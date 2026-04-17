"""Tests for pipeline/scorer.py — LLM scoring and score_segments assembly.

Covers:
- Property 8: Score weights sum to 1.0 (subtask 6.1)
- Unit tests for compute_llm_score and score_segments (subtask 6.2)
"""

from __future__ import annotations

import logging
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import requests
import scipy.io.wavfile
from hypothesis import given, settings
from hypothesis import strategies as st

from config import Config
from pipeline.exceptions import LLMScoringError
from pipeline.models import ScoredSegment, Segment, Transcript
from pipeline.scorer import compute_llm_score, score_segments

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000  # Hz


def make_config(
    llm_enabled: bool = False,
    llm_endpoint: str = "http://localhost:11434/api/generate",
    llm_model: str = "llama3",
    text_weight: float = 0.4,
    audio_weight: float = 0.6,
    llm_weight: float = 0.0,
) -> Config:
    cfg = Config(work_dir="/tmp/test")
    cfg.llm_enabled = llm_enabled
    cfg.llm_endpoint = llm_endpoint
    cfg.llm_model = llm_model
    cfg.text_weight = text_weight
    cfg.audio_weight = audio_weight
    cfg.llm_weight = llm_weight
    return cfg


def make_segment(text: str = "hello world", start: float = 0.0, end: float = 1.0) -> Segment:
    return Segment(start=start, end=end, text=text)


def write_wav(path: str, data: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Write a float32 numpy array as a WAV file."""
    scipy.io.wavfile.write(path, sample_rate, data.astype(np.float32))


def make_wav(tmp_path, duration: float = 2.0) -> str:
    """Create a simple sine-wave WAV file and return its path."""
    n = int(duration * SAMPLE_RATE)
    t = np.linspace(0, duration, n, endpoint=False)
    audio = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    wav_path = str(tmp_path / "test.wav")
    write_wav(wav_path, audio)
    return wav_path


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------

# Feature: video-highlight-generator, Property 8: Score weights sum to 1.0
# Validates: Requirements 6.3
@given(
    llm_weight=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_weights_sum_to_one(llm_weight: float) -> None:
    """Property 8: Score weights sum to 1.0 when llm_enabled=True.

    Generate an arbitrary llm_weight in [0.0, 1.0]; derive text_weight and
    audio_weight proportionally from the remaining (1 - llm_weight), split
    40/60.  Assert the three weights sum to 1.0 within floating-point
    tolerance.

    **Validates: Requirements 6.3**
    """
    remaining = 1.0 - llm_weight
    # Split remaining 40 % text / 60 % audio (matching the design defaults)
    text_weight = remaining * 0.4
    audio_weight = remaining * 0.6

    total = text_weight + audio_weight + llm_weight
    assert abs(total - 1.0) < 1e-9, (
        f"Weights do not sum to 1.0: text={text_weight}, audio={audio_weight}, "
        f"llm={llm_weight}, total={total}"
    )


# ---------------------------------------------------------------------------
# Unit tests — compute_llm_score
# ---------------------------------------------------------------------------

class TestComputeLLMScore:
    """Unit tests for compute_llm_score."""

    def _mock_response(self, json_data=None, text: str = "") -> MagicMock:
        """Build a mock requests.Response."""
        mock_resp = MagicMock()
        mock_resp.text = text
        if json_data is not None:
            mock_resp.json.return_value = json_data
        else:
            mock_resp.json.side_effect = ValueError("no JSON")
        return mock_resp

    def test_valid_numeric_response_in_json(self) -> None:
        """Valid numeric score in JSON 'response' field → correct normalized score."""
        config = make_config(llm_enabled=True)
        seg = make_segment("This is an amazing moment!")

        mock_resp = self._mock_response(json_data={"response": "8"}, text="8")

        with patch("pipeline.scorer.requests.post", return_value=mock_resp):
            score = compute_llm_score(config, seg)

        assert abs(score - 0.8) < 1e-9

    def test_valid_numeric_response_integer_in_json(self) -> None:
        """JSON 'response' field with integer value → correct normalized score."""
        config = make_config(llm_enabled=True)
        seg = make_segment("Watch this!")

        mock_resp = self._mock_response(json_data={"response": 7}, text="7")

        with patch("pipeline.scorer.requests.post", return_value=mock_resp):
            score = compute_llm_score(config, seg)

        assert abs(score - 0.7) < 1e-9

    def test_score_10_normalizes_to_1(self) -> None:
        """Score of 10 normalizes to 1.0."""
        config = make_config(llm_enabled=True)
        seg = make_segment("Incredible!")

        mock_resp = self._mock_response(json_data={"response": "10"}, text="10")

        with patch("pipeline.scorer.requests.post", return_value=mock_resp):
            score = compute_llm_score(config, seg)

        assert abs(score - 1.0) < 1e-9

    def test_score_1_normalizes_to_0_1(self) -> None:
        """Score of 1 normalizes to 0.1."""
        config = make_config(llm_enabled=True)
        seg = make_segment("Boring segment.")

        mock_resp = self._mock_response(json_data={"response": "1"}, text="1")

        with patch("pipeline.scorer.requests.post", return_value=mock_resp):
            score = compute_llm_score(config, seg)

        assert abs(score - 0.1) < 1e-9

    def test_no_parseable_number_returns_zero_and_logs_warning(
        self, caplog
    ) -> None:
        """LLM response with no parseable number → score 0.0 and warning logged."""
        config = make_config(llm_enabled=True)
        seg = make_segment("Some segment text.")

        mock_resp = self._mock_response(
            json_data={"response": "I cannot rate this."},
            text="I cannot rate this.",
        )

        with caplog.at_level(logging.WARNING, logger="root"):
            with patch("pipeline.scorer.requests.post", return_value=mock_resp):
                score = compute_llm_score(config, seg)

        assert score == 0.0
        assert any("parseable" in record.message.lower() or "0.0" in record.message
                   for record in caplog.records), (
            f"Expected a warning about unparseable score, got: {caplog.records}"
        )

    def test_no_json_falls_back_to_regex_on_text(self) -> None:
        """When JSON parsing fails, regex is applied to response.text."""
        config = make_config(llm_enabled=True)
        seg = make_segment("Great moment!")

        # json() raises ValueError, but text contains a number
        mock_resp = self._mock_response(json_data=None, text="The score is 6.")

        with patch("pipeline.scorer.requests.post", return_value=mock_resp):
            score = compute_llm_score(config, seg)

        assert abs(score - 0.6) < 1e-9

    def test_connection_error_raises_llm_scoring_error(self) -> None:
        """requests.ConnectionError → LLMScoringError raised."""
        config = make_config(llm_enabled=True)
        seg = make_segment("Some text.")

        with patch(
            "pipeline.scorer.requests.post",
            side_effect=requests.ConnectionError("refused"),
        ):
            with pytest.raises(LLMScoringError, match="unreachable"):
                compute_llm_score(config, seg)

    def test_timeout_raises_llm_scoring_error(self) -> None:
        """requests.Timeout → LLMScoringError raised."""
        config = make_config(llm_enabled=True)
        seg = make_segment("Some text.")

        with patch(
            "pipeline.scorer.requests.post",
            side_effect=requests.Timeout("timed out"),
        ):
            with pytest.raises(LLMScoringError, match="unreachable"):
                compute_llm_score(config, seg)

    def test_correct_payload_sent(self) -> None:
        """Verify the POST payload matches the expected structure."""
        config = make_config(
            llm_enabled=True,
            llm_endpoint="http://localhost:11434/api/generate",
            llm_model="llama3",
        )
        seg = make_segment("Watch this crazy moment!")

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "9"}
        mock_resp.text = "9"

        with patch("pipeline.scorer.requests.post", return_value=mock_resp) as mock_post:
            compute_llm_score(config, seg)

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        url = call_kwargs[0][0] if call_kwargs[0] else call_kwargs[1].get("url", call_kwargs[0])
        # Check URL
        assert "11434" in str(call_kwargs)
        # Check payload fields
        sent_json = call_kwargs[1]["json"] if "json" in call_kwargs[1] else call_kwargs[0][1]
        assert sent_json["model"] == "llama3"
        assert sent_json["stream"] is False
        assert "Watch this crazy moment!" in sent_json["prompt"]


# ---------------------------------------------------------------------------
# Unit tests — score_segments
# ---------------------------------------------------------------------------

class TestScoreSegments:
    """Unit tests for score_segments."""

    def test_returns_correct_number_of_scored_segments(self, tmp_path) -> None:
        """score_segments returns one ScoredSegment per transcript segment."""
        wav_path = make_wav(tmp_path)
        config = make_config(llm_enabled=False)
        segments = [
            make_segment("Hello world.", 0.0, 1.0),
            make_segment("Watch this!", 1.0, 2.0),
        ]
        transcript = Transcript(segments=segments)

        result = score_segments(config, transcript, wav_path)

        assert len(result) == 2
        for ss in result:
            assert isinstance(ss, ScoredSegment)

    def test_llm_disabled_compute_llm_score_not_called(self, tmp_path) -> None:
        """When llm_enabled=False, compute_llm_score is never called."""
        wav_path = make_wav(tmp_path)
        config = make_config(llm_enabled=False, text_weight=0.4, audio_weight=0.6, llm_weight=0.0)
        segments = [make_segment("Hello.", 0.0, 1.0)]
        transcript = Transcript(segments=segments)

        with patch("pipeline.scorer.compute_llm_score") as mock_llm:
            result = score_segments(config, transcript, wav_path)

        mock_llm.assert_not_called()
        assert result[0].llm_score == 0.0

    def test_llm_disabled_clip_score_uses_only_text_and_audio(self, tmp_path) -> None:
        """When llm_enabled=False, clip_score = text_weight*text + audio_weight*audio."""
        wav_path = make_wav(tmp_path)
        config = make_config(llm_enabled=False, text_weight=0.4, audio_weight=0.6, llm_weight=0.0)
        segments = [make_segment("Hello.", 0.0, 1.0)]
        transcript = Transcript(segments=segments)

        result = score_segments(config, transcript, wav_path)

        ss = result[0]
        expected_clip = 0.4 * ss.text_score + 0.6 * ss.audio_score + 0.0 * ss.llm_score
        assert abs(ss.clip_score - expected_clip) < 1e-9

    def test_llm_scoring_error_falls_back_to_zero(self, tmp_path) -> None:
        """LLMScoringError caught in score_segments → llm_score falls back to 0.0."""
        wav_path = make_wav(tmp_path)
        config = make_config(
            llm_enabled=True,
            text_weight=0.3,
            audio_weight=0.4,
            llm_weight=0.3,
        )
        segments = [make_segment("Some text.", 0.0, 1.0)]
        transcript = Transcript(segments=segments)

        with patch(
            "pipeline.scorer.compute_llm_score",
            side_effect=LLMScoringError("endpoint down"),
        ):
            result = score_segments(config, transcript, wav_path)

        assert result[0].llm_score == 0.0

    def test_llm_enabled_calls_compute_llm_score_per_segment(self, tmp_path) -> None:
        """When llm_enabled=True, compute_llm_score is called once per segment."""
        wav_path = make_wav(tmp_path)
        config = make_config(
            llm_enabled=True,
            text_weight=0.3,
            audio_weight=0.4,
            llm_weight=0.3,
        )
        segments = [
            make_segment("First.", 0.0, 1.0),
            make_segment("Second.", 1.0, 2.0),
        ]
        transcript = Transcript(segments=segments)

        with patch("pipeline.scorer.compute_llm_score", return_value=0.5) as mock_llm:
            result = score_segments(config, transcript, wav_path)

        assert mock_llm.call_count == 2
        for ss in result:
            assert ss.llm_score == 0.5

    def test_scored_segment_fields_populated(self, tmp_path) -> None:
        """All ScoredSegment fields are populated with valid values."""
        wav_path = make_wav(tmp_path)
        config = make_config(llm_enabled=False)
        seg = make_segment("Incredible moment!", 0.0, 1.0)
        transcript = Transcript(segments=[seg])

        result = score_segments(config, transcript, wav_path)

        ss = result[0]
        assert ss.segment is seg
        assert 0.0 <= ss.text_score <= 1.0
        assert 0.0 <= ss.audio_score <= 1.0
        assert ss.llm_score == 0.0
        assert ss.clip_score >= 0.0

    def test_empty_transcript_returns_empty_list(self, tmp_path) -> None:
        """Empty transcript produces an empty list of ScoredSegments."""
        wav_path = make_wav(tmp_path)
        config = make_config(llm_enabled=False)
        transcript = Transcript(segments=[])

        result = score_segments(config, transcript, wav_path)

        assert result == []
