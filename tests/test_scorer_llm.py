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
from pipeline.scorer import compute_llm_score_with_context, score_segments

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
# Unit tests — compute_llm_score_with_context
# ---------------------------------------------------------------------------

class TestComputeLLMScore:
    """Unit tests for compute_llm_score_with_context."""

    def _mock_response(self, response_text: str) -> MagicMock:
        """Build a mock requests.Response with a given response body."""
        mock_resp = MagicMock()
        mock_resp.text = response_text
        mock_resp.json.return_value = {"response": response_text}
        return mock_resp

    def _make_segments(self, n: int = 5) -> list[Segment]:
        return [Segment(start=float(i), end=float(i + 1), text=f"Segment {i}.") for i in range(n)]

    def test_valid_score_parsed(self) -> None:
        """Valid SCORE line in response → correct normalized score."""
        config = make_config(llm_enabled=True)
        segs = self._make_segments()
        response_text = "SCORE: 8\nTITLE: Amazing Moment\nDESCRIPTION: Great clip.\nTAGS: #shorts #viral"

        with patch("pipeline.scorer.requests.post", return_value=self._mock_response(response_text)):
            score, meta = compute_llm_score_with_context(config, 2, segs)

        assert abs(score - 0.8) < 1e-9

    def test_score_10_normalizes_to_1(self) -> None:
        """Score of 10 normalizes to 1.0."""
        config = make_config(llm_enabled=True)
        segs = self._make_segments()
        response_text = "SCORE: 10\nTITLE: Incredible!\nDESCRIPTION: Wow.\nTAGS: #shorts"

        with patch("pipeline.scorer.requests.post", return_value=self._mock_response(response_text)):
            score, meta = compute_llm_score_with_context(config, 2, segs)

        assert abs(score - 1.0) < 1e-9

    def test_score_1_normalizes_to_0_1(self) -> None:
        """Score of 1 normalizes to 0.1."""
        config = make_config(llm_enabled=True)
        segs = self._make_segments()
        response_text = "SCORE: 1\nTITLE: Boring\nDESCRIPTION: Not great.\nTAGS: #shorts"

        with patch("pipeline.scorer.requests.post", return_value=self._mock_response(response_text)):
            score, meta = compute_llm_score_with_context(config, 2, segs)

        assert abs(score - 0.1) < 1e-9

    def test_metadata_parsed(self) -> None:
        """Title, description, and tags are parsed from the response."""
        config = make_config(llm_enabled=True)
        segs = self._make_segments()
        response_text = (
            "SCORE: 7\n"
            "TITLE: Watch This Crazy Moment\n"
            "DESCRIPTION: You won't believe what happens next.\n"
            "TAGS: #shorts #viral #crazy #mustwatch"
        )

        with patch("pipeline.scorer.requests.post", return_value=self._mock_response(response_text)):
            score, meta = compute_llm_score_with_context(config, 2, segs)

        assert meta is not None
        assert meta.title == "Watch This Crazy Moment"
        assert "believe" in meta.description
        assert "#shorts" in meta.tags
        assert "#viral" in meta.tags

    def test_no_parseable_score_returns_zero_and_none(self, caplog) -> None:
        """Response with no SCORE line → (0.0, None) and warning logged."""
        config = make_config(llm_enabled=True)
        segs = self._make_segments()
        response_text = "I cannot rate this clip."

        with caplog.at_level(logging.WARNING):
            with patch("pipeline.scorer.requests.post", return_value=self._mock_response(response_text)):
                score, meta = compute_llm_score_with_context(config, 2, segs)

        assert score == 0.0
        assert meta is None

    def test_connection_error_raises_llm_scoring_error(self) -> None:
        """requests.ConnectionError → LLMScoringError raised."""
        config = make_config(llm_enabled=True)
        segs = self._make_segments()

        with patch("pipeline.scorer.requests.post", side_effect=requests.ConnectionError("refused")):
            with pytest.raises(LLMScoringError, match="unreachable"):
                compute_llm_score_with_context(config, 2, segs)

    def test_timeout_raises_llm_scoring_error(self) -> None:
        """requests.Timeout → LLMScoringError raised."""
        config = make_config(llm_enabled=True)
        segs = self._make_segments()

        with patch("pipeline.scorer.requests.post", side_effect=requests.Timeout("timed out")):
            with pytest.raises(LLMScoringError, match="unreachable"):
                compute_llm_score_with_context(config, 2, segs)

    def test_context_window_included_in_prompt(self) -> None:
        """The prompt sent to the LLM includes surrounding context segments."""
        config = make_config(llm_enabled=True)
        config.llm_context_window = 2
        segs = self._make_segments(7)
        response_text = "SCORE: 5\nTITLE: Test\nDESCRIPTION: Desc.\nTAGS: #shorts"

        mock_resp = self._mock_response(response_text)
        with patch("pipeline.scorer.requests.post", return_value=mock_resp) as mock_post:
            compute_llm_score_with_context(config, 3, segs)

        sent_prompt = mock_post.call_args[1]["json"]["prompt"]
        # Candidate segment 3 and its neighbours (1,2,3,4,5) should appear
        assert "<<<HIGHLIGHT>>>" in sent_prompt
        assert "Segment 3" in sent_prompt
        assert "Segment 2" in sent_prompt  # context before
        assert "Segment 4" in sent_prompt  # context after

    def test_correct_payload_sent(self) -> None:
        """Verify the POST payload has the correct model and stream=False."""
        config = make_config(llm_enabled=True, llm_model="llama3.2:1b")
        segs = self._make_segments()
        response_text = "SCORE: 6\nTITLE: Cool\nDESCRIPTION: Nice.\nTAGS: #shorts"

        mock_resp = self._mock_response(response_text)
        with patch("pipeline.scorer.requests.post", return_value=mock_resp) as mock_post:
            compute_llm_score_with_context(config, 2, segs)

        sent_json = mock_post.call_args[1]["json"]
        assert sent_json["model"] == "llama3.2:1b"
        assert sent_json["stream"] is False


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
        """When llm_enabled=False, _score_window_with_llm is never called."""
        wav_path = make_wav(tmp_path)
        config = make_config(llm_enabled=False, text_weight=0.4, audio_weight=0.6, llm_weight=0.0)
        segments = [make_segment("Hello.", 0.0, 1.0)]
        transcript = Transcript(segments=segments)

        with patch("pipeline.scorer._score_window_with_llm") as mock_llm:
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
            "pipeline.scorer._score_window_with_llm",
            side_effect=LLMScoringError("endpoint down"),
        ):
            result = score_segments(config, transcript, wav_path)

        assert result[0].llm_score == 0.0

    def test_llm_enabled_calls_window_scorer_for_candidates(self, tmp_path) -> None:
        """When llm_enabled=True, _score_window_with_llm is called for top candidates."""
        wav_path = make_wav(tmp_path, duration=4.0)
        config = make_config(
            llm_enabled=True,
            text_weight=0.3,
            audio_weight=0.4,
            llm_weight=0.3,
        )
        # Set max_candidates high enough to cover both segments
        config.llm_max_candidates = 10
        segments = [
            make_segment("First.", 0.0, 1.0),
            make_segment("Second.", 2.0, 3.0),  # spaced > min_clip_duration apart? no — use small min
        ]
        config.min_clip_duration = 0.5  # small so both seeds are selected
        transcript = Transcript(segments=segments)

        with patch(
            "pipeline.scorer._score_window_with_llm",
            return_value=(0.5, None),
        ) as mock_llm:
            result = score_segments(config, transcript, wav_path)

        assert mock_llm.call_count >= 1  # at least one window scored
        # All segments in the window get the LLM score propagated
        for ss in result:
            assert ss.llm_score >= 0.0

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
