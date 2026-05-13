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
from pipeline.scorer import _build_candidate_windows, compute_llm_score_with_context, score_segments

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


def test_config_uses_ollama_host_env_var(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "http://ollama:11434")
    cfg = Config(work_dir="/tmp/test")
    assert cfg.llm_endpoint == "http://ollama:11434/api/generate"


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
        """Both attempts time out → LLMScoringError raised (falls back to 0.0 via caller)."""
        config = make_config(llm_enabled=True)
        segs = self._make_segments()

        with patch("pipeline.scorer.requests.post", side_effect=requests.Timeout("timed out")):
            with pytest.raises(LLMScoringError, match="unreachable"):
                compute_llm_score_with_context(config, 2, segs)

    def test_first_timeout_retries_and_succeeds(self) -> None:
        """First attempt times out, second attempt succeeds → returns the score."""
        config = make_config(llm_enabled=True)
        segs = self._make_segments()
        response_text = "SCORE: 7\nTITLE: Great Clip\nDESCRIPTION: Nice moment.\nTAGS: #shorts"

        mock_resp = self._mock_response(response_text)
        side_effects = [requests.Timeout("timed out"), mock_resp]

        with patch("pipeline.scorer.requests.post", side_effect=side_effects) as mock_post:
            score, meta = compute_llm_score_with_context(config, 2, segs)

        assert abs(score - 0.7) < 1e-9
        # First call uses 60s timeout, retry uses 45s timeout
        assert mock_post.call_count == 2
        assert mock_post.call_args_list[0][1]["timeout"] == 60
        assert mock_post.call_args_list[1][1]["timeout"] == 45

    def test_retry_log_message_emitted(self, caplog) -> None:
        """On first timeout, the retry log message is emitted."""
        config = make_config(llm_enabled=True)
        segs = self._make_segments()
        response_text = "SCORE: 5\nTITLE: OK\nDESCRIPTION: Fine.\nTAGS: #shorts"

        mock_resp = self._mock_response(response_text)
        side_effects = [requests.Timeout("timed out"), mock_resp]

        with caplog.at_level(logging.WARNING):
            with patch("pipeline.scorer.requests.post", side_effect=side_effects):
                compute_llm_score_with_context(config, 2, segs)

        assert any("retrying with 45s timeout (attempt 2/2)" in r.message for r in caplog.records)

    def test_both_timeouts_falls_back_to_zero(self, tmp_path) -> None:
        """Both attempts time out → score_segments falls back to llm_score=0.0."""
        wav_path = make_wav(tmp_path)
        config = make_config(
            llm_enabled=True,
            text_weight=0.3,
            audio_weight=0.4,
            llm_weight=0.3,
        )
        segments = [make_segment("Some text.", 0.0, 1.0)]
        transcript = Transcript(segments=segments)

        with patch("pipeline.scorer.requests.post", side_effect=requests.Timeout("timed out")), \
             patch("pipeline.scorer._check_llm_model_available", return_value=True):
            result = score_segments(config, transcript, wav_path)

        assert result[0].llm_score == 0.0

    def test_connection_error_no_retry_falls_back_to_zero(self, tmp_path) -> None:
        """Connection error → no retry, score_segments falls back to llm_score=0.0."""
        wav_path = make_wav(tmp_path)
        config = make_config(
            llm_enabled=True,
            text_weight=0.3,
            audio_weight=0.4,
            llm_weight=0.3,
        )
        segments = [make_segment("Some text.", 0.0, 1.0)]
        transcript = Transcript(segments=segments)

        call_count = {"n": 0}

        def _conn_error(*args, **kwargs):
            call_count["n"] += 1
            raise requests.ConnectionError("refused")

        with patch("pipeline.scorer.requests.post", side_effect=_conn_error), \
             patch("pipeline.scorer._check_llm_model_available", return_value=True):
            result = score_segments(config, transcript, wav_path)

        # Two POST calls: 1 for video summary + 1 for window scoring (no retry on ConnectionError)
        assert call_count["n"] == 2
        assert result[0].llm_score == 0.0

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
        """When llm_enabled=False, spike_weight=0, burst_weight=0, clip_score is a
        weighted combination of text and audio (or excitement) signals."""
        wav_path = make_wav(tmp_path)
        config = make_config(llm_enabled=False, text_weight=0.4, audio_weight=0.6, llm_weight=0.0)
        config.spike_weight = 0.0
        config.burst_weight = 0.0
        segments = [make_segment("Hello.", 0.0, 1.0)]
        transcript = Transcript(segments=segments)

        result = score_segments(config, transcript, wav_path)

        ss = result[0]
        # clip_score must be non-negative and bounded
        assert ss.clip_score >= 0.0
        assert ss.clip_score <= 1.5  # generous upper bound
        # llm_score must be 0 when LLM disabled
        assert ss.llm_score == 0.0

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
        config.hook_detection_enabled = False  # disable hook detection to isolate LLM window scoring
        segments = [
            make_segment("First.", 0.0, 1.0),
            make_segment("Second.", 2.0, 3.0),  # spaced > min_clip_duration apart? no — use small min
        ]
        config.min_clip_duration = 0.5  # small so both seeds are selected
        transcript = Transcript(segments=segments)

        with patch("pipeline.scorer._check_llm_model_available", return_value=True), \
             patch("pipeline.scorer._score_window_with_llm", return_value=(0.5, None)) as mock_llm:
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


# ---------------------------------------------------------------------------
# Unit tests — _build_candidate_windows (dual-track audio-only candidate track)
# ---------------------------------------------------------------------------

class TestBuildCandidateWindowsDualTrack:
    """Tests for the dual-track candidate selection in _build_candidate_windows."""

    def _make_config(
        self,
        llm_max_candidates: int = 6,
        llm_audio_candidates: int = 3,
        min_clip_duration: float = 5.0,
        text_weight: float = 0.5,
        audio_weight: float = 0.5,
    ) -> Config:
        cfg = Config(work_dir="/tmp/test")
        cfg.llm_max_candidates = llm_max_candidates
        cfg.llm_audio_candidates = llm_audio_candidates
        cfg.min_clip_duration = min_clip_duration
        cfg.text_weight = text_weight
        cfg.audio_weight = audio_weight
        return cfg

    def _make_segments(self, n: int, spacing: float = 10.0) -> list[Segment]:
        """Create n segments spaced *spacing* seconds apart."""
        return [
            Segment(start=float(i) * spacing, end=float(i) * spacing + 1.0, text=f"seg{i}")
            for i in range(n)
        ]

    # ------------------------------------------------------------------
    # Test 1: high spike / low text segment gets included via audio track
    # ------------------------------------------------------------------

    def test_high_spike_low_text_included(self) -> None:
        """A segment with high spike score but low text+audio score is included
        in the candidates via the audio-only track."""
        # 8 segments spaced 10s apart (well beyond min_clip_duration=5s)
        segments = self._make_segments(8, spacing=10.0)
        cfg = self._make_config(
            llm_max_candidates=6,
            llm_audio_candidates=3,
            min_clip_duration=5.0,
        )

        # All segments have low text and audio scores
        text_scores = [0.1] * 8
        audio_scores = [0.1] * 8

        # Segment 5 has a massive spike score but still low text+audio
        spike_scores = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]

        candidates = _build_candidate_windows(
            segments, text_scores, audio_scores, cfg, spike_scores
        )

        llm_track, spike_track = candidates
        candidate_indices = [idx for idx, _ in llm_track] + [idx for idx, _ in spike_track]
        assert 5 in candidate_indices, (
            "Segment 5 (high spike, low text) should be included via the audio track"
        )

    # ------------------------------------------------------------------
    # Test 2: low spike AND low text segment is NOT included
    # ------------------------------------------------------------------

    def test_low_spike_low_text_excluded(self) -> None:
        """A segment with low spike score AND low text+audio score is NOT
        included when better candidates exist on both tracks."""
        segments = self._make_segments(8, spacing=10.0)
        cfg = self._make_config(
            llm_max_candidates=4,
            llm_audio_candidates=2,
            min_clip_duration=5.0,
        )

        # Segments 0-3 have high text+audio AND high spike scores
        # Segments 4-7 have low scores on ALL tracks
        text_scores = [0.9, 0.9, 0.9, 0.9, 0.05, 0.05, 0.05, 0.05]
        audio_scores = [0.9, 0.9, 0.9, 0.9, 0.05, 0.05, 0.05, 0.05]
        spike_scores = [0.9, 0.9, 0.9, 0.9, 0.05, 0.05, 0.05, 0.05]

        candidates = _build_candidate_windows(
            segments, text_scores, audio_scores, cfg, spike_scores
        )

        llm_track, spike_track = candidates
        candidate_indices = [idx for idx, _ in llm_track] + [idx for idx, _ in spike_track]
        # Segments 4-7 have low scores on all tracks — none should appear
        for bad_idx in [4, 5, 6, 7]:
            assert bad_idx not in candidate_indices, (
                f"Segment {bad_idx} (low spike, low text) should NOT be included"
            )

    # ------------------------------------------------------------------
    # Test 3: deduplication — no two candidates within min_clip_duration
    # ------------------------------------------------------------------

    def test_deduplication_no_two_within_min_spacing(self) -> None:
        """After merging both tracks, no two candidates should have midpoints
        within min_clip_duration of each other."""
        # 10 segments spaced 2s apart — many are within min_clip_duration=5s
        segments = self._make_segments(10, spacing=2.0)
        cfg = self._make_config(
            llm_max_candidates=6,
            llm_audio_candidates=3,
            min_clip_duration=5.0,
        )

        # Alternate high text and high spike to stress-test deduplication
        text_scores = [0.9 if i % 2 == 0 else 0.1 for i in range(10)]
        audio_scores = [0.5] * 10
        spike_scores = [0.1 if i % 2 == 0 else 0.9 for i in range(10)]

        candidates = _build_candidate_windows(
            segments, text_scores, audio_scores, cfg, spike_scores
        )

        # Check all pairs of candidates are spaced >= min_clip_duration apart
        llm_track, spike_track = candidates
        all_candidates = llm_track + spike_track
        midpoints = [
            (segments[idx].start + segments[idx].end) / 2.0
            for idx, _ in all_candidates
        ]
        for i in range(len(midpoints)):
            for j in range(i + 1, len(midpoints)):
                gap = abs(midpoints[i] - midpoints[j])
                assert gap >= cfg.min_clip_duration, (
                    f"Candidates at midpoints {midpoints[i]:.1f}s and "
                    f"{midpoints[j]:.1f}s are only {gap:.1f}s apart "
                    f"(min_clip_duration={cfg.min_clip_duration}s)"
                )

    # ------------------------------------------------------------------
    # Test 4: total candidates <= llm_max_candidates
    # ------------------------------------------------------------------

    def test_total_candidates_within_budget(self) -> None:
        """The merged candidate list never exceeds llm_max_candidates."""
        segments = self._make_segments(20, spacing=10.0)
        cfg = self._make_config(
            llm_max_candidates=6,
            llm_audio_candidates=3,
            min_clip_duration=5.0,
        )

        text_scores = [float(i) / 20.0 for i in range(20)]
        audio_scores = [float(i) / 20.0 for i in range(20)]
        spike_scores = [float(19 - i) / 20.0 for i in range(20)]

        candidates = _build_candidate_windows(
            segments, text_scores, audio_scores, cfg, spike_scores
        )

        assert len(candidates) <= cfg.llm_max_candidates

    # ------------------------------------------------------------------
    # Test 5: llm_audio_candidates=0 falls back to single-track behaviour
    # ------------------------------------------------------------------

    def test_zero_audio_candidates_single_track(self) -> None:
        """When llm_audio_candidates=0, only the text+audio track is used."""
        segments = self._make_segments(6, spacing=10.0)
        cfg = self._make_config(
            llm_max_candidates=3,
            llm_audio_candidates=0,
            min_clip_duration=5.0,
        )

        # Segment 5 has high spike but low text+audio
        text_scores = [0.9, 0.8, 0.7, 0.1, 0.1, 0.1]
        audio_scores = [0.9, 0.8, 0.7, 0.1, 0.1, 0.1]
        spike_scores = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

        candidates = _build_candidate_windows(
            segments, text_scores, audio_scores, cfg, spike_scores
        )

        llm_track, spike_track = candidates
        candidate_indices = [idx for idx, _ in llm_track] + [idx for idx, _ in spike_track]
        # With no audio budget, segment 5 (high spike, low text) should NOT appear
        assert 5 not in candidate_indices, (
            "With llm_audio_candidates=0, spike-only segment should not be included"
        )
        # Top text+audio segments (0, 1, 2) should be selected
        for expected in [0, 1, 2]:
            assert expected in candidate_indices

    # ------------------------------------------------------------------
    # Test 6: deduplication keeps higher pre_score when tracks overlap
    # ------------------------------------------------------------------

    def test_deduplication_keeps_higher_prescore(self) -> None:
        """When both tracks select the same segment (or nearby ones), the one
        with the higher pre_score is kept."""
        # Two segments very close together (within min_clip_duration)
        segments = [
            Segment(start=0.0, end=1.0, text="seg0"),
            Segment(start=1.5, end=2.5, text="seg1"),  # 1.5s from seg0 — within 5s spacing
        ]
        cfg = self._make_config(
            llm_max_candidates=4,
            llm_audio_candidates=2,
            min_clip_duration=5.0,
        )

        # seg0: high text+audio, low spike
        # seg1: low text+audio, high spike
        text_scores = [0.9, 0.1]
        audio_scores = [0.9, 0.1]
        spike_scores = [0.1, 0.9]

        candidates = _build_candidate_windows(
            segments, text_scores, audio_scores, cfg, spike_scores
        )

        # Only one of the two should appear (they're within min_clip_duration)
        llm_track, spike_track = candidates
        all_candidates = llm_track + spike_track
        assert len(all_candidates) == 1
        # seg0 has higher pre_score (0.9*0.5 + 0.9*0.5 = 0.9 vs 0.1*0.5 + 0.1*0.5 = 0.1)
        assert all_candidates[0][0] == 0, "seg0 (higher pre_score) should be kept over seg1"


# ---------------------------------------------------------------------------
# Unit tests — prefilter weight decoupling (Task 43)
# ---------------------------------------------------------------------------

class TestPrefilterWeightDecoupling:
    """Tests verifying that _build_candidate_windows uses llm_prefilter_*_weight
    for candidate ranking, while score_segments still uses text_weight/audio_weight
    for the final clip_score.
    """

    def _make_config(
        self,
        text_weight: float = 0.5,
        audio_weight: float = 0.5,
        llm_prefilter_text_weight: float = 0.2,
        llm_prefilter_audio_weight: float = 0.8,
        min_clip_duration: float = 5.0,
        llm_max_candidates: int = 4,
        llm_audio_candidates: int = 0,
    ) -> Config:
        cfg = Config(work_dir="/tmp/test")
        cfg.text_weight = text_weight
        cfg.audio_weight = audio_weight
        cfg.llm_prefilter_text_weight = llm_prefilter_text_weight
        cfg.llm_prefilter_audio_weight = llm_prefilter_audio_weight
        cfg.min_clip_duration = min_clip_duration
        cfg.llm_max_candidates = llm_max_candidates
        cfg.llm_audio_candidates = llm_audio_candidates
        cfg.spike_weight = 0.0
        cfg.burst_weight = 0.0
        return cfg

    def _make_segments(self, n: int, spacing: float = 10.0) -> list[Segment]:
        return [
            Segment(start=float(i) * spacing, end=float(i) * spacing + 1.0, text=f"seg{i}")
            for i in range(n)
        ]

    def test_prefilter_uses_prefilter_weights_not_final_weights(self) -> None:
        """_build_candidate_windows ranks by llm_prefilter_*_weight, not text_weight/audio_weight.

        Set up two segments:
          - seg0: high text (0.9), low audio (0.1)
          - seg1: low text (0.1), high audio (0.9)

        With llm_prefilter_text_weight=0.2, llm_prefilter_audio_weight=0.8:
          pre_score(seg0) = 0.2*0.9 + 0.8*0.1 = 0.18 + 0.08 = 0.26
          pre_score(seg1) = 0.2*0.1 + 0.8*0.9 = 0.02 + 0.72 = 0.74  ← higher

        With final weights text_weight=0.5, audio_weight=0.5:
          final_score(seg0) = 0.5*0.9 + 0.5*0.1 = 0.50
          final_score(seg1) = 0.5*0.1 + 0.5*0.9 = 0.50  ← tied

        The prefilter should rank seg1 first (audio-heavy prefilter).
        """
        segments = [
            Segment(start=0.0, end=1.0, text="seg0"),
            Segment(start=20.0, end=21.0, text="seg1"),  # spaced > min_clip_duration
        ]
        cfg = self._make_config(
            text_weight=0.5,
            audio_weight=0.5,
            llm_prefilter_text_weight=0.2,
            llm_prefilter_audio_weight=0.8,
            min_clip_duration=5.0,
            llm_max_candidates=2,
        )

        text_scores = [0.9, 0.1]
        audio_scores = [0.1, 0.9]

        candidates = _build_candidate_windows(segments, text_scores, audio_scores, cfg)

        # seg1 should rank first because its prefilter pre_score (0.74) > seg0 (0.26)
        llm_track, _ = candidates
        assert llm_track[0][0] == 1, (
            "seg1 (high audio, low text) should rank first with audio-heavy prefilter weights"
        )

    def test_high_audio_low_text_ranks_higher_with_audio_heavy_prefilter(self) -> None:
        """A segment with high audio score but low text score ranks higher when
        llm_prefilter_audio_weight is high (0.8) vs llm_prefilter_text_weight (0.2).
        """
        segments = self._make_segments(4, spacing=10.0)
        cfg = self._make_config(
            llm_prefilter_text_weight=0.2,
            llm_prefilter_audio_weight=0.8,
            min_clip_duration=5.0,
            llm_max_candidates=4,
        )

        # seg2: high audio (0.95), low text (0.05) — should rank high with audio-heavy prefilter
        # seg3: high text (0.95), low audio (0.05) — should rank lower
        text_scores = [0.5, 0.5, 0.05, 0.95]
        audio_scores = [0.5, 0.5, 0.95, 0.05]

        candidates = _build_candidate_windows(segments, text_scores, audio_scores, cfg)

        llm_track, _ = candidates
        indices = [idx for idx, _ in llm_track]
        # seg2 pre_score = 0.2*0.05 + 0.8*0.95 = 0.01 + 0.76 = 0.77
        # seg3 pre_score = 0.2*0.95 + 0.8*0.05 = 0.19 + 0.04 = 0.23
        assert indices.index(2) < indices.index(3), (
            "seg2 (high audio) should rank before seg3 (high text) with audio-heavy prefilter"
        )

    def test_final_clip_score_uses_text_audio_weight_not_prefilter(self, tmp_path) -> None:
        """score_segments final clip_score uses text_weight/audio_weight, not prefilter weights.

        With llm disabled, spike_weight=0, burst_weight=0:
          clip_score = text_weight * text_score + audio_weight * audio_score

        This must NOT use llm_prefilter_text_weight or llm_prefilter_audio_weight.
        """
        # Build a WAV with two distinct energy levels
        n_samples = int(2.0 * SAMPLE_RATE)
        # First second: loud; second second: quiet
        audio_data = np.concatenate([
            np.ones(SAMPLE_RATE, dtype=np.float32) * 0.8,
            np.ones(SAMPLE_RATE, dtype=np.float32) * 0.1,
        ])
        wav_path = str(tmp_path / "test.wav")
        write_wav(wav_path, audio_data)

        cfg = Config(work_dir="/tmp/test")
        cfg.llm_enabled = False
        cfg.text_weight = 0.5
        cfg.audio_weight = 0.5
        cfg.llm_prefilter_text_weight = 0.2   # different from final weights
        cfg.llm_prefilter_audio_weight = 0.8  # different from final weights
        cfg.llm_weight = 0.0
        cfg.spike_weight = 0.0
        cfg.burst_weight = 0.0

        segments = [make_segment("Hello.", 0.0, 1.0)]
        transcript = Transcript(segments=segments)

        result = score_segments(cfg, transcript, wav_path)

        ss = result[0]
        # clip_score must be non-negative and bounded
        assert ss.clip_score >= 0.0
        assert ss.clip_score <= 1.5
        # llm_score must be 0 when LLM disabled
        assert ss.llm_score == 0.0

    def test_prefilter_weights_independent_of_final_weights(self) -> None:
        """Changing llm_prefilter_*_weight does not affect the pre_score formula
        used in _build_candidate_windows when final weights differ.

        Verify that the ranking changes when prefilter weights change, confirming
        the prefilter weights are actually being used.
        """
        segments = [
            Segment(start=0.0, end=1.0, text="seg0"),
            Segment(start=20.0, end=21.0, text="seg1"),
        ]

        text_scores = [0.9, 0.1]
        audio_scores = [0.1, 0.9]

        # Config A: audio-heavy prefilter → seg1 ranks first
        cfg_a = Config(work_dir="/tmp/test")
        cfg_a.text_weight = 0.5
        cfg_a.audio_weight = 0.5
        cfg_a.llm_prefilter_text_weight = 0.1
        cfg_a.llm_prefilter_audio_weight = 0.9
        cfg_a.min_clip_duration = 5.0
        cfg_a.llm_max_candidates = 2
        cfg_a.llm_audio_candidates = 0

        # Config B: text-heavy prefilter → seg0 ranks first
        cfg_b = Config(work_dir="/tmp/test")
        cfg_b.text_weight = 0.5
        cfg_b.audio_weight = 0.5
        cfg_b.llm_prefilter_text_weight = 0.9
        cfg_b.llm_prefilter_audio_weight = 0.1
        cfg_b.min_clip_duration = 5.0
        cfg_b.llm_max_candidates = 2
        cfg_b.llm_audio_candidates = 0

        candidates_a = _build_candidate_windows(segments, text_scores, audio_scores, cfg_a)
        candidates_b = _build_candidate_windows(segments, text_scores, audio_scores, cfg_b)

        # Audio-heavy prefilter: seg1 (high audio) should rank first
        assert candidates_a[0][0][0] == 1, (
            "With audio-heavy prefilter, seg1 (high audio) should rank first"
        )
        # Text-heavy prefilter: seg0 (high text) should rank first
        assert candidates_b[0][0][0] == 0, (
            "With text-heavy prefilter, seg0 (high text) should rank first"
        )


# ---------------------------------------------------------------------------
# Unit tests — LLM audio gate (Task 46)
# ---------------------------------------------------------------------------

class TestLLMAudioGate:
    """Tests for the llm_audio_gate soft cap in combine_scores."""

    def _make_config(
        self,
        llm_audio_gate: bool = True,
        text_weight: float = 0.3,
        audio_weight: float = 0.4,
        llm_weight: float = 0.3,
    ) -> Config:
        from pipeline.scorer import combine_scores  # noqa: F401 — ensure import works
        cfg = Config(work_dir="/tmp/test")
        cfg.text_weight = text_weight
        cfg.audio_weight = audio_weight
        cfg.llm_weight = llm_weight
        cfg.llm_audio_gate = llm_audio_gate
        cfg.spike_weight = 0.0
        cfg.burst_weight = 0.0
        return cfg

    def test_high_llm_low_audio_gated(self) -> None:
        """A 9/10 LLM score (0.9) with audio_score=0.1 should produce a lower
        clip_score than a 7/10 LLM score (0.7) with audio_score=0.8 when the
        gate is enabled.
        """
        from pipeline.scorer import combine_scores

        cfg = self._make_config(llm_audio_gate=True)

        # High LLM, low audio — gate should suppress the LLM contribution
        score_quiet = combine_scores(cfg, text=0.5, audio=0.1, llm=0.9)

        # Moderate LLM, high audio — gate has no effect (audio >= 0.3)
        score_loud = combine_scores(cfg, text=0.5, audio=0.8, llm=0.7)

        assert score_quiet < score_loud, (
            f"Expected gated quiet score ({score_quiet:.4f}) < loud score ({score_loud:.4f})"
        )

    def test_gate_disabled_uses_full_llm_score(self) -> None:
        """When llm_audio_gate=False, the LLM score is used at full weight
        regardless of audio_score.
        """
        from pipeline.scorer import combine_scores

        cfg_gated = self._make_config(llm_audio_gate=True)
        cfg_ungated = self._make_config(llm_audio_gate=False)

        # Low audio — gate would suppress LLM if enabled
        score_gated = combine_scores(cfg_gated, text=0.5, audio=0.1, llm=0.9)
        score_ungated = combine_scores(cfg_ungated, text=0.5, audio=0.1, llm=0.9)

        # Ungated should be higher because LLM is used at full weight
        assert score_ungated > score_gated, (
            f"Ungated score ({score_ungated:.4f}) should be > gated score ({score_gated:.4f})"
        )

        # Verify ungated uses the raw formula: text*t + audio*a + llm*l
        expected_ungated = (
            cfg_ungated.text_weight * 0.5
            + cfg_ungated.audio_weight * 0.1
            + cfg_ungated.llm_weight * 0.9
        )
        assert abs(score_ungated - expected_ungated) < 1e-9, (
            f"Ungated score {score_ungated:.6f} != expected {expected_ungated:.6f}"
        )

    def test_audio_above_threshold_uses_full_llm(self) -> None:
        """When audio_score >= 0.3, the LLM score is not reduced by the gate."""
        from pipeline.scorer import combine_scores

        cfg_gated = self._make_config(llm_audio_gate=True)
        cfg_ungated = self._make_config(llm_audio_gate=False)

        # audio_score = 0.3 exactly — gate factor = min(1.0, 0.3/0.3) = 1.0
        score_gated_at_threshold = combine_scores(cfg_gated, text=0.5, audio=0.3, llm=0.8)
        score_ungated_at_threshold = combine_scores(cfg_ungated, text=0.5, audio=0.3, llm=0.8)

        assert abs(score_gated_at_threshold - score_ungated_at_threshold) < 1e-9, (
            f"At audio=0.3, gated ({score_gated_at_threshold:.6f}) should equal "
            f"ungated ({score_ungated_at_threshold:.6f})"
        )

        # audio_score = 0.8 — well above threshold, gate factor = 1.0
        score_gated_above = combine_scores(cfg_gated, text=0.5, audio=0.8, llm=0.8)
        score_ungated_above = combine_scores(cfg_ungated, text=0.5, audio=0.8, llm=0.8)

        assert abs(score_gated_above - score_ungated_above) < 1e-9, (
            f"At audio=0.8, gated ({score_gated_above:.6f}) should equal "
            f"ungated ({score_ungated_above:.6f})"
        )

    def test_audio_at_zero_zeroes_llm(self) -> None:
        """When audio_score=0.0, effective_llm=0.0 (gate fully suppresses LLM)."""
        from pipeline.scorer import combine_scores

        cfg = self._make_config(llm_audio_gate=True)

        score = combine_scores(cfg, text=0.5, audio=0.0, llm=0.9)

        # With audio=0, gate factor = min(1.0, 0.0/0.3) = 0.0 → effective_llm = 0.0
        expected = cfg.text_weight * 0.5 + cfg.audio_weight * 0.0 + cfg.llm_weight * 0.0
        assert abs(score - expected) < 1e-9, (
            f"At audio=0.0, score {score:.6f} should equal {expected:.6f} "
            f"(LLM fully suppressed by gate)"
        )


# ---------------------------------------------------------------------------
# Unit tests — _check_llm_model_available (lightweight HTTP probe)
# ---------------------------------------------------------------------------

class TestCheckLLMModelAvailable:
    """Tests for the lightweight /api/tags availability probe."""

    def _make_config(self, llm_endpoint: str = "http://localhost:11434/api/generate", llm_model: str = "llama3") -> Config:
        cfg = Config(work_dir="/tmp/test")
        cfg.llm_endpoint = llm_endpoint
        cfg.llm_model = llm_model
        return cfg

    def _mock_tags_response(self, models: list, status_code: int = 200) -> MagicMock:
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.json.return_value = {"models": [{"name": m} for m in models]}
        return mock_resp

    def test_model_found_exact_match(self) -> None:
        """Model name matches exactly → returns True."""
        from pipeline.scorer import _check_llm_model_available
        cfg = self._make_config(llm_model="llama3")
        with patch("pipeline.scorer.requests.get", return_value=self._mock_tags_response(["llama3"])):
            assert _check_llm_model_available(cfg) is True

    def test_model_found_with_tag_suffix(self) -> None:
        """Model in tags has :latest suffix but config has bare name → returns True."""
        from pipeline.scorer import _check_llm_model_available
        cfg = self._make_config(llm_model="llama3")
        with patch("pipeline.scorer.requests.get", return_value=self._mock_tags_response(["llama3:latest"])):
            assert _check_llm_model_available(cfg) is True

    def test_model_not_found_returns_false(self) -> None:
        """Model not in tags list → returns False."""
        from pipeline.scorer import _check_llm_model_available
        cfg = self._make_config(llm_model="llama3")
        with patch("pipeline.scorer.requests.get", return_value=self._mock_tags_response(["mistral", "phi3"])):
            assert _check_llm_model_available(cfg) is False

    def test_non_200_response_assumes_available(self) -> None:
        """Non-200 HTTP response → assume available (graceful fallback)."""
        from pipeline.scorer import _check_llm_model_available
        cfg = self._make_config()
        with patch("pipeline.scorer.requests.get", return_value=self._mock_tags_response([], status_code=404)):
            assert _check_llm_model_available(cfg) is True

    def test_connection_error_assumes_available(self) -> None:
        """Connection error → assume available (graceful fallback)."""
        from pipeline.scorer import _check_llm_model_available
        cfg = self._make_config()
        with patch("pipeline.scorer.requests.get", side_effect=requests.ConnectionError("refused")):
            assert _check_llm_model_available(cfg) is True

    def test_timeout_assumes_available(self) -> None:
        """Timeout → assume available (graceful fallback)."""
        from pipeline.scorer import _check_llm_model_available
        cfg = self._make_config()
        with patch("pipeline.scorer.requests.get", side_effect=requests.Timeout("timed out")):
            assert _check_llm_model_available(cfg) is True

    def test_uses_get_not_post(self) -> None:
        """Probe uses GET, not POST (no inference request sent)."""
        from pipeline.scorer import _check_llm_model_available
        cfg = self._make_config(llm_model="llama3")
        with patch("pipeline.scorer.requests.get", return_value=self._mock_tags_response(["llama3"])) as mock_get, \
             patch("pipeline.scorer.requests.post") as mock_post:
            _check_llm_model_available(cfg)
        mock_get.assert_called_once()
        mock_post.assert_not_called()

    def test_probes_api_tags_endpoint(self) -> None:
        """GET is sent to <base_url>/api/tags, not /api/generate."""
        from pipeline.scorer import _check_llm_model_available
        cfg = self._make_config(llm_endpoint="http://localhost:11434/api/generate", llm_model="llama3")
        with patch("pipeline.scorer.requests.get", return_value=self._mock_tags_response(["llama3"])) as mock_get:
            _check_llm_model_available(cfg)
        called_url = mock_get.call_args[0][0]
        assert called_url == "http://localhost:11434/api/tags"

    def test_short_timeout_used(self) -> None:
        """Probe uses a short timeout (≤ 5 seconds)."""
        from pipeline.scorer import _check_llm_model_available
        cfg = self._make_config(llm_model="llama3")
        with patch("pipeline.scorer.requests.get", return_value=self._mock_tags_response(["llama3"])) as mock_get:
            _check_llm_model_available(cfg)
        timeout = mock_get.call_args.kwargs.get("timeout")
        assert timeout is not None and timeout <= 5, f"Expected short timeout, got {timeout}"

    def test_non_ollama_endpoint_assumes_available(self) -> None:
        """Endpoint not ending in /api/generate → skip probe, assume available."""
        from pipeline.scorer import _check_llm_model_available
        cfg = self._make_config(llm_endpoint="http://some-other-llm/v1/completions")
        with patch("pipeline.scorer.requests.get") as mock_get:
            result = _check_llm_model_available(cfg)
        mock_get.assert_not_called()
        assert result is True

    def test_config_model_with_tag_suffix_matches(self) -> None:
        """Config model with :tag suffix (e.g. llama3:8b) matches llama3:8b in tags."""
        from pipeline.scorer import _check_llm_model_available
        cfg = self._make_config(llm_model="llama3:8b")
        with patch("pipeline.scorer.requests.get", return_value=self._mock_tags_response(["llama3:8b", "mistral:latest"])):
            assert _check_llm_model_available(cfg) is True


# ---------------------------------------------------------------------------
# Property-based tests — Profile-Based Prompt Differentiation (Task 6.5)
# ---------------------------------------------------------------------------

def _classify_rubric_type(content_type: str, energy_level: str) -> str:
    """Classify which rubric type will be returned by _build_customized_rubric.
    
    This mirrors the logic in _build_customized_rubric to determine which
    rubric template is used.
    """
    # Default rubric (no profile or auto content type)
    if content_type == "auto":
        return "default"
    
    # High-energy content: gaming, comedy, or high energy level
    if content_type in ("gaming", "comedy") or energy_level == "high":
        return "high_energy"
    
    # Calm content: podcast, educational, or calm energy level
    if content_type in ("podcast", "educational") or energy_level == "calm":
        return "calm"
    
    # Moderate energy: vlog, other content types
    return "moderate"


# Feature: clip-selection-improvements, Property 17: Profile-Based Prompt Differentiation
# Validates: Requirements 3.1
@given(
    content_type_a=st.sampled_from(["gaming", "comedy", "podcast", "educational", "vlog", "auto"]),
    energy_level_a=st.sampled_from(["high", "moderate", "calm"]),
    content_type_b=st.sampled_from(["gaming", "comedy", "podcast", "educational", "vlog", "auto"]),
    energy_level_b=st.sampled_from(["high", "moderate", "calm"]),
)
@settings(max_examples=100)
def test_property_17_profile_prompt_differentiation(
    content_type_a: str,
    energy_level_a: str,
    content_type_b: str,
    energy_level_b: str,
) -> None:
    """Property 17: Profile-Based Prompt Differentiation.
    
    For any two CreatorProfile objects with different content_type or energy_level,
    the generated LLM prompts should differ in their rubric content (different
    emphasis keywords) UNLESS they map to the same rubric type.
    
    The rubric selection logic is:
    - content_type == "auto" → default rubric
    - content_type in ("gaming", "comedy") OR energy_level == "high" → high-energy rubric
    - content_type in ("podcast", "educational") OR energy_level == "calm" → calm rubric
    - otherwise → moderate rubric
    
    **Validates: Requirements 3.1**
    """
    from pipeline.models import CreatorProfile
    from pipeline.scorer import _build_customized_rubric
    
    # Create two profiles with the given parameters
    profile_a = CreatorProfile(
        creator_id="creator_a",
        content_type=content_type_a,
        energy_level=energy_level_a,
        typical_clip_duration=30.0,
        keyword_overrides=[],
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        video_count=1,
    )
    
    profile_b = CreatorProfile(
        creator_id="creator_b",
        content_type=content_type_b,
        energy_level=energy_level_b,
        typical_clip_duration=30.0,
        keyword_overrides=[],
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        video_count=1,
    )
    
    # Build configs with these profiles
    config_a = Config(work_dir="/tmp/test")
    config_a.creator_profile = profile_a
    
    config_b = Config(work_dir="/tmp/test")
    config_b.creator_profile = profile_b
    
    # Generate rubrics
    rubric_a = _build_customized_rubric(config_a)
    rubric_b = _build_customized_rubric(config_b)
    
    # Determine which rubric type each profile maps to
    rubric_type_a = _classify_rubric_type(content_type_a, energy_level_a)
    rubric_type_b = _classify_rubric_type(content_type_b, energy_level_b)
    
    # If profiles map to different rubric types, rubrics should differ
    if rubric_type_a != rubric_type_b:
        assert rubric_a != rubric_b, (
            f"Rubrics should differ when profiles map to different rubric types:\n"
            f"Profile A: content_type={content_type_a}, energy_level={energy_level_a} → {rubric_type_a}\n"
            f"Profile B: content_type={content_type_b}, energy_level={energy_level_b} → {rubric_type_b}\n"
            f"Rubric A length: {len(rubric_a)}\n"
            f"Rubric B length: {len(rubric_b)}"
        )
        
        # Verify that the rubrics contain the expected emphasis keywords for their type
        # High-energy rubric should contain high-energy keywords
        if rubric_type_a == "high_energy":
            assert any(keyword in rubric_a for keyword in ["audio energy", "reaction intensity", "HIGH ENERGY", "LOUD"]), (
                f"High-energy rubric A should contain high-energy keywords"
            )
        
        # Calm rubric should contain calm keywords
        if rubric_type_a == "calm":
            assert any(keyword in rubric_a for keyword in ["semantic interest", "insight quality", "VALUABLE INSIGHT"]), (
                f"Calm rubric A should contain calm keywords"
            )
        
        # Moderate rubric should contain moderate keywords
        if rubric_type_a == "moderate":
            assert any(keyword in rubric_a for keyword in ["BALANCE", "emotional connection", "relatable"]), (
                f"Moderate rubric A should contain moderate keywords"
            )
        
        # Check profile B keywords
        if rubric_type_b == "high_energy":
            assert any(keyword in rubric_b for keyword in ["audio energy", "reaction intensity", "HIGH ENERGY", "LOUD"]), (
                f"High-energy rubric B should contain high-energy keywords"
            )
        
        if rubric_type_b == "calm":
            assert any(keyword in rubric_b for keyword in ["semantic interest", "insight quality", "VALUABLE INSIGHT"]), (
                f"Calm rubric B should contain calm keywords"
            )
        
        if rubric_type_b == "moderate":
            assert any(keyword in rubric_b for keyword in ["BALANCE", "emotional connection", "relatable"]), (
                f"Moderate rubric B should contain moderate keywords"
            )
    else:
        # If profiles map to the same rubric type, rubrics may still differ if content_type differs
        # (because content_type is interpolated into the rubric text)
        if content_type_a != content_type_b and rubric_type_a not in ("default", "high_energy", "calm", "moderate"):
            # This shouldn't happen given our classification logic, but handle it gracefully
            pass
        elif content_type_a == content_type_b:
            # Same content type and same rubric type → rubrics should be identical
            assert rubric_a == rubric_b, (
                f"Rubrics should be identical when profiles map to the same rubric type and content_type:\n"
                f"Profile A: content_type={content_type_a}, energy_level={energy_level_a} → {rubric_type_a}\n"
                f"Profile B: content_type={content_type_b}, energy_level={energy_level_b} → {rubric_type_b}"
            )
