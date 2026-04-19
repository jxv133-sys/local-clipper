"""Tests for pipeline/scorer.py — audio scoring.

Covers:
- Property 5: Audio score normalized (subtask 5.1)
- Unit tests for compute_audio_score (subtask 5.2)
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import scipy.io.wavfile
from hypothesis import given, settings
from hypothesis import strategies as st

from pipeline.models import Segment
from pipeline.scorer import compute_audio_score, compute_spike_score, compute_burst_score

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000  # Hz


def write_wav(path: str, data: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Write a float32 numpy array as a WAV file."""
    scipy.io.wavfile.write(path, sample_rate, data.astype(np.float32))


def make_segment(start: float, end: float) -> Segment:
    return Segment(start=start, end=end, text="")


def sine_wave(duration: float, freq: float = 440.0, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a sine wave of the given duration (seconds)."""
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------

# Feature: video-highlight-generator, Property 5: Audio score is normalized
# Validates: Requirements 4.3, 4.5
@given(
    segments=st.lists(
        st.builds(
            Segment,
            start=st.floats(min_value=0.0, max_value=9.0, allow_nan=False),
            end=st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
            text=st.just(""),
        ),
        min_size=1,
        max_size=10,
    )
)
@settings(max_examples=100)
def test_audio_score_normalized(segments: list[Segment]) -> None:
    """Property 5: Audio score is normalized — every value in [0.0, 1.0].

    **Validates: Requirements 4.3, 4.5**
    """
    # Create a 10-second synthetic WAV with random audio
    duration = 10.0
    rng = np.random.default_rng(seed=42)
    audio_data = rng.uniform(-1.0, 1.0, int(duration * SAMPLE_RATE)).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name

    try:
        write_wav(wav_path, audio_data)
        scores = compute_audio_score(segments, wav_path)

        assert len(scores) == len(segments)
        for score in scores:
            assert 0.0 <= score <= 1.0, f"Score {score} is out of [0.0, 1.0]"
    finally:
        os.unlink(wav_path)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestComputeAudioScore:
    """Unit tests for compute_audio_score."""

    def test_known_rms_normalized(self, tmp_path) -> None:
        """Segment with known audio samples produces expected normalized RMS."""
        # Two segments: one with amplitude 1.0, one with amplitude 0.5
        # RMS of amplitude-1.0 sine = 1/sqrt(2) ≈ 0.707
        # RMS of amplitude-0.5 sine = 0.5/sqrt(2) ≈ 0.354
        # After normalization: segment 0 → 1.0, segment 1 → 0.5
        duration = 1.0
        n = int(duration * SAMPLE_RATE)
        t = np.linspace(0, duration, n, endpoint=False)

        seg0_audio = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)  # amplitude 1.0
        seg1_audio = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)  # amplitude 0.5

        # Concatenate: segment 0 at [0, 1s), segment 1 at [1s, 2s)
        full_audio = np.concatenate([seg0_audio, seg1_audio])
        wav_path = str(tmp_path / "test.wav")
        write_wav(wav_path, full_audio)

        segments = [make_segment(0.0, 1.0), make_segment(1.0, 2.0)]
        scores = compute_audio_score(segments, wav_path)

        assert len(scores) == 2
        # Segment 0 has the highest RMS → normalized to 1.0
        assert abs(scores[0] - 1.0) < 1e-5
        # Segment 1 has half the amplitude → RMS is half → normalized to ~0.5
        assert abs(scores[1] - 0.5) < 1e-5

    def test_silent_segment_scores_zero(self, tmp_path) -> None:
        """Segment with all-zero audio samples receives audio score 0.0."""
        duration = 2.0
        n = int(duration * SAMPLE_RATE)
        # First second: sine wave; second second: silence
        t = np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False)
        audio = np.concatenate([
            np.sin(2 * np.pi * 440.0 * t).astype(np.float32),
            np.zeros(SAMPLE_RATE, dtype=np.float32),
        ])
        wav_path = str(tmp_path / "test.wav")
        write_wav(wav_path, audio)

        segments = [make_segment(0.0, 1.0), make_segment(1.0, 2.0)]
        scores = compute_audio_score(segments, wav_path)

        assert len(scores) == 2
        assert scores[1] == 0.0  # silent segment

    def test_segment_outside_wav_duration_scores_zero(self, tmp_path) -> None:
        """Segment whose time range is entirely outside the WAV duration scores 0.0."""
        duration = 1.0
        audio = sine_wave(duration)
        wav_path = str(tmp_path / "test.wav")
        write_wav(wav_path, audio)

        # Segment starts after the WAV ends
        segments = [make_segment(0.0, 1.0), make_segment(5.0, 6.0)]
        scores = compute_audio_score(segments, wav_path)

        assert len(scores) == 2
        assert scores[1] == 0.0  # out-of-range segment

    def test_all_silent_no_divide_by_zero(self, tmp_path) -> None:
        """All-silent audio returns all scores 0.0 without divide-by-zero error."""
        duration = 2.0
        audio = np.zeros(int(duration * SAMPLE_RATE), dtype=np.float32)
        wav_path = str(tmp_path / "test.wav")
        write_wav(wav_path, audio)

        segments = [make_segment(0.0, 1.0), make_segment(1.0, 2.0)]
        scores = compute_audio_score(segments, wav_path)

        assert len(scores) == 2
        assert all(s == 0.0 for s in scores)

    def test_single_segment_scores_one(self, tmp_path) -> None:
        """A single non-silent segment is normalized to 1.0."""
        audio = sine_wave(1.0)
        wav_path = str(tmp_path / "test.wav")
        write_wav(wav_path, audio)

        segments = [make_segment(0.0, 1.0)]
        scores = compute_audio_score(segments, wav_path)

        assert len(scores) == 1
        assert abs(scores[0] - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Unit tests for compute_spike_score
# ---------------------------------------------------------------------------

class TestComputeSpikeScore:
    """Unit tests for compute_spike_score."""

    def test_normal_segment_no_spike(self, tmp_path) -> None:
        """A segment with similar RMS to its baseline produces a low spike score."""
        # 35 seconds of uniform sine wave — baseline and segment have the same energy
        duration = 35.0
        n = int(duration * SAMPLE_RATE)
        t = np.linspace(0, duration, n, endpoint=False)
        audio = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
        wav_path = str(tmp_path / "uniform.wav")
        write_wav(wav_path, audio)

        # Segment at [30s, 35s] — baseline is [0s, 30s], same amplitude
        segments = [make_segment(30.0, 35.0)]
        scores = compute_spike_score(segments, wav_path)

        assert len(scores) == 1
        # ratio ≈ 1.0 → score ≈ 0.0
        assert scores[0] < 0.1, f"Expected near-zero spike score, got {scores[0]}"

    def test_spike_segment_ratio_above_3x(self, tmp_path) -> None:
        """A segment with RMS > 3x baseline produces a spike score near 1.0."""
        # 30s of quiet baseline (amplitude 0.1) + 5s of loud burst (amplitude 1.0)
        baseline_duration = 30.0
        burst_duration = 5.0
        n_baseline = int(baseline_duration * SAMPLE_RATE)
        n_burst = int(burst_duration * SAMPLE_RATE)

        t_baseline = np.linspace(0, baseline_duration, n_baseline, endpoint=False)
        t_burst = np.linspace(0, burst_duration, n_burst, endpoint=False)

        baseline_audio = (0.1 * np.sin(2 * np.pi * 440.0 * t_baseline)).astype(np.float32)
        burst_audio = np.sin(2 * np.pi * 440.0 * t_burst).astype(np.float32)

        full_audio = np.concatenate([baseline_audio, burst_audio])
        wav_path = str(tmp_path / "spike.wav")
        write_wav(wav_path, full_audio)

        # Segment is the burst at [30s, 35s]
        segments = [make_segment(30.0, 35.0)]
        scores = compute_spike_score(segments, wav_path)

        assert len(scores) == 1
        # burst RMS ≈ 0.707, baseline RMS ≈ 0.0707 → ratio ≈ 10x → score = 1.0
        assert scores[0] >= 0.9, f"Expected spike score near 1.0, got {scores[0]}"

    def test_silent_baseline_with_audio_segment(self, tmp_path) -> None:
        """A segment preceded by silence (baseline_rms == 0) scores 1.0."""
        # 30s of silence followed by 5s of audio
        n_silence = int(30.0 * SAMPLE_RATE)
        n_burst = int(5.0 * SAMPLE_RATE)
        t_burst = np.linspace(0, 5.0, n_burst, endpoint=False)

        silence = np.zeros(n_silence, dtype=np.float32)
        burst = np.sin(2 * np.pi * 440.0 * t_burst).astype(np.float32)

        full_audio = np.concatenate([silence, burst])
        wav_path = str(tmp_path / "silence_then_burst.wav")
        write_wav(wav_path, full_audio)

        segments = [make_segment(30.0, 35.0)]
        scores = compute_spike_score(segments, wav_path)

        assert len(scores) == 1
        assert scores[0] == 1.0, f"Expected 1.0 for silence→burst, got {scores[0]}"

    def test_all_silent_scores_zero(self, tmp_path) -> None:
        """All-silent audio produces spike scores of 0.0 without errors."""
        duration = 35.0
        audio = np.zeros(int(duration * SAMPLE_RATE), dtype=np.float32)
        wav_path = str(tmp_path / "silent.wav")
        write_wav(wav_path, audio)

        segments = [make_segment(0.0, 5.0), make_segment(30.0, 35.0)]
        scores = compute_spike_score(segments, wav_path)

        assert len(scores) == 2
        assert all(s == 0.0 for s in scores), f"Expected all zeros, got {scores}"

    def test_empty_segments_returns_empty(self, tmp_path) -> None:
        """Empty segment list returns empty list."""
        audio = sine_wave(5.0)
        wav_path = str(tmp_path / "test.wav")
        write_wav(wav_path, audio)

        scores = compute_spike_score([], wav_path)
        assert scores == []

    def test_scores_normalized_to_unit_interval(self, tmp_path) -> None:
        """All spike scores are in [0.0, 1.0]."""
        # Mix of quiet and loud segments
        n_quiet = int(5.0 * SAMPLE_RATE)
        n_loud = int(5.0 * SAMPLE_RATE)
        t = np.linspace(0, 5.0, n_quiet, endpoint=False)

        quiet = (0.05 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        loud = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

        # 30s baseline + quiet + loud
        baseline = np.zeros(int(30.0 * SAMPLE_RATE), dtype=np.float32)
        full_audio = np.concatenate([baseline, quiet, loud])
        wav_path = str(tmp_path / "mixed.wav")
        write_wav(wav_path, full_audio)

        segments = [
            make_segment(30.0, 35.0),  # quiet segment
            make_segment(35.0, 40.0),  # loud segment
        ]
        scores = compute_spike_score(segments, wav_path)

        assert len(scores) == 2
        for score in scores:
            assert 0.0 <= score <= 1.0, f"Score {score} out of [0.0, 1.0]"

    def test_segment_at_start_no_baseline(self, tmp_path) -> None:
        """A segment at t=0 has no baseline window — treated as silence before it."""
        # Segment starts at 0, so baseline window is [0, 0] → empty → rms=0
        audio = sine_wave(5.0)
        wav_path = str(tmp_path / "test.wav")
        write_wav(wav_path, audio)

        segments = [make_segment(0.0, 5.0)]
        scores = compute_spike_score(segments, wav_path)

        assert len(scores) == 1
        # baseline is empty (0s window) → rms=0 → silence-before-burst → score=1.0
        assert scores[0] == 1.0


# ---------------------------------------------------------------------------
# Unit tests for compute_burst_score
# ---------------------------------------------------------------------------

class TestComputeBurstScore:
    """Unit tests for compute_burst_score (silence-then-burst detection).

    The algorithm:
    - global_rms_mean = mean of per-segment RMS values (non-zero only)
    - global_rms_max  = max of per-segment RMS values
    - For each segment:
        silence_before = avg RMS of the 5s window immediately before segment start
        burst_score = 1.0 if silence_before < 0.1 * global_rms_mean
                              AND segment_rms > 0.5 * global_rms_max
                    else 0.0
    """

    def _make_wav(self, tmp_path, sections: list[tuple[float, float]]) -> tuple[str, list[Segment]]:
        """Build a WAV file from (amplitude, duration_seconds) pairs.

        Returns (wav_path, segments) where each segment spans exactly one section.
        """
        chunks = []
        segments = []
        t_cursor = 0.0
        for amp, dur in sections:
            n = int(dur * SAMPLE_RATE)
            t = np.linspace(0, dur, n, endpoint=False)
            chunk = (amp * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
            chunks.append(chunk)
            segments.append(make_segment(t_cursor, t_cursor + dur))
            t_cursor += dur
        audio = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
        wav_path = str(tmp_path / "burst_test.wav")
        write_wav(wav_path, audio)
        return wav_path, segments

    def test_burst_after_silence_scores_one(self, tmp_path) -> None:
        """A segment preceded by silence and that is loud gets burst_score = 1.0.

        Layout:
          [0s–5s]  silence (amp=0.0)   — the 5s pre-window before the loud segment
          [5s–10s] loud audio (amp=1.0) — the segment under test

        global_rms_mean ≈ RMS of loud segment (only non-zero segment)
        global_rms_max  ≈ RMS of loud segment
        silence_before  = RMS of [0s–5s] = 0.0  < 0.1 * global_rms_mean  ✓
        segment_rms     ≈ 0.707                  > 0.5 * global_rms_max   ✓
        → burst_score = 1.0
        """
        wav_path, segments = self._make_wav(tmp_path, [
            (0.0, 5.0),   # silence section (becomes the pre-window)
            (1.0, 5.0),   # loud segment
        ])
        # Only score the loud segment (index 1)
        scores = compute_burst_score([segments[1]], wav_path)
        assert len(scores) == 1
        assert scores[0] == 1.0, f"Expected 1.0 for silence→loud, got {scores[0]}"

    def test_normal_audio_before_segment_scores_zero(self, tmp_path) -> None:
        """A segment with normal (non-silent) audio before it gets burst_score = 0.0.

        Layout:
          [0s–5s]  moderate audio (amp=0.8) — pre-window is NOT silent
          [5s–10s] loud audio (amp=1.0)     — segment under test

        silence_before ≈ 0.566  which is NOT < 0.1 * global_rms_mean
        → burst_score = 0.0
        """
        wav_path, segments = self._make_wav(tmp_path, [
            (0.8, 5.0),   # moderate pre-window
            (1.0, 5.0),   # loud segment
        ])
        scores = compute_burst_score([segments[1]], wav_path)
        assert len(scores) == 1
        assert scores[0] == 0.0, f"Expected 0.0 for non-silent pre-window, got {scores[0]}"

    def test_loud_but_not_preceded_by_silence_scores_zero(self, tmp_path) -> None:
        """A loud segment NOT preceded by silence gets burst_score = 0.0.

        Layout:
          [0s–5s]  loud audio (amp=1.0)  — pre-window is loud, not silent
          [5s–10s] loud audio (amp=1.0)  — segment under test

        silence_before ≈ 0.707  which is NOT < 0.1 * global_rms_mean
        → burst_score = 0.0
        """
        wav_path, segments = self._make_wav(tmp_path, [
            (1.0, 5.0),   # loud pre-window
            (1.0, 5.0),   # loud segment
        ])
        scores = compute_burst_score([segments[1]], wav_path)
        assert len(scores) == 1
        assert scores[0] == 0.0, f"Expected 0.0 for loud pre-window, got {scores[0]}"

    def test_all_silent_audio_scores_zero(self, tmp_path) -> None:
        """All-silent audio produces burst_score = 0.0 for all segments."""
        duration = 10.0
        audio = np.zeros(int(duration * SAMPLE_RATE), dtype=np.float32)
        wav_path = str(tmp_path / "silent.wav")
        write_wav(wav_path, audio)

        segments = [make_segment(0.0, 5.0), make_segment(5.0, 10.0)]
        scores = compute_burst_score(segments, wav_path)

        assert len(scores) == 2
        assert all(s == 0.0 for s in scores), f"Expected all zeros, got {scores}"

    def test_empty_segment_list_returns_empty(self, tmp_path) -> None:
        """Empty segment list returns empty list."""
        audio = sine_wave(5.0)
        wav_path = str(tmp_path / "test.wav")
        write_wav(wav_path, audio)

        scores = compute_burst_score([], wav_path)
        assert scores == []

    def test_burst_score_is_binary(self, tmp_path) -> None:
        """burst_score is always exactly 0.0 or 1.0 (binary)."""
        # Mix: silence then loud, then moderate throughout
        wav_path, segments = self._make_wav(tmp_path, [
            (0.0, 5.0),   # silence
            (1.0, 5.0),   # loud burst
            (0.5, 5.0),   # moderate
            (0.3, 5.0),   # quiet-ish
        ])
        scores = compute_burst_score(segments, wav_path)

        assert len(scores) == len(segments)
        for score in scores:
            assert score in (0.0, 1.0), f"Expected binary score, got {score}"
