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
from pipeline.scorer import compute_audio_score

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
