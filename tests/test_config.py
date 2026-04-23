"""Unit tests for Config.__post_init__ validation."""

from __future__ import annotations

import pytest

from config import Config


def _base_config(**kwargs) -> Config:
    """Return a valid Config with work_dir set, overriding any fields via kwargs."""
    return Config(work_dir="/tmp/test", **kwargs)


# ---------------------------------------------------------------------------
# Weight sum validation (only when llm_enabled=True)
# ---------------------------------------------------------------------------

class TestWeightSumValidation:
    def test_valid_weights_when_llm_enabled(self):
        """Weights summing to 1.0 with LLM enabled should not raise."""
        cfg = _base_config(llm_enabled=True, text_weight=0.35, audio_weight=0.25, llm_weight=0.4)
        assert cfg.text_weight + cfg.audio_weight + cfg.llm_weight == pytest.approx(1.0)

    def test_invalid_weights_when_llm_enabled_raises(self):
        """Weights not summing to 1.0 with LLM enabled should raise ValueError."""
        with pytest.raises(ValueError, match="text_weight \\+ audio_weight \\+ llm_weight"):
            _base_config(llm_enabled=True, text_weight=0.5, audio_weight=0.5, llm_weight=0.4)

    def test_weights_not_validated_when_llm_disabled(self):
        """Weight sum is not checked when llm_enabled=False (default)."""
        # Default config has text=0.5, audio=0.5, llm=0.0 — sum is 1.0 anyway,
        # but even an invalid sum should be allowed when LLM is off.
        cfg = _base_config(llm_enabled=False, text_weight=0.9, audio_weight=0.9, llm_weight=0.0)
        assert cfg.llm_enabled is False  # no exception raised

    def test_weights_within_tolerance_accepted(self):
        """Weights within 1e-9 tolerance of 1.0 should be accepted."""
        # Floating-point arithmetic can produce tiny deviations
        cfg = _base_config(
            llm_enabled=True,
            text_weight=1 / 3,
            audio_weight=1 / 3,
            llm_weight=1 / 3,
        )
        # Should not raise (sum is 1.0 within floating-point precision)
        assert cfg.llm_enabled is True


# ---------------------------------------------------------------------------
# min_clip_duration <= max_clip_duration
# ---------------------------------------------------------------------------

class TestClipDurationValidation:
    def test_valid_clip_durations(self):
        """min <= max should not raise."""
        cfg = _base_config(min_clip_duration=10.0, max_clip_duration=60.0)
        assert cfg.min_clip_duration <= cfg.max_clip_duration

    def test_equal_clip_durations_valid(self):
        """min == max is allowed."""
        cfg = _base_config(min_clip_duration=30.0, max_clip_duration=30.0)
        assert cfg.min_clip_duration == cfg.max_clip_duration

    def test_min_greater_than_max_raises(self):
        """min > max should raise ValueError."""
        with pytest.raises(ValueError, match="min_clip_duration"):
            _base_config(min_clip_duration=60.0, max_clip_duration=30.0)


# ---------------------------------------------------------------------------
# 0.0 <= llm_audio_spike_percentage <= 1.0
# ---------------------------------------------------------------------------

class TestLlmAudioSpikePercentageValidation:
    def test_valid_spike_percentage_zero(self):
        """0.0 is a valid spike percentage."""
        cfg = _base_config(llm_audio_spike_percentage=0.0)
        assert cfg.llm_audio_spike_percentage == 0.0

    def test_valid_spike_percentage_one(self):
        """1.0 is a valid spike percentage."""
        cfg = _base_config(llm_audio_spike_percentage=1.0)
        assert cfg.llm_audio_spike_percentage == 1.0

    def test_valid_spike_percentage_middle(self):
        """0.5 is a valid spike percentage."""
        cfg = _base_config(llm_audio_spike_percentage=0.5)
        assert cfg.llm_audio_spike_percentage == 0.5

    def test_negative_spike_percentage_raises(self):
        """Negative value should raise ValueError."""
        with pytest.raises(ValueError, match="llm_audio_spike_percentage"):
            _base_config(llm_audio_spike_percentage=-0.1)

    def test_spike_percentage_above_one_raises(self):
        """Value > 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="llm_audio_spike_percentage"):
            _base_config(llm_audio_spike_percentage=1.1)


# ---------------------------------------------------------------------------
# audio_percentile_low < audio_percentile_high
# ---------------------------------------------------------------------------

class TestAudioPercentileValidation:
    def test_valid_percentiles(self):
        """low < high should not raise."""
        cfg = _base_config(audio_percentile_low=5.0, audio_percentile_high=95.0)
        assert cfg.audio_percentile_low < cfg.audio_percentile_high

    def test_equal_percentiles_raises(self):
        """low == high should raise ValueError."""
        with pytest.raises(ValueError, match="audio_percentile_low"):
            _base_config(audio_percentile_low=50.0, audio_percentile_high=50.0)

    def test_low_greater_than_high_raises(self):
        """low > high should raise ValueError."""
        with pytest.raises(ValueError, match="audio_percentile_low"):
            _base_config(audio_percentile_low=90.0, audio_percentile_high=10.0)


# ---------------------------------------------------------------------------
# excitement_volume_weight + excitement_pitch_weight == 1.0
# ---------------------------------------------------------------------------

class TestExcitementWeightValidation:
    def test_valid_excitement_weights(self):
        """Weights summing to 1.0 should not raise."""
        cfg = _base_config(excitement_volume_weight=0.6, excitement_pitch_weight=0.4)
        assert cfg.excitement_volume_weight + cfg.excitement_pitch_weight == pytest.approx(1.0)

    def test_excitement_weights_not_summing_to_one_raises(self):
        """Weights not summing to 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="excitement_volume_weight \\+ excitement_pitch_weight"):
            _base_config(excitement_volume_weight=0.5, excitement_pitch_weight=0.5 + 0.1)

    def test_excitement_weights_zero_and_one(self):
        """0.0 + 1.0 = 1.0 is valid."""
        cfg = _base_config(excitement_volume_weight=0.0, excitement_pitch_weight=1.0)
        assert cfg.excitement_volume_weight + cfg.excitement_pitch_weight == pytest.approx(1.0)

    def test_excitement_weights_within_tolerance(self):
        """Weights within 1e-9 of 1.0 should be accepted."""
        cfg = _base_config(excitement_volume_weight=0.5, excitement_pitch_weight=0.5)
        assert cfg.excitement_volume_weight + cfg.excitement_pitch_weight == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Default Config is always valid
# ---------------------------------------------------------------------------

class TestDefaultConfigIsValid:
    def test_default_config_does_not_raise(self, tmp_path):
        """The default Config (with only work_dir) should always be valid."""
        cfg = Config(work_dir=str(tmp_path))
        assert cfg is not None
