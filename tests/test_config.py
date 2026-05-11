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


# ---------------------------------------------------------------------------
# ShortsConfig fields — defaults and overrides
# ---------------------------------------------------------------------------

class TestShortsConfigDefaults:
    """Verify that every ShortsConfig field has the expected default value."""

    def setup_method(self):
        self.cfg = _base_config()

    def test_shorts_enabled_default(self):
        assert self.cfg.shorts_enabled is False

    def test_shorts_width_default(self):
        assert self.cfg.shorts_width == 1080

    def test_shorts_height_default(self):
        assert self.cfg.shorts_height == 1920

    def test_facecam_top_fraction_default(self):
        assert self.cfg.facecam_top_fraction == pytest.approx(0.35)

    def test_facecam_detection_enabled_default(self):
        assert self.cfg.facecam_detection_enabled is True

    def test_facecam_sample_duration_default(self):
        assert self.cfg.facecam_sample_duration == pytest.approx(10.0)

    def test_facecam_min_area_fraction_default(self):
        assert self.cfg.facecam_min_area_fraction == pytest.approx(0.04)

    def test_facecam_max_area_fraction_default(self):
        assert self.cfg.facecam_max_area_fraction == pytest.approx(0.30)

    def test_subtitle_style_default(self):
        assert self.cfg.subtitle_style == "bubble"

    def test_subtitle_font_size_default(self):
        assert self.cfg.subtitle_font_size == 72

    def test_subtitle_font_name_default(self):
        assert self.cfg.subtitle_font_name == "Impact"

    def test_subtitle_primary_color_default(self):
        assert self.cfg.subtitle_primary_color == "&H00FFFFFF"

    def test_subtitle_outline_color_default(self):
        assert self.cfg.subtitle_outline_color == "&H00000000"

    def test_subtitle_highlight_color_default(self):
        assert self.cfg.subtitle_highlight_color == "&H0000FFFF"

    def test_subtitle_outline_width_default(self):
        assert self.cfg.subtitle_outline_width == pytest.approx(4.0)

    def test_subtitle_shadow_depth_default(self):
        assert self.cfg.subtitle_shadow_depth == pytest.approx(2.0)

    def test_subtitle_margin_bottom_default(self):
        assert self.cfg.subtitle_margin_bottom == 80

    def test_subtitle_words_per_group_default(self):
        assert self.cfg.subtitle_words_per_group == 3


class TestShortsConfigOverrides:
    """Verify that ShortsConfig fields can be overridden at construction time."""

    def test_shorts_enabled_override(self):
        cfg = _base_config(shorts_enabled=True)
        assert cfg.shorts_enabled is True

    def test_shorts_width_override(self):
        cfg = _base_config(shorts_width=720)
        assert cfg.shorts_width == 720

    def test_shorts_height_override(self):
        cfg = _base_config(shorts_height=1280)
        assert cfg.shorts_height == 1280

    def test_facecam_top_fraction_override(self):
        cfg = _base_config(facecam_top_fraction=0.5)
        assert cfg.facecam_top_fraction == pytest.approx(0.5)

    def test_facecam_detection_enabled_override(self):
        cfg = _base_config(facecam_detection_enabled=False)
        assert cfg.facecam_detection_enabled is False

    def test_facecam_sample_duration_override(self):
        cfg = _base_config(facecam_sample_duration=5.0)
        assert cfg.facecam_sample_duration == pytest.approx(5.0)

    def test_facecam_min_area_fraction_override(self):
        cfg = _base_config(facecam_min_area_fraction=0.10)
        assert cfg.facecam_min_area_fraction == pytest.approx(0.10)

    def test_facecam_max_area_fraction_override(self):
        cfg = _base_config(facecam_max_area_fraction=0.50)
        assert cfg.facecam_max_area_fraction == pytest.approx(0.50)

    def test_subtitle_style_override(self):
        cfg = _base_config(subtitle_style="karaoke")
        assert cfg.subtitle_style == "karaoke"

    def test_subtitle_font_size_override(self):
        cfg = _base_config(subtitle_font_size=48)
        assert cfg.subtitle_font_size == 48

    def test_subtitle_font_name_override(self):
        cfg = _base_config(subtitle_font_name="Arial")
        assert cfg.subtitle_font_name == "Arial"

    def test_subtitle_primary_color_override(self):
        cfg = _base_config(subtitle_primary_color="&H000000FF")
        assert cfg.subtitle_primary_color == "&H000000FF"

    def test_subtitle_outline_color_override(self):
        cfg = _base_config(subtitle_outline_color="&H00FFFFFF")
        assert cfg.subtitle_outline_color == "&H00FFFFFF"

    def test_subtitle_highlight_color_override(self):
        cfg = _base_config(subtitle_highlight_color="&H0000FF00")
        assert cfg.subtitle_highlight_color == "&H0000FF00"

    def test_subtitle_outline_width_override(self):
        cfg = _base_config(subtitle_outline_width=2.0)
        assert cfg.subtitle_outline_width == pytest.approx(2.0)

    def test_subtitle_shadow_depth_override(self):
        cfg = _base_config(subtitle_shadow_depth=0.0)
        assert cfg.subtitle_shadow_depth == pytest.approx(0.0)

    def test_subtitle_margin_bottom_override(self):
        cfg = _base_config(subtitle_margin_bottom=120)
        assert cfg.subtitle_margin_bottom == 120

    def test_subtitle_words_per_group_override(self):
        cfg = _base_config(subtitle_words_per_group=5)
        assert cfg.subtitle_words_per_group == 5

    def test_multiple_shorts_fields_override_together(self):
        """Multiple shorts fields can be overridden simultaneously."""
        cfg = _base_config(
            shorts_enabled=True,
            shorts_width=720,
            shorts_height=1280,
            subtitle_style="popup",
            subtitle_font_size=60,
        )
        assert cfg.shorts_enabled is True
        assert cfg.shorts_width == 720
        assert cfg.shorts_height == 1280
        assert cfg.subtitle_style == "popup"
        assert cfg.subtitle_font_size == 60


# ---------------------------------------------------------------------------
# Creator Profile Weight Adjustment
# ---------------------------------------------------------------------------

class TestCreatorProfileWeightAdjustment:
    """Test adjust_weights_for_creator_profile() method."""

    def test_high_energy_increases_audio_weight(self):
        """High-energy profile should increase audio_weight to 60%."""
        from pipeline.models import CreatorProfile
        
        cfg = _base_config(text_weight=0.5, audio_weight=0.5)
        cfg.creator_profile = CreatorProfile(
            creator_id="test_creator",
            content_type="gaming",
            energy_level="high",
            typical_clip_duration=35.0,
            keyword_overrides=[],
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            video_count=1
        )
        
        cfg.adjust_weights_for_creator_profile()
        
        # High-energy: 40% text, 60% audio
        assert cfg.text_weight == pytest.approx(0.40)
        assert cfg.audio_weight == pytest.approx(0.60)

    def test_calm_increases_text_weight(self):
        """Calm profile should increase text_weight to 60%."""
        from pipeline.models import CreatorProfile
        
        cfg = _base_config(text_weight=0.5, audio_weight=0.5)
        cfg.creator_profile = CreatorProfile(
            creator_id="test_creator",
            content_type="podcast",
            energy_level="calm",
            typical_clip_duration=60.0,
            keyword_overrides=[],
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            video_count=1
        )
        
        cfg.adjust_weights_for_creator_profile()
        
        # Calm: 60% text, 40% audio
        assert cfg.text_weight == pytest.approx(0.60)
        assert cfg.audio_weight == pytest.approx(0.40)

    def test_moderate_keeps_balanced_weights(self):
        """Moderate profile should keep balanced 50/50 weights."""
        from pipeline.models import CreatorProfile
        
        cfg = _base_config(text_weight=0.5, audio_weight=0.5)
        cfg.creator_profile = CreatorProfile(
            creator_id="test_creator",
            content_type="vlog",
            energy_level="moderate",
            typical_clip_duration=45.0,
            keyword_overrides=[],
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            video_count=1
        )
        
        cfg.adjust_weights_for_creator_profile()
        
        # Moderate: 50% text, 50% audio
        assert cfg.text_weight == pytest.approx(0.50)
        assert cfg.audio_weight == pytest.approx(0.50)

    def test_no_profile_no_adjustment(self):
        """No profile should leave weights unchanged."""
        cfg = _base_config(text_weight=0.5, audio_weight=0.5)
        cfg.creator_profile = None
        
        original_text = cfg.text_weight
        original_audio = cfg.audio_weight
        
        cfg.adjust_weights_for_creator_profile()
        
        assert cfg.text_weight == original_text
        assert cfg.audio_weight == original_audio

    def test_adjustment_with_llm_enabled(self):
        """Weight adjustment should work with LLM enabled (non-LLM budget is 0.6)."""
        from pipeline.models import CreatorProfile
        
        cfg = _base_config(
            llm_enabled=True,
            text_weight=0.35,
            audio_weight=0.25,
            llm_weight=0.4
        )
        cfg.creator_profile = CreatorProfile(
            creator_id="test_creator",
            content_type="gaming",
            energy_level="high",
            typical_clip_duration=35.0,
            keyword_overrides=[],
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            video_count=1
        )
        
        cfg.adjust_weights_for_creator_profile()
        
        # Non-LLM budget = 0.35 + 0.25 = 0.6
        # High-energy: 40% of 0.6 = 0.24 text, 60% of 0.6 = 0.36 audio
        assert cfg.text_weight == pytest.approx(0.24)
        assert cfg.audio_weight == pytest.approx(0.36)
        assert cfg.llm_weight == pytest.approx(0.4)  # LLM weight unchanged
        
        # Total should still sum to 1.0
        assert cfg.text_weight + cfg.audio_weight + cfg.llm_weight == pytest.approx(1.0)

    def test_adjustment_preserves_weight_sum(self):
        """Weight adjustment should preserve the total weight sum."""
        from pipeline.models import CreatorProfile
        
        cfg = _base_config(text_weight=0.7, audio_weight=0.3)
        cfg.creator_profile = CreatorProfile(
            creator_id="test_creator",
            content_type="gaming",
            energy_level="high",
            typical_clip_duration=35.0,
            keyword_overrides=[],
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            video_count=1
        )
        
        original_sum = cfg.text_weight + cfg.audio_weight + cfg.llm_weight
        
        cfg.adjust_weights_for_creator_profile()
        
        new_sum = cfg.text_weight + cfg.audio_weight + cfg.llm_weight
        assert new_sum == pytest.approx(original_sum)

    def test_unknown_energy_level_no_change(self):
        """Unknown energy_level should leave weights unchanged."""
        from pipeline.models import CreatorProfile
        
        cfg = _base_config(text_weight=0.5, audio_weight=0.5)
        cfg.creator_profile = CreatorProfile(
            creator_id="test_creator",
            content_type="vlog",
            energy_level="unknown",  # Invalid value
            typical_clip_duration=45.0,
            keyword_overrides=[],
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            video_count=1
        )
        
        original_text = cfg.text_weight
        original_audio = cfg.audio_weight
        
        cfg.adjust_weights_for_creator_profile()
        
        # Should remain unchanged for unknown energy level
        assert cfg.text_weight == original_text
        assert cfg.audio_weight == original_audio
