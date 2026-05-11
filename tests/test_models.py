"""Tests for pipeline data models, including property-based tests."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pipeline.models import (
    CanvasLayout,
    CreatorProfile,
    EmotionFeatures,
    EngagementFeatures,
    FacecamRegion,
    FilterFragment,
    NaturalPause,
    Segment,
    SubtitleStyle,
    Transcript,
)
from pipeline.exceptions import PipelineError, ShortsFormattingError


# ---------------------------------------------------------------------------
# Property 1: Transcript serialization round-trip
# Validates: Requirements 2.7
# ---------------------------------------------------------------------------

segment_strategy = st.builds(
    Segment,
    start=st.floats(min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
    end=st.floats(min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
    text=st.text(),
)

transcript_strategy = st.builds(
    Transcript,
    segments=st.lists(segment_strategy),
)


creator_profile_strategy = st.builds(
    CreatorProfile,
    creator_id=st.text(min_size=1),
    content_type=st.sampled_from(["gaming", "podcast", "comedy", "vlog", "educational"]),
    energy_level=st.sampled_from(["high", "moderate", "calm"]),
    typical_clip_duration=st.floats(min_value=10.0, max_value=120.0, allow_nan=False, allow_infinity=False),
    keyword_overrides=st.lists(st.text()),
    created_at=st.text(),
    updated_at=st.text(),
    video_count=st.integers(min_value=0, max_value=10000),
)


# Feature: video-highlight-generator, Property 1: Transcript serialization round-trip
@given(transcript=transcript_strategy)
@settings(max_examples=100)
def test_transcript_roundtrip(transcript: Transcript) -> None:
    """For any valid Transcript, serializing to dict and deserializing back
    SHALL produce a Transcript that is structurally and value-equivalent to
    the original.

    Validates: Requirements 2.7
    """
    result = Transcript.from_dict(transcript.to_dict())
    assert result == transcript


# ---------------------------------------------------------------------------
# Property 2: CreatorProfile serialization round-trip
# Validates: Requirements 1.1, 1.2, 1.3, 1.4, 15.1, 15.2
# ---------------------------------------------------------------------------

# Feature: clip-selection-improvements, Property 2: CreatorProfile serialization round-trip
@given(profile=creator_profile_strategy)
@settings(max_examples=100)
def test_creator_profile_roundtrip_property(profile: CreatorProfile) -> None:
    """For any valid CreatorProfile, serializing to dict and deserializing back
    SHALL produce a CreatorProfile that is structurally and value-equivalent to
    the original.

    **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 15.1, 15.2**
    """
    result = CreatorProfile.from_dict(profile.to_dict())
    assert result == profile


# ---------------------------------------------------------------------------
# Unit tests for Transcript serialization
# ---------------------------------------------------------------------------

def test_transcript_to_dict_empty() -> None:
    """An empty Transcript serializes to a dict with an empty segments list."""
    t = Transcript(segments=[])
    assert t.to_dict() == {"segments": []}


def test_transcript_to_dict_single_segment() -> None:
    """A Transcript with one segment serializes correctly."""
    seg = Segment(start=1.0, end=2.5, text="hello world")
    t = Transcript(segments=[seg])
    d = t.to_dict()
    assert d == {"segments": [{"start": 1.0, "end": 2.5, "text": "hello world", "words": []}]}


def test_transcript_from_dict_empty() -> None:
    """Deserializing an empty segments dict produces an empty Transcript."""
    t = Transcript.from_dict({"segments": []})
    assert t == Transcript(segments=[])


def test_transcript_from_dict_single_segment() -> None:
    """Deserializing a dict with one segment produces the correct Transcript."""
    d = {"segments": [{"start": 0.0, "end": 3.0, "text": "test"}]}
    t = Transcript.from_dict(d)
    assert t.segments == [Segment(start=0.0, end=3.0, text="test")]


def test_transcript_roundtrip_manual() -> None:
    """Manual round-trip test with multiple segments."""
    segments = [
        Segment(start=0.0, end=1.5, text="first"),
        Segment(start=1.5, end=3.0, text="second"),
        Segment(start=3.0, end=5.0, text=""),
    ]
    t = Transcript(segments=segments)
    assert Transcript.from_dict(t.to_dict()) == t


def test_transcript_equality() -> None:
    """Two Transcripts with identical segments are equal."""
    seg = Segment(start=0.0, end=1.0, text="hi")
    assert Transcript(segments=[seg]) == Transcript(segments=[Segment(start=0.0, end=1.0, text="hi")])


def test_transcript_inequality() -> None:
    """Two Transcripts with different segments are not equal."""
    t1 = Transcript(segments=[Segment(start=0.0, end=1.0, text="a")])
    t2 = Transcript(segments=[Segment(start=0.0, end=1.0, text="b")])
    assert t1 != t2


# ---------------------------------------------------------------------------
# FacecamRegion tests
# ---------------------------------------------------------------------------

def test_facecam_region_instantiation() -> None:
    """FacecamRegion can be instantiated with all required fields."""
    region = FacecamRegion(x=10, y=20, width=300, height=200, corner="top-left", confidence=0.95)
    assert region is not None


def test_facecam_region_fields_accessible() -> None:
    """All FacecamRegion fields are accessible."""
    region = FacecamRegion(x=10, y=20, width=300, height=200, corner="top-right", confidence=0.8)
    assert region.x == 10
    assert region.y == 20
    assert region.width == 300
    assert region.height == 200
    assert region.corner == "top-right"
    assert region.confidence == 0.8


def test_facecam_region_confidence_is_float() -> None:
    """FacecamRegion.confidence is a float."""
    region = FacecamRegion(x=0, y=0, width=100, height=100, corner="bottom-left", confidence=0.5)
    assert isinstance(region.confidence, float)


# ---------------------------------------------------------------------------
# CanvasLayout tests
# ---------------------------------------------------------------------------

def test_canvas_layout_instantiation() -> None:
    """CanvasLayout can be instantiated with all required fields."""
    layout = CanvasLayout(
        canvas_width=1080,
        canvas_height=1920,
        facecam_x=0,
        facecam_y=0,
        facecam_width=1080,
        facecam_height=672,
        gameplay_x=0,
        gameplay_y=672,
        gameplay_width=1080,
        gameplay_height=1248,
    )
    assert layout is not None


def test_canvas_layout_all_fields_accessible() -> None:
    """All 10 CanvasLayout fields are accessible."""
    layout = CanvasLayout(
        canvas_width=1080,
        canvas_height=1920,
        facecam_x=0,
        facecam_y=0,
        facecam_width=1080,
        facecam_height=672,
        gameplay_x=0,
        gameplay_y=672,
        gameplay_width=1080,
        gameplay_height=1248,
    )
    assert layout.canvas_width == 1080
    assert layout.canvas_height == 1920
    assert layout.facecam_x == 0
    assert layout.facecam_y == 0
    assert layout.facecam_width == 1080
    assert layout.facecam_height == 672
    assert layout.gameplay_x == 0
    assert layout.gameplay_y == 672
    assert layout.gameplay_width == 1080
    assert layout.gameplay_height == 1248


# ---------------------------------------------------------------------------
# FilterFragment tests
# ---------------------------------------------------------------------------

def test_filter_fragment_instantiation() -> None:
    """FilterFragment can be instantiated with required fields."""
    frag = FilterFragment(
        filter_str="scale=1080:1920",
        input_label="[v0]",
        output_label="[canvas]",
    )
    assert frag is not None


def test_filter_fragment_extra_inputs_defaults_to_empty_list() -> None:
    """FilterFragment.extra_inputs defaults to an empty list."""
    frag = FilterFragment(
        filter_str="scale=1080:1920",
        input_label="[v0]",
        output_label="[canvas]",
    )
    assert frag.extra_inputs == []


def test_filter_fragment_extra_inputs_can_be_set() -> None:
    """FilterFragment.extra_inputs can be set to a list of strings."""
    frag = FilterFragment(
        filter_str="overlay",
        input_label="[base]",
        output_label="[out]",
        extra_inputs=["/path/to/overlay.png", "/path/to/mask.png"],
    )
    assert frag.extra_inputs == ["/path/to/overlay.png", "/path/to/mask.png"]


# ---------------------------------------------------------------------------
# SubtitleStyle tests
# ---------------------------------------------------------------------------

def test_subtitle_style_has_expected_members() -> None:
    """SubtitleStyle has BUBBLE, POPUP, HIGHLIGHT, KARAOKE values."""
    assert hasattr(SubtitleStyle, "BUBBLE")
    assert hasattr(SubtitleStyle, "POPUP")
    assert hasattr(SubtitleStyle, "HIGHLIGHT")
    assert hasattr(SubtitleStyle, "KARAOKE")


def test_subtitle_style_values() -> None:
    """SubtitleStyle enum values are the expected strings."""
    assert SubtitleStyle.BUBBLE.value == "bubble"
    assert SubtitleStyle.POPUP.value == "popup"
    assert SubtitleStyle.HIGHLIGHT.value == "highlight"
    assert SubtitleStyle.KARAOKE.value == "karaoke"


def test_subtitle_style_construct_from_string() -> None:
    """SubtitleStyle can be constructed from its string value."""
    assert SubtitleStyle("bubble") == SubtitleStyle.BUBBLE
    assert SubtitleStyle("popup") == SubtitleStyle.POPUP
    assert SubtitleStyle("highlight") == SubtitleStyle.HIGHLIGHT
    assert SubtitleStyle("karaoke") == SubtitleStyle.KARAOKE


# ---------------------------------------------------------------------------
# ShortsFormattingError tests
# ---------------------------------------------------------------------------

def test_shorts_formatting_error_is_subclass_of_pipeline_error() -> None:
    """ShortsFormattingError is a subclass of PipelineError."""
    assert issubclass(ShortsFormattingError, PipelineError)


def test_shorts_formatting_error_can_be_caught_as_pipeline_error() -> None:
    """ShortsFormattingError can be raised and caught as PipelineError."""
    with pytest.raises(PipelineError):
        raise ShortsFormattingError("ffmpeg command failed")


def test_shorts_formatting_error_carries_message() -> None:
    """ShortsFormattingError carries the provided message."""
    msg = "ffmpeg exited with code 1"
    err = ShortsFormattingError(msg)
    assert str(err) == msg


# ---------------------------------------------------------------------------
# CreatorProfile tests
# ---------------------------------------------------------------------------

def test_creator_profile_instantiation() -> None:
    """CreatorProfile can be instantiated with all required fields."""
    profile = CreatorProfile(
        creator_id="test_creator",
        content_type="gaming",
        energy_level="high",
        typical_clip_duration=35.0,
        keyword_overrides=["clutch", "gg"],
        created_at="2024-01-15T10:30:00Z",
        updated_at="2024-01-20T14:45:00Z",
        video_count=5,
    )
    assert profile is not None


def test_creator_profile_to_dict() -> None:
    """CreatorProfile.to_dict() serializes all fields correctly."""
    profile = CreatorProfile(
        creator_id="test_creator",
        content_type="podcast",
        energy_level="calm",
        typical_clip_duration=45.0,
        keyword_overrides=["interesting", "fascinating"],
        created_at="2024-01-15T10:30:00Z",
        updated_at="2024-01-20T14:45:00Z",
        video_count=10,
    )
    d = profile.to_dict()
    assert d == {
        "creator_id": "test_creator",
        "content_type": "podcast",
        "energy_level": "calm",
        "typical_clip_duration": 45.0,
        "keyword_overrides": ["interesting", "fascinating"],
        "created_at": "2024-01-15T10:30:00Z",
        "updated_at": "2024-01-20T14:45:00Z",
        "video_count": 10,
    }


def test_creator_profile_from_dict() -> None:
    """CreatorProfile.from_dict() deserializes correctly."""
    d = {
        "creator_id": "test_creator",
        "content_type": "comedy",
        "energy_level": "moderate",
        "typical_clip_duration": 30.0,
        "keyword_overrides": ["funny", "hilarious"],
        "created_at": "2024-01-15T10:30:00Z",
        "updated_at": "2024-01-20T14:45:00Z",
        "video_count": 3,
    }
    profile = CreatorProfile.from_dict(d)
    assert profile.creator_id == "test_creator"
    assert profile.content_type == "comedy"
    assert profile.energy_level == "moderate"
    assert profile.typical_clip_duration == 30.0
    assert profile.keyword_overrides == ["funny", "hilarious"]
    assert profile.created_at == "2024-01-15T10:30:00Z"
    assert profile.updated_at == "2024-01-20T14:45:00Z"
    assert profile.video_count == 3


def test_creator_profile_roundtrip() -> None:
    """CreatorProfile round-trip serialization preserves all data."""
    profile = CreatorProfile(
        creator_id="roundtrip_test",
        content_type="vlog",
        energy_level="high",
        typical_clip_duration=40.0,
        keyword_overrides=["amazing", "wow"],
        created_at="2024-01-15T10:30:00Z",
        updated_at="2024-01-20T14:45:00Z",
        video_count=7,
    )
    result = CreatorProfile.from_dict(profile.to_dict())
    assert result == profile


# ---------------------------------------------------------------------------
# NaturalPause tests
# ---------------------------------------------------------------------------

def test_natural_pause_instantiation() -> None:
    """NaturalPause can be instantiated with all required fields."""
    pause = NaturalPause(
        time=10.5,
        type="punctuation",
        confidence=0.9,
        context="This is a sentence.",
    )
    assert pause is not None


def test_natural_pause_fields_accessible() -> None:
    """All NaturalPause fields are accessible."""
    pause = NaturalPause(
        time=15.2,
        type="silence",
        confidence=0.8,
        context="... [silence] ...",
    )
    assert pause.time == 15.2
    assert pause.type == "silence"
    assert pause.confidence == 0.8
    assert pause.context == "... [silence] ..."


def test_natural_pause_types() -> None:
    """NaturalPause supports different pause types."""
    punctuation = NaturalPause(time=1.0, type="punctuation", confidence=0.9, context="End.")
    silence = NaturalPause(time=2.0, type="silence", confidence=0.8, context="...")
    breath = NaturalPause(time=3.0, type="breath", confidence=0.6, context="*breath*")
    
    assert punctuation.type == "punctuation"
    assert silence.type == "silence"
    assert breath.type == "breath"


# ---------------------------------------------------------------------------
# EmotionFeatures tests
# ---------------------------------------------------------------------------

def test_emotion_features_instantiation() -> None:
    """EmotionFeatures can be instantiated with all required fields."""
    emotion = EmotionFeatures(
        time=5.0,
        pitch_mean=200.0,
        pitch_std=50.0,
        volume_rms=0.6,
        spectral_centroid=2500.0,
        zero_crossing_rate=0.1,
        emotion="excitement",
        confidence=0.85,
    )
    assert emotion is not None


def test_emotion_features_fields_accessible() -> None:
    """All EmotionFeatures fields are accessible."""
    emotion = EmotionFeatures(
        time=10.5,
        pitch_mean=150.0,
        pitch_std=30.0,
        volume_rms=0.4,
        spectral_centroid=2000.0,
        zero_crossing_rate=0.15,
        emotion="laughter",
        confidence=0.9,
    )
    assert emotion.time == 10.5
    assert emotion.pitch_mean == 150.0
    assert emotion.pitch_std == 30.0
    assert emotion.volume_rms == 0.4
    assert emotion.spectral_centroid == 2000.0
    assert emotion.zero_crossing_rate == 0.15
    assert emotion.emotion == "laughter"
    assert emotion.confidence == 0.9


def test_emotion_features_emotion_types() -> None:
    """EmotionFeatures supports different emotion types."""
    laughter = EmotionFeatures(
        time=1.0, pitch_mean=200.0, pitch_std=50.0, volume_rms=0.5,
        spectral_centroid=2500.0, zero_crossing_rate=0.2, emotion="laughter", confidence=0.9
    )
    scream = EmotionFeatures(
        time=2.0, pitch_mean=450.0, pitch_std=80.0, volume_rms=0.9,
        spectral_centroid=3500.0, zero_crossing_rate=0.1, emotion="scream", confidence=0.95
    )
    excitement = EmotionFeatures(
        time=3.0, pitch_mean=350.0, pitch_std=60.0, volume_rms=0.7,
        spectral_centroid=3000.0, zero_crossing_rate=0.08, emotion="excitement", confidence=0.85
    )
    calm = EmotionFeatures(
        time=4.0, pitch_mean=150.0, pitch_std=20.0, volume_rms=0.2,
        spectral_centroid=1500.0, zero_crossing_rate=0.05, emotion="calm", confidence=0.8
    )
    neutral = EmotionFeatures(
        time=5.0, pitch_mean=180.0, pitch_std=30.0, volume_rms=0.4,
        spectral_centroid=2000.0, zero_crossing_rate=0.1, emotion="neutral", confidence=0.7
    )
    
    assert laughter.emotion == "laughter"
    assert scream.emotion == "scream"
    assert excitement.emotion == "excitement"
    assert calm.emotion == "calm"
    assert neutral.emotion == "neutral"


# ---------------------------------------------------------------------------
# EngagementFeatures tests
# ---------------------------------------------------------------------------

def test_engagement_features_instantiation() -> None:
    """EngagementFeatures can be instantiated with all required fields."""
    features = EngagementFeatures(
        duration=35.0,
        pacing_score=0.8,
        energy_curve=[0.5, 0.6, 0.7, 0.8, 0.7],
        hook_score=0.9,
        emotion_diversity=0.75,
        pause_quality=0.85,
    )
    assert features is not None


def test_engagement_features_fields_accessible() -> None:
    """All EngagementFeatures fields are accessible."""
    features = EngagementFeatures(
        duration=40.0,
        pacing_score=0.7,
        energy_curve=[0.4, 0.5, 0.6, 0.7, 0.6, 0.5],
        hook_score=0.85,
        emotion_diversity=0.8,
        pause_quality=0.9,
    )
    assert features.duration == 40.0
    assert features.pacing_score == 0.7
    assert features.energy_curve == [0.4, 0.5, 0.6, 0.7, 0.6, 0.5]
    assert features.hook_score == 0.85
    assert features.emotion_diversity == 0.8
    assert features.pause_quality == 0.9


def test_engagement_features_energy_curve_is_list() -> None:
    """EngagementFeatures.energy_curve is a list of floats."""
    features = EngagementFeatures(
        duration=30.0,
        pacing_score=0.75,
        energy_curve=[0.3, 0.4, 0.5],
        hook_score=0.8,
        emotion_diversity=0.7,
        pause_quality=0.8,
    )
    assert isinstance(features.energy_curve, list)
    assert all(isinstance(x, float) for x in features.energy_curve)
