"""Tests for pipeline data models, including property-based tests."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pipeline.models import (
    CanvasLayout,
    FacecamRegion,
    FilterFragment,
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
