"""Tests for pipeline data models, including property-based tests."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pipeline.models import Segment, Transcript


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
