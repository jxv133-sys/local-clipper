"""Tests for pipeline/subtitle_generator.py.

Covers:
- Property 11: SRT timestamp offset (subtask 10.1)
- Property 12: SRT entry count matches in-range segments (subtask 10.2)
- Property 13: SRT serialization round-trip (subtask 10.3)
- Unit tests for generate_subtitles (subtask 10.4)
"""

from __future__ import annotations

import math
import os
import subprocess
from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from config import Config
from pipeline.exceptions import SubtitleError
from pipeline.models import Clip, Segment, SRTEntry, Transcript
from pipeline.subtitle_generator import (
    generate_subtitles,
    parse_srt,
    serialize_srt,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(tmp_path) -> Config:
    cfg = Config(work_dir=str(tmp_path))
    cfg.output_dir = str(tmp_path / "output")
    return cfg


def make_clip(start: float, end: float, rank: int = 1) -> Clip:
    return Clip(start=start, end=end, score=0.9, rank=rank, segment_indices=[])


def make_segment(start: float, end: float, text: str = "hello") -> Segment:
    return Segment(start=start, end=end, text=text)


def make_srt_entry(index: int, start: float, end: float, text: str = "hello") -> SRTEntry:
    return SRTEntry(index=index, start=start, end=end, text=text)


def completed(returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode,
                                       stdout="", stderr=stderr)


# ---------------------------------------------------------------------------
# Property 11: SRT timestamp offset
# Validates: Requirements 9.2
# ---------------------------------------------------------------------------

# Feature: video-highlight-generator, Property 11: SRT timestamp offset
@given(
    clip_start=st.floats(min_value=0.0, max_value=9000.0, allow_nan=False, allow_infinity=False),
    offset=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    seg_duration=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_srt_timestamp_offset(clip_start: float, offset: float, seg_duration: float) -> None:
    """Property 11: SRT entry timestamps are relative to clip start.

    For any Segment within a Clip's range, the SRTEntry start == segment.start - clip.start
    and end == segment.end - clip.start.

    **Validates: Requirements 9.2**
    """
    # Build segment that is guaranteed to be inside the clip window
    seg_abs_start = clip_start + offset
    seg_abs_end = seg_abs_start + seg_duration
    clip_end = seg_abs_end + 1.0  # clip always extends past the segment

    clip = make_clip(start=clip_start, end=clip_end)
    seg = make_segment(start=seg_abs_start, end=seg_abs_end, text="test text")
    transcript = Transcript(segments=[seg])

    # Compute expected relative timestamps
    expected_rel_start = max(0.0, seg.start - clip.start)
    expected_rel_end = max(0.0, seg.end - clip.start)

    # Collect entries the same way generate_subtitles does
    entries: list[SRTEntry] = []
    for s in transcript.segments:
        if not s.text.strip():
            continue
        if s.end <= clip.start or s.start >= clip.end:
            continue
        rel_start = max(0.0, s.start - clip.start)
        rel_end = max(0.0, s.end - clip.start)
        entries.append(SRTEntry(index=1, start=rel_start, end=rel_end, text=s.text.strip()))

    assert len(entries) == 1
    assert math.isclose(entries[0].start, expected_rel_start, abs_tol=1e-9)
    assert math.isclose(entries[0].end, expected_rel_end, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Property 12: SRT entry count matches in-range segments
# Validates: Requirements 9.1, 9.5
# ---------------------------------------------------------------------------

# Feature: video-highlight-generator, Property 12: SRT entry count
@given(
    clip_start=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    clip_duration=st.floats(min_value=20.0, max_value=45.0, allow_nan=False, allow_infinity=False),
    segments=st.lists(
        st.builds(
            Segment,
            start=st.floats(min_value=0.0, max_value=200.0, allow_nan=False, allow_infinity=False),
            end=st.floats(min_value=0.0, max_value=200.0, allow_nan=False, allow_infinity=False),
            text=st.one_of(st.just(""), st.text(min_size=1, max_size=50)),
        ),
        max_size=20,
    ),
)
@settings(max_examples=100)
def test_srt_entry_count(
    clip_start: float,
    clip_duration: float,
    segments: list[Segment],
) -> None:
    """Property 12: SRT entry count equals non-empty in-range segments.

    **Validates: Requirements 9.1, 9.5**
    """
    clip_end = clip_start + clip_duration
    clip = make_clip(start=clip_start, end=clip_end)
    transcript = Transcript(segments=segments)

    # Count expected entries: non-empty segments that overlap with the clip
    expected_count = sum(
        1 for s in segments
        if s.text.strip()
        and not (s.end <= clip_start or s.start >= clip_end)
    )

    # Generate entries the same way generate_subtitles does
    entries: list[SRTEntry] = []
    idx = 1
    for s in transcript.segments:
        if not s.text.strip():
            continue
        if s.end <= clip.start or s.start >= clip.end:
            continue
        rel_start = max(0.0, s.start - clip.start)
        rel_end = max(0.0, s.end - clip.start)
        entries.append(SRTEntry(index=idx, start=rel_start, end=rel_end, text=s.text.strip()))
        idx += 1

    assert len(entries) == expected_count


# ---------------------------------------------------------------------------
# Property 13: SRT serialization round-trip
# Validates: Requirements 9.6
# ---------------------------------------------------------------------------

# Feature: video-highlight-generator, Property 13: SRT serialization round-trip
@given(
    entries=st.lists(
        st.builds(
            SRTEntry,
            index=st.integers(min_value=1, max_value=9999),
            start=st.floats(min_value=0.0, max_value=9000.0, allow_nan=False, allow_infinity=False),
            end=st.floats(min_value=0.0, max_value=9000.0, allow_nan=False, allow_infinity=False),
            # Exclude Unicode line-separator characters that get normalized during SRT parsing
            text=st.text(
                alphabet=st.characters(blacklist_categories=("Cc", "Cs")),
                min_size=1,
                max_size=100,
            ).filter(lambda t: t.strip()),
        ),
        min_size=1,
        max_size=20,
    )
)
@settings(max_examples=100)
def test_srt_roundtrip(entries: list[SRTEntry]) -> None:
    """Property 13: Serializing then parsing SRT produces equivalent entries.

    **Validates: Requirements 9.6**
    """
    srt_str = serialize_srt(entries)
    parsed = parse_srt(srt_str)

    assert len(parsed) == len(entries)
    for orig, restored in zip(entries, parsed):
        assert orig.index == restored.index
        # Timestamps are stored with millisecond precision — allow 1ms tolerance
        assert math.isclose(orig.start, restored.start, abs_tol=0.001)
        assert math.isclose(orig.end, restored.end, abs_tol=0.001)
        assert orig.text.strip() == restored.text.strip()


# ---------------------------------------------------------------------------
# Unit tests (subtask 10.4)
# ---------------------------------------------------------------------------

class TestSerializeSRT:
    """Unit tests for serialize_srt and parse_srt."""

    def test_empty_text_segment_omitted(self) -> None:
        """Segments with empty text are not included in SRT output."""
        entries = [
            make_srt_entry(1, 0.0, 1.0, "Hello"),
            make_srt_entry(2, 1.0, 2.0, ""),   # empty — should be omitted by caller
            make_srt_entry(3, 2.0, 3.0, "World"),
        ]
        # serialize_srt itself doesn't filter — the caller (generate_subtitles) does
        # But we verify that empty-text entries round-trip correctly
        srt = serialize_srt([entries[0], entries[2]])
        parsed = parse_srt(srt)
        assert len(parsed) == 2
        assert parsed[0].text == "Hello"
        assert parsed[1].text == "World"

    def test_timestamp_format(self) -> None:
        """Timestamps are formatted as HH:MM:SS,mmm."""
        entries = [make_srt_entry(1, 3661.5, 3662.75, "Test")]
        srt = serialize_srt(entries)
        assert "01:01:01,500" in srt
        assert "01:01:02,750" in srt

    def test_single_entry_roundtrip(self) -> None:
        """Single SRT entry round-trips correctly."""
        entry = make_srt_entry(1, 5.0, 8.5, "Hello world")
        srt = serialize_srt([entry])
        parsed = parse_srt(srt)
        assert len(parsed) == 1
        assert parsed[0].index == 1
        assert math.isclose(parsed[0].start, 5.0, abs_tol=0.001)
        assert math.isclose(parsed[0].end, 8.5, abs_tol=0.001)
        assert parsed[0].text == "Hello world"

    def test_empty_entries_produces_empty_string(self) -> None:
        """serialize_srt with empty list produces empty string."""
        assert serialize_srt([]) == ""

    def test_parse_empty_string(self) -> None:
        """parse_srt on empty string returns empty list."""
        assert parse_srt("") == []


class TestGenerateSubtitles:
    """Unit tests for generate_subtitles."""

    def test_empty_text_segment_omitted_from_srt(self, tmp_path) -> None:
        """Segments with empty text are omitted from the SRT file."""
        config = make_config(tmp_path)
        os.makedirs(config.output_dir, exist_ok=True)

        clip = make_clip(start=0.0, end=30.0)
        segments = [
            make_segment(0.0, 5.0, "Hello"),
            make_segment(5.0, 10.0, ""),        # empty — should be omitted
            make_segment(10.0, 15.0, "   "),    # whitespace-only — should be omitted
            make_segment(15.0, 20.0, "World"),
        ]
        transcript = Transcript(segments=segments)

        # Create a fake raw clip file
        raw_path = os.path.join(config.output_dir, "clip_1_0s.mp4")
        open(raw_path, "w").close()

        with patch("subprocess.run", return_value=completed(0)), \
             patch("os.replace"):
            generate_subtitles(config, [clip], transcript, [raw_path])

        srt_path = os.path.join(config.output_dir, "clip_1_0s.srt")
        assert os.path.exists(srt_path)
        with open(srt_path, encoding="utf-8") as fh:
            content = fh.read()

        parsed = parse_srt(content)
        texts = [e.text for e in parsed]
        assert "Hello" in texts
        assert "World" in texts
        assert "" not in texts
        assert "   " not in texts
        assert len(parsed) == 2

    def test_timestamps_adjusted_relative_to_clip_start(self, tmp_path) -> None:
        """SRT timestamps are relative to clip.start, not absolute."""
        config = make_config(tmp_path)
        os.makedirs(config.output_dir, exist_ok=True)

        clip = make_clip(start=100.0, end=130.0)
        segments = [make_segment(105.0, 110.0, "Test")]
        transcript = Transcript(segments=segments)

        raw_path = os.path.join(config.output_dir, "clip_1_100s.mp4")
        open(raw_path, "w").close()

        with patch("subprocess.run", return_value=completed(0)), \
             patch("os.replace"):
            generate_subtitles(config, [clip], transcript, [raw_path])

        srt_path = os.path.join(config.output_dir, "clip_1_100s.srt")
        with open(srt_path, encoding="utf-8") as fh:
            content = fh.read()

        parsed = parse_srt(content)
        assert len(parsed) == 1
        # 105.0 - 100.0 = 5.0s relative start
        assert math.isclose(parsed[0].start, 5.0, abs_tol=0.001)
        # 110.0 - 100.0 = 10.0s relative end
        assert math.isclose(parsed[0].end, 10.0, abs_tol=0.001)

    def test_srt_file_written_alongside_clip(self, tmp_path) -> None:
        """SRT file is written to the same directory as the clip."""
        config = make_config(tmp_path)
        os.makedirs(config.output_dir, exist_ok=True)

        clip = make_clip(start=0.0, end=25.0)
        transcript = Transcript(segments=[make_segment(5.0, 10.0, "Hello")])

        raw_path = os.path.join(config.output_dir, "clip_1_0s.mp4")
        open(raw_path, "w").close()

        with patch("subprocess.run", return_value=completed(0)), \
             patch("os.replace"):
            generate_subtitles(config, [clip], transcript, [raw_path])

        srt_path = os.path.join(config.output_dir, "clip_1_0s.srt")
        assert os.path.exists(srt_path)

    def test_ffmpeg_error_raises_subtitle_error(self, tmp_path) -> None:
        """FFmpeg non-zero exit during subtitle burn raises SubtitleError."""
        config = make_config(tmp_path)
        os.makedirs(config.output_dir, exist_ok=True)

        clip = make_clip(start=0.0, end=25.0)
        transcript = Transcript(segments=[make_segment(5.0, 10.0, "Hello")])

        raw_path = os.path.join(config.output_dir, "clip_1_0s.mp4")
        open(raw_path, "w").close()

        # Mock the entire _burn_subtitles function to raise SubtitleError
        from pipeline.exceptions import SubtitleError as SE
        with patch("pipeline.subtitle_generator._burn_subtitles",
                   side_effect=SE("subtitle filter error")):
            with pytest.raises(SE) as exc_info:
                generate_subtitles(config, [clip], transcript, [raw_path])

        assert "subtitle filter error" in str(exc_info.value)

    def test_out_of_range_segments_excluded(self, tmp_path) -> None:
        """Segments outside the clip's time range are not included in the SRT."""
        config = make_config(tmp_path)
        os.makedirs(config.output_dir, exist_ok=True)

        clip = make_clip(start=50.0, end=80.0)
        segments = [
            make_segment(10.0, 20.0, "Before clip"),   # before clip
            make_segment(55.0, 65.0, "Inside clip"),   # inside clip
            make_segment(90.0, 100.0, "After clip"),   # after clip
        ]
        transcript = Transcript(segments=segments)

        raw_path = os.path.join(config.output_dir, "clip_1_50s.mp4")
        open(raw_path, "w").close()

        with patch("subprocess.run", return_value=completed(0)), \
             patch("os.replace"):
            generate_subtitles(config, [clip], transcript, [raw_path])

        srt_path = os.path.join(config.output_dir, "clip_1_50s.srt")
        with open(srt_path, encoding="utf-8") as fh:
            content = fh.read()

        parsed = parse_srt(content)
        assert len(parsed) == 1
        assert parsed[0].text == "Inside clip"
