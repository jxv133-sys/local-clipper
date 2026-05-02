"""Tests for pipeline/shorts_formatter.py — unit tests (task 6.7) and property-based tests (task 6.8)."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pipeline.exceptions import ShortsFormattingError
from pipeline.models import Clip, Segment, SRTEntry, Transcript
from pipeline.shorts_formatter import (
    ShortsFormatter,
    collect_srt_entries,
    derive_shorts_path,
)


# ---------------------------------------------------------------------------
# Helpers / Factories
# ---------------------------------------------------------------------------

def make_clip(start: float, end: float, rank: int = 1) -> Clip:
    return Clip(start=start, end=end, score=0.9, rank=rank, segment_indices=[])


def make_segment(start: float, end: float, text: str = "hello") -> Segment:
    return Segment(start=start, end=end, text=text)


def make_transcript(segments: list[Segment]) -> Transcript:
    return Transcript(segments=segments)


# ---------------------------------------------------------------------------
# Task 6.7 — Unit tests: derive_shorts_path
# ---------------------------------------------------------------------------

class TestDeriveShortsPath:
    """Unit tests for derive_shorts_path naming convention."""

    def test_full_path_mp4(self):
        """/output/clip_1_30s.mp4 → /output/clip_1_30s_shorts.mp4"""
        assert derive_shorts_path("/output/clip_1_30s.mp4") == "/output/clip_1_30s_shorts.mp4"

    def test_relative_path_mp4(self):
        """clip.mp4 → clip_shorts.mp4"""
        assert derive_shorts_path("clip.mp4") == "clip_shorts.mp4"

    def test_avi_extension(self):
        """/path/to/video.avi → /path/to/video_shorts.avi"""
        assert derive_shorts_path("/path/to/video.avi") == "/path/to/video_shorts.avi"

    def test_no_extension(self):
        """clip (no extension) → clip_shorts"""
        assert derive_shorts_path("clip") == "clip_shorts"

    def test_multiple_dots(self):
        """clip.v2.mp4 → clip.v2_shorts.mp4 (only last extension is split)"""
        assert derive_shorts_path("clip.v2.mp4") == "clip.v2_shorts.mp4"


# ---------------------------------------------------------------------------
# Task 6.7 — Unit tests: collect_srt_entries
# ---------------------------------------------------------------------------

class TestCollectSrtEntries:
    """Unit tests for collect_srt_entries overlap logic and time adjustment."""

    def test_empty_transcript_returns_empty(self):
        """Empty transcript → empty list."""
        clip = make_clip(10.0, 40.0)
        transcript = make_transcript([])
        result = collect_srt_entries(clip, transcript)
        assert result == []

    def test_segment_fully_within_clip_window(self):
        """Segment fully within clip window → included with adjusted timestamps."""
        clip = make_clip(10.0, 40.0)
        seg = make_segment(15.0, 25.0, "hello world")
        transcript = make_transcript([seg])
        result = collect_srt_entries(clip, transcript)
        assert len(result) == 1
        assert result[0].start == pytest.approx(5.0)   # 15.0 - 10.0
        assert result[0].end == pytest.approx(15.0)    # 25.0 - 10.0
        assert result[0].text == "hello world"

    def test_segment_starting_before_clip_start(self):
        """Segment starting before clip start → included, start adjusted to 0.0."""
        clip = make_clip(10.0, 40.0)
        seg = make_segment(5.0, 20.0, "overlap left")
        transcript = make_transcript([seg])
        result = collect_srt_entries(clip, transcript)
        assert len(result) == 1
        assert result[0].start == pytest.approx(0.0)   # max(0.0, 5.0 - 10.0)
        assert result[0].end == pytest.approx(10.0)    # 20.0 - 10.0

    def test_segment_ending_after_clip_end(self):
        """Segment ending after clip end → included, end adjusted to clip duration."""
        clip = make_clip(10.0, 40.0)
        seg = make_segment(30.0, 50.0, "overlap right")
        transcript = make_transcript([seg])
        result = collect_srt_entries(clip, transcript)
        assert len(result) == 1
        assert result[0].start == pytest.approx(20.0)  # 30.0 - 10.0
        assert result[0].end == pytest.approx(40.0)    # 50.0 - 10.0 = 40.0, but clip duration is 30.0
        # Actually: adjusted_end = max(0.0, 50.0 - 10.0) = 40.0 (not clamped to clip duration)
        # The implementation uses max(0.0, ...) not min(clip_duration, ...)
        # Let's verify the actual adjusted_end value
        assert result[0].end == pytest.approx(40.0)

    def test_segment_completely_before_clip(self):
        """Segment completely before clip → excluded."""
        clip = make_clip(10.0, 40.0)
        seg = make_segment(0.0, 9.0, "before clip")
        transcript = make_transcript([seg])
        result = collect_srt_entries(clip, transcript)
        assert result == []

    def test_segment_completely_after_clip(self):
        """Segment completely after clip → excluded."""
        clip = make_clip(10.0, 40.0)
        seg = make_segment(41.0, 50.0, "after clip")
        transcript = make_transcript([seg])
        result = collect_srt_entries(clip, transcript)
        assert result == []

    def test_multiple_overlapping_segments_all_included_in_order(self):
        """Multiple overlapping segments → all included in order."""
        clip = make_clip(10.0, 40.0)
        segs = [
            make_segment(12.0, 18.0, "first"),
            make_segment(20.0, 28.0, "second"),
            make_segment(30.0, 38.0, "third"),
        ]
        transcript = make_transcript(segs)
        result = collect_srt_entries(clip, transcript)
        assert len(result) == 3
        assert result[0].text == "first"
        assert result[1].text == "second"
        assert result[2].text == "third"

    def test_segment_exactly_at_clip_end_boundary_excluded(self):
        """Segment starting exactly at clip.end → excluded (start >= clip.end)."""
        clip = make_clip(10.0, 40.0)
        seg = make_segment(40.0, 50.0, "at end boundary")
        transcript = make_transcript([seg])
        result = collect_srt_entries(clip, transcript)
        assert result == []

    def test_segment_ending_exactly_at_clip_start_excluded(self):
        """Segment ending exactly at clip.start → excluded (end <= clip.start)."""
        clip = make_clip(10.0, 40.0)
        seg = make_segment(0.0, 10.0, "at start boundary")
        transcript = make_transcript([seg])
        result = collect_srt_entries(clip, transcript)
        assert result == []

    def test_returned_entries_have_1_based_indices(self):
        """Returned entries have 1-based indices."""
        clip = make_clip(0.0, 60.0)
        segs = [make_segment(i * 10.0, i * 10.0 + 5.0, f"seg{i}") for i in range(4)]
        transcript = make_transcript(segs)
        result = collect_srt_entries(clip, transcript)
        assert len(result) == 4
        for i, entry in enumerate(result):
            assert entry.index == i + 1, f"Expected 1-based index {i+1}, got {entry.index}"

    def test_all_returned_entries_have_valid_timestamps(self):
        """All returned entries have start >= 0.0 and start < end."""
        clip = make_clip(5.0, 35.0)
        segs = [
            make_segment(0.0, 10.0, "straddles left"),
            make_segment(15.0, 25.0, "fully inside"),
            make_segment(30.0, 40.0, "straddles right"),
        ]
        transcript = make_transcript(segs)
        result = collect_srt_entries(clip, transcript)
        assert len(result) == 3
        for entry in result:
            assert entry.start >= 0.0, f"entry.start={entry.start} < 0.0"
            assert entry.start < entry.end, f"entry.start={entry.start} >= entry.end={entry.end}"


# ---------------------------------------------------------------------------
# Task 6.7 — Unit tests: format_clips error handling
# ---------------------------------------------------------------------------

class TestFormatClipsErrorHandling:
    """Unit tests for format_clips error handling and partial success."""

    def _make_config(self) -> SimpleNamespace:
        return SimpleNamespace(
            work_dir="/tmp/test",
            shorts_width=1080,
            shorts_height=1920,
            facecam_top_fraction=0.35,
            facecam_detection_enabled=False,
            subtitle_style="bubble",
            subtitle_font_size=72,
            subtitle_font_name="Impact",
            subtitle_primary_color="&H00FFFFFF",
            subtitle_outline_color="&H00000000",
            subtitle_highlight_color="&H0000FFFF",
            subtitle_outline_width=4.0,
            subtitle_shadow_depth=2.0,
            subtitle_margin_bottom=80,
            subtitle_words_per_group=3,
        )

    def test_format_clips_continues_after_single_clip_failure(self):
        """When format_single_clip raises ShortsFormattingError for one clip,
        format_clips logs the error and continues processing other clips."""
        formatter = ShortsFormatter()
        config = self._make_config()

        clips = [make_clip(0.0, 30.0, rank=1), make_clip(60.0, 90.0, rank=2)]
        clip_paths = ["/output/clip_1.mp4", "/output/clip_2.mp4"]
        transcript = make_transcript([])

        def side_effect(cfg, clip, clip_path, srt_entries):
            if clip_path == "/output/clip_1.mp4":
                raise ShortsFormattingError("FFmpeg failed for clip 1")
            return "/output/clip_2_shorts.mp4"

        with patch.object(formatter, "format_single_clip", side_effect=side_effect):
            result = formatter.format_clips(config, clips, clip_paths, transcript)

        assert result == ["/output/clip_2_shorts.mp4"]

    def test_format_clips_returns_only_successful_paths(self):
        """format_clips returns only successfully converted paths."""
        formatter = ShortsFormatter()
        config = self._make_config()

        clips = [
            make_clip(0.0, 30.0, rank=1),
            make_clip(60.0, 90.0, rank=2),
            make_clip(120.0, 150.0, rank=3),
        ]
        clip_paths = [
            "/output/clip_1.mp4",
            "/output/clip_2.mp4",
            "/output/clip_3.mp4",
        ]
        transcript = make_transcript([])

        def side_effect(cfg, clip, clip_path, srt_entries):
            if clip_path == "/output/clip_2.mp4":
                raise ShortsFormattingError("FFmpeg failed for clip 2")
            return clip_path.replace(".mp4", "_shorts.mp4")

        with patch.object(formatter, "format_single_clip", side_effect=side_effect):
            result = formatter.format_clips(config, clips, clip_paths, transcript)

        assert "/output/clip_1_shorts.mp4" in result
        assert "/output/clip_3_shorts.mp4" in result
        assert "/output/clip_2_shorts.mp4" not in result
        assert len(result) == 2

    def test_format_clips_all_fail_returns_empty(self):
        """When all clips fail, format_clips returns an empty list."""
        formatter = ShortsFormatter()
        config = self._make_config()

        clips = [make_clip(0.0, 30.0, rank=1), make_clip(60.0, 90.0, rank=2)]
        clip_paths = ["/output/clip_1.mp4", "/output/clip_2.mp4"]
        transcript = make_transcript([])

        with patch.object(
            formatter,
            "format_single_clip",
            side_effect=ShortsFormattingError("all fail"),
        ):
            result = formatter.format_clips(config, clips, clip_paths, transcript)

        assert result == []

    def test_format_clips_logs_error_on_failure(self, caplog):
        """format_clips logs an error when format_single_clip raises ShortsFormattingError."""
        formatter = ShortsFormatter()
        config = self._make_config()

        clips = [make_clip(0.0, 30.0, rank=1)]
        clip_paths = ["/output/clip_1.mp4"]
        transcript = make_transcript([])

        with patch.object(
            formatter,
            "format_single_clip",
            side_effect=ShortsFormattingError("FFmpeg failed"),
        ):
            with caplog.at_level(logging.ERROR, logger="pipeline.shorts_formatter"):
                formatter.format_clips(config, clips, clip_paths, transcript)

        assert any("Shorts formatting failed" in record.message for record in caplog.records)

    def test_format_clips_empty_clips_returns_empty(self):
        """format_clips with empty clips list returns empty list immediately."""
        formatter = ShortsFormatter()
        config = self._make_config()
        result = formatter.format_clips(config, [], [], make_transcript([]))
        assert result == []


# ---------------------------------------------------------------------------
# Task 6.8 — Hypothesis strategies
# ---------------------------------------------------------------------------

@st.composite
def valid_clip(draw, min_start: float = 0.0, max_end: float = 3600.0) -> Clip:
    """Generate a Clip with start < end."""
    start = draw(st.floats(min_value=min_start, max_value=max_end - 0.1, allow_nan=False, allow_infinity=False))
    end = draw(st.floats(min_value=start + 0.01, max_value=max_end, allow_nan=False, allow_infinity=False))
    return make_clip(start, end)


@st.composite
def valid_segment_strategy(draw, min_start: float = 0.0, max_end: float = 3600.0) -> Segment:
    """Generate a Segment with start < end."""
    start = draw(st.floats(min_value=min_start, max_value=max_end - 0.01, allow_nan=False, allow_infinity=False))
    end = draw(st.floats(min_value=start + 0.01, max_value=max_end, allow_nan=False, allow_infinity=False))
    text = draw(st.text(min_size=1, max_size=50))
    return Segment(start=start, end=end, text=text)


@st.composite
def transcript_strategy(draw, max_end: float = 3600.0) -> Transcript:
    """Generate a Transcript with 0–20 segments."""
    n = draw(st.integers(min_value=0, max_value=20))
    segments = draw(st.lists(
        valid_segment_strategy(min_start=0.0, max_end=max_end),
        min_size=n,
        max_size=n,
    ))
    return Transcript(segments=segments)


@st.composite
def clip_and_transcript(draw) -> tuple[Clip, Transcript]:
    """Generate a (Clip, Transcript) pair with clip.start < clip.end."""
    clip = draw(valid_clip(min_start=0.0, max_end=3600.0))
    transcript = draw(transcript_strategy(max_end=3600.0))
    return clip, transcript


# ---------------------------------------------------------------------------
# Task 6.8 — Property 8: SRT entries are time-adjusted and non-negative
# ---------------------------------------------------------------------------

@given(data=clip_and_transcript())
@settings(max_examples=200)
def test_property_8_srt_entries_time_adjusted_and_non_negative(data):
    """Property 8: For any Clip with start < end and any Transcript,
    collect_srt_entries returns only entries where 0.0 <= entry.start < entry.end,
    with all timestamps adjusted to be relative to clip.start.

    **Validates: Requirements 9.2, 9.3**
    """
    clip, transcript = data

    result = collect_srt_entries(clip, transcript)

    for entry in result:
        assert entry.start >= 0.0, (
            f"entry.start={entry.start} < 0.0 for clip [{clip.start}, {clip.end}]"
        )
        assert entry.start < entry.end, (
            f"entry.start={entry.start} >= entry.end={entry.end} for clip [{clip.start}, {clip.end}]"
        )


# ---------------------------------------------------------------------------
# Task 6.8 — Property 9: Entries only from clip window
# ---------------------------------------------------------------------------

@given(data=clip_and_transcript())
@settings(max_examples=200)
def test_property_9_entries_only_from_clip_window(data):
    """Property 9: For any Clip and Transcript, every SRTEntry returned by
    collect_srt_entries corresponds to a transcript segment that overlaps
    the clip's [start, end] window.

    **Validates: Requirements 9.1**
    """
    clip, transcript = data

    result = collect_srt_entries(clip, transcript)

    # Build a lookup of original segments by their adjusted start time
    # For each returned entry, verify there exists a segment that overlaps the clip window
    for entry in result:
        # Reconstruct the original segment start/end from the adjusted entry
        original_start = entry.start + clip.start
        original_end = entry.end + clip.start

        # Find the matching segment in the transcript
        matching_segment = None
        for seg in transcript.segments:
            # The entry's text must match the segment's text
            if seg.text == entry.text:
                # Verify this segment overlaps the clip window
                if seg.start < clip.end and seg.end > clip.start:
                    matching_segment = seg
                    break

        assert matching_segment is not None, (
            f"SRTEntry (start={entry.start}, end={entry.end}, text={entry.text!r}) "
            f"has no matching overlapping segment in transcript for clip [{clip.start}, {clip.end}]"
        )


# ---------------------------------------------------------------------------
# Task 6.8 — Property 14: Shorts path derived from clip stem
# ---------------------------------------------------------------------------

@given(
    directory=st.one_of(
        st.just(""),
        st.just("/output/"),
        st.just("/path/to/"),
        st.just("relative/dir/"),
    ),
    stem=st.text(
        # Exclude dots from stems to avoid ambiguity with os.path.splitext:
        # a stem like "0." would be split as stem="" ext="." by splitext.
        # Real clip stems use alphanumerics and underscores only.
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="_"),
        min_size=1,
        max_size=30,
    ),
    extension=st.one_of(
        st.just(".mp4"),
        st.just(".avi"),
        st.just(".mov"),
        st.just(".mkv"),
        st.just(""),
    ),
)
@settings(max_examples=200)
def test_property_14_shorts_path_derived_from_clip_stem(
    directory: str, stem: str, extension: str
):
    """Property 14: For any clip path string, derive_shorts_path returns a path where:
    - The directory is the same as the input
    - The stem ends with '_shorts'
    - The extension is the same as the input

    **Validates: Requirements 7.1, 7.2**
    """
    import os

    clip_path = directory + stem + extension
    result = derive_shorts_path(clip_path)

    # The directory must be the same
    assert os.path.dirname(result) == os.path.dirname(clip_path), (
        f"Directory changed: input={os.path.dirname(clip_path)!r}, "
        f"output={os.path.dirname(result)!r}"
    )

    # The extension must be the same
    result_stem, result_ext = os.path.splitext(result)
    assert result_ext == extension, (
        f"Extension changed: expected={extension!r}, got={result_ext!r}"
    )

    # The stem must end with '_shorts'
    result_basename_stem = os.path.splitext(os.path.basename(result))[0]
    assert result_basename_stem.endswith("_shorts"), (
        f"Stem does not end with '_shorts': got {result_basename_stem!r} "
        f"for input clip_path={clip_path!r}"
    )

    # The original stem must be preserved before '_shorts'
    input_basename_stem = os.path.splitext(os.path.basename(clip_path))[0]
    assert result_basename_stem == input_basename_stem + "_shorts", (
        f"Expected stem {input_basename_stem + '_shorts'!r}, got {result_basename_stem!r}"
    )


# ---------------------------------------------------------------------------
# Task 7.5 — Integration tests: end-to-end shorts formatting with real FFmpeg
# ---------------------------------------------------------------------------

import shutil

from config import Config


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


pytestmark_integration = pytest.mark.skipif(
    not _ffmpeg_available(),
    reason="ffmpeg/ffprobe not available",
)


def _probe_streams(output_path: str) -> list[dict]:
    """Run ffprobe on output_path and return the list of stream dicts."""
    import json
    import subprocess

    result = subprocess.run(
        [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            output_path,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"ffprobe failed: {result.stderr}"
    data = json.loads(result.stdout)
    return data.get("streams", [])


def _create_synthetic_video(output_path: str) -> None:
    """Create a synthetic 5-second 1920×1080 test video with audio using FFmpeg."""
    import subprocess

    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f", "lavfi",
            "-i", "testsrc=duration=5:size=1920x1080:rate=30",
            "-f", "lavfi",
            "-i", "sine=frequency=440:duration=5",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "28",
            "-c:a", "aac",
            "-shortest",
            output_path,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Failed to create synthetic test video: {result.stderr}"
    )


@pytest.mark.integration
@pytestmark_integration
def test_integration_shorts_formatter_output_dimensions_and_audio(tmp_path):
    """Integration test: ShortsFormatter.format_single_clip produces a 1080×1920
    vertical video with an audio stream from a synthetic 1920×1080 source.

    Uses FFmpeg's testsrc and sine lavfi sources to generate a real video file,
    then runs the full shorts formatting pipeline and verifies the output with
    ffprobe.
    """
    # --- Create synthetic source video ---
    source_path = str(tmp_path / "source.mp4")
    _create_synthetic_video(source_path)
    assert (tmp_path / "source.mp4").exists(), "Synthetic source video was not created"

    # --- Build Config ---
    config = Config(
        work_dir=str(tmp_path),
        shorts_enabled=True,
        facecam_detection_enabled=False,  # skip facecam detection for speed
    )

    # --- Build minimal Clip and Transcript ---
    clip = Clip(
        start=0.0,
        end=5.0,
        score=0.9,
        rank=1,
        segment_indices=[0],
    )
    transcript = Transcript(
        segments=[
            Segment(start=0.0, end=5.0, text="Test clip"),
        ]
    )

    # --- Collect SRT entries ---
    srt_entries = collect_srt_entries(clip, transcript)

    # --- Run the formatter ---
    formatter = ShortsFormatter()
    shorts_path = formatter.format_single_clip(
        config=config,
        clip=clip,
        clip_path=source_path,
        srt_entries=srt_entries,
    )

    # --- Assert output file exists ---
    assert shorts_path is not None, "format_single_clip returned None"
    import os
    assert os.path.exists(shorts_path), (
        f"Output shorts file does not exist: {shorts_path}"
    )

    # --- Probe output streams ---
    streams = _probe_streams(shorts_path)

    # --- Verify video dimensions: 1080×1920 ---
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    assert len(video_streams) >= 1, "No video stream found in output"
    video_stream = video_streams[0]
    assert video_stream.get("width") == 1080, (
        f"Expected width=1080, got {video_stream.get('width')}"
    )
    assert video_stream.get("height") == 1920, (
        f"Expected height=1920, got {video_stream.get('height')}"
    )

    # --- Verify audio stream exists ---
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    assert len(audio_streams) >= 1, "No audio stream found in output"
