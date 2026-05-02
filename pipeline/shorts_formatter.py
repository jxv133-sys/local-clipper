"""ShortsFormatter: orchestrates the full clip-to-shorts conversion pipeline.

Stage 8 of the pipeline — consumes final clip paths from Stage 7 and produces
parallel *_shorts.mp4 files in 9:16 vertical format.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

from pipeline.animated_subtitle_renderer import AnimatedSubtitleRenderer
from pipeline.exceptions import ShortsFormattingError
from pipeline.facecam_relocator import FacecamRelocator
from pipeline.frame_reformatter import FrameReformatter, compute_canvas_layout
from pipeline.models import Clip, SRTEntry, SubtitleStyle, Transcript

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def derive_shorts_path(clip_path: str) -> str:
    """Derive the shorts output path by appending '_shorts' to the clip stem.

    Uses os.path.splitext to split the path into stem and extension, then
    appends '_shorts' before the extension.

    Args:
        clip_path: Path to the original clip file.

    Returns:
        Path with '_shorts' appended to the stem, e.g.
        '/output/clip_1_30s.mp4' → '/output/clip_1_30s_shorts.mp4'.

    Examples:
        >>> derive_shorts_path('/output/clip_1_30s.mp4')
        '/output/clip_1_30s_shorts.mp4'
        >>> derive_shorts_path('clip.mp4')
        'clip_shorts.mp4'
    """
    stem, ext = os.path.splitext(clip_path)
    return stem + "_shorts" + ext


def collect_srt_entries(clip: Clip, transcript: Transcript) -> list[SRTEntry]:
    """Collect and time-adjust SRTEntry objects for a clip from the full transcript.

    Filters transcript segments that overlap the clip's [start, end] window.
    A segment overlaps if segment.start < clip.end AND segment.end > clip.start.

    Timestamps are adjusted to be relative to clip.start:
        adjusted_start = max(0.0, segment.start - clip.start)
        adjusted_end   = max(0.0, segment.end   - clip.start)

    Entries where adjusted_start >= adjusted_end are skipped.

    Args:
        clip:       Clip with start/end times defining the window.
        transcript: Full Transcript with all segments.

    Returns:
        List of SRTEntry objects with 1-based indices and clip-relative timestamps.
        Returns an empty list if no segments overlap the clip window.
    """
    entries: list[SRTEntry] = []
    index = 1

    for segment in transcript.segments:
        # Overlap check: segment overlaps [clip.start, clip.end]
        if segment.start >= clip.end or segment.end <= clip.start:
            continue

        adjusted_start = max(0.0, segment.start - clip.start)
        adjusted_end = max(0.0, segment.end - clip.start)

        # Skip degenerate entries
        if adjusted_start >= adjusted_end:
            continue

        entries.append(
            SRTEntry(
                index=index,
                start=adjusted_start,
                end=adjusted_end,
                text=segment.text,
            )
        )
        index += 1

    return entries


# ---------------------------------------------------------------------------
# ShortsFormatter
# ---------------------------------------------------------------------------

class ShortsFormatter:
    """Orchestrates the full clip-to-shorts conversion pipeline.

    Coordinates FrameReformatter, FacecamRelocator, and AnimatedSubtitleRenderer
    to assemble a single composite FFmpeg filter graph per clip and run FFmpeg
    exactly once per clip.

    Responsibilities:
    - Probe source video dimensions using ffprobe.
    - Compute canvas layout from config.
    - Detect facecam (or use blur fallback).
    - Collect and time-adjust SRT entries per clip.
    - Assemble filter_complex from all three sub-components.
    - Run FFmpeg with libx264 -preset fast -crf 23 and audio copy.
    - Return shorts paths in rank order.
    - Raise ShortsFormattingError on non-zero FFmpeg exit.
    """

    def format_single_clip(
        self,
        config,
        clip: Clip,
        clip_path: str,
        srt_entries: list[SRTEntry],
    ) -> str:
        """Convert one clip to vertical 9:16 format.

        Probes source dimensions, computes canvas layout, detects facecam,
        builds the composite filter_complex, and runs FFmpeg once.

        Args:
            config:      Config object with shorts settings.
            clip:        Clip with start/end times and rank.
            clip_path:   Path to the source clip file.
            srt_entries: Pre-collected SRTEntry list for this clip.

        Returns:
            Path to the *_shorts.mp4 output file.

        Raises:
            ShortsFormattingError: If FFmpeg exits with a non-zero return code.
        """
        shorts_path = derive_shorts_path(clip_path)
        
        # Ensure output directory exists
        output_dir = os.path.dirname(shorts_path)
        os.makedirs(output_dir, exist_ok=True)

        # --- Step 1: Probe source video dimensions ---
        src_width, src_height = _probe_video_dimensions(clip_path)

        # --- Step 2: Compute canvas layout ---
        layout = compute_canvas_layout(config)

        # --- Step 3: Build canvas filter fragment ---
        reformatter = FrameReformatter()
        canvas_frag = reformatter.build_canvas_filter(src_width, src_height, layout)

        # --- Step 4: Detect facecam and build facecam/blur filter fragment ---
        relocator = FacecamRelocator()

        if config.facecam_detection_enabled:
            region = relocator.detect_facecam(
                clip_path=clip_path,
                frame_width=src_width,
                frame_height=src_height,
                config=config,
            )
        else:
            region = None

        if region is not None:
            facecam_frag = relocator.build_facecam_filter(
                region=region,
                canvas_width=layout.canvas_width,
                canvas_height=layout.canvas_height,
                top_third_height=layout.facecam_height,
            )
        else:
            if config.facecam_detection_enabled:
                logger.info(
                    "No facecam detected for clip %s; using blur fallback",
                    clip_path,
                )
            facecam_frag = relocator.build_blur_fallback_filter(
                canvas_width=layout.canvas_width,
                canvas_height=layout.canvas_height,
                top_third_height=layout.facecam_height,
            )

        # Normalise the facecam/blur output label to "[with_facecam]" so the
        # subtitle filter can always reference it by the same name.
        # The blur fallback uses "[with_blur_fill]" internally; we rename it.
        facecam_filter_str = facecam_frag.filter_str
        if facecam_frag.output_label == "[with_blur_fill]":
            facecam_filter_str = facecam_filter_str.replace(
                "[with_blur_fill]", "[with_facecam]"
            )

        # --- Step 5: Build subtitle filter fragment ---
        subtitle_style_str = getattr(config, "subtitle_style", "bubble")
        try:
            subtitle_style = SubtitleStyle(subtitle_style_str)
        except ValueError:
            logger.warning(
                "Unknown subtitle_style %r; falling back to BUBBLE", subtitle_style_str
            )
            subtitle_style = SubtitleStyle.BUBBLE

        renderer = AnimatedSubtitleRenderer()
        subtitle_frag = renderer.build_subtitle_filter(
            srt_entries=srt_entries,
            style=subtitle_style,
            canvas_width=layout.canvas_width,
            canvas_height=layout.canvas_height,
            gameplay_region_top=layout.gameplay_y,
            config=config,
            work_dir=config.work_dir,
        )

        # --- Step 6: Assemble filter_complex ---
        # Chain: canvas → facecam overlay → subtitle burn
        # The facecam filter takes [0:v] (for cropping the facecam) and [canvas] (for overlay)
        # The subtitle filter takes [with_facecam] and produces [final]
        # 
        # We need to chain these properly:
        # 1. Canvas filter: [0:v] → [canvas]
        # 2. Facecam filter: [0:v] + [canvas] → [with_facecam]
        # 3. Subtitle filter: [with_facecam] → [final]
        #
        # Using semicolons to separate independent filter chains
        filter_complex = (
            canvas_frag.filter_str
            + ";"
            + facecam_filter_str
            + ";"
            + subtitle_frag.filter_str
        )
        
        logger.info("Canvas filter: %s", canvas_frag.filter_str)
        logger.info("Facecam filter: %s", facecam_filter_str)
        logger.info("Subtitle filter: %s", subtitle_frag.filter_str)

        # --- Step 7: Run FFmpeg ---
        cmd = [
            "ffmpeg",
            "-nostdin",              # Disable interactive input
            "-y",                    # overwrite output without prompting
            "-i", clip_path,
            "-filter_complex", filter_complex,
            "-map", "[final]",
            "-map", "0:a",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "copy",
            shorts_path,
        ]

        logger.info("Running FFmpeg for shorts clip: %s", shorts_path)
        logger.debug("FFmpeg command: %s", cmd)
        logger.debug("Filter complex: %s", filter_complex)
        
        # Verify ASS file exists if using subtitle filter
        if "ass=" in filter_complex or "subtitles=" in filter_complex:
            # Extract the ASS file path from the filter string
            # Pattern: ass=/path/to/file.ass[label] → capture /path/to/file.ass
            # Match everything between = and the first [
            ass_match = re.search(r'(?:ass|subtitles)=([^[]+)', filter_complex)
            if ass_match:
                ass_path = ass_match.group(1).replace("\\:", ":")
                if not os.path.exists(ass_path):
                    logger.warning("ASS file not found: %s", ass_path)
                else:
                    logger.info("ASS file exists: %s", ass_path)

        try:
            # Use Popen with proper pipe handling to avoid deadlocks
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, stderr = process.communicate(timeout=600)
            result_returncode = process.returncode
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            raise ShortsFormattingError(
                f"FFmpeg timeout for {clip_path!r} (exceeded 600s)"
            )
        
        if result_returncode != 0:
            logger.error("FFmpeg stderr: %s", stderr)
            logger.error("FFmpeg stdout: %s", stdout)
            raise ShortsFormattingError(
                f"FFmpeg failed for {clip_path!r} (exit {result_returncode}):\n"
                f"{stderr}"
            )

        # Verify output file was created
        if not os.path.exists(shorts_path):
            raise ShortsFormattingError(
                f"FFmpeg completed but output file not created: {shorts_path}"
            )
        
        output_size = os.path.getsize(shorts_path)
        logger.info("Shorts clip written: %s (size: %d bytes)", shorts_path, output_size)
        return shorts_path

    def format_clips(
        self,
        config,
        clips: list[Clip],
        clip_paths: list[str],
        transcript: Transcript,
    ) -> list[str]:
        """Convert each clip to vertical 9:16 format using parallel processing.

        Uses ThreadPoolExecutor with max_workers = min(len(clips), cpu_count)
        to process clips concurrently.  Results are returned in rank order
        (matching the input order), regardless of thread completion order.

        Clips that fail with ShortsFormattingError are logged and skipped;
        only successfully converted paths are included in the return value.

        Args:
            config:     Config object with shorts settings.
            clips:      List of Clip objects (rank-ordered).
            clip_paths: List of paths to exported .mp4 files (same order as clips).
            transcript: Full Transcript for subtitle timing.

        Returns:
            List of paths to *_shorts.mp4 files, in rank order.
        """
        if not clips:
            return []

        max_workers = min(len(clips), os.cpu_count() or 1)

        # Map future → original index so we can restore rank order
        results: dict[int, str] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {}
            for i, (clip, clip_path) in enumerate(zip(clips, clip_paths)):
                srt_entries = collect_srt_entries(clip, transcript)
                future = executor.submit(
                    self.format_single_clip,
                    config,
                    clip,
                    clip_path,
                    srt_entries,
                )
                future_to_index[future] = i

            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    shorts_path = future.result()
                    results[idx] = shorts_path
                except ShortsFormattingError as exc:
                    logger.error(
                        "Shorts formatting failed for clip %d (%s): %s",
                        idx,
                        clip_paths[idx],
                        exc,
                    )

        # Return results in rank order (original input order)
        return [results[i] for i in sorted(results)]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _probe_video_dimensions(clip_path: str) -> tuple[int, int]:
    """Probe the video file to get its width and height using ffprobe.

    Uses list-form subprocess arguments to prevent shell injection.

    Args:
        clip_path: Path to the video file.

    Returns:
        (width, height) as positive integers.

    Raises:
        ValueError: If ffprobe fails or cannot determine dimensions.
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        clip_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise ValueError(f"ffprobe not found: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"ffprobe subprocess error: {exc}") from exc

    if result.returncode != 0:
        raise ValueError(
            f"ffprobe failed for {clip_path!r} (exit {result.returncode}):\n"
            f"{result.stderr}"
        )

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"ffprobe output is not valid JSON: {exc}") from exc

    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            width = stream.get("width")
            height = stream.get("height")
            if width and height and int(width) > 0 and int(height) > 0:
                return int(width), int(height)

    raise ValueError(
        f"Could not determine video dimensions from ffprobe output for {clip_path!r}"
    )
