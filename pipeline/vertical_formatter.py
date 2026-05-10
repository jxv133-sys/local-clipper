"""VerticalFormatter: applies confirmed facecam placement to clips and generates
vertical (9:16) output files.

This module implements the core vertical formatting engine for the Mini Video Editor.
It reuses FrameReformatter and FacecamRelocator from the existing pipeline.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

from pipeline.frame_reformatter import FrameReformatter
from pipeline.models import CanvasLayout, FacecamRegion, VerticalFormattingJob

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resolution scaling helper
# ---------------------------------------------------------------------------

def scale_region_to_resolution(
    region: FacecamRegion,
    from_resolution: tuple[int, int],
    to_resolution: tuple[int, int],
) -> FacecamRegion:
    """Proportionally adjust a FacecamRegion from one resolution to another.

    When a clip has a different source resolution than the reference clip used
    to define the facecam region, this function scales the region coordinates
    proportionally so the facecam covers the same relative area.

    Args:
        region:           The FacecamRegion defined in ``from_resolution`` space.
        from_resolution:  (width, height) of the reference frame.
        to_resolution:    (width, height) of the target frame.

    Returns:
        A new FacecamRegion with coordinates scaled to ``to_resolution``.
    """
    from_w, from_h = from_resolution
    to_w, to_h = to_resolution

    if from_w == 0 or from_h == 0:
        return region

    x_scale = to_w / from_w
    y_scale = to_h / from_h

    return FacecamRegion(
        x=round(region.x * x_scale),
        y=round(region.y * y_scale),
        width=round(region.width * x_scale),
        height=round(region.height * y_scale),
        corner=region.corner,
        confidence=region.confidence,
    )


# ---------------------------------------------------------------------------
# Output file naming
# ---------------------------------------------------------------------------

def get_output_path(
    input_path: str,
    settings: dict,
    output_dir: str,
) -> str:
    """Compute the output file path for a vertically-formatted clip.

    By default, appends ``_vertical`` before the file extension.  The caller
    can override this via ``settings["suffix"]`` or ``settings["prefix"]``.

    Args:
        input_path:  Path to the original source clip.
        settings:    User settings dict.  Recognised keys:
                       - ``"suffix"`` (str): suffix to append before extension
                         (default ``"_vertical"``).
                       - ``"prefix"`` (str): prefix to prepend to the filename.
        output_dir:  Directory where the output file should be saved.

    Returns:
        Absolute path string for the output file.
    """
    p = Path(input_path)
    stem = p.stem
    suffix = p.suffix  # e.g. ".mp4"

    name_suffix = settings.get("suffix", "_vertical")
    name_prefix = settings.get("prefix", "")

    new_name = f"{name_prefix}{stem}{name_suffix}{suffix}"
    return str(Path(output_dir) / new_name)


# ---------------------------------------------------------------------------
# FFmpeg filter generation
# ---------------------------------------------------------------------------

def _build_vertical_filter(
    src_width: int,
    src_height: int,
    facecam_region: FacecamRegion,
    layout: CanvasLayout,
) -> str:
    """Build the complete FFmpeg filter_complex string for vertical formatting.

    The filter chain uses CROP-BASED approach (not letterboxing):
    1. Split input into two streams for parallel processing
    2. Stream 1 (gameplay): Crop center of source to 9:16 aspect ratio,
       scale to gameplay region, pad to full canvas with gameplay at bottom
    3. Stream 2 (facecam): Crop facecam from source, scale to fit within
       facecam region while preserving aspect ratio
    4. Overlay facecam on canvas at specified position

    This matches the preview endpoint logic to ensure WYSIWYG.

    Args:
        src_width:      Source video width in pixels.
        src_height:     Source video height in pixels.
        facecam_region: Confirmed facecam placement in source frame coordinates.
        layout:         9:16 canvas layout.

    Returns:
        A filter_complex string suitable for ``ffmpeg -filter_complex``.
    """
    # Calculate crop dimensions for 9:16 gameplay region
    # Target aspect ratio for gameplay: 9:16
    gameplay_target_w = layout.gameplay_width
    gameplay_target_h = layout.gameplay_height
    gameplay_aspect = gameplay_target_w / gameplay_target_h  # 9/16 = 0.5625
    
    # Source aspect ratio
    src_aspect = src_width / src_height
    
    # Crop source to match gameplay aspect ratio (9:16)
    if src_aspect > gameplay_aspect:
        # Source is wider - crop width (center horizontally)
        crop_h = src_height
        crop_w = round(src_height * gameplay_aspect)
        crop_x = (src_width - crop_w) // 2
        crop_y = 0
    else:
        # Source is taller - crop height (center vertically)
        crop_w = src_width
        crop_h = round(src_width / gameplay_aspect)
        crop_x = 0
        crop_y = (src_height - crop_h) // 2
    
    # Build gameplay filter: crop to 9:16, scale to gameplay region, pad to canvas
    gameplay_filter = (
        f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y},"
        f"scale={gameplay_target_w}:{gameplay_target_h},"
        f"pad={layout.canvas_width}:{layout.canvas_height}:0:{layout.gameplay_y}:black"
    )
    
    # Build facecam crop and scale filter
    # Scale to fit within facecam region, preserving aspect ratio
    facecam_target_w = layout.facecam_width
    facecam_target_h = layout.facecam_height
    
    crop_w = facecam_region.width
    crop_h = facecam_region.height
    
    # Calculate scale dimensions to fit within target while preserving aspect ratio
    scale_w = facecam_target_w
    scale_h = round(crop_h * facecam_target_w / crop_w) if crop_w > 0 else facecam_target_h
    
    if scale_h > facecam_target_h:
        scale_h = facecam_target_h
        scale_w = round(crop_w * facecam_target_h / crop_h) if crop_h > 0 else facecam_target_w
    
    # Centre horizontally within the facecam region
    overlay_x = layout.facecam_x + (facecam_target_w - scale_w) // 2
    overlay_y = layout.facecam_y
    
    facecam_filter = (
        f"crop={facecam_region.width}:{facecam_region.height}:{facecam_region.x}:{facecam_region.y},"
        f"scale={scale_w}:{scale_h}"
    )
    
    # Log the filter details for debugging
    logger.info(f"Gameplay crop: {crop_w}x{crop_h} at ({crop_x},{crop_y}) from {src_width}x{src_height}")
    logger.info(f"Gameplay filter: {gameplay_filter}")
    logger.info(f"Facecam filter: {facecam_filter}")
    logger.info(f"Facecam overlay position: ({overlay_x},{overlay_y})")
    
    # Complete filter chain:
    # [0:v] -> split into two streams
    # Stream 1: crop and build canvas with gameplay in bottom region -> [canvas]
    # Stream 2: crop and scale facecam -> [facecam]
    # [canvas][facecam] -> overlay facecam on top -> output
    filter_complex = (
        f"[0:v]split=2[v1][v2];"
        f"[v1]{gameplay_filter}[canvas];"
        f"[v2]{facecam_filter}[facecam];"
        f"[canvas][facecam]overlay={overlay_x}:{overlay_y}[with_facecam]"
    )
    
    return filter_complex


# ---------------------------------------------------------------------------
# Core formatter
# ---------------------------------------------------------------------------

class VerticalFormatter:
    """Applies a confirmed facecam placement to a clip and produces a vertical output.

    Responsibilities:
    - Scale facecam region coordinates when the clip resolution differs from
      the reference resolution.
    - Build the FFmpeg filter chain (canvas + facecam overlay).
    - Encode the output file, preserving the original audio track.
    """

    def apply_placement_to_clip(
        self,
        clip_path: str,
        facecam_region: FacecamRegion,
        canvas_layout: CanvasLayout,
        output_path: str,
        config,
        reference_resolution: tuple[int, int] | None = None,
        clip_resolution: tuple[int, int] | None = None,
        transcript: "Transcript | None" = None,
        clip_start: float = 0.0,
        clip_end: float = 0.0,
        settings: dict | None = None,
    ) -> None:
        """Apply the facecam placement to a single clip and encode the output.

        If ``reference_resolution`` and ``clip_resolution`` are both provided
        and differ, the facecam region coordinates are proportionally scaled
        before building the filter chain.

        If ``settings["burn_subtitles"]`` is True and transcript data is provided,
        animated subtitles will be burned into the output using the specified style.

        Args:
            clip_path:            Path to the source clip.
            facecam_region:       Confirmed facecam placement (in reference
                                  resolution coordinates).
            canvas_layout:        9:16 canvas layout.
            output_path:          Destination path for the encoded output.
            config:               Config object (used for codec/quality defaults).
            reference_resolution: (width, height) of the reference clip used
                                  to define the facecam region.
            clip_resolution:      (width, height) of this specific clip.
            transcript:           Full transcript with all segments (optional, for subtitles).
            clip_start:           Start time of this clip in the source video (seconds).
            clip_end:             End time of this clip in the source video (seconds).
            settings:             User settings dict (may contain burn_subtitles, subtitle_style).

        Raises:
            subprocess.CalledProcessError: If FFmpeg encoding fails.
            RuntimeError: If the clip resolution cannot be determined.
        """
        settings = settings or {}
        
        # Determine actual clip resolution
        if clip_resolution is None:
            clip_resolution = _probe_resolution(clip_path)

        src_w, src_h = clip_resolution

        # Scale facecam region if resolutions differ
        effective_region = facecam_region
        if reference_resolution is not None and reference_resolution != clip_resolution:
            effective_region = scale_region_to_resolution(
                facecam_region, reference_resolution, clip_resolution
            )

        # Build filter chain
        filter_complex = _build_vertical_filter(src_w, src_h, effective_region, canvas_layout)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        # Check if we need to burn subtitles
        burn_subtitles = settings.get("burn_subtitles", False)
        subtitle_style_str = settings.get("subtitle_style", "bubble")
        
        # Map output label based on whether we're burning subtitles
        video_output_label = "[with_facecam]"
        
        if burn_subtitles and transcript is not None and clip_start < clip_end:
            # Generate subtitles for this clip's time range
            from pipeline.models import SRTEntry, SubtitleStyle
            from pipeline.animated_subtitle_renderer import AnimatedSubtitleRenderer
            
            # Parse subtitle style
            try:
                subtitle_style = SubtitleStyle(subtitle_style_str)
            except ValueError:
                logger.warning(
                    "Invalid subtitle style '%s', falling back to BUBBLE",
                    subtitle_style_str,
                )
                subtitle_style = SubtitleStyle.BUBBLE
            
            # Extract SRT entries for this clip's time range
            srt_entries: list[SRTEntry] = []
            entry_index = 1
            
            for seg in transcript.segments:
                if not seg.text.strip():
                    continue
                # Include segment if it overlaps with the clip window
                if seg.end <= clip_start or seg.start >= clip_end:
                    continue
                
                # Adjust timestamps to be relative to clip start
                rel_start = max(0.0, seg.start - clip_start)
                rel_end = min(clip_end - clip_start, seg.end - clip_start)
                
                if seg.words:
                    # Word-level splitting: group into short phrases (1-4 words)
                    from pipeline.subtitle_generator import _word_level_entries
                    new_entries = _word_level_entries(seg, clip_start, entry_index)
                    srt_entries.extend(new_entries)
                    entry_index += len(new_entries)
                else:
                    # Fallback: segment-level entry
                    srt_entries.append(
                        SRTEntry(
                            index=entry_index,
                            start=rel_start,
                            end=rel_end,
                            text=seg.text.strip(),
                        )
                    )
                    entry_index += 1
            
            if srt_entries:
                # Generate ASS subtitle filter
                renderer = AnimatedSubtitleRenderer()
                work_dir = os.path.dirname(output_path)
                
                subtitle_filter = renderer.build_subtitle_filter(
                    srt_entries=srt_entries,
                    style=subtitle_style,
                    canvas_width=canvas_layout.canvas_width,
                    canvas_height=canvas_layout.canvas_height,
                    gameplay_region_top=canvas_layout.gameplay_y,
                    config=config,
                    work_dir=work_dir,
                )
                
                # Append subtitle filter to the filter chain
                filter_complex += ";" + subtitle_filter.filter_str
                video_output_label = subtitle_filter.output_label
                
                logger.info(
                    "Burning %d subtitle entries into clip with style %s",
                    len(srt_entries), subtitle_style.value,
                )
            else:
                logger.info(
                    "No subtitle entries found for clip time range [%.1f, %.1f]",
                    clip_start, clip_end,
                )

        # Build FFmpeg command
        crf = getattr(config, "output_crf", 23)
        codec = getattr(config, "output_codec", "libx264")
        preset = getattr(config, "output_preset", "ultrafast")  # Fast encoding for vertical formatting

        cmd = [
            "ffmpeg",
            "-y",                    # overwrite output
            "-i", clip_path,
            "-filter_complex", filter_complex,
            "-map", video_output_label,
            "-map", "0:a?",          # preserve audio if present
            "-c:v", codec,
            "-preset", preset,       # Encoding speed preset
            "-crf", str(crf),
            "-c:a", "copy",
            "-threads", "0",         # Use all available CPU threads
            "-movflags", "+faststart",  # Enable fast start for web playback
            output_path,
        ]

        logger.info("Encoding vertical clip: %s → %s", clip_path, output_path)
        logger.debug("FFmpeg command: %s", " ".join(cmd))

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode, cmd, result.stdout, result.stderr
            )

        logger.info("Encoded vertical clip successfully: %s", output_path)


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_vertical_formatting_job(job: VerticalFormattingJob) -> None:
    """Process a VerticalFormattingJob: apply placement to all clips in the batch.

    Iterates through ``job.clips``, calls ``VerticalFormatter.apply_placement_to_clip()``
    for each, tracks progress, handles errors, and respects cancellation.

    After all clips are processed (or the job is cancelled), the job status is
    updated to ``"done"``, ``"cancelled"``, or ``"failed"`` as appropriate.

    Args:
        job: The VerticalFormattingJob to process.  Must be in ``"queued"`` or
             ``"running"`` state.  The job is mutated in-place.
    """
    # Respect pre-cancellation: if the job was cancelled before we started,
    # do not process any clips.
    if job.status == "cancelled":
        logger.info("Job %s was cancelled before processing started", job.job_id)
        return

    formatter = VerticalFormatter()
    job.start_processing()

    reference_resolution: tuple[int, int] | None = None

    # Determine reference resolution from the first clip that has one
    for clip in job.clips:
        res = clip.get("resolution")
        if res:
            reference_resolution = tuple(res)  # type: ignore[assignment]
            break

    # Load transcript if subtitle burning is enabled
    transcript = None
    if job.settings.get("burn_subtitles", False):
        # Try to load transcript from the job's clip batch
        # The transcript should be stored alongside the clips in the output directory
        transcript_path = Path(job.output_dir) / "transcript.json"
        if transcript_path.exists():
            try:
                import json as _json
                from pipeline.models import Transcript
                with open(transcript_path, "r", encoding="utf-8") as fh:
                    transcript_data = _json.load(fh)
                transcript = Transcript.from_dict(transcript_data)
                logger.info("Loaded transcript with %d segments for subtitle burning", len(transcript.segments))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load transcript for subtitle burning: %s", exc)
        else:
            logger.warning("Transcript file not found at %s, subtitles will be skipped", transcript_path)

    for clip in job.clips:
        # Check for cancellation before processing each clip
        if job.status == "cancelled":
            logger.info("Job %s cancelled; stopping batch processing", job.job_id)
            return

        clip_path = clip.get("path", "")
        clip_name = clip.get("name", os.path.basename(clip_path))
        clip_resolution_raw = clip.get("resolution")
        clip_resolution: tuple[int, int] | None = (
            tuple(clip_resolution_raw) if clip_resolution_raw else None  # type: ignore[assignment]
        )
        
        # Extract clip timing information if available
        clip_start = clip.get("start", 0.0)
        clip_end = clip.get("end", 0.0)

        # Update current clip display
        job.current_clip = clip_name

        # Compute output path
        output_path = get_output_path(clip_path, job.settings, job.output_dir)

        try:
            formatter.apply_placement_to_clip(
                clip_path=clip_path,
                facecam_region=job.facecam_region,
                canvas_layout=job.canvas_layout,
                output_path=output_path,
                config=_make_job_config(job),
                reference_resolution=reference_resolution,
                clip_resolution=clip_resolution,
                transcript=transcript,
                clip_start=clip_start,
                clip_end=clip_end,
                settings=job.settings,
            )

            # Handle replacement / backup
            if job.settings.get("replace_originals", False):
                if job.settings.get("backup", True):
                    _backup_clip(clip_path, job.output_dir)
                _replace_clip(clip_path, output_path)

            job.increment_progress(clip_name)
            logger.info(
                "Job %s: processed clip %s (%d/%d)",
                job.job_id, clip_name, job.clips_processed, job.clips_total,
            )

        except Exception as exc:  # noqa: BLE001
            error_msg = f"Failed to process clip '{clip_name}': {exc}"
            logger.error("Job %s: %s", job.job_id, error_msg)
            job.add_error(error_msg)
            # Continue processing remaining clips

    # Final status
    if job.status != "cancelled":
        job.complete_processing()


# ---------------------------------------------------------------------------
# Backup and replacement helpers
# ---------------------------------------------------------------------------

def create_backup_directory(output_dir: str) -> str:
    """Create a timestamped backup directory inside ``output_dir``.

    Args:
        output_dir: Base output directory.

    Returns:
        Path to the newly created backup directory.
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(output_dir, f"backup_{timestamp}")
    os.makedirs(backup_dir, exist_ok=True)
    return backup_dir


def _backup_clip(clip_path: str, output_dir: str) -> str:
    """Copy a clip to the backup directory.

    Args:
        clip_path:  Path to the original clip.
        output_dir: Base output directory (backup dir is created inside it).

    Returns:
        Path to the backup copy.
    """
    backup_dir = create_backup_directory(output_dir)
    dest = os.path.join(backup_dir, os.path.basename(clip_path))
    shutil.copy2(clip_path, dest)
    logger.info("Backed up %s → %s", clip_path, dest)
    return dest


def _replace_clip(original_path: str, new_path: str) -> None:
    """Replace the original clip with the new vertical version.

    Args:
        original_path: Path to the original clip to be replaced.
        new_path:      Path to the new vertical clip.
    """
    shutil.move(new_path, original_path)
    logger.info("Replaced %s with vertical version", original_path)


def backup_clips(clip_paths: list[str], output_dir: str) -> str:
    """Create a backup of multiple clips in a single timestamped directory.

    Args:
        clip_paths: List of paths to original clips.
        output_dir: Base output directory.

    Returns:
        Path to the backup directory containing all copies.
    """
    backup_dir = create_backup_directory(output_dir)
    for clip_path in clip_paths:
        dest = os.path.join(backup_dir, os.path.basename(clip_path))
        shutil.copy2(clip_path, dest)
        logger.info("Backed up %s → %s", clip_path, dest)
    return backup_dir


def replace_clips_with_vertical(
    original_paths: list[str],
    vertical_paths: list[str],
    output_dir: str,
    backup: bool = True,
) -> str | None:
    """Replace original clips with their vertical versions.

    Optionally creates a backup of the originals first.

    Args:
        original_paths: Paths to the original clips.
        vertical_paths: Paths to the corresponding vertical clips.
        output_dir:     Base output directory (used for backup).
        backup:         If True, back up originals before replacing.

    Returns:
        Path to the backup directory if backup was created, else None.
    """
    backup_dir: str | None = None

    if backup:
        backup_dir = backup_clips(original_paths, output_dir)

    for orig, vert in zip(original_paths, vertical_paths):
        shutil.move(vert, orig)
        logger.info("Replaced %s with vertical version", orig)

    return backup_dir


def restore_from_backup(backup_dir: str, original_paths: list[str]) -> None:
    """Restore original clips from a backup directory.

    Args:
        backup_dir:     Path to the backup directory created by ``backup_clips()``.
        original_paths: Paths to the original clip locations to restore to.

    Raises:
        FileNotFoundError: If a backup file is not found.
    """
    for original_path in original_paths:
        filename = os.path.basename(original_path)
        backup_path = os.path.join(backup_dir, filename)
        if not os.path.exists(backup_path):
            raise FileNotFoundError(
                f"Backup file not found: {backup_path}"
            )
        shutil.copy2(backup_path, original_path)
        logger.info("Restored %s from backup %s", original_path, backup_path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _probe_resolution(clip_path: str) -> tuple[int, int]:
    """Use ffprobe to determine the resolution of a video clip.

    Args:
        clip_path: Path to the video file.

    Returns:
        (width, height) tuple.

    Raises:
        RuntimeError: If ffprobe fails or returns no stream info.
    """
    import json as _json

    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-select_streams", "v:0",
        clip_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = _json.loads(result.stdout)
        streams = data.get("streams", [])
        if not streams:
            raise RuntimeError(f"No video streams found in {clip_path}")
        w = streams[0]["width"]
        h = streams[0]["height"]
        return int(w), int(h)
    except (KeyError, ValueError, OSError) as exc:
        raise RuntimeError(f"Failed to probe resolution of {clip_path}: {exc}") from exc


def _make_job_config(job: VerticalFormattingJob):
    """Create a minimal config-like object from job settings."""
    from types import SimpleNamespace
    return SimpleNamespace(
        output_crf=job.settings.get("crf", 23),
        output_codec=job.settings.get("codec", "libx264"),
    )
