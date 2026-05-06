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

    The filter chain:
    1. ``[canvas]``: scale + pad the source to the full 9:16 canvas with the
       gameplay region at the bottom (reuses FrameReformatter).
    2. ``[facecam_scaled]``: crop the facecam from the source, scale to fit
       the top region of the canvas while preserving aspect ratio.
    3. ``[with_facecam]``: overlay the facecam on the canvas at (overlay_x, 0).

    Args:
        src_width:      Source video width in pixels.
        src_height:     Source video height in pixels.
        facecam_region: Confirmed facecam placement in source frame coordinates.
        layout:         9:16 canvas layout.

    Returns:
        A filter_complex string suitable for ``ffmpeg -filter_complex``.
    """
    reformatter = FrameReformatter()
    canvas_fragment = reformatter.build_canvas_filter(src_width, src_height, layout)

    # Facecam crop + scale
    crop_w = facecam_region.width
    crop_h = facecam_region.height
    crop_x = facecam_region.x
    crop_y = facecam_region.y

    facecam_target_w = layout.facecam_width
    facecam_target_h = layout.facecam_height

    # Scale to fit within facecam region, preserving aspect ratio
    scale_w = facecam_target_w
    scale_h = round(crop_h * facecam_target_w / crop_w) if crop_w > 0 else facecam_target_h

    if scale_h > facecam_target_h:
        scale_h = facecam_target_h
        scale_w = round(crop_w * facecam_target_h / crop_h) if crop_h > 0 else facecam_target_w

    # Centre horizontally within the facecam region
    overlay_x = (facecam_target_w - scale_w) // 2

    # Build the complete filter_complex
    # canvas_fragment.filter_str is e.g.:
    #   "[0:v]scale=...,pad=...[canvas]"
    facecam_filter = (
        f"[0:v]crop={crop_w}:{crop_h}:{crop_x}:{crop_y},"
        f"scale={scale_w}:{scale_h}[facecam_scaled]"
    )
    overlay_filter = f"[canvas][facecam_scaled]overlay={overlay_x}:0[with_facecam]"

    return f"{canvas_fragment.filter_str};{facecam_filter};{overlay_filter}"


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
    ) -> None:
        """Apply the facecam placement to a single clip and encode the output.

        If ``reference_resolution`` and ``clip_resolution`` are both provided
        and differ, the facecam region coordinates are proportionally scaled
        before building the filter chain.

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

        Raises:
            subprocess.CalledProcessError: If FFmpeg encoding fails.
            RuntimeError: If the clip resolution cannot be determined.
        """
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

        # Build FFmpeg command
        crf = getattr(config, "output_crf", 23)
        codec = getattr(config, "output_codec", "libx264")

        cmd = [
            "ffmpeg",
            "-y",                    # overwrite output
            "-i", clip_path,
            "-filter_complex", filter_complex,
            "-map", "[with_facecam]",
            "-map", "0:a?",          # preserve audio if present
            "-c:v", codec,
            "-crf", str(crf),
            "-c:a", "copy",
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
