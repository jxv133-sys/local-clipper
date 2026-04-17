"""Clip extraction stage: extracts video clips from the source file using FFmpeg."""

from __future__ import annotations

import os
import subprocess

from config import Config
from pipeline.exceptions import ClipExtractionError
from pipeline.models import Clip


def extract_clips(config: Config, clips: list[Clip], video_path: str) -> list[str]:
    """Extract each Clip from *video_path* using FFmpeg.

    Attempts stream-copy first to preserve original quality.  If the probed
    duration of the stream-copied file differs from the requested duration by
    more than 1 second, re-extracts with re-encoding for accurate cut points.

    Output files are named ``clip_<rank>_<start_seconds>s.mp4`` and written to
    ``config.output_dir``.

    Args:
        config: Pipeline configuration (``output_dir`` must be set).
        clips: List of Clip objects to extract, sorted by rank.
        video_path: Path to the source video file.

    Returns:
        List of paths to the extracted ``.mp4`` files, in rank order.

    Raises:
        ClipExtractionError: If FFmpeg exits with a non-zero return code.
    """
    os.makedirs(config.output_dir, exist_ok=True)

    output_paths: list[str] = []

    for clip in sorted(clips, key=lambda c: c.rank):
        filename = f"clip_{clip.rank}_{int(clip.start)}s.mp4"
        output_path = os.path.join(config.output_dir, filename)

        # --- Attempt 1: stream-copy (no re-encoding) ---
        _run_ffmpeg(
            [
                "ffmpeg", "-y",
                "-ss", str(clip.start),
                "-to", str(clip.end),
                "-i", video_path,
                "-c", "copy",
                output_path,
            ],
            output_path,
        )

        # --- Verify duration via ffprobe ---
        requested_duration = clip.end - clip.start
        probed_duration = _probe_duration(output_path)

        if probed_duration is not None and abs(probed_duration - requested_duration) > 1.0:
            # --- Attempt 2: re-encode for accurate cut points ---
            _run_ffmpeg(
                [
                    "ffmpeg", "-y",
                    "-ss", str(clip.start),
                    "-to", str(clip.end),
                    "-i", video_path,
                    output_path,
                ],
                output_path,
            )

        output_paths.append(output_path)

    return output_paths


def _run_ffmpeg(cmd: list[str], output_path: str) -> None:
    """Run an FFmpeg command and raise ClipExtractionError on failure.

    Args:
        cmd: The FFmpeg command as a list of strings.
        output_path: The expected output file path (used in error messages).

    Raises:
        ClipExtractionError: If FFmpeg exits with a non-zero return code.
    """
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise ClipExtractionError(
            f"FFmpeg failed (exit code {result.returncode}) while writing "
            f"'{output_path}'. stderr: {result.stderr.strip()}"
        )


def _probe_duration(file_path: str) -> float | None:
    """Use ffprobe to get the duration of a media file in seconds.

    Returns None if ffprobe fails or the output cannot be parsed.
    """
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        return None
    try:
        return float(result.stdout.strip())
    except (ValueError, AttributeError):
        return None
