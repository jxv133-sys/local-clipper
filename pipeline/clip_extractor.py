"""Clip extraction stage: extracts video clips from the source file using FFmpeg."""

from __future__ import annotations

import concurrent.futures
import logging
import os
import subprocess
import time

from config import Config
from pipeline.audio_extractor import FFMPEG_VERSION  # noqa: F401 — used for version-specific flag selection
from pipeline.exceptions import ClipExtractionError
from pipeline.models import Clip

logger = logging.getLogger(__name__)


def trim_clip_silence(clip_path: str, config: Config, clip_rank: int = 0) -> str:
    """Trim leading/trailing silence from a clip using ffmpeg silenceremove.
    
    Note: This function re-encodes audio which can sometimes cause issues.
    Consider disabling silence trimming if audio problems occur.
    """
    stem, ext = os.path.splitext(clip_path)
    trimmed_path = f"{stem}_trimmed{ext}"

    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", clip_path,
                "-af",
                (
                    "silenceremove="
                    "start_periods=1:start_silence=0.5:start_threshold=-50dB:"
                    "stop_periods=1:stop_silence=0.5:stop_threshold=-50dB"
                ),
                "-c:v", "copy",          # don't re-encode video during silence trim
                "-c:a", "aac", "-b:a", "128k",  # re-encode audio with silenceremove filter
                "-shortest",             # Ensure audio and video stay in sync
                "-movflags", "+faststart",
                trimmed_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            logger.warning(
                "  Clip #%d: silence trim ffmpeg failed — keeping original. stderr: %s",
                clip_rank, result.stderr.strip(),
            )
            return clip_path

        trimmed_duration = _probe_duration(trimmed_path)
        if trimmed_duration is None:
            logger.warning("  Clip #%d: could not probe trimmed duration — keeping original", clip_rank)
            try:
                os.remove(trimmed_path)
            except OSError:
                pass
            return clip_path

        if trimmed_duration < config.min_clip_duration:
            logger.info("  Clip #%d: silence trim skipped (would shorten below min_clip_duration)", clip_rank)
            try:
                os.remove(trimmed_path)
            except OSError:
                pass
            return clip_path

        original_duration = _probe_duration(clip_path) or 0.0
        os.replace(trimmed_path, clip_path)
        logger.info("  Clip #%d: trimmed silence, duration %.1fs → %.1fs",
                    clip_rank, original_duration, trimmed_duration)
        return clip_path

    except Exception as exc:
        logger.warning("  Clip #%d: silence trim error — keeping original: %s", clip_rank, exc)
        try:
            os.remove(trimmed_path)
        except OSError:
            pass
        return clip_path


def _extract_single_clip(config: Config, clip: Clip, video_path: str) -> tuple[int, str]:
    """Extract a single clip and return ``(clip.rank, output_path)``."""
    t0 = time.time()
    filename = f"clip_{clip.rank}_{int(clip.start)}s.mp4"
    output_path = os.path.join(config.output_dir, filename)
    requested_duration = clip.end - clip.start

    # Input-side seek (-ss before -i) jumps to nearest keyframe instantly.
    # Re-encode to H.264/AAC for compatibility; ultrafast keeps it quick.
    extract_cmd = [
        "ffmpeg", "-y",
        "-ss", str(clip.start),
        "-i", video_path,
        "-t", str(requested_duration),
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-avoid_negative_ts", "make_zero",
        "-movflags", "+faststart",
        output_path,
    ]

    _run_ffmpeg(extract_cmd, output_path)

    if config.trim_silence:
        output_path = trim_clip_silence(output_path, config, clip_rank=clip.rank)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024) if os.path.exists(output_path) else 0.0
    logger.info("  Clip #%d: %s (%.2f MB, %.1fs)", clip.rank, output_path, file_size_mb, time.time() - t0)
    return clip.rank, output_path


def extract_clips(config: Config, clips: list[Clip], video_path: str) -> list[str]:
    """Extract each Clip from *video_path* using FFmpeg, running concurrently.

    Always re-encodes to H.264/AAC for maximum compatibility and to prevent
    audio issues. Output files are named ``clip_<rank>_<start_seconds>s.mp4``
    and written to ``config.output_dir``.

    Args:
        config: Pipeline configuration (``output_dir`` must be set).
        clips: List of Clip objects to extract.
        video_path: Path to the source video file.

    Returns:
        List of paths to the extracted ``.mp4`` files, in rank order.

    Raises:
        ClipExtractionError: If FFmpeg exits with a non-zero return code.
    """
    if not clips:
        return []

    # Validate output_dir before any work
    if config.output_dir.startswith(('http://', 'https://', 'ftp://')):
        raise ClipExtractionError(
            f"Invalid output directory: '{config.output_dir}'. "
            f"Output directory must be a local file path, not a URL."
        )

    os.makedirs(config.output_dir, exist_ok=True)

    logger.info("ClipExtractor starting — %d clip(s) to extract", len(clips))
    t0_total = time.time()

    max_workers = min(len(clips), os.cpu_count() or 4)
    results: list[tuple[int, str]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_extract_single_clip, config, clip, video_path): clip
            for clip in clips
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except ClipExtractionError:
                raise

    # Sort by rank to preserve rank-ordered output regardless of completion order
    results.sort(key=lambda t: t[0])
    output_paths = [path for _, path in results]

    logger.info("ClipExtractor complete — %d clip(s) in %.1fs", len(output_paths), time.time() - t0_total)
    return output_paths


def _run_ffmpeg(cmd: list[str], output_path: str) -> None:
    """Run an FFmpeg command and raise ClipExtractionError on failure."""
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


def _log_input_streams(file_path: str, clip_rank: int = 0) -> None:
    """Log the streams present in an input file for debugging.
    
    Args:
        file_path: Path to the input file to inspect.
        clip_rank: Clip rank for logging context.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_streams",
                file_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
        
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            streams = data.get("streams", [])
            
            stream_info = []
            for i, s in enumerate(streams):
                codec_type = s.get("codec_type", "unknown")
                codec_name = s.get("codec_name", "unknown")
                stream_info.append(f"#{i}:{codec_type}:{codec_name}")
            
            logger.info("  Clip #%d: Input file streams: %s", clip_rank, ", ".join(stream_info))
        else:
            logger.warning("  Clip #%d: Could not probe input streams", clip_rank)
            
    except Exception as exc:
        logger.warning("  Clip #%d: Error probing input streams: %s", clip_rank, exc)


def _log_output_streams(file_path: str, clip_rank: int = 0) -> None:
    """Log the streams present in an output file for debugging.
    
    Args:
        file_path: Path to the output file to inspect.
        clip_rank: Clip rank for logging context.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_streams",
                file_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
        
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            streams = data.get("streams", [])
            
            stream_info = []
            for s in streams:
                codec_type = s.get("codec_type", "unknown")
                codec_name = s.get("codec_name", "unknown")
                stream_info.append(f"{codec_type}:{codec_name}")
            
            logger.info("  Clip #%d: Output file streams: %s", clip_rank, ", ".join(stream_info))
        else:
            logger.warning("  Clip #%d: Could not probe output streams", clip_rank)
            
    except Exception as exc:
        logger.warning("  Clip #%d: Error probing output streams: %s", clip_rank, exc)


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


def generate_thumbnail(clip_path: str) -> str | None:
    """Generate a JPEG thumbnail at the midpoint of a clip.

    Uses ffprobe to determine the clip duration, then extracts a single frame
    at the midpoint with ffmpeg.  Falls back to multiple positions if seeking fails.

    The thumbnail is saved alongside the clip as ``<clip_stem>_thumb.jpg``.

    Args:
        clip_path: Path to the clip ``.mp4`` file.

    Returns:
        The path to the generated thumbnail on success, or ``None`` if
        thumbnail generation fails (non-fatal).
    """
    try:
        duration = _probe_duration(clip_path)
        
        # Determine seek positions to try
        if duration is not None and duration > 2.0:
            # If we have duration info, try midpoint first, then fallbacks
            seek_positions = [duration / 2.0, 1.0, 0.5]
        else:
            # If no duration info, try conservative positions
            seek_positions = [1.0, 0.5, 0.1]

        stem = os.path.splitext(clip_path)[0]
        thumb_path = f"{stem}_thumb.jpg"

        # Try each seek position until one succeeds
        for seek_pos in seek_positions:
            try:
                result = subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-ss", str(seek_pos),
                        "-i", clip_path,
                        "-frames:v", "1",
                        "-q:v", "2",
                        "-f", "image2",  # Explicitly specify output format
                        thumb_path,
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=10,  # Add timeout to prevent hanging
                )
                
                if result.returncode == 0:
                    logger.info("  Thumbnail: %s (seek position: %.1fs)", thumb_path, seek_pos)
                    return thumb_path
                    
            except subprocess.TimeoutExpired:
                logger.warning("Thumbnail generation timed out for %s at position %.1fs", clip_path, seek_pos)
                continue
            except Exception:
                # Continue to next position on any error
                continue

        # If all positions failed, log the last error
        logger.warning("Thumbnail generation failed for %s: could not seek to any position", clip_path)
        return None
        
    except Exception as exc:
        logger.warning("Thumbnail generation error for %s: %s", clip_path, exc)
        return None
