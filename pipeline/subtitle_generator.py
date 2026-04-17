"""Subtitle generation stage: creates SRT files and burns subtitles into clips."""

from __future__ import annotations

import os
import re
import subprocess

from config import Config
from pipeline.exceptions import SubtitleError
from pipeline.models import Clip, SRTEntry, Transcript


# ---------------------------------------------------------------------------
# SRT serialization / parsing
# ---------------------------------------------------------------------------

def _seconds_to_srt_time(seconds: float) -> str:
    """Convert a float seconds value to SRT timestamp format HH:MM:SS,mmm."""
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    # Guard against rounding pushing millis to 1000
    if millis >= 1000:
        millis = 999
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _srt_time_to_seconds(time_str: str) -> float:
    """Parse an SRT timestamp string HH:MM:SS,mmm to float seconds."""
    # Accept both comma and period as decimal separator
    time_str = time_str.replace(",", ".")
    parts = time_str.split(":")
    hours = int(parts[0])
    minutes = int(parts[1])
    sec_parts = parts[2].split(".")
    secs = int(sec_parts[0])
    millis = int(sec_parts[1]) if len(sec_parts) > 1 else 0
    return hours * 3600 + minutes * 60 + secs + millis / 1000.0


def serialize_srt(entries: list[SRTEntry]) -> str:
    """Serialize a list of SRTEntry objects to an SRT-format string.

    Args:
        entries: List of SRTEntry objects to serialize.

    Returns:
        A string in SRT format.
    """
    blocks: list[str] = []
    for entry in entries:
        start_ts = _seconds_to_srt_time(entry.start)
        end_ts = _seconds_to_srt_time(entry.end)
        blocks.append(f"{entry.index}\n{start_ts} --> {end_ts}\n{entry.text}")
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def parse_srt(srt_content: str) -> list[SRTEntry]:
    """Parse an SRT-format string into a list of SRTEntry objects.

    Args:
        srt_content: A string in SRT format.

    Returns:
        List of SRTEntry objects.
    """
    entries: list[SRTEntry] = []
    # Split on blank lines between blocks
    blocks = re.split(r"\n\s*\n", srt_content.strip())
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        if len(lines) < 3:
            continue
        try:
            index = int(lines[0].strip())
        except ValueError:
            continue
        # Parse timestamp line
        ts_match = re.match(
            r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})",
            lines[1].strip(),
        )
        if not ts_match:
            continue
        start = _srt_time_to_seconds(ts_match.group(1))
        end = _srt_time_to_seconds(ts_match.group(2))
        text = "\n".join(lines[2:])
        entries.append(SRTEntry(index=index, start=start, end=end, text=text))
    return entries


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def generate_subtitles(
    config: Config,
    clips: list[Clip],
    transcript: Transcript,
    clip_paths: list[str],
) -> list[str]:
    """Burn subtitles into each extracted clip.

    For each clip:
    1. Collect non-empty Segments whose time range falls within [clip.start, clip.end].
    2. Adjust timestamps to be relative to clip.start.
    3. Serialize to SRT and write alongside the clip in config.output_dir.
    4. Use FFmpeg to burn the SRT into the clip video.

    Args:
        config: Pipeline configuration.
        clips: List of Clip objects (in rank order).
        transcript: Full transcript with all segments.
        clip_paths: List of paths to the extracted (raw) clip .mp4 files, in rank order.

    Returns:
        List of paths to the final subtitle-burned .mp4 files, in rank order.

    Raises:
        SubtitleError: If FFmpeg exits with a non-zero return code.
    """
    os.makedirs(config.output_dir, exist_ok=True)
    final_paths: list[str] = []

    for clip, raw_path in zip(clips, clip_paths):
        # 1. Collect in-range, non-empty segments
        srt_entries: list[SRTEntry] = []
        entry_index = 1
        for seg in transcript.segments:
            if not seg.text.strip():
                continue
            # Include segment if it overlaps with the clip window
            if seg.end <= clip.start or seg.start >= clip.end:
                continue
            rel_start = max(0.0, seg.start - clip.start)
            rel_end = max(0.0, seg.end - clip.start)
            srt_entries.append(
                SRTEntry(index=entry_index, start=rel_start, end=rel_end, text=seg.text.strip())
            )
            entry_index += 1

        # 2. Write SRT file alongside the clip
        base = os.path.splitext(raw_path)[0]
        srt_path = base + ".srt"
        with open(srt_path, "w", encoding="utf-8") as fh:
            fh.write(serialize_srt(srt_entries))

        # 3. Burn subtitles into the clip using FFmpeg
        # Output to a temp name then replace the original
        burned_path = base + "_subtitled.mp4"
        _burn_subtitles(raw_path, srt_path, burned_path)

        # Replace the raw clip with the subtitled version
        os.replace(burned_path, raw_path)
        final_paths.append(raw_path)

    return final_paths


def _burn_subtitles(video_path: str, srt_path: str, output_path: str) -> None:
    """Burn SRT subtitles into a video using FFmpeg drawtext filter.

    Uses drawtext instead of the subtitles filter to avoid requiring libass,
    which is not included in the standard Homebrew FFmpeg build.

    Args:
        video_path: Path to the input video file.
        srt_path: Path to the SRT subtitle file.
        output_path: Path for the output video with burned-in subtitles.

    Raises:
        SubtitleError: If FFmpeg exits with a non-zero return code.
    """
    # Parse the SRT file to get timed text entries
    with open(srt_path, "r", encoding="utf-8") as fh:
        entries = parse_srt(fh.read())

    if not entries:
        # No subtitles — just copy the video as-is
        import shutil
        shutil.copy2(video_path, output_path)
        return

    # Build a drawtext filter chain — one drawtext per subtitle entry.
    # Each entry is shown only during its time window using 'enable' expression.
    filter_parts: list[str] = []
    for entry in entries:
        # Escape text for FFmpeg drawtext: backslash, colon, single quote, newline
        text = entry.text.strip()
        text = text.replace("\\", "\\\\")
        text = text.replace("'", "\\'")
        text = text.replace(":", "\\:")
        text = text.replace("\n", " ")

        start_s = entry.start
        end_s = entry.end

        filter_parts.append(
            f"drawtext=text='{text}'"
            f":fontsize=24"
            f":fontcolor=white"
            f":borderw=2"
            f":bordercolor=black"
            f":x=(w-text_w)/2"
            f":y=h-th-40"
            f":enable='between(t,{start_s:.3f},{end_s:.3f})'"
        )

    # Chain all drawtext filters together
    vf = ",".join(filter_parts)

    # Use full path to ffmpeg since it may not be on PATH in all environments
    import shutil as _shutil
    ffmpeg_bin = _shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"

    cmd = [
        ffmpeg_bin, "-y",
        "-i", video_path,
        "-vf", vf,
        "-c:a", "copy",
        output_path,
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise SubtitleError(
            f"FFmpeg failed (exit code {result.returncode}) while burning subtitles "
            f"into '{video_path}'. stderr: {result.stderr.strip()}"
        )
