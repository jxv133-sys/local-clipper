"""Subtitle generation stage: creates SRT files and burns subtitles into clips."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import time

from config import Config
from pipeline.audio_extractor import FFMPEG_VERSION  # noqa: F401 — used for version-specific flag selection
from pipeline.exceptions import SubtitleError
from pipeline.models import Clip, SRTEntry, Transcript

logger = logging.getLogger(__name__)


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
# Word-level subtitle helpers
# ---------------------------------------------------------------------------

_WORDS_PER_GROUP = 4  # group words into short phrases for readability


def _word_level_entries(seg, clip_start: float, start_index: int) -> list[SRTEntry]:
    """Split a segment into SRT entries grouped by word boundaries.

    Groups words into phrases of up to _WORDS_PER_GROUP words each.
    Timestamps are adjusted to be relative to clip_start.

    Args:
        seg: Segment with a non-empty .words list.
        clip_start: Absolute start time of the clip (seconds).
        start_index: 1-based SRT index for the first entry produced.

    Returns:
        List of SRTEntry objects, one per word group.
    """
    entries: list[SRTEntry] = []
    words = seg.words
    idx = start_index
    i = 0
    while i < len(words):
        group = words[i : i + _WORDS_PER_GROUP]
        group_text = "".join(w.word for w in group).strip()
        if group_text:
            rel_start = max(0.0, group[0].start - clip_start)
            rel_end = max(0.0, group[-1].end - clip_start)
            entries.append(SRTEntry(index=idx, start=rel_start, end=rel_end, text=group_text))
            idx += 1
        i += _WORDS_PER_GROUP
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

    logger.info("SubtitleGenerator starting — %d clip(s)", len(clips))
    t0_total = time.time()

    for clip, raw_path in zip(clips, clip_paths):
        t0 = time.time()
        # 1. Collect in-range, non-empty segments
        srt_entries: list[SRTEntry] = []
        entry_index = 1
        for seg in transcript.segments:
            if not seg.text.strip():
                continue
            # Include segment if it overlaps with the clip window
            if seg.end <= clip.start or seg.start >= clip.end:
                continue
            if seg.words:
                # Word-level splitting: group into short phrases
                new_entries = _word_level_entries(seg, clip.start, entry_index)
                srt_entries.extend(new_entries)
                entry_index += len(new_entries)
            else:
                # Fallback: segment-level entry (openai-whisper path)
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

        # 3. Optionally burn subtitles into the clip
        if config.burn_subtitles:
            if not srt_entries:
                logger.info(
                    "[SubtitleGenerator] Clip #%d has no transcript segments — skipping subtitle burn",
                    clip.rank,
                )
                # No overlapping segments — raw clip is already the final output; nothing to do
            else:
                burned_path = base + "_subtitled.mp4"
                _burn_subtitles(raw_path, srt_path, burned_path)
                os.replace(burned_path, raw_path)
                logger.info("  Clip #%d: %d subtitle entry(ies) burned, output: %s (%.1fs)",
                            clip.rank, len(srt_entries), raw_path, time.time() - t0)
        else:
            logger.info("  Clip #%d: %d subtitle entry(ies) written to SRT (burn disabled), output: %s (%.1fs)",
                        clip.rank, len(srt_entries), raw_path, time.time() - t0)

        final_paths.append(raw_path)

    logger.info("SubtitleGenerator complete — %d clip(s) in %.1fs", len(final_paths), time.time() - t0_total)
    return final_paths


def _burn_subtitles(video_path: str, srt_path: str, output_path: str) -> None:
    """Burn SRT subtitles into a video using Pillow-rendered overlays.

    Renders each subtitle entry as a transparent PNG image using Pillow,
    then uses FFmpeg's overlay filter to composite them onto the video.
    This approach works with any FFmpeg build (no libass or freetype required).

    Args:
        video_path: Path to the input video file.
        srt_path: Path to the SRT subtitle file.
        output_path: Path for the output video with burned-in subtitles.

    Raises:
        SubtitleError: If FFmpeg exits with a non-zero return code.
    """
    import shutil as _shutil
    import tempfile

    with open(srt_path, "r", encoding="utf-8") as fh:
        entries = parse_srt(fh.read())

    if not entries:
        _shutil.copy2(video_path, output_path)
        return

    ffmpeg_bin = _shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"

    # Get video dimensions
    probe = subprocess.run(
        [ffmpeg_bin, "-i", video_path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    width, height = 1280, 720  # sensible defaults
    for line in probe.stderr.splitlines():
        if "Video:" in line and "x" in line:
            import re
            m = re.search(r"(\d{3,5})x(\d{3,5})", line)
            if m:
                width, height = int(m.group(1)), int(m.group(2))
                break

    # Render subtitle images with Pillow
    try:
        from PIL import Image, ImageDraw, ImageFont
        _pillow_available = True
    except ImportError:
        _pillow_available = False

    if not _pillow_available:
        # No Pillow — copy without subtitles
        _shutil.copy2(video_path, output_path)
        return

    tmp_dir = tempfile.mkdtemp(prefix="subs_")
    try:
        # Probe source clip duration so image inputs don't truncate the output
        probe_dur = subprocess.run(
            [ffmpeg_bin, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            clip_duration = float(probe_dur.stdout.strip())
        except (ValueError, AttributeError):
            clip_duration = None

        # Build a filter_complex with one overlay per subtitle entry
        # Each subtitle is a PNG image shown only during its time window
        overlay_inputs: list[str] = []
        filter_parts: list[str] = []
        input_args: list[str] = [ffmpeg_bin, "-y", "-i", video_path]

        font_size = max(24, height // 22)
        pad = 8
        bottom_margin = max(40, height // 18)

        for i, entry in enumerate(entries):
            text = entry.text.strip().replace("\n", " ")
            if not text:
                continue

            # Render text onto transparent image
            img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)

            # Try to load a system font, fall back to default
            font = None
            _selected_font_path = None
            for font_path in [
                # macOS
                "/System/Library/Fonts/Helvetica.ttc",
                "/System/Library/Fonts/Arial.ttf",
                "/Library/Fonts/Arial.ttf",
                # Linux
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
                # Windows
                "C:/Windows/Fonts/arial.ttf",
                "C:/Windows/Fonts/Arial.ttf",
            ]:
                if os.path.exists(font_path):
                    try:
                        font = ImageFont.truetype(font_path, font_size)
                        _selected_font_path = font_path
                        break
                    except Exception:
                        pass
            if font is None:
                font = ImageFont.load_default()
                logger.debug("Subtitle font: using FFmpeg default (no font found)")
            else:
                logger.debug("Subtitle font selected: %s", _selected_font_path)

            # Measure text
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            x = (width - tw) // 2
            y = height - th - bottom_margin

            # Draw shadow/border
            for dx, dy in [(-2, -2), (2, -2), (-2, 2), (2, 2),
                           (-1, 0), (1, 0), (0, -1), (0, 1)]:
                draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0, 220))
            # Draw text
            draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))

            png_path = os.path.join(tmp_dir, f"sub_{i:04d}.png")
            img.save(png_path)

            input_args += ["-loop", "1", "-t", str(clip_duration) if clip_duration else "60", "-i", png_path]
            overlay_inputs.append((i + 1, entry.start, entry.end))

        if not overlay_inputs:
            _shutil.copy2(video_path, output_path)
            return

        # Build filter_complex: chain overlays
        # [0:v][1]overlay=enable='between(t,s,e)'[v1]; [v1][2]overlay=...
        fc_parts: list[str] = []
        prev = "[0:v]"
        for idx, (inp_idx, start, end) in enumerate(overlay_inputs):
            out_label = f"[v{idx + 1}]" if idx < len(overlay_inputs) - 1 else "[vout]"
            fc_parts.append(
                f"{prev}[{inp_idx}:v]overlay=0:0:enable='between(t,{start:.3f},{end:.3f})'{out_label}"
            )
            prev = out_label

        filter_complex = ";".join(fc_parts)

        cmd = input_args + [
            "-filter_complex", filter_complex,
            "-map", "[vout]",
            "-map", "0:a:0",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-c:a", "copy",
            "-threads", "0",
            "-movflags", "+faststart",
        ]
        # Pin output duration to source so still-image inputs can't truncate audio
        if clip_duration is not None:
            cmd += ["-t", str(clip_duration)]
        cmd.append(output_path)

        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )

        if result.returncode != 0:
            raise SubtitleError(
                f"FFmpeg failed (exit code {result.returncode}) while burning subtitles "
                f"into '{video_path}'. stderr: {result.stderr.strip()}"
            )
    finally:
        _shutil.rmtree(tmp_dir, ignore_errors=True)
