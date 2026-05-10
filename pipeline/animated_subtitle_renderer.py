"""AnimatedSubtitleRenderer: generates ASS subtitle files with animation tags
and builds FFmpeg filter fragments for burning them into vertical shorts clips."""

from __future__ import annotations

import logging
import math
import os
import subprocess
import tempfile

from pipeline.models import FilterFragment, SRTEntry, SubtitleStyle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def escape_ass_text(text: str) -> str:
    """Escape special characters in subtitle text for safe insertion into ASS files.

    ASS override tags use { and } as delimiters, and backslash as an escape
    character.  Any literal occurrences of these characters in subtitle text
    must be escaped so they are not misinterpreted as override tags.

    Escaping order matters: backslash must be escaped first so that the
    backslashes introduced by subsequent escapes are not double-escaped.

    Args:
        text: Raw subtitle text string.

    Returns:
        Escaped string safe for insertion into an ASS Dialogue line.
    """
    # Escape backslash first, then braces
    text = text.replace("\\", "\\\\")
    text = text.replace("{", "\\{")
    text = text.replace("}", "\\}")
    return text


def _centiseconds_to_ass_time(cs: int) -> str:
    """Convert a centiseconds integer to ASS time format "H:MM:SS.cc".

    Args:
        cs: Time in centiseconds (non-negative integer).

    Returns:
        ASS-format time string, e.g. "0:00:01.00" for 100 cs.

    Examples:
        >>> _centiseconds_to_ass_time(100)
        '0:00:01.00'
        >>> _centiseconds_to_ass_time(6000)
        '0:01:00.00'
    """
    cs = max(0, cs)
    total_seconds = cs // 100
    centis = cs % 100
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _build_ass_header(
    canvas_width: int,
    canvas_height: int,
    style: SubtitleStyle,
    config,
) -> str:
    """Build the ASS file header with Script Info, V4+ Styles, and Events sections.

    Args:
        canvas_width:  Canvas width in pixels (used for PlayResX).
        canvas_height: Canvas height in pixels (used for PlayResY).
        style:         SubtitleStyle enum value (affects style parameters).
        config:        Config object with subtitle font/colour settings.

    Returns:
        Complete ASS header string ending with the Events Format line.
    """
    font_name = getattr(config, "subtitle_font_name", "Impact")
    font_size = getattr(config, "subtitle_font_size", 72)
    primary_color = getattr(config, "subtitle_primary_color", "&H00FFFFFF")
    outline_color = getattr(config, "subtitle_outline_color", "&H00000000")
    outline_width = getattr(config, "subtitle_outline_width", 4.0)
    shadow_depth = getattr(config, "subtitle_shadow_depth", 2.0)

    # Bold is always 1 for shorts subtitles (high-impact visual style)
    bold = 1

    # ASS colour format: &HAABBGGRR (alpha, blue, green, red)
    # The config stores colours in ASS format already.

    header_lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        f"PlayResX: {canvas_width}",
        f"PlayResY: {canvas_height}",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        (
            f"Style: Default,{font_name},{font_size},"
            f"{primary_color},&H000000FF,"
            f"{outline_color},&H00000000,"
            f"{bold},0,0,0,"
            f"100,100,0,0,1,"
            f"{outline_width:.0f},{shadow_depth:.0f},"
            f"2,10,10,10,1"
        ),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    return "\n".join(header_lines) + "\n"


# ---------------------------------------------------------------------------
# Word-by-word highlighting builder (TikTok style)
# ---------------------------------------------------------------------------

def _build_word_by_word_line(
    entry: SRTEntry,
    cx: int,
    subtitle_y: int,
    style: SubtitleStyle,
    config,
) -> str:
    """Build an ASS Dialogue text with word-by-word highlighting (TikTok style).

    All words in the group are shown at once, but each word is highlighted
    individually as it's spoken using ASS animation tags with actual word timings.

    Args:
        entry:      SRTEntry with start/end times, text, and optional word_timings.
        cx:         Horizontal centre position in pixels.
        subtitle_y: Vertical position in pixels.
        style:      SubtitleStyle enum value.
        config:     Config object with subtitle settings.

    Returns:
        ASS override + text string for the Dialogue line's Text field.
    """
    # If we have word-level timings, use them for precise highlighting
    if entry.word_timings:
        words_with_timing = entry.word_timings
    else:
        # Fallback: split text and divide duration equally
        words = entry.text.upper().split()
        if not words:
            return f"{{\\an2\\pos({cx},{subtitle_y})}}"
        
        total_cs = round((entry.end - entry.start) * 100)
        word_cs = round(total_cs / max(len(words), 1))
        
        words_with_timing = [
            (word, entry.start + (i * word_cs / 100), entry.start + ((i + 1) * word_cs / 100))
            for i, word in enumerate(words)
        ]

    if not words_with_timing:
        return f"{{\\an2\\pos({cx},{subtitle_y})}}"

    # Get colors for highlighting
    primary_color = getattr(config, "subtitle_primary_color", "&H00FFFFFF")  # White
    highlight_color = getattr(config, "subtitle_highlight_color", "&H0000FFFF")  # Yellow

    parts = [f"{{\\an2\\pos({cx},{subtitle_y})}}"]
    
    if style == SubtitleStyle.BUBBLE:
        # Bubble: Scale pop on each word as it's highlighted
        for word, word_start, word_end in words_with_timing:
            escaped_word = escape_ass_text(word.upper().strip())
            # Calculate timing relative to entry start (in centiseconds)
            start_cs = round((word_start - entry.start) * 100)
            end_cs = round((word_end - entry.start) * 100)
            
            # Word starts dimmed, scales up when highlighted, then dims again
            parts.append(
                f"{{\\1c{primary_color}\\alpha&H80&"  # Start dimmed
                f"\\t({start_cs},{start_cs + 50},\\1c{highlight_color}\\alpha&H00&\\fscx110\\fscy110)"  # Highlight + scale
                f"\\t({end_cs - 50},{end_cs},\\1c{primary_color}\\alpha&H80&\\fscx100\\fscy100)}}"  # Dim again
                f"{escaped_word} "
            )
    
    elif style == SubtitleStyle.POPUP:
        # Popup: Each word pops in as it's spoken
        for word, word_start, word_end in words_with_timing:
            escaped_word = escape_ass_text(word.upper().strip())
            start_cs = round((word_start - entry.start) * 100)
            
            # Word starts invisible, pops in when spoken
            parts.append(
                f"{{\\alpha&HFF&"  # Start invisible
                f"\\t({start_cs},{start_cs + 80},\\alpha&H00&\\fscx100\\fscy100)}}"  # Pop in
                f"{escaped_word} "
            )
    
    elif style == SubtitleStyle.HIGHLIGHT:
        # Highlight: Background color changes for each word
        for word, word_start, word_end in words_with_timing:
            escaped_word = escape_ass_text(word.upper().strip())
            start_cs = round((word_start - entry.start) * 100)
            end_cs = round((word_end - entry.start) * 100)
            
            # Word starts normal, gets highlighted background, then normal again
            parts.append(
                f"{{\\1c{primary_color}"
                f"\\t({start_cs},{start_cs + 50},\\1c{highlight_color}\\bord6)"  # Highlight
                f"\\t({end_cs - 50},{end_cs},\\1c{primary_color}\\bord4)}}"  # Normal
                f"{escaped_word} "
            )
    
    elif style == SubtitleStyle.KARAOKE:
        # Karaoke: Use native \k tags for color change with actual word durations
        for i, (word, word_start, word_end) in enumerate(words_with_timing):
            escaped_word = escape_ass_text(word.upper().strip())
            # Calculate word duration in centiseconds
            word_duration_cs = round((word_end - word_start) * 100)
            if i < len(words_with_timing) - 1:
                parts.append(f"{{\\k{word_duration_cs}}}{escaped_word} ")
            else:
                parts.append(f"{{\\k{word_duration_cs}}}{escaped_word}")
        return "".join(parts).rstrip()
    
    return "".join(parts).rstrip()


# ---------------------------------------------------------------------------
# Karaoke line builder (legacy - now handled by _build_word_by_word_line)
# ---------------------------------------------------------------------------

def _build_karaoke_line(
    entry: SRTEntry,
    cx: int,
    subtitle_y: int,
) -> str:
    """Build an ASS karaoke Dialogue text with per-word \\k timing tags.

    Each word is assigned an equal share of the entry's total duration.
    The \\k tag value is in centiseconds.

    Args:
        entry:      SRTEntry with start/end times and text.
        cx:         Horizontal centre position in pixels.
        subtitle_y: Vertical position in pixels.

    Returns:
        ASS override + text string for the Dialogue line's Text field.
    """
    words = entry.text.upper().split()
    if not words:
        return f"{{\\an2\\pos({cx},{subtitle_y})}}"

    total_cs = round((entry.end - entry.start) * 100)
    word_cs = round(total_cs / max(len(words), 1))

    parts = [f"{{\\an2\\pos({cx},{subtitle_y})}}"]
    for i, word in enumerate(words):
        escaped_word = escape_ass_text(word)
        if i < len(words) - 1:
            parts.append(f"{{\\k{word_cs}}}{escaped_word} ")
        else:
            parts.append(f"{{\\k{word_cs}}}{escaped_word}")

    return "".join(parts)


# ---------------------------------------------------------------------------
# AnimatedSubtitleRenderer
# ---------------------------------------------------------------------------

class AnimatedSubtitleRenderer:
    """Generates ASS subtitle files with animation tags and builds FFmpeg filter
    fragments for burning them into vertical shorts clips.

    Responsibilities:
    - Convert SRTEntry list to ASS format with animation override tags.
    - Support four visual styles: BUBBLE, POPUP, HIGHLIGHT, KARAOKE.
    - Position subtitles in the lower portion of the gameplay region.
    - Apply word-level timing using actual word timestamps from transcript.
    - Skip invalid entries (start >= end or negative timestamps) with a warning.
    - Escape special characters in subtitle text.
    """

    def generate_ass_file(
        self,
        srt_entries: list[SRTEntry],
        style: SubtitleStyle,
        output_path: str,
        canvas_width: int,
        canvas_height: int,
        gameplay_region_top: int,
        config,
    ) -> str:
        """Write an ASS subtitle file with animation tags for the given style.

        For each valid SRTEntry:
        - Converts start/end times to centiseconds.
        - Uppercases and escapes the text.
        - Builds a Dialogue line with style-specific ASS override tags.
        - Uses word-level timings if available for precise highlighting.

        Invalid entries (start >= end or negative timestamps) are skipped with
        a WARNING log message.

        Args:
            srt_entries:         List of SRTEntry objects to render.
            style:               SubtitleStyle enum value.
            output_path:         File path to write the .ass file.
            canvas_width:        Canvas width in pixels.
            canvas_height:       Canvas height in pixels.
            gameplay_region_top: Y coordinate of the top of the gameplay region.
            config:              Config object with subtitle settings.

        Returns:
            output_path (the path to the written .ass file).
        """
        subtitle_margin_bottom = getattr(config, "subtitle_margin_bottom", 80)

        # Subtitle anchor: bottom-centre of the canvas, margin_bottom px from bottom
        cx = canvas_width // 2
        subtitle_y = canvas_height - subtitle_margin_bottom

        header = _build_ass_header(canvas_width, canvas_height, style, config)
        dialogue_lines: list[str] = []

        for entry in srt_entries:
            # --- Task 5.7: Invalid entry filtering ---
            if entry.start < 0 or entry.end < 0:
                logger.warning(
                    "Skipping SRTEntry %d: negative timestamp (start=%.3f, end=%.3f)",
                    entry.index, entry.start, entry.end,
                )
                continue
            if entry.start >= entry.end:
                logger.warning(
                    "Skipping SRTEntry %d: start (%.3f) >= end (%.3f)",
                    entry.index, entry.start, entry.end,
                )
                continue

            start_cs = math.floor(entry.start * 100)
            end_cs = math.floor(entry.end * 100)

            start_str = _centiseconds_to_ass_time(start_cs)
            end_str = _centiseconds_to_ass_time(end_cs)

            # --- Task 5.8: Escape special characters ---
            upper_text = entry.text.upper()
            escaped_text = escape_ass_text(upper_text)

            # --- Build style-specific text with word-by-word highlighting ---
            # Use TikTok-style word-by-word highlighting with actual word timings
            text = _build_word_by_word_line(entry, cx, subtitle_y, style, config)

            dialogue_line = (
                f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{text}"
            )
            dialogue_lines.append(dialogue_line)

        # Write the ASS file
        content = header + "\n".join(dialogue_lines)
        if dialogue_lines:
            content += "\n"

        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(content)

        logger.info(
            "AnimatedSubtitleRenderer: wrote %d dialogue line(s) to %s",
            len(dialogue_lines), output_path,
        )

        return output_path

    def build_subtitle_filter(
        self,
        srt_entries: list[SRTEntry],
        style: SubtitleStyle,
        canvas_width: int,
        canvas_height: int,
        gameplay_region_top: int,
        config,
        work_dir: str,
    ) -> FilterFragment:
        """Build an FFmpeg filter fragment that burns animated subtitles into the clip.

        Generates a temporary ASS file in work_dir, then returns a FilterFragment
        referencing it via the 'ass' filter.  Falls back to the 'subtitles' filter
        if libass is not available.

        Args:
            srt_entries:         List of SRTEntry objects to render.
            style:               SubtitleStyle enum value.
            canvas_width:        Canvas width in pixels.
            canvas_height:       Canvas height in pixels.
            gameplay_region_top: Y coordinate of the top of the gameplay region.
            config:              Config object with subtitle settings.
            work_dir:            Directory in which to write the temporary ASS file.

        Returns:
            FilterFragment with input_label "[with_facecam]" and
            output_label "[final]".
        """
        os.makedirs(work_dir, exist_ok=True)

        # Generate a unique ASS file path in work_dir
        fd, ass_path = tempfile.mkstemp(suffix=".ass", dir=work_dir)
        os.close(fd)

        self.generate_ass_file(
            srt_entries=srt_entries,
            style=style,
            output_path=ass_path,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            gameplay_region_top=gameplay_region_top,
            config=config,
        )

        # Detect whether the 'ass' filter (libass) is available
        use_ass_filter = _is_ass_filter_available()

        # Escape the path for use in an FFmpeg filter string
        # Colons must be escaped on all platforms; backslashes on Windows
        escaped_ass_path = ass_path.replace("\\", "/").replace(":", "\\:")

        if use_ass_filter:
            filter_str = f"[with_facecam]ass={escaped_ass_path}[final]"
        else:
            logger.warning(
                "libass not available; falling back to subtitles filter "
                "for animated subtitle rendering"
            )
            filter_str = f"[with_facecam]subtitles={escaped_ass_path}[final]"

        return FilterFragment(
            filter_str=filter_str,
            input_label="[with_facecam]",
            output_label="[final]",
        )


# ---------------------------------------------------------------------------
# libass availability check
# ---------------------------------------------------------------------------

def _is_ass_filter_available() -> bool:
    """Return True if FFmpeg is compiled with libass (supports the 'ass' filter).

    Runs 'ffmpeg -filters' and checks for 'ass' in the output.  Returns False
    if ffmpeg is not found or the check fails for any reason.
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-filters"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = result.stdout + result.stderr
        # Look for a line containing " ass " (the filter name with surrounding spaces)
        for line in output.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1] == "ass":
                return True
        return False
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return False
