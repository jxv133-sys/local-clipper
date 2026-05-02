"""Tests for AnimatedSubtitleRenderer: unit tests (task 5.10) and property-based tests (task 5.11)."""

from __future__ import annotations

import re
import tempfile
import os
from types import SimpleNamespace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pipeline.animated_subtitle_renderer import AnimatedSubtitleRenderer, escape_ass_text
from pipeline.models import SRTEntry, SubtitleStyle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(
    subtitle_font_name: str = "Impact",
    subtitle_font_size: int = 72,
    subtitle_primary_color: str = "&H00FFFFFF",
    subtitle_outline_color: str = "&H00000000",
    subtitle_highlight_color: str = "&H0000FFFF",
    subtitle_outline_width: float = 4.0,
    subtitle_shadow_depth: float = 2.0,
    subtitle_margin_bottom: int = 80,
) -> SimpleNamespace:
    """Return a minimal config-like object for testing."""
    return SimpleNamespace(
        subtitle_font_name=subtitle_font_name,
        subtitle_font_size=subtitle_font_size,
        subtitle_primary_color=subtitle_primary_color,
        subtitle_outline_color=subtitle_outline_color,
        subtitle_highlight_color=subtitle_highlight_color,
        subtitle_outline_width=subtitle_outline_width,
        subtitle_shadow_depth=subtitle_shadow_depth,
        subtitle_margin_bottom=subtitle_margin_bottom,
    )


DEFAULT_CONFIG = make_config()
CANVAS_WIDTH = 1080
CANVAS_HEIGHT = 1920
GAMEPLAY_REGION_TOP = 672  # 35% of 1920


def make_entry(
    index: int = 1,
    start: float = 1.0,
    end: float = 3.0,
    text: str = "hello world",
) -> SRTEntry:
    """Return a valid SRTEntry for testing."""
    return SRTEntry(index=index, start=start, end=end, text=text)


def read_ass_file(path: str) -> str:
    """Read and return the content of an ASS file."""
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def get_dialogue_lines(content: str) -> list[str]:
    """Extract all Dialogue: lines from ASS file content."""
    return [line for line in content.splitlines() if line.startswith("Dialogue:")]


# ---------------------------------------------------------------------------
# Task 5.10 — Unit tests for generate_ass_file
# ---------------------------------------------------------------------------

class TestGenerateAssFileStructure:
    """Tests that generate_ass_file always produces a valid ASS file structure."""

    def setup_method(self):
        self.renderer = AnimatedSubtitleRenderer()

    def _generate(self, tmp_path, entries, style=SubtitleStyle.BUBBLE):
        output_path = str(tmp_path / "test.ass")
        self.renderer.generate_ass_file(
            srt_entries=entries,
            style=style,
            output_path=output_path,
            canvas_width=CANVAS_WIDTH,
            canvas_height=CANVAS_HEIGHT,
            gameplay_region_top=GAMEPLAY_REGION_TOP,
            config=DEFAULT_CONFIG,
        )
        return read_ass_file(output_path)

    # --- Section headers present for each style ---

    @pytest.mark.parametrize("style", list(SubtitleStyle))
    def test_script_info_section_present(self, tmp_path, style):
        content = self._generate(tmp_path, [make_entry()], style)
        assert "[Script Info]" in content

    @pytest.mark.parametrize("style", list(SubtitleStyle))
    def test_v4_styles_section_present(self, tmp_path, style):
        content = self._generate(tmp_path, [make_entry()], style)
        assert "[V4+ Styles]" in content

    @pytest.mark.parametrize("style", list(SubtitleStyle))
    def test_events_section_present(self, tmp_path, style):
        content = self._generate(tmp_path, [make_entry()], style)
        assert "[Events]" in content

    # --- At least one Dialogue line for a valid entry ---

    @pytest.mark.parametrize("style", list(SubtitleStyle))
    def test_dialogue_line_present_for_valid_entry(self, tmp_path, style):
        content = self._generate(tmp_path, [make_entry()], style)
        dialogue_lines = get_dialogue_lines(content)
        assert len(dialogue_lines) >= 1

    # --- Text is uppercased in Dialogue lines ---

    @pytest.mark.parametrize("style", list(SubtitleStyle))
    def test_text_is_uppercased(self, tmp_path, style):
        entry = make_entry(text="hello world")
        content = self._generate(tmp_path, [entry], style)
        dialogue_lines = get_dialogue_lines(content)
        assert len(dialogue_lines) == 1
        line = dialogue_lines[0]
        # KARAOKE style inserts \k tags between words, so check each word individually
        assert "HELLO" in line
        assert "WORLD" in line
        assert "hello" not in line
        assert "world" not in line


class TestGenerateAssFileBubbleStyle:
    """Tests for BUBBLE style-specific ASS tags."""

    def setup_method(self):
        self.renderer = AnimatedSubtitleRenderer()

    def test_bubble_fscx110_tag_present(self, tmp_path):
        output_path = str(tmp_path / "test.ass")
        self.renderer.generate_ass_file(
            srt_entries=[make_entry()],
            style=SubtitleStyle.BUBBLE,
            output_path=output_path,
            canvas_width=CANVAS_WIDTH,
            canvas_height=CANVAS_HEIGHT,
            gameplay_region_top=GAMEPLAY_REGION_TOP,
            config=DEFAULT_CONFIG,
        )
        content = read_ass_file(output_path)
        dialogue_lines = get_dialogue_lines(content)
        assert len(dialogue_lines) == 1
        assert r"\fscx110" in dialogue_lines[0]

    def test_bubble_fscy110_tag_present(self, tmp_path):
        output_path = str(tmp_path / "test.ass")
        self.renderer.generate_ass_file(
            srt_entries=[make_entry()],
            style=SubtitleStyle.BUBBLE,
            output_path=output_path,
            canvas_width=CANVAS_WIDTH,
            canvas_height=CANVAS_HEIGHT,
            gameplay_region_top=GAMEPLAY_REGION_TOP,
            config=DEFAULT_CONFIG,
        )
        content = read_ass_file(output_path)
        dialogue_lines = get_dialogue_lines(content)
        assert r"\fscy110" in dialogue_lines[0]

    def test_bubble_scale_animation_tag_present(self, tmp_path):
        output_path = str(tmp_path / "test.ass")
        self.renderer.generate_ass_file(
            srt_entries=[make_entry()],
            style=SubtitleStyle.BUBBLE,
            output_path=output_path,
            canvas_width=CANVAS_WIDTH,
            canvas_height=CANVAS_HEIGHT,
            gameplay_region_top=GAMEPLAY_REGION_TOP,
            config=DEFAULT_CONFIG,
        )
        content = read_ass_file(output_path)
        dialogue_lines = get_dialogue_lines(content)
        assert r"\t(0,80" in dialogue_lines[0]


class TestGenerateAssFilePopupStyle:
    """Tests for POPUP style-specific ASS tags."""

    def setup_method(self):
        self.renderer = AnimatedSubtitleRenderer()

    def test_popup_fad_tag_present(self, tmp_path):
        output_path = str(tmp_path / "test.ass")
        self.renderer.generate_ass_file(
            srt_entries=[make_entry()],
            style=SubtitleStyle.POPUP,
            output_path=output_path,
            canvas_width=CANVAS_WIDTH,
            canvas_height=CANVAS_HEIGHT,
            gameplay_region_top=GAMEPLAY_REGION_TOP,
            config=DEFAULT_CONFIG,
        )
        content = read_ass_file(output_path)
        dialogue_lines = get_dialogue_lines(content)
        assert len(dialogue_lines) == 1
        assert r"\fad(80,0)" in dialogue_lines[0]

    def test_popup_scale_zero_tags_present(self, tmp_path):
        output_path = str(tmp_path / "test.ass")
        self.renderer.generate_ass_file(
            srt_entries=[make_entry()],
            style=SubtitleStyle.POPUP,
            output_path=output_path,
            canvas_width=CANVAS_WIDTH,
            canvas_height=CANVAS_HEIGHT,
            gameplay_region_top=GAMEPLAY_REGION_TOP,
            config=DEFAULT_CONFIG,
        )
        content = read_ass_file(output_path)
        dialogue_lines = get_dialogue_lines(content)
        assert r"\fscx0\fscy0" in dialogue_lines[0]


class TestGenerateAssFileHighlightStyle:
    """Tests for HIGHLIGHT style-specific ASS tags."""

    def setup_method(self):
        self.renderer = AnimatedSubtitleRenderer()

    def test_highlight_3c_tag_present(self, tmp_path):
        output_path = str(tmp_path / "test.ass")
        self.renderer.generate_ass_file(
            srt_entries=[make_entry()],
            style=SubtitleStyle.HIGHLIGHT,
            output_path=output_path,
            canvas_width=CANVAS_WIDTH,
            canvas_height=CANVAS_HEIGHT,
            gameplay_region_top=GAMEPLAY_REGION_TOP,
            config=DEFAULT_CONFIG,
        )
        content = read_ass_file(output_path)
        dialogue_lines = get_dialogue_lines(content)
        assert len(dialogue_lines) == 1
        assert r"\3c" in dialogue_lines[0]

    def test_highlight_bord6_tag_present(self, tmp_path):
        output_path = str(tmp_path / "test.ass")
        self.renderer.generate_ass_file(
            srt_entries=[make_entry()],
            style=SubtitleStyle.HIGHLIGHT,
            output_path=output_path,
            canvas_width=CANVAS_WIDTH,
            canvas_height=CANVAS_HEIGHT,
            gameplay_region_top=GAMEPLAY_REGION_TOP,
            config=DEFAULT_CONFIG,
        )
        content = read_ass_file(output_path)
        dialogue_lines = get_dialogue_lines(content)
        assert r"\bord6" in dialogue_lines[0]


class TestGenerateAssFileKaraokeStyle:
    """Tests for KARAOKE style-specific ASS tags."""

    def setup_method(self):
        self.renderer = AnimatedSubtitleRenderer()

    def test_karaoke_k_tags_present(self, tmp_path):
        output_path = str(tmp_path / "test.ass")
        self.renderer.generate_ass_file(
            srt_entries=[make_entry(text="one two three")],
            style=SubtitleStyle.KARAOKE,
            output_path=output_path,
            canvas_width=CANVAS_WIDTH,
            canvas_height=CANVAS_HEIGHT,
            gameplay_region_top=GAMEPLAY_REGION_TOP,
            config=DEFAULT_CONFIG,
        )
        content = read_ass_file(output_path)
        dialogue_lines = get_dialogue_lines(content)
        assert len(dialogue_lines) == 1
        # \k tags should be present for karaoke style
        assert r"\k" in dialogue_lines[0]


class TestGenerateAssFileEmptyEntries:
    """Tests for generate_ass_file with an empty entry list."""

    def setup_method(self):
        self.renderer = AnimatedSubtitleRenderer()

    def test_empty_entries_produces_valid_ass_file(self, tmp_path):
        output_path = str(tmp_path / "test.ass")
        self.renderer.generate_ass_file(
            srt_entries=[],
            style=SubtitleStyle.BUBBLE,
            output_path=output_path,
            canvas_width=CANVAS_WIDTH,
            canvas_height=CANVAS_HEIGHT,
            gameplay_region_top=GAMEPLAY_REGION_TOP,
            config=DEFAULT_CONFIG,
        )
        content = read_ass_file(output_path)
        assert "[Script Info]" in content
        assert "[V4+ Styles]" in content
        assert "[Events]" in content

    def test_empty_entries_produces_no_dialogue_lines(self, tmp_path):
        output_path = str(tmp_path / "test.ass")
        self.renderer.generate_ass_file(
            srt_entries=[],
            style=SubtitleStyle.BUBBLE,
            output_path=output_path,
            canvas_width=CANVAS_WIDTH,
            canvas_height=CANVAS_HEIGHT,
            gameplay_region_top=GAMEPLAY_REGION_TOP,
            config=DEFAULT_CONFIG,
        )
        content = read_ass_file(output_path)
        dialogue_lines = get_dialogue_lines(content)
        assert len(dialogue_lines) == 0


class TestGenerateAssFileInvalidTimingEntries:
    """Tests for generate_ass_file with invalid timing entries."""

    def setup_method(self):
        self.renderer = AnimatedSubtitleRenderer()

    def _generate(self, tmp_path, entries):
        output_path = str(tmp_path / "test.ass")
        self.renderer.generate_ass_file(
            srt_entries=entries,
            style=SubtitleStyle.BUBBLE,
            output_path=output_path,
            canvas_width=CANVAS_WIDTH,
            canvas_height=CANVAS_HEIGHT,
            gameplay_region_top=GAMEPLAY_REGION_TOP,
            config=DEFAULT_CONFIG,
        )
        return read_ass_file(output_path)

    def test_start_equals_end_is_skipped(self, tmp_path):
        """Entry with start == end should produce no Dialogue line."""
        entry = SRTEntry(index=1, start=2.0, end=2.0, text="bad entry")
        content = self._generate(tmp_path, [entry])
        dialogue_lines = get_dialogue_lines(content)
        assert len(dialogue_lines) == 0

    def test_start_greater_than_end_is_skipped(self, tmp_path):
        """Entry with start > end should produce no Dialogue line."""
        entry = SRTEntry(index=1, start=5.0, end=2.0, text="bad entry")
        content = self._generate(tmp_path, [entry])
        dialogue_lines = get_dialogue_lines(content)
        assert len(dialogue_lines) == 0

    def test_negative_start_is_skipped(self, tmp_path):
        """Entry with negative start should produce no Dialogue line."""
        entry = SRTEntry(index=1, start=-1.0, end=2.0, text="bad entry")
        content = self._generate(tmp_path, [entry])
        dialogue_lines = get_dialogue_lines(content)
        assert len(dialogue_lines) == 0

    def test_negative_end_is_skipped(self, tmp_path):
        """Entry with negative end should produce no Dialogue line."""
        entry = SRTEntry(index=1, start=1.0, end=-1.0, text="bad entry")
        content = self._generate(tmp_path, [entry])
        dialogue_lines = get_dialogue_lines(content)
        assert len(dialogue_lines) == 0

    def test_valid_mixed_with_invalid_only_valid_appears(self, tmp_path):
        """Only valid entries should appear in output when mixed with invalid ones."""
        entries = [
            SRTEntry(index=1, start=1.0, end=3.0, text="valid one"),
            SRTEntry(index=2, start=5.0, end=5.0, text="invalid equal"),
            SRTEntry(index=3, start=4.0, end=6.0, text="valid two"),
            SRTEntry(index=4, start=-1.0, end=2.0, text="invalid negative start"),
            SRTEntry(index=5, start=7.0, end=9.0, text="valid three"),
            SRTEntry(index=6, start=10.0, end=8.0, text="invalid start gt end"),
        ]
        content = self._generate(tmp_path, entries)
        dialogue_lines = get_dialogue_lines(content)
        # Only 3 valid entries should appear
        assert len(dialogue_lines) == 3
        # Valid entries' text should be present (uppercased)
        full_content = "\n".join(dialogue_lines)
        assert "VALID ONE" in full_content
        assert "VALID TWO" in full_content
        assert "VALID THREE" in full_content
        # Invalid entries' text should NOT be present
        assert "INVALID EQUAL" not in full_content
        assert "INVALID NEGATIVE START" not in full_content
        assert "INVALID START GT END" not in full_content


# ---------------------------------------------------------------------------
# Task 5.11 — Property-based tests (hypothesis)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Generate arbitrary SRTEntry objects (may be valid or invalid)
@st.composite
def srt_entry_strategy(draw, index: int = 1):
    """Generate an arbitrary SRTEntry (may have invalid timing).

    When the entry would be 'valid' (start >= 0, end >= 0, start < end),
    we use integer centiseconds to ensure floor(end*100) > floor(start*100),
    avoiding floating-point edge cases.
    Invalid entries (negative timestamps or start >= end) are generated freely.
    """
    kind = draw(st.sampled_from(["valid", "invalid_neg_start", "invalid_neg_end", "invalid_start_ge_end"]))
    if kind == "valid":
        # Use integer seconds to avoid floating-point precision issues
        # floor(n * 100) = n * 100 exactly for integer n, so timestamps are always distinct
        start = draw(st.integers(min_value=0, max_value=99))
        end = draw(st.integers(min_value=start + 1, max_value=start + 60))
        start = float(start)
        end = float(end)
    elif kind == "invalid_neg_start":
        start = draw(st.floats(min_value=-5.0, max_value=-0.001, allow_nan=False, allow_infinity=False))
        end = draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False))
    elif kind == "invalid_neg_end":
        start = draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False))
        end = draw(st.floats(min_value=-5.0, max_value=-0.001, allow_nan=False, allow_infinity=False))
    else:  # invalid_start_ge_end
        end = draw(st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False))
        start = draw(st.floats(min_value=end, max_value=end + 50.0, allow_nan=False, allow_infinity=False))
    text = draw(st.text(min_size=0, max_size=100))
    return SRTEntry(index=index, start=start, end=end, text=text)


@st.composite
def valid_srt_entry_strategy(draw, index: int = 1):
    """Generate a valid SRTEntry (start >= 0, start < end, with at least 1 second gap).
    
    Uses integer seconds to avoid floating-point precision issues with floor().
    The renderer uses floor(x * 100) to convert to centiseconds, so using integer
    seconds guarantees floor(start * 100) < floor(end * 100).
    """
    start = draw(st.integers(min_value=0, max_value=99))
    end = draw(st.integers(min_value=start + 1, max_value=start + 60))
    text = draw(st.text(min_size=1, max_size=80))
    return SRTEntry(index=index, start=float(start), end=float(end), text=text)


@st.composite
def invalid_srt_entry_strategy(draw, index: int = 1):
    """Generate an invalid SRTEntry (start >= end or negative timestamps).
    
    All generated entries are guaranteed to be invalid per the renderer's filter:
    - neg_start: start < 0
    - neg_end: end < 0
    - start_eq_end: start == end (both >= 0)
    - start_gt_end: start > end (both >= 0)
    """
    kind = draw(st.sampled_from(["start_eq_end", "start_gt_end", "neg_start", "neg_end"]))
    if kind == "start_eq_end":
        t = draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False))
        start, end = t, t
    elif kind == "start_gt_end":
        end = draw(st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False))
        # Use a fixed offset to guarantee start > end regardless of float precision
        start = end + draw(st.floats(min_value=0.1, max_value=50.0, allow_nan=False, allow_infinity=False))
    elif kind == "neg_start":
        start = draw(st.floats(min_value=-100.0, max_value=-0.001, allow_nan=False, allow_infinity=False))
        end = draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False))
    else:  # neg_end
        start = draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False))
        end = draw(st.floats(min_value=-100.0, max_value=-0.001, allow_nan=False, allow_infinity=False))
    text = draw(st.text(min_size=1, max_size=80))
    return SRTEntry(index=index, start=start, end=end, text=text)


def _is_valid_entry(entry: SRTEntry) -> bool:
    """Return True if the entry has valid timing matching the renderer's filter:
    start >= 0, end >= 0, start < end."""
    return entry.start >= 0 and entry.end >= 0 and entry.start < entry.end


def _parse_ass_time(time_str: str) -> float:
    """Parse an ASS time string 'H:MM:SS.cc' to seconds."""
    # Format: H:MM:SS.cc
    h, rest = time_str.split(":", 1)
    m, rest2 = rest.split(":", 1)
    s, cs = rest2.split(".", 1)
    return int(h) * 3600 + int(m) * 60 + int(s) + int(cs) / 100.0


def _extract_dialogue_times(dialogue_line: str) -> tuple[float, float]:
    """Extract start and end times from a Dialogue line."""
    # Format: Dialogue: 0,H:MM:SS.cc,H:MM:SS.cc,Default,,0,0,0,,text
    parts = dialogue_line.split(",")
    start_str = parts[1]
    end_str = parts[2]
    return _parse_ass_time(start_str), _parse_ass_time(end_str)


# ---------------------------------------------------------------------------
# Property 10: ASS file always has valid structure
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(entries=st.lists(srt_entry_strategy(), min_size=0, max_size=20))
def test_property_10_ass_file_always_has_valid_structure(entries):
    """
    Property 10: For any list of SRTEntry objects (including empty),
    generate_ass_file output contains "[Script Info]", "[V4+ Styles]", "[Events]",
    and all Dialogue lines have start_time < end_time.

    Validates: Requirements 13.1, 13.2, 5.1
    """
    renderer = AnimatedSubtitleRenderer()
    with tempfile.NamedTemporaryFile(suffix=".ass", delete=False) as fh:
        output_path = fh.name
    try:
        renderer.generate_ass_file(
            srt_entries=entries,
            style=SubtitleStyle.BUBBLE,
            output_path=output_path,
            canvas_width=CANVAS_WIDTH,
            canvas_height=CANVAS_HEIGHT,
            gameplay_region_top=GAMEPLAY_REGION_TOP,
            config=DEFAULT_CONFIG,
        )
        content = read_ass_file(output_path)
    finally:
        os.unlink(output_path)

    # All three section headers must be present
    assert "[Script Info]" in content, "Missing [Script Info] section"
    assert "[V4+ Styles]" in content, "Missing [V4+ Styles] section"
    assert "[Events]" in content, "Missing [Events] section"

    # Every Dialogue line must have start_time < end_time
    # (only entries with >= 0.01s gap produce distinct ASS centisecond timestamps)
    for line in get_dialogue_lines(content):
        start_t, end_t = _extract_dialogue_times(line)
        assert start_t < end_t, (
            f"Dialogue line has start_time ({start_t}) >= end_time ({end_t}): {line!r}"
        )
# ---------------------------------------------------------------------------
# Property 11: Invalid entries are excluded from ASS output
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    valid_entries=st.lists(valid_srt_entry_strategy(), min_size=0, max_size=10),
    invalid_entries=st.lists(invalid_srt_entry_strategy(), min_size=1, max_size=10),
)
def test_property_11_invalid_entries_excluded(valid_entries, invalid_entries):
    """
    Property 11: For any list of SRTEntry objects that contains entries with
    start >= end or negative timestamps mixed with valid entries, generate_ass_file
    produces an ASS file whose Dialogue lines correspond only to the valid entries.

    Validates: Requirements 5.10, 11.4
    """
    # Re-index all entries to avoid duplicate indices
    all_entries = []
    for i, e in enumerate(valid_entries + invalid_entries, start=1):
        all_entries.append(SRTEntry(index=i, start=e.start, end=e.end, text=e.text))

    renderer = AnimatedSubtitleRenderer()
    with tempfile.NamedTemporaryFile(suffix=".ass", delete=False) as fh:
        output_path = fh.name
    try:
        renderer.generate_ass_file(
            srt_entries=all_entries,
            style=SubtitleStyle.BUBBLE,
            output_path=output_path,
            canvas_width=CANVAS_WIDTH,
            canvas_height=CANVAS_HEIGHT,
            gameplay_region_top=GAMEPLAY_REGION_TOP,
            config=DEFAULT_CONFIG,
        )
        content = read_ass_file(output_path)
    finally:
        os.unlink(output_path)

    dialogue_lines = get_dialogue_lines(content)

    # Count valid entries in the combined list
    valid_count = sum(1 for e in all_entries if _is_valid_entry(e))

    assert len(dialogue_lines) == valid_count, (
        f"Expected {valid_count} Dialogue lines (valid entries only), "
        f"got {len(dialogue_lines)}"
    )

    # All Dialogue lines must have valid timing
    for line in dialogue_lines:
        start_t, end_t = _extract_dialogue_times(line)
        assert start_t < end_t, (
            f"Dialogue line has start_time ({start_t}) >= end_time ({end_t}): {line!r}"
        )


# ---------------------------------------------------------------------------
# Property 12: Subtitle text is always uppercased in ASS output
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    text=st.text(
        alphabet=st.characters(
            # Exclude characters that would be escaped in ASS (backslash, braces)
            # to keep the test focused on uppercasing rather than escaping
            blacklist_characters="\\{}",
            min_codepoint=32,
            max_codepoint=127,
        ),
        min_size=1,
        max_size=50,
    ),
    style=st.sampled_from(list(SubtitleStyle)),
)
def test_property_12_text_always_uppercased(text, style):
    """
    Property 12: For any SRTEntry with arbitrary text and any SubtitleStyle,
    the corresponding Dialogue line in the generated ASS file contains the
    uppercase version of the entry's text (after escaping).

    Validates: Requirements 5.8
    """
    entry = SRTEntry(index=1, start=1.0, end=3.0, text=text)
    renderer = AnimatedSubtitleRenderer()
    with tempfile.NamedTemporaryFile(suffix=".ass", delete=False) as fh:
        output_path = fh.name
    try:
        renderer.generate_ass_file(
            srt_entries=[entry],
            style=style,
            output_path=output_path,
            canvas_width=CANVAS_WIDTH,
            canvas_height=CANVAS_HEIGHT,
            gameplay_region_top=GAMEPLAY_REGION_TOP,
            config=DEFAULT_CONFIG,
        )
        content = read_ass_file(output_path)
    finally:
        os.unlink(output_path)

    dialogue_lines = get_dialogue_lines(content)

    assert len(dialogue_lines) == 1, (
        f"Expected 1 Dialogue line for valid entry, got {len(dialogue_lines)}"
    )

    # The uppercased, escaped text should appear in the Dialogue line
    # For KARAOKE style, text is split into words and reassembled with \k tags,
    # so whitespace may be normalized. Check that each word appears in the line.
    upper_text = text.upper()
    if style == SubtitleStyle.KARAOKE:
        # KARAOKE splits by words, so check each word appears
        words = upper_text.split()
        if words:
            for word in words:
                escaped_word = escape_ass_text(word)
                assert escaped_word in dialogue_lines[0], (
                    f"Expected word {escaped_word!r} in KARAOKE Dialogue line: {dialogue_lines[0]!r}"
                )
        else:
            # Empty text after split — just check the line exists (already asserted above)
            pass
    else:
        expected_text = escape_ass_text(upper_text)
        assert expected_text in dialogue_lines[0], (
            f"Expected uppercased text {expected_text!r} in Dialogue line: {dialogue_lines[0]!r}"
        )


# ---------------------------------------------------------------------------
# Property 13: Subtitle positions are within canvas bounds
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    canvas_width=st.integers(min_value=100, max_value=3840),
    canvas_height=st.integers(min_value=100, max_value=7680),
    subtitle_margin_bottom=st.integers(min_value=0, max_value=99),
)
def test_property_13_subtitle_positions_within_canvas_bounds(
    canvas_width, canvas_height, subtitle_margin_bottom
):
    """
    Property 13: For any canvas dimensions (canvas_width, canvas_height) and
    subtitle_margin_bottom, the subtitle position in the Dialogue line is within
    canvas bounds: 0 <= x <= canvas_width and 0 <= y <= canvas_height.

    Validates: Requirements 5.7, 13.4
    """
    config = make_config(subtitle_margin_bottom=subtitle_margin_bottom)
    entry = SRTEntry(index=1, start=1.0, end=3.0, text="test subtitle")
    renderer = AnimatedSubtitleRenderer()
    with tempfile.NamedTemporaryFile(suffix=".ass", delete=False) as fh:
        output_path = fh.name
    try:
        renderer.generate_ass_file(
            srt_entries=[entry],
            style=SubtitleStyle.BUBBLE,
            output_path=output_path,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            gameplay_region_top=0,
            config=config,
        )
        content = read_ass_file(output_path)
    finally:
        os.unlink(output_path)

    dialogue_lines = get_dialogue_lines(content)

    assert len(dialogue_lines) == 1, (
        f"Expected 1 Dialogue line, got {len(dialogue_lines)}"
    )

    # Extract \pos(x,y) from the Dialogue line
    match = re.search(r"\\pos\((\d+),(\d+)\)", dialogue_lines[0])
    assert match is not None, (
        f"Could not find \\pos(x,y) in Dialogue line: {dialogue_lines[0]!r}"
    )
    pos_x = int(match.group(1))
    pos_y = int(match.group(2))

    assert 0 <= pos_x <= canvas_width, (
        f"pos_x={pos_x} is outside [0, {canvas_width}]"
    )
    assert 0 <= pos_y <= canvas_height, (
        f"pos_y={pos_y} is outside [0, {canvas_height}]"
    )
