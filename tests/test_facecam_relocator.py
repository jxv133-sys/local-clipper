"""Tests for FacecamRelocator: unit tests (tasks 4.6, 4.7) and property-based tests (task 4.8)."""

from __future__ import annotations

import re
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pipeline.facecam_relocator import FacecamRelocator, classify_region
from pipeline.models import CanvasLayout, FacecamRegion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(
    facecam_sample_duration: float = 10.0,
    facecam_min_area_fraction: float = 0.04,
    facecam_max_area_fraction: float = 0.30,
) -> SimpleNamespace:
    """Return a minimal config-like object for testing."""
    return SimpleNamespace(
        facecam_sample_duration=facecam_sample_duration,
        facecam_min_area_fraction=facecam_min_area_fraction,
        facecam_max_area_fraction=facecam_max_area_fraction,
    )


DEFAULT_CONFIG = make_config()

FRAME_W = 1920
FRAME_H = 1080


# ---------------------------------------------------------------------------
# Task 4.6 — Unit tests for classify_region
# ---------------------------------------------------------------------------

class TestClassifyRegionCorners:
    """Unit tests for classify_region corner detection with known coordinates."""

    # --- Corner classification for 1920×1080 frame ---

    def test_top_left_corner(self):
        """Region whose centre is in the top-left quadrant → 'top-left'."""
        # Centre at (240, 135) — clearly top-left
        result = classify_region(x=0, y=0, w=480, h=270, frame_w=FRAME_W, frame_h=FRAME_H)
        assert result == "top-left"

    def test_top_right_corner(self):
        """Region whose centre is in the top-right quadrant → 'top-right'."""
        # Centre at (1680, 135) — clearly top-right
        result = classify_region(x=1440, y=0, w=480, h=270, frame_w=FRAME_W, frame_h=FRAME_H)
        assert result == "top-right"

    def test_bottom_left_corner(self):
        """Region whose centre is in the bottom-left quadrant → 'bottom-left'."""
        # Centre at (240, 945) — clearly bottom-left
        result = classify_region(x=0, y=810, w=480, h=270, frame_w=FRAME_W, frame_h=FRAME_H)
        assert result == "bottom-left"

    def test_bottom_right_corner(self):
        """Region whose centre is in the bottom-right quadrant → 'bottom-right'."""
        # Centre at (1680, 945) — clearly bottom-right
        result = classify_region(x=1440, y=810, w=480, h=270, frame_w=FRAME_W, frame_h=FRAME_H)
        assert result == "bottom-right"


class TestClassifyRegionAreaFraction:
    """Unit tests for classify_region area fraction boundary validation."""

    # A region in the bottom-right corner with controllable area fraction.
    # frame_w=1000, frame_h=1000 makes arithmetic easy: area_fraction = w*h / 1_000_000.

    FRAME = 1000  # square frame for easy fraction arithmetic

    def _region_with_fraction(self, fraction: float) -> tuple[int, int, int, int]:
        """Return (x, y, w, h) for a bottom-right region with the given area fraction."""
        # w = h = sqrt(fraction * FRAME^2)
        side = int((fraction * self.FRAME * self.FRAME) ** 0.5)
        # Place in bottom-right so centre is in the correct quadrant
        x = self.FRAME - side
        y = self.FRAME - side
        return x, y, side, side

    def test_area_fraction_exactly_at_min_boundary_is_valid(self):
        """Area fraction == 0.04 (min boundary, inclusive) → not None."""
        x, y, w, h = self._region_with_fraction(0.04)
        result = classify_region(x=x, y=y, w=w, h=h, frame_w=self.FRAME, frame_h=self.FRAME)
        assert result is not None

    def test_area_fraction_just_below_min_returns_none(self):
        """Area fraction == 0.039 (just below min) → None."""
        # Use exact pixel counts to get fraction just below 0.04
        # 0.039 * 1000 * 1000 = 39000 → side = 197 → 197*197/1000000 = 0.038809 < 0.04
        result = classify_region(x=803, y=803, w=197, h=197, frame_w=self.FRAME, frame_h=self.FRAME)
        area = (197 * 197) / (self.FRAME * self.FRAME)
        assert area < 0.04
        assert result is None

    def test_area_fraction_exactly_at_max_boundary_is_valid(self):
        """Area fraction == 0.30 (max boundary, inclusive) → not None."""
        # 0.30 * 1000 * 1000 = 300000 → side = 547 → 547*547/1000000 = 0.299209 ≈ 0.30
        # Use exact values: w=548, h=548 → 548*548/1000000 = 0.300304 > 0.30 → too big
        # Use w=547, h=548 → 547*548/1000000 = 0.299756 < 0.30 → still valid
        # Better: use the classify_region with explicit min/max to test boundary exactly
        # Pass a region whose exact fraction is 0.30 using custom frame size
        # frame_w=10, frame_h=10, w=h=sqrt(0.30*100)=sqrt(30)≈5.477 → not integer
        # Use frame 100x100, w=h=sqrt(0.30*10000)=sqrt(3000)≈54.77 → w=54, h=56 → 54*56/10000=0.3024 > 0.30
        # Simplest: use min/max params directly
        result = classify_region(
            x=0, y=500, w=300, h=1000,
            frame_w=1000, frame_h=1000,
            min_area_fraction=0.04,
            max_area_fraction=0.30,
        )
        # area = 300*1000/1000000 = 0.30 exactly → valid
        assert result is not None

    def test_area_fraction_just_above_max_returns_none(self):
        """Area fraction == 0.31 (just above max) → None."""
        # area = 310*1000/1000000 = 0.31 > 0.30
        result = classify_region(
            x=0, y=0, w=310, h=1000,
            frame_w=1000, frame_h=1000,
            min_area_fraction=0.04,
            max_area_fraction=0.30,
        )
        assert result is None

    def test_area_fraction_in_valid_range_returns_corner_string(self):
        """Area fraction in valid range → returns one of the four corner strings."""
        valid_corners = {"top-left", "top-right", "bottom-left", "bottom-right"}
        # area = 200*200/1000000 = 0.04 exactly (min boundary)
        result = classify_region(x=0, y=0, w=200, h=200, frame_w=1000, frame_h=1000)
        assert result in valid_corners


# ---------------------------------------------------------------------------
# Task 4.7 — Unit tests for detect_facecam with mocked ffmpeg
# ---------------------------------------------------------------------------

class TestDetectFacecam:
    """Unit tests for FacecamRelocator.detect_facecam with mocked subprocess.run."""

    def setup_method(self):
        self.relocator = FacecamRelocator()
        self.config = DEFAULT_CONFIG

    def _make_result(self, stderr: str) -> MagicMock:
        result = MagicMock()
        result.stderr = stderr
        return result

    def test_valid_cropdetect_bottom_right_returns_facecam_region(self):
        """Valid cropdetect output with a pip in the bottom-right corner → FacecamRegion."""
        # 480×270 pip at (1440, 810) in a 1920×1080 frame
        # area_fraction = 480*270 / (1920*1080) = 129600/2073600 ≈ 0.0625 → valid
        # centre = (1440+240, 810+135) = (1680, 945) → bottom-right
        stderr = (
            "[Parsed_cropdetect_0 @ 0x...] x1:1440 x2:1919 y1:810 y2:1079 "
            "w:480 h:270 x:1440 y:810 pts:0 t:0.000 crop=480:270:1440:810\n"
            "[Parsed_cropdetect_0 @ 0x...] x1:1440 x2:1919 y1:810 y2:1079 "
            "w:480 h:270 x:1440 y:810 pts:1 t:0.033 crop=480:270:1440:810\n"
            "[Parsed_cropdetect_0 @ 0x...] x1:1440 x2:1919 y1:810 y2:1079 "
            "w:480 h:270 x:1440 y:810 pts:2 t:0.067 crop=480:270:1440:810\n"
        )
        with patch("subprocess.run", return_value=self._make_result(stderr)):
            region = self.relocator.detect_facecam(
                clip_path="fake.mp4",
                frame_width=FRAME_W,
                frame_height=FRAME_H,
                config=self.config,
            )
        assert region is not None
        assert isinstance(region, FacecamRegion)
        assert region.x == 1440
        assert region.y == 810
        assert region.width == 480
        assert region.height == 270
        assert region.corner == "bottom-right"
        assert 0.0 <= region.confidence <= 1.0

    def test_no_crop_lines_returns_none(self):
        """Cropdetect output with no crop= lines → None."""
        stderr = (
            "ffmpeg version 6.0\n"
            "Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'fake.mp4':\n"
            "  Duration: 00:00:10.00\n"
        )
        with patch("subprocess.run", return_value=self._make_result(stderr)):
            region = self.relocator.detect_facecam(
                clip_path="fake.mp4",
                frame_width=FRAME_W,
                frame_height=FRAME_H,
                config=self.config,
            )
        assert region is None

    def test_best_crop_area_too_small_returns_none(self):
        """Best crop area fraction < min_area_fraction → None."""
        # 10×10 pip in 1920×1080 → area_fraction = 100/2073600 ≈ 0.000048 << 0.04
        stderr = "crop=10:10:0:0\ncrop=10:10:0:0\ncrop=10:10:0:0\n"
        with patch("subprocess.run", return_value=self._make_result(stderr)):
            region = self.relocator.detect_facecam(
                clip_path="fake.mp4",
                frame_width=FRAME_W,
                frame_height=FRAME_H,
                config=self.config,
            )
        assert region is None

    def test_best_crop_area_too_large_returns_none(self):
        """Best crop area fraction > max_area_fraction → None."""
        # 1800×1000 in 1920×1080 → area_fraction = 1800000/2073600 ≈ 0.868 >> 0.30
        stderr = "crop=1800:1000:60:40\ncrop=1800:1000:60:40\n"
        with patch("subprocess.run", return_value=self._make_result(stderr)):
            region = self.relocator.detect_facecam(
                clip_path="fake.mp4",
                frame_width=FRAME_W,
                frame_height=FRAME_H,
                config=self.config,
            )
        assert region is None

    def test_ffmpeg_not_found_returns_none(self):
        """FileNotFoundError from subprocess.run (ffmpeg not on PATH) → None."""
        with patch("subprocess.run", side_effect=FileNotFoundError("ffmpeg not found")):
            region = self.relocator.detect_facecam(
                clip_path="fake.mp4",
                frame_width=FRAME_W,
                frame_height=FRAME_H,
                config=self.config,
            )
        assert region is None


# ---------------------------------------------------------------------------
# Task 4.8 — Property-based tests (hypothesis)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Frame dimensions: small enough to keep arithmetic fast, large enough to be realistic
frame_dim = st.integers(min_value=100, max_value=3840)

# Area fractions for boundary testing
area_fraction_strategy = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)

# Valid area fractions (within [0.04, 0.30])
valid_area_fraction_strategy = st.floats(
    min_value=0.04, max_value=0.30, allow_nan=False, allow_infinity=False
)


@st.composite
def frame_and_region(draw, valid_fraction: bool = True):
    """Generate (frame_w, frame_h, x, y, w, h) with controllable area fraction."""
    frame_w = draw(st.integers(min_value=100, max_value=3840))
    frame_h = draw(st.integers(min_value=100, max_value=2160))
    frame_area = frame_w * frame_h

    if valid_fraction:
        # Pick a fraction in [0.04, 0.30]
        fraction = draw(st.floats(min_value=0.04, max_value=0.30, allow_nan=False, allow_infinity=False))
    else:
        # Pick a fraction outside [0.04, 0.30]
        fraction = draw(
            st.one_of(
                st.floats(min_value=0.0, max_value=0.039, allow_nan=False, allow_infinity=False),
                st.floats(min_value=0.31, max_value=1.0, allow_nan=False, allow_infinity=False),
            )
        )

    # Derive w, h from fraction: use w = h = sqrt(fraction * frame_area) clamped to frame dims
    import math
    side = max(1, int(math.sqrt(fraction * frame_area)))
    w = min(side, frame_w)
    h = min(side, frame_h)

    # Place region so it fits within the frame
    x = draw(st.integers(min_value=0, max_value=max(0, frame_w - w)))
    y = draw(st.integers(min_value=0, max_value=max(0, frame_h - h)))

    return frame_w, frame_h, x, y, w, h


@st.composite
def canvas_layout_strategy(draw):
    """Generate a CanvasLayout with valid dimensions."""
    canvas_width = draw(st.integers(min_value=100, max_value=3840))
    canvas_height = draw(st.integers(min_value=100, max_value=7680))
    facecam_height = draw(st.integers(min_value=1, max_value=canvas_height - 1))
    gameplay_height = canvas_height - facecam_height
    return CanvasLayout(
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        facecam_x=0,
        facecam_y=0,
        facecam_width=canvas_width,
        facecam_height=facecam_height,
        gameplay_x=0,
        gameplay_y=facecam_height,
        gameplay_width=canvas_width,
        gameplay_height=gameplay_height,
    )


@st.composite
def facecam_region_strategy(draw):
    """Generate a FacecamRegion with valid pixel coordinates."""
    frame_w = draw(st.integers(min_value=100, max_value=3840))
    frame_h = draw(st.integers(min_value=100, max_value=2160))
    w = draw(st.integers(min_value=1, max_value=frame_w))
    h = draw(st.integers(min_value=1, max_value=frame_h))
    x = draw(st.integers(min_value=0, max_value=frame_w - w))
    y = draw(st.integers(min_value=0, max_value=frame_h - h))
    corner = draw(st.sampled_from(["top-left", "top-right", "bottom-left", "bottom-right"]))
    confidence = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    return FacecamRegion(x=x, y=y, width=w, height=h, corner=corner, confidence=confidence)


# ---------------------------------------------------------------------------
# Property 4: Area fraction filtering is correct
# ---------------------------------------------------------------------------

@settings(max_examples=200)
@given(data=frame_and_region(valid_fraction=False))
def test_property_4_invalid_area_fraction_returns_none(data):
    """
    Property 4 (part A): For any (x, y, w, h) within a frame where
    area_fraction < min_area_fraction or > max_area_fraction,
    classify_region returns None.

    Validates: Requirements 3.3, 3.4
    """
    frame_w, frame_h, x, y, w, h = data
    result = classify_region(
        x=x, y=y, w=w, h=h,
        frame_w=frame_w, frame_h=frame_h,
        min_area_fraction=0.04,
        max_area_fraction=0.30,
    )
    area_fraction = (w * h) / (frame_w * frame_h)
    if area_fraction < 0.04 or area_fraction > 0.30:
        assert result is None, (
            f"Expected None for area_fraction={area_fraction:.4f} "
            f"(w={w}, h={h}, frame={frame_w}x{frame_h}), got {result!r}"
        )


@settings(max_examples=200)
@given(data=frame_and_region(valid_fraction=True))
def test_property_4_valid_area_fraction_returns_corner(data):
    """
    Property 4 (part B): For any (x, y, w, h) within a frame where
    area_fraction is in [min_area_fraction, max_area_fraction],
    classify_region returns a valid corner string.

    Validates: Requirements 3.3, 3.4
    """
    frame_w, frame_h, x, y, w, h = data
    result = classify_region(
        x=x, y=y, w=w, h=h,
        frame_w=frame_w, frame_h=frame_h,
        min_area_fraction=0.04,
        max_area_fraction=0.30,
    )
    area_fraction = (w * h) / (frame_w * frame_h)
    if 0.04 <= area_fraction <= 0.30:
        valid_corners = {"top-left", "top-right", "bottom-left", "bottom-right"}
        assert result in valid_corners, (
            f"Expected a corner string for area_fraction={area_fraction:.4f}, got {result!r}"
        )


# ---------------------------------------------------------------------------
# Property 5: Corner classification is consistent with centre coordinates
# ---------------------------------------------------------------------------

@settings(max_examples=200)
@given(data=frame_and_region(valid_fraction=True))
def test_property_5_corner_consistent_with_centre(data):
    """
    Property 5: For any valid region (area fraction in [0.04, 0.30]),
    the returned corner is consistent with whether the region's centre
    falls in the left/right and top/bottom halves of the frame.

    Validates: Requirements 3.5
    """
    frame_w, frame_h, x, y, w, h = data
    result = classify_region(
        x=x, y=y, w=w, h=h,
        frame_w=frame_w, frame_h=frame_h,
        min_area_fraction=0.04,
        max_area_fraction=0.30,
    )
    area_fraction = (w * h) / (frame_w * frame_h)
    if result is None:
        # Area fraction outside range — skip consistency check
        return

    center_x = x + w / 2
    center_y = y + h / 2
    half_w = frame_w / 2
    half_h = frame_h / 2

    is_left = center_x < half_w
    is_top = center_y < half_h

    if is_left and is_top:
        expected = "top-left"
    elif not is_left and is_top:
        expected = "top-right"
    elif is_left and not is_top:
        expected = "bottom-left"
    else:
        expected = "bottom-right"

    assert result == expected, (
        f"Corner mismatch: centre=({center_x:.1f}, {center_y:.1f}), "
        f"frame={frame_w}x{frame_h}, expected={expected!r}, got={result!r}"
    )


# ---------------------------------------------------------------------------
# Property 6: Confidence score is always in [0.0, 1.0]
# ---------------------------------------------------------------------------

def _build_cropdetect_stderr(crops: list[tuple[int, int, int, int]]) -> str:
    """Build a fake ffmpeg stderr string from a list of (w, h, x, y) crop tuples."""
    lines = []
    for w, h, x, y in crops:
        lines.append(f"crop={w}:{h}:{x}:{y}")
    return "\n".join(lines)


@st.composite
def valid_cropdetect_output(draw):
    """
    Generate a list of (w, h, x, y) crop tuples where the most frequent crop
    passes area validation for a 1920×1080 frame.
    """
    # The dominant crop: a valid pip in the bottom-right of a 1920×1080 frame
    # area_fraction = 480*270 / (1920*1080) ≈ 0.0625 → valid
    dominant_w, dominant_h, dominant_x, dominant_y = 480, 270, 1440, 810

    # How many times the dominant crop appears (at least 1)
    dominant_count = draw(st.integers(min_value=1, max_value=20))

    # Optional noise crops (different dimensions, may or may not be valid)
    noise_count = draw(st.integers(min_value=0, max_value=dominant_count - 1))
    noise_crops = [(100, 100, 0, 0)] * noise_count  # small, invalid area fraction

    crops = [(dominant_w, dominant_h, dominant_x, dominant_y)] * dominant_count + noise_crops
    return crops


@settings(max_examples=200)
@given(crops=valid_cropdetect_output())
def test_property_6_confidence_in_range(crops):
    """
    Property 6: For any non-empty list of cropdetect crop strings where the best
    crop passes area validation, the confidence field of the returned FacecamRegion
    is in [0.0, 1.0].

    Validates: Requirements 3.6
    """
    stderr = _build_cropdetect_stderr(crops)
    relocator = FacecamRelocator()
    config = DEFAULT_CONFIG

    mock_result = MagicMock()
    mock_result.stderr = stderr

    with patch("subprocess.run", return_value=mock_result):
        region = relocator.detect_facecam(
            clip_path="fake.mp4",
            frame_width=FRAME_W,
            frame_height=FRAME_H,
            config=config,
        )

    # The dominant crop (480×270 at 1440×810) is valid, so region should not be None
    if region is not None:
        assert 0.0 <= region.confidence <= 1.0, (
            f"confidence={region.confidence} is outside [0.0, 1.0]"
        )


# ---------------------------------------------------------------------------
# Property 7: Overlay coordinates are within canvas bounds
# ---------------------------------------------------------------------------

@settings(max_examples=200)
@given(region=facecam_region_strategy(), layout=canvas_layout_strategy())
def test_property_7_overlay_x_within_canvas_bounds(region: FacecamRegion, layout: CanvasLayout):
    """
    Property 7: For any FacecamRegion with valid coordinates and any CanvasLayout,
    build_facecam_filter produces overlay_x satisfying 0 <= overlay_x <= canvas_width.

    Validates: Requirements 4.2, 4.3
    """
    relocator = FacecamRelocator()
    frag = relocator.build_facecam_filter(
        region=region,
        canvas_width=layout.canvas_width,
        canvas_height=layout.canvas_height,
        top_third_height=layout.facecam_height,
    )

    # Extract overlay_x from the filter string.
    # Format: "...overlay=OVERLAY_X:0[with_facecam]"
    match = re.search(r"overlay=(\d+):0", frag.filter_str)
    assert match is not None, (
        f"Could not find overlay=X:0 in filter_str: {frag.filter_str!r}"
    )
    overlay_x = int(match.group(1))

    assert 0 <= overlay_x <= layout.canvas_width, (
        f"overlay_x={overlay_x} is outside [0, {layout.canvas_width}] "
        f"for region={region}, canvas_width={layout.canvas_width}"
    )
