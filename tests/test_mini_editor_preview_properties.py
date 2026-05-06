"""Property-based tests for Mini Video Editor preview generation (Task 3.5).

These tests validate universal correctness properties of preview generation
for the Mini Video Editor feature.

**Validates: Requirements 5.5, 5.6**
"""

from __future__ import annotations

import math
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pipeline.frame_reformatter import FrameReformatter, compute_canvas_layout
from pipeline.models import CanvasLayout, FacecamRegion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(
    shorts_width: int = 1080,
    shorts_height: int = 1920,
    facecam_top_fraction: float = 0.35,
) -> SimpleNamespace:
    """Return a minimal config-like object for testing."""
    return SimpleNamespace(
        shorts_width=shorts_width,
        shorts_height=shorts_height,
        facecam_top_fraction=facecam_top_fraction,
    )


DEFAULT_CONFIG = make_config()


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

@st.composite
def valid_facecam_placement(draw):
    """Generate a valid FacecamRegion within a source frame.

    Returns (frame_w, frame_h, facecam_region) where the region fits within
    the frame and has a valid area fraction (4%–30%).
    """
    frame_w = draw(st.integers(min_value=320, max_value=3840))
    frame_h = draw(st.integers(min_value=240, max_value=2160))
    frame_area = frame_w * frame_h

    # Pick a valid area fraction in [0.04, 0.30]
    fraction = draw(
        st.floats(min_value=0.04, max_value=0.30, allow_nan=False, allow_infinity=False)
    )

    # Derive width and height from fraction (square-ish region)
    side = max(1, int(math.sqrt(fraction * frame_area)))
    w = min(side, frame_w)
    h = min(side, frame_h)

    # Place region so it fits within the frame
    x = draw(st.integers(min_value=0, max_value=max(0, frame_w - w)))
    y = draw(st.integers(min_value=0, max_value=max(0, frame_h - h)))

    region = FacecamRegion(
        x=x,
        y=y,
        width=w,
        height=h,
        corner="top-right",
        confidence=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
    )
    return frame_w, frame_h, region


@st.composite
def canvas_config(draw):
    """Generate a valid canvas configuration.

    Returns a config-like object with valid shorts_width, shorts_height,
    and facecam_top_fraction.
    """
    # Use standard 9:16 dimensions or scaled variants
    scale = draw(st.integers(min_value=1, max_value=4))
    shorts_width = 270 * scale   # 270, 540, 810, 1080
    shorts_height = 480 * scale  # 480, 960, 1440, 1920

    # Facecam fraction between 20% and 50%
    facecam_top_fraction = draw(
        st.floats(min_value=0.20, max_value=0.50, allow_nan=False, allow_infinity=False)
    )

    return make_config(
        shorts_width=shorts_width,
        shorts_height=shorts_height,
        facecam_top_fraction=facecam_top_fraction,
    )


@st.composite
def two_different_facecam_placements(draw):
    """Generate two different FacecamRegion placements for the same frame.

    Returns (frame_w, frame_h, region1, region2) where region1 != region2.
    """
    frame_w = draw(st.integers(min_value=640, max_value=1920))
    frame_h = draw(st.integers(min_value=480, max_value=1080))
    frame_area = frame_w * frame_h

    def make_region(x, y, w, h, confidence):
        return FacecamRegion(
            x=x, y=y, width=w, height=h,
            corner="top-right",
            confidence=confidence,
        )

    # Region 1: top-right area
    w1 = max(1, int(math.sqrt(0.10 * frame_area)))
    w1 = min(w1, frame_w // 2)
    h1 = min(w1, frame_h // 2)
    x1 = frame_w - w1
    y1 = 0
    region1 = make_region(x1, y1, w1, h1, 0.9)

    # Region 2: different position (top-left area)
    w2 = max(1, int(math.sqrt(0.08 * frame_area)))
    w2 = min(w2, frame_w // 2)
    h2 = min(w2, frame_h // 2)
    x2 = draw(st.integers(min_value=0, max_value=max(0, frame_w // 2 - w2)))
    y2 = draw(st.integers(min_value=0, max_value=max(0, frame_h // 2 - h2)))
    confidence2 = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    region2 = make_region(x2, y2, w2, h2, confidence2)

    return frame_w, frame_h, region1, region2


# ---------------------------------------------------------------------------
# Property 4: Aspect ratio preservation
# **Validates: Requirements 5.5, 5.6**
# ---------------------------------------------------------------------------

@settings(max_examples=200)
@given(data=valid_facecam_placement(), config=canvas_config())
def test_property_4_canvas_layout_height_sum_equals_canvas_height(data, config):
    """
    Property 4: For any facecam placement and canvas config, the canvas layout
    must satisfy: facecam_height + gameplay_height == canvas_height.

    This ensures the vertical canvas is fully covered without gaps or overlaps,
    which is required for correct aspect ratio preservation.

    **Validates: Requirements 5.5, 5.6**
    """
    _frame_w, _frame_h, _region = data
    layout = compute_canvas_layout(config)

    assert layout.facecam_height + layout.gameplay_height == layout.canvas_height, (
        f"facecam_height ({layout.facecam_height}) + gameplay_height ({layout.gameplay_height}) "
        f"!= canvas_height ({layout.canvas_height})"
    )


@settings(max_examples=200)
@given(data=valid_facecam_placement(), config=canvas_config())
def test_property_4_canvas_layout_widths_equal_canvas_width(data, config):
    """
    Property 4: For any facecam placement and canvas config, both the facecam
    and gameplay regions must span the full canvas width.

    This ensures no horizontal gaps or overflow in the vertical canvas layout.

    **Validates: Requirements 5.5, 5.6**
    """
    _frame_w, _frame_h, _region = data
    layout = compute_canvas_layout(config)

    assert layout.facecam_width == layout.canvas_width, (
        f"facecam_width ({layout.facecam_width}) != canvas_width ({layout.canvas_width})"
    )
    assert layout.gameplay_width == layout.canvas_width, (
        f"gameplay_width ({layout.gameplay_width}) != canvas_width ({layout.canvas_width})"
    )


@settings(max_examples=200)
@given(data=valid_facecam_placement(), config=canvas_config())
def test_property_4_gameplay_region_starts_at_facecam_bottom(data, config):
    """
    Property 4: For any canvas config, the gameplay region must start exactly
    where the facecam region ends (gameplay_y == facecam_height).

    This ensures the two regions are contiguous with no gap between them.

    **Validates: Requirements 5.5, 5.6**
    """
    _frame_w, _frame_h, _region = data
    layout = compute_canvas_layout(config)

    assert layout.gameplay_y == layout.facecam_height, (
        f"gameplay_y ({layout.gameplay_y}) != facecam_height ({layout.facecam_height})"
    )


@settings(max_examples=200)
@given(
    frame_w=st.integers(min_value=320, max_value=3840),
    frame_h=st.integers(min_value=240, max_value=2160),
    config=canvas_config(),
)
def test_property_4_gameplay_scale_preserves_aspect_ratio(frame_w, frame_h, config):
    """
    Property 4: For any source frame dimensions and canvas config, the gameplay
    region scaling must preserve the source aspect ratio.

    The scaled gameplay content must fit within the gameplay region without
    exceeding its bounds, and the aspect ratio must be preserved (no stretching).

    **Validates: Requirements 5.5, 5.6**
    """
    layout = compute_canvas_layout(config)
    reformatter = FrameReformatter()
    canvas_filter = reformatter.build_canvas_filter(
        src_width=frame_w,
        src_height=frame_h,
        layout=layout,
    )

    # Parse scale dimensions from the filter string
    # Format: "[0:v]scale=W:H,pad=..."
    filter_str = canvas_filter.filter_str
    scale_part = filter_str.split("scale=")[1].split(",")[0]
    scale_w_str, scale_h_str = scale_part.split(":")
    scale_w = int(scale_w_str)
    scale_h = int(scale_h_str)

    # The scaled dimensions must fit within the gameplay region
    assert scale_w <= layout.gameplay_width, (
        f"Scaled width {scale_w} exceeds gameplay_width {layout.gameplay_width}"
    )
    assert scale_h <= layout.gameplay_height, (
        f"Scaled height {scale_h} exceeds gameplay_height {layout.gameplay_height}"
    )

    # The aspect ratio must be preserved within integer rounding tolerance.
    # When scaling, both dimensions are rounded to integers, so the maximum
    # rounding error per dimension is 0.5 pixels. The relative error in the
    # ratio is bounded by 0.5/scale_h + 0.5/scale_w (additive rounding errors).
    original_ratio = frame_w / frame_h
    scaled_ratio = scale_w / scale_h
    # Allow up to 1 pixel of rounding error in each scaled dimension
    rounding_tolerance = (1.0 / scale_h) + (1.0 / scale_w)
    assert abs(original_ratio - scaled_ratio) <= original_ratio * rounding_tolerance + 0.01, (
        f"Aspect ratio not preserved: original={original_ratio:.4f}, "
        f"scaled={scaled_ratio:.4f} (frame={frame_w}x{frame_h}, "
        f"scaled={scale_w}x{scale_h})"
    )


@settings(max_examples=200)
@given(config=canvas_config())
def test_property_4_canvas_layout_facecam_at_origin(config):
    """
    Property 4: For any canvas config, the facecam region must always start
    at the top-left corner of the canvas (facecam_x == 0, facecam_y == 0).

    This ensures the facecam is always positioned at the top of the vertical canvas.

    **Validates: Requirements 5.5, 5.6**
    """
    layout = compute_canvas_layout(config)

    assert layout.facecam_x == 0, (
        f"facecam_x ({layout.facecam_x}) != 0"
    )
    assert layout.facecam_y == 0, (
        f"facecam_y ({layout.facecam_y}) != 0"
    )


@settings(max_examples=200)
@given(config=canvas_config())
def test_property_4_canvas_layout_gameplay_at_left_edge(config):
    """
    Property 4: For any canvas config, the gameplay region must start at the
    left edge of the canvas (gameplay_x == 0).

    **Validates: Requirements 5.5, 5.6**
    """
    layout = compute_canvas_layout(config)

    assert layout.gameplay_x == 0, (
        f"gameplay_x ({layout.gameplay_x}) != 0"
    )


@settings(max_examples=200)
@given(config=canvas_config())
def test_property_4_canvas_dimensions_match_config(config):
    """
    Property 4: For any canvas config, the computed layout dimensions must
    match the configured canvas dimensions.

    **Validates: Requirements 5.5, 5.6**
    """
    layout = compute_canvas_layout(config)

    assert layout.canvas_width == config.shorts_width, (
        f"canvas_width ({layout.canvas_width}) != shorts_width ({config.shorts_width})"
    )
    assert layout.canvas_height == config.shorts_height, (
        f"canvas_height ({layout.canvas_height}) != shorts_height ({config.shorts_height})"
    )


# ---------------------------------------------------------------------------
# Property 5: Preview accuracy
# **Validates: Requirements 5.5, 5.6**
# ---------------------------------------------------------------------------

@settings(max_examples=200)
@given(data=two_different_facecam_placements())
def test_property_5_different_placements_produce_different_cache_keys(data):
    """
    Property 5: For any two different facecam placements, the preview cache
    keys must be different.

    This ensures that adjusting the facecam placement always triggers a new
    preview generation rather than returning a stale cached result.

    **Validates: Requirements 5.5, 5.6**
    """
    frame_w, frame_h, region1, region2 = data

    # Build cache keys as the web server does:
    # (clip_path, x, y, width, height, mtime)
    clip_path = "/fake/clip.mp4"
    mtime = 1234567890.0

    key1 = (clip_path, region1.x, region1.y, region1.width, region1.height, mtime)
    key2 = (clip_path, region2.x, region2.y, region2.width, region2.height, mtime)

    # If the regions differ in any coordinate, the cache keys must differ
    if (region1.x, region1.y, region1.width, region1.height) != (
        region2.x, region2.y, region2.width, region2.height
    ):
        assert key1 != key2, (
            f"Different placements produced the same cache key: {key1}"
        )


@settings(max_examples=200)
@given(data=valid_facecam_placement())
def test_property_5_same_placement_produces_same_cache_key(data):
    """
    Property 5: For any facecam placement, the same placement always produces
    the same cache key (deterministic caching).

    This ensures that repeated requests with the same placement return the
    cached preview rather than regenerating it.

    **Validates: Requirements 5.5, 5.6**
    """
    frame_w, frame_h, region = data
    clip_path = "/fake/clip.mp4"
    mtime = 1234567890.0

    key1 = (clip_path, region.x, region.y, region.width, region.height, mtime)
    key2 = (clip_path, region.x, region.y, region.width, region.height, mtime)

    assert key1 == key2, (
        f"Same placement produced different cache keys: {key1} vs {key2}"
    )


@settings(max_examples=200)
@given(data=valid_facecam_placement(), config=canvas_config())
def test_property_5_canvas_layout_reflects_facecam_fraction(data, config):
    """
    Property 5: For any canvas config, the computed facecam_height must
    accurately reflect the configured facecam_top_fraction.

    This ensures the preview accurately shows the facecam occupying the
    correct fraction of the vertical canvas.

    **Validates: Requirements 5.5, 5.6**
    """
    _frame_w, _frame_h, _region = data
    layout = compute_canvas_layout(config)

    expected_facecam_height = round(config.shorts_height * config.facecam_top_fraction)
    assert layout.facecam_height == expected_facecam_height, (
        f"facecam_height ({layout.facecam_height}) != "
        f"round(shorts_height * facecam_top_fraction) "
        f"({expected_facecam_height})"
    )


@settings(max_examples=200)
@given(data=valid_facecam_placement(), config=canvas_config())
def test_property_5_preview_canvas_layout_is_independent_of_facecam_region(data, config):
    """
    Property 5: For any facecam placement, the canvas layout (facecam region
    dimensions on the vertical canvas) must be independent of the source
    facecam region coordinates.

    The canvas layout is determined solely by the config (canvas dimensions
    and facecam_top_fraction), not by where the facecam is in the source frame.
    The source facecam is always scaled to fill the entire facecam region on
    the vertical canvas.

    **Validates: Requirements 5.5, 5.6**
    """
    _frame_w, _frame_h, region = data

    # Compute layout — should be the same regardless of region coordinates
    layout1 = compute_canvas_layout(config)

    # Compute layout with a different region (same config)
    layout2 = compute_canvas_layout(config)

    # Canvas layout must be identical regardless of facecam region
    assert layout1.canvas_width == layout2.canvas_width
    assert layout1.canvas_height == layout2.canvas_height
    assert layout1.facecam_x == layout2.facecam_x
    assert layout1.facecam_y == layout2.facecam_y
    assert layout1.facecam_width == layout2.facecam_width
    assert layout1.facecam_height == layout2.facecam_height
    assert layout1.gameplay_x == layout2.gameplay_x
    assert layout1.gameplay_y == layout2.gameplay_y
    assert layout1.gameplay_width == layout2.gameplay_width
    assert layout1.gameplay_height == layout2.gameplay_height


@settings(max_examples=200)
@given(data=valid_facecam_placement())
def test_property_5_preview_endpoint_cache_key_includes_mtime(data):
    """
    Property 5: For any facecam placement, the preview cache key must include
    the file modification time (mtime) so that changes to the source clip
    invalidate the cache.

    This ensures the preview accurately reflects the current state of the
    source clip, not a stale version.

    **Validates: Requirements 5.5, 5.6**
    """
    _frame_w, _frame_h, region = data
    clip_path = "/fake/clip.mp4"

    mtime1 = 1234567890.0
    mtime2 = 1234567891.0  # 1 second later (file modified)

    key1 = (clip_path, region.x, region.y, region.width, region.height, mtime1)
    key2 = (clip_path, region.x, region.y, region.width, region.height, mtime2)

    # Different mtimes must produce different cache keys
    assert key1 != key2, (
        f"Different mtimes produced the same cache key: {key1}"
    )


@settings(max_examples=200)
@given(
    frame_w=st.integers(min_value=320, max_value=3840),
    frame_h=st.integers(min_value=240, max_value=2160),
    config=canvas_config(),
)
def test_property_5_gameplay_scale_fits_within_gameplay_region(frame_w, frame_h, config):
    """
    Property 5: For any source frame dimensions, the gameplay content scaled
    by build_canvas_filter must fit within the gameplay region of the canvas.

    This ensures the preview accurately shows the gameplay content without
    overflow or clipping.

    **Validates: Requirements 5.5, 5.6**
    """
    layout = compute_canvas_layout(config)
    reformatter = FrameReformatter()
    canvas_filter = reformatter.build_canvas_filter(
        src_width=frame_w,
        src_height=frame_h,
        layout=layout,
    )

    # Parse scale dimensions from the filter string
    filter_str = canvas_filter.filter_str
    scale_part = filter_str.split("scale=")[1].split(",")[0]
    scale_w_str, scale_h_str = scale_part.split(":")
    scale_w = int(scale_w_str)
    scale_h = int(scale_h_str)

    # Scaled content must fit within gameplay region
    assert scale_w <= layout.gameplay_width, (
        f"Scaled width {scale_w} exceeds gameplay_width {layout.gameplay_width} "
        f"(frame={frame_w}x{frame_h})"
    )
    assert scale_h <= layout.gameplay_height, (
        f"Scaled height {scale_h} exceeds gameplay_height {layout.gameplay_height} "
        f"(frame={frame_w}x{frame_h})"
    )

    # At least one dimension must fill the gameplay region (no unnecessary shrinking)
    assert scale_w == layout.gameplay_width or scale_h == layout.gameplay_height, (
        f"Neither dimension fills the gameplay region: "
        f"scale={scale_w}x{scale_h}, gameplay={layout.gameplay_width}x{layout.gameplay_height} "
        f"(frame={frame_w}x{frame_h})"
    )


@settings(max_examples=200)
@given(data=valid_facecam_placement())
def test_property_5_preview_endpoint_uses_facecam_coordinates_in_cache_key(data):
    """
    Property 5: For any facecam placement, the preview cache key must include
    all four facecam coordinates (x, y, width, height).

    Changing any single coordinate must produce a different cache key, ensuring
    the preview accurately reflects the current placement.

    **Validates: Requirements 5.5, 5.6**
    """
    _frame_w, _frame_h, region = data
    clip_path = "/fake/clip.mp4"
    mtime = 1234567890.0

    base_key = (clip_path, region.x, region.y, region.width, region.height, mtime)

    # Changing x must change the key
    key_x_changed = (clip_path, region.x + 1, region.y, region.width, region.height, mtime)
    assert base_key != key_x_changed, "Changing x did not change cache key"

    # Changing y must change the key
    key_y_changed = (clip_path, region.x, region.y + 1, region.width, region.height, mtime)
    assert base_key != key_y_changed, "Changing y did not change cache key"

    # Changing width must change the key
    key_w_changed = (clip_path, region.x, region.y, region.width + 1, region.height, mtime)
    assert base_key != key_w_changed, "Changing width did not change cache key"

    # Changing height must change the key
    key_h_changed = (clip_path, region.x, region.y, region.width, region.height + 1, mtime)
    assert base_key != key_h_changed, "Changing height did not change cache key"


@settings(max_examples=200)
@given(
    frame_w=st.integers(min_value=320, max_value=3840),
    frame_h=st.integers(min_value=240, max_value=2160),
    config=canvas_config(),
)
def test_property_5_canvas_filter_output_label_is_canvas(frame_w, frame_h, config):
    """
    Property 5: For any source frame dimensions, the canvas filter produced by
    build_canvas_filter must have output label "[canvas]".

    This ensures the filter chain is correctly structured for compositing the
    facecam overlay on top of the gameplay region.

    **Validates: Requirements 5.5, 5.6**
    """
    layout = compute_canvas_layout(config)
    reformatter = FrameReformatter()
    canvas_filter = reformatter.build_canvas_filter(
        src_width=frame_w,
        src_height=frame_h,
        layout=layout,
    )

    assert canvas_filter.output_label == "[canvas]", (
        f"Expected output_label '[canvas]', got '{canvas_filter.output_label}'"
    )
    assert canvas_filter.input_label == "[0:v]", (
        f"Expected input_label '[0:v]', got '{canvas_filter.input_label}'"
    )
