"""Property-based tests for vertical formatter (Task 6.4).

Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5, 14.1, 14.4
"""

from __future__ import annotations

import math
import os
import tempfile
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from pipeline.models import CanvasLayout, FacecamRegion, VerticalFormattingJob
from pipeline.vertical_formatter import (
    VerticalFormatter,
    _build_vertical_filter,
    get_output_path,
    scale_region_to_resolution,
    process_vertical_formatting_job,
)
from pipeline.frame_reformatter import compute_canvas_layout


# ---------------------------------------------------------------------------
# Helpers / Strategies
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


@st.composite
def valid_facecam_region(draw, frame_w=None, frame_h=None):
    """Generate a valid FacecamRegion within a source frame."""
    if frame_w is None:
        frame_w = draw(st.integers(min_value=320, max_value=1920))
    if frame_h is None:
        frame_h = draw(st.integers(min_value=240, max_value=1080))

    frame_area = frame_w * frame_h
    fraction = draw(
        st.floats(min_value=0.04, max_value=0.30, allow_nan=False, allow_infinity=False)
    )
    side = max(1, int(math.sqrt(fraction * frame_area)))
    w = min(side, frame_w)
    h = min(side, frame_h)
    x = draw(st.integers(min_value=0, max_value=max(0, frame_w - w)))
    y = draw(st.integers(min_value=0, max_value=max(0, frame_h - h)))

    return FacecamRegion(
        x=x,
        y=y,
        width=w,
        height=h,
        corner=draw(st.sampled_from(["top-left", "top-right", "bottom-left", "bottom-right"])),
        confidence=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
    )


@st.composite
def canvas_config(draw):
    """Generate a valid canvas configuration."""
    scale = draw(st.integers(min_value=1, max_value=4))
    shorts_width = 270 * scale
    shorts_height = 480 * scale
    facecam_top_fraction = draw(
        st.floats(min_value=0.20, max_value=0.50, allow_nan=False, allow_infinity=False)
    )
    return make_config(
        shorts_width=shorts_width,
        shorts_height=shorts_height,
        facecam_top_fraction=facecam_top_fraction,
    )


@st.composite
def resolution_pair(draw):
    """Generate two different resolutions (reference and target)."""
    ref_w = draw(st.integers(min_value=320, max_value=1920))
    ref_h = draw(st.integers(min_value=240, max_value=1080))
    # Scale factor between 0.5x and 2x
    scale_x = draw(st.floats(min_value=0.5, max_value=2.0, allow_nan=False, allow_infinity=False))
    scale_y = draw(st.floats(min_value=0.5, max_value=2.0, allow_nan=False, allow_infinity=False))
    tgt_w = max(1, round(ref_w * scale_x))
    tgt_h = max(1, round(ref_h * scale_y))
    return (ref_w, ref_h), (tgt_w, tgt_h)


def make_canvas_layout(config=None) -> CanvasLayout:
    """Build a CanvasLayout from config."""
    if config is None:
        config = DEFAULT_CONFIG
    return compute_canvas_layout(config)


def make_job(
    clips: list[dict],
    facecam_region: FacecamRegion,
    canvas_layout: CanvasLayout,
    output_dir: str,
    settings_dict: dict | None = None,
) -> VerticalFormattingJob:
    """Create a VerticalFormattingJob for testing."""
    return VerticalFormattingJob(
        job_id=str(uuid.uuid4()),
        session_id=str(uuid.uuid4()),
        clip_batch_id=str(uuid.uuid4()),
        facecam_region=facecam_region,
        canvas_layout=canvas_layout,
        settings=settings_dict or {},
        clips=clips,
        output_dir=output_dir,
    )


# ---------------------------------------------------------------------------
# Property 11: Consistent placement
# **Validates: Requirements 7.1, 7.2, 7.3, 7.4**
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    region=valid_facecam_region(),
    config=canvas_config(),
    n_clips=st.integers(min_value=1, max_value=5),
)
def test_property_11_same_filter_applied_to_all_clips(region, config, n_clips):
    """
    Property 11: For any batch of clips with the same facecam_region and
    canvas_layout, the FFmpeg filter_complex string generated for each clip
    must be identical (assuming the same source resolution).

    This ensures the same placement is applied consistently across all clips
    in the batch.

    **Validates: Requirements 7.1, 7.2, 7.3, 7.4**
    """
    layout = compute_canvas_layout(config)
    src_w, src_h = 1920, 1080  # same resolution for all clips

    # Generate filter for each clip — must all be identical
    filters = [
        _build_vertical_filter(src_w, src_h, region, layout)
        for _ in range(n_clips)
    ]

    # All filters must be identical
    assert len(set(filters)) == 1, (
        f"Expected identical filters for all {n_clips} clips, "
        f"but got {len(set(filters))} distinct filters"
    )


@settings(max_examples=100)
@given(
    region=valid_facecam_region(),
    config=canvas_config(),
)
def test_property_11_filter_contains_canvas_and_facecam_and_overlay(region, config):
    """
    Property 11: For any facecam region and canvas config, the generated
    filter_complex must contain all three required stages:
    - canvas (gameplay scaling + padding)
    - facecam crop + scale
    - overlay compositing

    **Validates: Requirements 7.1, 7.2, 7.3, 7.4**
    """
    layout = compute_canvas_layout(config)
    src_w, src_h = 1920, 1080

    filter_str = _build_vertical_filter(src_w, src_h, region, layout)

    # Must contain canvas output label
    assert "[canvas]" in filter_str, (
        f"Filter missing '[canvas]' label: {filter_str}"
    )
    # Must contain facecam_scaled label
    assert "[facecam_scaled]" in filter_str, (
        f"Filter missing '[facecam_scaled]' label: {filter_str}"
    )
    # Must contain overlay output label
    assert "[with_facecam]" in filter_str, (
        f"Filter missing '[with_facecam]' label: {filter_str}"
    )
    # Must contain crop filter with the region coordinates
    assert f"crop={region.width}:{region.height}:{region.x}:{region.y}" in filter_str, (
        f"Filter missing expected crop parameters: {filter_str}"
    )
    # Must contain overlay filter
    assert "overlay=" in filter_str, (
        f"Filter missing 'overlay=' directive: {filter_str}"
    )


@settings(max_examples=100)
@given(
    region=valid_facecam_region(),
    config=canvas_config(),
)
def test_property_11_filter_is_deterministic(region, config):
    """
    Property 11: For any facecam region and canvas config, the filter
    generation must be deterministic — calling it twice with the same inputs
    must produce the same output.

    **Validates: Requirements 7.1, 7.2, 7.3, 7.4**
    """
    layout = compute_canvas_layout(config)
    src_w, src_h = 1280, 720

    filter1 = _build_vertical_filter(src_w, src_h, region, layout)
    filter2 = _build_vertical_filter(src_w, src_h, region, layout)

    assert filter1 == filter2, (
        f"Filter generation is not deterministic: '{filter1}' != '{filter2}'"
    )


@settings(max_examples=100)
@given(
    region=valid_facecam_region(),
    config=canvas_config(),
)
def test_property_11_job_facecam_region_unchanged_after_filter_generation(region, config):
    """
    Property 11: For any facecam region, generating the FFmpeg filter must
    not mutate the original FacecamRegion object.

    This ensures the same region object can be safely reused across all clips
    in the batch.

    **Validates: Requirements 7.1, 7.2, 7.3, 7.4**
    """
    layout = compute_canvas_layout(config)
    src_w, src_h = 1920, 1080

    # Record original values
    orig_x, orig_y = region.x, region.y
    orig_w, orig_h = region.width, region.height
    orig_corner = region.corner
    orig_confidence = region.confidence

    _build_vertical_filter(src_w, src_h, region, layout)

    # Region must be unchanged
    assert region.x == orig_x
    assert region.y == orig_y
    assert region.width == orig_w
    assert region.height == orig_h
    assert region.corner == orig_corner
    assert region.confidence == orig_confidence


# ---------------------------------------------------------------------------
# Property 12: Resolution scaling
# **Validates: Requirements 7.5, 14.1**
# ---------------------------------------------------------------------------

@settings(max_examples=200)
@given(
    region=valid_facecam_region(),
    resolutions=resolution_pair(),
)
def test_property_12_scaled_region_preserves_relative_position(region, resolutions):
    """
    Property 12: For any facecam region and resolution pair, the scaled region
    must preserve the relative position of the facecam within the frame.

    The ratio (x / frame_width) and (y / frame_height) must be the same in
    both the original and scaled regions (within rounding tolerance).

    **Validates: Requirements 7.5, 14.1**
    """
    (ref_w, ref_h), (tgt_w, tgt_h) = resolutions
    assume(ref_w > 0 and ref_h > 0 and tgt_w > 0 and tgt_h > 0)

    scaled = scale_region_to_resolution(region, (ref_w, ref_h), (tgt_w, tgt_h))

    # Relative x position must be preserved
    if ref_w > 0 and tgt_w > 0:
        orig_rel_x = region.x / ref_w
        scaled_rel_x = scaled.x / tgt_w
        # Allow 1 pixel of rounding error
        tolerance = 1.0 / tgt_w + 1.0 / ref_w
        assert abs(orig_rel_x - scaled_rel_x) <= tolerance + 0.001, (
            f"Relative x position not preserved: "
            f"orig={orig_rel_x:.4f}, scaled={scaled_rel_x:.4f} "
            f"(region.x={region.x}, ref_w={ref_w}, tgt_w={tgt_w}, scaled.x={scaled.x})"
        )

    # Relative y position must be preserved
    if ref_h > 0 and tgt_h > 0:
        orig_rel_y = region.y / ref_h
        scaled_rel_y = scaled.y / tgt_h
        tolerance = 1.0 / tgt_h + 1.0 / ref_h
        assert abs(orig_rel_y - scaled_rel_y) <= tolerance + 0.001, (
            f"Relative y position not preserved: "
            f"orig={orig_rel_y:.4f}, scaled={scaled_rel_y:.4f} "
            f"(region.y={region.y}, ref_h={ref_h}, tgt_h={tgt_h}, scaled.y={scaled.y})"
        )


@settings(max_examples=200)
@given(
    region=valid_facecam_region(),
    resolutions=resolution_pair(),
)
def test_property_12_scaled_region_preserves_relative_size(region, resolutions):
    """
    Property 12: For any facecam region and resolution pair, the scaled region
    must preserve the relative size of the facecam within the frame.

    The ratio (width / frame_width) and (height / frame_height) must be the
    same in both the original and scaled regions (within rounding tolerance).

    **Validates: Requirements 7.5, 14.1**
    """
    (ref_w, ref_h), (tgt_w, tgt_h) = resolutions
    assume(ref_w > 0 and ref_h > 0 and tgt_w > 0 and tgt_h > 0)
    assume(region.width > 0 and region.height > 0)

    scaled = scale_region_to_resolution(region, (ref_w, ref_h), (tgt_w, tgt_h))

    # Relative width must be preserved
    orig_rel_w = region.width / ref_w
    scaled_rel_w = scaled.width / tgt_w
    tolerance_w = 1.0 / tgt_w + 1.0 / ref_w
    assert abs(orig_rel_w - scaled_rel_w) <= tolerance_w + 0.001, (
        f"Relative width not preserved: "
        f"orig={orig_rel_w:.4f}, scaled={scaled_rel_w:.4f} "
        f"(region.width={region.width}, ref_w={ref_w}, tgt_w={tgt_w}, scaled.width={scaled.width})"
    )

    # Relative height must be preserved
    orig_rel_h = region.height / ref_h
    scaled_rel_h = scaled.height / tgt_h
    tolerance_h = 1.0 / tgt_h + 1.0 / ref_h
    assert abs(orig_rel_h - scaled_rel_h) <= tolerance_h + 0.001, (
        f"Relative height not preserved: "
        f"orig={orig_rel_h:.4f}, scaled={scaled_rel_h:.4f} "
        f"(region.height={region.height}, ref_h={ref_h}, tgt_h={tgt_h}, scaled.height={scaled.height})"
    )


@settings(max_examples=200)
@given(region=valid_facecam_region())
def test_property_12_identity_scaling_returns_same_coordinates(region):
    """
    Property 12: Scaling a region from a resolution to the same resolution
    must return a region with identical coordinates.

    **Validates: Requirements 7.5, 14.1**
    """
    resolution = (1920, 1080)
    scaled = scale_region_to_resolution(region, resolution, resolution)

    assert scaled.x == region.x, f"x changed: {region.x} -> {scaled.x}"
    assert scaled.y == region.y, f"y changed: {region.y} -> {scaled.y}"
    assert scaled.width == region.width, f"width changed: {region.width} -> {scaled.width}"
    assert scaled.height == region.height, f"height changed: {region.height} -> {scaled.height}"
    assert scaled.corner == region.corner
    assert scaled.confidence == region.confidence


@settings(max_examples=200)
@given(
    region=valid_facecam_region(),
    resolutions=resolution_pair(),
)
def test_property_12_scaling_preserves_corner_and_confidence(region, resolutions):
    """
    Property 12: Scaling a region must preserve the corner classification and
    confidence score — only the pixel coordinates change.

    **Validates: Requirements 7.5, 14.1**
    """
    (ref_w, ref_h), (tgt_w, tgt_h) = resolutions
    scaled = scale_region_to_resolution(region, (ref_w, ref_h), (tgt_w, tgt_h))

    assert scaled.corner == region.corner, (
        f"Corner changed after scaling: {region.corner} -> {scaled.corner}"
    )
    assert scaled.confidence == region.confidence, (
        f"Confidence changed after scaling: {region.confidence} -> {scaled.confidence}"
    )


@settings(max_examples=200)
@given(
    region=valid_facecam_region(),
    resolutions=resolution_pair(),
)
def test_property_12_round_trip_scaling_is_approximately_identity(region, resolutions):
    """
    Property 12: Scaling a region from resolution A to B and back to A must
    return approximately the original coordinates (within rounding tolerance).

    **Validates: Requirements 7.5, 14.1**
    """
    (ref_w, ref_h), (tgt_w, tgt_h) = resolutions
    assume(ref_w > 0 and ref_h > 0 and tgt_w > 0 and tgt_h > 0)

    # Scale A -> B -> A
    scaled_to_tgt = scale_region_to_resolution(region, (ref_w, ref_h), (tgt_w, tgt_h))
    scaled_back = scale_region_to_resolution(scaled_to_tgt, (tgt_w, tgt_h), (ref_w, ref_h))

    # Allow 1 pixel of rounding error per round trip
    assert abs(scaled_back.x - region.x) <= 1, (
        f"Round-trip x not preserved: {region.x} -> {scaled_to_tgt.x} -> {scaled_back.x}"
    )
    assert abs(scaled_back.y - region.y) <= 1, (
        f"Round-trip y not preserved: {region.y} -> {scaled_to_tgt.y} -> {scaled_back.y}"
    )
    assert abs(scaled_back.width - region.width) <= 1, (
        f"Round-trip width not preserved: {region.width} -> {scaled_to_tgt.width} -> {scaled_back.width}"
    )
    assert abs(scaled_back.height - region.height) <= 1, (
        f"Round-trip height not preserved: {region.height} -> {scaled_to_tgt.height} -> {scaled_back.height}"
    )


# ---------------------------------------------------------------------------
# Property 13: Aspect ratio preservation
# **Validates: Requirements 14.1, 14.4**
# ---------------------------------------------------------------------------

@settings(max_examples=200)
@given(
    src_w=st.integers(min_value=320, max_value=3840),
    src_h=st.integers(min_value=240, max_value=2160),
    config=canvas_config(),
)
def test_property_13_gameplay_scale_preserves_source_aspect_ratio(src_w, src_h, config):
    """
    Property 13: For any source video dimensions, the gameplay region scaling
    must preserve the source aspect ratio (no stretching or distortion).

    The scaled gameplay content must have the same width/height ratio as the
    source video, within integer rounding tolerance.

    **Validates: Requirements 14.1, 14.4**
    """
    from pipeline.frame_reformatter import FrameReformatter

    layout = compute_canvas_layout(config)
    reformatter = FrameReformatter()
    canvas_filter = reformatter.build_canvas_filter(src_w, src_h, layout)

    # Parse scale dimensions from the filter string
    # Format: "[0:v]scale=W:H,pad=..."
    filter_str = canvas_filter.filter_str
    scale_part = filter_str.split("scale=")[1].split(",")[0]
    scale_w_str, scale_h_str = scale_part.split(":")
    scale_w = int(scale_w_str)
    scale_h = int(scale_h_str)

    # Aspect ratio must be preserved within rounding tolerance
    original_ratio = src_w / src_h
    scaled_ratio = scale_w / scale_h
    # Allow up to 1 pixel of rounding error in each dimension
    rounding_tolerance = (1.0 / scale_h) + (1.0 / scale_w)
    assert abs(original_ratio - scaled_ratio) <= original_ratio * rounding_tolerance + 0.01, (
        f"Aspect ratio not preserved: original={original_ratio:.4f}, "
        f"scaled={scaled_ratio:.4f} (src={src_w}x{src_h}, scaled={scale_w}x{scale_h})"
    )


@settings(max_examples=200)
@given(
    src_w=st.integers(min_value=320, max_value=3840),
    src_h=st.integers(min_value=240, max_value=2160),
    config=canvas_config(),
)
def test_property_13_gameplay_scale_fits_within_gameplay_region(src_w, src_h, config):
    """
    Property 13: For any source video dimensions, the scaled gameplay content
    must fit within the gameplay region of the canvas (no overflow).

    **Validates: Requirements 14.1, 14.4**
    """
    from pipeline.frame_reformatter import FrameReformatter

    layout = compute_canvas_layout(config)
    reformatter = FrameReformatter()
    canvas_filter = reformatter.build_canvas_filter(src_w, src_h, layout)

    filter_str = canvas_filter.filter_str
    scale_part = filter_str.split("scale=")[1].split(",")[0]
    scale_w_str, scale_h_str = scale_part.split(":")
    scale_w = int(scale_w_str)
    scale_h = int(scale_h_str)

    assert scale_w <= layout.gameplay_width, (
        f"Scaled width {scale_w} exceeds gameplay_width {layout.gameplay_width} "
        f"(src={src_w}x{src_h})"
    )
    assert scale_h <= layout.gameplay_height, (
        f"Scaled height {scale_h} exceeds gameplay_height {layout.gameplay_height} "
        f"(src={src_w}x{src_h})"
    )


@settings(max_examples=200)
@given(
    region=valid_facecam_region(),
    config=canvas_config(),
)
def test_property_13_facecam_scale_fits_within_facecam_region(region, config):
    """
    Property 13: For any facecam region, the scaled facecam content in the
    vertical filter must fit within the facecam region of the canvas.

    **Validates: Requirements 14.1, 14.4**
    """
    assume(region.width > 0 and region.height > 0)

    layout = compute_canvas_layout(config)
    src_w, src_h = 1920, 1080

    filter_str = _build_vertical_filter(src_w, src_h, region, layout)

    # Parse the facecam scale from the filter string
    # Format: "...[0:v]crop=W:H:X:Y,scale=SW:SH[facecam_scaled]..."
    facecam_part = filter_str.split("[facecam_scaled]")[0]
    scale_part = facecam_part.split("scale=")[-1]
    scale_w_str, scale_h_str = scale_part.split(":")
    scale_w = int(scale_w_str)
    scale_h = int(scale_h_str)

    assert scale_w <= layout.facecam_width, (
        f"Facecam scaled width {scale_w} exceeds facecam_width {layout.facecam_width}"
    )
    assert scale_h <= layout.facecam_height, (
        f"Facecam scaled height {scale_h} exceeds facecam_height {layout.facecam_height}"
    )


@settings(max_examples=200)
@given(
    region=valid_facecam_region(),
    config=canvas_config(),
)
def test_property_13_facecam_scale_preserves_aspect_ratio(region, config):
    """
    Property 13: For any facecam region, the facecam scaling in the vertical
    filter must preserve the source facecam aspect ratio (no stretching).

    **Validates: Requirements 14.1, 14.4**
    """
    assume(region.width > 0 and region.height > 0)

    layout = compute_canvas_layout(config)
    src_w, src_h = 1920, 1080

    filter_str = _build_vertical_filter(src_w, src_h, region, layout)

    # Parse the facecam scale from the filter string
    facecam_part = filter_str.split("[facecam_scaled]")[0]
    scale_part = facecam_part.split("scale=")[-1]
    scale_w_str, scale_h_str = scale_part.split(":")
    scale_w = int(scale_w_str)
    scale_h = int(scale_h_str)

    # Aspect ratio must be preserved within rounding tolerance
    original_ratio = region.width / region.height
    scaled_ratio = scale_w / scale_h
    rounding_tolerance = (1.0 / scale_h) + (1.0 / scale_w)
    assert abs(original_ratio - scaled_ratio) <= original_ratio * rounding_tolerance + 0.01, (
        f"Facecam aspect ratio not preserved: original={original_ratio:.4f}, "
        f"scaled={scaled_ratio:.4f} "
        f"(region={region.width}x{region.height}, scaled={scale_w}x{scale_h})"
    )


@settings(max_examples=100)
@given(
    config=canvas_config(),
    src_w=st.integers(min_value=320, max_value=3840),
    src_h=st.integers(min_value=240, max_value=2160),
)
def test_property_13_vertical_source_handled_without_overflow(config, src_w, src_h):
    """
    Property 13: For any source video (including vertical 9:16 sources), the
    gameplay scaling must not overflow the gameplay region.

    This verifies that vertical source videos (9:16) are handled correctly
    alongside horizontal (16:9) and square (1:1) sources.

    **Validates: Requirements 14.1, 14.2, 14.3, 14.4**
    """
    from pipeline.frame_reformatter import FrameReformatter

    layout = compute_canvas_layout(config)
    reformatter = FrameReformatter()
    canvas_filter = reformatter.build_canvas_filter(src_w, src_h, layout)

    filter_str = canvas_filter.filter_str
    scale_part = filter_str.split("scale=")[1].split(",")[0]
    scale_w_str, scale_h_str = scale_part.split(":")
    scale_w = int(scale_w_str)
    scale_h = int(scale_h_str)

    # Scaled content must never exceed the gameplay region
    assert scale_w <= layout.gameplay_width, (
        f"Overflow: scale_w={scale_w} > gameplay_width={layout.gameplay_width} "
        f"(src={src_w}x{src_h})"
    )
    assert scale_h <= layout.gameplay_height, (
        f"Overflow: scale_h={scale_h} > gameplay_height={layout.gameplay_height} "
        f"(src={src_w}x{src_h})"
    )
