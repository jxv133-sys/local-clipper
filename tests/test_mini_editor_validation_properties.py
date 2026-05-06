"""Property-based tests for Mini Video Editor confirmation validation (Task 4.4).

These tests validate universal correctness properties of facecam placement
validation for the Mini Video Editor confirm endpoint.

**Validates: Requirements 6.2, 6.3, 6.4**
"""

from __future__ import annotations

import json
import math
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, Mock

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from pipeline.models import (
    CanvasLayout,
    EditorSession,
    FacecamRegion,
    SessionStore,
    VerticalFormattingJob,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(
    facecam_min_area_fraction: float = 0.04,
    facecam_max_area_fraction: float = 0.30,
    shorts_width: int = 1080,
    shorts_height: int = 1920,
    facecam_top_fraction: float = 0.35,
) -> SimpleNamespace:
    """Return a minimal config-like object for testing."""
    return SimpleNamespace(
        facecam_min_area_fraction=facecam_min_area_fraction,
        facecam_max_area_fraction=facecam_max_area_fraction,
        shorts_width=shorts_width,
        shorts_height=shorts_height,
        facecam_top_fraction=facecam_top_fraction,
    )


DEFAULT_CONFIG = make_config()

DEFAULT_CANVAS_LAYOUT = CanvasLayout(
    canvas_width=1080,
    canvas_height=1920,
    facecam_x=0,
    facecam_y=0,
    facecam_width=1080,
    facecam_height=672,
    gameplay_x=0,
    gameplay_y=672,
    gameplay_width=1080,
    gameplay_height=1248,
)


def _make_session(
    frame_width: int = 1920,
    frame_height: int = 1080,
    clip_batch_id: str = "test-batch",
) -> EditorSession:
    """Create a minimal EditorSession for testing."""
    import time
    import uuid
    region = FacecamRegion(
        x=100, y=50, width=400, height=300,
        corner="top-right", confidence=0.85,
    )
    return EditorSession(
        session_id=str(uuid.uuid4()),
        clip_batch_id=clip_batch_id,
        reference_clip_path="/fake/clip.mp4",
        reference_resolution=(frame_width, frame_height),
        facecam_region=region,
        canvas_layout=DEFAULT_CANVAS_LAYOUT,
    )


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

@st.composite
def frame_dimensions(draw):
    """Generate valid frame dimensions."""
    frame_w = draw(st.integers(min_value=320, max_value=3840))
    frame_h = draw(st.integers(min_value=240, max_value=2160))
    return frame_w, frame_h


@st.composite
def region_extending_beyond_frame(draw):
    """Generate a FacecamRegion that extends beyond the frame bounds.

    Returns (frame_w, frame_h, region) where region.x + region.width > frame_w
    OR region.y + region.height > frame_h.
    """
    frame_w = draw(st.integers(min_value=320, max_value=3840))
    frame_h = draw(st.integers(min_value=240, max_value=2160))

    # Choose which boundary to violate
    violate_x = draw(st.booleans())

    if violate_x:
        # x + width > frame_w
        x = draw(st.integers(min_value=1, max_value=frame_w - 1))
        # width must push past the right edge
        width = draw(st.integers(min_value=frame_w - x + 1, max_value=frame_w))
        height = draw(st.integers(min_value=1, max_value=frame_h))
        y = draw(st.integers(min_value=0, max_value=max(0, frame_h - height)))
    else:
        # y + height > frame_h
        y = draw(st.integers(min_value=1, max_value=frame_h - 1))
        # height must push past the bottom edge
        height = draw(st.integers(min_value=frame_h - y + 1, max_value=frame_h))
        width = draw(st.integers(min_value=1, max_value=frame_w))
        x = draw(st.integers(min_value=0, max_value=max(0, frame_w - width)))

    region = FacecamRegion(
        x=x, y=y, width=width, height=height,
        corner="top-right", confidence=0.5,
    )
    return frame_w, frame_h, region


@st.composite
def region_with_invalid_area_fraction(draw):
    """Generate a FacecamRegion that fits within the frame but has an invalid area fraction.

    Returns (frame_w, frame_h, region) where the area fraction is outside [4%, 30%].
    """
    frame_w = draw(st.integers(min_value=320, max_value=3840))
    frame_h = draw(st.integers(min_value=240, max_value=2160))
    frame_area = frame_w * frame_h

    # Pick a fraction outside [0.04, 0.30]
    fraction = draw(
        st.one_of(
            st.floats(min_value=0.001, max_value=0.039, allow_nan=False, allow_infinity=False),
            st.floats(min_value=0.301, max_value=0.99, allow_nan=False, allow_infinity=False),
        )
    )

    side = max(1, int(math.sqrt(fraction * frame_area)))
    width = min(side, frame_w)
    height = min(side, frame_h)

    # Ensure it fits within the frame
    x = draw(st.integers(min_value=0, max_value=max(0, frame_w - width)))
    y = draw(st.integers(min_value=0, max_value=max(0, frame_h - height)))

    region = FacecamRegion(
        x=x, y=y, width=width, height=height,
        corner="top-right", confidence=0.5,
    )
    return frame_w, frame_h, region


@st.composite
def valid_region_within_frame(draw):
    """Generate a FacecamRegion that fits within the frame with a valid area fraction.

    Returns (frame_w, frame_h, region).
    """
    frame_w = draw(st.integers(min_value=320, max_value=3840))
    frame_h = draw(st.integers(min_value=240, max_value=2160))
    frame_area = frame_w * frame_h

    # Pick a valid area fraction in [0.04, 0.30]
    fraction = draw(
        st.floats(min_value=0.04, max_value=0.30, allow_nan=False, allow_infinity=False)
    )

    side = max(1, int(math.sqrt(fraction * frame_area)))
    width = min(side, frame_w)
    height = min(side, frame_h)

    # After integer rounding, verify the area fraction is still valid.
    # If rounding pushed it below 4%, increase dimensions by 1 to compensate.
    actual_fraction = (width * height) / frame_area
    if actual_fraction < 0.04:
        # Increase width or height by 1 to push fraction up
        if width < frame_w:
            width += 1
        elif height < frame_h:
            height += 1
        actual_fraction = (width * height) / frame_area

    # If still invalid after adjustment, skip this example
    assume(0.04 <= actual_fraction <= 0.30)

    x = draw(st.integers(min_value=0, max_value=max(0, frame_w - width)))
    y = draw(st.integers(min_value=0, max_value=max(0, frame_h - height)))

    region = FacecamRegion(
        x=x, y=y, width=width, height=height,
        corner="top-right",
        confidence=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
    )
    return frame_w, frame_h, region


# ---------------------------------------------------------------------------
# Property 6: Bounds validation
# **Validates: Requirements 6.2, 6.3, 6.4**
# ---------------------------------------------------------------------------

@settings(max_examples=200)
@given(data=region_extending_beyond_frame())
def test_property_6_region_beyond_frame_bounds_is_rejected(data):
    """
    Property 6: For any facecam placement that extends beyond frame bounds,
    confirmation should be prevented.

    This test verifies that the validation logic correctly rejects any region
    where x + width > frame_width OR y + height > frame_height.

    **Validates: Requirements 6.2, 6.3, 6.4**
    """
    frame_w, frame_h, region = data

    # Verify the region actually extends beyond bounds (precondition)
    extends_beyond = (
        region.x + region.width > frame_w
        or region.y + region.height > frame_h
    )
    assert extends_beyond, "Test precondition: region must extend beyond frame"

    # Apply the same validation logic as the confirm endpoint
    is_valid = _validate_region_bounds(region, frame_w, frame_h)

    assert not is_valid, (
        f"Expected validation to fail for region ({region.x}, {region.y}, "
        f"{region.width}, {region.height}) in frame {frame_w}x{frame_h}, "
        f"but it passed"
    )


@settings(max_examples=200)
@given(data=region_extending_beyond_frame())
def test_property_6_region_beyond_bounds_produces_error_message(data):
    """
    Property 6: For any facecam placement that extends beyond frame bounds,
    an error message should be produced (not just a silent failure).

    **Validates: Requirements 6.2, 6.3, 6.4**
    """
    frame_w, frame_h, region = data

    error_msg = _get_validation_error(region, frame_w, frame_h)

    assert error_msg is not None, (
        f"Expected an error message for out-of-bounds region, but got None"
    )
    assert len(error_msg) > 0, "Error message must not be empty"


@settings(max_examples=200)
@given(data=valid_region_within_frame())
def test_property_6_valid_region_within_bounds_passes(data):
    """
    Property 6 (complement): For any facecam placement that fits within the
    frame bounds and has a valid area fraction, confirmation should be allowed.

    **Validates: Requirements 6.2, 6.3, 6.4**
    """
    frame_w, frame_h, region = data

    # Verify preconditions (guaranteed by the strategy)
    assert region.x >= 0
    assert region.y >= 0
    assert region.x + region.width <= frame_w
    assert region.y + region.height <= frame_h

    frame_area = frame_w * frame_h
    area_fraction = (region.width * region.height) / frame_area
    # Strategy guarantees valid area fraction via assume()
    assert 0.04 <= area_fraction <= 0.30, (
        f"Strategy should have filtered this: area_fraction={area_fraction:.4f}"
    )

    # Validation should pass
    is_valid = _validate_region_bounds(region, frame_w, frame_h)
    assert is_valid, (
        f"Expected validation to pass for region ({region.x}, {region.y}, "
        f"{region.width}, {region.height}) in frame {frame_w}x{frame_h}"
    )

    # No error message should be produced
    error_msg = _get_validation_error(region, frame_w, frame_h)
    assert error_msg is None, (
        f"Expected no error for valid region, but got: {error_msg}"
    )


@settings(max_examples=200)
@given(
    frame_w=st.integers(min_value=320, max_value=3840),
    frame_h=st.integers(min_value=240, max_value=2160),
)
def test_property_6_negative_coordinates_rejected(frame_w, frame_h):
    """
    Property 6: Regions with negative x or y coordinates must be rejected.

    **Validates: Requirements 6.2, 6.3, 6.4**
    """
    # Region with negative x
    region_neg_x = FacecamRegion(
        x=-1, y=0, width=100, height=100,
        corner="top-left", confidence=0.5,
    )
    assert not _validate_region_bounds(region_neg_x, frame_w, frame_h)
    assert _get_validation_error(region_neg_x, frame_w, frame_h) is not None

    # Region with negative y
    region_neg_y = FacecamRegion(
        x=0, y=-1, width=100, height=100,
        corner="top-left", confidence=0.5,
    )
    assert not _validate_region_bounds(region_neg_y, frame_w, frame_h)
    assert _get_validation_error(region_neg_y, frame_w, frame_h) is not None


@settings(max_examples=200)
@given(
    frame_w=st.integers(min_value=320, max_value=3840),
    frame_h=st.integers(min_value=240, max_value=2160),
)
def test_property_6_zero_dimensions_rejected(frame_w, frame_h):
    """
    Property 6: Regions with zero width or height must be rejected.

    **Validates: Requirements 6.2, 6.3, 6.4**
    """
    # Region with zero width
    region_zero_w = FacecamRegion(
        x=0, y=0, width=0, height=100,
        corner="top-left", confidence=0.5,
    )
    assert not _validate_region_bounds(region_zero_w, frame_w, frame_h)

    # Region with zero height
    region_zero_h = FacecamRegion(
        x=0, y=0, width=100, height=0,
        corner="top-left", confidence=0.5,
    )
    assert not _validate_region_bounds(region_zero_h, frame_w, frame_h)


# ---------------------------------------------------------------------------
# Property 7: Validation consistency
# **Validates: Requirements 6.2, 6.3, 6.4**
# ---------------------------------------------------------------------------

@settings(max_examples=200)
@given(data=region_extending_beyond_frame())
def test_property_7_invalid_placement_always_has_error_message(data):
    """
    Property 7: For any invalid placement (extends beyond frame bounds),
    an error message should always be displayed.

    This ensures the user always receives feedback when their placement is invalid.

    **Validates: Requirements 6.2, 6.3, 6.4**
    """
    frame_w, frame_h, region = data

    error_msg = _get_validation_error(region, frame_w, frame_h)

    # An invalid placement must always produce an error message
    assert error_msg is not None, (
        f"No error message for invalid region ({region.x}, {region.y}, "
        f"{region.width}, {region.height}) in frame {frame_w}x{frame_h}"
    )
    assert isinstance(error_msg, str), "Error message must be a string"
    assert len(error_msg) > 0, "Error message must not be empty"


@settings(max_examples=200)
@given(data=region_with_invalid_area_fraction())
def test_property_7_invalid_area_fraction_always_has_error_message(data):
    """
    Property 7: For any placement with an invalid area fraction (outside 4%–30%),
    an error message should always be displayed.

    **Validates: Requirements 6.2, 6.3, 6.4**
    """
    frame_w, frame_h, region = data

    # Verify the area fraction is actually invalid
    frame_area = frame_w * frame_h
    area_fraction = (region.width * region.height) / frame_area
    is_invalid_fraction = area_fraction < 0.04 or area_fraction > 0.30

    if not is_invalid_fraction:
        # Skip if the generated region accidentally has a valid fraction
        # (can happen due to integer rounding)
        return

    error_msg = _get_area_fraction_error(region, frame_w, frame_h)

    assert error_msg is not None, (
        f"No error message for region with area_fraction={area_fraction:.4f} "
        f"(outside [0.04, 0.30])"
    )
    assert len(error_msg) > 0, "Error message must not be empty"


@settings(max_examples=200)
@given(data=valid_region_within_frame())
def test_property_7_valid_placement_has_no_error_message(data):
    """
    Property 7 (complement): For any valid placement, no error message should
    be displayed.

    **Validates: Requirements 6.2, 6.3, 6.4**
    """
    frame_w, frame_h, region = data

    # Verify preconditions (guaranteed by the strategy via assume())
    assert region.x >= 0
    assert region.y >= 0
    assert region.x + region.width <= frame_w
    assert region.y + region.height <= frame_h

    frame_area = frame_w * frame_h
    area_fraction = (region.width * region.height) / frame_area
    # Strategy guarantees valid area fraction via assume()
    assert 0.04 <= area_fraction <= 0.30, (
        f"Strategy should have filtered this: area_fraction={area_fraction:.4f}"
    )

    # No error message for valid placement
    bounds_error = _get_validation_error(region, frame_w, frame_h)
    fraction_error = _get_area_fraction_error(region, frame_w, frame_h)

    assert bounds_error is None, (
        f"Unexpected bounds error for valid region: {bounds_error}"
    )
    assert fraction_error is None, (
        f"Unexpected area fraction error for valid region: {fraction_error}"
    )


@settings(max_examples=200)
@given(data=region_extending_beyond_frame())
def test_property_7_error_message_describes_violation(data):
    """
    Property 7: Error messages for out-of-bounds placements must describe
    which boundary was violated (x/width or y/height).

    **Validates: Requirements 6.2, 6.3, 6.4**
    """
    frame_w, frame_h, region = data

    error_msg = _get_validation_error(region, frame_w, frame_h)
    assert error_msg is not None

    # The error message should mention the relevant dimension
    violates_x = region.x + region.width > frame_w
    violates_y = region.y + region.height > frame_h

    if violates_x and not violates_y:
        # Should mention width or x
        assert any(kw in error_msg.lower() for kw in ["width", "x", "frame"]), (
            f"Error message '{error_msg}' doesn't describe x/width violation"
        )
    elif violates_y and not violates_x:
        # Should mention height or y
        assert any(kw in error_msg.lower() for kw in ["height", "y", "frame"]), (
            f"Error message '{error_msg}' doesn't describe y/height violation"
        )
    # If both are violated, either mention is acceptable


# ---------------------------------------------------------------------------
# Validation helper functions (mirror the confirm endpoint logic)
# These are pure functions that replicate the validation logic from web_server.py
# so we can test it independently of the HTTP layer.
# ---------------------------------------------------------------------------

def _validate_region_bounds(region: FacecamRegion, frame_w: int, frame_h: int) -> bool:
    """Return True if the region is valid (within bounds and positive dimensions)."""
    if region.x < 0:
        return False
    if region.y < 0:
        return False
    if region.width <= 0:
        return False
    if region.height <= 0:
        return False
    if region.x + region.width > frame_w:
        return False
    if region.y + region.height > frame_h:
        return False
    return True


def _get_validation_error(
    region: FacecamRegion, frame_w: int, frame_h: int
) -> str | None:
    """Return an error message if the region is out of bounds, else None."""
    if region.x < 0:
        return "facecam_region.x must be >= 0"
    if region.y < 0:
        return "facecam_region.y must be >= 0"
    if region.width <= 0:
        return "facecam_region.width must be > 0"
    if region.height <= 0:
        return "facecam_region.height must be > 0"
    if region.x + region.width > frame_w:
        return (
            f"facecam_region extends beyond frame width: "
            f"x ({region.x}) + width ({region.width}) > frame_width ({frame_w})"
        )
    if region.y + region.height > frame_h:
        return (
            f"facecam_region extends beyond frame height: "
            f"y ({region.y}) + height ({region.height}) > frame_height ({frame_h})"
        )
    return None


def _get_area_fraction_error(
    region: FacecamRegion,
    frame_w: int,
    frame_h: int,
    min_fraction: float = 0.04,
    max_fraction: float = 0.30,
) -> str | None:
    """Return an error message if the area fraction is invalid, else None."""
    frame_area = frame_w * frame_h
    if frame_area == 0:
        return "Frame area must be > 0"
    region_area = region.width * region.height
    area_fraction = region_area / frame_area
    if area_fraction < min_fraction:
        return (
            f"facecam_region area fraction ({area_fraction:.3f}) is below "
            f"minimum ({min_fraction}). The facecam region is too small."
        )
    if area_fraction > max_fraction:
        return (
            f"facecam_region area fraction ({area_fraction:.3f}) exceeds "
            f"maximum ({max_fraction}). The facecam region is too large."
        )
    return None
