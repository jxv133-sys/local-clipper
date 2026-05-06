"""Property-based tests for Mini Video Editor facecam detection (Task 2.3).

These tests validate universal correctness properties of facecam detection
for the Mini Video Editor feature.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from hypothesis import given, settings
from hypothesis import strategies as st

from pipeline.facecam_relocator import FacecamRelocator, classify_region
from pipeline.models import FacecamRegion


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
# Strategies
# ---------------------------------------------------------------------------

@st.composite
def region_with_area_fraction(draw, min_frac: float, max_frac: float):
    """Generate (frame_w, frame_h, x, y, w, h) with area fraction in [min_frac, max_frac]."""
    frame_w = draw(st.integers(min_value=100, max_value=3840))
    frame_h = draw(st.integers(min_value=100, max_value=2160))
    frame_area = frame_w * frame_h

    # Pick a fraction in the specified range
    fraction = draw(st.floats(min_value=min_frac, max_value=max_frac, allow_nan=False, allow_infinity=False))

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
def invalid_area_fraction_region(draw):
    """Generate (frame_w, frame_h, x, y, w, h) with area fraction outside [0.04, 0.30]."""
    frame_w = draw(st.integers(min_value=100, max_value=3840))
    frame_h = draw(st.integers(min_value=100, max_value=2160))
    frame_area = frame_w * frame_h

    # Pick a fraction outside [0.04, 0.30]
    fraction = draw(
        st.one_of(
            st.floats(min_value=0.0001, max_value=0.0399, allow_nan=False, allow_infinity=False),
            st.floats(min_value=0.3001, max_value=0.99, allow_nan=False, allow_infinity=False),
        )
    )

    # Derive w, h from fraction
    import math
    side = max(1, int(math.sqrt(fraction * frame_area)))
    w = min(side, frame_w)
    h = min(side, frame_h)

    # Place region so it fits within the frame
    x = draw(st.integers(min_value=0, max_value=max(0, frame_w - w)))
    y = draw(st.integers(min_value=0, max_value=max(0, frame_h - h)))

    return frame_w, frame_h, x, y, w, h


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
    noise_count = draw(st.integers(min_value=0, max_value=max(0, dominant_count - 1)))
    noise_crops = [(100, 100, 0, 0)] * noise_count  # small, invalid area fraction

    crops = [(dominant_w, dominant_h, dominant_x, dominant_y)] * dominant_count + noise_crops
    return crops


def _build_cropdetect_stderr(crops: list[tuple[int, int, int, int]]) -> str:
    """Build a fake ffmpeg stderr string from a list of (w, h, x, y) crop tuples."""
    lines = []
    for w, h, x, y in crops:
        lines.append(f"crop={w}:{h}:{x}:{y}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Property 1: Area fraction validation
# **Validates: Requirements 3.3, 3.4, 3.5**
# ---------------------------------------------------------------------------

@settings(max_examples=200)
@given(data=invalid_area_fraction_region())
def test_property_1_area_fraction_outside_bounds_rejected(data):
    """
    Property 1: For any detected region, if its area fraction is outside [4%, 30%],
    it should be rejected.

    This test verifies that classify_region returns None for regions with area
    fractions outside the valid range [0.04, 0.30].

    **Validates: Requirements 3.3, 3.4, 3.5**
    """
    frame_w, frame_h, x, y, w, h = data
    result = classify_region(
        x=x, y=y, w=w, h=h,
        frame_w=frame_w, frame_h=frame_h,
        min_area_fraction=0.04,
        max_area_fraction=0.30,
    )
    
    area_fraction = (w * h) / (frame_w * frame_h)
    
    # If area fraction is outside [0.04, 0.30], result must be None
    if area_fraction < 0.04 or area_fraction > 0.30:
        assert result is None, (
            f"Expected None for area_fraction={area_fraction:.4f} "
            f"(outside [0.04, 0.30]), but got {result!r}"
        )


@settings(max_examples=200)
@given(data=region_with_area_fraction(min_frac=0.04, max_frac=0.30))
def test_property_1_area_fraction_within_bounds_accepted(data):
    """
    Property 1 (complement): For any detected region, if its area fraction is
    within [4%, 30%], it should be accepted (not None).

    This test verifies that classify_region returns a valid corner string for
    regions with area fractions within the valid range [0.04, 0.30].

    **Validates: Requirements 3.3, 3.4, 3.5**
    """
    frame_w, frame_h, x, y, w, h = data
    result = classify_region(
        x=x, y=y, w=w, h=h,
        frame_w=frame_w, frame_h=frame_h,
        min_area_fraction=0.04,
        max_area_fraction=0.30,
    )
    
    area_fraction = (w * h) / (frame_w * frame_h)
    
    # If area fraction is within [0.04, 0.30], result must be a valid corner
    if 0.04 <= area_fraction <= 0.30:
        valid_corners = {"top-left", "top-right", "bottom-left", "bottom-right"}
        assert result in valid_corners, (
            f"Expected a corner string for area_fraction={area_fraction:.4f} "
            f"(within [0.04, 0.30]), but got {result!r}"
        )


# ---------------------------------------------------------------------------
# Property 2: Confidence bounds
# **Validates: Requirements 3.3, 3.4, 3.5**
# ---------------------------------------------------------------------------

@settings(max_examples=200)
@given(crops=valid_cropdetect_output())
def test_property_2_confidence_in_valid_range(crops):
    """
    Property 2: For any set of detected crops, confidence should be in [0.0, 1.0].

    This test verifies that the confidence score returned by detect_facecam is
    always within the valid range [0.0, 1.0], regardless of the cropdetect output.

    **Validates: Requirements 3.3, 3.4, 3.5**
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

    # If a region is detected, confidence must be in [0.0, 1.0]
    if region is not None:
        assert 0.0 <= region.confidence <= 1.0, (
            f"confidence={region.confidence} is outside [0.0, 1.0]"
        )


@settings(max_examples=100)
@given(
    dominant_count=st.integers(min_value=1, max_value=100),
    noise_count=st.integers(min_value=0, max_value=50),
)
def test_property_2_confidence_reflects_frequency(dominant_count, noise_count):
    """
    Property 2 (extended): Confidence should reflect the frequency of the most
    common crop relative to the total number of crops.

    This test verifies that confidence = dominant_count / total_count.

    **Validates: Requirements 3.3, 3.4, 3.5**
    """
    # Create crops with known frequency distribution
    dominant_crop = (480, 270, 1440, 810)  # valid area fraction
    noise_crop = (100, 100, 0, 0)  # invalid area fraction
    
    crops = [dominant_crop] * dominant_count + [noise_crop] * noise_count
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

    if region is not None:
        total_count = dominant_count + noise_count
        expected_confidence = dominant_count / total_count
        
        # Allow small floating-point tolerance
        assert abs(region.confidence - expected_confidence) < 0.001, (
            f"Expected confidence={expected_confidence:.4f}, "
            f"but got {region.confidence:.4f}"
        )


# ---------------------------------------------------------------------------
# Property 3: Corner classification
# **Validates: Requirements 3.3, 3.4, 3.5**
# ---------------------------------------------------------------------------

@settings(max_examples=200)
@given(data=region_with_area_fraction(min_frac=0.04, max_frac=0.30))
def test_property_3_corner_classification_exactly_one(data):
    """
    Property 3: For any detected region, it should be classified into exactly
    one of four corners.

    This test verifies that classify_region returns exactly one of the four
    valid corner strings: "top-left", "top-right", "bottom-left", "bottom-right".

    **Validates: Requirements 3.3, 3.4, 3.5**
    """
    frame_w, frame_h, x, y, w, h = data
    result = classify_region(
        x=x, y=y, w=w, h=h,
        frame_w=frame_w, frame_h=frame_h,
        min_area_fraction=0.04,
        max_area_fraction=0.30,
    )
    
    area_fraction = (w * h) / (frame_w * frame_h)
    
    # If area fraction is valid, result must be exactly one of the four corners
    if 0.04 <= area_fraction <= 0.30:
        valid_corners = {"top-left", "top-right", "bottom-left", "bottom-right"}
        assert result in valid_corners, (
            f"Expected exactly one corner from {valid_corners}, but got {result!r}"
        )
        
        # Verify it's a string (not None or other type)
        assert isinstance(result, str), (
            f"Expected corner to be a string, but got type {type(result)}"
        )


@settings(max_examples=200)
@given(data=region_with_area_fraction(min_frac=0.04, max_frac=0.30))
def test_property_3_corner_classification_consistent_with_position(data):
    """
    Property 3 (extended): For any detected region, the corner classification
    should be consistent with the region's centre position relative to the
    frame midpoint.

    This test verifies that the corner classification matches the quadrant
    where the region's centre is located.

    **Validates: Requirements 3.3, 3.4, 3.5**
    """
    frame_w, frame_h, x, y, w, h = data
    result = classify_region(
        x=x, y=y, w=w, h=h,
        frame_w=frame_w, frame_h=frame_h,
        min_area_fraction=0.04,
        max_area_fraction=0.30,
    )
    
    area_fraction = (w * h) / (frame_w * frame_h)
    
    # If area fraction is valid, verify corner matches centre position
    if 0.04 <= area_fraction <= 0.30 and result is not None:
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


@settings(max_examples=100)
@given(
    frame_w=st.integers(min_value=100, max_value=3840),
    frame_h=st.integers(min_value=100, max_value=2160),
)
def test_property_3_all_four_corners_possible(frame_w, frame_h):
    """
    Property 3 (extended): Verify that all four corners are possible classifications
    for valid regions in different positions.

    This test creates regions in each quadrant and verifies they are classified
    into the correct corner.

    **Validates: Requirements 3.3, 3.4, 3.5**
    """
    # Create a region with valid area fraction (10% of frame)
    import math
    area_fraction = 0.10
    side = max(1, int(math.sqrt(area_fraction * frame_w * frame_h)))
    w = min(side, frame_w // 2)
    h = min(side, frame_h // 2)
    
    # Test all four corners
    corners_to_test = [
        ("top-left", 0, 0),
        ("top-right", frame_w - w, 0),
        ("bottom-left", 0, frame_h - h),
        ("bottom-right", frame_w - w, frame_h - h),
    ]
    
    for expected_corner, x, y in corners_to_test:
        result = classify_region(
            x=x, y=y, w=w, h=h,
            frame_w=frame_w, frame_h=frame_h,
            min_area_fraction=0.04,
            max_area_fraction=0.30,
        )
        
        # Verify the region is valid and classified correctly
        area_frac = (w * h) / (frame_w * frame_h)
        if 0.04 <= area_frac <= 0.30:
            assert result == expected_corner, (
                f"Expected {expected_corner} for region at ({x}, {y}), "
                f"but got {result!r}"
            )
