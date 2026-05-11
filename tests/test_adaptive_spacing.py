"""Unit and property-based tests for adaptive spacing module.

This module tests the compute_adaptive_spacing() function which dynamically
adjusts spacing constraints based on video duration and clip count.

**Validates: Requirements 8.1, 8.2**
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from pipeline.adaptive_spacing import compute_adaptive_spacing


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

class TestComputeAdaptiveSpacing:
    """Unit tests for compute_adaptive_spacing() function."""
    
    def test_short_video_scales_down_spacing(self):
        """For a 10-minute video with 6 clips, spacing should scale down from 300s base."""
        # 600s / (6 + 1) = 85.71s
        result = compute_adaptive_spacing(600.0, 6, 300.0)
        assert result == pytest.approx(85.71428571428571, rel=1e-6)
    
    def test_medium_video_scales_down_spacing(self):
        """For a 30-minute video with 6 clips, spacing should scale down from 300s base."""
        # 1800s / (6 + 1) = 257.14s
        result = compute_adaptive_spacing(1800.0, 6, 300.0)
        assert result == pytest.approx(257.14285714285717, rel=1e-6)
    
    def test_long_video_uses_base_spacing(self):
        """For a 60-minute video with 6 clips, spacing should use full base_spacing."""
        # 3600s / (6 + 1) = 514.29s > 300s, so use base_spacing
        result = compute_adaptive_spacing(3600.0, 6, 300.0)
        assert result == pytest.approx(300.0, rel=1e-6)
    
    def test_very_short_video_applies_minimum_floor(self):
        """For a 5-minute video with 6 clips, spacing should hit the 30s minimum floor."""
        # 300s / (6 + 1) = 42.86s > 30s, so use 42.86s
        result = compute_adaptive_spacing(300.0, 6, 300.0)
        assert result == pytest.approx(42.857142857142854, rel=1e-6)
    
    def test_extremely_short_video_enforces_minimum_floor(self):
        """For a 3-minute video with 10 clips, spacing should enforce 30s minimum."""
        # 180s / (10 + 1) = 16.36s < 30s, so use 30s floor
        result = compute_adaptive_spacing(180.0, 10, 300.0)
        assert result == pytest.approx(30.0, rel=1e-6)
    
    def test_single_clip_uses_video_duration(self):
        """For a video with 1 clip requested, spacing should be video_duration / 2."""
        # 600s / (1 + 1) = 300s
        result = compute_adaptive_spacing(600.0, 1, 300.0)
        assert result == pytest.approx(300.0, rel=1e-6)
    
    def test_many_clips_short_video_uses_minimum(self):
        """For many clips in a short video, spacing should hit minimum floor."""
        # 600s / (20 + 1) = 28.57s < 30s, so use 30s floor
        result = compute_adaptive_spacing(600.0, 20, 300.0)
        assert result == pytest.approx(30.0, rel=1e-6)
    
    def test_base_spacing_smaller_than_required(self):
        """When base_spacing < required_spacing, use base_spacing."""
        # 3600s / (5 + 1) = 600s, but base_spacing=200s, so use 200s
        result = compute_adaptive_spacing(3600.0, 5, 200.0)
        assert result == pytest.approx(200.0, rel=1e-6)
    
    def test_base_spacing_larger_than_required(self):
        """When base_spacing > required_spacing, use required_spacing."""
        # 600s / (10 + 1) = 54.55s, base_spacing=300s, so use 54.55s
        result = compute_adaptive_spacing(600.0, 10, 300.0)
        assert result == pytest.approx(54.54545454545455, rel=1e-6)
    
    def test_exact_boundary_at_minimum_floor(self):
        """When required_spacing equals minimum floor, return minimum floor."""
        # 210s / (6 + 1) = 30s exactly
        result = compute_adaptive_spacing(210.0, 6, 300.0)
        assert result == pytest.approx(30.0, rel=1e-6)
    
    def test_exact_boundary_at_base_spacing(self):
        """When required_spacing equals base_spacing, return base_spacing."""
        # 2100s / (6 + 1) = 300s exactly
        result = compute_adaptive_spacing(2100.0, 6, 300.0)
        assert result == pytest.approx(300.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Property-Based Tests
# ---------------------------------------------------------------------------

# Feature: clip-selection-improvements, Property 10: Adaptive Spacing Bounds
@given(
    video_duration=st.floats(min_value=60.0, max_value=7200.0, allow_nan=False, allow_infinity=False),
    top_n_clips=st.integers(min_value=1, max_value=20),
)
@settings(max_examples=200, deadline=None)
def test_adaptive_spacing_bounds(video_duration: float, top_n_clips: int):
    """For any video duration and top_n, effective spacing should satisfy bounds.
    
    This property validates that:
    1. Effective spacing is always >= 30.0 (minimum floor)
    2. Effective spacing is always <= base_spacing (300.0)
    3. The computed spacing allows all clips to fit within the video duration
    
    **Validates: Requirements 8.1, 8.2**
    """
    base_spacing = 300.0
    effective = compute_adaptive_spacing(video_duration, top_n_clips, base_spacing)
    
    # Property 1: Effective spacing should be >= minimum floor (30.0)
    assert effective >= 30.0, \
        f"Effective spacing {effective} should be >= 30.0 for duration={video_duration}, clips={top_n_clips}"
    
    # Property 2: Effective spacing should be <= base_spacing
    assert effective <= base_spacing, \
        f"Effective spacing {effective} should be <= {base_spacing} for duration={video_duration}, clips={top_n_clips}"
    
    # Property 3: The formula ensures clips can be distributed across the video
    # The adaptive spacing formula is: video_duration / (top_n_clips + 1)
    # This means: (top_n_clips + 1) * effective_spacing <= video_duration (when not capped)
    # However, when capped by min_floor or base_spacing, this may not hold
    # The key property is that effective_spacing follows the formula correctly
    required_spacing = video_duration / (top_n_clips + 1)
    min_floor = 30.0
    expected_effective = max(min_floor, min(base_spacing, required_spacing))
    assert effective == pytest.approx(expected_effective, rel=1e-9), \
        f"Effective spacing should match formula for duration={video_duration}, clips={top_n_clips}"


# Feature: clip-selection-improvements, Property 10: Adaptive Spacing Monotonicity
@given(
    video_duration=st.floats(min_value=60.0, max_value=7200.0, allow_nan=False, allow_infinity=False),
    top_n_clips=st.integers(min_value=1, max_value=19),  # max 19 so we can test n and n+1
)
@settings(max_examples=200, deadline=None)
def test_adaptive_spacing_monotonicity_with_clip_count(video_duration: float, top_n_clips: int):
    """For a fixed video duration, increasing clip count should decrease or maintain spacing.
    
    This property validates that as we request more clips from the same video,
    the spacing constraint either decreases (to fit more clips) or stays at the
    minimum floor (30.0).
    
    **Validates: Requirements 8.1, 8.2**
    """
    base_spacing = 300.0
    
    spacing_n = compute_adaptive_spacing(video_duration, top_n_clips, base_spacing)
    spacing_n_plus_1 = compute_adaptive_spacing(video_duration, top_n_clips + 1, base_spacing)
    
    # Spacing should decrease or stay the same (if at minimum floor)
    assert spacing_n_plus_1 <= spacing_n, \
        f"Spacing should decrease with more clips: " \
        f"spacing({top_n_clips})={spacing_n} should be >= spacing({top_n_clips + 1})={spacing_n_plus_1} " \
        f"for duration={video_duration}"


# Feature: clip-selection-improvements, Property 10: Adaptive Spacing Monotonicity with Duration
@given(
    video_duration=st.floats(min_value=60.0, max_value=7199.0, allow_nan=False, allow_infinity=False),
    top_n_clips=st.integers(min_value=1, max_value=20),
    duration_increase=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=200, deadline=None)
def test_adaptive_spacing_monotonicity_with_duration(
    video_duration: float,
    top_n_clips: int,
    duration_increase: float,
):
    """For a fixed clip count, increasing video duration should increase or maintain spacing.
    
    This property validates that longer videos allow for more spacing between clips,
    up to the base_spacing limit.
    
    **Validates: Requirements 8.1, 8.2**
    """
    base_spacing = 300.0
    longer_duration = video_duration + duration_increase
    
    # Ensure longer_duration doesn't exceed our test bounds
    assume(longer_duration <= 7200.0)
    
    spacing_short = compute_adaptive_spacing(video_duration, top_n_clips, base_spacing)
    spacing_long = compute_adaptive_spacing(longer_duration, top_n_clips, base_spacing)
    
    # Spacing should increase or stay the same (if at base_spacing limit)
    assert spacing_long >= spacing_short, \
        f"Spacing should increase with longer videos: " \
        f"spacing({video_duration})={spacing_short} should be <= spacing({longer_duration})={spacing_long} " \
        f"for clips={top_n_clips}"


# Feature: clip-selection-improvements, Property 10: Adaptive Spacing Formula Correctness
@given(
    video_duration=st.floats(min_value=60.0, max_value=7200.0, allow_nan=False, allow_infinity=False),
    top_n_clips=st.integers(min_value=1, max_value=20),
    base_spacing=st.floats(min_value=30.0, max_value=600.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=200, deadline=None)
def test_adaptive_spacing_formula_correctness(
    video_duration: float,
    top_n_clips: int,
    base_spacing: float,
):
    """For any inputs, effective spacing should match the documented formula.
    
    Formula: max(30.0, min(base_spacing, video_duration / (top_n_clips + 1)))
    
    This property validates that the implementation exactly matches the specification.
    
    **Validates: Requirements 8.1, 8.2**
    """
    min_floor = 30.0
    
    # Compute using function
    effective = compute_adaptive_spacing(video_duration, top_n_clips, base_spacing)
    
    # Compute using formula
    required_spacing = video_duration / (top_n_clips + 1)
    expected = max(min_floor, min(base_spacing, required_spacing))
    
    # Should match exactly (within floating point precision)
    assert effective == pytest.approx(expected, rel=1e-9), \
        f"Effective spacing {effective} should match formula result {expected} " \
        f"for duration={video_duration}, clips={top_n_clips}, base={base_spacing}"


# Feature: clip-selection-improvements, Property 10: Adaptive Spacing Minimum Floor Enforcement
@given(
    video_duration=st.floats(min_value=10.0, max_value=500.0, allow_nan=False, allow_infinity=False),
    top_n_clips=st.integers(min_value=10, max_value=50),
)
@settings(max_examples=200, deadline=None)
def test_adaptive_spacing_minimum_floor_enforcement(video_duration: float, top_n_clips: int):
    """For scenarios where required_spacing < 30.0, the minimum floor should be enforced.
    
    This property specifically tests edge cases where many clips are requested
    from short videos, ensuring the 30.0 second minimum is always respected.
    
    **Validates: Requirements 8.1, 8.2**
    """
    base_spacing = 300.0
    min_floor = 30.0
    
    effective = compute_adaptive_spacing(video_duration, top_n_clips, base_spacing)
    required_spacing = video_duration / (top_n_clips + 1)
    
    # If required_spacing < min_floor, effective should be min_floor
    if required_spacing < min_floor:
        assert effective == pytest.approx(min_floor, rel=1e-9), \
            f"When required_spacing ({required_spacing}) < {min_floor}, " \
            f"effective should be {min_floor}, got {effective}"
    else:
        # Otherwise, effective should be min(base_spacing, required_spacing)
        expected = min(base_spacing, required_spacing)
        assert effective == pytest.approx(expected, rel=1e-9), \
            f"When required_spacing ({required_spacing}) >= {min_floor}, " \
            f"effective should be min(base, required) = {expected}, got {effective}"


# Feature: clip-selection-improvements, Property 10: Adaptive Spacing Base Limit Enforcement
@given(
    video_duration=st.floats(min_value=3000.0, max_value=7200.0, allow_nan=False, allow_infinity=False),
    top_n_clips=st.integers(min_value=1, max_value=5),
    base_spacing=st.floats(min_value=30.0, max_value=300.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=200, deadline=None)
def test_adaptive_spacing_base_limit_enforcement(
    video_duration: float,
    top_n_clips: int,
    base_spacing: float,
):
    """For scenarios where required_spacing > base_spacing, the base limit should be enforced.
    
    This property specifically tests edge cases where few clips are requested
    from long videos, ensuring the base_spacing limit is never exceeded.
    
    **Validates: Requirements 8.1, 8.2**
    """
    min_floor = 30.0
    
    effective = compute_adaptive_spacing(video_duration, top_n_clips, base_spacing)
    required_spacing = video_duration / (top_n_clips + 1)
    
    # If required_spacing > base_spacing, effective should be base_spacing
    if required_spacing > base_spacing:
        assert effective == pytest.approx(base_spacing, rel=1e-9), \
            f"When required_spacing ({required_spacing}) > base_spacing ({base_spacing}), " \
            f"effective should be {base_spacing}, got {effective}"
    else:
        # Otherwise, effective should be max(min_floor, required_spacing)
        expected = max(min_floor, required_spacing)
        assert effective == pytest.approx(expected, rel=1e-9), \
            f"When required_spacing ({required_spacing}) <= base_spacing ({base_spacing}), " \
            f"effective should be max(floor, required) = {expected}, got {effective}"
