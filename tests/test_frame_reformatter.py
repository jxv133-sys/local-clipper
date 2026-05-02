"""Tests for FrameReformatter: unit tests (task 3.4) and property-based tests (task 3.5)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pipeline.frame_reformatter import FrameReformatter, compute_canvas_layout
from pipeline.models import CanvasLayout


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


def default_layout() -> CanvasLayout:
    return compute_canvas_layout(DEFAULT_CONFIG)


# ---------------------------------------------------------------------------
# Task 3.4 — Unit tests for build_canvas_filter
# ---------------------------------------------------------------------------

class TestBuildCanvasFilter:
    """Unit tests for FrameReformatter.build_canvas_filter across common aspect ratios."""

    def setup_method(self):
        self.reformatter = FrameReformatter()
        self.layout = default_layout()

    # --- 16:9 source (1920×1080) ---

    def test_16_9_canvas_dimensions_in_filter_str(self):
        frag = self.reformatter.build_canvas_filter(1920, 1080, self.layout)
        assert "1080" in frag.filter_str
        assert "1920" in frag.filter_str

    def test_16_9_pad_dimensions(self):
        frag = self.reformatter.build_canvas_filter(1920, 1080, self.layout)
        assert "pad=1080:1920" in frag.filter_str

    def test_16_9_black_fill(self):
        frag = self.reformatter.build_canvas_filter(1920, 1080, self.layout)
        assert "black" in frag.filter_str

    def test_16_9_output_label(self):
        frag = self.reformatter.build_canvas_filter(1920, 1080, self.layout)
        assert frag.output_label == "[canvas]"

    def test_16_9_input_label(self):
        frag = self.reformatter.build_canvas_filter(1920, 1080, self.layout)
        assert frag.input_label == "[0:v]"

    # --- 4:3 source (1280×960) ---

    def test_4_3_canvas_dimensions_in_filter_str(self):
        frag = self.reformatter.build_canvas_filter(1280, 960, self.layout)
        assert "1080" in frag.filter_str
        assert "1920" in frag.filter_str

    def test_4_3_pad_dimensions(self):
        frag = self.reformatter.build_canvas_filter(1280, 960, self.layout)
        assert "pad=1080:1920" in frag.filter_str

    def test_4_3_black_fill(self):
        frag = self.reformatter.build_canvas_filter(1280, 960, self.layout)
        assert "black" in frag.filter_str

    def test_4_3_output_label(self):
        frag = self.reformatter.build_canvas_filter(1280, 960, self.layout)
        assert frag.output_label == "[canvas]"

    def test_4_3_input_label(self):
        frag = self.reformatter.build_canvas_filter(1280, 960, self.layout)
        assert frag.input_label == "[0:v]"

    # --- 1:1 source (1080×1080) ---

    def test_1_1_canvas_dimensions_in_filter_str(self):
        frag = self.reformatter.build_canvas_filter(1080, 1080, self.layout)
        assert "1080" in frag.filter_str
        assert "1920" in frag.filter_str

    def test_1_1_pad_dimensions(self):
        frag = self.reformatter.build_canvas_filter(1080, 1080, self.layout)
        assert "pad=1080:1920" in frag.filter_str

    def test_1_1_black_fill(self):
        frag = self.reformatter.build_canvas_filter(1080, 1080, self.layout)
        assert "black" in frag.filter_str

    def test_1_1_output_label(self):
        frag = self.reformatter.build_canvas_filter(1080, 1080, self.layout)
        assert frag.output_label == "[canvas]"

    def test_1_1_input_label(self):
        frag = self.reformatter.build_canvas_filter(1080, 1080, self.layout)
        assert frag.input_label == "[0:v]"

    # --- 9:16 source (1080×1920) ---

    def test_9_16_canvas_dimensions_in_filter_str(self):
        frag = self.reformatter.build_canvas_filter(1080, 1920, self.layout)
        assert "1080" in frag.filter_str
        assert "1920" in frag.filter_str

    def test_9_16_pad_dimensions(self):
        frag = self.reformatter.build_canvas_filter(1080, 1920, self.layout)
        assert "pad=1080:1920" in frag.filter_str

    def test_9_16_black_fill(self):
        frag = self.reformatter.build_canvas_filter(1080, 1920, self.layout)
        assert "black" in frag.filter_str

    def test_9_16_output_label(self):
        frag = self.reformatter.build_canvas_filter(1080, 1920, self.layout)
        assert frag.output_label == "[canvas]"

    def test_9_16_input_label(self):
        frag = self.reformatter.build_canvas_filter(1080, 1920, self.layout)
        assert frag.input_label == "[0:v]"


class TestComputeCanvasLayoutInvariants:
    """Unit tests for compute_canvas_layout invariants."""

    def test_facecam_plus_gameplay_equals_canvas_height(self):
        layout = compute_canvas_layout(DEFAULT_CONFIG)
        assert layout.facecam_height + layout.gameplay_height == layout.canvas_height

    def test_facecam_width_equals_canvas_width(self):
        layout = compute_canvas_layout(DEFAULT_CONFIG)
        assert layout.facecam_width == layout.canvas_width

    def test_gameplay_width_equals_canvas_width(self):
        layout = compute_canvas_layout(DEFAULT_CONFIG)
        assert layout.gameplay_width == layout.canvas_width

    def test_gameplay_y_equals_facecam_height(self):
        layout = compute_canvas_layout(DEFAULT_CONFIG)
        assert layout.gameplay_y == layout.facecam_height

    def test_facecam_x_is_zero(self):
        layout = compute_canvas_layout(DEFAULT_CONFIG)
        assert layout.facecam_x == 0

    def test_facecam_y_is_zero(self):
        layout = compute_canvas_layout(DEFAULT_CONFIG)
        assert layout.facecam_y == 0


# ---------------------------------------------------------------------------
# Task 3.5 — Property-based tests (hypothesis)
# ---------------------------------------------------------------------------

# Strategies
src_dimensions = st.integers(min_value=1, max_value=7680)
src_height_strategy = st.integers(min_value=1, max_value=4320)
facecam_fraction_strategy = st.floats(
    min_value=0.001, max_value=0.999, allow_nan=False, allow_infinity=False
)
positive_dim_strategy = st.integers(min_value=1, max_value=7680)


@settings(max_examples=200)
@given(src_w=src_dimensions, src_h=src_height_strategy)
def test_property_1_canvas_dimensions_always_exact(src_w: int, src_h: int):
    """
    Property 1: For any (src_w, src_h) in [1, 7680] × [1, 4320],
    build_canvas_filter filter_str contains "pad=1080:1920" (canvas dimensions always exact).

    Validates: Requirements 2.2, 2.6
    """
    reformatter = FrameReformatter()
    layout = compute_canvas_layout(DEFAULT_CONFIG)
    frag = reformatter.build_canvas_filter(src_w, src_h, layout)
    assert "pad=1080:1920" in frag.filter_str, (
        f"Expected 'pad=1080:1920' in filter_str for src={src_w}x{src_h}, "
        f"got: {frag.filter_str}"
    )


@settings(max_examples=200)
@given(
    src_w=st.integers(min_value=1, max_value=7680),
    src_h=st.integers(min_value=1, max_value=4320),
    facecam_top_fraction=facecam_fraction_strategy,
)
def test_property_2_gameplay_positioned_below_facecam(
    src_w: int, src_h: int, facecam_top_fraction: float
):
    """
    Property 2: For any positive source dimensions and facecam_top_fraction in (0.0, 1.0),
    the gameplay video's pad_y in the filter_str is >= round(canvas_height * facecam_top_fraction).

    Validates: Requirements 2.3, 2.4, 2.5
    """
    config = make_config(facecam_top_fraction=facecam_top_fraction)
    layout = compute_canvas_layout(config)
    reformatter = FrameReformatter()
    frag = reformatter.build_canvas_filter(src_w, src_h, layout)

    # Extract pad_y from filter_str: "pad=W:H:pad_x:pad_y:black"
    # Format: scale=W:H,pad=CW:CH:PX:PY:black[canvas]
    filter_str = frag.filter_str
    pad_part = filter_str.split("pad=")[1]  # "1080:1920:PX:PY:black[canvas]"
    parts = pad_part.split(":")
    # parts: ["1080", "1920", "PX", "PY", "black[canvas]"]
    pad_y = int(parts[3])

    expected_min_pad_y = round(layout.canvas_height * facecam_top_fraction)
    assert pad_y >= expected_min_pad_y, (
        f"pad_y={pad_y} should be >= {expected_min_pad_y} "
        f"(facecam_top_fraction={facecam_top_fraction}, src={src_w}x{src_h})"
    )


@settings(max_examples=200)
@given(
    shorts_width=positive_dim_strategy,
    shorts_height=positive_dim_strategy,
    facecam_top_fraction=facecam_fraction_strategy,
)
def test_property_3_canvas_layout_invariants(
    shorts_width: int, shorts_height: int, facecam_top_fraction: float
):
    """
    Property 3: For any shorts_width > 0, shorts_height > 0, facecam_top_fraction in (0.0, 1.0),
    compute_canvas_layout satisfies all five invariants.

    Validates: Requirements 8.2, 8.3, 8.4, 8.5
    """
    config = make_config(
        shorts_width=shorts_width,
        shorts_height=shorts_height,
        facecam_top_fraction=facecam_top_fraction,
    )
    layout = compute_canvas_layout(config)

    # Invariant 1: facecam_height + gameplay_height == canvas_height
    assert layout.facecam_height + layout.gameplay_height == layout.canvas_height, (
        f"facecam_height({layout.facecam_height}) + gameplay_height({layout.gameplay_height}) "
        f"!= canvas_height({layout.canvas_height})"
    )

    # Invariant 2: facecam_width == canvas_width
    assert layout.facecam_width == layout.canvas_width, (
        f"facecam_width({layout.facecam_width}) != canvas_width({layout.canvas_width})"
    )

    # Invariant 3: gameplay_width == canvas_width
    assert layout.gameplay_width == layout.canvas_width, (
        f"gameplay_width({layout.gameplay_width}) != canvas_width({layout.canvas_width})"
    )

    # Invariant 4: gameplay_y == facecam_height
    assert layout.gameplay_y == layout.facecam_height, (
        f"gameplay_y({layout.gameplay_y}) != facecam_height({layout.facecam_height})"
    )

    # Invariant 5a: facecam_x == 0
    assert layout.facecam_x == 0, f"facecam_x={layout.facecam_x} != 0"

    # Invariant 5b: facecam_y == 0
    assert layout.facecam_y == 0, f"facecam_y={layout.facecam_y} != 0"
