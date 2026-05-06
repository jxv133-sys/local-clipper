"""Property-based tests for Mini Video Editor undo/redo functionality (Task 5.3).

These tests validate universal correctness properties of the undo/redo history
management for the Mini Video Editor feature.

**Validates: Requirements 18.1, 18.2, 18.3, 18.5**
"""

from __future__ import annotations

import copy
import uuid
from dataclasses import replace
from typing import List

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from pipeline.models import (
    CanvasLayout,
    EditorSession,
    FacecamRegion,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def make_region(x: int = 0, y: int = 0, w: int = 100, h: int = 100,
                corner: str = "top-right", confidence: float = 0.5) -> FacecamRegion:
    """Create a FacecamRegion with given parameters."""
    return FacecamRegion(x=x, y=y, width=w, height=h, corner=corner, confidence=confidence)


def make_session(initial_region: FacecamRegion | None = None) -> EditorSession:
    """Create a fresh EditorSession for testing."""
    if initial_region is None:
        initial_region = make_region(100, 50, 400, 300)
    return EditorSession(
        session_id=str(uuid.uuid4()),
        clip_batch_id="test-batch",
        reference_clip_path="/fake/clip.mp4",
        reference_resolution=(1920, 1080),
        facecam_region=initial_region,
        canvas_layout=DEFAULT_CANVAS_LAYOUT,
    )


def region_as_tuple(r: FacecamRegion) -> tuple:
    """Convert a FacecamRegion to a comparable tuple."""
    return (r.x, r.y, r.width, r.height, r.corner, r.confidence)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

@st.composite
def facecam_region(draw):
    """Generate a random FacecamRegion."""
    x = draw(st.integers(min_value=0, max_value=1800))
    y = draw(st.integers(min_value=0, max_value=1000))
    w = draw(st.integers(min_value=10, max_value=500))
    h = draw(st.integers(min_value=10, max_value=400))
    corner = draw(st.sampled_from(["top-left", "top-right", "bottom-left", "bottom-right"]))
    confidence = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    return make_region(x, y, w, h, corner, confidence)


@st.composite
def sequence_of_regions(draw, min_size: int = 1, max_size: int = 10):
    """Generate a non-empty list of FacecamRegion objects."""
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    return [draw(facecam_region()) for _ in range(n)]


# ---------------------------------------------------------------------------
# Property 8: Undo restoration
# **Validates: Requirements 18.1, 18.2**
# ---------------------------------------------------------------------------

@settings(max_examples=200)
@given(adjustments=sequence_of_regions(min_size=1, max_size=8))
def test_property_8_undo_restores_previous_state(adjustments):
    """
    Property 8: For any adjustment, undo should restore the previous state.

    After making N adjustments and then undoing once, the current facecam_region
    must equal the state before the last adjustment.

    **Validates: Requirements 18.1, 18.2**
    """
    session = make_session()
    initial_region = session.facecam_region

    # Apply all adjustments, recording state before each
    states = [initial_region]
    for adj in adjustments:
        session.push_undo(session.facecam_region)
        session.facecam_region = adj
        states.append(adj)

    # Undo each adjustment and verify the previous state is restored
    for i in range(len(adjustments) - 1, -1, -1):
        expected_state = states[i]
        previous = session.pop_undo()

        assert previous is not None, (
            f"pop_undo() returned None at step {i}, expected a region"
        )
        assert region_as_tuple(previous) == region_as_tuple(expected_state), (
            f"Undo at step {i} restored wrong state: "
            f"got {region_as_tuple(previous)}, expected {region_as_tuple(expected_state)}"
        )

        session.facecam_region = previous


@settings(max_examples=200)
@given(adjustments=sequence_of_regions(min_size=2, max_size=8))
def test_property_8_undo_history_decreases_by_one(adjustments):
    """
    Property 8: Each undo operation must decrease the undo history length by exactly 1.

    **Validates: Requirements 18.1, 18.2**
    """
    session = make_session()

    # Apply all adjustments
    for adj in adjustments:
        session.push_undo(session.facecam_region)
        session.facecam_region = adj

    initial_history_len = len(session.undo_history)

    # Undo once
    session.pop_undo()

    assert len(session.undo_history) == initial_history_len - 1, (
        f"Expected undo_history length {initial_history_len - 1}, "
        f"got {len(session.undo_history)}"
    )


@settings(max_examples=200)
@given(adjustments=sequence_of_regions(min_size=1, max_size=8))
def test_property_8_undo_empty_history_returns_none(adjustments):
    """
    Property 8: When undo history is empty, pop_undo() must return None.

    **Validates: Requirements 18.1, 18.2**
    """
    session = make_session()

    # Apply and undo all adjustments to empty the history
    for adj in adjustments:
        session.push_undo(session.facecam_region)
        session.facecam_region = adj

    # Drain the undo history
    while session.undo_history:
        session.pop_undo()

    # Now undo on empty history must return None
    result = session.pop_undo()
    assert result is None, (
        f"Expected None when undo history is empty, got {result!r}"
    )


@settings(max_examples=200)
@given(adjustments=sequence_of_regions(min_size=1, max_size=8))
def test_property_8_undo_history_length_equals_adjustment_count(adjustments):
    """
    Property 8: After N adjustments, the undo history must contain exactly N entries.

    **Validates: Requirements 18.1, 18.2**
    """
    session = make_session()

    for i, adj in enumerate(adjustments):
        session.push_undo(session.facecam_region)
        session.facecam_region = adj
        assert len(session.undo_history) == i + 1, (
            f"After {i + 1} adjustments, expected undo_history length {i + 1}, "
            f"got {len(session.undo_history)}"
        )


# ---------------------------------------------------------------------------
# Property 9: Redo reapplication
# **Validates: Requirements 18.1, 18.3**
# ---------------------------------------------------------------------------

@settings(max_examples=200)
@given(adjustments=sequence_of_regions(min_size=1, max_size=8))
def test_property_9_redo_reapplies_undone_adjustment(adjustments):
    """
    Property 9: For any undone adjustment, redo should reapply it.

    After making N adjustments, undoing them all, and then redoing them all,
    the final state must equal the state after the last adjustment.

    **Validates: Requirements 18.1, 18.3**
    """
    session = make_session()
    initial_region = session.facecam_region

    # Apply all adjustments
    for adj in adjustments:
        session.push_undo(session.facecam_region)
        session.facecam_region = adj

    final_state = session.facecam_region

    # Undo all adjustments
    while session.undo_history:
        prev = session.pop_undo()
        session.push_redo(session.facecam_region)
        session.facecam_region = prev

    # Verify we're back to initial state
    assert region_as_tuple(session.facecam_region) == region_as_tuple(initial_region), (
        "After undoing all adjustments, should be back to initial state"
    )

    # Redo all adjustments
    while session.redo_history:
        next_region = session.pop_redo()
        session.undo_history.append(session.facecam_region)
        session.facecam_region = next_region

    # Verify we're back to the final state
    assert region_as_tuple(session.facecam_region) == region_as_tuple(final_state), (
        f"After redoing all adjustments, expected final state "
        f"{region_as_tuple(final_state)}, got {region_as_tuple(session.facecam_region)}"
    )


@settings(max_examples=200)
@given(adjustments=sequence_of_regions(min_size=2, max_size=8))
def test_property_9_redo_history_decreases_by_one(adjustments):
    """
    Property 9: Each redo operation must decrease the redo history length by exactly 1.

    **Validates: Requirements 18.1, 18.3**
    """
    session = make_session()

    # Apply all adjustments
    for adj in adjustments:
        session.push_undo(session.facecam_region)
        session.facecam_region = adj

    # Undo all to populate redo history
    while session.undo_history:
        prev = session.pop_undo()
        session.push_redo(session.facecam_region)
        session.facecam_region = prev

    initial_redo_len = len(session.redo_history)
    assert initial_redo_len > 0, "Redo history should be non-empty after undoing"

    # Redo once
    session.pop_redo()

    assert len(session.redo_history) == initial_redo_len - 1, (
        f"Expected redo_history length {initial_redo_len - 1}, "
        f"got {len(session.redo_history)}"
    )


@settings(max_examples=200)
@given(adjustments=sequence_of_regions(min_size=1, max_size=8))
def test_property_9_redo_empty_history_returns_none(adjustments):
    """
    Property 9: When redo history is empty, pop_redo() must return None.

    **Validates: Requirements 18.1, 18.3**
    """
    session = make_session()

    # Fresh session has empty redo history
    result = session.pop_redo()
    assert result is None, (
        f"Expected None when redo history is empty, got {result!r}"
    )


@settings(max_examples=200)
@given(adjustments=sequence_of_regions(min_size=1, max_size=8))
def test_property_9_new_adjustment_clears_redo_history(adjustments):
    """
    Property 9: Making a new adjustment after undoing must clear the redo history.

    This ensures the redo history is invalidated when the user makes a new
    adjustment, preventing stale redo states.

    **Validates: Requirements 18.1, 18.3**
    """
    session = make_session()

    # Apply at least one adjustment
    session.push_undo(session.facecam_region)
    session.facecam_region = adjustments[0]

    # Undo it (populates redo history)
    prev = session.pop_undo()
    session.push_redo(session.facecam_region)
    session.facecam_region = prev

    assert len(session.redo_history) > 0, "Redo history should be non-empty after undo"

    # Make a new adjustment — push_undo clears redo history
    new_region = make_region(999, 999, 50, 50)
    session.push_undo(session.facecam_region)
    session.facecam_region = new_region

    assert len(session.redo_history) == 0, (
        f"Expected redo history to be cleared after new adjustment, "
        f"but it has {len(session.redo_history)} entries"
    )


@settings(max_examples=200)
@given(
    adjustments=sequence_of_regions(min_size=2, max_size=8),
    undo_count=st.integers(min_value=1, max_value=7),
)
def test_property_9_undo_then_redo_restores_exact_state(adjustments, undo_count):
    """
    Property 9: Undoing K steps and then redoing K steps must restore the
    exact state before the undos.

    **Validates: Requirements 18.1, 18.3**
    """
    session = make_session()

    # Apply all adjustments
    for adj in adjustments:
        session.push_undo(session.facecam_region)
        session.facecam_region = adj

    # Clamp undo_count to available history
    actual_undo_count = min(undo_count, len(session.undo_history))
    assume(actual_undo_count > 0)

    # Record state before undoing
    state_before_undo = session.facecam_region

    # Undo K steps
    for _ in range(actual_undo_count):
        prev = session.pop_undo()
        if prev is None:
            break
        session.push_redo(session.facecam_region)
        session.facecam_region = prev

    # Redo K steps
    for _ in range(actual_undo_count):
        next_r = session.pop_redo()
        if next_r is None:
            break
        session.undo_history.append(session.facecam_region)
        session.facecam_region = next_r

    # Must be back to state before undos
    assert region_as_tuple(session.facecam_region) == region_as_tuple(state_before_undo), (
        f"After undo+redo cycle, expected {region_as_tuple(state_before_undo)}, "
        f"got {region_as_tuple(session.facecam_region)}"
    )


# ---------------------------------------------------------------------------
# Property 10: History clearing
# **Validates: Requirements 18.1, 18.5**
# ---------------------------------------------------------------------------

@settings(max_examples=200)
@given(adjustments=sequence_of_regions(min_size=1, max_size=8))
def test_property_10_clear_history_empties_both_stacks(adjustments):
    """
    Property 10: For any confirmation or close, undo/redo history should be cleared.

    After calling clear_history(), both undo_history and redo_history must be empty.

    **Validates: Requirements 18.1, 18.5**
    """
    session = make_session()

    # Apply adjustments to populate undo history
    for adj in adjustments:
        session.push_undo(session.facecam_region)
        session.facecam_region = adj

    # Undo some to populate redo history
    if session.undo_history:
        prev = session.pop_undo()
        session.push_redo(session.facecam_region)
        session.facecam_region = prev

    # Verify histories are non-empty before clearing
    # (at least one of them should be non-empty)
    assert len(session.undo_history) > 0 or len(session.redo_history) > 0, (
        "At least one history should be non-empty before clearing"
    )

    # Clear history (simulates confirm or cancel)
    session.clear_history()

    assert len(session.undo_history) == 0, (
        f"Expected undo_history to be empty after clear_history(), "
        f"but it has {len(session.undo_history)} entries"
    )
    assert len(session.redo_history) == 0, (
        f"Expected redo_history to be empty after clear_history(), "
        f"but it has {len(session.redo_history)} entries"
    )


@settings(max_examples=200)
@given(adjustments=sequence_of_regions(min_size=1, max_size=8))
def test_property_10_clear_history_does_not_affect_current_region(adjustments):
    """
    Property 10: Clearing history must not change the current facecam_region.

    The current placement should be preserved even after history is cleared.

    **Validates: Requirements 18.1, 18.5**
    """
    session = make_session()

    # Apply adjustments
    for adj in adjustments:
        session.push_undo(session.facecam_region)
        session.facecam_region = adj

    current_region_before = session.facecam_region

    # Clear history
    session.clear_history()

    # Current region must be unchanged
    assert region_as_tuple(session.facecam_region) == region_as_tuple(current_region_before), (
        f"clear_history() changed the current facecam_region: "
        f"before={region_as_tuple(current_region_before)}, "
        f"after={region_as_tuple(session.facecam_region)}"
    )


@settings(max_examples=200)
@given(adjustments=sequence_of_regions(min_size=1, max_size=8))
def test_property_10_after_clear_undo_returns_none(adjustments):
    """
    Property 10: After clearing history, pop_undo() must return None.

    **Validates: Requirements 18.1, 18.5**
    """
    session = make_session()

    # Apply adjustments
    for adj in adjustments:
        session.push_undo(session.facecam_region)
        session.facecam_region = adj

    # Clear history
    session.clear_history()

    # Undo on empty history must return None
    result = session.pop_undo()
    assert result is None, (
        f"Expected None after clear_history(), got {result!r}"
    )


@settings(max_examples=200)
@given(adjustments=sequence_of_regions(min_size=1, max_size=8))
def test_property_10_after_clear_redo_returns_none(adjustments):
    """
    Property 10: After clearing history, pop_redo() must return None.

    **Validates: Requirements 18.1, 18.5**
    """
    session = make_session()

    # Apply adjustments and undo some to populate redo history
    for adj in adjustments:
        session.push_undo(session.facecam_region)
        session.facecam_region = adj

    if session.undo_history:
        prev = session.pop_undo()
        session.push_redo(session.facecam_region)
        session.facecam_region = prev

    # Clear history
    session.clear_history()

    # Redo on empty history must return None
    result = session.pop_redo()
    assert result is None, (
        f"Expected None after clear_history(), got {result!r}"
    )


@settings(max_examples=200)
@given(adjustments=sequence_of_regions(min_size=1, max_size=8))
def test_property_10_clear_history_is_idempotent(adjustments):
    """
    Property 10: Calling clear_history() multiple times must be safe (idempotent).

    **Validates: Requirements 18.1, 18.5**
    """
    session = make_session()

    # Apply adjustments
    for adj in adjustments:
        session.push_undo(session.facecam_region)
        session.facecam_region = adj

    # Clear multiple times
    session.clear_history()
    session.clear_history()
    session.clear_history()

    assert len(session.undo_history) == 0
    assert len(session.redo_history) == 0
    assert session.pop_undo() is None
    assert session.pop_redo() is None


# ---------------------------------------------------------------------------
# Additional invariant tests
# ---------------------------------------------------------------------------

@settings(max_examples=200)
@given(adjustments=sequence_of_regions(min_size=1, max_size=8))
def test_undo_redo_symmetry(adjustments):
    """
    Invariant: The undo/redo stacks are symmetric — undoing and redoing the
    same number of steps returns to the original state.

    **Validates: Requirements 18.1, 18.2, 18.3**
    """
    session = make_session()

    # Apply all adjustments
    for adj in adjustments:
        session.push_undo(session.facecam_region)
        session.facecam_region = adj

    final_state = session.facecam_region
    n_undos = len(session.undo_history)

    # Undo all
    for _ in range(n_undos):
        prev = session.pop_undo()
        if prev is None:
            break
        session.push_redo(session.facecam_region)
        session.facecam_region = prev

    # Redo all
    for _ in range(n_undos):
        nxt = session.pop_redo()
        if nxt is None:
            break
        session.undo_history.append(session.facecam_region)
        session.facecam_region = nxt

    assert region_as_tuple(session.facecam_region) == region_as_tuple(final_state), (
        "Full undo+redo cycle did not restore final state"
    )


@settings(max_examples=200)
@given(adjustments=sequence_of_regions(min_size=1, max_size=8))
def test_push_undo_clears_redo_history(adjustments):
    """
    Invariant: push_undo() must clear the redo history (new adjustment invalidates redo).

    **Validates: Requirements 18.1, 18.3**
    """
    session = make_session()

    # Apply one adjustment and undo it to populate redo history
    session.push_undo(session.facecam_region)
    session.facecam_region = adjustments[0]

    prev = session.pop_undo()
    session.push_redo(session.facecam_region)
    session.facecam_region = prev

    assert len(session.redo_history) > 0, "Redo history should be non-empty"

    # Make a new adjustment via push_undo — this should clear redo history
    session.push_undo(session.facecam_region)

    assert len(session.redo_history) == 0, (
        f"push_undo() should clear redo history, but it has "
        f"{len(session.redo_history)} entries"
    )
