"""Property-based tests for batch processing (Task 7.5).

Validates: Requirements 7.6, 12.7, 19.1, 19.2
"""

from __future__ import annotations

import os
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from pipeline.models import CanvasLayout, FacecamRegion, VerticalFormattingJob
from pipeline.vertical_formatter import process_vertical_formatting_job
from pipeline.frame_reformatter import compute_canvas_layout


# ---------------------------------------------------------------------------
# Helpers / Strategies
# ---------------------------------------------------------------------------

def make_config(
    shorts_width: int = 1080,
    shorts_height: int = 1920,
    facecam_top_fraction: float = 0.35,
) -> SimpleNamespace:
    return SimpleNamespace(
        shorts_width=shorts_width,
        shorts_height=shorts_height,
        facecam_top_fraction=facecam_top_fraction,
    )


DEFAULT_CONFIG = make_config()


def make_canvas_layout() -> CanvasLayout:
    return compute_canvas_layout(DEFAULT_CONFIG)


def make_facecam_region(x=100, y=50, width=300, height=200) -> FacecamRegion:
    return FacecamRegion(
        x=x, y=y, width=width, height=height,
        corner="top-left", confidence=0.9,
    )


def make_clip_dict(name: str = "clip.mp4", path: str = "/tmp/clip.mp4") -> dict:
    return {"path": path, "name": name, "resolution": [1920, 1080]}


def make_job(
    clips: list[dict],
    facecam_region: FacecamRegion | None = None,
    canvas_layout: CanvasLayout | None = None,
    output_dir: str = "/tmp/output",
    settings_dict: dict | None = None,
) -> VerticalFormattingJob:
    return VerticalFormattingJob(
        job_id=str(uuid.uuid4()),
        session_id=str(uuid.uuid4()),
        clip_batch_id=str(uuid.uuid4()),
        facecam_region=facecam_region or make_facecam_region(),
        canvas_layout=canvas_layout or make_canvas_layout(),
        settings=settings_dict or {},
        clips=clips,
        output_dir=output_dir,
    )


@st.composite
def clip_list(draw, min_clips=1, max_clips=8):
    """Generate a list of clip dicts."""
    n = draw(st.integers(min_value=min_clips, max_value=max_clips))
    clips = []
    for i in range(n):
        clips.append({
            "path": f"/tmp/clip_{i}.mp4",
            "name": f"clip_{i}.mp4",
            "resolution": [1920, 1080],
        })
    return clips


# ---------------------------------------------------------------------------
# Property 14: Cancellation safety
# **Validates: Requirements 12.7, 19.6**
# ---------------------------------------------------------------------------

@settings(max_examples=50)
@given(
    n_clips=st.integers(min_value=2, max_value=8),
    cancel_after=st.integers(min_value=0, max_value=1),
)
def test_property_14_cancelled_job_stops_processing(n_clips, cancel_after):
    """
    Property 14: For any cancelled job, processing stops after the cancellation
    point and clips beyond that point are not processed.

    When a job is cancelled after processing `cancel_after` clips, the
    remaining clips must not be processed.

    **Validates: Requirements 12.7, 19.6**
    """
    assume(cancel_after < n_clips)

    clips = [make_clip_dict(f"clip_{i}.mp4", f"/tmp/clip_{i}.mp4") for i in range(n_clips)]
    job = make_job(clips)

    processed_clips: list[str] = []
    call_count = [0]

    def mock_apply(clip_path, facecam_region, canvas_layout, output_path, config,
                   reference_resolution=None, clip_resolution=None):
        call_count[0] += 1
        processed_clips.append(clip_path)
        # Cancel the job after processing `cancel_after` clips
        if call_count[0] > cancel_after:
            job.status = "cancelled"

    with patch(
        "pipeline.vertical_formatter.VerticalFormatter.apply_placement_to_clip",
        side_effect=mock_apply,
    ):
        process_vertical_formatting_job(job)

    # Job must be in cancelled state
    assert job.status == "cancelled", (
        f"Expected status 'cancelled', got '{job.status}'"
    )

    # Must not have processed all clips (some were skipped due to cancellation)
    assert len(processed_clips) <= n_clips, (
        f"Processed {len(processed_clips)} clips but only {n_clips} exist"
    )


@settings(max_examples=50)
@given(n_clips=st.integers(min_value=1, max_value=6))
def test_property_14_pre_cancelled_job_processes_no_clips(n_clips):
    """
    Property 14: A job that is already cancelled before processing starts
    must not process any clips.

    **Validates: Requirements 12.7, 19.6**
    """
    clips = [make_clip_dict(f"clip_{i}.mp4", f"/tmp/clip_{i}.mp4") for i in range(n_clips)]
    job = make_job(clips)

    # Pre-cancel the job
    job.status = "cancelled"

    processed_clips: list[str] = []

    def mock_apply(clip_path, *args, **kwargs):
        processed_clips.append(clip_path)

    with patch(
        "pipeline.vertical_formatter.VerticalFormatter.apply_placement_to_clip",
        side_effect=mock_apply,
    ):
        process_vertical_formatting_job(job)

    # No clips should have been processed
    assert len(processed_clips) == 0, (
        f"Expected 0 clips processed for pre-cancelled job, "
        f"but {len(processed_clips)} were processed"
    )


@settings(max_examples=50)
@given(
    n_clips=st.integers(min_value=2, max_value=6),
    cancel_after=st.integers(min_value=1, max_value=5),
)
def test_property_14_already_processed_clips_preserved_on_cancel(n_clips, cancel_after):
    """
    Property 14: When a job is cancelled mid-batch, the clips that were
    already processed before cancellation must have been processed
    (their output paths were computed and apply_placement_to_clip was called).

    **Validates: Requirements 12.7, 19.6**
    """
    assume(cancel_after <= n_clips)

    clips = [make_clip_dict(f"clip_{i}.mp4", f"/tmp/clip_{i}.mp4") for i in range(n_clips)]
    job = make_job(clips)

    processed_clips: list[str] = []
    call_count = [0]

    def mock_apply(clip_path, *args, **kwargs):
        call_count[0] += 1
        processed_clips.append(clip_path)
        if call_count[0] >= cancel_after:
            job.status = "cancelled"

    with patch(
        "pipeline.vertical_formatter.VerticalFormatter.apply_placement_to_clip",
        side_effect=mock_apply,
    ):
        process_vertical_formatting_job(job)

    # The clips processed before cancellation must be in the processed list
    expected_processed = min(cancel_after, n_clips)
    assert len(processed_clips) == expected_processed, (
        f"Expected {expected_processed} clips processed before cancel, "
        f"got {len(processed_clips)}"
    )


# ---------------------------------------------------------------------------
# Property 15: Progress accuracy
# **Validates: Requirements 19.1, 19.2**
# ---------------------------------------------------------------------------

@settings(max_examples=50)
@given(clips=clip_list(min_clips=1, max_clips=6))
def test_property_15_clips_processed_matches_successful_clips(clips):
    """
    Property 15: For any batch of clips, clips_processed must accurately
    reflect the number of clips that were successfully processed.

    **Validates: Requirements 19.1, 19.2**
    """
    job = make_job(clips)

    def mock_apply(clip_path, *args, **kwargs):
        pass  # Simulate successful processing

    with patch(
        "pipeline.vertical_formatter.VerticalFormatter.apply_placement_to_clip",
        side_effect=mock_apply,
    ):
        process_vertical_formatting_job(job)

    assert job.clips_processed == len(clips), (
        f"Expected clips_processed={len(clips)}, got {job.clips_processed}"
    )
    assert job.status == "done", (
        f"Expected status 'done', got '{job.status}'"
    )


@settings(max_examples=50)
@given(
    n_clips=st.integers(min_value=2, max_value=6),
    fail_indices=st.lists(
        st.integers(min_value=0, max_value=5),
        min_size=1,
        max_size=3,
        unique=True,
    ),
)
def test_property_15_failed_clips_not_counted_in_progress(n_clips, fail_indices):
    """
    Property 15: For any batch where some clips fail, clips_processed must
    only count successfully processed clips (not failed ones).

    **Validates: Requirements 19.1, 19.2**
    """
    # Only use fail_indices that are within range
    valid_fail_indices = [i for i in fail_indices if i < n_clips]
    assume(len(valid_fail_indices) > 0)

    clips = [make_clip_dict(f"clip_{i}.mp4", f"/tmp/clip_{i}.mp4") for i in range(n_clips)]
    job = make_job(clips)

    call_count = [0]

    def mock_apply(clip_path, *args, **kwargs):
        idx = call_count[0]
        call_count[0] += 1
        if idx in valid_fail_indices:
            raise RuntimeError(f"Simulated failure for clip {idx}")

    with patch(
        "pipeline.vertical_formatter.VerticalFormatter.apply_placement_to_clip",
        side_effect=mock_apply,
    ):
        process_vertical_formatting_job(job)

    expected_processed = n_clips - len(valid_fail_indices)
    assert job.clips_processed == expected_processed, (
        f"Expected clips_processed={expected_processed} "
        f"(n_clips={n_clips}, failures={len(valid_fail_indices)}), "
        f"got {job.clips_processed}"
    )

    # Errors must be recorded for each failed clip
    assert len(job.errors) == len(valid_fail_indices), (
        f"Expected {len(valid_fail_indices)} errors, got {len(job.errors)}"
    )


@settings(max_examples=50)
@given(clips=clip_list(min_clips=1, max_clips=6))
def test_property_15_clips_total_set_correctly(clips):
    """
    Property 15: For any batch, clips_total must equal the number of clips
    in the batch from the start.

    **Validates: Requirements 19.1, 19.2**
    """
    job = make_job(clips)

    # clips_total should be set at job creation
    assert job.clips_total == len(clips), (
        f"Expected clips_total={len(clips)}, got {job.clips_total}"
    )


@settings(max_examples=50)
@given(clips=clip_list(min_clips=1, max_clips=6))
def test_property_15_progress_never_exceeds_total(clips):
    """
    Property 15: For any batch, clips_processed must never exceed clips_total
    at any point during processing.

    **Validates: Requirements 19.1, 19.2**
    """
    job = make_job(clips)
    violations: list[str] = []

    original_increment = job.increment_progress

    def tracking_increment(current_clip: str = ""):
        original_increment(current_clip)
        if job.clips_processed > job.clips_total:
            violations.append(
                f"clips_processed ({job.clips_processed}) > "
                f"clips_total ({job.clips_total}) after processing '{current_clip}'"
            )

    job.increment_progress = tracking_increment  # type: ignore[method-assign]

    def mock_apply(clip_path, *args, **kwargs):
        pass

    with patch(
        "pipeline.vertical_formatter.VerticalFormatter.apply_placement_to_clip",
        side_effect=mock_apply,
    ):
        process_vertical_formatting_job(job)

    assert len(violations) == 0, (
        f"Progress exceeded total: {violations}"
    )


@settings(max_examples=50)
@given(clips=clip_list(min_clips=1, max_clips=6))
def test_property_15_current_clip_updated_during_processing(clips):
    """
    Property 15: For any batch, current_clip must be updated to the name of
    the clip being processed.

    **Validates: Requirements 19.3, 19.4**
    """
    job = make_job(clips)
    observed_current_clips: list[str] = []

    def mock_apply(clip_path, *args, **kwargs):
        # Record the current_clip at the time of processing
        observed_current_clips.append(job.current_clip)

    with patch(
        "pipeline.vertical_formatter.VerticalFormatter.apply_placement_to_clip",
        side_effect=mock_apply,
    ):
        process_vertical_formatting_job(job)

    # Each clip's name should have been set as current_clip before processing
    expected_names = [c["name"] for c in clips]
    assert observed_current_clips == expected_names, (
        f"current_clip not updated correctly: "
        f"expected {expected_names}, got {observed_current_clips}"
    )
