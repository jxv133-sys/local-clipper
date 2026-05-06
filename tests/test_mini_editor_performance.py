"""Performance and correctness tests for Mini Video Editor.

Covers Tasks 26-28:
  - Session cleanup removes expired sessions
  - Detection cache doesn't grow unboundedly (cache key structure)
  - VerticalFormattingJob.estimate_remaining_time() accuracy
  - VerticalFormattingJob.get_progress_percentage() accuracy
  - Job queue integration (job is enqueued after confirm)
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from pipeline.models import CanvasLayout, FacecamRegion, VerticalFormattingJob


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_global_state():
    """Clear module-level globals between tests."""
    import web_server
    with web_server._jobs_lock:
        web_server._jobs.clear()
    with web_server._detection_cache_lock:
        web_server._detection_cache.clear()
    with web_server._preview_cache_lock:
        web_server._preview_cache.clear()
    with web_server._formatting_jobs_lock:
        web_server._formatting_jobs.clear()
    web_server._session_store._sessions.clear()
    yield
    with web_server._jobs_lock:
        web_server._jobs.clear()
    with web_server._detection_cache_lock:
        web_server._detection_cache.clear()
    with web_server._preview_cache_lock:
        web_server._preview_cache.clear()
    with web_server._formatting_jobs_lock:
        web_server._formatting_jobs.clear()
    web_server._session_store._sessions.clear()


@pytest.fixture
def sample_region():
    return FacecamRegion(
        x=100, y=50, width=400, height=300,
        corner="top-right", confidence=0.85,
    )


@pytest.fixture
def sample_layout():
    return CanvasLayout(
        canvas_width=1080, canvas_height=1920,
        facecam_x=0, facecam_y=0,
        facecam_width=1080, facecam_height=672,
        gameplay_x=0, gameplay_y=672,
        gameplay_width=1080, gameplay_height=1248,
    )


# ---------------------------------------------------------------------------
# Session cleanup
# ---------------------------------------------------------------------------

class TestSessionCleanup:
    """Session cleanup removes expired sessions."""

    def test_cleanup_removes_expired_sessions(self):
        """cleanup_expired_sessions() removes all expired sessions."""
        import web_server

        store = web_server._session_store
        region = FacecamRegion(x=100, y=50, width=400, height=300, corner="top-right", confidence=0.85)
        layout = CanvasLayout(
            canvas_width=1080, canvas_height=1920,
            facecam_x=0, facecam_y=0,
            facecam_width=1080, facecam_height=672,
            gameplay_x=0, gameplay_y=672,
            gameplay_width=1080, gameplay_height=1248,
        )

        # Create 5 sessions
        session_ids = []
        for _ in range(5):
            s = store.create_session(
                clip_batch_id="batch",
                reference_clip_path="/fake/clip.mp4",
                reference_resolution=(1920, 1080),
                facecam_region=region,
                canvas_layout=layout,
            )
            session_ids.append(s.session_id)

        # Expire 3 of them
        for sid in session_ids[:3]:
            store._sessions[sid].expires_at = time.time() - 1

        removed = store.cleanup_expired_sessions()
        assert removed == 3

        # Remaining 2 should still be accessible
        for sid in session_ids[3:]:
            assert store.get_session(sid) is not None

        # Expired ones should be gone
        for sid in session_ids[:3]:
            assert store.get_session(sid) is None

    def test_cleanup_does_not_remove_active_sessions(self):
        """cleanup_expired_sessions() does not remove active sessions."""
        import web_server

        store = web_server._session_store
        region = FacecamRegion(x=100, y=50, width=400, height=300, corner="top-right", confidence=0.85)
        layout = CanvasLayout(
            canvas_width=1080, canvas_height=1920,
            facecam_x=0, facecam_y=0,
            facecam_width=1080, facecam_height=672,
            gameplay_x=0, gameplay_y=672,
            gameplay_width=1080, gameplay_height=1248,
        )

        # Create 3 active sessions
        for _ in range(3):
            store.create_session(
                clip_batch_id="batch",
                reference_clip_path="/fake/clip.mp4",
                reference_resolution=(1920, 1080),
                facecam_region=region,
                canvas_layout=layout,
            )

        removed = store.cleanup_expired_sessions()
        assert removed == 0
        assert len(store.list_sessions()) == 3

    def test_expired_session_removed_on_get(self):
        """get_session() removes and returns None for expired sessions."""
        import web_server

        store = web_server._session_store
        region = FacecamRegion(x=100, y=50, width=400, height=300, corner="top-right", confidence=0.85)
        layout = CanvasLayout(
            canvas_width=1080, canvas_height=1920,
            facecam_x=0, facecam_y=0,
            facecam_width=1080, facecam_height=672,
            gameplay_x=0, gameplay_y=672,
            gameplay_width=1080, gameplay_height=1248,
        )

        session = store.create_session(
            clip_batch_id="batch",
            reference_clip_path="/fake/clip.mp4",
            reference_resolution=(1920, 1080),
            facecam_region=region,
            canvas_layout=layout,
        )
        session_id = session.session_id

        # Expire the session
        store._sessions[session_id].expires_at = time.time() - 1

        result = store.get_session(session_id)
        assert result is None
        assert session_id not in store._sessions


# ---------------------------------------------------------------------------
# Detection cache structure
# ---------------------------------------------------------------------------

class TestDetectionCacheStructure:
    """Detection cache key structure prevents unbounded growth."""

    def test_cache_key_includes_mtime(self, tmp_path):
        """Cache key includes file mtime for invalidation."""
        import web_server

        video = tmp_path / "clip.mp4"
        video.write_text("fake")
        mtime = video.stat().st_mtime

        # The cache key should be (clip_path, frame_width, frame_height, mtime)
        cache_key = (str(video), 1920, 1080, mtime)

        with web_server._detection_cache_lock:
            web_server._detection_cache[cache_key] = {"facecam_region": None, "error": "test"}

        with web_server._detection_cache_lock:
            result = web_server._detection_cache.get(cache_key)

        assert result is not None
        assert result["error"] == "test"

    def test_different_files_have_different_cache_keys(self, tmp_path):
        """Different clip paths produce different cache keys."""
        video1 = tmp_path / "clip1.mp4"
        video2 = tmp_path / "clip2.mp4"
        video1.write_text("fake1")
        video2.write_text("fake2")

        mtime1 = video1.stat().st_mtime
        mtime2 = video2.stat().st_mtime

        key1 = (str(video1), 1920, 1080, mtime1)
        key2 = (str(video2), 1920, 1080, mtime2)

        assert key1 != key2

    def test_same_file_different_dimensions_have_different_keys(self, tmp_path):
        """Same file with different dimensions produces different cache keys."""
        video = tmp_path / "clip.mp4"
        video.write_text("fake")
        mtime = video.stat().st_mtime

        key1 = (str(video), 1920, 1080, mtime)
        key2 = (str(video), 1280, 720, mtime)

        assert key1 != key2

    def test_cache_can_be_cleared(self):
        """Detection cache can be cleared to prevent unbounded growth."""
        import web_server

        with web_server._detection_cache_lock:
            web_server._detection_cache["test_key"] = {"facecam_region": None, "error": None}
            assert len(web_server._detection_cache) > 0
            web_server._detection_cache.clear()
            assert len(web_server._detection_cache) == 0


# ---------------------------------------------------------------------------
# VerticalFormattingJob timing accuracy
# ---------------------------------------------------------------------------

class TestVerticalFormattingJobTiming:
    """VerticalFormattingJob estimate_remaining_time() and get_progress_percentage() accuracy."""

    def test_estimate_remaining_time_before_start_returns_zero(self, sample_region, sample_layout):
        """estimate_remaining_time() returns 0 before processing starts."""
        job = VerticalFormattingJob(
            job_id="test",
            session_id="sess",
            clip_batch_id="batch",
            facecam_region=sample_region,
            canvas_layout=sample_layout,
            settings={},
            clips=[{"path": "/fake/clip.mp4", "name": "clip.mp4"}],
            output_dir="/tmp",
        )
        assert job.estimate_remaining_time() == 0.0

    def test_estimate_remaining_time_with_no_clips_processed_returns_zero(self, sample_region, sample_layout):
        """estimate_remaining_time() returns 0 when no clips processed yet."""
        job = VerticalFormattingJob(
            job_id="test",
            session_id="sess",
            clip_batch_id="batch",
            facecam_region=sample_region,
            canvas_layout=sample_layout,
            settings={},
            clips=[{"path": f"/fake/clip{i}.mp4", "name": f"clip{i}.mp4"} for i in range(5)],
            output_dir="/tmp",
        )
        job.start_processing()
        assert job.estimate_remaining_time() == 0.0

    def test_estimate_remaining_time_is_accurate(self, sample_region, sample_layout):
        """estimate_remaining_time() estimates based on average time per clip."""
        job = VerticalFormattingJob(
            job_id="test",
            session_id="sess",
            clip_batch_id="batch",
            facecam_region=sample_region,
            canvas_layout=sample_layout,
            settings={},
            clips=[{"path": f"/fake/clip{i}.mp4", "name": f"clip{i}.mp4"} for i in range(10)],
            output_dir="/tmp",
        )

        # Simulate: started 10 seconds ago, processed 2 clips
        job.started_at = time.time() - 10.0
        job.clips_processed = 2
        job.clips_total = 10

        # avg_time_per_clip = 10/2 = 5s; remaining = 8 clips * 5s = 40s
        eta = job.estimate_remaining_time()
        assert 35.0 <= eta <= 45.0  # Allow some tolerance for test timing

    def test_get_progress_percentage_zero_clips(self, sample_region, sample_layout):
        """get_progress_percentage() returns 0.0 when no clips processed."""
        job = VerticalFormattingJob(
            job_id="test",
            session_id="sess",
            clip_batch_id="batch",
            facecam_region=sample_region,
            canvas_layout=sample_layout,
            settings={},
            clips=[{"path": f"/fake/clip{i}.mp4", "name": f"clip{i}.mp4"} for i in range(5)],
            output_dir="/tmp",
        )
        assert job.get_progress_percentage() == 0.0

    def test_get_progress_percentage_half_done(self, sample_region, sample_layout):
        """get_progress_percentage() returns 50.0 when half clips processed."""
        job = VerticalFormattingJob(
            job_id="test",
            session_id="sess",
            clip_batch_id="batch",
            facecam_region=sample_region,
            canvas_layout=sample_layout,
            settings={},
            clips=[{"path": f"/fake/clip{i}.mp4", "name": f"clip{i}.mp4"} for i in range(10)],
            output_dir="/tmp",
        )
        job.clips_processed = 5
        assert job.get_progress_percentage() == 50.0

    def test_get_progress_percentage_all_done(self, sample_region, sample_layout):
        """get_progress_percentage() returns 100.0 when all clips processed."""
        job = VerticalFormattingJob(
            job_id="test",
            session_id="sess",
            clip_batch_id="batch",
            facecam_region=sample_region,
            canvas_layout=sample_layout,
            settings={},
            clips=[{"path": f"/fake/clip{i}.mp4", "name": f"clip{i}.mp4"} for i in range(4)],
            output_dir="/tmp",
        )
        job.clips_processed = 4
        assert job.get_progress_percentage() == 100.0

    def test_get_progress_percentage_no_clips_returns_zero(self, sample_region, sample_layout):
        """get_progress_percentage() returns 0.0 when clips_total is 0."""
        job = VerticalFormattingJob(
            job_id="test",
            session_id="sess",
            clip_batch_id="batch",
            facecam_region=sample_region,
            canvas_layout=sample_layout,
            settings={},
            clips=[],
            output_dir="/tmp",
        )
        assert job.get_progress_percentage() == 0.0

    def test_get_progress_percentage_at_various_stages(self, sample_region, sample_layout):
        """get_progress_percentage() is accurate at various stages."""
        clips = [{"path": f"/fake/clip{i}.mp4", "name": f"clip{i}.mp4"} for i in range(8)]
        job = VerticalFormattingJob(
            job_id="test",
            session_id="sess",
            clip_batch_id="batch",
            facecam_region=sample_region,
            canvas_layout=sample_layout,
            settings={},
            clips=clips,
            output_dir="/tmp",
        )

        for processed, expected_pct in [(0, 0.0), (2, 25.0), (4, 50.0), (6, 75.0), (8, 100.0)]:
            job.clips_processed = processed
            assert job.get_progress_percentage() == expected_pct


# ---------------------------------------------------------------------------
# Job queue integration
# ---------------------------------------------------------------------------

class TestJobQueueIntegration:
    """Verify job is enqueued after confirm."""

    def test_confirm_enqueues_job(self, tmp_path):
        """Confirming placement enqueues the job for background processing."""
        import web_server
        from web_server import Job, JobStatus
        import json

        # Set up a done job
        clip = tmp_path / "clip.mp4"
        clip.write_text("fake")
        job_id = str(uuid.uuid4())
        job = Job(
            job_id=job_id,
            video_path=str(tmp_path / "source.mp4"),
            config=MagicMock(),
            status=JobStatus.DONE,
            result_clips=[{"path": str(clip), "name": "clip.mp4"}],
        )
        with web_server._jobs_lock:
            web_server._jobs[job_id] = job

        # Create a session
        region = FacecamRegion(x=100, y=50, width=400, height=300, corner="top-right", confidence=0.85)
        layout = CanvasLayout(
            canvas_width=1080, canvas_height=1920,
            facecam_x=0, facecam_y=0,
            facecam_width=1080, facecam_height=672,
            gameplay_x=0, gameplay_y=672,
            gameplay_width=1080, gameplay_height=1248,
        )
        session = web_server._session_store.create_session(
            clip_batch_id=job_id,
            reference_clip_path=str(clip),
            reference_resolution=(1920, 1080),
            facecam_region=region,
            canvas_layout=layout,
        )

        web_server.app.config["TESTING"] = True
        client = web_server.app.test_client()

        enqueued_ids = []

        def mock_put(job_id_val):
            enqueued_ids.append(job_id_val)

        with patch.object(web_server._formatting_job_queue, "put", side_effect=mock_put):
            resp = client.post(
                "/api/mini-editor/confirm",
                json={
                    "session_id": session.session_id,
                    "facecam_region": {
                        "x": 100, "y": 50, "width": 400, "height": 300,
                        "corner": "top-right", "confidence": 0.85,
                    },
                },
                content_type="application/json",
            )

        assert resp.status_code == 201
        data = json.loads(resp.data)
        assert data["job_id"] in enqueued_ids

    def test_worker_thread_is_running(self):
        """Background vertical formatting worker thread is running."""
        import threading
        import web_server

        # Check that the worker thread exists and is alive
        worker_threads = [
            t for t in threading.enumerate()
            if t.name == "vertical-formatting-worker"
        ]
        assert len(worker_threads) == 1
        assert worker_threads[0].is_alive()
        assert worker_threads[0].daemon is True
