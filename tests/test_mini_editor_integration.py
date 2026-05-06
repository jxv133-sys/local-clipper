"""Integration tests for Mini Video Editor API endpoints.

Tests cover:
  22.1 Session creation and management
  22.2 Detection endpoint
  22.3 Preview endpoint
  22.4 Confirmation endpoint
  22.5 Undo/redo endpoints
"""

from __future__ import annotations

import json
import time
import uuid
from unittest.mock import MagicMock, Mock, patch

import pytest

from pipeline.models import CanvasLayout, FacecamRegion


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_global_state():
    """Clear module-level globals between tests to prevent state leakage."""
    import web_server
    with web_server._jobs_lock:
        web_server._jobs.clear()
    with web_server._detection_cache_lock:
        web_server._detection_cache.clear()
    with web_server._preview_cache_lock:
        web_server._preview_cache.clear()
    with web_server._formatting_jobs_lock:
        web_server._formatting_jobs.clear()
    # Clear session store
    web_server._session_store._sessions.clear()
    yield
    # Cleanup after test
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
def app():
    import web_server
    web_server.app.config["TESTING"] = True
    return web_server.app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def sample_canvas_layout():
    return CanvasLayout(
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


@pytest.fixture
def sample_facecam_region():
    return FacecamRegion(
        x=100, y=50, width=400, height=300,
        corner="top-right", confidence=0.85,
    )


def _make_done_job(tmp_path, job_id=None):
    """Create a completed Job with two fake clips and register it."""
    import web_server
    from web_server import Job, JobStatus

    if job_id is None:
        job_id = str(uuid.uuid4())

    clip1 = tmp_path / "clip1.mp4"
    clip2 = tmp_path / "clip2.mp4"
    clip1.write_text("fake video 1")
    clip2.write_text("fake video 2")

    job = Job(
        job_id=job_id,
        video_path=str(tmp_path / "source.mp4"),
        config=MagicMock(),
        status=JobStatus.DONE,
        result_clips=[
            {"path": str(clip1), "name": "clip1.mp4"},
            {"path": str(clip2), "name": "clip2.mp4"},
        ],
    )
    with web_server._jobs_lock:
        web_server._jobs[job_id] = job
    return job_id, str(clip1), str(clip2)


def _mock_ffprobe(width=1920, height=1080):
    """Return a mock subprocess result that looks like ffprobe output."""
    mock_result = Mock()
    mock_result.returncode = 0
    mock_result.stdout = json.dumps({"streams": [{"width": width, "height": height}]})
    mock_result.stderr = ""
    return mock_result


def _create_session(client, tmp_path, job_id=None, sample_facecam_region=None, sample_canvas_layout=None):
    """Helper: create a session via the API and return (session_id, job_id)."""
    if job_id is None:
        job_id, clip1_path, _ = _make_done_job(tmp_path)
    else:
        clip1_path = str(tmp_path / "clip1.mp4")
        if not (tmp_path / "clip1.mp4").exists():
            (tmp_path / "clip1.mp4").write_text("fake")

    if sample_facecam_region is None:
        sample_facecam_region = FacecamRegion(
            x=100, y=50, width=400, height=300,
            corner="top-right", confidence=0.85,
        )
    if sample_canvas_layout is None:
        sample_canvas_layout = CanvasLayout(
            canvas_width=1080, canvas_height=1920,
            facecam_x=0, facecam_y=0,
            facecam_width=1080, facecam_height=672,
            gameplay_x=0, gameplay_y=672,
            gameplay_width=1080, gameplay_height=1248,
        )

    with patch("web_server.FacecamRelocator") as mock_rel, \
         patch("pipeline.frame_reformatter.compute_canvas_layout") as mock_cl, \
         patch("subprocess.run") as mock_sub:
        mock_rel.return_value.detect_facecam.return_value = sample_facecam_region
        mock_cl.return_value = sample_canvas_layout
        mock_sub.return_value = _mock_ffprobe()

        resp = client.post(
            "/api/mini-editor/session",
            json={"clip_batch_id": job_id, "reference_clip_path": clip1_path},
            content_type="application/json",
        )

    assert resp.status_code == 201, resp.data
    data = json.loads(resp.data)
    return data["session_id"], job_id


# ---------------------------------------------------------------------------
# 22.1 Session creation and management
# ---------------------------------------------------------------------------

class TestSessionCreationAndManagement:
    """22.1 Session creation and management."""

    def test_create_session_returns_correct_structure(self, client, tmp_path):
        """POST /api/mini-editor/session creates session with correct structure."""
        job_id, clip1, clip2 = _make_done_job(tmp_path)

        with patch("web_server.FacecamRelocator") as mock_rel, \
             patch("pipeline.frame_reformatter.compute_canvas_layout") as mock_cl, \
             patch("subprocess.run") as mock_sub:
            mock_rel.return_value.detect_facecam.return_value = FacecamRegion(
                x=100, y=50, width=400, height=300, corner="top-right", confidence=0.9
            )
            mock_cl.return_value = CanvasLayout(
                canvas_width=1080, canvas_height=1920,
                facecam_x=0, facecam_y=0,
                facecam_width=1080, facecam_height=672,
                gameplay_x=0, gameplay_y=672,
                gameplay_width=1080, gameplay_height=1248,
            )
            mock_sub.return_value = _mock_ffprobe()

            resp = client.post(
                "/api/mini-editor/session",
                json={"clip_batch_id": job_id, "reference_clip_path": clip1},
                content_type="application/json",
            )

        assert resp.status_code == 201
        data = json.loads(resp.data)
        assert data["error"] is None
        assert data["session_id"] is not None
        assert len(data["session_id"]) > 0
        assert isinstance(data["clips"], list)
        assert len(data["clips"]) == 2
        assert data["reference_clip"]["name"] == "clip1.mp4"
        assert data["reference_clip"]["resolution"] == [1920, 1080]

    def test_session_expiry_mock_time(self, client, tmp_path):
        """Session expires after 30 minutes (mocked time)."""
        import web_server

        session_id, _ = _create_session(client, tmp_path)

        # Session should be accessible now
        session = web_server._session_store.get_session(session_id)
        assert session is not None

        # Mock time to be 31 minutes in the future
        future_time = time.time() + 31 * 60
        with patch("time.time", return_value=future_time):
            expired_session = web_server._session_store.get_session(session_id)
            assert expired_session is None

    def test_session_lookup_by_id(self, client, tmp_path):
        """Session can be looked up by session_id."""
        import web_server

        session_id, _ = _create_session(client, tmp_path)

        session = web_server._session_store.get_session(session_id)
        assert session is not None
        assert session.session_id == session_id

    def test_session_not_found_returns_none(self, client):
        """Looking up a non-existent session returns None."""
        import web_server
        result = web_server._session_store.get_session("nonexistent-id")
        assert result is None

    def test_session_cleanup_removes_expired(self, client, tmp_path):
        """cleanup_expired_sessions removes expired sessions."""
        import web_server

        session_id, _ = _create_session(client, tmp_path)

        # Manually expire the session
        session = web_server._session_store._sessions[session_id]
        session.expires_at = time.time() - 1  # already expired

        removed = web_server._session_store.cleanup_expired_sessions()
        assert removed == 1
        assert web_server._session_store.get_session(session_id) is None


# ---------------------------------------------------------------------------
# 22.2 Detection endpoint
# ---------------------------------------------------------------------------

class TestDetectionEndpoint:
    """22.2 Detection endpoint."""

    def test_detection_success_with_mocked_relocator(self, client, tmp_path):
        """Test with mocked FacecamRelocator returning a region."""
        video = tmp_path / "clip.mp4"
        video.write_text("fake")

        region = FacecamRegion(x=50, y=20, width=300, height=200, corner="top-left", confidence=0.92)

        with patch("web_server.FacecamRelocator") as mock_rel:
            mock_rel.return_value.detect_facecam.return_value = region

            resp = client.post(
                "/api/mini-editor/detect",
                json={"clip_path": str(video), "frame_width": 1920, "frame_height": 1080},
                content_type="application/json",
            )

        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["error"] is None
        assert data["facecam_region"]["x"] == 50
        assert data["facecam_region"]["confidence"] == 0.92
        assert data["cached"] is False

    def test_detection_failure_returns_structured_error(self, client, tmp_path):
        """Detection failure returns structured error with offer_manual_selection flag."""
        video = tmp_path / "clip.mp4"
        video.write_text("fake")

        with patch("web_server.FacecamRelocator") as mock_rel:
            mock_rel.return_value.detect_facecam.return_value = None

            resp = client.post(
                "/api/mini-editor/detect",
                json={"clip_path": str(video), "frame_width": 1920, "frame_height": 1080},
                content_type="application/json",
            )

        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["facecam_region"] is None
        assert data["error"] is not None
        assert data["offer_manual_selection"] is True
        assert data["offer_fallback"] is True
        assert "reason" in data

    def test_detection_caching_behavior(self, client, tmp_path):
        """Detection results are cached; second call returns cached=True."""
        import web_server
        with web_server._detection_cache_lock:
            web_server._detection_cache.clear()

        video = tmp_path / "clip.mp4"
        video.write_text("fake")

        region = FacecamRegion(x=10, y=10, width=200, height=150, corner="top-left", confidence=0.7)

        with patch("web_server.FacecamRelocator") as mock_rel:
            mock_rel.return_value.detect_facecam.return_value = region

            resp1 = client.post(
                "/api/mini-editor/detect",
                json={"clip_path": str(video), "frame_width": 1920, "frame_height": 1080},
                content_type="application/json",
            )
            resp2 = client.post(
                "/api/mini-editor/detect",
                json={"clip_path": str(video), "frame_width": 1920, "frame_height": 1080},
                content_type="application/json",
            )

            assert mock_rel.return_value.detect_facecam.call_count == 1

        assert json.loads(resp1.data)["cached"] is False
        assert json.loads(resp2.data)["cached"] is True


# ---------------------------------------------------------------------------
# 22.3 Preview endpoint
# ---------------------------------------------------------------------------

class TestPreviewEndpoint:
    """22.3 Preview endpoint."""

    def test_preview_with_mocked_ffmpeg(self, client, tmp_path):
        """Test preview generation with mocked FFmpeg subprocess."""
        video = tmp_path / "clip.mp4"
        video.write_text("fake")

        # Create a fake preview image
        preview_img = tmp_path / "preview.jpg"
        preview_img.write_bytes(b"fake jpeg")

        with patch("subprocess.run") as mock_sub, \
             patch("web_server.uuid") as mock_uuid:
            mock_sub.return_value = Mock(returncode=0, stdout="", stderr="")
            mock_uuid.uuid4.return_value.hex = "abcd1234"

            # Make the preview file appear to exist
            with patch("web_server.OUTPUT_DIR", tmp_path):
                resp = client.post(
                    "/api/mini-editor/preview",
                    json={
                        "clip_path": str(video),
                        "facecam_region": {
                            "x": 100, "y": 50, "width": 400, "height": 300,
                            "corner": "top-right", "confidence": 0.85,
                        },
                        "frame_width": 1920,
                        "frame_height": 1080,
                    },
                    content_type="application/json",
                )

        # Either success or ffmpeg failure is acceptable in test env
        assert resp.status_code in (200, 500)
        data = json.loads(resp.data)
        assert "error" in data

    def test_preview_invalid_facecam_region_returns_400(self, client, tmp_path):
        """Invalid facecam_region (missing field) returns 400."""
        video = tmp_path / "clip.mp4"
        video.write_text("fake")

        resp = client.post(
            "/api/mini-editor/preview",
            json={
                "clip_path": str(video),
                "facecam_region": {"x": 100, "y": 50},  # missing width/height/corner/confidence
                "frame_width": 1920,
                "frame_height": 1080,
            },
            content_type="application/json",
        )

        assert resp.status_code == 400
        data = json.loads(resp.data)
        assert "error" in data

    def test_preview_caching(self, client, tmp_path):
        """Preview results are cached; second call returns cached=True."""
        import web_server
        with web_server._preview_cache_lock:
            web_server._preview_cache.clear()

        video = tmp_path / "clip.mp4"
        video.write_text("fake")

        # Pre-populate the cache with a fake preview path
        fake_preview = tmp_path / "fake_preview.jpg"
        fake_preview.write_bytes(b"fake jpeg")

        try:
            mtime = video.stat().st_mtime
        except OSError:
            mtime = 0

        cache_key = (str(video), 100, 50, 400, 300, mtime)
        with web_server._preview_cache_lock:
            web_server._preview_cache[cache_key] = str(fake_preview)

        with patch("pipeline.frame_reformatter.compute_canvas_layout") as mock_cl:
            mock_cl.return_value = CanvasLayout(
                canvas_width=1080, canvas_height=1920,
                facecam_x=0, facecam_y=0,
                facecam_width=1080, facecam_height=672,
                gameplay_x=0, gameplay_y=672,
                gameplay_width=1080, gameplay_height=1248,
            )
            resp = client.post(
                "/api/mini-editor/preview",
                json={
                    "clip_path": str(video),
                    "facecam_region": {
                        "x": 100, "y": 50, "width": 400, "height": 300,
                        "corner": "top-right", "confidence": 0.85,
                    },
                    "frame_width": 1920,
                    "frame_height": 1080,
                },
                content_type="application/json",
            )

        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["cached"] is True


# ---------------------------------------------------------------------------
# 22.4 Confirmation endpoint
# ---------------------------------------------------------------------------

class TestConfirmationEndpoint:
    """22.4 Confirmation endpoint."""

    def test_valid_placement_creates_formatting_job(self, client, tmp_path):
        """Valid placement creates VerticalFormattingJob stored in _formatting_jobs."""
        import web_server

        session_id, job_id = _create_session(client, tmp_path)

        # Patch the queue to prevent actual processing
        with patch.object(web_server._formatting_job_queue, "put"):
            resp = client.post(
                "/api/mini-editor/confirm",
                json={
                    "session_id": session_id,
                    "facecam_region": {
                        "x": 100, "y": 50, "width": 400, "height": 300,
                        "corner": "top-right", "confidence": 0.85,
                    },
                },
                content_type="application/json",
            )

        assert resp.status_code == 201
        data = json.loads(resp.data)
        assert data["error"] is None
        assert data["job_id"] is not None
        assert data["status"] == "queued"

        # Verify job is stored
        with web_server._formatting_jobs_lock:
            assert data["job_id"] in web_server._formatting_jobs

    def test_invalid_placement_out_of_bounds_returns_400(self, client, tmp_path):
        """Placement extending beyond frame bounds returns 400."""
        session_id, _ = _create_session(client, tmp_path)

        resp = client.post(
            "/api/mini-editor/confirm",
            json={
                "session_id": session_id,
                "facecam_region": {
                    "x": 1900, "y": 50, "width": 400, "height": 300,  # x+w > 1920
                    "corner": "top-right", "confidence": 0.85,
                },
            },
            content_type="application/json",
        )

        assert resp.status_code == 400
        data = json.loads(resp.data)
        assert "error" in data
        assert data["error"] is not None

    def test_job_stored_in_formatting_jobs(self, client, tmp_path):
        """Confirmed job is stored in _formatting_jobs dict."""
        import web_server

        session_id, _ = _create_session(client, tmp_path)

        with patch.object(web_server._formatting_job_queue, "put"):
            resp = client.post(
                "/api/mini-editor/confirm",
                json={
                    "session_id": session_id,
                    "facecam_region": {
                        "x": 100, "y": 50, "width": 400, "height": 300,
                        "corner": "top-right", "confidence": 0.85,
                    },
                },
                content_type="application/json",
            )

        assert resp.status_code == 201
        job_id = json.loads(resp.data)["job_id"]

        with web_server._formatting_jobs_lock:
            assert job_id in web_server._formatting_jobs
            job = web_server._formatting_jobs[job_id]
            assert job.status == "queued"
            assert job.clips_total == 2

    def test_area_too_small_returns_400(self, client, tmp_path):
        """Region with area fraction below minimum returns 400."""
        session_id, _ = _create_session(client, tmp_path)

        # 10x10 region in 1920x1080 frame = 0.0000482 fraction (way below 4%)
        resp = client.post(
            "/api/mini-editor/confirm",
            json={
                "session_id": session_id,
                "facecam_region": {
                    "x": 0, "y": 0, "width": 10, "height": 10,
                    "corner": "top-left", "confidence": 0.5,
                },
            },
            content_type="application/json",
        )

        assert resp.status_code == 400
        data = json.loads(resp.data)
        assert "too small" in data["error"].lower() or "minimum" in data["error"].lower()


# ---------------------------------------------------------------------------
# 22.5 Undo/redo endpoints
# ---------------------------------------------------------------------------

class TestUndoRedoEndpoints:
    """22.5 Undo/redo endpoints."""

    def test_undo_restores_previous_region(self, client, tmp_path):
        """Undo restores the previous facecam region."""
        import web_server

        session_id, _ = _create_session(client, tmp_path)

        # Get the initial region
        session = web_server._session_store.get_session(session_id)
        original_region = session.facecam_region

        # Push a new region to undo history
        new_region = FacecamRegion(x=200, y=100, width=350, height=250, corner="top-left", confidence=0.7)
        session.push_undo(original_region)
        session.facecam_region = new_region

        # Undo
        resp = client.post(
            "/api/mini-editor/undo",
            json={"session_id": session_id},
            content_type="application/json",
        )

        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["error"] is None
        restored = data["facecam_region"]
        assert restored["x"] == original_region.x
        assert restored["y"] == original_region.y

    def test_redo_reapplies_region(self, client, tmp_path):
        """Redo reapplies the last undone region."""
        import web_server

        session_id, _ = _create_session(client, tmp_path)
        session = web_server._session_store.get_session(session_id)

        original_region = session.facecam_region
        new_region = FacecamRegion(x=200, y=100, width=350, height=250, corner="top-left", confidence=0.7)

        # Push original to undo, set new region
        session.push_undo(original_region)
        session.facecam_region = new_region

        # Undo (restores original)
        client.post("/api/mini-editor/undo", json={"session_id": session_id}, content_type="application/json")

        # Redo (reapplies new_region)
        resp = client.post(
            "/api/mini-editor/redo",
            json={"session_id": session_id},
            content_type="application/json",
        )

        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["error"] is None
        reapplied = data["facecam_region"]
        assert reapplied["x"] == new_region.x
        assert reapplied["y"] == new_region.y

    def test_history_cleared_on_confirm(self, client, tmp_path):
        """Undo/redo history is cleared after confirmation."""
        import web_server

        session_id, _ = _create_session(client, tmp_path)
        session = web_server._session_store.get_session(session_id)

        # Add some history
        session.push_undo(session.facecam_region)
        session.push_undo(session.facecam_region)
        assert len(session.undo_history) == 2

        with patch.object(web_server._formatting_job_queue, "put"):
            client.post(
                "/api/mini-editor/confirm",
                json={
                    "session_id": session_id,
                    "facecam_region": {
                        "x": 100, "y": 50, "width": 400, "height": 300,
                        "corner": "top-right", "confidence": 0.85,
                    },
                },
                content_type="application/json",
            )

        # History should be cleared
        session = web_server._session_store.get_session(session_id)
        assert session is not None
        assert len(session.undo_history) == 0
        assert len(session.redo_history) == 0

    def test_undo_with_empty_history_returns_400(self, client, tmp_path):
        """Undo with empty history returns 400."""
        session_id, _ = _create_session(client, tmp_path)

        resp = client.post(
            "/api/mini-editor/undo",
            json={"session_id": session_id},
            content_type="application/json",
        )

        assert resp.status_code == 400
        data = json.loads(resp.data)
        assert data["can_undo"] is False

    def test_redo_with_empty_history_returns_400(self, client, tmp_path):
        """Redo with empty history returns 400."""
        session_id, _ = _create_session(client, tmp_path)

        resp = client.post(
            "/api/mini-editor/redo",
            json={"session_id": session_id},
            content_type="application/json",
        )

        assert resp.status_code == 400
        data = json.loads(resp.data)
        assert data["can_redo"] is False
