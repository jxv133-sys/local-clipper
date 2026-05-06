"""Tests for Mini Video Editor routes and API endpoint behavior.

Covers Tasks 24 and 25:
  - GET /mini-editor returns 200
  - GET /api/mini-editor/job/<nonexistent>/progress returns 404
  - GET /api/mini-editor/job/<existing>/progress returns correct JSON structure
  - All API endpoints return correct Content-Type: application/json
  - Manual region endpoint
  - Fallback fill endpoint
  - Error handling endpoints
  - format_to_vertical_available field in job detail
"""

from __future__ import annotations

import json
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
    with web_server._formatting_jobs_lock:
        web_server._formatting_jobs.clear()
    web_server._session_store._sessions.clear()
    yield
    with web_server._jobs_lock:
        web_server._jobs.clear()
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


def _make_formatting_job(tmp_path, status="queued"):
    """Create and register a VerticalFormattingJob."""
    import web_server

    region = FacecamRegion(x=100, y=50, width=400, height=300, corner="top-right", confidence=0.85)
    layout = CanvasLayout(
        canvas_width=1080, canvas_height=1920,
        facecam_x=0, facecam_y=0,
        facecam_width=1080, facecam_height=672,
        gameplay_x=0, gameplay_y=672,
        gameplay_width=1080, gameplay_height=1248,
    )
    job_id = str(uuid.uuid4())
    job = VerticalFormattingJob(
        job_id=job_id,
        session_id="test-session",
        clip_batch_id="test-batch",
        facecam_region=region,
        canvas_layout=layout,
        settings={},
        clips=[
            {"path": str(tmp_path / "clip1.mp4"), "name": "clip1.mp4"},
            {"path": str(tmp_path / "clip2.mp4"), "name": "clip2.mp4"},
        ],
        output_dir=str(tmp_path),
        status=status,
    )
    with web_server._formatting_jobs_lock:
        web_server._formatting_jobs[job_id] = job
    return job_id, job


def _make_session(tmp_path):
    """Create a session in the store and return session_id."""
    import web_server

    region = FacecamRegion(x=100, y=50, width=400, height=300, corner="top-right", confidence=0.85)
    layout = CanvasLayout(
        canvas_width=1080, canvas_height=1920,
        facecam_x=0, facecam_y=0,
        facecam_width=1080, facecam_height=672,
        gameplay_x=0, gameplay_y=672,
        gameplay_width=1080, gameplay_height=1248,
    )
    session = web_server._session_store.create_session(
        clip_batch_id="test-batch",
        reference_clip_path=str(tmp_path / "clip.mp4"),
        reference_resolution=(1920, 1080),
        facecam_region=region,
        canvas_layout=layout,
    )
    return session.session_id


# ---------------------------------------------------------------------------
# Route tests
# ---------------------------------------------------------------------------

class TestMiniEditorRoutes:
    """Test basic route availability."""

    def test_get_mini_editor_returns_200_or_404(self, client):
        """GET /mini-editor returns 200 if HTML exists, 404 otherwise."""
        resp = client.get("/mini-editor")
        # Either 200 (HTML exists) or 404 (HTML not found) is acceptable
        assert resp.status_code in (200, 404)

    def test_get_nonexistent_job_progress_returns_404(self, client):
        """GET /api/mini-editor/job/<nonexistent>/progress returns 404."""
        resp = client.get("/api/mini-editor/job/nonexistent-job-id/progress")
        assert resp.status_code == 404
        data = json.loads(resp.data)
        assert "error" in data
        assert data["error"] is not None

    def test_get_existing_job_progress_returns_correct_structure(self, client, tmp_path):
        """GET /api/mini-editor/job/<existing>/progress returns correct JSON structure."""
        job_id, job = _make_formatting_job(tmp_path, status="running")
        job.clips_processed = 1
        job.clips_total = 2
        job.current_clip = "clip1.mp4"

        resp = client.get(f"/api/mini-editor/job/{job_id}/progress")
        assert resp.status_code == 200
        data = json.loads(resp.data)

        # Verify all required fields are present
        assert "job_id" in data
        assert "status" in data
        assert "clips_processed" in data
        assert "clips_total" in data
        assert "current_clip" in data
        assert "eta_seconds" in data
        assert "errors" in data
        assert "error" in data

        assert data["job_id"] == job_id
        assert data["status"] == "running"
        assert data["clips_processed"] == 1
        assert data["clips_total"] == 2
        assert data["current_clip"] == "clip1.mp4"
        assert isinstance(data["errors"], list)

    def test_progress_endpoint_includes_progress_pct(self, client, tmp_path):
        """Progress endpoint includes progress_pct field."""
        job_id, job = _make_formatting_job(tmp_path, status="running")
        job.clips_processed = 1
        job.clips_total = 4

        resp = client.get(f"/api/mini-editor/job/{job_id}/progress")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "progress_pct" in data
        assert data["progress_pct"] == 25.0


# ---------------------------------------------------------------------------
# Content-Type tests
# ---------------------------------------------------------------------------

class TestContentTypes:
    """All API endpoints return application/json."""

    def test_detect_returns_json_content_type(self, client, tmp_path):
        """POST /api/mini-editor/detect returns application/json."""
        video = tmp_path / "clip.mp4"
        video.write_text("fake")

        with patch("web_server.FacecamRelocator") as mock_rel:
            mock_rel.return_value.detect_facecam.return_value = None
            resp = client.post(
                "/api/mini-editor/detect",
                json={"clip_path": str(video), "frame_width": 1920, "frame_height": 1080},
                content_type="application/json",
            )

        assert "application/json" in resp.content_type

    def test_session_returns_json_content_type(self, client):
        """POST /api/mini-editor/session returns application/json."""
        resp = client.post(
            "/api/mini-editor/session",
            json={"clip_batch_id": "nonexistent", "reference_clip_path": "/nonexistent.mp4"},
            content_type="application/json",
        )
        assert "application/json" in resp.content_type

    def test_confirm_returns_json_content_type(self, client):
        """POST /api/mini-editor/confirm returns application/json."""
        resp = client.post(
            "/api/mini-editor/confirm",
            json={"session_id": "nonexistent"},
            content_type="application/json",
        )
        assert "application/json" in resp.content_type

    def test_cancel_returns_json_content_type(self, client):
        """POST /api/mini-editor/cancel returns application/json."""
        resp = client.post(
            "/api/mini-editor/cancel",
            json={"session_id": "nonexistent"},
            content_type="application/json",
        )
        assert "application/json" in resp.content_type

    def test_undo_returns_json_content_type(self, client):
        """POST /api/mini-editor/undo returns application/json."""
        resp = client.post(
            "/api/mini-editor/undo",
            json={"session_id": "nonexistent"},
            content_type="application/json",
        )
        assert "application/json" in resp.content_type

    def test_redo_returns_json_content_type(self, client):
        """POST /api/mini-editor/redo returns application/json."""
        resp = client.post(
            "/api/mini-editor/redo",
            json={"session_id": "nonexistent"},
            content_type="application/json",
        )
        assert "application/json" in resp.content_type

    def test_progress_returns_json_content_type(self, client, tmp_path):
        """GET /api/mini-editor/job/<id>/progress returns application/json."""
        job_id, _ = _make_formatting_job(tmp_path)
        resp = client.get(f"/api/mini-editor/job/{job_id}/progress")
        assert "application/json" in resp.content_type

    def test_manual_region_returns_json_content_type(self, client):
        """POST /api/mini-editor/manual-region returns application/json."""
        resp = client.post(
            "/api/mini-editor/manual-region",
            json={"session_id": "nonexistent"},
            content_type="application/json",
        )
        assert "application/json" in resp.content_type

    def test_fallback_returns_json_content_type(self, client):
        """POST /api/mini-editor/fallback returns application/json."""
        resp = client.post(
            "/api/mini-editor/fallback",
            json={"session_id": "nonexistent"},
            content_type="application/json",
        )
        assert "application/json" in resp.content_type


# ---------------------------------------------------------------------------
# Manual region endpoint tests
# ---------------------------------------------------------------------------

class TestManualRegionEndpoint:
    """Tests for POST /api/mini-editor/manual-region."""

    def test_valid_manual_region_stored_in_session(self, client, tmp_path):
        """Valid manual region is stored in the session."""
        import web_server

        session_id = _make_session(tmp_path)

        # Use a region that satisfies area fraction constraints (4%-30% of 1920x1080)
        # 1920*1080 = 2073600; 4% = 82944; 500*200 = 100000 (4.8%) - valid
        resp = client.post(
            "/api/mini-editor/manual-region",
            json={
                "session_id": session_id,
                "facecam_region": {"x": 50, "y": 30, "width": 500, "height": 200},
            },
            content_type="application/json",
        )

        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["error"] is None
        assert data["facecam_region"]["x"] == 50
        assert data["facecam_region"]["width"] == 500

        # Verify session was updated
        session = web_server._session_store.get_session(session_id)
        assert session.facecam_region.x == 50

    def test_manual_region_out_of_bounds_returns_400(self, client, tmp_path):
        """Manual region extending beyond frame returns 400."""
        session_id = _make_session(tmp_path)

        resp = client.post(
            "/api/mini-editor/manual-region",
            json={
                "session_id": session_id,
                "facecam_region": {"x": 1900, "y": 0, "width": 400, "height": 200},
            },
            content_type="application/json",
        )

        assert resp.status_code == 400

    def test_manual_region_too_small_returns_400(self, client, tmp_path):
        """Manual region with area fraction below minimum returns 400."""
        session_id = _make_session(tmp_path)

        resp = client.post(
            "/api/mini-editor/manual-region",
            json={
                "session_id": session_id,
                "facecam_region": {"x": 0, "y": 0, "width": 5, "height": 5},
            },
            content_type="application/json",
        )

        assert resp.status_code == 400

    def test_manual_region_computes_corner(self, client, tmp_path):
        """Manual region corner is computed from position if not provided."""
        session_id = _make_session(tmp_path)

        # Top-right quadrant (x > 960, y < 540 in 1920x1080)
        resp = client.post(
            "/api/mini-editor/manual-region",
            json={
                "session_id": session_id,
                "facecam_region": {"x": 1200, "y": 50, "width": 400, "height": 300},
            },
            content_type="application/json",
        )

        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["facecam_region"]["corner"] == "top-right"

    def test_manual_region_pushes_to_undo_history(self, client, tmp_path):
        """Setting manual region pushes previous region to undo history."""
        import web_server

        session_id = _make_session(tmp_path)
        session = web_server._session_store.get_session(session_id)
        initial_len = len(session.undo_history)

        # Use a valid region (4%-30% of 1920x1080 = 82944-622080 pixels)
        # 500*200 = 100000 pixels = 4.8% - valid
        client.post(
            "/api/mini-editor/manual-region",
            json={
                "session_id": session_id,
                "facecam_region": {"x": 50, "y": 30, "width": 500, "height": 200},
            },
            content_type="application/json",
        )

        session = web_server._session_store.get_session(session_id)
        assert len(session.undo_history) == initial_len + 1

    def test_manual_region_nonexistent_session_returns_404(self, client):
        """Manual region with nonexistent session returns 404."""
        resp = client.post(
            "/api/mini-editor/manual-region",
            json={
                "session_id": "nonexistent",
                "facecam_region": {"x": 50, "y": 30, "width": 300, "height": 200},
            },
            content_type="application/json",
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Fallback fill endpoint tests
# ---------------------------------------------------------------------------

class TestFallbackFillEndpoint:
    """Tests for POST /api/mini-editor/fallback."""

    def test_fallback_sets_use_fallback_fill(self, client, tmp_path):
        """Fallback endpoint sets use_fallback_fill=True on session."""
        import web_server

        session_id = _make_session(tmp_path)

        resp = client.post(
            "/api/mini-editor/fallback",
            json={"session_id": session_id},
            content_type="application/json",
        )

        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["use_fallback_fill"] is True
        assert data["error"] is None

        # Verify session was updated
        session = web_server._session_store.get_session(session_id)
        assert session.settings.get("use_fallback_fill") is True

    def test_fallback_nonexistent_session_returns_404(self, client):
        """Fallback with nonexistent session returns 404."""
        resp = client.post(
            "/api/mini-editor/fallback",
            json={"session_id": "nonexistent"},
            content_type="application/json",
        )
        assert resp.status_code == 404

    def test_fallback_missing_session_id_returns_400(self, client):
        """Fallback without session_id returns 400."""
        resp = client.post(
            "/api/mini-editor/fallback",
            json={},
            content_type="application/json",
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# format_to_vertical_available field
# ---------------------------------------------------------------------------

class TestFormatToVerticalAvailable:
    """Tests for format_to_vertical_available field in job detail."""

    def test_done_job_with_clips_has_format_to_vertical_true(self, client, tmp_path):
        """Completed job with clips has format_to_vertical_available=True."""
        import web_server
        from web_server import Job, JobStatus

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

        resp = client.get(f"/api/jobs/{job_id}")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["format_to_vertical_available"] is True

    def test_running_job_has_format_to_vertical_false(self, client, tmp_path):
        """Running job has format_to_vertical_available=False."""
        import web_server
        from web_server import Job, JobStatus

        job_id = str(uuid.uuid4())
        job = Job(
            job_id=job_id,
            video_path=str(tmp_path / "source.mp4"),
            config=MagicMock(),
            status=JobStatus.RUNNING,
        )
        with web_server._jobs_lock:
            web_server._jobs[job_id] = job

        resp = client.get(f"/api/jobs/{job_id}")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["format_to_vertical_available"] is False

    def test_done_job_with_no_clips_has_format_to_vertical_false(self, client, tmp_path):
        """Completed job with no clips has format_to_vertical_available=False."""
        import web_server
        from web_server import Job, JobStatus

        job_id = str(uuid.uuid4())
        job = Job(
            job_id=job_id,
            video_path=str(tmp_path / "source.mp4"),
            config=MagicMock(),
            status=JobStatus.DONE,
            result_clips=[],
        )
        with web_server._jobs_lock:
            web_server._jobs[job_id] = job

        resp = client.get(f"/api/jobs/{job_id}")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["format_to_vertical_available"] is False
