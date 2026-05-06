"""Tests for mini video editor API endpoints."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock

import pytest

from pipeline.models import FacecamRegion, CanvasLayout


@pytest.fixture
def app():
    """Create a Flask test client."""
    # Import here to avoid circular imports
    import web_server
    web_server.app.config["TESTING"] = True
    return web_server.app


@pytest.fixture
def client(app):
    """Create a test client for the Flask app."""
    return app.test_client()


@pytest.fixture
def sample_facecam_region():
    """Create a sample facecam region for testing."""
    return FacecamRegion(
        x=100,
        y=50,
        width=400,
        height=300,
        corner="top-left",
        confidence=0.85,
    )


class TestDetectFacecamEndpoint:
    """Tests for POST /api/mini-editor/detect endpoint."""

    def test_detect_success(self, client, sample_facecam_region, tmp_path):
        """Test successful facecam detection."""
        # Create a temporary video file
        video_file = tmp_path / "test_video.mp4"
        video_file.write_text("fake video content")

        with patch("web_server.FacecamRelocator") as mock_relocator_class:
            mock_relocator = MagicMock()
            mock_relocator_class.return_value = mock_relocator
            mock_relocator.detect_facecam.return_value = sample_facecam_region

            response = client.post(
                "/api/mini-editor/detect",
                json={
                    "clip_path": str(video_file),
                    "frame_width": 1920,
                    "frame_height": 1080,
                },
                content_type="application/json",
            )

            assert response.status_code == 200
            data = json.loads(response.data)
            assert data["error"] is None
            assert data["facecam_region"] is not None
            assert data["facecam_region"]["x"] == 100
            assert data["facecam_region"]["y"] == 50
            assert data["facecam_region"]["width"] == 400
            assert data["facecam_region"]["height"] == 300
            assert data["facecam_region"]["corner"] == "top-left"
            assert data["facecam_region"]["confidence"] == 0.85

    def test_detect_no_facecam_found(self, client, tmp_path):
        """Test detection when no facecam is found."""
        video_file = tmp_path / "test_video.mp4"
        video_file.write_text("fake video content")

        with patch("web_server.FacecamRelocator") as mock_relocator_class:
            mock_relocator = MagicMock()
            mock_relocator_class.return_value = mock_relocator
            mock_relocator.detect_facecam.return_value = None

            response = client.post(
                "/api/mini-editor/detect",
                json={
                    "clip_path": str(video_file),
                    "frame_width": 1920,
                    "frame_height": 1080,
                },
                content_type="application/json",
            )

            assert response.status_code == 200
            data = json.loads(response.data)
            assert data["facecam_region"] is None
            assert data["error"] == "No valid facecam region detected"

    def test_detect_missing_clip_path(self, client):
        """Test detection with missing clip_path."""
        response = client.post(
            "/api/mini-editor/detect",
            json={
                "frame_width": 1920,
                "frame_height": 1080,
            },
            content_type="application/json",
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert "clip_path is required" in data["error"]

    def test_detect_missing_frame_width(self, client, tmp_path):
        """Test detection with missing frame_width."""
        video_file = tmp_path / "test_video.mp4"
        video_file.write_text("fake video content")

        response = client.post(
            "/api/mini-editor/detect",
            json={
                "clip_path": str(video_file),
                "frame_height": 1080,
            },
            content_type="application/json",
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert "frame_width must be an integer" in data["error"]

    def test_detect_missing_frame_height(self, client, tmp_path):
        """Test detection with missing frame_height."""
        video_file = tmp_path / "test_video.mp4"
        video_file.write_text("fake video content")

        response = client.post(
            "/api/mini-editor/detect",
            json={
                "clip_path": str(video_file),
                "frame_width": 1920,
            },
            content_type="application/json",
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert "frame_height must be an integer" in data["error"]

    def test_detect_invalid_frame_width_type(self, client, tmp_path):
        """Test detection with invalid frame_width type."""
        video_file = tmp_path / "test_video.mp4"
        video_file.write_text("fake video content")

        response = client.post(
            "/api/mini-editor/detect",
            json={
                "clip_path": str(video_file),
                "frame_width": "not an integer",
                "frame_height": 1080,
            },
            content_type="application/json",
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert "frame_width must be an integer" in data["error"]

    def test_detect_clip_not_found(self, client):
        """Test detection with non-existent clip file."""
        response = client.post(
            "/api/mini-editor/detect",
            json={
                "clip_path": "/nonexistent/video.mp4",
                "frame_width": 1920,
                "frame_height": 1080,
            },
            content_type="application/json",
        )

        assert response.status_code == 404
        data = json.loads(response.data)
        assert "Clip not found" in data["error"]

    def test_detect_with_config_overrides(self, client, sample_facecam_region, tmp_path):
        """Test detection with custom config overrides."""
        video_file = tmp_path / "test_video.mp4"
        video_file.write_text("fake video content")

        with patch("web_server.FacecamRelocator") as mock_relocator_class:
            mock_relocator = MagicMock()
            mock_relocator_class.return_value = mock_relocator
            mock_relocator.detect_facecam.return_value = sample_facecam_region

            response = client.post(
                "/api/mini-editor/detect",
                json={
                    "clip_path": str(video_file),
                    "frame_width": 1920,
                    "frame_height": 1080,
                    "config": {
                        "facecam_sample_duration": 5.0,
                        "facecam_min_area_fraction": 0.05,
                    },
                },
                content_type="application/json",
            )

            assert response.status_code == 200
            data = json.loads(response.data)
            assert data["error"] is None
            assert data["facecam_region"] is not None

            # Verify config was passed to detect_facecam
            mock_relocator.detect_facecam.assert_called_once()
            call_args = mock_relocator.detect_facecam.call_args
            config = call_args.kwargs["config"]
            assert config.facecam_sample_duration == 5.0
            assert config.facecam_min_area_fraction == 0.05

    def test_detect_invalid_json(self, client):
        """Test detection with invalid JSON body."""
        response = client.post(
            "/api/mini-editor/detect",
            data="not json",
            content_type="application/json",
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert "Request body must be JSON" in data["error"]

    def test_detect_internal_error(self, client, tmp_path):
        """Test detection with internal error during processing."""
        video_file = tmp_path / "test_video.mp4"
        video_file.write_text("fake video content")

        with patch("web_server.FacecamRelocator") as mock_relocator_class:
            mock_relocator = MagicMock()
            mock_relocator_class.return_value = mock_relocator
            mock_relocator.detect_facecam.side_effect = Exception("Test error")

            response = client.post(
                "/api/mini-editor/detect",
                json={
                    "clip_path": str(video_file),
                    "frame_width": 1920,
                    "frame_height": 1080,
                },
                content_type="application/json",
            )

            assert response.status_code == 500
            data = json.loads(response.data)
            assert data["facecam_region"] is None
            assert "Internal error" in data["error"]

    def test_detect_caching(self, client, sample_facecam_region, tmp_path):
        """Test that detection results are cached and reused."""
        video_file = tmp_path / "test_video.mp4"
        video_file.write_text("fake video content")

        # Clear the cache before testing
        import web_server
        with web_server._detection_cache_lock:
            web_server._detection_cache.clear()

        with patch("web_server.FacecamRelocator") as mock_relocator_class:
            mock_relocator = MagicMock()
            mock_relocator_class.return_value = mock_relocator
            mock_relocator.detect_facecam.return_value = sample_facecam_region

            # First request - should call detect_facecam
            response1 = client.post(
                "/api/mini-editor/detect",
                json={
                    "clip_path": str(video_file),
                    "frame_width": 1920,
                    "frame_height": 1080,
                },
                content_type="application/json",
            )

            assert response1.status_code == 200
            data1 = json.loads(response1.data)
            assert data1["cached"] is False
            assert data1["facecam_region"] is not None
            assert mock_relocator.detect_facecam.call_count == 1

            # Second request with same parameters - should use cache
            response2 = client.post(
                "/api/mini-editor/detect",
                json={
                    "clip_path": str(video_file),
                    "frame_width": 1920,
                    "frame_height": 1080,
                },
                content_type="application/json",
            )

            assert response2.status_code == 200
            data2 = json.loads(response2.data)
            assert data2["cached"] is True
            assert data2["facecam_region"] == data1["facecam_region"]
            # Should still be 1 - not called again
            assert mock_relocator.detect_facecam.call_count == 1

    def test_detect_cache_invalidation_on_file_update(self, client, sample_facecam_region, tmp_path):
        """Test that cache is invalidated when file is modified."""
        video_file = tmp_path / "test_video.mp4"
        video_file.write_text("fake video content")

        # Clear the cache before testing
        import web_server
        import time
        with web_server._detection_cache_lock:
            web_server._detection_cache.clear()

        with patch("web_server.FacecamRelocator") as mock_relocator_class:
            mock_relocator = MagicMock()
            mock_relocator_class.return_value = mock_relocator
            mock_relocator.detect_facecam.return_value = sample_facecam_region

            # First request
            response1 = client.post(
                "/api/mini-editor/detect",
                json={
                    "clip_path": str(video_file),
                    "frame_width": 1920,
                    "frame_height": 1080,
                },
                content_type="application/json",
            )

            assert response1.status_code == 200
            data1 = json.loads(response1.data)
            assert data1["cached"] is False
            assert mock_relocator.detect_facecam.call_count == 1

            # Modify the file (update mtime)
            time.sleep(0.01)  # Ensure mtime changes
            video_file.write_text("modified video content")

            # Second request after file modification - should not use cache
            response2 = client.post(
                "/api/mini-editor/detect",
                json={
                    "clip_path": str(video_file),
                    "frame_width": 1920,
                    "frame_height": 1080,
                },
                content_type="application/json",
            )

            assert response2.status_code == 200
            data2 = json.loads(response2.data)
            assert data2["cached"] is False
            # Should be called again due to cache invalidation
            assert mock_relocator.detect_facecam.call_count == 2

    def test_detect_cache_different_dimensions(self, client, sample_facecam_region, tmp_path):
        """Test that cache is separate for different frame dimensions."""
        video_file = tmp_path / "test_video.mp4"
        video_file.write_text("fake video content")

        # Clear the cache before testing
        import web_server
        with web_server._detection_cache_lock:
            web_server._detection_cache.clear()

        with patch("web_server.FacecamRelocator") as mock_relocator_class:
            mock_relocator = MagicMock()
            mock_relocator_class.return_value = mock_relocator
            mock_relocator.detect_facecam.return_value = sample_facecam_region

            # First request with 1920x1080
            response1 = client.post(
                "/api/mini-editor/detect",
                json={
                    "clip_path": str(video_file),
                    "frame_width": 1920,
                    "frame_height": 1080,
                },
                content_type="application/json",
            )

            assert response1.status_code == 200
            data1 = json.loads(response1.data)
            assert data1["cached"] is False
            assert mock_relocator.detect_facecam.call_count == 1

            # Second request with different dimensions - should not use cache
            response2 = client.post(
                "/api/mini-editor/detect",
                json={
                    "clip_path": str(video_file),
                    "frame_width": 1280,
                    "frame_height": 720,
                },
                content_type="application/json",
            )

            assert response2.status_code == 200
            data2 = json.loads(response2.data)
            assert data2["cached"] is False
            # Should be called again with different dimensions
            assert mock_relocator.detect_facecam.call_count == 2

    def test_detect_cache_no_facecam_found(self, client, tmp_path):
        """Test that cache works correctly when no facecam is found."""
        video_file = tmp_path / "test_video.mp4"
        video_file.write_text("fake video content")

        # Clear the cache before testing
        import web_server
        with web_server._detection_cache_lock:
            web_server._detection_cache.clear()

        with patch("web_server.FacecamRelocator") as mock_relocator_class:
            mock_relocator = MagicMock()
            mock_relocator_class.return_value = mock_relocator
            mock_relocator.detect_facecam.return_value = None

            # First request - no facecam found
            response1 = client.post(
                "/api/mini-editor/detect",
                json={
                    "clip_path": str(video_file),
                    "frame_width": 1920,
                    "frame_height": 1080,
                },
                content_type="application/json",
            )

            assert response1.status_code == 200
            data1 = json.loads(response1.data)
            assert data1["cached"] is False
            assert data1["facecam_region"] is None
            assert mock_relocator.detect_facecam.call_count == 1

            # Second request - should use cached "no facecam" result
            response2 = client.post(
                "/api/mini-editor/detect",
                json={
                    "clip_path": str(video_file),
                    "frame_width": 1920,
                    "frame_height": 1080,
                },
                content_type="application/json",
            )

            assert response2.status_code == 200
            data2 = json.loads(response2.data)
            assert data2["cached"] is True
            assert data2["facecam_region"] is None
            # Should still be 1 - not called again
            assert mock_relocator.detect_facecam.call_count == 1



class TestCreateMiniEditorSession:
    """Tests for POST /api/mini-editor/session endpoint."""

    def test_create_session_success(self, client, sample_facecam_region, tmp_path):
        """Test successful session creation."""
        # Create temporary video files
        video1 = tmp_path / "clip1.mp4"
        video2 = tmp_path / "clip2.mp4"
        video1.write_text("fake video 1")
        video2.write_text("fake video 2")

        # Mock the job
        import web_server
        from web_server import Job, JobStatus
        
        job_id = "test-job-123"
        job = Job(
            job_id=job_id,
            video_path=str(tmp_path / "source.mp4"),
            config=MagicMock(),
            status=JobStatus.DONE,
            result_clips=[
                {"path": str(video1), "name": "clip1.mp4"},
                {"path": str(video2), "name": "clip2.mp4"},
            ],
        )
        
        with web_server._jobs_lock:
            web_server._jobs[job_id] = job

        # Mock FFprobe to return video resolution
        mock_ffprobe_output = json.dumps({
            "streams": [{"width": 1920, "height": 1080}]
        })
        
        # Mock FacecamRelocator
        with patch("web_server.FacecamRelocator") as mock_relocator_class, \
             patch("pipeline.frame_reformatter.compute_canvas_layout") as mock_canvas_layout, \
             patch("subprocess.run") as mock_subprocess:
            
            mock_relocator = MagicMock()
            mock_relocator_class.return_value = mock_relocator
            mock_relocator.detect_facecam.return_value = sample_facecam_region
            
            mock_canvas_layout.return_value = CanvasLayout(
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
            
            # Mock subprocess.run to return FFprobe output
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = mock_ffprobe_output
            mock_result.stderr = ""
            mock_subprocess.return_value = mock_result

            response = client.post(
                "/api/mini-editor/session",
                json={
                    "clip_batch_id": job_id,
                    "reference_clip_path": str(video1),
                },
                content_type="application/json",
            )

            assert response.status_code == 201
            data = json.loads(response.data)
            assert data["error"] is None
            assert data["session_id"] is not None
            assert len(data["clips"]) == 2
            assert data["clips"][0]["name"] == "clip1.mp4"
            assert data["clips"][0]["resolution"] == [1920, 1080]
            assert data["reference_clip"]["name"] == "clip1.mp4"
            assert data["reference_clip"]["resolution"] == [1920, 1080]

    def test_create_session_missing_clip_batch_id(self, client):
        """Test session creation with missing clip_batch_id."""
        response = client.post(
            "/api/mini-editor/session",
            json={
                "reference_clip_path": "/path/to/clip.mp4",
            },
            content_type="application/json",
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert "clip_batch_id is required" in data["error"]

    def test_create_session_missing_reference_clip_path(self, client):
        """Test session creation with missing reference_clip_path."""
        response = client.post(
            "/api/mini-editor/session",
            json={
                "clip_batch_id": "test-job-123",
            },
            content_type="application/json",
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert "reference_clip_path is required" in data["error"]

    def test_create_session_reference_clip_not_found(self, client):
        """Test session creation with non-existent reference clip."""
        response = client.post(
            "/api/mini-editor/session",
            json={
                "clip_batch_id": "test-job-123",
                "reference_clip_path": "/nonexistent/clip.mp4",
            },
            content_type="application/json",
        )

        assert response.status_code == 404
        data = json.loads(response.data)
        assert "Reference clip not found" in data["error"]

    def test_create_session_job_not_found(self, client, tmp_path):
        """Test session creation with non-existent job."""
        video = tmp_path / "clip.mp4"
        video.write_text("fake video")

        response = client.post(
            "/api/mini-editor/session",
            json={
                "clip_batch_id": "nonexistent-job",
                "reference_clip_path": str(video),
            },
            content_type="application/json",
        )

        assert response.status_code == 404
        data = json.loads(response.data)
        assert "Job not found" in data["error"]

    def test_create_session_job_not_complete(self, client, tmp_path):
        """Test session creation with incomplete job."""
        video = tmp_path / "clip.mp4"
        video.write_text("fake video")

        # Mock the job with RUNNING status
        import web_server
        from web_server import Job, JobStatus
        
        job_id = "test-job-running"
        job = Job(
            job_id=job_id,
            video_path=str(tmp_path / "source.mp4"),
            config=MagicMock(),
            status=JobStatus.RUNNING,
        )
        
        with web_server._jobs_lock:
            web_server._jobs[job_id] = job

        response = client.post(
            "/api/mini-editor/session",
            json={
                "clip_batch_id": job_id,
                "reference_clip_path": str(video),
            },
            content_type="application/json",
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert "Job is not complete yet" in data["error"]

    def test_create_session_job_has_no_clips(self, client, tmp_path):
        """Test session creation with job that has no clips."""
        video = tmp_path / "clip.mp4"
        video.write_text("fake video")

        # Mock the job with no clips
        import web_server
        from web_server import Job, JobStatus
        
        job_id = "test-job-no-clips"
        job = Job(
            job_id=job_id,
            video_path=str(tmp_path / "source.mp4"),
            config=MagicMock(),
            status=JobStatus.DONE,
            result_clips=[],
        )
        
        with web_server._jobs_lock:
            web_server._jobs[job_id] = job

        response = client.post(
            "/api/mini-editor/session",
            json={
                "clip_batch_id": job_id,
                "reference_clip_path": str(video),
            },
            content_type="application/json",
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert "Job has no clips" in data["error"]

    def test_create_session_with_detection_failure(self, client, tmp_path):
        """Test session creation when facecam detection fails (should use default)."""
        video1 = tmp_path / "clip1.mp4"
        video1.write_text("fake video 1")

        # Mock the job
        import web_server
        from web_server import Job, JobStatus
        
        job_id = "test-job-no-facecam"
        job = Job(
            job_id=job_id,
            video_path=str(tmp_path / "source.mp4"),
            config=MagicMock(),
            status=JobStatus.DONE,
            result_clips=[
                {"path": str(video1), "name": "clip1.mp4"},
            ],
        )
        
        with web_server._jobs_lock:
            web_server._jobs[job_id] = job

        # Mock FFprobe to return video resolution
        mock_ffprobe_output = json.dumps({
            "streams": [{"width": 1920, "height": 1080}]
        })
        
        # Mock FacecamRelocator to return None (detection failure)
        with patch("web_server.FacecamRelocator") as mock_relocator_class, \
             patch("pipeline.frame_reformatter.compute_canvas_layout") as mock_canvas_layout, \
             patch("subprocess.run") as mock_subprocess:
            
            mock_relocator = MagicMock()
            mock_relocator_class.return_value = mock_relocator
            mock_relocator.detect_facecam.return_value = None  # Detection fails
            
            mock_canvas_layout.return_value = CanvasLayout(
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
            
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = mock_ffprobe_output
            mock_result.stderr = ""
            mock_subprocess.return_value = mock_result

            response = client.post(
                "/api/mini-editor/session",
                json={
                    "clip_batch_id": job_id,
                    "reference_clip_path": str(video1),
                },
                content_type="application/json",
            )

            # Should succeed with default facecam region
            assert response.status_code == 201
            data = json.loads(response.data)
            assert data["error"] is None
            assert data["session_id"] is not None

    def test_create_session_ffprobe_failure(self, client, tmp_path):
        """Test session creation when FFprobe fails."""
        video1 = tmp_path / "clip1.mp4"
        video1.write_text("fake video 1")

        # Mock the job
        import web_server
        from web_server import Job, JobStatus
        
        job_id = "test-job-ffprobe-fail"
        job = Job(
            job_id=job_id,
            video_path=str(tmp_path / "source.mp4"),
            config=MagicMock(),
            status=JobStatus.DONE,
            result_clips=[
                {"path": str(video1), "name": "clip1.mp4"},
            ],
        )
        
        with web_server._jobs_lock:
            web_server._jobs[job_id] = job

        # Mock subprocess.run to return error
        with patch("subprocess.run") as mock_subprocess:
            mock_result = Mock()
            mock_result.returncode = 1
            mock_result.stdout = ""
            mock_result.stderr = "FFprobe error"
            mock_subprocess.return_value = mock_result

            response = client.post(
                "/api/mini-editor/session",
                json={
                    "clip_batch_id": job_id,
                    "reference_clip_path": str(video1),
                },
                content_type="application/json",
            )

            assert response.status_code == 500
            data = json.loads(response.data)
            assert "Failed to get video resolution" in data["error"]

    def test_create_session_invalid_json(self, client):
        """Test session creation with invalid JSON body."""
        response = client.post(
            "/api/mini-editor/session",
            data="not json",
            content_type="application/json",
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert "Request body must be JSON" in data["error"]


class TestConfirmPlacementEndpoint:
    """Tests for POST /api/mini-editor/confirm endpoint."""

    def _make_session(self, client, tmp_path, frame_width=1920, frame_height=1080):
        """Helper to create a session in the store and return its ID."""
        import web_server
        from pipeline.models import FacecamRegion, CanvasLayout, EditorSession
        import uuid

        region = FacecamRegion(
            x=100, y=50, width=400, height=300,
            corner="top-right", confidence=0.85,
        )
        canvas = CanvasLayout(
            canvas_width=1080, canvas_height=1920,
            facecam_x=0, facecam_y=0,
            facecam_width=1080, facecam_height=672,
            gameplay_x=0, gameplay_y=672,
            gameplay_width=1080, gameplay_height=1248,
        )
        session = web_server._session_store.create_session(
            clip_batch_id="test-batch",
            reference_clip_path="/fake/clip.mp4",
            reference_resolution=(frame_width, frame_height),
            facecam_region=region,
            canvas_layout=canvas,
        )

        # Register a fake completed job for the batch
        from web_server import Job, JobStatus
        video = tmp_path / "source.mp4"
        video.write_text("fake")
        clip1 = tmp_path / "clip1.mp4"
        clip1.write_text("fake clip")
        job = Job(
            job_id="test-batch",
            video_path=str(video),
            config=MagicMock(),
            status=JobStatus.DONE,
            result_clips=[{"path": str(clip1), "name": "clip1.mp4"}],
        )
        with web_server._jobs_lock:
            web_server._jobs["test-batch"] = job

        return session.session_id

    def test_confirm_valid_placement(self, client, tmp_path):
        """Test confirming a valid facecam placement creates a formatting job."""
        session_id = self._make_session(client, tmp_path)

        response = client.post(
            "/api/mini-editor/confirm",
            json={
                "session_id": session_id,
                "facecam_region": {
                    "x": 100,
                    "y": 50,
                    "width": 400,
                    "height": 300,
                    "corner": "top-right",
                    "confidence": 0.85,
                },
            },
            content_type="application/json",
        )

        assert response.status_code == 201
        data = json.loads(response.data)
        assert data["error"] is None
        assert data["job_id"] is not None
        assert data["status"] == "queued"

    def test_confirm_missing_session_id(self, client):
        """Test confirm with missing session_id."""
        response = client.post(
            "/api/mini-editor/confirm",
            json={
                "facecam_region": {
                    "x": 100, "y": 50, "width": 400, "height": 300,
                    "corner": "top-right", "confidence": 0.85,
                },
            },
            content_type="application/json",
        )
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "session_id is required" in data["error"]

    def test_confirm_missing_facecam_region(self, client, tmp_path):
        """Test confirm with missing facecam_region."""
        session_id = self._make_session(client, tmp_path)
        response = client.post(
            "/api/mini-editor/confirm",
            json={"session_id": session_id},
            content_type="application/json",
        )
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "facecam_region is required" in data["error"]

    def test_confirm_session_not_found(self, client):
        """Test confirm with non-existent session."""
        response = client.post(
            "/api/mini-editor/confirm",
            json={
                "session_id": "nonexistent-session",
                "facecam_region": {
                    "x": 100, "y": 50, "width": 400, "height": 300,
                    "corner": "top-right", "confidence": 0.85,
                },
            },
            content_type="application/json",
        )
        assert response.status_code == 404
        data = json.loads(response.data)
        assert "Session not found" in data["error"]

    def test_confirm_region_extends_beyond_frame_width(self, client, tmp_path):
        """Test confirm rejects region that extends beyond frame width."""
        session_id = self._make_session(client, tmp_path, frame_width=1920, frame_height=1080)

        response = client.post(
            "/api/mini-editor/confirm",
            json={
                "session_id": session_id,
                "facecam_region": {
                    "x": 1800,
                    "y": 50,
                    "width": 200,   # 1800 + 200 = 2000 > 1920
                    "height": 200,
                    "corner": "top-right",
                    "confidence": 0.5,
                },
            },
            content_type="application/json",
        )
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "frame width" in data["error"].lower() or "width" in data["error"].lower()

    def test_confirm_region_extends_beyond_frame_height(self, client, tmp_path):
        """Test confirm rejects region that extends beyond frame height."""
        session_id = self._make_session(client, tmp_path, frame_width=1920, frame_height=1080)

        response = client.post(
            "/api/mini-editor/confirm",
            json={
                "session_id": session_id,
                "facecam_region": {
                    "x": 100,
                    "y": 900,
                    "width": 200,
                    "height": 300,   # 900 + 300 = 1200 > 1080
                    "corner": "bottom-right",
                    "confidence": 0.5,
                },
            },
            content_type="application/json",
        )
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "frame height" in data["error"].lower() or "height" in data["error"].lower()

    def test_confirm_region_too_small(self, client, tmp_path):
        """Test confirm rejects region with area fraction below 4%."""
        session_id = self._make_session(client, tmp_path, frame_width=1920, frame_height=1080)
        # 1920*1080 = 2,073,600; 4% = 82,944; 10x10 = 100 << 82,944
        response = client.post(
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
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "too small" in data["error"].lower() or "minimum" in data["error"].lower()

    def test_confirm_region_too_large(self, client, tmp_path):
        """Test confirm rejects region with area fraction above 30%."""
        session_id = self._make_session(client, tmp_path, frame_width=1920, frame_height=1080)
        # 30% of 1920*1080 = 622,080; 900*700 = 630,000 > 622,080
        response = client.post(
            "/api/mini-editor/confirm",
            json={
                "session_id": session_id,
                "facecam_region": {
                    "x": 0, "y": 0, "width": 900, "height": 700,
                    "corner": "top-left", "confidence": 0.5,
                },
            },
            content_type="application/json",
        )
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "too large" in data["error"].lower() or "maximum" in data["error"].lower()

    def test_confirm_clears_undo_redo_history(self, client, tmp_path):
        """Test that confirming clears the undo/redo history."""
        import web_server
        session_id = self._make_session(client, tmp_path)

        # Add some undo history
        session = web_server._session_store.get_session(session_id)
        from pipeline.models import FacecamRegion
        session.push_undo(FacecamRegion(x=0, y=0, width=200, height=200, corner="top-left", confidence=0.5))

        assert len(session.undo_history) > 0

        response = client.post(
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
        assert response.status_code == 201

        # History should be cleared
        session = web_server._session_store.get_session(session_id)
        assert len(session.undo_history) == 0
        assert len(session.redo_history) == 0

    def test_confirm_invalid_json(self, client):
        """Test confirm with invalid JSON body."""
        response = client.post(
            "/api/mini-editor/confirm",
            data="not json",
            content_type="application/json",
        )
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "Request body must be JSON" in data["error"]


class TestCancelEditorSessionEndpoint:
    """Tests for POST /api/mini-editor/cancel endpoint."""

    def _make_session(self, tmp_path):
        """Helper to create a session in the store and return its ID."""
        import web_server
        from pipeline.models import FacecamRegion, CanvasLayout

        region = FacecamRegion(
            x=100, y=50, width=400, height=300,
            corner="top-right", confidence=0.85,
        )
        canvas = CanvasLayout(
            canvas_width=1080, canvas_height=1920,
            facecam_x=0, facecam_y=0,
            facecam_width=1080, facecam_height=672,
            gameplay_x=0, gameplay_y=672,
            gameplay_width=1080, gameplay_height=1248,
        )
        session = web_server._session_store.create_session(
            clip_batch_id="test-batch",
            reference_clip_path="/fake/clip.mp4",
            reference_resolution=(1920, 1080),
            facecam_region=region,
            canvas_layout=canvas,
        )
        return session.session_id

    def test_cancel_existing_session(self, client, tmp_path):
        """Test cancelling an existing session."""
        session_id = self._make_session(tmp_path)

        response = client.post(
            "/api/mini-editor/cancel",
            json={"session_id": session_id},
            content_type="application/json",
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "cancelled"
        assert data["error"] is None

    def test_cancel_removes_session(self, client, tmp_path):
        """Test that cancelling removes the session from the store."""
        import web_server
        session_id = self._make_session(tmp_path)

        client.post(
            "/api/mini-editor/cancel",
            json={"session_id": session_id},
            content_type="application/json",
        )

        # Session should no longer exist
        session = web_server._session_store.get_session(session_id)
        assert session is None

    def test_cancel_nonexistent_session_returns_cancelled(self, client):
        """Test cancelling a non-existent session returns cancelled (graceful)."""
        response = client.post(
            "/api/mini-editor/cancel",
            json={"session_id": "nonexistent-session"},
            content_type="application/json",
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "cancelled"
        assert data["error"] is None

    def test_cancel_missing_session_id(self, client):
        """Test cancel with missing session_id."""
        response = client.post(
            "/api/mini-editor/cancel",
            json={},
            content_type="application/json",
        )
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "session_id is required" in data["error"]

    def test_cancel_invalid_json(self, client):
        """Test cancel with invalid JSON body."""
        response = client.post(
            "/api/mini-editor/cancel",
            data="not json",
            content_type="application/json",
        )
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "Request body must be JSON" in data["error"]

    def test_cancel_clears_undo_redo_history(self, client, tmp_path):
        """Test that cancelling clears undo/redo history before removing session."""
        import web_server
        from pipeline.models import FacecamRegion
        session_id = self._make_session(tmp_path)

        # Add some undo history
        session = web_server._session_store.get_session(session_id)
        session.push_undo(FacecamRegion(x=0, y=0, width=200, height=200, corner="top-left", confidence=0.5))

        response = client.post(
            "/api/mini-editor/cancel",
            json={"session_id": session_id},
            content_type="application/json",
        )
        assert response.status_code == 200
        # Session is deleted — no history to check, but no error either
        assert json.loads(response.data)["status"] == "cancelled"


class TestUndoEndpoint:
    """Tests for POST /api/mini-editor/undo endpoint."""

    def _make_session_with_history(self, tmp_path, n_adjustments=3):
        """Create a session with N adjustments in undo history."""
        import web_server
        from pipeline.models import FacecamRegion, CanvasLayout

        initial_region = FacecamRegion(
            x=100, y=50, width=400, height=300,
            corner="top-right", confidence=0.85,
        )
        canvas = CanvasLayout(
            canvas_width=1080, canvas_height=1920,
            facecam_x=0, facecam_y=0,
            facecam_width=1080, facecam_height=672,
            gameplay_x=0, gameplay_y=672,
            gameplay_width=1080, gameplay_height=1248,
        )
        session = web_server._session_store.create_session(
            clip_batch_id="test-batch",
            reference_clip_path="/fake/clip.mp4",
            reference_resolution=(1920, 1080),
            facecam_region=initial_region,
            canvas_layout=canvas,
        )

        # Add adjustments to undo history
        for i in range(n_adjustments):
            session.push_undo(session.facecam_region)
            session.facecam_region = FacecamRegion(
                x=100 + i * 10, y=50 + i * 5,
                width=400, height=300,
                corner="top-right", confidence=0.8,
            )

        return session.session_id, initial_region

    def test_undo_restores_previous_region(self, client, tmp_path):
        """Test that undo restores the previous facecam region."""
        import web_server
        session_id, _ = self._make_session_with_history(tmp_path, n_adjustments=2)

        session = web_server._session_store.get_session(session_id)
        expected_region = session.undo_history[-1]  # top of undo stack

        response = client.post(
            "/api/mini-editor/undo",
            json={"session_id": session_id},
            content_type="application/json",
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["error"] is None
        assert data["facecam_region"] is not None
        assert data["facecam_region"]["x"] == expected_region.x
        assert data["facecam_region"]["y"] == expected_region.y

    def test_undo_empty_history_returns_400(self, client, tmp_path):
        """Test that undo with empty history returns 400."""
        import web_server
        from pipeline.models import FacecamRegion, CanvasLayout

        region = FacecamRegion(x=100, y=50, width=400, height=300, corner="top-right", confidence=0.85)
        canvas = CanvasLayout(
            canvas_width=1080, canvas_height=1920,
            facecam_x=0, facecam_y=0,
            facecam_width=1080, facecam_height=672,
            gameplay_x=0, gameplay_y=672,
            gameplay_width=1080, gameplay_height=1248,
        )
        session = web_server._session_store.create_session(
            clip_batch_id="test-batch",
            reference_clip_path="/fake/clip.mp4",
            reference_resolution=(1920, 1080),
            facecam_region=region,
            canvas_layout=canvas,
        )

        response = client.post(
            "/api/mini-editor/undo",
            json={"session_id": session.session_id},
            content_type="application/json",
        )
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "Nothing to undo" in data["error"]
        assert data["can_undo"] is False

    def test_undo_missing_session_id(self, client):
        """Test undo with missing session_id."""
        response = client.post(
            "/api/mini-editor/undo",
            json={},
            content_type="application/json",
        )
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "session_id is required" in data["error"]

    def test_undo_session_not_found(self, client):
        """Test undo with non-existent session."""
        response = client.post(
            "/api/mini-editor/undo",
            json={"session_id": "nonexistent"},
            content_type="application/json",
        )
        assert response.status_code == 404

    def test_undo_populates_redo_history(self, client, tmp_path):
        """Test that undo pushes the current region to redo history."""
        import web_server
        session_id, _ = self._make_session_with_history(tmp_path, n_adjustments=2)

        session = web_server._session_store.get_session(session_id)
        current_region_before_undo = session.facecam_region
        assert len(session.redo_history) == 0

        response = client.post(
            "/api/mini-editor/undo",
            json={"session_id": session_id},
            content_type="application/json",
        )
        assert response.status_code == 200

        session = web_server._session_store.get_session(session_id)
        assert len(session.redo_history) == 1
        assert session.redo_history[0].x == current_region_before_undo.x

    def test_undo_returns_can_undo_can_redo_flags(self, client, tmp_path):
        """Test that undo response includes can_undo and can_redo flags."""
        session_id, _ = self._make_session_with_history(tmp_path, n_adjustments=1)

        response = client.post(
            "/api/mini-editor/undo",
            json={"session_id": session_id},
            content_type="application/json",
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "can_undo" in data
        assert "can_redo" in data
        assert data["can_undo"] is False  # Only 1 adjustment, now undone
        assert data["can_redo"] is True   # Can redo the undone adjustment


class TestRedoEndpoint:
    """Tests for POST /api/mini-editor/redo endpoint."""

    def _make_session_with_redo_history(self, tmp_path, n_adjustments=2):
        """Create a session with redo history by applying and undoing adjustments."""
        import web_server
        from pipeline.models import FacecamRegion, CanvasLayout

        initial_region = FacecamRegion(
            x=100, y=50, width=400, height=300,
            corner="top-right", confidence=0.85,
        )
        canvas = CanvasLayout(
            canvas_width=1080, canvas_height=1920,
            facecam_x=0, facecam_y=0,
            facecam_width=1080, facecam_height=672,
            gameplay_x=0, gameplay_y=672,
            gameplay_width=1080, gameplay_height=1248,
        )
        session = web_server._session_store.create_session(
            clip_batch_id="test-batch",
            reference_clip_path="/fake/clip.mp4",
            reference_resolution=(1920, 1080),
            facecam_region=initial_region,
            canvas_layout=canvas,
        )

        # Apply adjustments
        adjusted_regions = []
        for i in range(n_adjustments):
            session.push_undo(session.facecam_region)
            new_region = FacecamRegion(
                x=200 + i * 10, y=100 + i * 5,
                width=400, height=300,
                corner="top-right", confidence=0.8,
            )
            session.facecam_region = new_region
            adjusted_regions.append(new_region)

        # Undo all to populate redo history
        while session.undo_history:
            prev = session.pop_undo()
            session.push_redo(session.facecam_region)
            session.facecam_region = prev

        return session.session_id, adjusted_regions

    def test_redo_reapplies_undone_adjustment(self, client, tmp_path):
        """Test that redo reapplies the last undone adjustment."""
        import web_server
        session_id, adjusted_regions = self._make_session_with_redo_history(tmp_path, n_adjustments=2)

        session = web_server._session_store.get_session(session_id)
        expected_region = session.redo_history[-1]  # top of redo stack

        response = client.post(
            "/api/mini-editor/redo",
            json={"session_id": session_id},
            content_type="application/json",
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["error"] is None
        assert data["facecam_region"] is not None
        assert data["facecam_region"]["x"] == expected_region.x
        assert data["facecam_region"]["y"] == expected_region.y

    def test_redo_empty_history_returns_400(self, client, tmp_path):
        """Test that redo with empty redo history returns 400."""
        import web_server
        from pipeline.models import FacecamRegion, CanvasLayout

        region = FacecamRegion(x=100, y=50, width=400, height=300, corner="top-right", confidence=0.85)
        canvas = CanvasLayout(
            canvas_width=1080, canvas_height=1920,
            facecam_x=0, facecam_y=0,
            facecam_width=1080, facecam_height=672,
            gameplay_x=0, gameplay_y=672,
            gameplay_width=1080, gameplay_height=1248,
        )
        session = web_server._session_store.create_session(
            clip_batch_id="test-batch",
            reference_clip_path="/fake/clip.mp4",
            reference_resolution=(1920, 1080),
            facecam_region=region,
            canvas_layout=canvas,
        )

        response = client.post(
            "/api/mini-editor/redo",
            json={"session_id": session.session_id},
            content_type="application/json",
        )
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "Nothing to redo" in data["error"]
        assert data["can_redo"] is False

    def test_redo_missing_session_id(self, client):
        """Test redo with missing session_id."""
        response = client.post(
            "/api/mini-editor/redo",
            json={},
            content_type="application/json",
        )
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "session_id is required" in data["error"]

    def test_redo_session_not_found(self, client):
        """Test redo with non-existent session."""
        response = client.post(
            "/api/mini-editor/redo",
            json={"session_id": "nonexistent"},
            content_type="application/json",
        )
        assert response.status_code == 404

    def test_redo_populates_undo_history(self, client, tmp_path):
        """Test that redo pushes the current region to undo history."""
        import web_server
        session_id, _ = self._make_session_with_redo_history(tmp_path, n_adjustments=2)

        session = web_server._session_store.get_session(session_id)
        current_region_before_redo = session.facecam_region
        initial_undo_len = len(session.undo_history)

        response = client.post(
            "/api/mini-editor/redo",
            json={"session_id": session_id},
            content_type="application/json",
        )
        assert response.status_code == 200

        session = web_server._session_store.get_session(session_id)
        assert len(session.undo_history) == initial_undo_len + 1
        assert session.undo_history[-1].x == current_region_before_redo.x

    def test_redo_returns_can_undo_can_redo_flags(self, client, tmp_path):
        """Test that redo response includes can_undo and can_redo flags."""
        session_id, _ = self._make_session_with_redo_history(tmp_path, n_adjustments=1)

        response = client.post(
            "/api/mini-editor/redo",
            json={"session_id": session_id},
            content_type="application/json",
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "can_undo" in data
        assert "can_redo" in data
        assert data["can_undo"] is True   # Can undo the reapplied adjustment
        assert data["can_redo"] is False  # Only 1 adjustment, now redone

    def test_undo_then_redo_roundtrip(self, client, tmp_path):
        """Test that undo followed by redo restores the original state."""
        import web_server
        from pipeline.models import FacecamRegion, CanvasLayout

        initial_region = FacecamRegion(
            x=100, y=50, width=400, height=300,
            corner="top-right", confidence=0.85,
        )
        canvas = CanvasLayout(
            canvas_width=1080, canvas_height=1920,
            facecam_x=0, facecam_y=0,
            facecam_width=1080, facecam_height=672,
            gameplay_x=0, gameplay_y=672,
            gameplay_width=1080, gameplay_height=1248,
        )
        session = web_server._session_store.create_session(
            clip_batch_id="test-batch",
            reference_clip_path="/fake/clip.mp4",
            reference_resolution=(1920, 1080),
            facecam_region=initial_region,
            canvas_layout=canvas,
        )
        session_id = session.session_id

        # Make an adjustment
        adjusted_region = FacecamRegion(
            x=200, y=100, width=400, height=300,
            corner="top-right", confidence=0.8,
        )
        session.push_undo(session.facecam_region)
        session.facecam_region = adjusted_region

        # Undo
        undo_resp = client.post(
            "/api/mini-editor/undo",
            json={"session_id": session_id},
            content_type="application/json",
        )
        assert undo_resp.status_code == 200
        undo_data = json.loads(undo_resp.data)
        assert undo_data["facecam_region"]["x"] == initial_region.x

        # Redo
        redo_resp = client.post(
            "/api/mini-editor/redo",
            json={"session_id": session_id},
            content_type="application/json",
        )
        assert redo_resp.status_code == 200
        redo_data = json.loads(redo_resp.data)
        assert redo_data["facecam_region"]["x"] == adjusted_region.x
