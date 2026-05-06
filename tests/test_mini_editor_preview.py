"""Tests for the mini-editor preview endpoint."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from config import Config
from pipeline.models import FacecamRegion


@pytest.fixture
def client():
    """Create a Flask test client."""
    from web_server import app
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_facecam_region():
    """Sample facecam region for testing."""
    return {
        "x": 100,
        "y": 50,
        "width": 400,
        "height": 300,
        "corner": "top-right",
        "confidence": 0.85,
    }


@pytest.fixture
def sample_canvas_layout():
    """Sample canvas layout for testing."""
    return {
        "canvas_width": 1080,
        "canvas_height": 1920,
        "facecam_x": 0,
        "facecam_y": 0,
        "facecam_width": 1080,
        "facecam_height": 672,
        "gameplay_x": 0,
        "gameplay_y": 672,
        "gameplay_width": 1080,
        "gameplay_height": 1248,
    }


class TestGeneratePreviewEndpoint:
    """Tests for POST /api/mini-editor/preview endpoint."""

    def test_preview_success(self, client, sample_facecam_region, sample_canvas_layout, tmp_path):
        """Test successful preview generation."""
        # Create a dummy video file
        video_file = tmp_path / "test_video.mp4"
        video_file.write_text("dummy video content")

        # Create a dummy preview image that FFmpeg would generate
        preview_image = tmp_path / "preview_test.jpg"
        preview_image.write_text("dummy preview image")

        with patch("subprocess.run") as mock_run, \
             patch("web_server.OUTPUT_DIR", tmp_path), \
             patch("web_server.uuid.uuid4") as mock_uuid:
            
            # Mock UUID for predictable preview filename
            mock_uuid.return_value.hex = "abcd1234" * 4
            
            # Mock FFmpeg success
            mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
            
            # Mock the preview file creation
            def create_preview(*args, **kwargs):
                preview_path = tmp_path / "preview_abcd1234.jpg"
                preview_path.write_text("generated preview")
                return MagicMock(returncode=0, stderr="", stdout="")
            
            mock_run.side_effect = create_preview

            response = client.post(
                "/api/mini-editor/preview",
                json={
                    "clip_path": str(video_file),
                    "facecam_region": sample_facecam_region,
                    "frame_width": 1920,
                    "frame_height": 1080,
                },
            )

            assert response.status_code == 200
            data = response.get_json()
            assert data["error"] is None
            assert data["cached"] is False
            assert "preview_image_url" in data
            assert data["preview_image_url"].startswith("/output/preview_")
            assert "canvas_layout" in data
            assert data["canvas_layout"]["canvas_width"] == 1080
            assert data["canvas_layout"]["canvas_height"] == 1920

    def test_preview_missing_clip_path(self, client, sample_facecam_region):
        """Test preview with missing clip_path."""
        response = client.post(
            "/api/mini-editor/preview",
            json={
                "facecam_region": sample_facecam_region,
                "frame_width": 1920,
                "frame_height": 1080,
            },
        )

        assert response.status_code == 400
        data = response.get_json()
        assert "clip_path is required" in data["error"]

    def test_preview_missing_facecam_region(self, client, tmp_path):
        """Test preview with missing facecam_region."""
        video_file = tmp_path / "test_video.mp4"
        video_file.write_text("dummy video content")

        response = client.post(
            "/api/mini-editor/preview",
            json={
                "clip_path": str(video_file),
                "frame_width": 1920,
                "frame_height": 1080,
            },
        )

        assert response.status_code == 400
        data = response.get_json()
        assert "facecam_region is required" in data["error"]

    def test_preview_missing_frame_dimensions(self, client, sample_facecam_region, tmp_path):
        """Test preview with missing frame dimensions."""
        video_file = tmp_path / "test_video.mp4"
        video_file.write_text("dummy video content")

        response = client.post(
            "/api/mini-editor/preview",
            json={
                "clip_path": str(video_file),
                "facecam_region": sample_facecam_region,
            },
        )

        assert response.status_code == 400
        data = response.get_json()
        assert "frame_width must be an integer" in data["error"]

    def test_preview_invalid_facecam_region(self, client, tmp_path):
        """Test preview with invalid facecam_region structure."""
        video_file = tmp_path / "test_video.mp4"
        video_file.write_text("dummy video content")

        response = client.post(
            "/api/mini-editor/preview",
            json={
                "clip_path": str(video_file),
                "facecam_region": {
                    "x": 100,
                    "y": 50,
                    # Missing width, height, corner, confidence
                },
                "frame_width": 1920,
                "frame_height": 1080,
            },
        )

        assert response.status_code == 400
        data = response.get_json()
        assert "facecam_region.width is required" in data["error"]

    def test_preview_nonexistent_clip(self, client, sample_facecam_region):
        """Test preview with non-existent clip file."""
        response = client.post(
            "/api/mini-editor/preview",
            json={
                "clip_path": "/nonexistent/video.mp4",
                "facecam_region": sample_facecam_region,
                "frame_width": 1920,
                "frame_height": 1080,
            },
        )

        assert response.status_code == 404
        data = response.get_json()
        assert "Clip not found" in data["error"]

    def test_preview_ffmpeg_failure(self, client, sample_facecam_region, tmp_path):
        """Test preview when FFmpeg fails."""
        video_file = tmp_path / "test_video.mp4"
        video_file.write_text("dummy video content")

        with patch("subprocess.run") as mock_run, \
             patch("web_server.OUTPUT_DIR", tmp_path):
            
            # Mock FFmpeg failure
            mock_run.return_value = MagicMock(
                returncode=1,
                stderr="FFmpeg error: invalid codec",
                stdout=""
            )

            response = client.post(
                "/api/mini-editor/preview",
                json={
                    "clip_path": str(video_file),
                    "facecam_region": sample_facecam_region,
                    "frame_width": 1920,
                    "frame_height": 1080,
                },
            )

            assert response.status_code == 500
            data = response.get_json()
            assert "Failed to generate preview" in data["error"]

    def test_preview_caching(self, client, sample_facecam_region, tmp_path):
        """Test that preview results are cached."""
        video_file = tmp_path / "test_video.mp4"
        video_file.write_text("dummy video content")

        with patch("subprocess.run") as mock_run, \
             patch("web_server.OUTPUT_DIR", tmp_path), \
             patch("web_server.uuid.uuid4") as mock_uuid:
            
            # Mock UUID for predictable preview filename
            mock_uuid_obj = MagicMock()
            mock_uuid_obj.hex = "cached1234" * 4
            mock_uuid.return_value = mock_uuid_obj
            
            # The actual filename will be preview_{first 8 chars of hex}.jpg
            preview_filename = f"preview_{mock_uuid_obj.hex[:8]}.jpg"
            preview_path = tmp_path / preview_filename
            
            def create_preview(*args, **kwargs):
                # Create the actual preview file so caching works
                preview_path.write_text("generated preview")
                return MagicMock(returncode=0, stderr="", stdout="")
            
            mock_run.side_effect = create_preview

            # First request - should call FFmpeg
            response1 = client.post(
                "/api/mini-editor/preview",
                json={
                    "clip_path": str(video_file),
                    "facecam_region": sample_facecam_region,
                    "frame_width": 1920,
                    "frame_height": 1080,
                },
            )

            assert response1.status_code == 200
            data1 = response1.get_json()
            assert data1["cached"] is False
            assert mock_run.call_count == 1
            
            # Verify the preview file was created
            assert preview_path.exists()

            # Second request with same parameters - should use cache
            response2 = client.post(
                "/api/mini-editor/preview",
                json={
                    "clip_path": str(video_file),
                    "facecam_region": sample_facecam_region,
                    "frame_width": 1920,
                    "frame_height": 1080,
                },
            )

            assert response2.status_code == 200
            data2 = response2.get_json()
            assert data2["cached"] is True  # Should use cache now
            assert mock_run.call_count == 1  # Should not call FFmpeg again

    def test_preview_cache_invalidation_on_file_change(self, client, sample_facecam_region, tmp_path):
        """Test that cache is invalidated when file is modified."""
        video_file = tmp_path / "test_video.mp4"
        video_file.write_text("dummy video content")

        with patch("subprocess.run") as mock_run, \
             patch("web_server.OUTPUT_DIR", tmp_path), \
             patch("web_server.uuid.uuid4") as mock_uuid:
            
            # Mock UUID for predictable preview filename
            mock_uuid.return_value.hex = "modified123" * 4
            
            # Mock the preview file creation
            def create_preview(*args, **kwargs):
                preview_path = tmp_path / f"preview_{mock_uuid.return_value.hex[:8]}.jpg"
                preview_path.write_text("generated preview")
                return MagicMock(returncode=0, stderr="", stdout="")
            
            mock_run.side_effect = create_preview

            # First request
            response1 = client.post(
                "/api/mini-editor/preview",
                json={
                    "clip_path": str(video_file),
                    "facecam_region": sample_facecam_region,
                    "frame_width": 1920,
                    "frame_height": 1080,
                },
            )

            assert response1.status_code == 200
            assert mock_run.call_count == 1

            # Modify the video file (change mtime)
            import time
            time.sleep(0.01)  # Ensure mtime changes
            video_file.write_text("modified video content")

            # Second request after file modification - should not use cache
            response2 = client.post(
                "/api/mini-editor/preview",
                json={
                    "clip_path": str(video_file),
                    "facecam_region": sample_facecam_region,
                    "frame_width": 1920,
                    "frame_height": 1080,
                },
            )

            assert response2.status_code == 200
            assert mock_run.call_count == 2  # Should call FFmpeg again

    def test_preview_different_facecam_regions(self, client, sample_facecam_region, tmp_path):
        """Test that different facecam regions generate different previews."""
        video_file = tmp_path / "test_video.mp4"
        video_file.write_text("dummy video content")

        with patch("subprocess.run") as mock_run, \
             patch("web_server.OUTPUT_DIR", tmp_path), \
             patch("web_server.uuid.uuid4") as mock_uuid:
            
            # Mock UUID for predictable preview filename
            mock_uuid.return_value.hex = "different12" * 4
            
            # Mock the preview file creation
            def create_preview(*args, **kwargs):
                preview_path = tmp_path / f"preview_{mock_uuid.return_value.hex[:8]}.jpg"
                preview_path.write_text("generated preview")
                return MagicMock(returncode=0, stderr="", stdout="")
            
            mock_run.side_effect = create_preview

            # First request with original facecam region
            response1 = client.post(
                "/api/mini-editor/preview",
                json={
                    "clip_path": str(video_file),
                    "facecam_region": sample_facecam_region,
                    "frame_width": 1920,
                    "frame_height": 1080,
                },
            )

            assert response1.status_code == 200
            assert mock_run.call_count == 1

            # Second request with different facecam region - should not use cache
            different_region = sample_facecam_region.copy()
            different_region["x"] = 200  # Different x position
            
            response2 = client.post(
                "/api/mini-editor/preview",
                json={
                    "clip_path": str(video_file),
                    "facecam_region": different_region,
                    "frame_width": 1920,
                    "frame_height": 1080,
                },
            )

            assert response2.status_code == 200
            assert mock_run.call_count == 2  # Should call FFmpeg again

    def test_preview_invalid_json(self, client):
        """Test preview with invalid JSON body."""
        response = client.post(
            "/api/mini-editor/preview",
            data="not json",
            content_type="application/json",
        )

        assert response.status_code == 400
        data = response.get_json()
        assert "Request body must be JSON" in data["error"]

    def test_preview_with_config_overrides(self, client, sample_facecam_region, tmp_path):
        """Test preview with custom config overrides."""
        video_file = tmp_path / "test_video.mp4"
        video_file.write_text("dummy video content")

        with patch("subprocess.run") as mock_run, \
             patch("web_server.OUTPUT_DIR", tmp_path), \
             patch("web_server.uuid.uuid4") as mock_uuid:
            
            # Mock UUID for predictable preview filename
            mock_uuid.return_value.hex = "config1234" * 4
            
            # Mock the preview file creation
            def create_preview(*args, **kwargs):
                preview_path = tmp_path / f"preview_{mock_uuid.return_value.hex[:8]}.jpg"
                preview_path.write_text("generated preview")
                return MagicMock(returncode=0, stderr="", stdout="")
            
            mock_run.side_effect = create_preview

            response = client.post(
                "/api/mini-editor/preview",
                json={
                    "clip_path": str(video_file),
                    "facecam_region": sample_facecam_region,
                    "frame_width": 1920,
                    "frame_height": 1080,
                    "config": {
                        "shorts_width": 1080,
                        "shorts_height": 1920,
                        "facecam_top_fraction": 0.4,  # Custom fraction
                    },
                },
            )

            assert response.status_code == 200
            data = response.get_json()
            assert data["error"] is None
            # Canvas layout should reflect custom config
            assert data["canvas_layout"]["canvas_width"] == 1080
            assert data["canvas_layout"]["canvas_height"] == 1920
