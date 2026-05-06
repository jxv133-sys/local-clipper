"""Integration tests for Mini Video Editor batch processing.

Tests cover:
  23.1 Single clip formatting (apply_placement_to_clip)
  23.2 Batch processing with multiple clips
  23.3 Resolution scaling
  23.4 Backup and replacement
  23.5 Cancellation
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, call

import pytest

from pipeline.models import CanvasLayout, FacecamRegion, VerticalFormattingJob
from pipeline.vertical_formatter import (
    VerticalFormatter,
    backup_clips,
    get_output_path,
    process_vertical_formatting_job,
    replace_clips_with_vertical,
    restore_from_backup,
    scale_region_to_resolution,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_region():
    return FacecamRegion(
        x=100, y=50, width=400, height=300,
        corner="top-right", confidence=0.85,
    )


@pytest.fixture
def sample_layout():
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
def mock_config():
    cfg = MagicMock()
    cfg.output_crf = 23
    cfg.output_codec = "libx264"
    return cfg


def _make_job(tmp_path, clips, region, layout, settings=None):
    """Create a VerticalFormattingJob for testing."""
    return VerticalFormattingJob(
        job_id="test-job-001",
        session_id="test-session-001",
        clip_batch_id="test-batch-001",
        facecam_region=region,
        canvas_layout=layout,
        settings=settings or {},
        clips=clips,
        output_dir=str(tmp_path),
        status="queued",
    )


# ---------------------------------------------------------------------------
# 23.1 Single clip formatting
# ---------------------------------------------------------------------------

class TestSingleClipFormatting:
    """23.1 Single clip formatting."""

    def test_apply_placement_to_clip_calls_ffmpeg(self, tmp_path, sample_region, sample_layout, mock_config):
        """apply_placement_to_clip() calls FFmpeg with correct arguments."""
        clip = tmp_path / "clip.mp4"
        clip.write_text("fake")
        output = tmp_path / "clip_vertical.mp4"

        formatter = VerticalFormatter()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

            formatter.apply_placement_to_clip(
                clip_path=str(clip),
                facecam_region=sample_region,
                canvas_layout=sample_layout,
                output_path=str(output),
                config=mock_config,
                clip_resolution=(1920, 1080),
            )

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "ffmpeg" in cmd[0]
        assert str(clip) in cmd
        assert str(output) in cmd

    def test_output_path_uses_vertical_suffix(self, tmp_path):
        """Output path appends _vertical suffix by default."""
        input_path = "/some/dir/clip_001.mp4"
        output = get_output_path(input_path, {}, str(tmp_path))
        assert output.endswith("clip_001_vertical.mp4")
        assert str(tmp_path) in output

    def test_output_path_custom_suffix(self, tmp_path):
        """Output path uses custom suffix from settings."""
        input_path = "/some/dir/clip_001.mp4"
        output = get_output_path(input_path, {"suffix": "_9x16"}, str(tmp_path))
        assert output.endswith("clip_001_9x16.mp4")

    def test_output_path_custom_prefix(self, tmp_path):
        """Output path uses custom prefix from settings."""
        input_path = "/some/dir/clip_001.mp4"
        output = get_output_path(input_path, {"prefix": "vertical_", "suffix": ""}, str(tmp_path))
        assert "vertical_clip_001" in output

    def test_audio_preservation_in_ffmpeg_command(self, tmp_path, sample_region, sample_layout, mock_config):
        """FFmpeg command includes -c:a copy to preserve audio."""
        clip = tmp_path / "clip.mp4"
        clip.write_text("fake")
        output = tmp_path / "clip_vertical.mp4"

        formatter = VerticalFormatter()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

            formatter.apply_placement_to_clip(
                clip_path=str(clip),
                facecam_region=sample_region,
                canvas_layout=sample_layout,
                output_path=str(output),
                config=mock_config,
                clip_resolution=(1920, 1080),
            )

        cmd = mock_run.call_args[0][0]
        # Find -c:a copy in the command
        cmd_str = " ".join(cmd)
        assert "-c:a copy" in cmd_str

    def test_ffmpeg_failure_raises_error(self, tmp_path, sample_region, sample_layout, mock_config):
        """FFmpeg failure raises CalledProcessError."""
        import subprocess
        clip = tmp_path / "clip.mp4"
        clip.write_text("fake")
        output = tmp_path / "clip_vertical.mp4"

        formatter = VerticalFormatter()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=1, stdout="", stderr="FFmpeg error")

            with pytest.raises(subprocess.CalledProcessError):
                formatter.apply_placement_to_clip(
                    clip_path=str(clip),
                    facecam_region=sample_region,
                    canvas_layout=sample_layout,
                    output_path=str(output),
                    config=mock_config,
                    clip_resolution=(1920, 1080),
                )


# ---------------------------------------------------------------------------
# 23.2 Batch processing
# ---------------------------------------------------------------------------

class TestBatchProcessing:
    """23.2 Batch processing with multiple clips."""

    def test_process_job_processes_all_clips(self, tmp_path, sample_region, sample_layout):
        """process_vertical_formatting_job() processes all clips in the batch."""
        clips = [
            {"path": str(tmp_path / f"clip{i}.mp4"), "name": f"clip{i}.mp4", "resolution": [1920, 1080]}
            for i in range(3)
        ]
        for c in clips:
            Path(c["path"]).write_text("fake")

        job = _make_job(tmp_path, clips, sample_region, sample_layout)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
            process_vertical_formatting_job(job)

        assert job.status == "done"
        assert job.clips_processed == 3
        assert len(job.errors) == 0

    def test_progress_tracking_updates_correctly(self, tmp_path, sample_region, sample_layout):
        """Progress counter increments after each clip."""
        clips = [
            {"path": str(tmp_path / f"clip{i}.mp4"), "name": f"clip{i}.mp4", "resolution": [1920, 1080]}
            for i in range(4)
        ]
        for c in clips:
            Path(c["path"]).write_text("fake")

        job = _make_job(tmp_path, clips, sample_region, sample_layout)
        progress_snapshots = []

        original_increment = job.increment_progress

        def tracking_increment(clip_name=""):
            original_increment(clip_name)
            progress_snapshots.append(job.clips_processed)

        job.increment_progress = tracking_increment

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
            process_vertical_formatting_job(job)

        assert progress_snapshots == [1, 2, 3, 4]
        assert job.get_progress_percentage() == 100.0

    def test_error_in_one_clip_does_not_stop_others(self, tmp_path, sample_region, sample_layout):
        """Error in one clip is logged but processing continues."""
        clips = [
            {"path": str(tmp_path / f"clip{i}.mp4"), "name": f"clip{i}.mp4", "resolution": [1920, 1080]}
            for i in range(3)
        ]
        for c in clips:
            Path(c["path"]).write_text("fake")

        job = _make_job(tmp_path, clips, sample_region, sample_layout)

        call_count = [0]

        def mock_run_side_effect(cmd, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:  # Fail on second clip
                return Mock(returncode=1, stdout="", stderr="Encoding error")
            return Mock(returncode=0, stdout="", stderr="")

        with patch("subprocess.run", side_effect=mock_run_side_effect):
            process_vertical_formatting_job(job)

        assert job.status == "done"
        assert job.clips_processed == 2  # 2 succeeded
        assert len(job.errors) == 1
        assert "clip1.mp4" in job.errors[0]


# ---------------------------------------------------------------------------
# 23.3 Resolution scaling
# ---------------------------------------------------------------------------

class TestResolutionScaling:
    """23.3 Resolution scaling."""

    def test_scale_region_1920x1080_to_1280x720(self):
        """Scale region from 1920x1080 to 1280x720 (2/3 scale)."""
        region = FacecamRegion(x=300, y=150, width=600, height=450, corner="top-right", confidence=0.9)
        scaled = scale_region_to_resolution(region, (1920, 1080), (1280, 720))

        assert scaled.x == 200   # 300 * (1280/1920)
        assert scaled.y == 100   # 150 * (720/1080)
        assert scaled.width == 400   # 600 * (1280/1920)
        assert scaled.height == 300  # 450 * (720/1080)
        assert scaled.corner == "top-right"
        assert scaled.confidence == 0.9

    def test_scale_region_vertical_source(self):
        """Scale region for vertical source (9:16 aspect ratio)."""
        region = FacecamRegion(x=50, y=100, width=200, height=150, corner="top-left", confidence=0.8)
        # From 1080x1920 to 540x960 (half scale)
        scaled = scale_region_to_resolution(region, (1080, 1920), (540, 960))

        assert scaled.x == 25
        assert scaled.y == 50
        assert scaled.width == 100
        assert scaled.height == 75

    def test_scale_region_square_source(self):
        """Scale region for square source (1:1 aspect ratio)."""
        region = FacecamRegion(x=100, y=100, width=200, height=200, corner="top-left", confidence=0.75)
        # From 1000x1000 to 500x500 (half scale)
        scaled = scale_region_to_resolution(region, (1000, 1000), (500, 500))

        assert scaled.x == 50
        assert scaled.y == 50
        assert scaled.width == 100
        assert scaled.height == 100

    def test_scale_region_same_resolution_unchanged(self):
        """Scaling to same resolution returns identical coordinates."""
        region = FacecamRegion(x=100, y=50, width=400, height=300, corner="top-right", confidence=0.85)
        scaled = scale_region_to_resolution(region, (1920, 1080), (1920, 1080))

        assert scaled.x == region.x
        assert scaled.y == region.y
        assert scaled.width == region.width
        assert scaled.height == region.height

    def test_scale_region_zero_from_resolution_returns_original(self):
        """Zero from_resolution returns original region without error."""
        region = FacecamRegion(x=100, y=50, width=400, height=300, corner="top-right", confidence=0.85)
        result = scale_region_to_resolution(region, (0, 0), (1920, 1080))
        assert result is region


# ---------------------------------------------------------------------------
# 23.4 Backup and replacement
# ---------------------------------------------------------------------------

class TestBackupAndReplacement:
    """23.4 Backup and replacement."""

    def test_backup_clips_creates_timestamped_directory(self, tmp_path):
        """backup_clips() creates a timestamped backup directory."""
        clips = []
        for i in range(3):
            p = tmp_path / f"clip{i}.mp4"
            p.write_text(f"content {i}")
            clips.append(str(p))

        backup_dir = backup_clips(clips, str(tmp_path))

        assert os.path.isdir(backup_dir)
        assert "backup_" in os.path.basename(backup_dir)
        # All clips should be in the backup dir
        for clip in clips:
            backup_copy = os.path.join(backup_dir, os.path.basename(clip))
            assert os.path.exists(backup_copy)

    def test_replace_clips_with_vertical_replaces_files(self, tmp_path):
        """replace_clips_with_vertical() replaces original files."""
        originals = []
        verticals = []
        for i in range(2):
            orig = tmp_path / f"clip{i}.mp4"
            vert = tmp_path / f"clip{i}_vertical.mp4"
            orig.write_text(f"original {i}")
            vert.write_text(f"vertical {i}")
            originals.append(str(orig))
            verticals.append(str(vert))

        replace_clips_with_vertical(originals, verticals, str(tmp_path), backup=False)

        # Originals should now contain vertical content
        for i, orig in enumerate(originals):
            assert Path(orig).read_text() == f"vertical {i}"

    def test_replace_clips_creates_backup_when_enabled(self, tmp_path):
        """replace_clips_with_vertical() creates backup when backup=True."""
        originals = []
        verticals = []
        for i in range(2):
            orig = tmp_path / f"clip{i}.mp4"
            vert = tmp_path / f"clip{i}_vertical.mp4"
            orig.write_text(f"original {i}")
            vert.write_text(f"vertical {i}")
            originals.append(str(orig))
            verticals.append(str(vert))

        backup_dir = replace_clips_with_vertical(originals, verticals, str(tmp_path), backup=True)

        assert backup_dir is not None
        assert os.path.isdir(backup_dir)
        # Backup should contain original content
        for i, orig in enumerate(originals):
            backup_copy = os.path.join(backup_dir, os.path.basename(orig))
            assert Path(backup_copy).read_text() == f"original {i}"

    def test_restore_from_backup_restores_originals(self, tmp_path):
        """restore_from_backup() restores original clips from backup."""
        originals = []
        for i in range(2):
            orig = tmp_path / f"clip{i}.mp4"
            orig.write_text(f"original {i}")
            originals.append(str(orig))

        # Create backup
        backup_dir = backup_clips(originals, str(tmp_path))

        # Overwrite originals
        for i, orig in enumerate(originals):
            Path(orig).write_text(f"overwritten {i}")

        # Restore
        restore_from_backup(backup_dir, originals)

        for i, orig in enumerate(originals):
            assert Path(orig).read_text() == f"original {i}"

    def test_restore_from_backup_raises_if_backup_missing(self, tmp_path):
        """restore_from_backup() raises FileNotFoundError if backup file missing."""
        orig = tmp_path / "clip.mp4"
        orig.write_text("content")

        fake_backup_dir = str(tmp_path / "backup_nonexistent")
        os.makedirs(fake_backup_dir)

        with pytest.raises(FileNotFoundError):
            restore_from_backup(fake_backup_dir, [str(orig)])


# ---------------------------------------------------------------------------
# 23.5 Cancellation
# ---------------------------------------------------------------------------

class TestBatchCancellation:
    """23.5 Batch processing cancellation."""

    def test_cancelled_job_stops_processing(self, tmp_path, sample_region, sample_layout):
        """Cancelling a job mid-batch stops processing."""
        clips = [
            {"path": str(tmp_path / f"clip{i}.mp4"), "name": f"clip{i}.mp4", "resolution": [1920, 1080]}
            for i in range(5)
        ]
        for c in clips:
            Path(c["path"]).write_text("fake")

        job = _make_job(tmp_path, clips, sample_region, sample_layout)

        call_count = [0]

        def mock_run_side_effect(cmd, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                # Cancel the job after second clip starts
                job.status = "cancelled"
            return Mock(returncode=0, stdout="", stderr="")

        with patch("subprocess.run", side_effect=mock_run_side_effect):
            process_vertical_formatting_job(job)

        assert job.status == "cancelled"
        # Should have processed at most 2 clips before cancellation
        assert job.clips_processed <= 2

    def test_pre_cancelled_job_processes_nothing(self, tmp_path, sample_region, sample_layout):
        """Job cancelled before processing starts processes no clips."""
        clips = [
            {"path": str(tmp_path / f"clip{i}.mp4"), "name": f"clip{i}.mp4", "resolution": [1920, 1080]}
            for i in range(3)
        ]
        for c in clips:
            Path(c["path"]).write_text("fake")

        job = _make_job(tmp_path, clips, sample_region, sample_layout)
        job.status = "cancelled"  # Cancel before processing

        with patch("subprocess.run") as mock_run:
            process_vertical_formatting_job(job)
            mock_run.assert_not_called()

        assert job.clips_processed == 0

    def test_already_processed_clips_preserved_on_cancel(self, tmp_path, sample_region, sample_layout):
        """Clips processed before cancellation are preserved."""
        clips = [
            {"path": str(tmp_path / f"clip{i}.mp4"), "name": f"clip{i}.mp4", "resolution": [1920, 1080]}
            for i in range(4)
        ]
        for c in clips:
            Path(c["path"]).write_text("fake")

        job = _make_job(tmp_path, clips, sample_region, sample_layout)

        call_count = [0]

        def mock_run_side_effect(cmd, **kwargs):
            call_count[0] += 1
            if call_count[0] == 3:
                job.status = "cancelled"
            return Mock(returncode=0, stdout="", stderr="")

        with patch("subprocess.run", side_effect=mock_run_side_effect):
            process_vertical_formatting_job(job)

        # At least 2 clips were processed before cancellation
        assert job.clips_processed >= 2
        assert job.status == "cancelled"
