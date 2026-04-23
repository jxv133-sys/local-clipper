"""Unit tests for pipeline/clip_extractor.py."""

from __future__ import annotations

import os
import subprocess
from unittest.mock import call, patch

import pytest

from config import Config
from pipeline.clip_extractor import extract_clips
from pipeline.exceptions import ClipExtractionError
from pipeline.models import Clip


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(tmp_path) -> Config:
    cfg = Config(work_dir=str(tmp_path))
    cfg.output_dir = str(tmp_path / "output")
    return cfg


def make_clip(rank: int, start: float, end: float, score: float = 0.9) -> Clip:
    return Clip(start=start, end=end, score=score, rank=rank, segment_indices=[])


def completed(returncode: int = 0, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode,
                                       stdout=stdout, stderr=stderr)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExtractClipsSuccess:
    """Happy-path: stream-copy succeeds and duration matches."""

    def test_returns_correct_output_path(self, tmp_path):
        """Returns the expected clip_<rank>_<start>s.mp4 path."""
        config = make_config(tmp_path)
        clip = make_clip(rank=1, start=10.0, end=35.0)

        # stream-copy succeeds; ffprobe returns matching duration
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                completed(0),                    # ffmpeg stream-copy
                completed(0, stdout="25.0\n"),   # ffprobe
            ]
            paths = extract_clips(config, [clip], "/fake/video.mp4")

        assert len(paths) == 1
        assert paths[0].endswith("clip_1_10s.mp4")

    def test_output_file_in_output_dir(self, tmp_path):
        """Output file is placed inside config.output_dir."""
        config = make_config(tmp_path)
        clip = make_clip(rank=1, start=5.0, end=30.0)

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                completed(0),
                completed(0, stdout="25.0\n"),
            ]
            paths = extract_clips(config, [clip], "/fake/video.mp4")

        assert paths[0].startswith(config.output_dir)

    def test_ffmpeg_called_with_stream_copy_flags(self, tmp_path):
        """FFmpeg is invoked with -c copy for stream-copy extraction."""
        config = make_config(tmp_path)
        clip = make_clip(rank=1, start=0.0, end=25.0)

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                completed(0),
                completed(0, stdout="25.0\n"),
            ]
            extract_clips(config, [clip], "/fake/video.mp4")

        first_call_cmd = mock_run.call_args_list[0][0][0]
        assert "-c" in first_call_cmd
        assert "copy" in first_call_cmd

    def test_multiple_clips_returned_in_rank_order(self, tmp_path):
        """Multiple clips are returned in ascending rank order."""
        config = make_config(tmp_path)
        clips = [
            make_clip(rank=2, start=60.0, end=85.0, score=0.7),
            make_clip(rank=1, start=10.0, end=35.0, score=0.9),
        ]

        with patch("subprocess.run") as mock_run:
            # 2 clips × (1 ffmpeg + 1 ffprobe) = 4 calls
            mock_run.side_effect = [
                completed(0), completed(0, stdout="25.0\n"),
                completed(0), completed(0, stdout="25.0\n"),
            ]
            paths = extract_clips(config, clips, "/fake/video.mp4")

        assert len(paths) == 2
        assert "clip_1_" in paths[0]
        assert "clip_2_" in paths[1]


class TestOutputDirectoryCreation:
    """Output directory is created if it does not exist."""

    def test_creates_output_dir_if_missing(self, tmp_path):
        """config.output_dir is created before writing clips."""
        config = make_config(tmp_path)
        # Ensure the output dir does NOT exist yet
        assert not os.path.exists(config.output_dir)

        clip = make_clip(rank=1, start=0.0, end=25.0)

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                completed(0),
                completed(0, stdout="25.0\n"),
            ]
            extract_clips(config, [clip], "/fake/video.mp4")

        assert os.path.exists(config.output_dir)

    def test_does_not_fail_if_output_dir_already_exists(self, tmp_path):
        """No error if output_dir already exists."""
        config = make_config(tmp_path)
        os.makedirs(config.output_dir, exist_ok=True)

        clip = make_clip(rank=1, start=0.0, end=25.0)

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                completed(0),
                completed(0, stdout="25.0\n"),
            ]
            # Should not raise
            extract_clips(config, [clip], "/fake/video.mp4")


class TestReEncodeOnDurationMismatch:
    """Re-encode is triggered when stream-copy duration differs by > 1s."""

    def test_reencode_invoked_on_duration_mismatch(self, tmp_path):
        """When probed duration differs by > 1s, a second FFmpeg call without -c copy is made."""
        config = make_config(tmp_path)
        clip = make_clip(rank=1, start=10.0, end=35.0)  # requested = 25s

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                completed(0),                    # ffmpeg stream-copy
                completed(0, stdout="22.0\n"),   # ffprobe: 22s vs 25s → diff = 3s > 1s
                completed(0),                    # ffmpeg re-encode
            ]
            extract_clips(config, [clip], "/fake/video.mp4")

        assert mock_run.call_count == 3
        # Third call should NOT have -c copy
        reencode_cmd = mock_run.call_args_list[2][0][0]
        assert "-c" not in reencode_cmd or "copy" not in reencode_cmd

    def test_no_reencode_when_duration_within_1s(self, tmp_path):
        """When probed duration is within 1s of requested, no re-encode occurs."""
        config = make_config(tmp_path)
        clip = make_clip(rank=1, start=10.0, end=35.0)  # requested = 25s

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                completed(0),                    # ffmpeg stream-copy
                completed(0, stdout="24.5\n"),   # ffprobe: 24.5s vs 25s → diff = 0.5s ≤ 1s
            ]
            extract_clips(config, [clip], "/fake/video.mp4")

        # Only 2 calls: stream-copy + ffprobe
        assert mock_run.call_count == 2

    def test_no_reencode_when_ffprobe_fails(self, tmp_path):
        """When ffprobe fails (returns None), no re-encode is triggered."""
        config = make_config(tmp_path)
        clip = make_clip(rank=1, start=10.0, end=35.0)

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                completed(0),          # ffmpeg stream-copy
                completed(1, stderr="ffprobe error"),  # ffprobe fails
            ]
            extract_clips(config, [clip], "/fake/video.mp4")

        # Only 2 calls: stream-copy + ffprobe (no re-encode since duration is None)
        assert mock_run.call_count == 2


class TestFFmpegFailure:
    """FFmpeg non-zero exit → ClipExtractionError with stderr in message."""

    def test_raises_on_nonzero_exit(self, tmp_path):
        """ClipExtractionError is raised when FFmpeg exits with non-zero code."""
        config = make_config(tmp_path)
        clip = make_clip(rank=1, start=0.0, end=25.0)
        stderr_msg = "Invalid data found when processing input"

        with patch("subprocess.run", return_value=completed(1, stderr=stderr_msg)):
            with pytest.raises(ClipExtractionError) as exc_info:
                extract_clips(config, [clip], "/fake/video.mp4")

        assert stderr_msg in str(exc_info.value)

    def test_error_message_includes_exit_code(self, tmp_path):
        """ClipExtractionError message includes the FFmpeg exit code."""
        config = make_config(tmp_path)
        clip = make_clip(rank=1, start=0.0, end=25.0)

        with patch("subprocess.run", return_value=completed(2, stderr="some error")):
            with pytest.raises(ClipExtractionError) as exc_info:
                extract_clips(config, [clip], "/fake/video.mp4")

        assert "2" in str(exc_info.value)

    def test_empty_clips_returns_empty_list(self, tmp_path):
        """Empty clips list returns empty list without calling FFmpeg."""
        config = make_config(tmp_path)

        with patch("subprocess.run") as mock_run:
            result = extract_clips(config, [], "/fake/video.mp4")

        assert result == []
        mock_run.assert_not_called()


# ---------------------------------------------------------------------------
# generate_thumbnail tests
# ---------------------------------------------------------------------------

class TestGenerateThumbnail:
    """Tests for generate_thumbnail()."""

    from pipeline.clip_extractor import generate_thumbnail

    def test_returns_thumb_path_on_success(self, tmp_path):
        """Returns the thumbnail path when ffprobe and ffmpeg both succeed."""
        from pipeline.clip_extractor import generate_thumbnail

        clip_path = str(tmp_path / "clip_1_10s.mp4")

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                completed(0, stdout="20.0\n"),  # ffprobe duration
                completed(0),                   # ffmpeg thumbnail
            ]
            result = generate_thumbnail(clip_path)

        assert result == str(tmp_path / "clip_1_10s_thumb.jpg")

    def test_thumbnail_saved_alongside_clip(self, tmp_path):
        """Thumbnail path uses <clip_stem>_thumb.jpg naming."""
        from pipeline.clip_extractor import generate_thumbnail

        clip_path = str(tmp_path / "clip_2_60s.mp4")

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                completed(0, stdout="30.0\n"),
                completed(0),
            ]
            result = generate_thumbnail(clip_path)

        assert result is not None
        assert result.endswith("clip_2_60s_thumb.jpg")
        assert os.path.dirname(result) == str(tmp_path)

    def test_midpoint_passed_to_ffmpeg(self, tmp_path):
        """ffmpeg -ss is called with the midpoint (duration / 2)."""
        from pipeline.clip_extractor import generate_thumbnail

        clip_path = str(tmp_path / "clip_1_0s.mp4")

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                completed(0, stdout="40.0\n"),  # duration = 40s → mid = 20.0
                completed(0),
            ]
            generate_thumbnail(clip_path)

        ffmpeg_cmd = mock_run.call_args_list[1][0][0]
        assert "-ss" in ffmpeg_cmd
        ss_idx = ffmpeg_cmd.index("-ss")
        assert float(ffmpeg_cmd[ss_idx + 1]) == 20.0

    def test_falls_back_to_1s_when_ffprobe_fails(self, tmp_path):
        """Falls back to 1.0s midpoint when ffprobe returns non-zero."""
        from pipeline.clip_extractor import generate_thumbnail

        clip_path = str(tmp_path / "clip_1_0s.mp4")

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                completed(1, stderr="ffprobe error"),  # ffprobe fails
                completed(0),                          # ffmpeg still runs
            ]
            result = generate_thumbnail(clip_path)

        assert result is not None
        ffmpeg_cmd = mock_run.call_args_list[1][0][0]
        ss_idx = ffmpeg_cmd.index("-ss")
        assert float(ffmpeg_cmd[ss_idx + 1]) == 1.0

    def test_returns_none_when_ffmpeg_fails(self, tmp_path):
        """Returns None (non-fatal) when ffmpeg thumbnail extraction fails."""
        from pipeline.clip_extractor import generate_thumbnail

        clip_path = str(tmp_path / "clip_1_0s.mp4")

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                completed(0, stdout="20.0\n"),
                completed(1, stderr="ffmpeg error"),
            ]
            result = generate_thumbnail(clip_path)

        assert result is None

    def test_returns_none_on_unexpected_exception(self, tmp_path):
        """Returns None (non-fatal) when an unexpected exception occurs."""
        from pipeline.clip_extractor import generate_thumbnail

        clip_path = str(tmp_path / "clip_1_0s.mp4")

        with patch("subprocess.run", side_effect=OSError("no ffmpeg")):
            result = generate_thumbnail(clip_path)

        assert result is None
