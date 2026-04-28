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
    cfg.trim_silence = False  # disable by default so existing tests aren't affected
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
        """FFmpeg is invoked with libx264 re-encode for reliable extraction."""
        config = make_config(tmp_path)
        clip = make_clip(rank=1, start=0.0, end=25.0)

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [completed(0)]
            extract_clips(config, [clip], "/fake/video.mp4")

        first_call_cmd = mock_run.call_args_list[0][0][0]
        assert "libx264" in first_call_cmd

    def test_multiple_clips_returned_in_rank_order(self, tmp_path):
        """Multiple clips are returned in ascending rank order."""
        config = make_config(tmp_path)
        clips = [
            make_clip(rank=2, start=60.0, end=85.0, score=0.7),
            make_clip(rank=1, start=10.0, end=35.0, score=0.9),
        ]

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [completed(0), completed(0)]
            paths = extract_clips(config, clips, "/fake/video.mp4")

        assert len(paths) == 2
        assert "clip_1_" in paths[0]
        assert "clip_2_" in paths[1]


class TestOutputDirectoryCreation:
    """Output directory is created if it does not exist."""

    def test_creates_output_dir_if_missing(self, tmp_path):
        """config.output_dir is created before writing clips."""
        config = make_config(tmp_path)
        assert not os.path.exists(config.output_dir)
        clip = make_clip(rank=1, start=0.0, end=25.0)
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [completed(0)]
            extract_clips(config, [clip], "/fake/video.mp4")
        assert os.path.exists(config.output_dir)

    def test_does_not_fail_if_output_dir_already_exists(self, tmp_path):
        """No error if output_dir already exists."""
        config = make_config(tmp_path)
        os.makedirs(config.output_dir, exist_ok=True)
        clip = make_clip(rank=1, start=0.0, end=25.0)
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [completed(0)]
            extract_clips(config, [clip], "/fake/video.mp4")


class TestReEncodeOnDurationMismatch:
    """Single-pass re-encode always produces correct duration."""

    def test_reencode_invoked_on_duration_mismatch(self, tmp_path):
        """Single FFmpeg call always re-encodes — no stream-copy fallback."""
        config = make_config(tmp_path)
        clip = make_clip(rank=1, start=10.0, end=35.0)

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [completed(0)]
            extract_clips(config, [clip], "/fake/video.mp4")

        assert mock_run.call_count == 1
        cmd = mock_run.call_args_list[0][0][0]
        assert "libx264" in cmd

    def test_no_reencode_when_duration_within_1s(self, tmp_path):
        """Single-pass extraction — always exactly one FFmpeg call."""
        config = make_config(tmp_path)
        clip = make_clip(rank=1, start=10.0, end=35.0)

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [completed(0)]
            extract_clips(config, [clip], "/fake/video.mp4")

        assert mock_run.call_count == 1

    def test_no_reencode_when_ffprobe_fails(self, tmp_path):
        """Single-pass extraction — ffprobe not called during extraction."""
        config = make_config(tmp_path)
        clip = make_clip(rank=1, start=10.0, end=35.0)

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [completed(0)]
            extract_clips(config, [clip], "/fake/video.mp4")

        assert mock_run.call_count == 1


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


# ---------------------------------------------------------------------------
# trim_clip_silence tests
# ---------------------------------------------------------------------------

class TestURLValidation:
    """Test that URLs are rejected as output directories."""

    def test_rejects_http_url_as_output_dir(self, tmp_path):
        """HTTP URLs are rejected as output directories."""
        config = make_config(tmp_path)
        config.output_dir = "http://example.com"
        clip = make_clip(rank=1, start=0.0, end=25.0)

        with pytest.raises(ClipExtractionError, match="Invalid output directory.*URL"):
            extract_clips(config, [clip], "/fake/video.mp4")

    def test_rejects_https_url_as_output_dir(self, tmp_path):
        """HTTPS URLs are rejected as output directories."""
        config = make_config(tmp_path)
        config.output_dir = "https://youtu.be/E98O-HlcjtY"
        clip = make_clip(rank=1, start=0.0, end=25.0)

        with pytest.raises(ClipExtractionError, match="Invalid output directory.*URL"):
            extract_clips(config, [clip], "/fake/video.mp4")

    def test_rejects_ftp_url_as_output_dir(self, tmp_path):
        """FTP URLs are rejected as output directories."""
        config = make_config(tmp_path)
        config.output_dir = "ftp://example.com/path"
        clip = make_clip(rank=1, start=0.0, end=25.0)

        with pytest.raises(ClipExtractionError, match="Invalid output directory.*URL"):
            extract_clips(config, [clip], "/fake/video.mp4")


class TestTrimClipSilence:
    """Tests for trim_clip_silence()."""

    def test_replaces_original_when_trimmed_duration_above_min(self, tmp_path):
        """Replaces the original clip when trimmed duration >= min_clip_duration."""
        from pipeline.clip_extractor import trim_clip_silence

        config = make_config(tmp_path)
        config.min_clip_duration = 20.0
        clip_path = str(tmp_path / "clip_1_10s.mp4")
        open(clip_path, "w").close()  # create dummy file

        with patch("subprocess.run") as mock_run, \
             patch("os.replace") as mock_replace, \
             patch("pipeline.clip_extractor._probe_duration") as mock_probe:
            mock_run.return_value = completed(0)
            # original duration, then trimmed duration
            mock_probe.side_effect = [25.0, 22.0]
            result = trim_clip_silence(clip_path, config, clip_rank=1)

        mock_replace.assert_called_once()
        assert result == clip_path

    def test_keeps_original_when_trimmed_below_min_clip_duration(self, tmp_path):
        """Keeps original clip when trimmed duration < min_clip_duration."""
        from pipeline.clip_extractor import trim_clip_silence

        config = make_config(tmp_path)
        config.min_clip_duration = 30.0
        clip_path = str(tmp_path / "clip_1_10s.mp4")
        open(clip_path, "w").close()

        trimmed_path = str(tmp_path / "clip_1_10s_trimmed.mp4")
        open(trimmed_path, "w").close()

        with patch("subprocess.run") as mock_run, \
             patch("pipeline.clip_extractor._probe_duration") as mock_probe:
            mock_run.return_value = completed(0)
            # trimmed duration is below min_clip_duration
            mock_probe.side_effect = [25.0, 15.0]
            result = trim_clip_silence(clip_path, config, clip_rank=1)

        # Original path returned unchanged
        assert result == clip_path
        # Trimmed file should be removed
        assert not os.path.exists(trimmed_path)

    def test_returns_original_when_ffmpeg_fails(self, tmp_path):
        """Returns original path when ffmpeg silenceremove fails."""
        from pipeline.clip_extractor import trim_clip_silence

        config = make_config(tmp_path)
        config.min_clip_duration = 20.0
        clip_path = str(tmp_path / "clip_1_10s.mp4")
        open(clip_path, "w").close()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = completed(1, stderr="ffmpeg error")
            result = trim_clip_silence(clip_path, config, clip_rank=1)

        assert result == clip_path

    def test_returns_original_when_probe_fails(self, tmp_path):
        """Returns original path when ffprobe cannot determine trimmed duration."""
        from pipeline.clip_extractor import trim_clip_silence

        config = make_config(tmp_path)
        config.min_clip_duration = 20.0
        clip_path = str(tmp_path / "clip_1_10s.mp4")
        open(clip_path, "w").close()

        trimmed_path = str(tmp_path / "clip_1_10s_trimmed.mp4")
        open(trimmed_path, "w").close()

        with patch("subprocess.run") as mock_run, \
             patch("pipeline.clip_extractor._probe_duration", return_value=None):
            mock_run.return_value = completed(0)
            result = trim_clip_silence(clip_path, config, clip_rank=1)

        assert result == clip_path

    def test_returns_original_on_unexpected_exception(self, tmp_path):
        """Returns original path gracefully when an unexpected exception occurs."""
        from pipeline.clip_extractor import trim_clip_silence

        config = make_config(tmp_path)
        config.min_clip_duration = 20.0
        clip_path = str(tmp_path / "clip_1_10s.mp4")
        open(clip_path, "w").close()

        with patch("subprocess.run", side_effect=OSError("no ffmpeg")):
            result = trim_clip_silence(clip_path, config, clip_rank=1)

        assert result == clip_path

    def test_trim_silence_disabled_skips_trim(self, tmp_path):
        """When config.trim_silence=False, trim_clip_silence is not called during extraction."""
        config = make_config(tmp_path)
        config.trim_silence = False
        clip = make_clip(rank=1, start=10.0, end=40.0)

        with patch("subprocess.run") as mock_run, \
             patch("pipeline.clip_extractor.trim_clip_silence") as mock_trim:
            mock_run.side_effect = [
                completed(0),                    # ffmpeg stream-copy
                completed(0, stdout="30.0\n"),   # ffprobe
            ]
            extract_clips(config, [clip], "/fake/video.mp4")

        mock_trim.assert_not_called()

    def test_trim_silence_enabled_calls_trim(self, tmp_path):
        """When config.trim_silence=True, trim_clip_silence is called after extraction."""
        config = make_config(tmp_path)
        config.trim_silence = True
        clip = make_clip(rank=1, start=10.0, end=40.0)

        with patch("subprocess.run") as mock_run, \
             patch("pipeline.clip_extractor.trim_clip_silence") as mock_trim:
            mock_run.side_effect = [
                completed(0),                    # ffmpeg stream-copy
                completed(0, stdout="30.0\n"),   # ffprobe
            ]
            mock_trim.return_value = str(tmp_path / "output" / "clip_1_10s.mp4")
            extract_clips(config, [clip], "/fake/video.mp4")

        mock_trim.assert_called_once()
