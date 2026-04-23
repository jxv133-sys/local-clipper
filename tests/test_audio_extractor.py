"""Unit tests for pipeline/audio_extractor.py."""

import os
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from config import Config
from pipeline.audio_extractor import extract_audio
from pipeline.exceptions import AudioExtractionError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(tmp_path) -> Config:
    """Return a minimal Config whose work_dir points to *tmp_path*."""
    return Config(work_dir=str(tmp_path))


def _completed_process(returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess:
    """Build a fake CompletedProcess for use with mock subprocess.run."""
    return subprocess.CompletedProcess(
        args=[],
        returncode=returncode,
        stdout="",
        stderr=stderr,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExtractAudioSuccess:
    """Happy-path: valid video file → returns .wav path inside work_dir."""

    def test_returns_wav_path_in_work_dir(self, tmp_path):
        """A successful FFmpeg run returns <work_dir>/audio.wav."""
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake video content")

        config = _make_config(tmp_path)

        with patch("shutil.which", return_value="/usr/bin/ffmpeg"), \
             patch("subprocess.run", return_value=_completed_process(0, "")) as mock_run:

            result = extract_audio(config, str(video))

        expected = os.path.join(str(tmp_path), "audio.wav")
        assert result == expected

    def test_ffmpeg_called_with_correct_arguments(self, tmp_path):
        """FFmpeg is invoked with -ac 1 -ar 16000 -vn flags."""
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake video content")

        config = _make_config(tmp_path)

        with patch("shutil.which", return_value="/usr/bin/ffmpeg"), \
             patch("subprocess.run", return_value=_completed_process(0, "")) as mock_run:

            extract_audio(config, str(video))

        call_args = mock_run.call_args[0][0]  # first positional arg is the command list
        assert "-ac" in call_args
        assert "1" in call_args
        assert "-ar" in call_args
        assert "16000" in call_args
        assert "-vn" in call_args
        assert str(video) in call_args


class TestExtractAudioMissingVideoFile:
    """Missing video file → FileNotFoundError."""

    def test_raises_file_not_found_for_nonexistent_path(self, tmp_path):
        config = _make_config(tmp_path)
        missing = str(tmp_path / "nonexistent.mp4")

        with pytest.raises(FileNotFoundError) as exc_info:
            extract_audio(config, missing)

        assert missing in str(exc_info.value)

    def test_error_message_contains_path(self, tmp_path):
        config = _make_config(tmp_path)
        missing = "/some/totally/missing/file.mp4"

        with pytest.raises(FileNotFoundError) as exc_info:
            extract_audio(config, missing)

        assert "/some/totally/missing/file.mp4" in str(exc_info.value)


class TestExtractAudioFFmpegNotOnPath:
    """FFmpeg not on PATH → AudioExtractionError."""

    def test_raises_audio_extraction_error_when_ffmpeg_missing(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake video content")

        config = _make_config(tmp_path)

        with patch("shutil.which", return_value=None):
            with pytest.raises(AudioExtractionError) as exc_info:
                extract_audio(config, str(video))

        assert "FFmpeg" in str(exc_info.value)
        assert "PATH" in str(exc_info.value)

    def test_subprocess_not_called_when_ffmpeg_missing(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake video content")

        config = _make_config(tmp_path)

        with patch("shutil.which", return_value=None), \
             patch("subprocess.run") as mock_run:
            with pytest.raises(AudioExtractionError):
                extract_audio(config, str(video))

        mock_run.assert_not_called()


class TestExtractAudioNoAudioTrack:
    """FFmpeg stderr contains no-audio indicator → AudioExtractionError."""

    @pytest.mark.parametrize("stderr_msg", [
        "no audio",
        "No audio",
        "NO AUDIO",
        "does not contain any stream",
        "Does not contain any stream",
    ])
    def test_raises_on_no_audio_indicator_in_stderr(self, tmp_path, stderr_msg):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake video content")

        config = _make_config(tmp_path)

        with patch("shutil.which", return_value="/usr/bin/ffmpeg"), \
             patch("subprocess.run", return_value=_completed_process(0, stderr_msg)):

            with pytest.raises(AudioExtractionError) as exc_info:
                extract_audio(config, str(video))

        assert "No audio" in str(exc_info.value) or "audio" in str(exc_info.value).lower()

    def test_error_message_includes_video_path(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake video content")

        config = _make_config(tmp_path)

        with patch("shutil.which", return_value="/usr/bin/ffmpeg"), \
             patch("subprocess.run", return_value=_completed_process(0, "no audio track found")):

            with pytest.raises(AudioExtractionError) as exc_info:
                extract_audio(config, str(video))

        assert str(video) in str(exc_info.value)


class TestExtractAudioNonZeroExit:
    """FFmpeg non-zero exit code → AudioExtractionError with stderr in message."""

    def test_raises_on_nonzero_exit_code(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake video content")

        config = _make_config(tmp_path)
        stderr_output = "Invalid data found when processing input"

        with patch("shutil.which", return_value="/usr/bin/ffmpeg"), \
             patch("subprocess.run", return_value=_completed_process(1, stderr_output)):

            with pytest.raises(AudioExtractionError) as exc_info:
                extract_audio(config, str(video))

        error_msg = str(exc_info.value)
        assert stderr_output in error_msg

    def test_error_message_includes_exit_code(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake video content")

        config = _make_config(tmp_path)

        with patch("shutil.which", return_value="/usr/bin/ffmpeg"), \
             patch("subprocess.run", return_value=_completed_process(2, "some error")):

            with pytest.raises(AudioExtractionError) as exc_info:
                extract_audio(config, str(video))

        assert "2" in str(exc_info.value)

    def test_raises_on_various_nonzero_exit_codes(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake video content")

        config = _make_config(tmp_path)

        for code in (1, 2, 127, 255):
            with patch("shutil.which", return_value="/usr/bin/ffmpeg"), \
                 patch("subprocess.run", return_value=_completed_process(code, f"error code {code}")):

                with pytest.raises(AudioExtractionError):
                    extract_audio(config, str(video))


# ---------------------------------------------------------------------------
# FFmpeg version detection tests
# ---------------------------------------------------------------------------

class TestDetectFFmpegVersion:
    """Tests for _detect_ffmpeg_version()."""

    def test_parses_major_version_from_typical_output(self):
        """Parses major version from a typical 'ffmpeg version X.Y.Z' line."""
        from pipeline.audio_extractor import _detect_ffmpeg_version

        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout="ffmpeg version 6.1.1 Copyright (c) 2000-2023 the FFmpeg developers\n",
            stderr="",
        )
        with patch("subprocess.run", return_value=mock_result):
            version = _detect_ffmpeg_version()

        assert version == 6

    def test_parses_version_4(self):
        """Parses major version 4 correctly."""
        from pipeline.audio_extractor import _detect_ffmpeg_version

        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout="ffmpeg version 4.4.2-0ubuntu0.22.04.1\n",
            stderr="",
        )
        with patch("subprocess.run", return_value=mock_result):
            version = _detect_ffmpeg_version()

        assert version == 4

    def test_returns_zero_when_ffmpeg_not_found(self):
        """Returns 0 when ffmpeg binary is not found (FileNotFoundError)."""
        from pipeline.audio_extractor import _detect_ffmpeg_version

        with patch("subprocess.run", side_effect=FileNotFoundError("ffmpeg not found")):
            version = _detect_ffmpeg_version()

        assert version == 0

    def test_returns_zero_on_malformed_output(self):
        """Returns 0 when ffmpeg output doesn't contain a parseable version."""
        from pipeline.audio_extractor import _detect_ffmpeg_version

        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout="some unexpected output with no version\n",
            stderr="",
        )
        with patch("subprocess.run", return_value=mock_result):
            version = _detect_ffmpeg_version()

        assert version == 0

    def test_returns_zero_on_empty_output(self):
        """Returns 0 when ffmpeg produces no stdout."""
        from pipeline.audio_extractor import _detect_ffmpeg_version

        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout="",
            stderr="",
        )
        with patch("subprocess.run", return_value=mock_result):
            version = _detect_ffmpeg_version()

        assert version == 0

    def test_ffmpeg_version_constant_is_int(self):
        """FFMPEG_VERSION module constant is an integer."""
        from pipeline.audio_extractor import FFMPEG_VERSION

        assert isinstance(FFMPEG_VERSION, int)
