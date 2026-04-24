"""Audio extraction stage: extracts a mono 16kHz WAV from a video file using FFmpeg."""

import logging
import os
import re
import shutil
import subprocess
import time

from config import Config
from pipeline.exceptions import AudioExtractionError

logger = logging.getLogger(__name__)


def _detect_ffmpeg_version() -> int:
    """Detect the major version of the installed FFmpeg binary.

    Runs ``ffmpeg -version`` and parses the first line for the version number.

    Returns:
        The major version as an int (e.g. ``6`` for "ffmpeg version 6.1.1").
        Returns ``0`` if ffmpeg is not found or the version cannot be parsed.
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        first_line = result.stdout.splitlines()[0] if result.stdout.strip() else ""
        match = re.search(r"ffmpeg version (\d+)", first_line, re.IGNORECASE)
        if match:
            major = int(match.group(1))
            logger.debug("[FFmpeg] Detected version %d.x", major)
            return major
    except (FileNotFoundError, OSError):
        pass
    except Exception:
        pass
    return 0


# Detected at import time so all modules share a single probe.
FFMPEG_VERSION: int = _detect_ffmpeg_version()

# Phrases in FFmpeg stderr that indicate no audio track was found
_NO_AUDIO_INDICATORS = [
    "no audio",
    "does not contain any stream",
    "audio stream not found",
]


def extract_audio(config: Config, video_path: str) -> str:
    """Extract audio from *video_path* to a mono 16 kHz WAV file.

    The output file is written to ``<config.work_dir>/audio.wav``.

    Args:
        config: Pipeline configuration (``work_dir`` must be set).
        video_path: Absolute or relative path to the input video file.

    Returns:
        The path to the extracted ``.wav`` file.

    Raises:
        FileNotFoundError: If *video_path* does not exist on disk.
        AudioExtractionError: If FFmpeg is not found on PATH, if the video
            contains no audio track, or if FFmpeg exits with a non-zero code.
    """
    logger.info("AudioExtractor starting — input: %s", video_path)
    t0 = time.time()

    # 1. Verify the input file exists.
    if not os.path.exists(video_path):
        raise FileNotFoundError(
            f"Video file not found: '{video_path}'"
        )

    # 2. Verify FFmpeg is available on PATH.
    if shutil.which("ffmpeg") is None:
        raise AudioExtractionError(
            "FFmpeg is not installed or not accessible on PATH. "
            "Please install FFmpeg and ensure it is in your PATH."
        )

    output_wav = os.path.join(config.work_dir, "audio.wav")

    # 3. Invoke FFmpeg to extract audio.
    cmd = [
        "ffmpeg",
        "-y",           # overwrite output without prompting
        "-threads", "0", # use all available CPU cores
        "-i", video_path,
        "-ac", "1",     # mono
        "-ar", "16000", # 16 kHz sample rate
        "-vn",          # no video
        output_wav,
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    stderr_lower = result.stderr.lower()

    # 4. Detect "no audio" conditions in FFmpeg stderr.
    for indicator in _NO_AUDIO_INDICATORS:
        if indicator in stderr_lower:
            raise AudioExtractionError(
                f"No audio track found in '{video_path}'. "
                f"FFmpeg output: {result.stderr.strip()}"
            )

    # 5. Raise on non-zero exit code.
    if result.returncode != 0:
        raise AudioExtractionError(
            f"FFmpeg exited with code {result.returncode} while processing "
            f"'{video_path}'. stderr: {result.stderr.strip()}"
        )

    # 6. Return the path to the extracted WAV file.
    elapsed = time.time() - t0
    logger.info("AudioExtractor complete — output: %s (%.1fs)", output_wav, elapsed)
    return output_wav
