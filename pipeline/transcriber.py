"""Transcription stage: transcribes a WAV file using the local Whisper model."""

import json
import os

try:
    import whisper  # noqa: F401 – imported at module level so tests can patch it
except ImportError:  # pragma: no cover – whisper may not be installed in test envs
    whisper = None  # type: ignore[assignment]

from config import Config
from pipeline.exceptions import TranscriptionError
from pipeline.models import Segment, Transcript


def transcribe(config: Config, wav_path: str) -> Transcript:
    """Transcribe *wav_path* using the local Whisper model.

    The resulting :class:`Transcript` is serialized to
    ``<config.work_dir>/transcript.json`` before being returned.

    Args:
        config: Pipeline configuration (``work_dir`` and ``whisper_model``
            must be set).
        wav_path: Absolute or relative path to the input ``.wav`` file.

    Returns:
        A :class:`Transcript` containing one :class:`Segment` per Whisper
        segment.  The segment list is empty when no speech is detected.

    Raises:
        FileNotFoundError: If *wav_path* does not exist on disk.
        TranscriptionError: If the Whisper model cannot be loaded.
    """
    # 1. Verify the input file exists.
    if not os.path.exists(wav_path):
        raise FileNotFoundError(
            f"WAV file not found: '{wav_path}'"
        )

    # 2. Load the Whisper model (wrap failures in TranscriptionError).
    try:
        model = whisper.load_model(config.whisper_model)
    except Exception as exc:
        raise TranscriptionError(
            f"Failed to load Whisper model '{config.whisper_model}': {exc}"
        ) from exc

    # 3. Run transcription with word-level timestamps.
    result = model.transcribe(wav_path, word_timestamps=True)

    # 4. Map Whisper segments to Segment dataclass instances.
    raw_segments = result.get("segments", []) or []
    segments = [
        Segment(
            start=float(seg["start"]),
            end=float(seg["end"]),
            text=seg["text"],
        )
        for seg in raw_segments
    ]

    transcript = Transcript(segments=segments)

    # 5. Serialize the Transcript to JSON in the working directory.
    transcript_path = os.path.join(config.work_dir, "transcript.json")
    with open(transcript_path, "w", encoding="utf-8") as fh:
        json.dump(transcript.to_dict(), fh, ensure_ascii=False, indent=2)

    return transcript
