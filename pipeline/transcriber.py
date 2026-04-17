"""Transcription stage: transcribes a WAV file using the local Whisper model.

Uses faster-whisper (CTranslate2 backend) when available for 4-8x speedup
over openai-whisper on CPU. Falls back to openai-whisper if not installed.
"""

import json
import os

# Try faster-whisper first (significantly faster on CPU)
try:
    from faster_whisper import WhisperModel as FasterWhisperModel
    _FASTER_WHISPER_AVAILABLE = True
except ImportError:
    _FASTER_WHISPER_AVAILABLE = False

# Fall back to openai-whisper
try:
    import whisper  # noqa: F401
except ImportError:
    whisper = None  # type: ignore[assignment]

from config import Config
from pipeline.exceptions import TranscriptionError
from pipeline.models import Segment, Transcript


def transcribe(config: Config, wav_path: str) -> Transcript:
    """Transcribe *wav_path* using the local Whisper model.

    Prefers faster-whisper (CTranslate2) for speed. Falls back to
    openai-whisper if faster-whisper is not installed.

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
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"WAV file not found: '{wav_path}'")

    if _FASTER_WHISPER_AVAILABLE:
        segments = _transcribe_faster_whisper(config, wav_path)
    else:
        segments = _transcribe_openai_whisper(config, wav_path)

    transcript = Transcript(segments=segments)

    # Serialize to JSON in the working directory
    transcript_path = os.path.join(config.work_dir, "transcript.json")
    with open(transcript_path, "w", encoding="utf-8") as fh:
        json.dump(transcript.to_dict(), fh, ensure_ascii=False, indent=2)

    return transcript


def _transcribe_faster_whisper(config: Config, wav_path: str) -> list[Segment]:
    """Transcribe using faster-whisper (CTranslate2 backend).

    Uses int8 quantization on CPU for maximum speed.
    On Apple Silicon this runs on CPU with BLAS acceleration.
    """
    try:
        # int8 quantization gives ~2x additional speedup on CPU with minimal
        # accuracy loss for highlight detection purposes
        model = FasterWhisperModel(
            config.whisper_model,
            device="cpu",
            compute_type="int8",
        )
    except Exception as exc:
        raise TranscriptionError(
            f"Failed to load faster-whisper model '{config.whisper_model}': {exc}"
        ) from exc

    # num_workers uses all available CPU cores for parallel decoding
    import multiprocessing
    num_workers = max(1, multiprocessing.cpu_count() - 1)

    raw_segments, _info = model.transcribe(
        wav_path,
        word_timestamps=True,
        num_workers=num_workers,
        vad_filter=True,          # skip silent sections — big speedup
        vad_parameters=dict(
            min_silence_duration_ms=500,
        ),
    )

    # faster-whisper returns a generator — consume it
    segments: list[Segment] = []
    for seg in raw_segments:
        segments.append(Segment(
            start=float(seg.start),
            end=float(seg.end),
            text=seg.text,
        ))

    return segments


def _transcribe_openai_whisper(config: Config, wav_path: str) -> list[Segment]:
    """Transcribe using openai-whisper (fallback)."""
    try:
        model = whisper.load_model(config.whisper_model)
    except Exception as exc:
        raise TranscriptionError(
            f"Failed to load Whisper model '{config.whisper_model}': {exc}"
        ) from exc

    result = model.transcribe(wav_path, word_timestamps=True)

    raw_segments = result.get("segments", []) or []
    return [
        Segment(
            start=float(seg["start"]),
            end=float(seg["end"]),
            text=seg["text"],
        )
        for seg in raw_segments
    ]
