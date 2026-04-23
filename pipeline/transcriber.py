"""Transcription stage: transcribes a WAV file using the local Whisper model.

Uses faster-whisper (CTranslate2 backend) when available for 4-8x speedup
over openai-whisper on CPU. Falls back to openai-whisper if not installed.
"""

import hashlib
import json
import logging
import os
import time
import wave

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
from pipeline.models import Segment, Transcript, WordTimestamp

logger = logging.getLogger(__name__)


_VAD_GAP_THRESHOLD = 0.5  # seconds — gaps larger than this count as a silent section


def _compute_cache_key(video_path: str, whisper_model: str, file_mtime: float) -> str:
    """Return a hex digest uniquely identifying a (video_path, model, mtime) triple."""
    raw = f"{video_path}|{whisper_model}|{file_mtime}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_path(cache_dir: str, key: str) -> str:
    """Return the full path to the cache file for *key*."""
    return os.path.join(cache_dir, f"{key}.json")


def _get_wav_duration(wav_path: str) -> float:
    """Return the duration of a WAV file in seconds."""
    with wave.open(wav_path, "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate)


def _log_vad_removed(segments: list[Segment], audio_duration: float) -> None:
    """Log how much audio VAD removed, if any.

    Computes gaps between consecutive segments (and before the first /
    after the last) and logs the total removed time and silent section count
    at INFO level when there is at least one silent section.

    Args:
        segments: Transcribed segments (sorted by start time).
        audio_duration: Total duration of the source WAV file in seconds.
    """
    if not segments:
        # No segments at all — the entire file was silent; nothing useful to log
        # (the "no speech detected" message already covers this case).
        return

    silent_sections = 0
    total_removed = 0.0

    # Gap before the first segment
    gap_before = segments[0].start
    if gap_before > _VAD_GAP_THRESHOLD:
        silent_sections += 1
        total_removed += gap_before

    # Gaps between consecutive segments
    for i in range(len(segments) - 1):
        gap = segments[i + 1].start - segments[i].end
        if gap > _VAD_GAP_THRESHOLD:
            silent_sections += 1
            total_removed += gap

    # Gap after the last segment
    gap_after = audio_duration - segments[-1].end
    if gap_after > _VAD_GAP_THRESHOLD:
        silent_sections += 1
        total_removed += gap_after

    if silent_sections > 0:
        total_seconds = int(round(total_removed))
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        logger.info(
            "[Transcriber] VAD removed %d:%02d of audio across %d silent section%s",
            minutes,
            seconds,
            silent_sections,
            "s" if silent_sections != 1 else "",
        )


def transcribe(config: Config, wav_path: str, progress_callback=None) -> Transcript:
    """Transcribe *wav_path* using the local Whisper model.

    Prefers faster-whisper (CTranslate2) for speed. Falls back to
    openai-whisper if faster-whisper is not installed.

    The resulting :class:`Transcript` is serialized to
    ``<config.work_dir>/transcript.json`` before being returned.

    Args:
        config: Pipeline configuration (``work_dir`` and ``whisper_model``
            must be set).
        wav_path: Absolute or relative path to the input ``.wav`` file.
        progress_callback: Optional callback function(percentage: int) called
            during transcription to report progress (10-60%).

    Returns:
        A :class:`Transcript` containing one :class:`Segment` per Whisper
        segment.  The segment list is empty when no speech is detected.

    Raises:
        FileNotFoundError: If *wav_path* does not exist on disk.
        TranscriptionError: If the Whisper model cannot be loaded.
    """
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"WAV file not found: '{wav_path}'")

    logger.info("Transcriber starting — wav: %s, model: %s", wav_path, config.whisper_model)
    t0 = time.time()

    # ------------------------------------------------------------------
    # Cache lookup — skip Whisper if a valid cached transcript exists
    # ------------------------------------------------------------------
    if config.use_cache:
        try:
            mtime = os.path.getmtime(wav_path)
            key = _compute_cache_key(wav_path, config.whisper_model, mtime)
            cached_file = _cache_path(config.cache_dir, key)
            if os.path.exists(cached_file):
                with open(cached_file, encoding="utf-8") as fh:
                    data = json.load(fh)
                transcript = Transcript.from_dict(data)
                logger.info("[Transcriber] Loaded transcript from cache (skipping Whisper)")
                # Still write to work_dir so downstream stages can find it
                transcript_path = os.path.join(config.work_dir, "transcript.json")
                with open(transcript_path, "w", encoding="utf-8") as fh:
                    json.dump(transcript.to_dict(), fh, ensure_ascii=False, indent=2)
                return transcript
        except Exception:
            # Non-fatal — fall through to normal transcription
            pass

    if _FASTER_WHISPER_AVAILABLE:
        segments = _transcribe_faster_whisper(config, wav_path, progress_callback)
    else:
        segments = _transcribe_openai_whisper(config, wav_path, progress_callback)

    transcript = Transcript(segments=segments)

    # Log VAD-removed time ranges (only when faster-whisper VAD filter is active)
    try:
        audio_duration = _get_wav_duration(wav_path)
        _log_vad_removed(segments, audio_duration)
    except Exception:
        # Non-fatal — don't let duration reading break transcription
        pass

    # Serialize to JSON in the working directory
    transcript_path = os.path.join(config.work_dir, "transcript.json")
    with open(transcript_path, "w", encoding="utf-8") as fh:
        json.dump(transcript.to_dict(), fh, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Cache write — persist transcript so future runs can skip Whisper
    # ------------------------------------------------------------------
    if config.use_cache:
        try:
            mtime = os.path.getmtime(wav_path)
            key = _compute_cache_key(wav_path, config.whisper_model, mtime)
            cached_file = _cache_path(config.cache_dir, key)
            os.makedirs(config.cache_dir, exist_ok=True)
            with open(cached_file, "w", encoding="utf-8") as fh:
                json.dump(transcript.to_dict(), fh, ensure_ascii=False, indent=2)
        except Exception:
            # Non-fatal — caching is best-effort
            pass

    elapsed = time.time() - t0
    if segments:
        logger.info(
            "Transcriber complete — %d segment(s) in %.1fs",
            len(segments),
            elapsed,
        )
    else:
        logger.info(
            "Transcriber complete — no speech detected (%.1fs)", elapsed
        )

    return transcript


def _transcribe_faster_whisper(config: Config, wav_path: str, progress_callback=None) -> list[Segment]:
    """Transcribe using faster-whisper (CTranslate2 backend).

    Uses int8 quantization on CPU for maximum speed.
    On Apple Silicon this runs on CPU with BLAS acceleration.
    
    Args:
        config: Pipeline configuration.
        wav_path: Path to WAV file.
        progress_callback: Optional callback(percentage) for progress updates.
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

    # Get audio duration to estimate progress
    try:
        audio_duration = _get_wav_duration(wav_path)
    except Exception:
        audio_duration = None

    # Resolve language: None means auto-detect (Whisper default)
    language = None if config.language == "auto" else config.language

    # num_workers is not supported in all versions — use beam_size for speed
    raw_segments, _info = model.transcribe(
        wav_path,
        word_timestamps=True,
        vad_filter=True,          # skip silent sections — big speedup
        vad_parameters=dict(
            min_silence_duration_ms=500,
        ),
        beam_size=1,              # greedy decoding — faster, minimal accuracy loss
        language=language,
    )

    # faster-whisper returns a generator — consume it and track progress
    segments: list[Segment] = []
    last_reported_pct = 10
    
    for seg in raw_segments:
        words: list[WordTimestamp] = []
        if seg.words:
            for w in seg.words:
                words.append(WordTimestamp(
                    word=w.word,
                    start=float(w.start),
                    end=float(w.end),
                ))
        segments.append(Segment(
            start=float(seg.start),
            end=float(seg.end),
            text=seg.text,
            words=words,
        ))
        
        # Report progress based on how far through the audio we've transcribed
        if progress_callback and audio_duration and audio_duration > 0:
            # Map segment end time to progress percentage (10% → 60%)
            progress_pct = int(10 + (seg.end / audio_duration) * 50)
            progress_pct = min(60, max(10, progress_pct))
            
            # Only report when percentage increases by at least 5%
            if progress_pct >= last_reported_pct + 5:
                progress_callback(progress_pct)
                last_reported_pct = progress_pct

    # Ensure we report 60% at the end
    if progress_callback:
        progress_callback(60)

    return segments


def _transcribe_openai_whisper(config: Config, wav_path: str, progress_callback=None) -> list[Segment]:
    """Transcribe using openai-whisper (fallback).
    
    Args:
        config: Pipeline configuration.
        wav_path: Path to WAV file.
        progress_callback: Optional callback(percentage) for progress updates.
    """
    try:
        model = whisper.load_model(config.whisper_model)
    except Exception as exc:
        raise TranscriptionError(
            f"Failed to load Whisper model '{config.whisper_model}': {exc}"
        ) from exc

    # Report initial progress
    if progress_callback:
        progress_callback(15)

    # Resolve language: None means auto-detect (Whisper default)
    language = None if config.language == "auto" else config.language

    result = model.transcribe(wav_path, word_timestamps=True, language=language)

    # Report completion
    if progress_callback:
        progress_callback(60)

    raw_segments = result.get("segments", []) or []
    return [
        Segment(
            start=float(seg["start"]),
            end=float(seg["end"]),
            text=seg["text"],
        )
        for seg in raw_segments
    ]
