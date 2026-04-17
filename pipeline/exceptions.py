"""Custom exceptions for the video highlight generator pipeline."""


class PipelineError(Exception):
    """Base exception for all pipeline errors."""


class AudioExtractionError(PipelineError):
    """Raised when audio extraction fails.

    Conditions:
    - No audio track found in the input video
    - FFmpeg is not installed or not accessible on PATH
    - FFmpeg exits with a non-zero return code
    """


class TranscriptionError(PipelineError):
    """Raised when transcription fails.

    Conditions:
    - Whisper model load failure
    """


class ClipExtractionError(PipelineError):
    """Raised when clip extraction fails.

    Conditions:
    - FFmpeg exits with a non-zero return code during clip extraction
    """


class SubtitleError(PipelineError):
    """Raised when subtitle generation fails.

    Conditions:
    - FFmpeg exits with a non-zero return code during subtitle burning
    """


class LLMScoringError(PipelineError):
    """Raised when LLM scoring fails.

    Conditions:
    - LLM endpoint is unreachable
    Note: This is non-fatal; callers should catch it and fall back to score 0.0
    """
