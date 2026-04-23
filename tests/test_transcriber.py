"""Unit tests for pipeline/transcriber.py."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from config import Config
from pipeline.exceptions import TranscriptionError
from pipeline.models import Segment, Transcript
from pipeline.transcriber import transcribe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(tmp_path) -> Config:
    """Return a minimal Config whose work_dir points to *tmp_path*."""
    return Config(work_dir=str(tmp_path))


def _fake_whisper_result(segments: list[dict]) -> dict:
    """Build a fake Whisper transcription result dict."""
    return {"segments": segments, "text": " ".join(s["text"] for s in segments)}


def _make_fake_model(result: dict) -> MagicMock:
    """Return a mock Whisper model whose .transcribe() returns *result*."""
    model = MagicMock()
    model.transcribe.return_value = result
    return model


# Force all tests to use the openai-whisper fallback path so we can mock it.
# faster-whisper is tested separately; these tests cover the core transcribe() logic.
@pytest.fixture(autouse=True)
def use_openai_whisper_backend():
    with patch("pipeline.transcriber._FASTER_WHISPER_AVAILABLE", False):
        yield


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTranscribeSuccess:
    """Happy-path: valid WAV → Transcript with correct segments + JSON written."""

    def test_returns_transcript_with_correct_segments(self, tmp_path):
        """transcribe() maps Whisper segments to Segment dataclass instances."""
        wav = tmp_path / "audio.wav"
        wav.write_bytes(b"fake wav content")

        config = _make_config(tmp_path)

        whisper_segments = [
            {"start": 0.0, "end": 2.5, "text": "Hello world"},
            {"start": 2.5, "end": 5.0, "text": "This is a test"},
        ]
        fake_model = _make_fake_model(_fake_whisper_result(whisper_segments))

        with patch("pipeline.transcriber.whisper") as mock_whisper:
            mock_whisper.load_model.return_value = fake_model
            result = transcribe(config, str(wav))

        assert isinstance(result, Transcript)
        assert len(result.segments) == 2

        seg0 = result.segments[0]
        assert seg0.start == 0.0
        assert seg0.end == 2.5
        assert seg0.text == "Hello world"

        seg1 = result.segments[1]
        assert seg1.start == 2.5
        assert seg1.end == 5.0
        assert seg1.text == "This is a test"

    def test_whisper_called_with_word_timestamps(self, tmp_path):
        """transcribe() calls model.transcribe with word_timestamps=True."""
        wav = tmp_path / "audio.wav"
        wav.write_bytes(b"fake wav content")

        config = _make_config(tmp_path)
        fake_model = _make_fake_model(_fake_whisper_result([]))

        with patch("pipeline.transcriber.whisper") as mock_whisper:
            mock_whisper.load_model.return_value = fake_model
            transcribe(config, str(wav))

        fake_model.transcribe.assert_called_once_with(str(wav), word_timestamps=True, language=None)

    def test_writes_json_to_work_dir(self, tmp_path):
        """transcribe() serializes the Transcript to <work_dir>/transcript.json."""
        wav = tmp_path / "audio.wav"
        wav.write_bytes(b"fake wav content")

        config = _make_config(tmp_path)

        whisper_segments = [
            {"start": 1.0, "end": 3.0, "text": "Some speech"},
        ]
        fake_model = _make_fake_model(_fake_whisper_result(whisper_segments))

        with patch("pipeline.transcriber.whisper") as mock_whisper:
            mock_whisper.load_model.return_value = fake_model
            transcribe(config, str(wav))

        json_path = tmp_path / "transcript.json"
        assert json_path.exists(), "transcript.json was not written to work_dir"

    def test_json_content_matches_transcript(self, tmp_path):
        """The written JSON file contains the correct segment data."""
        wav = tmp_path / "audio.wav"
        wav.write_bytes(b"fake wav content")

        config = _make_config(tmp_path)

        whisper_segments = [
            {"start": 0.5, "end": 4.0, "text": "Check this out"},
        ]
        fake_model = _make_fake_model(_fake_whisper_result(whisper_segments))

        with patch("pipeline.transcriber.whisper") as mock_whisper:
            mock_whisper.load_model.return_value = fake_model
            transcribe(config, str(wav))

        json_path = tmp_path / "transcript.json"
        with open(json_path, encoding="utf-8") as fh:
            data = json.load(fh)

        assert "segments" in data
        assert len(data["segments"]) == 1
        assert data["segments"][0]["start"] == 0.5
        assert data["segments"][0]["end"] == 4.0
        assert data["segments"][0]["text"] == "Check this out"

    def test_whisper_model_loaded_with_config_model_name(self, tmp_path):
        """whisper.load_model is called with config.whisper_model."""
        wav = tmp_path / "audio.wav"
        wav.write_bytes(b"fake wav content")

        config = _make_config(tmp_path)
        config.whisper_model = "small"

        fake_model = _make_fake_model(_fake_whisper_result([]))

        with patch("pipeline.transcriber.whisper") as mock_whisper:
            mock_whisper.load_model.return_value = fake_model
            transcribe(config, str(wav))

        mock_whisper.load_model.assert_called_once_with("small")


class TestTranscribeMissingWavFile:
    """Missing WAV file → FileNotFoundError."""

    def test_raises_file_not_found_for_nonexistent_path(self, tmp_path):
        config = _make_config(tmp_path)
        missing = str(tmp_path / "nonexistent.wav")

        with pytest.raises(FileNotFoundError) as exc_info:
            transcribe(config, missing)

        assert missing in str(exc_info.value)

    def test_error_message_contains_path(self, tmp_path):
        config = _make_config(tmp_path)
        missing = "/some/totally/missing/audio.wav"

        with pytest.raises(FileNotFoundError) as exc_info:
            transcribe(config, missing)

        assert "/some/totally/missing/audio.wav" in str(exc_info.value)

    def test_whisper_not_called_when_file_missing(self, tmp_path):
        config = _make_config(tmp_path)
        missing = str(tmp_path / "nonexistent.wav")

        with patch("pipeline.transcriber.whisper") as mock_whisper:
            with pytest.raises(FileNotFoundError):
                transcribe(config, missing)

        mock_whisper.load_model.assert_not_called()


class TestTranscribeNoSpeech:
    """Whisper returns no segments → Transcript with empty segment list."""

    def test_returns_empty_transcript_when_no_segments(self, tmp_path):
        wav = tmp_path / "audio.wav"
        wav.write_bytes(b"fake wav content")

        config = _make_config(tmp_path)
        fake_model = _make_fake_model({"segments": [], "text": ""})

        with patch("pipeline.transcriber.whisper") as mock_whisper:
            mock_whisper.load_model.return_value = fake_model
            result = transcribe(config, str(wav))

        assert isinstance(result, Transcript)
        assert result.segments == []

    def test_returns_empty_transcript_when_segments_is_none(self, tmp_path):
        """Handles Whisper returning None for segments key."""
        wav = tmp_path / "audio.wav"
        wav.write_bytes(b"fake wav content")

        config = _make_config(tmp_path)
        fake_model = _make_fake_model({"segments": None, "text": ""})

        with patch("pipeline.transcriber.whisper") as mock_whisper:
            mock_whisper.load_model.return_value = fake_model
            result = transcribe(config, str(wav))

        assert isinstance(result, Transcript)
        assert result.segments == []

    def test_writes_empty_json_when_no_segments(self, tmp_path):
        """Even with no segments, transcript.json is written."""
        wav = tmp_path / "audio.wav"
        wav.write_bytes(b"fake wav content")

        config = _make_config(tmp_path)
        fake_model = _make_fake_model({"segments": [], "text": ""})

        with patch("pipeline.transcriber.whisper") as mock_whisper:
            mock_whisper.load_model.return_value = fake_model
            transcribe(config, str(wav))

        json_path = tmp_path / "transcript.json"
        assert json_path.exists()

        with open(json_path, encoding="utf-8") as fh:
            data = json.load(fh)

        assert data["segments"] == []


class TestTranscribeModelLoadFailure:
    """Whisper model load failure → TranscriptionError."""

    def test_raises_transcription_error_on_model_load_failure(self, tmp_path):
        wav = tmp_path / "audio.wav"
        wav.write_bytes(b"fake wav content")

        config = _make_config(tmp_path)

        with patch("pipeline.transcriber.whisper") as mock_whisper:
            mock_whisper.load_model.side_effect = RuntimeError("model not found")

            with pytest.raises(TranscriptionError) as exc_info:
                transcribe(config, str(wav))

        assert "model not found" in str(exc_info.value) or "Failed to load" in str(exc_info.value)

    def test_transcription_error_includes_model_name(self, tmp_path):
        wav = tmp_path / "audio.wav"
        wav.write_bytes(b"fake wav content")

        config = _make_config(tmp_path)
        config.whisper_model = "large"

        with patch("pipeline.transcriber.whisper") as mock_whisper:
            mock_whisper.load_model.side_effect = Exception("CUDA out of memory")

            with pytest.raises(TranscriptionError) as exc_info:
                transcribe(config, str(wav))

        assert "large" in str(exc_info.value)


class TestTranscribeJsonRoundTrip:
    """JSON file round-trips correctly: deserialize and compare."""

    def test_json_roundtrip_single_segment(self, tmp_path):
        """Deserializing the written JSON produces an equivalent Transcript."""
        wav = tmp_path / "audio.wav"
        wav.write_bytes(b"fake wav content")

        config = _make_config(tmp_path)

        whisper_segments = [
            {"start": 10.0, "end": 15.5, "text": "Round-trip test"},
        ]
        fake_model = _make_fake_model(_fake_whisper_result(whisper_segments))

        with patch("pipeline.transcriber.whisper") as mock_whisper:
            mock_whisper.load_model.return_value = fake_model
            original = transcribe(config, str(wav))

        json_path = tmp_path / "transcript.json"
        with open(json_path, encoding="utf-8") as fh:
            data = json.load(fh)

        restored = Transcript.from_dict(data)

        assert len(restored.segments) == len(original.segments)
        for orig_seg, rest_seg in zip(original.segments, restored.segments):
            assert orig_seg.start == rest_seg.start
            assert orig_seg.end == rest_seg.end
            assert orig_seg.text == rest_seg.text

    def test_json_roundtrip_multiple_segments(self, tmp_path):
        """Round-trip works correctly with multiple segments."""
        wav = tmp_path / "audio.wav"
        wav.write_bytes(b"fake wav content")

        config = _make_config(tmp_path)

        whisper_segments = [
            {"start": 0.0, "end": 3.0, "text": "First segment"},
            {"start": 3.0, "end": 7.5, "text": "Second segment"},
            {"start": 7.5, "end": 12.0, "text": "Third segment"},
        ]
        fake_model = _make_fake_model(_fake_whisper_result(whisper_segments))

        with patch("pipeline.transcriber.whisper") as mock_whisper:
            mock_whisper.load_model.return_value = fake_model
            original = transcribe(config, str(wav))

        json_path = tmp_path / "transcript.json"
        with open(json_path, encoding="utf-8") as fh:
            data = json.load(fh)

        restored = Transcript.from_dict(data)

        assert restored == original

    def test_json_roundtrip_empty_transcript(self, tmp_path):
        """Round-trip works correctly with an empty segment list."""
        wav = tmp_path / "audio.wav"
        wav.write_bytes(b"fake wav content")

        config = _make_config(tmp_path)
        fake_model = _make_fake_model({"segments": [], "text": ""})

        with patch("pipeline.transcriber.whisper") as mock_whisper:
            mock_whisper.load_model.return_value = fake_model
            original = transcribe(config, str(wav))

        json_path = tmp_path / "transcript.json"
        with open(json_path, encoding="utf-8") as fh:
            data = json.load(fh)

        restored = Transcript.from_dict(data)

        assert restored == original
        assert restored.segments == []


# ---------------------------------------------------------------------------
# Helpers for WAV-duration-aware tests
# ---------------------------------------------------------------------------

import struct
import wave as _wave_module


def _write_real_wav(path, duration_seconds: float, sample_rate: int = 16000) -> None:
    """Write a minimal valid WAV file of the given duration (silence)."""
    num_frames = int(duration_seconds * sample_rate)
    with _wave_module.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        # Write silence (all zeros)
        wf.writeframes(b"\x00\x00" * num_frames)


# ---------------------------------------------------------------------------
# VAD logging tests
# ---------------------------------------------------------------------------

class TestVadRemovedLogging:
    """Verify that VAD-removed time is logged when gaps exist between segments."""

    def test_vad_log_emitted_when_gaps_between_segments(self, tmp_path, caplog):
        """transcribe() logs VAD-removed message when segments have gaps > 0.5s."""
        import logging

        wav = tmp_path / "audio.wav"
        # 60-second audio file
        _write_real_wav(wav, duration_seconds=60.0)

        config = _make_config(tmp_path)

        # Segments cover 0-5s and 35-40s, leaving a ~30s gap in the middle
        # and a ~20s tail — both > 0.5s threshold
        whisper_segments = [
            {"start": 0.0, "end": 5.0, "text": "Hello"},
            {"start": 35.0, "end": 40.0, "text": "World"},
        ]
        fake_model = _make_fake_model(_fake_whisper_result(whisper_segments))

        with patch("pipeline.transcriber.whisper") as mock_whisper:
            mock_whisper.load_model.return_value = fake_model
            with caplog.at_level(logging.INFO, logger="pipeline.transcriber"):
                transcribe(config, str(wav))

        vad_messages = [r.message for r in caplog.records if "VAD removed" in r.message]
        assert len(vad_messages) == 1, f"Expected 1 VAD log message, got: {vad_messages}"

        msg = vad_messages[0]
        # Should mention "silent sections" (plural, since there are 2 gaps)
        assert "silent section" in msg
        assert "[Transcriber] VAD removed" in msg

    def test_vad_log_not_emitted_when_no_gaps(self, tmp_path, caplog):
        """transcribe() does NOT log VAD message when segments cover the full audio."""
        import logging

        wav = tmp_path / "audio.wav"
        _write_real_wav(wav, duration_seconds=10.0)

        config = _make_config(tmp_path)

        # Segments cover the full 10 seconds with no gaps
        whisper_segments = [
            {"start": 0.0, "end": 5.0, "text": "First half"},
            {"start": 5.0, "end": 10.0, "text": "Second half"},
        ]
        fake_model = _make_fake_model(_fake_whisper_result(whisper_segments))

        with patch("pipeline.transcriber.whisper") as mock_whisper:
            mock_whisper.load_model.return_value = fake_model
            with caplog.at_level(logging.INFO, logger="pipeline.transcriber"):
                transcribe(config, str(wav))

        vad_messages = [r.message for r in caplog.records if "VAD removed" in r.message]
        assert len(vad_messages) == 0, f"Expected no VAD log message, got: {vad_messages}"

    def test_vad_log_format_minutes_and_seconds(self, tmp_path, caplog):
        """VAD log message uses MM:SS format (e.g. '32:24')."""
        import logging

        # 2000-second audio; segments cover only 10s, leaving ~1990s removed
        wav = tmp_path / "audio.wav"
        _write_real_wav(wav, duration_seconds=2000.0)

        config = _make_config(tmp_path)

        whisper_segments = [
            {"start": 0.0, "end": 5.0, "text": "Start"},
            {"start": 5.0, "end": 10.0, "text": "End"},
        ]
        fake_model = _make_fake_model(_fake_whisper_result(whisper_segments))

        with patch("pipeline.transcriber.whisper") as mock_whisper:
            mock_whisper.load_model.return_value = fake_model
            with caplog.at_level(logging.INFO, logger="pipeline.transcriber"):
                transcribe(config, str(wav))

        vad_messages = [r.message for r in caplog.records if "VAD removed" in r.message]
        assert len(vad_messages) == 1
        msg = vad_messages[0]
        # 1990 seconds = 33 minutes 10 seconds → "33:10"
        assert "33:10" in msg

    def test_vad_log_singular_section(self, tmp_path, caplog):
        """VAD log uses singular 'section' when only one gap exists."""
        import logging

        wav = tmp_path / "audio.wav"
        _write_real_wav(wav, duration_seconds=20.0)

        config = _make_config(tmp_path)

        # One gap: segment ends at 5s, audio ends at 20s → 15s tail gap
        whisper_segments = [
            {"start": 0.0, "end": 5.0, "text": "Only speech"},
        ]
        fake_model = _make_fake_model(_fake_whisper_result(whisper_segments))

        with patch("pipeline.transcriber.whisper") as mock_whisper:
            mock_whisper.load_model.return_value = fake_model
            with caplog.at_level(logging.INFO, logger="pipeline.transcriber"):
                transcribe(config, str(wav))

        vad_messages = [r.message for r in caplog.records if "VAD removed" in r.message]
        assert len(vad_messages) == 1
        msg = vad_messages[0]
        assert "1 silent section" in msg
        assert "sections" not in msg


# ---------------------------------------------------------------------------
# Transcript caching tests
# ---------------------------------------------------------------------------

import wave as _wave_module2


def _write_minimal_wav(path) -> None:
    """Write a minimal valid WAV file (0.1s silence)."""
    with _wave_module2.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 1600)


class TestTranscriptCaching:
    """Verify transcript caching: write on first run, load on second run."""

    def _make_config_with_cache(self, tmp_path) -> Config:
        cfg = Config(work_dir=str(tmp_path))
        cfg.cache_dir = str(tmp_path / "cache")
        cfg.use_cache = True
        return cfg

    def test_cache_file_written_after_transcription(self, tmp_path):
        """After transcribing, a cache file is written to cache_dir."""
        wav = tmp_path / "audio.wav"
        _write_minimal_wav(wav)

        config = self._make_config_with_cache(tmp_path)

        whisper_segments = [{"start": 0.0, "end": 2.0, "text": "Hello"}]
        fake_model = _make_fake_model(_fake_whisper_result(whisper_segments))

        with patch("pipeline.transcriber.whisper") as mock_whisper:
            mock_whisper.load_model.return_value = fake_model
            transcribe(config, str(wav))

        cache_files = list((tmp_path / "cache").glob("*.json"))
        assert len(cache_files) == 1, "Expected exactly one cache file"

    def test_cache_hit_skips_whisper(self, tmp_path):
        """On second run with same file, Whisper is NOT called."""
        wav = tmp_path / "audio.wav"
        _write_minimal_wav(wav)

        config = self._make_config_with_cache(tmp_path)

        whisper_segments = [{"start": 0.0, "end": 2.0, "text": "Cached segment"}]
        fake_model = _make_fake_model(_fake_whisper_result(whisper_segments))

        # First run — populates cache
        with patch("pipeline.transcriber.whisper") as mock_whisper:
            mock_whisper.load_model.return_value = fake_model
            transcribe(config, str(wav))

        # Second run — should load from cache
        with patch("pipeline.transcriber.whisper") as mock_whisper2:
            mock_whisper2.load_model.return_value = fake_model
            result = transcribe(config, str(wav))

        mock_whisper2.load_model.assert_not_called()
        assert len(result.segments) == 1
        assert result.segments[0].text == "Cached segment"

    def test_cache_hit_logs_correct_message(self, tmp_path, caplog):
        """Cache hit logs '[Transcriber] Loaded transcript from cache (skipping Whisper)'."""
        import logging

        wav = tmp_path / "audio.wav"
        _write_minimal_wav(wav)

        config = self._make_config_with_cache(tmp_path)

        whisper_segments = [{"start": 0.0, "end": 1.0, "text": "Test"}]
        fake_model = _make_fake_model(_fake_whisper_result(whisper_segments))

        # First run — populate cache
        with patch("pipeline.transcriber.whisper") as mock_whisper:
            mock_whisper.load_model.return_value = fake_model
            transcribe(config, str(wav))

        # Second run — should hit cache and log
        with patch("pipeline.transcriber.whisper") as mock_whisper2:
            mock_whisper2.load_model.return_value = fake_model
            with caplog.at_level(logging.INFO, logger="pipeline.transcriber"):
                transcribe(config, str(wav))

        cache_msgs = [r.message for r in caplog.records if "Loaded transcript from cache" in r.message]
        assert len(cache_msgs) == 1
        assert "skipping Whisper" in cache_msgs[0]

    def test_no_cache_flag_forces_retranscription(self, tmp_path):
        """When use_cache=False, Whisper is always called even if cache exists."""
        wav = tmp_path / "audio.wav"
        _write_minimal_wav(wav)

        config = self._make_config_with_cache(tmp_path)

        whisper_segments = [{"start": 0.0, "end": 2.0, "text": "Fresh"}]
        fake_model = _make_fake_model(_fake_whisper_result(whisper_segments))

        # First run — populate cache
        with patch("pipeline.transcriber.whisper") as mock_whisper:
            mock_whisper.load_model.return_value = fake_model
            transcribe(config, str(wav))

        # Second run with use_cache=False — must call Whisper again
        config.use_cache = False
        with patch("pipeline.transcriber.whisper") as mock_whisper2:
            mock_whisper2.load_model.return_value = fake_model
            transcribe(config, str(wav))

        mock_whisper2.load_model.assert_called_once()

    def test_cache_key_differs_for_different_models(self, tmp_path):
        """Different whisper_model values produce different cache files."""
        wav = tmp_path / "audio.wav"
        _write_minimal_wav(wav)

        config_base = Config(
            work_dir=str(tmp_path),
            whisper_model="base",
            cache_dir=str(tmp_path / "cache"),
            use_cache=True,
        )
        config_small = Config(
            work_dir=str(tmp_path),
            whisper_model="small",
            cache_dir=str(tmp_path / "cache"),
            use_cache=True,
        )

        whisper_segments = [{"start": 0.0, "end": 1.0, "text": "Hi"}]
        fake_model = _make_fake_model(_fake_whisper_result(whisper_segments))

        with patch("pipeline.transcriber.whisper") as mock_whisper:
            mock_whisper.load_model.return_value = fake_model
            transcribe(config_base, str(wav))
            transcribe(config_small, str(wav))

        cache_files = list((tmp_path / "cache").glob("*.json"))
        assert len(cache_files) == 2, "Expected separate cache files for different models"

    def test_cache_written_to_work_dir_on_hit(self, tmp_path):
        """On a cache hit, transcript.json is still written to work_dir."""
        wav = tmp_path / "audio.wav"
        _write_minimal_wav(wav)

        config = self._make_config_with_cache(tmp_path)

        whisper_segments = [{"start": 0.0, "end": 1.0, "text": "Cached"}]
        fake_model = _make_fake_model(_fake_whisper_result(whisper_segments))

        # First run
        with patch("pipeline.transcriber.whisper") as mock_whisper:
            mock_whisper.load_model.return_value = fake_model
            transcribe(config, str(wav))

        # Remove work_dir transcript to confirm it gets re-written on cache hit
        work_transcript = tmp_path / "transcript.json"
        work_transcript.unlink()
        assert not work_transcript.exists()

        # Second run — cache hit should re-write transcript.json
        with patch("pipeline.transcriber.whisper") as mock_whisper2:
            mock_whisper2.load_model.return_value = fake_model
            transcribe(config, str(wav))

        assert work_transcript.exists(), "transcript.json should be written on cache hit"
