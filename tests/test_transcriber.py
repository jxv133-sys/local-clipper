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

        fake_model.transcribe.assert_called_once_with(str(wav), word_timestamps=True)

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
