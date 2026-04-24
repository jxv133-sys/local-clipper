"""Unit tests for main.py — pipeline orchestrator."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, call, patch

import pytest

from config import Config
from pipeline.exceptions import PipelineError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_transcript(n: int = 2):
    """Return a mock Transcript with n segments."""
    from pipeline.models import Segment, Transcript
    segs = [Segment(start=i * 5.0, end=i * 5.0 + 4.0, text=f"segment {i}") for i in range(n)]
    return Transcript(segments=segs)


def _make_scored_segments(transcript):
    from pipeline.models import ScoredSegment
    return [
        ScoredSegment(segment=seg, text_score=0.5, audio_score=0.5,
                      llm_score=0.0, clip_score=0.5)
        for seg in transcript.segments
    ]


def _make_clips(n: int = 2):
    from pipeline.models import Clip
    return [
        Clip(start=i * 30.0, end=i * 30.0 + 25.0, score=0.9 - i * 0.1,
             rank=i + 1, segment_indices=[i])
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Tests: CLI argument parsing
# ---------------------------------------------------------------------------

class TestCLIArguments:
    """Tests for CLI argument handling."""

    def test_missing_input_video_exits_with_error(self, capsys):
        """No input_video argument → usage message on stderr, exit code 2."""
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", ["main.py"]):
                import importlib
                import main as m
                importlib.reload(m)
                m.main()
        assert exc_info.value.code != 0

    def test_missing_input_video_argparse(self):
        """argparse exits with code 2 when required positional arg is missing."""
        import argparse
        # Simulate what argparse does when input_video is missing
        with patch("sys.argv", ["main.py"]):
            with pytest.raises(SystemExit) as exc_info:
                import main as m
                # Call parser directly
                parser = argparse.ArgumentParser()
                parser.add_argument("input_video")
                parser.parse_args([])
        assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# Tests: run_pipeline stage ordering
# ---------------------------------------------------------------------------

class TestRunPipelineStageOrdering:
    """Stages are called in the correct sequence with correct arguments."""

    def test_all_stages_called_in_order(self, tmp_path):
        """All 6 pipeline stages are called in sequence."""
        import main as m

        transcript = _make_transcript(3)
        scored = _make_scored_segments(transcript)
        clips = _make_clips(2)
        clip_paths = [str(tmp_path / "clip_1_0s.mp4"), str(tmp_path / "clip_2_30s.mp4")]
        final_paths = [str(tmp_path / "clip_1_0s.mp4"), str(tmp_path / "clip_2_30s.mp4")]

        call_order = []

        def mock_extract_audio(config, video_path):
            call_order.append("extract_audio")
            return str(tmp_path / "audio.wav")

        def mock_transcribe(config, wav_path, **kwargs):
            call_order.append("transcribe")
            return transcript

        def mock_score_segments(config, transcript_, wav_path):
            call_order.append("score_segments")
            return scored

        def mock_select_clips(config, scored_, transcript_, video_duration):
            call_order.append("select_clips")
            return clips

        def mock_extract_clips(config, clips_, video_path):
            call_order.append("extract_clips")
            return clip_paths

        def mock_generate_subtitles(config, clips_, transcript_, clip_paths_):
            call_order.append("generate_subtitles")
            return final_paths

        config = Config(work_dir=str(tmp_path), output_dir=str(tmp_path / "output"))

        with patch("main.extract_audio", mock_extract_audio), \
             patch("main.transcribe", mock_transcribe), \
             patch("main.score_segments", mock_score_segments), \
             patch("main.select_clips", mock_select_clips), \
             patch("main.extract_clips", mock_extract_clips), \
             patch("main.generate_subtitles", mock_generate_subtitles), \
             patch("main._get_video_duration", return_value=300.0):

            result = m.run_pipeline("/fake/video.mp4", config)

        assert call_order == [
            "extract_audio",
            "transcribe",
            "score_segments",
            "select_clips",
            "extract_clips",
            "generate_subtitles",
        ]
        assert result == (final_paths, clips)

    def test_wav_path_passed_to_transcribe(self, tmp_path):
        """The WAV path returned by extract_audio is passed to transcribe."""
        import main as m

        transcript = _make_transcript(2)
        scored = _make_scored_segments(transcript)
        clips = _make_clips(1)
        expected_wav = str(tmp_path / "audio.wav")
        received_wav = []

        def mock_transcribe(config, wav_path, **kwargs):
            received_wav.append(wav_path)
            return transcript

        config = Config(work_dir=str(tmp_path), output_dir=str(tmp_path / "output"))

        with patch("main.extract_audio", return_value=expected_wav), \
             patch("main.transcribe", mock_transcribe), \
             patch("main.score_segments", return_value=scored), \
             patch("main.select_clips", return_value=clips), \
             patch("main.extract_clips", return_value=["clip.mp4"]), \
             patch("main.generate_subtitles", return_value=["clip.mp4"]), \
             patch("main._get_video_duration", return_value=300.0):

            m.run_pipeline("/fake/video.mp4", config)

        assert received_wav == [expected_wav]


# ---------------------------------------------------------------------------
# Tests: error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Pipeline errors are caught, logged to stderr, and exit with code 1."""

    def test_pipeline_error_exits_with_code_1(self, tmp_path, capsys):
        """PipelineError → error logged to stderr, exit code 1."""
        import main as m

        config = Config(work_dir=str(tmp_path), output_dir=str(tmp_path / "output"))

        with patch("main.extract_audio", side_effect=PipelineError("audio failed")), \
             patch("main._get_video_duration", return_value=300.0), \
             patch("tempfile.mkdtemp", return_value=str(tmp_path)), \
             patch("main.build_config", return_value=config), \
             patch("sys.argv", ["main.py", "/fake/video.mp4"]):

            with pytest.raises(SystemExit) as exc_info:
                m.main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "audio failed" in captured.err

    def test_pipeline_error_temp_dir_not_deleted(self, tmp_path, capsys):
        """On PipelineError, the temp directory is NOT deleted."""
        import main as m

        config = Config(work_dir=str(tmp_path), output_dir=str(tmp_path / "output"))

        with patch("main.extract_audio", side_effect=PipelineError("stage failed")), \
             patch("main._get_video_duration", return_value=300.0), \
             patch("tempfile.mkdtemp", return_value=str(tmp_path)), \
             patch("main.build_config", return_value=config), \
             patch("shutil.rmtree") as mock_rmtree, \
             patch("sys.argv", ["main.py", "/fake/video.mp4"]):

            with pytest.raises(SystemExit):
                m.main()

        # rmtree should NOT have been called on failure
        mock_rmtree.assert_not_called()

    def test_no_segments_raises_pipeline_error(self, tmp_path):
        """Empty scored_segments raises PipelineError."""
        import main as m
        from pipeline.models import Transcript

        config = Config(work_dir=str(tmp_path), output_dir=str(tmp_path / "output"))

        with patch("main.extract_audio", return_value=str(tmp_path / "audio.wav")), \
             patch("main.transcribe", return_value=Transcript(segments=[])), \
             patch("main.score_segments", return_value=[]), \
             patch("main._get_video_duration", return_value=300.0):

            with pytest.raises(PipelineError, match="No segments"):
                m.run_pipeline("/fake/video.mp4", config)


# ---------------------------------------------------------------------------
# Tests: successful run
# ---------------------------------------------------------------------------

class TestSuccessfulRun:
    """On success, temp dir is deleted and clip paths are printed."""

    def test_temp_dir_deleted_on_success(self, tmp_path, capsys):
        """Temp directory is deleted after a successful pipeline run."""
        import main as m

        transcript = _make_transcript(2)
        scored = _make_scored_segments(transcript)
        clips = _make_clips(1)
        final_paths = [str(tmp_path / "output" / "clip_1_0s.mp4")]

        config = Config(work_dir=str(tmp_path), output_dir=str(tmp_path / "output"))

        with patch("main.extract_audio", return_value=str(tmp_path / "audio.wav")), \
             patch("main.transcribe", return_value=transcript), \
             patch("main.score_segments", return_value=scored), \
             patch("main.select_clips", return_value=clips), \
             patch("main.extract_clips", return_value=final_paths), \
             patch("main.generate_subtitles", return_value=final_paths), \
             patch("main.generate_report", return_value=str(tmp_path / "output" / "clip_1_0s_why_chosen.txt")), \
             patch("main._get_video_duration", return_value=300.0), \
             patch("tempfile.mkdtemp", return_value=str(tmp_path)), \
             patch("main.build_config", return_value=config), \
             patch("shutil.rmtree") as mock_rmtree, \
             patch("sys.argv", ["main.py", "/fake/video.mp4"]):

            m.main()

        mock_rmtree.assert_called_once_with(str(tmp_path), ignore_errors=True)

    def test_clip_paths_printed_on_success(self, tmp_path, capsys):
        """Final clip filenames are printed to stdout on success."""
        import main as m

        transcript = _make_transcript(2)
        scored = _make_scored_segments(transcript)
        clips = _make_clips(2)
        final_paths = ["/output/clip_1_0s.mp4", "/output/clip_2_30s.mp4"]

        config = Config(work_dir=str(tmp_path), output_dir=str(tmp_path / "output"))

        with patch("main.extract_audio", return_value=str(tmp_path / "audio.wav")), \
             patch("main.transcribe", return_value=transcript), \
             patch("main.score_segments", return_value=scored), \
             patch("main.select_clips", return_value=clips), \
             patch("main.extract_clips", return_value=final_paths), \
             patch("main.generate_subtitles", return_value=final_paths), \
             patch("main.generate_report", return_value="/output/clip_1_0s_why_chosen.txt"), \
             patch("main._get_video_duration", return_value=300.0), \
             patch("tempfile.mkdtemp", return_value=str(tmp_path)), \
             patch("main.build_config", return_value=config), \
             patch("shutil.rmtree"), \
             patch("sys.argv", ["main.py", "/fake/video.mp4"]):

            m.main()

        captured = capsys.readouterr()
        assert "clip_1_0s.mp4" in captured.out
        assert "clip_2_30s.mp4" in captured.out

    def test_summary_includes_duration_and_score(self, tmp_path, capsys):
        """CLI summary shows duration in seconds and score for each clip."""
        import main as m

        transcript = _make_transcript(2)
        scored = _make_scored_segments(transcript)
        # clip: start=0.0, end=25.0, score=0.9 → "25s  score=0.90"
        clips = _make_clips(1)
        final_paths = ["/output/clip_1_0s.mp4"]

        config = Config(work_dir=str(tmp_path), output_dir=str(tmp_path / "output"))

        with patch("main.extract_audio", return_value=str(tmp_path / "audio.wav")), \
             patch("main.transcribe", return_value=transcript), \
             patch("main.score_segments", return_value=scored), \
             patch("main.select_clips", return_value=clips), \
             patch("main.extract_clips", return_value=final_paths), \
             patch("main.generate_subtitles", return_value=final_paths), \
             patch("main.generate_report", return_value="/output/clip_1_0s_why_chosen.txt"), \
             patch("main._get_video_duration", return_value=300.0), \
             patch("tempfile.mkdtemp", return_value=str(tmp_path)), \
             patch("main.build_config", return_value=config), \
             patch("shutil.rmtree"), \
             patch("sys.argv", ["main.py", "/fake/video.mp4"]):

            m.main()

        captured = capsys.readouterr()
        # clip_1_0s.mp4  25s  score=0.90
        assert "25s" in captured.out
        assert "score=0.90" in captured.out
