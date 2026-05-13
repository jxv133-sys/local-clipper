"""Integration tests for video summary in LLM prompts.

Tests that video summary is correctly integrated into _score_window_with_llm prompts.

**Validates: Requirements 2.4, 2.5**
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from config import Config
from pipeline.models import Segment, Transcript
from pipeline.scorer import _score_window_with_llm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(
    llm_enabled: bool = True,
    llm_endpoint: str = "http://localhost:11434/api/generate",
    llm_model: str = "llama3",
) -> Config:
    cfg = Config(work_dir="/tmp/test")
    cfg.llm_enabled = llm_enabled
    cfg.llm_endpoint = llm_endpoint
    cfg.llm_model = llm_model
    return cfg


def make_segment(text: str = "hello world", start: float = 0.0, end: float = 1.0) -> Segment:
    return Segment(start=start, end=end, text=text)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

def test_video_summary_included_in_prompt() -> None:
    """Test that video summary is prepended to LLM prompt with correct format.
    
    **Validates: Requirement 2.4**
    """
    config = make_config()
    
    # Create test segments
    segments = [
        make_segment("This is segment 1", start=0.0, end=3.0),
        make_segment("This is segment 2", start=3.0, end=6.0),
        make_segment("This is segment 3", start=6.0, end=9.0),
    ]
    
    video_summary = "This is a gaming video with high energy and frequent reactions."
    
    # Mock _call_llm to capture the prompt
    with patch("pipeline.scorer._call_llm") as mock_llm:
        mock_llm.return_value = (
            "SCORE: 7\n"
            "TITLE: Amazing Gaming Moment\n"
            "DESCRIPTION: Player makes an incredible play.\n"
            "TAGS: #gaming #shorts #viral"
        )
        
        # Call _score_window_with_llm with video summary
        score, metadata = _score_window_with_llm(
            config=config,
            seed_idx=1,
            all_segments=segments,
            all_audio_scores=[0.5, 0.8, 0.6],
            all_raw_rms=[0.3, 0.6, 0.4],
            global_rms_mean=0.5,
            global_rms_max=1.0,
            video_summary=video_summary,
        )
        
        # Verify LLM was called
        assert mock_llm.call_count == 1
        
        # Extract the prompt
        prompt = mock_llm.call_args[0][1]
        
        # Verify video summary is included with correct format
        assert "VIDEO CONTEXT:" in prompt
        assert video_summary in prompt
        
        # Verify format: "VIDEO CONTEXT: {summary}\n\n"
        expected_format = f"VIDEO CONTEXT: {video_summary}\n\n"
        assert expected_format in prompt
        
        # Verify the video context appears before the window transcript
        video_context_pos = prompt.find("VIDEO CONTEXT:")
        window_transcript_pos = prompt.find("WINDOW TRANSCRIPT:")
        assert video_context_pos < window_transcript_pos, \
            "VIDEO CONTEXT should appear before WINDOW TRANSCRIPT"


def test_video_summary_not_included_when_empty() -> None:
    """Test that when video summary is empty, VIDEO CONTEXT is not included."""
    config = make_config()
    
    segments = [
        make_segment("This is segment 1", start=0.0, end=3.0),
        make_segment("This is segment 2", start=3.0, end=6.0),
    ]
    
    with patch("pipeline.scorer._call_llm") as mock_llm:
        mock_llm.return_value = (
            "SCORE: 5\n"
            "TITLE: Test Clip\n"
            "DESCRIPTION: Test description.\n"
            "TAGS: #test"
        )
        
        # Call without video summary (empty string)
        score, metadata = _score_window_with_llm(
            config=config,
            seed_idx=0,
            all_segments=segments,
            video_summary="",
        )
        
        # Extract the prompt
        prompt = mock_llm.call_args[0][1]
        
        # Verify VIDEO CONTEXT is not included when summary is empty
        assert "VIDEO CONTEXT:" not in prompt


def test_prompt_instructs_relative_scoring() -> None:
    """Test that prompt instructs LLM to score relative to video baseline.
    
    **Validates: Requirement 2.5**
    """
    config = make_config()
    
    segments = [
        make_segment("This is segment 1", start=0.0, end=3.0),
        make_segment("This is segment 2", start=3.0, end=6.0),
    ]
    
    video_summary = "This is a calm podcast with moderate energy."
    
    with patch("pipeline.scorer._call_llm") as mock_llm:
        mock_llm.return_value = (
            "SCORE: 6\n"
            "TITLE: Interesting Point\n"
            "DESCRIPTION: Host makes an insightful observation.\n"
            "TAGS: #podcast"
        )
        
        # Call with video summary
        score, metadata = _score_window_with_llm(
            config=config,
            seed_idx=0,
            all_segments=segments,
            video_summary=video_summary,
        )
        
        # Extract the prompt
        prompt = mock_llm.call_args[0][1]
        
        # Verify prompt contains instructions about relative scoring
        assert "Score this moment relative to THIS VIDEO'S baseline" in prompt
        assert "not generic content" in prompt
        assert "Use the VIDEO CONTEXT above" in prompt


def test_video_summary_with_special_characters() -> None:
    """Test that video summary with special characters is handled correctly."""
    config = make_config()
    
    segments = [
        make_segment("Test segment", start=0.0, end=3.0),
    ]
    
    # Video summary with special characters
    video_summary = 'This is a "gaming" video with $pecial ch@racters & symbols!'
    
    with patch("pipeline.scorer._call_llm") as mock_llm:
        mock_llm.return_value = (
            "SCORE: 7\n"
            "TITLE: Test\n"
            "DESCRIPTION: Test.\n"
            "TAGS: #test"
        )
        
        # Call with video summary containing special characters
        score, metadata = _score_window_with_llm(
            config=config,
            seed_idx=0,
            all_segments=segments,
            video_summary=video_summary,
        )
        
        # Extract the prompt
        prompt = mock_llm.call_args[0][1]
        
        # Verify video summary is included correctly with special characters
        assert video_summary in prompt
        assert "VIDEO CONTEXT:" in prompt


def test_video_summary_with_newlines() -> None:
    """Test that video summary with newlines is handled correctly."""
    config = make_config()
    
    segments = [
        make_segment("Test segment", start=0.0, end=3.0),
    ]
    
    # Video summary with newlines
    video_summary = "This is a gaming video.\nIt has high energy.\nFrequent reactions occur."
    
    with patch("pipeline.scorer._call_llm") as mock_llm:
        mock_llm.return_value = (
            "SCORE: 8\n"
            "TITLE: Test\n"
            "DESCRIPTION: Test.\n"
            "TAGS: #test"
        )
        
        # Call with video summary containing newlines
        score, metadata = _score_window_with_llm(
            config=config,
            seed_idx=0,
            all_segments=segments,
            video_summary=video_summary,
        )
        
        # Extract the prompt
        prompt = mock_llm.call_args[0][1]
        
        # Verify video summary is included correctly with newlines
        assert video_summary in prompt
        assert "VIDEO CONTEXT:" in prompt


def test_video_summary_very_long() -> None:
    """Test that very long video summaries are handled correctly."""
    config = make_config()
    
    segments = [
        make_segment("Test segment", start=0.0, end=3.0),
    ]
    
    # Very long video summary (500+ characters)
    video_summary = (
        "This is a very long video summary that contains a lot of information about the video. "
        "It describes the content type, main topics, energy level, and recurring themes in great detail. "
        "The video is a gaming stream where the creator plays Minecraft survival mode. "
        "Throughout the video, the creator builds a base, fights mobs, and explores caves. "
        "The energy level is consistently high with frequent excited reactions to unexpected events. "
        "Key recurring themes include resource gathering, combat encounters, and building projects."
    )
    
    with patch("pipeline.scorer._call_llm") as mock_llm:
        mock_llm.return_value = (
            "SCORE: 9\n"
            "TITLE: Epic Moment\n"
            "DESCRIPTION: Amazing gameplay.\n"
            "TAGS: #gaming"
        )
        
        # Call with very long video summary
        score, metadata = _score_window_with_llm(
            config=config,
            seed_idx=0,
            all_segments=segments,
            video_summary=video_summary,
        )
        
        # Extract the prompt
        prompt = mock_llm.call_args[0][1]
        
        # Verify video summary is included correctly
        assert video_summary in prompt
        assert "VIDEO CONTEXT:" in prompt


def test_video_summary_position_in_prompt() -> None:
    """Test that video summary appears in the correct position in the prompt."""
    config = make_config()
    
    segments = [
        make_segment("Test segment", start=0.0, end=3.0),
    ]
    
    video_summary = "This is a test video summary."
    
    with patch("pipeline.scorer._call_llm") as mock_llm:
        mock_llm.return_value = (
            "SCORE: 6\n"
            "TITLE: Test\n"
            "DESCRIPTION: Test.\n"
            "TAGS: #test"
        )
        
        # Call with video summary
        score, metadata = _score_window_with_llm(
            config=config,
            seed_idx=0,
            all_segments=segments,
            video_summary=video_summary,
        )
        
        # Extract the prompt
        prompt = mock_llm.call_args[0][1]
        
        # Verify order of elements in prompt
        intro_pos = prompt.find("You are a strict YouTube Shorts editor")
        video_context_pos = prompt.find("VIDEO CONTEXT:")
        window_transcript_pos = prompt.find("WINDOW TRANSCRIPT:")
        
        # VIDEO CONTEXT should appear after intro and before WINDOW TRANSCRIPT
        assert intro_pos < video_context_pos < window_transcript_pos, \
            "VIDEO CONTEXT should appear between intro and WINDOW TRANSCRIPT"
