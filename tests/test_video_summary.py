"""Tests for video summary generation in pipeline/scorer.py.

Covers:
- generate_video_summary() function
- Sampling strategy
- Caching behavior
- LLM integration
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from config import Config
from pipeline.models import Segment, Transcript
from pipeline.scorer import generate_video_summary, _video_summary_cache


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


def make_transcript(num_segments: int = 100) -> Transcript:
    """Create a transcript with num_segments segments."""
    segments = []
    for i in range(num_segments):
        start = i * 3.0
        end = start + 3.0
        text = f"This is segment {i} with some test content."
        segments.append(make_segment(text=text, start=start, end=end))
    return Transcript(segments=segments)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_generate_video_summary_empty_transcript() -> None:
    """Test that empty transcript returns a fallback summary."""
    config = make_config()
    transcript = Transcript(segments=[])
    
    summary = generate_video_summary(config, transcript)
    
    assert summary == "Empty video with no transcript content."


def test_generate_video_summary_sampling_rate() -> None:
    """Test that sampling rate is calculated correctly (len(segments) // 20)."""
    config = make_config()
    transcript = make_transcript(num_segments=100)
    
    # Mock _call_llm to capture the prompt
    with patch("pipeline.scorer._call_llm") as mock_llm:
        mock_llm.return_value = "Test summary response"
        
        summary = generate_video_summary(config, transcript)
        
        # Verify LLM was called
        assert mock_llm.call_count == 1
        
        # Extract the prompt
        prompt = mock_llm.call_args[0][1]
        
        # Count how many segments appear in the condensed transcript
        # Sample rate should be 100 // 20 = 5, so we expect ~20 segments
        segment_count = prompt.count("[")  # Each segment starts with [timestamp]
        
        # Should be around 20 segments (±2 for edge cases)
        assert 18 <= segment_count <= 22, f"Expected ~20 segments, got {segment_count}"


def test_generate_video_summary_max_words_limit() -> None:
    """Test that condensed transcript respects 500-word limit."""
    config = make_config()
    
    # Create transcript with very long segments
    segments = []
    for i in range(50):
        start = i * 3.0
        end = start + 3.0
        # Each segment has 20 words
        text = " ".join([f"word{j}" for j in range(20)])
        segments.append(make_segment(text=text, start=start, end=end))
    
    transcript = Transcript(segments=segments)
    
    with patch("pipeline.scorer._call_llm") as mock_llm:
        mock_llm.return_value = "Test summary response"
        
        summary = generate_video_summary(config, transcript)
        
        # Extract the condensed transcript from the prompt
        prompt = mock_llm.call_args[0][1]
        
        # Extract the condensed transcript section
        start_marker = "Below is a condensed transcript"
        end_marker = "Provide a 2-3 sentence summary"
        
        start_idx = prompt.find(start_marker)
        end_idx = prompt.find(end_marker)
        
        condensed_section = prompt[start_idx:end_idx]
        
        # Count words in condensed transcript
        words = condensed_section.split()
        word_count = len([w for w in words if w.startswith("word")])
        
        # Should be <= 500 words
        assert word_count <= 500, f"Expected <= 500 words, got {word_count}"


def test_generate_video_summary_caching() -> None:
    """Test that video summary is cached and reused."""
    config = make_config()
    transcript = make_transcript(num_segments=50)
    video_path = "/path/to/video.mp4"
    
    # Clear cache
    _video_summary_cache.clear()
    
    with patch("pipeline.scorer._call_llm") as mock_llm:
        mock_llm.return_value = "Cached summary response"
        
        # First call should invoke LLM
        summary1 = generate_video_summary(config, transcript, video_path)
        assert mock_llm.call_count == 1
        assert summary1 == "Cached summary response"
        
        # Second call should use cache
        summary2 = generate_video_summary(config, transcript, video_path)
        assert mock_llm.call_count == 1  # Still 1, not called again
        assert summary2 == "Cached summary response"
        
        # Verify cache contains the entry
        assert video_path in _video_summary_cache
        assert _video_summary_cache[video_path] == "Cached summary response"


def test_generate_video_summary_no_cache_key() -> None:
    """Test that summary generation works without video_path (no caching)."""
    config = make_config()
    transcript = make_transcript(num_segments=50)
    
    with patch("pipeline.scorer._call_llm") as mock_llm:
        mock_llm.return_value = "Uncached summary response"
        
        # Call without video_path
        summary = generate_video_summary(config, transcript, video_path="")
        
        assert mock_llm.call_count == 1
        assert summary == "Uncached summary response"


def test_generate_video_summary_llm_error_fallback() -> None:
    """Test that LLM errors result in a fallback summary."""
    config = make_config()
    transcript = make_transcript(num_segments=50)
    video_path = "/path/to/video.mp4"
    
    # Clear cache
    _video_summary_cache.clear()
    
    from pipeline.exceptions import LLMScoringError
    
    with patch("pipeline.scorer._call_llm") as mock_llm:
        mock_llm.side_effect = LLMScoringError("LLM unavailable")
        
        summary = generate_video_summary(config, transcript, video_path)
        
        # Should return fallback summary
        assert "LLM summary unavailable" in summary
        assert "50 segments" in summary
        
        # Fallback should still be cached
        assert video_path in _video_summary_cache


def test_generate_video_summary_empty_llm_response() -> None:
    """Test that empty LLM response results in a fallback summary."""
    config = make_config()
    transcript = make_transcript(num_segments=50)
    
    with patch("pipeline.scorer._call_llm") as mock_llm:
        mock_llm.return_value = ""  # Empty response
        
        summary = generate_video_summary(config, transcript)
        
        # Should return fallback summary
        assert "Content type and themes unclear" in summary
        assert "50 segments" in summary


def test_generate_video_summary_strips_summary_prefix() -> None:
    """Test that 'Summary:' prefix is stripped from LLM response."""
    config = make_config()
    transcript = make_transcript(num_segments=50)
    
    with patch("pipeline.scorer._call_llm") as mock_llm:
        mock_llm.return_value = "Summary: This is a gaming video with high energy."
        
        summary = generate_video_summary(config, transcript)
        
        # Should strip the "Summary:" prefix
        assert summary == "This is a gaming video with high energy."
        assert not summary.startswith("Summary:")


def test_generate_video_summary_prompt_format() -> None:
    """Test that the LLM prompt contains all required elements."""
    config = make_config()
    transcript = make_transcript(num_segments=100)
    
    with patch("pipeline.scorer._call_llm") as mock_llm:
        mock_llm.return_value = "Test summary"
        
        generate_video_summary(config, transcript)
        
        # Extract the prompt
        prompt = mock_llm.call_args[0][1]
        
        # Verify prompt contains required elements
        assert "analyzing a video transcript" in prompt
        assert "condensed transcript" in prompt
        assert "Content type" in prompt
        assert "Main topics or activities" in prompt
        assert "Overall energy level" in prompt
        assert "Key recurring themes" in prompt
        assert "Summary:" in prompt
        
        # Verify duration is included
        assert "-minute video" in prompt


def test_generate_video_summary_integration() -> None:
    """Integration test: verify full flow with realistic transcript."""
    config = make_config()
    
    # Create a realistic gaming transcript
    segments = [
        make_segment("Oh my god, did you see that?", start=0.0, end=3.0),
        make_segment("That was insane!", start=3.0, end=5.0),
        make_segment("Let me try this again.", start=5.0, end=8.0),
        make_segment("Watch this, watch this!", start=8.0, end=11.0),
        make_segment("No way, I can't believe it!", start=11.0, end=14.0),
    ] * 20  # Repeat to get 100 segments
    
    transcript = Transcript(segments=segments)
    
    with patch("pipeline.scorer._call_llm") as mock_llm:
        mock_llm.return_value = (
            "This is a high-energy gaming video where the creator attempts "
            "challenging gameplay moments with frequent excited reactions. "
            "The content features repeated attempts and surprise outcomes."
        )
        
        summary = generate_video_summary(config, transcript, video_path="/test/video.mp4")
        
        # Verify summary is returned
        assert "high-energy gaming video" in summary
        assert len(summary) > 50  # Should be a substantial summary
        
        # Verify caching worked
        assert "/test/video.mp4" in _video_summary_cache


def test_generate_video_summary_single_segment() -> None:
    """Test that a transcript with a single segment is handled correctly."""
    config = make_config()
    transcript = Transcript(segments=[make_segment("Single segment.", start=0.0, end=3.0)])
    
    with patch("pipeline.scorer._call_llm") as mock_llm:
        mock_llm.return_value = "Brief video with minimal content."
        
        summary = generate_video_summary(config, transcript)
        
        assert mock_llm.call_count == 1
        assert summary == "Brief video with minimal content."


def test_generate_video_summary_very_short_segments() -> None:
    """Test handling of transcript with many very short segments."""
    config = make_config()
    
    # Create 100 segments with only 1-2 words each
    segments = []
    for i in range(100):
        start = i * 0.5
        end = start + 0.5
        text = f"Word{i}"
        segments.append(make_segment(text=text, start=start, end=end))
    
    transcript = Transcript(segments=segments)
    
    with patch("pipeline.scorer._call_llm") as mock_llm:
        mock_llm.return_value = "Fast-paced content with brief utterances."
        
        summary = generate_video_summary(config, transcript)
        
        assert mock_llm.call_count == 1
        assert summary == "Fast-paced content with brief utterances."


def test_generate_video_summary_segments_with_empty_text() -> None:
    """Test that segments with empty text are skipped in condensed transcript."""
    config = make_config()
    
    segments = []
    for i in range(60):
        start = i * 3.0
        end = start + 3.0
        # Make segments at indices 5, 15, 25, 35, 45, 55 empty (every 10th starting at 5)
        # This ensures some sampled segments will be empty
        text = "" if i % 10 == 5 else f"Segment {i} content."
        segments.append(make_segment(text=text, start=start, end=end))
    
    transcript = Transcript(segments=segments)
    
    with patch("pipeline.scorer._call_llm") as mock_llm:
        mock_llm.return_value = "Video with intermittent speech."
        
        summary = generate_video_summary(config, transcript)
        
        # Extract the prompt to verify empty segments were skipped
        prompt = mock_llm.call_args[0][1]
        
        # With 60 segments and sample_rate=20, step_size = 60//20 = 3
        # So we sample segments at indices: 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57
        # Of these, indices 15 and 45 are empty (i % 10 == 5)
        # So we should see 20 - 2 = 18 segments with content
        
        # Count actual segment lines (those with timestamps and content)
        lines_with_content = [line for line in prompt.split('\n') if line.strip().startswith('[') and 'Segment' in line]
        
        # Should have 18 segments (20 sampled - 2 empty)
        assert len(lines_with_content) == 18, f"Expected 18 segments (2 empty ones skipped), got {len(lines_with_content)}"


def test_generate_video_summary_very_long_video() -> None:
    """Test handling of very long video (many segments)."""
    config = make_config()
    
    # Create a 2-hour video (1200 segments at 6 seconds each)
    segments = []
    for i in range(1200):
        start = i * 6.0
        end = start + 6.0
        text = f"Long video segment {i} with content."
        segments.append(make_segment(text=text, start=start, end=end))
    
    transcript = Transcript(segments=segments)
    
    with patch("pipeline.scorer._call_llm") as mock_llm:
        mock_llm.return_value = "Extended content spanning multiple hours."
        
        summary = generate_video_summary(config, transcript)
        
        # Verify LLM was called
        assert mock_llm.call_count == 1
        
        # Extract the prompt
        prompt = mock_llm.call_args[0][1]
        
        # Verify duration is calculated correctly (should be ~120 minutes)
        assert "120" in prompt or "119" in prompt or "121" in prompt
        
        # Verify sampling still respects word limit
        condensed_section = prompt.split("Below is a condensed transcript")[1].split("Provide a 2-3 sentence summary")[0]
        words = condensed_section.split()
        word_count = len([w for w in words if "segment" in w.lower()])
        
        # Should still be reasonable number of samples
        assert word_count <= 500


def test_generate_video_summary_cache_different_videos() -> None:
    """Test that different video paths have separate cache entries."""
    config = make_config()
    transcript1 = make_transcript(num_segments=50)
    transcript2 = make_transcript(num_segments=50)
    
    video_path1 = "/path/to/video1.mp4"
    video_path2 = "/path/to/video2.mp4"
    
    # Clear cache
    _video_summary_cache.clear()
    
    with patch("pipeline.scorer._call_llm") as mock_llm:
        mock_llm.side_effect = ["Summary for video 1", "Summary for video 2"]
        
        # Generate summaries for both videos
        summary1 = generate_video_summary(config, transcript1, video_path1)
        summary2 = generate_video_summary(config, transcript2, video_path2)
        
        # Both should have been called
        assert mock_llm.call_count == 2
        
        # Both should be cached separately
        assert video_path1 in _video_summary_cache
        assert video_path2 in _video_summary_cache
        assert _video_summary_cache[video_path1] == "Summary for video 1"
        assert _video_summary_cache[video_path2] == "Summary for video 2"


def test_generate_video_summary_llm_timeout() -> None:
    """Test that LLM timeout is handled gracefully."""
    config = make_config()
    transcript = make_transcript(num_segments=50)
    video_path = "/path/to/video.mp4"
    
    # Clear cache
    _video_summary_cache.clear()
    
    from pipeline.exceptions import LLMScoringError
    
    with patch("pipeline.scorer._call_llm") as mock_llm:
        mock_llm.side_effect = LLMScoringError("LLM endpoint unreachable: Request timed out")
        
        summary = generate_video_summary(config, transcript, video_path)
        
        # Should return fallback summary
        assert "LLM summary unavailable" in summary
        assert "50 segments" in summary
        
        # Fallback should still be cached
        assert video_path in _video_summary_cache


def test_generate_video_summary_llm_connection_error() -> None:
    """Test that LLM connection errors are handled gracefully."""
    config = make_config()
    transcript = make_transcript(num_segments=50)
    video_path = "/path/to/video.mp4"
    
    # Clear cache
    _video_summary_cache.clear()
    
    from pipeline.exceptions import LLMScoringError
    
    with patch("pipeline.scorer._call_llm") as mock_llm:
        mock_llm.side_effect = LLMScoringError("LLM endpoint unreachable: Connection refused")
        
        summary = generate_video_summary(config, transcript, video_path)
        
        # Should return fallback summary
        assert "LLM summary unavailable" in summary
        
        # Fallback should still be cached
        assert video_path in _video_summary_cache


def test_generate_video_summary_custom_sample_rate() -> None:
    """Test that custom video_summary_sample_rate is respected."""
    config = make_config()
    config.video_summary_sample_rate = 10  # Sample every 10th segment instead of 20
    
    transcript = make_transcript(num_segments=100)
    
    with patch("pipeline.scorer._call_llm") as mock_llm:
        mock_llm.return_value = "Test summary with custom sample rate"
        
        summary = generate_video_summary(config, transcript)
        
        # Extract the prompt
        prompt = mock_llm.call_args[0][1]
        
        # Count segments in condensed transcript
        segment_count = prompt.count("[")
        
        # With sample rate of 10, we expect ~10 segments (100 // 10)
        assert 8 <= segment_count <= 12, f"Expected ~10 segments with sample_rate=10, got {segment_count}"


def test_generate_video_summary_whitespace_handling() -> None:
    """Test that segments with only whitespace are handled correctly."""
    config = make_config()
    
    segments = []
    for i in range(50):
        start = i * 3.0
        end = start + 3.0
        # Mix of normal text, whitespace-only, and empty
        if i % 3 == 0:
            text = f"Segment {i} content."
        elif i % 3 == 1:
            text = "   \t\n  "  # Whitespace only
        else:
            text = ""
        segments.append(make_segment(text=text, start=start, end=end))
    
    transcript = Transcript(segments=segments)
    
    with patch("pipeline.scorer._call_llm") as mock_llm:
        mock_llm.return_value = "Video with sparse speech."
        
        summary = generate_video_summary(config, transcript)
        
        # Should complete without errors
        assert mock_llm.call_count == 1
        assert summary == "Video with sparse speech."
