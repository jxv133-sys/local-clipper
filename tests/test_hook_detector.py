"""Tests for LLM-based hook detection."""

import pytest
from unittest.mock import patch, MagicMock

from config import Config
from pipeline.models import Segment
from pipeline.hook_detector import (
    detect_hooks,
    get_hook_score_at_time,
    get_hook_score_for_window,
    Hook,
    _call_llm_for_hook,
)


@pytest.fixture
def config():
    """Create a test config with LLM enabled."""
    cfg = Config(work_dir="/tmp/test")
    cfg.llm_enabled = True
    cfg.llm_endpoint = "http://localhost:11434/api/generate"
    cfg.llm_model = "llama3"
    return cfg


@pytest.fixture
def segments():
    """Create test segments."""
    return [
        Segment(start=0.0, end=2.0, text="Hello everyone"),
        Segment(start=2.0, end=4.0, text="What if I told you"),
        Segment(start=4.0, end=6.0, text="something amazing happened"),
        Segment(start=6.0, end=8.0, text="that will blow your mind"),
        Segment(start=8.0, end=10.0, text="Let me explain"),
    ]


class TestCallLLMForHook:
    def test_successful_hook_detection(self):
        """Test successful hook detection with valid JSON response."""
        with patch('requests.post') as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "response": '{"hook_score": 0.85, "hook_type": "question"}'
            }
            mock_post.return_value = mock_response
            
            score, hook_type = _call_llm_for_hook(
                "http://localhost:11434/api/generate",
                "llama3",
                "What if I told you something amazing?"
            )
            
            assert score == 0.85
            assert hook_type == "question"
    
    def test_score_clamped_to_range(self):
        """Test that scores outside 0-1 are clamped."""
        with patch('requests.post') as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "response": '{"hook_score": 1.5, "hook_type": "reveal"}'
            }
            mock_post.return_value = mock_response
            
            score, hook_type = _call_llm_for_hook(
                "http://localhost:11434/api/generate",
                "llama3",
                "Test text"
            )
            
            assert score == 1.0  # Clamped to max
            assert hook_type == "reveal"
    
    def test_invalid_hook_type_defaults_to_none(self):
        """Test that invalid hook types default to 'none'."""
        with patch('requests.post') as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "response": '{"hook_score": 0.7, "hook_type": "invalid_type"}'
            }
            mock_post.return_value = mock_response
            
            score, hook_type = _call_llm_for_hook(
                "http://localhost:11434/api/generate",
                "llama3",
                "Test text"
            )
            
            assert score == 0.7
            assert hook_type == "none"
    
    def test_empty_response_returns_zero(self):
        """Test that empty LLM response returns 0.0 score."""
        with patch('requests.post') as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {"response": ""}
            mock_post.return_value = mock_response
            
            score, hook_type = _call_llm_for_hook(
                "http://localhost:11434/api/generate",
                "llama3",
                "Test text"
            )
            
            assert score == 0.0
            assert hook_type == "none"
    
    def test_json_parse_error_returns_zero(self):
        """Test that JSON parse errors return 0.0 score."""
        with patch('requests.post') as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "response": "This is not valid JSON"
            }
            mock_post.return_value = mock_response
            
            score, hook_type = _call_llm_for_hook(
                "http://localhost:11434/api/generate",
                "llama3",
                "Test text"
            )
            
            assert score == 0.0
            assert hook_type == "none"


class TestDetectHooks:
    def test_llm_disabled_returns_empty(self, segments):
        """Test that hook detection returns empty list when LLM disabled."""
        config = Config(work_dir="/tmp/test")
        config.llm_enabled = False
        
        hooks = detect_hooks(config, segments)
        
        assert hooks == []
    
    def test_empty_segments_returns_empty(self, config):
        """Test that empty segments list returns empty hooks."""
        hooks = detect_hooks(config, [])
        
        assert hooks == []
    
    def test_sliding_window_with_stride(self, config, segments):
        """Test that sliding window works with specified stride."""
        with patch('pipeline.hook_detector._call_llm_for_hook') as mock_llm:
            # Return high score for all windows
            mock_llm.return_value = (0.8, "question")
            
            hooks = detect_hooks(
                config,
                segments,
                window_size=2,
                stride=1,
                min_words=1,
                score_threshold=0.4
            )
            
            # Should have multiple hooks due to sliding window
            assert len(hooks) > 0
            # Each hook should have valid time range
            for hook in hooks:
                assert hook.start_time >= 0.0
                assert hook.end_time > hook.start_time
                assert hook.hook_score == 0.8
                assert hook.hook_type == "question"
    
    def test_score_threshold_filters_low_scores(self, config, segments):
        """Test that hooks below threshold are filtered out."""
        with patch('pipeline.hook_detector._call_llm_for_hook') as mock_llm:
            # Return low score
            mock_llm.return_value = (0.3, "none")
            
            hooks = detect_hooks(
                config,
                segments,
                window_size=3,
                stride=2,
                min_words=1,
                score_threshold=0.4
            )
            
            # Should have no hooks (all below threshold)
            assert len(hooks) == 0
    
    def test_min_words_filters_short_windows(self, config):
        """Test that windows with too few words are skipped."""
        short_segments = [
            Segment(start=0.0, end=1.0, text="Hi"),
            Segment(start=1.0, end=2.0, text="There"),
        ]
        
        with patch('pipeline.hook_detector._call_llm_for_hook') as mock_llm:
            mock_llm.return_value = (0.8, "question")
            
            hooks = detect_hooks(
                config,
                short_segments,
                window_size=2,
                stride=1,
                min_words=10,  # Require 10 words
                score_threshold=0.4
            )
            
            # Should have no hooks (not enough words)
            assert len(hooks) == 0
    
    def test_hook_contains_window_text(self, config, segments):
        """Test that detected hook contains the window text."""
        with patch('pipeline.hook_detector._call_llm_for_hook') as mock_llm:
            mock_llm.return_value = (0.8, "reveal")
            
            hooks = detect_hooks(
                config,
                segments,
                window_size=2,
                stride=5,  # Only one window
                min_words=1,
                score_threshold=0.4
            )
            
            assert len(hooks) == 1
            assert "Hello everyone" in hooks[0].text
            assert "What if I told you" in hooks[0].text


class TestGetHookScoreAtTime:
    def test_time_within_hook_returns_score(self):
        """Test that time within hook range returns the hook score."""
        hooks = [
            Hook(start_time=2.0, end_time=6.0, hook_score=0.8, hook_type="question", text="test"),
        ]
        
        score = get_hook_score_at_time(hooks, 4.0)
        
        assert score == 0.8
    
    def test_time_outside_hook_returns_zero(self):
        """Test that time outside hook range returns 0.0."""
        hooks = [
            Hook(start_time=2.0, end_time=6.0, hook_score=0.8, hook_type="question", text="test"),
        ]
        
        score = get_hook_score_at_time(hooks, 10.0)
        
        assert score == 0.0
    
    def test_multiple_hooks_returns_max(self):
        """Test that multiple overlapping hooks return the maximum score."""
        hooks = [
            Hook(start_time=2.0, end_time=6.0, hook_score=0.6, hook_type="question", text="test1"),
            Hook(start_time=4.0, end_time=8.0, hook_score=0.9, hook_type="reveal", text="test2"),
        ]
        
        score = get_hook_score_at_time(hooks, 5.0)
        
        assert score == 0.9  # Max of 0.6 and 0.9
    
    def test_empty_hooks_returns_zero(self):
        """Test that empty hooks list returns 0.0."""
        score = get_hook_score_at_time([], 5.0)
        
        assert score == 0.0


class TestGetHookScoreForWindow:
    def test_window_overlaps_hook_returns_score(self):
        """Test that overlapping window returns hook score."""
        hooks = [
            Hook(start_time=2.0, end_time=6.0, hook_score=0.8, hook_type="question", text="test"),
        ]
        
        score = get_hook_score_for_window(hooks, 4.0, 8.0)
        
        assert score == 0.8
    
    def test_window_contains_hook_returns_score(self):
        """Test that window containing hook returns score."""
        hooks = [
            Hook(start_time=3.0, end_time=5.0, hook_score=0.7, hook_type="emotional", text="test"),
        ]
        
        score = get_hook_score_for_window(hooks, 2.0, 6.0)
        
        assert score == 0.7
    
    def test_hook_contains_window_returns_score(self):
        """Test that hook containing window returns score."""
        hooks = [
            Hook(start_time=2.0, end_time=8.0, hook_score=0.9, hook_type="reveal", text="test"),
        ]
        
        score = get_hook_score_for_window(hooks, 4.0, 6.0)
        
        assert score == 0.9
    
    def test_no_overlap_returns_zero(self):
        """Test that non-overlapping window returns 0.0."""
        hooks = [
            Hook(start_time=2.0, end_time=4.0, hook_score=0.8, hook_type="question", text="test"),
        ]
        
        score = get_hook_score_for_window(hooks, 6.0, 8.0)
        
        assert score == 0.0
    
    def test_multiple_hooks_returns_max(self):
        """Test that multiple overlapping hooks return maximum score."""
        hooks = [
            Hook(start_time=2.0, end_time=5.0, hook_score=0.6, hook_type="question", text="test1"),
            Hook(start_time=4.0, end_time=7.0, hook_score=0.9, hook_type="reveal", text="test2"),
            Hook(start_time=6.0, end_time=9.0, hook_score=0.7, hook_type="emotional", text="test3"),
        ]
        
        score = get_hook_score_for_window(hooks, 3.0, 8.0)
        
        assert score == 0.9  # Max of all overlapping hooks
    
    def test_empty_hooks_returns_zero(self):
        """Test that empty hooks list returns 0.0."""
        score = get_hook_score_for_window([], 2.0, 6.0)
        
        assert score == 0.0
