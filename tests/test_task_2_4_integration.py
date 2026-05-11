"""Integration test for Task 2.4: Phrase detection in scorer.py.

This test verifies that the phrase detection integration works correctly
in the context of the full scoring pipeline.

Validates Requirements 4.3, 4.5, 19.2.
"""

from __future__ import annotations

import logging

import pytest

from config import Config
from pipeline.models import Segment
from pipeline.scorer import compute_text_score


def test_phrase_detection_integration_end_to_end() -> None:
    """End-to-end test: phrase detection increases text score in compute_text_score."""
    # Setup config with phrase keywords
    config = Config(work_dir="/tmp/test")
    config.phrase_keywords = ["oh my god", "no way", "watch this"]
    config.phrase_weight = 4.0
    config.text_pattern_weight = 0.0  # Disable pattern blending for cleaner test
    
    # Test segment with phrase
    segment_with_phrase = Segment(
        start=10.0,
        end=13.0,
        text="Oh my god, that was incredible!"
    )
    
    # Test segment without phrase
    segment_without_phrase = Segment(
        start=20.0,
        end=23.0,
        text="That was incredible!"
    )
    
    # Compute scores
    score_with_phrase = compute_text_score(config, segment_with_phrase)
    score_without_phrase = compute_text_score(config, segment_without_phrase)
    
    # Verify phrase detection increases score
    assert score_with_phrase > score_without_phrase, (
        f"Expected phrase detection to increase score, but got "
        f"with_phrase={score_with_phrase:.4f} <= without_phrase={score_without_phrase:.4f}"
    )
    
    # Verify scores are normalized
    assert 0.0 <= score_with_phrase <= 1.0
    assert 0.0 <= score_without_phrase <= 1.0


def test_phrase_detection_logging(caplog) -> None:
    """Verify phrase detection logs at DEBUG level."""
    # Setup config
    config = Config(work_dir="/tmp/test")
    config.phrase_keywords = ["oh my god"]
    config.phrase_weight = 4.0
    config.text_pattern_weight = 0.0
    
    # Test segment with phrase
    segment = Segment(start=10.0, end=13.0, text="Oh my god!")
    
    # Capture logs at DEBUG level
    with caplog.at_level(logging.DEBUG):
        compute_text_score(config, segment)
    
    # Verify DEBUG log was created
    debug_logs = [record for record in caplog.records if record.levelname == "DEBUG"]
    assert len(debug_logs) > 0, "Expected DEBUG log for phrase detection"
    
    # Verify log contains phrase information
    log_messages = [record.message for record in debug_logs]
    phrase_logs = [msg for msg in log_messages if "Phrase detected" in msg and "oh my god" in msg]
    assert len(phrase_logs) > 0, f"Expected phrase detection log, got: {log_messages}"


def test_multiple_phrases_cumulative_score() -> None:
    """Multiple phrase matches should cumulatively increase the score."""
    config = Config(work_dir="/tmp/test")
    config.phrase_keywords = ["oh my god", "no way"]
    config.phrase_weight = 4.0
    config.text_pattern_weight = 0.0
    
    # Segment with one phrase
    one_phrase = Segment(start=0.0, end=3.0, text="Oh my god!")
    
    # Segment with two phrases
    two_phrases = Segment(start=0.0, end=5.0, text="Oh my god, no way!")
    
    score_one = compute_text_score(config, one_phrase)
    score_two = compute_text_score(config, two_phrases)
    
    assert score_two > score_one, (
        f"Expected two phrases to score higher than one, but got "
        f"two={score_two:.4f} <= one={score_one:.4f}"
    )


def test_phrase_weight_configuration() -> None:
    """Verify phrase_weight configuration affects scoring."""
    # Config with low phrase weight
    config_low = Config(work_dir="/tmp/test")
    config_low.phrase_keywords = ["oh my god"]
    config_low.phrase_weight = 1.0
    config_low.text_pattern_weight = 0.0
    
    # Config with high phrase weight
    config_high = Config(work_dir="/tmp/test")
    config_high.phrase_keywords = ["oh my god"]
    config_high.phrase_weight = 10.0
    config_high.text_pattern_weight = 0.0
    
    segment = Segment(start=0.0, end=3.0, text="Oh my god!")
    
    score_low = compute_text_score(config_low, segment)
    score_high = compute_text_score(config_high, segment)
    
    assert score_high > score_low, (
        f"Expected higher phrase_weight to produce higher score, but got "
        f"high={score_high:.4f} <= low={score_low:.4f}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
