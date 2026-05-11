"""Tests for phrase detection integration in scorer.py (Task 2.4).

Validates Requirements 4.3, 4.5, 19.2.
"""

from __future__ import annotations

import pytest

from config import Config
from pipeline.models import Segment
from pipeline.scorer import compute_text_score


def make_config(
    phrase_keywords: list[str] | None = None,
    phrase_weight: float = 4.0,
) -> Config:
    """Create a test config with phrase detection settings."""
    cfg = Config(work_dir="/tmp/test")
    if phrase_keywords is not None:
        cfg.phrase_keywords = phrase_keywords
    cfg.phrase_weight = phrase_weight
    # Disable pattern blending for cleaner test assertions
    cfg.text_pattern_weight = 0.0
    return cfg


def make_segment(text: str, start: float = 0.0, end: float = 1.0) -> Segment:
    """Create a test segment."""
    return Segment(start=start, end=end, text=text)


class TestPhraseDetectionIntegration:
    """Tests for phrase detection integration in compute_text_score."""

    def test_phrase_match_increases_score(self) -> None:
        """Segment with a phrase keyword scores higher than one without."""
        config = make_config(phrase_keywords=["oh my god"])
        without = make_segment("that was amazing")
        with_phrase = make_segment("oh my god that was amazing")
        assert compute_text_score(config, with_phrase) > compute_text_score(config, without)

    def test_phrase_weight_higher_than_single_keyword(self) -> None:
        """Phrase match scores higher than individual keyword matches."""
        config = make_config(phrase_keywords=["oh my god"], phrase_weight=4.0)
        # Clear regular keywords to isolate phrase scoring
        config.keywords = []
        
        # Segment with phrase
        phrase_seg = make_segment("oh my god")
        phrase_score = compute_text_score(config, phrase_seg)
        
        # Segment with individual words (not as phrase)
        # Since we cleared keywords, this should score lower
        individual_seg = make_segment("oh my")
        individual_score = compute_text_score(config, individual_seg)
        
        assert phrase_score > individual_score

    def test_phrase_case_insensitive(self) -> None:
        """Phrase matching is case-insensitive."""
        config = make_config(phrase_keywords=["oh my god"])
        lower = make_segment("oh my god that was crazy")
        upper = make_segment("OH MY GOD that was crazy")
        mixed = make_segment("Oh My God that was crazy")
        
        assert compute_text_score(config, lower) == compute_text_score(config, upper)
        assert compute_text_score(config, lower) == compute_text_score(config, mixed)

    def test_phrase_word_boundaries(self) -> None:
        """Phrase matching respects word boundaries."""
        config = make_config(phrase_keywords=["no way"])
        # "no way" as separate words should match
        match_seg = make_segment("no way that happened")
        # "noway" as one word should NOT match
        no_match_seg = make_segment("noway that happened")
        
        assert compute_text_score(config, match_seg) > compute_text_score(config, no_match_seg)

    def test_multiple_phrase_matches(self) -> None:
        """Multiple phrase matches each add to the score."""
        config = make_config(phrase_keywords=["oh my god"], phrase_weight=4.0)
        one = make_segment("oh my god that was great")
        two = make_segment("oh my god oh my god that was great")
        
        assert compute_text_score(config, two) > compute_text_score(config, one)

    def test_phrase_with_punctuation(self) -> None:
        """Phrase matching works with punctuation."""
        config = make_config(phrase_keywords=["oh my god"])
        with_punct = make_segment("Oh my god! That was amazing!")
        without_punct = make_segment("That was amazing")
        
        assert compute_text_score(config, with_punct) > compute_text_score(config, without_punct)

    def test_default_phrase_keywords(self) -> None:
        """Config has default phrase keywords."""
        config = Config(work_dir="/tmp/test")
        assert hasattr(config, "phrase_keywords")
        assert isinstance(config.phrase_keywords, list)
        assert len(config.phrase_keywords) > 0
        # Check for some expected defaults
        expected = {"oh my god", "no way", "watch this"}
        assert expected.issubset(set(config.phrase_keywords))

    def test_default_phrase_weight(self) -> None:
        """Config has default phrase_weight of 4.0."""
        config = Config(work_dir="/tmp/test")
        assert hasattr(config, "phrase_weight")
        assert config.phrase_weight == 4.0

    def test_phrase_score_normalized(self) -> None:
        """Text score with many phrase matches remains in [0.0, 1.0]."""
        config = make_config(phrase_keywords=["oh my god", "no way"], phrase_weight=4.0)
        # Pile on many phrase matches
        text = "oh my god no way " * 20
        seg = make_segment(text)
        score = compute_text_score(config, seg)
        assert 0.0 <= score <= 1.0

    def test_empty_phrase_keywords_list(self) -> None:
        """Empty phrase_keywords list doesn't crash."""
        config = make_config(phrase_keywords=[])
        seg = make_segment("oh my god that was amazing")
        score = compute_text_score(config, seg)
        assert 0.0 <= score <= 1.0

    def test_phrase_detection_with_other_scoring_components(self) -> None:
        """Phrase detection works alongside other scoring components."""
        config = make_config(phrase_keywords=["oh my god"], phrase_weight=4.0)
        config.keywords = ["crazy"]
        config.reaction_keywords = ["wow"]
        
        # Segment with phrase, keyword, reaction, and punctuation
        rich_seg = make_segment("oh my god wow that was crazy!")
        # Segment with just base text
        base_seg = make_segment("that was something")
        
        assert compute_text_score(config, rich_seg) > compute_text_score(config, base_seg)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
