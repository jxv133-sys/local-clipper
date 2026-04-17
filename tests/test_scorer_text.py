"""Tests for pipeline/scorer.py — text scoring and score combination logic.

Covers:
- Property 2: Text score determinism (subtask 4.1)
- Property 3: Text score normalized (subtask 4.2)
- Property 4: Text score monotonicity (subtask 4.3)
- Property 6: Clip score equals weighted sum (subtask 4.4)
- Property 7: Clip score monotonicity with text score (subtask 4.5)
- Unit tests for text scoring and combine_scores (subtask 4.6)
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from config import Config
from pipeline.models import Segment
from pipeline.scorer import combine_scores, compute_text_score

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = Config(work_dir="/tmp/test")


def make_config(
    text_weight: float = 0.4,
    audio_weight: float = 0.6,
    llm_weight: float = 0.0,
    keywords: list[str] | None = None,
) -> Config:
    cfg = Config(work_dir="/tmp/test")
    cfg.text_weight = text_weight
    cfg.audio_weight = audio_weight
    cfg.llm_weight = llm_weight
    if keywords is not None:
        cfg.keywords = keywords
    return cfg


def make_segment(text: str, start: float = 0.0, end: float = 1.0) -> Segment:
    return Segment(start=start, end=end, text=text)


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------

# Feature: video-highlight-generator, Property 2: Text score determinism
# Validates: Requirements 3.6
@given(
    seg=st.builds(
        Segment,
        text=st.text(),
        start=st.just(0.0),
        end=st.just(1.0),
    )
)
@settings(max_examples=100)
def test_text_score_determinism(seg: Segment) -> None:
    """Property 2: Text score determinism — same segment always returns same score.

    **Validates: Requirements 3.6**
    """
    score1 = compute_text_score(DEFAULT_CONFIG, seg)
    score2 = compute_text_score(DEFAULT_CONFIG, seg)
    assert score1 == score2


# Feature: video-highlight-generator, Property 3: Text score is normalized
# Validates: Requirements 3.5
@given(
    seg=st.builds(
        Segment,
        text=st.text(),
        start=st.just(0.0),
        end=st.just(1.0),
    )
)
@settings(max_examples=100)
def test_text_score_normalized(seg: Segment) -> None:
    """Property 3: Text score is normalized — always in [0.0, 1.0].

    **Validates: Requirements 3.5**
    """
    score = compute_text_score(DEFAULT_CONFIG, seg)
    assert 0.0 <= score <= 1.0


# Feature: video-highlight-generator, Property 4: Text score monotonicity
# Validates: Requirements 3.2, 3.3, 3.4
@given(
    base_text=st.text(),
    extra_chars=st.text(
        alphabet=st.characters(blacklist_categories=("Zs", "Cc", "Cs")),
        min_size=1,
    ),
)
@settings(max_examples=100)
def test_text_score_monotonicity(base_text: str, extra_chars: str) -> None:
    """Property 4: Text score monotonicity — enriching text never decreases score.

    Appending a keyword, '!', '?', or extra non-whitespace characters to a
    base text must produce a score >= the base score.

    **Validates: Requirements 3.2, 3.3, 3.4**
    """
    config = make_config(keywords=["crazy", "important"])
    base_seg = make_segment(base_text)
    base_score = compute_text_score(config, base_seg)

    # Append a keyword
    keyword_seg = make_segment(base_text + " crazy")
    assert compute_text_score(config, keyword_seg) >= base_score

    # Append '!'
    exclaim_seg = make_segment(base_text + "!")
    assert compute_text_score(config, exclaim_seg) >= base_score

    # Append '?'
    question_seg = make_segment(base_text + "?")
    assert compute_text_score(config, question_seg) >= base_score

    # Append extra non-whitespace characters
    extra_seg = make_segment(base_text + extra_chars)
    assert compute_text_score(config, extra_seg) >= base_score


# Feature: video-highlight-generator, Property 6: Clip score equals weighted sum
# Validates: Requirements 6.1, 6.4
@given(
    text_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    audio_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    llm_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    text_weight=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    audio_weight=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    llm_weight=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_clip_score_weighted_sum(
    text_score: float,
    audio_score: float,
    llm_score: float,
    text_weight: float,
    audio_weight: float,
    llm_weight: float,
) -> None:
    """Property 6: Clip score equals weighted sum.

    combine_scores returns exactly text_w * text + audio_w * audio + llm_w * llm.

    **Validates: Requirements 6.1, 6.4**
    """
    config = make_config(
        text_weight=text_weight,
        audio_weight=audio_weight,
        llm_weight=llm_weight,
    )
    result = combine_scores(config, text_score, audio_score, llm_score)
    expected = text_weight * text_score + audio_weight * audio_score + llm_weight * llm_score
    assert math.isclose(result, max(0.0, expected), rel_tol=1e-9, abs_tol=1e-12)


# Feature: video-highlight-generator, Property 7: Clip score monotonicity with text score
# Validates: Requirements 6.5
@given(
    text_a=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    text_b=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    audio=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    llm=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    text_weight=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    audio_weight=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    llm_weight=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_clip_score_monotone_text(
    text_a: float,
    text_b: float,
    audio: float,
    llm: float,
    text_weight: float,
    audio_weight: float,
    llm_weight: float,
) -> None:
    """Property 7: Clip score monotonicity with text score.

    If text_a >= text_b and audio/llm are equal, combine_scores(text_a) >= combine_scores(text_b).

    **Validates: Requirements 6.5**
    """
    # Ensure text_a >= text_b
    high_text, low_text = max(text_a, text_b), min(text_a, text_b)

    config = make_config(
        text_weight=text_weight,
        audio_weight=audio_weight,
        llm_weight=llm_weight,
    )
    score_high = combine_scores(config, high_text, audio, llm)
    score_low = combine_scores(config, low_text, audio, llm)
    assert score_high >= score_low


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestComputeTextScore:
    """Unit tests for compute_text_score."""

    def test_keyword_increases_score(self) -> None:
        """Segment with a known keyword scores higher than one without."""
        config = make_config(keywords=["crazy"])
        without = make_segment("This is a normal sentence.")
        with_kw = make_segment("This is crazy stuff.")
        assert compute_text_score(config, with_kw) > compute_text_score(config, without)

    def test_exclamation_increases_score(self) -> None:
        """Segment with '!' scores higher than the same text without it."""
        config = DEFAULT_CONFIG
        base = make_segment("Wow that is amazing")
        excited = make_segment("Wow that is amazing!")
        assert compute_text_score(config, excited) > compute_text_score(config, base)

    def test_question_mark_increases_score(self) -> None:
        """Segment with '?' scores higher than the same text without it."""
        config = DEFAULT_CONFIG
        base = make_segment("Did you see that")
        questioned = make_segment("Did you see that?")
        assert compute_text_score(config, questioned) > compute_text_score(config, base)

    def test_longer_text_increases_score(self) -> None:
        """Longer segment text scores higher than shorter text."""
        config = DEFAULT_CONFIG
        short = make_segment("Hi.")
        long_ = make_segment("Hi. " + "a" * 100)
        assert compute_text_score(config, long_) > compute_text_score(config, short)

    def test_empty_text_returns_zero(self) -> None:
        """Empty text should produce a score of 0.0."""
        config = DEFAULT_CONFIG
        seg = make_segment("")
        assert compute_text_score(config, seg) == 0.0

    def test_score_in_range(self) -> None:
        """Score is always in [0.0, 1.0]."""
        config = make_config(keywords=["crazy", "important"])
        texts = [
            "",
            "hello",
            "crazy crazy crazy crazy crazy crazy crazy crazy crazy crazy",
            "!" * 100,
            "?" * 100,
            "a" * 1000,
        ]
        for text in texts:
            score = compute_text_score(config, make_segment(text))
            assert 0.0 <= score <= 1.0, f"Score {score} out of range for text: {text!r}"

    def test_keyword_case_insensitive(self) -> None:
        """Keywords are matched case-insensitively."""
        config = make_config(keywords=["crazy"])
        lower = make_segment("that was crazy")
        upper = make_segment("that was CRAZY")
        mixed = make_segment("that was CrAzY")
        assert compute_text_score(config, lower) == compute_text_score(config, upper)
        assert compute_text_score(config, lower) == compute_text_score(config, mixed)

    def test_multiple_keyword_occurrences(self) -> None:
        """Multiple keyword occurrences each add to the score."""
        config = make_config(keywords=["wow"])
        one = make_segment("wow that was great")
        two = make_segment("wow wow that was great")
        assert compute_text_score(config, two) > compute_text_score(config, one)


class TestCombineScores:
    """Unit tests for combine_scores."""

    def test_known_weights_and_scores(self) -> None:
        """combine_scores returns the correct weighted sum for known inputs."""
        config = make_config(text_weight=0.4, audio_weight=0.6, llm_weight=0.0)
        result = combine_scores(config, text=0.5, audio=0.8, llm=None)
        expected = 0.4 * 0.5 + 0.6 * 0.8 + 0.0 * 0.0
        assert math.isclose(result, expected, rel_tol=1e-9)

    def test_llm_none_with_zero_llm_weight(self) -> None:
        """combine_scores with llm=None and llm_weight=0.0 equals text+audio only."""
        config = make_config(text_weight=0.4, audio_weight=0.6, llm_weight=0.0)
        result_none = combine_scores(config, text=0.5, audio=0.8, llm=None)
        result_zero = combine_scores(config, text=0.5, audio=0.8, llm=0.0)
        assert math.isclose(result_none, result_zero, rel_tol=1e-9)

    def test_llm_none_treated_as_zero(self) -> None:
        """llm=None is treated as 0.0 in the weighted sum."""
        config = make_config(text_weight=0.3, audio_weight=0.4, llm_weight=0.3)
        result_none = combine_scores(config, text=0.5, audio=0.5, llm=None)
        result_zero = combine_scores(config, text=0.5, audio=0.5, llm=0.0)
        assert math.isclose(result_none, result_zero, rel_tol=1e-9)

    def test_result_non_negative(self) -> None:
        """combine_scores result is always >= 0.0."""
        config = make_config(text_weight=0.4, audio_weight=0.6, llm_weight=0.0)
        result = combine_scores(config, text=0.0, audio=0.0, llm=None)
        assert result >= 0.0

    def test_all_weights_contribute(self) -> None:
        """All three weights contribute to the final score when non-zero."""
        config = make_config(text_weight=0.3, audio_weight=0.4, llm_weight=0.3)
        result = combine_scores(config, text=1.0, audio=1.0, llm=1.0)
        expected = 0.3 * 1.0 + 0.4 * 1.0 + 0.3 * 1.0
        assert math.isclose(result, expected, rel_tol=1e-9)

    def test_zero_text_weight_ignores_text(self) -> None:
        """When text_weight=0.0, text score does not affect the result."""
        config = make_config(text_weight=0.0, audio_weight=1.0, llm_weight=0.0)
        result_low = combine_scores(config, text=0.0, audio=0.5, llm=None)
        result_high = combine_scores(config, text=1.0, audio=0.5, llm=None)
        assert math.isclose(result_low, result_high, rel_tol=1e-9)
