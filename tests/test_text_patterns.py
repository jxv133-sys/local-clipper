"""Tests for text pattern matching utilities."""

import pytest

from pipeline.text_patterns import (
    analyze_text_patterns,
    detect_question,
    detect_repetition,
    detect_laughter,
    detect_hinglish,
    detect_story_phrase,
    detect_emotional_words,
)


class TestDetectQuestion:
    def test_question_mark_detected(self):
        is_q, score = detect_question("What is this?")
        assert is_q is True
        assert score == 0.3
    
    def test_no_question_mark(self):
        is_q, score = detect_question("This is a statement")
        assert is_q is False
        assert score == 0.0
    
    def test_multiple_questions(self):
        is_q, score = detect_question("What? Why? How?")
        assert is_q is True
        assert score == 0.3


class TestDetectRepetition:
    def test_immediate_repetition(self):
        has_rep, score = detect_repetition("wait wait wait")
        assert has_rep is True
        assert score == 0.2
    
    def test_no_repetition(self):
        has_rep, score = detect_repetition("this is unique text")
        assert has_rep is False
        assert score == 0.0
    
    def test_repetition_in_window(self):
        has_rep, score = detect_repetition("I said no I said yes")
        assert has_rep is True
        assert score == 0.2
    
    def test_short_words_ignored(self):
        # Short words like "a", "is" shouldn't trigger repetition
        has_rep, score = detect_repetition("a a a")
        assert has_rep is False
        assert score == 0.0
    
    def test_too_short_text(self):
        has_rep, score = detect_repetition("hi")
        assert has_rep is False
        assert score == 0.0


class TestDetectLaughter:
    def test_laughter_marker(self):
        has_laugh, score = detect_laughter("That was funny (laughter)")
        assert has_laugh is True
        assert score == 0.5
    
    def test_haha(self):
        has_laugh, score = detect_laughter("haha that's great")
        assert has_laugh is True
        assert score == 0.5
    
    def test_lol(self):
        has_laugh, score = detect_laughter("lol nice")
        assert has_laugh is True
        assert score == 0.5
    
    def test_no_laughter(self):
        has_laugh, score = detect_laughter("This is serious")
        assert has_laugh is False
        assert score == 0.0
    
    def test_case_insensitive(self):
        has_laugh, score = detect_laughter("HAHA LMAO")
        assert has_laugh is True
        assert score == 0.5


class TestDetectHinglish:
    def test_kya_detected(self):
        has_hindi, score = detect_hinglish("kya bhai what happened")
        assert has_hindi is True
        assert score == 0.25
    
    def test_yaar_detected(self):
        has_hindi, score = detect_hinglish("come on yaar")
        assert has_hindi is True
        assert score == 0.25
    
    def test_no_hinglish(self):
        has_hindi, score = detect_hinglish("This is pure English")
        assert has_hindi is False
        assert score == 0.0
    
    def test_case_insensitive(self):
        has_hindi, score = detect_hinglish("KYA is happening")
        assert has_hindi is True
        assert score == 0.25


class TestDetectStoryPhrase:
    def test_so_basically(self):
        has_story, score = detect_story_phrase("So basically what happened was")
        assert has_story is True
        assert score == 0.2
    
    def test_let_me_tell_you(self):
        has_story, score = detect_story_phrase("Let me tell you about this")
        assert has_story is True
        assert score == 0.2
    
    def test_one_time(self):
        has_story, score = detect_story_phrase("One time I saw something crazy")
        assert has_story is True
        assert score == 0.2
    
    def test_no_story_phrase(self):
        has_story, score = detect_story_phrase("Just regular commentary")
        assert has_story is False
        assert score == 0.0
    
    def test_case_insensitive(self):
        has_story, score = detect_story_phrase("SO BASICALLY this happened")
        assert has_story is True
        assert score == 0.2


class TestDetectEmotionalWords:
    def test_love(self):
        has_emotion, score = detect_emotional_words("I love this game")
        assert has_emotion is True
        assert score == 0.15
    
    def test_hate(self):
        has_emotion, score = detect_emotional_words("I hate this level")
        assert has_emotion is True
        assert score == 0.15
    
    def test_amazing(self):
        has_emotion, score = detect_emotional_words("That was amazing")
        assert has_emotion is True
        assert score == 0.15
    
    def test_no_emotional_words(self):
        has_emotion, score = detect_emotional_words("This is neutral text")
        assert has_emotion is False
        assert score == 0.0
    
    def test_case_insensitive(self):
        has_emotion, score = detect_emotional_words("AMAZING play")
        assert has_emotion is True
        assert score == 0.15


class TestAnalyzeTextPatterns:
    def test_multiple_signals(self):
        text = "What? That was amazing! haha"
        result = analyze_text_patterns(text)
        
        assert "Question" in result.signals
        assert "Emotional" in result.signals
        assert "Laughter" in result.signals
        assert result.score > 0.0
    
    def test_no_signals(self):
        text = "Just regular commentary here"
        result = analyze_text_patterns(text)
        
        assert len(result.signals) == 0
        assert result.score == 0.0
    
    def test_score_capped_at_one(self):
        # Text with many signals that would exceed 1.0
        text = "What? So basically I love this haha wait wait kya bhai amazing"
        result = analyze_text_patterns(text)
        
        assert result.score <= 1.0
    
    def test_hinglish_gaming_content(self):
        text = "kya yaar what is this?"
        result = analyze_text_patterns(text)
        
        assert "Question" in result.signals
        assert "Question (HI)" in result.signals
        assert result.score > 0.0
    
    def test_story_with_emotion(self):
        text = "So basically one time I saw something amazing"
        result = analyze_text_patterns(text)
        
        assert "Story (HI)" in result.signals
        assert "Emotional" in result.signals
        assert result.score > 0.0
    
    def test_repetition_with_laughter(self):
        text = "wait wait wait haha"
        result = analyze_text_patterns(text)
        
        assert "Repetition" in result.signals
        assert "Laughter" in result.signals
        assert result.score == 0.7  # 0.2 + 0.5
