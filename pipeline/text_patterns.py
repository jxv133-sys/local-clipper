"""Text pattern matching utilities for semantic analysis.

Provides heuristic-based text scoring patterns including:
- Question detection
- Word repetition
- Laughter markers
- Hinglish keywords
- Story phrases
- Emotional words
"""

import re
from dataclasses import dataclass


@dataclass
class TextSignals:
    """Detected text signals and their contributions to the score."""
    signals: list[str]  # Names of detected signals
    score: float        # Total score contribution (0.0-1.0)


# Hinglish question words (common in Indian English gaming/streaming)
HINGLISH_QUESTIONS = [
    "kya", "kaise", "kahan", "kab", "kyun", "kaun",
    "yaar", "bhai", "arre", "arey",
]

# Story/narrative phrases that indicate engaging content
STORY_PHRASES = [
    "so basically", "let me tell you", "here's what happened",
    "the thing is", "you know what", "i remember when",
    "one time", "this one time", "back when", "the other day",
]

# Emotional words that indicate strong reactions
EMOTIONAL_WORDS = [
    "love", "hate", "amazing", "terrible", "awesome", "awful",
    "fantastic", "horrible", "brilliant", "disgusting",
    "beautiful", "ugly", "perfect", "disaster",
]

# Laughter markers (Whisper sometimes transcribes these)
LAUGHTER_MARKERS = [
    "(laughter)", "(laughing)", "[laughter]", "[laughing]",
    "haha", "hahaha", "lol", "lmao",
]


def detect_question(text: str) -> tuple[bool, float]:
    """Detect if text contains a question.
    
    Returns:
        (is_question, score_boost)
    """
    if "?" in text:
        return True, 0.3
    return False, 0.0


def detect_repetition(text: str) -> tuple[bool, float]:
    """Detect word repetition patterns.
    
    Looks for the same word appearing multiple times in close proximity.
    
    Returns:
        (has_repetition, score_boost)
    """
    words = text.lower().split()
    if len(words) < 3:
        return False, 0.0
    
    # Check for immediate repetition (word appears 2+ times consecutively)
    for i in range(len(words) - 1):
        if words[i] == words[i + 1] and len(words[i]) > 2:  # Ignore short words like "a", "is"
            return True, 0.2
    
    # Check for repetition within a 5-word window
    for i in range(len(words) - 4):
        window = words[i:i+5]
        if len(set(window)) < len(window) - 1:  # At least 2 repeated words
            return True, 0.2
    
    return False, 0.0


def detect_laughter(text: str) -> tuple[bool, float]:
    """Detect laughter markers in text.
    
    Returns:
        (has_laughter, score_boost)
    """
    text_lower = text.lower()
    for marker in LAUGHTER_MARKERS:
        if marker.lower() in text_lower:
            return True, 0.5
    return False, 0.0


def detect_hinglish(text: str) -> tuple[bool, float]:
    """Detect Hinglish keywords (common in Indian English content).
    
    Returns:
        (has_hinglish, score_boost)
    """
    text_lower = text.lower()
    words = text_lower.split()
    
    for keyword in HINGLISH_QUESTIONS:
        if keyword in words:
            return True, 0.25
    
    return False, 0.0


def detect_story_phrase(text: str) -> tuple[bool, float]:
    """Detect story/narrative phrases that indicate engaging content.
    
    Returns:
        (has_story_phrase, score_boost)
    """
    text_lower = text.lower()
    
    for phrase in STORY_PHRASES:
        if phrase in text_lower:
            return True, 0.2
    
    return False, 0.0


def detect_emotional_words(text: str) -> tuple[bool, float]:
    """Detect emotional words that indicate strong reactions.
    
    Returns:
        (has_emotional_words, score_boost)
    """
    text_lower = text.lower()
    # Remove punctuation for word matching
    import string
    text_clean = text_lower.translate(str.maketrans('', '', string.punctuation))
    words = text_clean.split()
    
    for word in EMOTIONAL_WORDS:
        if word in words:
            return True, 0.15
    
    return False, 0.0


def analyze_text_patterns(text: str) -> TextSignals:
    """Analyze text for all heuristic patterns and return detected signals.
    
    Args:
        text: Text segment to analyze
        
    Returns:
        TextSignals with list of detected signal names and total score
    """
    signals = []
    score = 0.0
    
    # Check all patterns
    is_question, q_score = detect_question(text)
    if is_question:
        signals.append("Question")
        score += q_score
    
    has_repetition, r_score = detect_repetition(text)
    if has_repetition:
        signals.append("Repetition")
        score += r_score
    
    has_laughter, l_score = detect_laughter(text)
    if has_laughter:
        signals.append("Laughter")
        score += l_score
    
    has_hinglish, h_score = detect_hinglish(text)
    if has_hinglish:
        signals.append("Question (HI)")
        score += h_score
    
    has_story, s_score = detect_story_phrase(text)
    if has_story:
        signals.append("Story (HI)")
        score += s_score
    
    has_emotional, e_score = detect_emotional_words(text)
    if has_emotional:
        signals.append("Emotional")
        score += e_score
    
    # Cap at 1.0
    score = min(score, 1.0)
    
    return TextSignals(signals=signals, score=score)
