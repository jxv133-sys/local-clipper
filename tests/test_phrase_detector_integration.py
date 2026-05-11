"""Integration tests for phrase detector with realistic scenarios.

Tests the phrase detector with realistic text examples that would be
encountered in video transcripts.

**Validates: Requirements 4.1, 4.2, 4.4, 4.6**
"""

import pytest

from pipeline.phrase_detector import detect_phrases


class TestRealisticScenarios:
    """Test phrase detection with realistic video transcript scenarios."""
    
    def test_gaming_commentary(self):
        """Test with typical gaming commentary phrases."""
        text = "Oh my god, watch this play! No way, that was insane!"
        phrases = ["oh my god", "watch this", "no way"]
        result = detect_phrases(text, phrases)
        
        assert len(result) == 3
        assert result[0] == ("oh my god", 0, 9)
        assert result[1] == ("watch this", 11, 21)
        assert result[2] == ("no way", 28, 34)
    
    def test_reaction_video(self):
        """Test with reaction video phrases."""
        text = "Are you kidding me? I can't believe what I just saw!"
        phrases = ["are you kidding", "i can't believe"]
        result = detect_phrases(text, phrases)
        
        assert len(result) == 2
        assert result[0] == ("are you kidding", 0, 15)
        assert result[1] == ("i can't believe", 20, 35)
    
    def test_podcast_transcript(self):
        """Test with podcast-style transcript."""
        text = "Look at this data. What the hell is going on here?"
        phrases = ["look at this", "what the hell"]
        result = detect_phrases(text, phrases)
        
        assert len(result) == 2
        assert result[0] == ("look at this", 0, 12)
        assert result[1] == ("what the hell", 19, 32)
    
    def test_multiple_occurrences_in_conversation(self):
        """Test with phrases occurring multiple times in conversation."""
        text = "Oh my god, did you see that? Oh my god, I can't believe it!"
        phrases = ["oh my god", "i can't believe"]
        result = detect_phrases(text, phrases)
        
        assert len(result) == 3
        # Two "oh my god" matches
        assert result[0] == ("oh my god", 0, 9)
        assert result[1] == ("oh my god", 29, 38)
        # One "i can't believe" match
        assert result[2] == ("i can't believe", 40, 55)
    
    def test_mixed_case_natural_speech(self):
        """Test with natural speech patterns and mixed case."""
        text = "NO WAY! Watch This amazing moment. Oh My God!"
        phrases = ["no way", "watch this", "oh my god"]
        result = detect_phrases(text, phrases)
        
        assert len(result) == 3
        assert result[0][0] == "no way"
        assert result[1][0] == "watch this"
        assert result[2][0] == "oh my god"
    
    def test_phrase_with_surrounding_noise(self):
        """Test phrases surrounded by filler words and noise."""
        text = "Um, like, oh my god, you know, that's crazy, no way!"
        phrases = ["oh my god", "no way"]
        result = detect_phrases(text, phrases)
        
        assert len(result) == 2
        assert result[0] == ("oh my god", 10, 19)
        assert result[1] == ("no way", 45, 51)
    
    def test_no_false_positives_similar_words(self):
        """Test that similar but different phrases don't match."""
        text = "Oh my goodness, that's amazing!"
        phrases = ["oh my god"]  # Should NOT match "oh my goodness"
        result = detect_phrases(text, phrases)
        
        assert len(result) == 0
    
    def test_phrase_in_longer_sentence(self):
        """Test phrase detection in longer, complex sentences."""
        text = ("I was just sitting there, and then, oh my god, "
                "the most incredible thing happened that I can't believe!")
        phrases = ["oh my god", "i can't believe"]
        result = detect_phrases(text, phrases)
        
        assert len(result) == 2
        assert result[0][0] == "oh my god"
        assert result[1][0] == "i can't believe"
    
    def test_default_phrase_list(self):
        """Test with the default phrase list from the design document."""
        default_phrases = [
            "oh my god", "no way", "watch this", "look at this",
            "are you kidding", "i can't believe", "what the hell"
        ]
        
        text = "Oh my god, no way! Watch this, are you kidding me?"
        result = detect_phrases(text, default_phrases)
        
        # Should find 4 phrases
        assert len(result) == 4
        assert any(match[0] == "oh my god" for match in result)
        assert any(match[0] == "no way" for match in result)
        assert any(match[0] == "watch this" for match in result)
        assert any(match[0] == "are you kidding" for match in result)


class TestPerformance:
    """Test performance characteristics of phrase detection."""
    
    def test_large_text_performance(self):
        """Test that phrase detection performs well on large text."""
        # Create a large text (simulating a long video transcript)
        large_text = " ".join(["word"] * 10000) + " oh my god " + " ".join(["word"] * 10000)
        phrases = ["oh my god"]
        
        # Should complete quickly and find the phrase
        result = detect_phrases(large_text, phrases)
        assert len(result) == 1
        assert result[0][0] == "oh my god"
    
    def test_many_phrases_performance(self):
        """Test performance with many phrases to search for."""
        text = "oh my god, this is amazing!"
        # Create a large list of phrases
        many_phrases = [f"phrase {i}" for i in range(1000)] + ["oh my god"]
        
        # Should complete quickly and find the matching phrase
        result = detect_phrases(text, many_phrases)
        assert len(result) == 1
        assert result[0][0] == "oh my god"
