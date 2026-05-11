"""Tests for phrase detection module.

This module tests the phrase detection functionality for multi-word keyword
matching with word boundaries and case-insensitive matching.

**Validates: Requirements 4.1, 4.2, 4.4, 4.6**
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pipeline.phrase_detector import detect_phrases


class TestDetectPhrasesBasic:
    """Basic unit tests for phrase detection."""
    
    def test_single_phrase_match(self):
        """Test matching a single phrase in text."""
        result = detect_phrases("Oh my god, that was amazing!", ["oh my god"])
        assert len(result) == 1
        assert result[0] == ("oh my god", 0, 9)
    
    def test_multiple_phrase_matches(self):
        """Test matching multiple different phrases."""
        text = "Oh my god, no way, watch this!"
        phrases = ["oh my god", "no way", "watch this"]
        result = detect_phrases(text, phrases)
        assert len(result) == 3
        assert result[0] == ("oh my god", 0, 9)
        assert result[1] == ("no way", 11, 17)
        assert result[2] == ("watch this", 19, 29)
    
    def test_case_insensitive_matching(self):
        """Test that matching is case-insensitive."""
        # Test various case combinations
        assert detect_phrases("OH MY GOD", ["oh my god"]) == [("oh my god", 0, 9)]
        assert detect_phrases("Oh My God", ["oh my god"]) == [("oh my god", 0, 9)]
        assert detect_phrases("oh my god", ["OH MY GOD"]) == [("OH MY GOD", 0, 9)]
        assert detect_phrases("oH mY gOd", ["oh my god"]) == [("oh my god", 0, 9)]
    
    def test_word_boundary_enforcement(self):
        """Test that word boundaries are enforced (no partial matches)."""
        # "ohmygod" should NOT match "oh my god"
        result = detect_phrases("ohmygod", ["oh my god"])
        assert len(result) == 0
        
        # "oh my godly" should NOT match "oh my god"
        result = detect_phrases("oh my godly", ["oh my god"])
        assert len(result) == 0
        
        # "myoh my god" should NOT match at the start
        result = detect_phrases("myoh my god", ["oh my god"])
        assert len(result) == 0
    
    def test_word_boundary_with_punctuation(self):
        """Test that punctuation acts as word boundary."""
        # Punctuation should act as word boundaries
        assert detect_phrases("Oh my god!", ["oh my god"]) == [("oh my god", 0, 9)]
        assert detect_phrases("Oh my god.", ["oh my god"]) == [("oh my god", 0, 9)]
        assert detect_phrases("Oh my god?", ["oh my god"]) == [("oh my god", 0, 9)]
        assert detect_phrases("'Oh my god'", ["oh my god"]) == [("oh my god", 1, 10)]
        assert detect_phrases('"Oh my god"', ["oh my god"]) == [("oh my god", 1, 10)]
    
    def test_partial_phrase_no_match(self):
        """Test that partial phrases don't match longer phrases."""
        # "oh my" should NOT match "oh my god"
        result = detect_phrases("oh my", ["oh my god"])
        assert len(result) == 0
        
        # "my god" should NOT match "oh my god"
        result = detect_phrases("my god", ["oh my god"])
        assert len(result) == 0
    
    def test_overlapping_phrases(self):
        """Test handling of overlapping phrase matches."""
        text = "Oh my, oh my god!"
        phrases = ["oh my", "oh my god"]
        result = detect_phrases(text, phrases)
        # Should find both "oh my" at position 0 and 7, and "oh my god" at position 7
        assert len(result) == 3
        assert ("oh my", 0, 5) in result
        assert ("oh my", 7, 12) in result
        assert ("oh my god", 7, 16) in result  # "oh my god" ends at position 16 (not 17)
    
    def test_repeated_phrase(self):
        """Test matching the same phrase multiple times."""
        text = "no way, no way, no way!"
        result = detect_phrases(text, ["no way"])
        assert len(result) == 3
        assert result[0] == ("no way", 0, 6)
        assert result[1] == ("no way", 8, 14)
        assert result[2] == ("no way", 16, 22)
    
    def test_empty_text(self):
        """Test with empty text."""
        result = detect_phrases("", ["oh my god"])
        assert len(result) == 0
    
    def test_empty_phrases_list(self):
        """Test with empty phrases list."""
        result = detect_phrases("Oh my god", [])
        assert len(result) == 0
    
    def test_empty_phrase_in_list(self):
        """Test with empty string in phrases list."""
        result = detect_phrases("Oh my god", ["", "no way"])
        # Should skip empty phrase and find "no way" if present
        assert len(result) == 0
    
    def test_phrase_not_in_text(self):
        """Test when phrase is not present in text."""
        result = detect_phrases("This is some text", ["oh my god"])
        assert len(result) == 0
    
    def test_special_regex_characters_in_phrase(self):
        """Test that special regex characters are escaped properly."""
        # Phrases with special regex characters should be matched literally
        result = detect_phrases("What the hell?", ["what the hell"])
        assert len(result) == 1
        assert result[0] == ("what the hell", 0, 13)
        
        # Test with parentheses
        result = detect_phrases("I can't believe (this)", ["can't believe"])
        assert len(result) == 1
        
        # Test with brackets
        result = detect_phrases("Look at [this]", ["look at"])
        assert len(result) == 1
    
    def test_sorted_by_start_position(self):
        """Test that results are sorted by start position."""
        text = "watch this, oh my god, no way"
        phrases = ["no way", "oh my god", "watch this"]  # Intentionally unsorted
        result = detect_phrases(text, phrases)
        # Results should be sorted by start position
        assert result[0][1] < result[1][1] < result[2][1]
        assert result[0] == ("watch this", 0, 10)
        assert result[1] == ("oh my god", 12, 21)
        assert result[2] == ("no way", 23, 29)
    
    def test_phrase_at_text_boundaries(self):
        """Test phrases at the start and end of text."""
        # Phrase at start
        result = detect_phrases("oh my god is here", ["oh my god"])
        assert len(result) == 1
        assert result[0] == ("oh my god", 0, 9)
        
        # Phrase at end
        result = detect_phrases("here is oh my god", ["oh my god"])
        assert len(result) == 1
        assert result[0] == ("oh my god", 8, 17)
        
        # Phrase is entire text
        result = detect_phrases("oh my god", ["oh my god"])
        assert len(result) == 1
        assert result[0] == ("oh my god", 0, 9)
    
    def test_whitespace_handling(self):
        """Test handling of various whitespace."""
        # Multiple spaces
        result = detect_phrases("oh  my  god", ["oh my god"])
        assert len(result) == 0  # Should not match due to extra spaces
        
        # Tabs and newlines
        result = detect_phrases("oh\tmy\tgod", ["oh my god"])
        assert len(result) == 0  # Should not match
        
        result = detect_phrases("oh\nmy\ngod", ["oh my god"])
        assert len(result) == 0  # Should not match


class TestDetectPhrasesProperties:
    """Property-based tests for phrase detection."""
    
    # Feature: clip-selection-improvements, Property 3: Phrase Detection with Word Boundaries
    @given(
        phrase=st.sampled_from(["oh my god", "no way", "watch this", "look at this"]),
        prefix=st.sampled_from(["", "Hello ", "Well, "]),
        suffix=st.sampled_from(["", "!", ".", " there"]),
    )
    @settings(max_examples=100)
    def test_phrase_always_found_with_boundaries(self, phrase, prefix, suffix):
        """For any phrase with valid word boundaries, it should be detected."""
        text = prefix + phrase + suffix
        result = detect_phrases(text, [phrase])
        
        # Should find exactly one match
        assert len(result) == 1
        assert result[0][0] == phrase
        # Verify the matched text is correct
        matched_text = text[result[0][1]:result[0][2]]
        assert matched_text.lower() == phrase.lower()
    
    # Feature: clip-selection-improvements, Property 3: Phrase Detection with Word Boundaries
    @given(
        phrase=st.sampled_from(["oh my god", "no way", "watch this"]),
        case_transform=st.sampled_from([str.upper, str.lower, str.title]),
    )
    @settings(max_examples=100)
    def test_case_insensitive_property(self, phrase, case_transform):
        """For any phrase and case transformation, matching should be case-insensitive."""
        transformed_text = case_transform(phrase)
        result = detect_phrases(transformed_text, [phrase])
        
        # Should always find the phrase regardless of case
        assert len(result) == 1
        assert result[0][0] == phrase
    
    # Feature: clip-selection-improvements, Property 3: Phrase Detection with Word Boundaries
    @given(
        phrase=st.sampled_from(["oh my god", "no way", "watch this"]),
        connector=st.sampled_from(["", "-", "_"]),
    )
    @settings(max_examples=100)
    def test_no_match_without_word_boundaries(self, phrase, connector):
        """For any phrase without word boundaries, it should NOT be detected."""
        # Create text without word boundaries by removing spaces
        no_boundary_text = phrase.replace(" ", connector)
        if connector == "":  # Only test when there are no spaces
            result = detect_phrases(no_boundary_text, [phrase])
            # Should NOT find a match when word boundaries are violated
            assert len(result) == 0
    
    # Feature: clip-selection-improvements, Property 3: Phrase Detection with Word Boundaries
    @given(
        phrases=st.lists(
            st.sampled_from(["oh my god", "no way", "watch this", "look at this"]),
            min_size=1,
            max_size=4,
            unique=True,
        )
    )
    @settings(max_examples=100)
    def test_results_sorted_by_position(self, phrases):
        """For any list of phrases, results should be sorted by start position."""
        # Create text with all phrases in a specific order
        text = ", ".join(phrases)
        result = detect_phrases(text, phrases)
        
        # Verify results are sorted by start position
        for i in range(len(result) - 1):
            assert result[i][1] <= result[i + 1][1]
    
    # Feature: clip-selection-improvements, Property 3: Phrase Detection with Word Boundaries
    @given(
        phrase=st.sampled_from(["oh my god", "no way", "watch this"]),
        repeat_count=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=100)
    def test_repeated_phrase_detection(self, phrase, repeat_count):
        """For any phrase repeated N times, should detect N matches."""
        text = ", ".join([phrase] * repeat_count)
        result = detect_phrases(text, [phrase])
        
        # Should find exactly repeat_count matches
        assert len(result) == repeat_count
        # All matches should be for the same phrase
        assert all(match[0] == phrase for match in result)


class TestDetectPhrasesEdgeCases:
    """Edge case tests for phrase detection."""
    
    def test_unicode_text(self):
        """Test with unicode characters."""
        result = detect_phrases("Oh my god 😱", ["oh my god"])
        assert len(result) == 1
        assert result[0] == ("oh my god", 0, 9)
    
    def test_very_long_text(self):
        """Test with very long text."""
        # Create a long text with the phrase somewhere in the middle
        long_text = "word " * 1000 + "oh my god" + " word" * 1000
        result = detect_phrases(long_text, ["oh my god"])
        assert len(result) == 1
        assert result[0][0] == "oh my god"
    
    def test_many_phrases(self):
        """Test with many phrases in the list."""
        phrases = [f"phrase {i}" for i in range(100)]
        text = "phrase 50 is here"
        result = detect_phrases(text, phrases)
        assert len(result) == 1
        assert result[0] == ("phrase 50", 0, 9)
    
    def test_phrase_with_apostrophe(self):
        """Test phrases containing apostrophes."""
        result = detect_phrases("I can't believe this", ["can't believe"])
        assert len(result) == 1
        assert result[0] == ("can't believe", 2, 15)
    
    def test_phrase_with_numbers(self):
        """Test phrases containing numbers."""
        result = detect_phrases("Look at this 123 thing", ["look at this"])
        assert len(result) == 1
        assert result[0] == ("look at this", 0, 12)
