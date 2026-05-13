"""Unit tests for semantic deduplication module.

Tests the compute_semantic_similarity() function and helper functions.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from hypothesis import given, settings
from hypothesis import strategies as st

from pipeline.semantic_dedup import compute_semantic_similarity, deduplicate_semantic, _extract_clip_text
from pipeline.models import Clip, Transcript, Segment


class TestExtractClipText:
    """Tests for _extract_clip_text helper function."""
    
    def test_extract_single_segment(self):
        """Extract text from a clip with one segment."""
        transcript = Transcript(segments=[
            Segment(start=0.0, end=5.0, text="Hello world"),
            Segment(start=5.0, end=10.0, text="How are you"),
        ])
        clip = Clip(start=0.0, end=5.0, score=0.8, rank=1, segment_indices=[0])
        
        text = _extract_clip_text(clip, transcript)
        assert text == "Hello world"
    
    def test_extract_multiple_segments(self):
        """Extract text from a clip with multiple segments."""
        transcript = Transcript(segments=[
            Segment(start=0.0, end=5.0, text="Hello world"),
            Segment(start=5.0, end=10.0, text="How are you"),
            Segment(start=10.0, end=15.0, text="I am fine"),
        ])
        clip = Clip(start=0.0, end=15.0, score=0.8, rank=1, segment_indices=[0, 1, 2])
        
        text = _extract_clip_text(clip, transcript)
        assert text == "Hello world How are you I am fine"
    
    def test_extract_empty_clip(self):
        """Extract text from a clip with no segments."""
        transcript = Transcript(segments=[
            Segment(start=0.0, end=5.0, text="Hello world"),
        ])
        clip = Clip(start=0.0, end=0.0, score=0.0, rank=1, segment_indices=[])
        
        text = _extract_clip_text(clip, transcript)
        assert text == ""
    
    def test_extract_out_of_bounds_indices(self):
        """Extract text with out-of-bounds segment indices (should skip them)."""
        transcript = Transcript(segments=[
            Segment(start=0.0, end=5.0, text="Hello world"),
        ])
        clip = Clip(start=0.0, end=5.0, score=0.8, rank=1, segment_indices=[0, 5, 10])
        
        text = _extract_clip_text(clip, transcript)
        assert text == "Hello world"
    
    def test_extract_preserves_order(self):
        """Extract text preserves segment order."""
        transcript = Transcript(segments=[
            Segment(start=0.0, end=5.0, text="First"),
            Segment(start=5.0, end=10.0, text="Second"),
            Segment(start=10.0, end=15.0, text="Third"),
        ])
        clip = Clip(start=0.0, end=15.0, score=0.8, rank=1, segment_indices=[2, 0, 1])
        
        text = _extract_clip_text(clip, transcript)
        # Should preserve the order in segment_indices
        assert text == "Third First Second"


class TestComputeSemanticSimilarity:
    """Tests for compute_semantic_similarity function."""
    
    def test_identical_clips_high_similarity(self):
        """Identical clips should have similarity close to 1.0."""
        transcript = Transcript(segments=[
            Segment(start=0.0, end=5.0, text="This is a test sentence about machine learning"),
        ])
        clip_a = Clip(start=0.0, end=5.0, score=0.8, rank=1, segment_indices=[0])
        clip_b = Clip(start=0.0, end=5.0, score=0.7, rank=2, segment_indices=[0])
        
        similarity = compute_semantic_similarity(clip_a, clip_b, transcript)
        
        # Identical text should have similarity very close to 1.0
        assert 0.99 <= similarity <= 1.0
    
    def test_similar_meaning_high_similarity(self):
        """Clips with similar meanings should have high similarity."""
        transcript = Transcript(segments=[
            Segment(start=0.0, end=5.0, text="The cat sat on the mat"),
            Segment(start=5.0, end=10.0, text="A feline rested on the rug"),
        ])
        clip_a = Clip(start=0.0, end=5.0, score=0.8, rank=1, segment_indices=[0])
        clip_b = Clip(start=5.0, end=10.0, score=0.7, rank=2, segment_indices=[1])
        
        similarity = compute_semantic_similarity(clip_a, clip_b, transcript)
        
        # Similar meaning should have high similarity (>0.5)
        # Note: Semantic similarity can vary based on model, so we use a reasonable threshold
        assert similarity > 0.5
    
    def test_different_topics_low_similarity(self):
        """Clips with different topics should have low similarity."""
        transcript = Transcript(segments=[
            Segment(start=0.0, end=5.0, text="Machine learning is a subset of artificial intelligence"),
            Segment(start=5.0, end=10.0, text="I love eating pizza and pasta for dinner"),
        ])
        clip_a = Clip(start=0.0, end=5.0, score=0.8, rank=1, segment_indices=[0])
        clip_b = Clip(start=5.0, end=10.0, score=0.7, rank=2, segment_indices=[1])
        
        similarity = compute_semantic_similarity(clip_a, clip_b, transcript)
        
        # Different topics should have low similarity (<0.5)
        assert similarity < 0.5
    
    def test_empty_clip_returns_zero(self):
        """Empty clip text should return 0.0 similarity."""
        transcript = Transcript(segments=[
            Segment(start=0.0, end=5.0, text="Hello world"),
        ])
        clip_a = Clip(start=0.0, end=5.0, score=0.8, rank=1, segment_indices=[0])
        clip_b = Clip(start=5.0, end=10.0, score=0.7, rank=2, segment_indices=[])
        
        similarity = compute_semantic_similarity(clip_a, clip_b, transcript)
        
        assert similarity == 0.0
    
    def test_whitespace_only_returns_zero(self):
        """Whitespace-only clip text should return 0.0 similarity."""
        transcript = Transcript(segments=[
            Segment(start=0.0, end=5.0, text="Hello world"),
            Segment(start=5.0, end=10.0, text="   "),
        ])
        clip_a = Clip(start=0.0, end=5.0, score=0.8, rank=1, segment_indices=[0])
        clip_b = Clip(start=5.0, end=10.0, score=0.7, rank=2, segment_indices=[1])
        
        similarity = compute_semantic_similarity(clip_a, clip_b, transcript)
        
        assert similarity == 0.0
    
    def test_similarity_in_valid_range(self):
        """Similarity score should always be in [0.0, 1.0]."""
        transcript = Transcript(segments=[
            Segment(start=0.0, end=5.0, text="The quick brown fox jumps over the lazy dog"),
            Segment(start=5.0, end=10.0, text="Python is a programming language"),
        ])
        clip_a = Clip(start=0.0, end=5.0, score=0.8, rank=1, segment_indices=[0])
        clip_b = Clip(start=5.0, end=10.0, score=0.7, rank=2, segment_indices=[1])
        
        similarity = compute_semantic_similarity(clip_a, clip_b, transcript)
        
        assert 0.0 <= similarity <= 1.0
    
    def test_similarity_is_symmetric(self):
        """Similarity should be symmetric: sim(A, B) == sim(B, A)."""
        transcript = Transcript(segments=[
            Segment(start=0.0, end=5.0, text="Machine learning algorithms"),
            Segment(start=5.0, end=10.0, text="Deep neural networks"),
        ])
        clip_a = Clip(start=0.0, end=5.0, score=0.8, rank=1, segment_indices=[0])
        clip_b = Clip(start=5.0, end=10.0, score=0.7, rank=2, segment_indices=[1])
        
        similarity_ab = compute_semantic_similarity(clip_a, clip_b, transcript)
        similarity_ba = compute_semantic_similarity(clip_b, clip_a, transcript)
        
        # Should be exactly equal (or very close due to floating point)
        assert abs(similarity_ab - similarity_ba) < 1e-6
    
    def test_multiple_segments_similarity(self):
        """Clips with multiple segments should compute similarity correctly."""
        transcript = Transcript(segments=[
            Segment(start=0.0, end=5.0, text="I went to the store"),
            Segment(start=5.0, end=10.0, text="to buy some groceries"),
            Segment(start=10.0, end=15.0, text="I visited the market"),
            Segment(start=15.0, end=20.0, text="to purchase food items"),
        ])
        clip_a = Clip(start=0.0, end=10.0, score=0.8, rank=1, segment_indices=[0, 1])
        clip_b = Clip(start=10.0, end=20.0, score=0.7, rank=2, segment_indices=[2, 3])
        
        similarity = compute_semantic_similarity(clip_a, clip_b, transcript)
        
        # Similar meaning across multiple segments should have high similarity
        assert similarity > 0.6


class TestSemanticSimilarityEdgeCases:
    """Edge case tests for semantic similarity."""
    
    def test_very_long_text_similarity(self):
        """Very long text should be handled correctly."""
        long_text_a = " ".join(["This is sentence number " + str(i) for i in range(100)])
        long_text_b = " ".join(["This is sentence number " + str(i) for i in range(100)])
        
        transcript = Transcript(segments=[
            Segment(start=0.0, end=100.0, text=long_text_a),
            Segment(start=100.0, end=200.0, text=long_text_b),
        ])
        clip_a = Clip(start=0.0, end=100.0, score=0.8, rank=1, segment_indices=[0])
        clip_b = Clip(start=100.0, end=200.0, score=0.7, rank=2, segment_indices=[1])
        
        similarity = compute_semantic_similarity(clip_a, clip_b, transcript)
        
        # Identical long text should have high similarity
        assert similarity > 0.99
    
    def test_special_characters_similarity(self):
        """Text with special characters should be handled correctly."""
        transcript = Transcript(segments=[
            Segment(start=0.0, end=5.0, text="Hello! How are you? I'm fine :)"),
            Segment(start=5.0, end=10.0, text="Hi! How are you doing? I am good :)"),
        ])
        clip_a = Clip(start=0.0, end=5.0, score=0.8, rank=1, segment_indices=[0])
        clip_b = Clip(start=5.0, end=10.0, score=0.7, rank=2, segment_indices=[1])
        
        similarity = compute_semantic_similarity(clip_a, clip_b, transcript)
        
        # Similar meaning with special characters should have high similarity
        assert similarity > 0.6
    
    def test_numbers_and_text_similarity(self):
        """Text with numbers should be handled correctly."""
        transcript = Transcript(segments=[
            Segment(start=0.0, end=5.0, text="I have 3 apples and 2 oranges"),
            Segment(start=5.0, end=10.0, text="I own three apples and two oranges"),
        ])
        clip_a = Clip(start=0.0, end=5.0, score=0.8, rank=1, segment_indices=[0])
        clip_b = Clip(start=5.0, end=10.0, score=0.7, rank=2, segment_indices=[1])
        
        similarity = compute_semantic_similarity(clip_a, clip_b, transcript)
        
        # Similar meaning with numbers should have high similarity
        assert similarity > 0.6
    
    def test_single_word_clips(self):
        """Single word clips should compute similarity correctly."""
        transcript = Transcript(segments=[
            Segment(start=0.0, end=1.0, text="cat"),
            Segment(start=1.0, end=2.0, text="dog"),
        ])
        clip_a = Clip(start=0.0, end=1.0, score=0.8, rank=1, segment_indices=[0])
        clip_b = Clip(start=1.0, end=2.0, score=0.7, rank=2, segment_indices=[1])
        
        similarity = compute_semantic_similarity(clip_a, clip_b, transcript)
        
        # Different single words should have low similarity
        assert 0.0 <= similarity <= 1.0
    
    def test_unicode_text_similarity(self):
        """Unicode text should be handled correctly."""
        transcript = Transcript(segments=[
            Segment(start=0.0, end=5.0, text="Hello 世界 🌍"),
            Segment(start=5.0, end=10.0, text="Hello world 🌎"),
        ])
        clip_a = Clip(start=0.0, end=5.0, score=0.8, rank=1, segment_indices=[0])
        clip_b = Clip(start=5.0, end=10.0, score=0.7, rank=2, segment_indices=[1])
        
        similarity = compute_semantic_similarity(clip_a, clip_b, transcript)
        
        # Should handle unicode without crashing
        assert 0.0 <= similarity <= 1.0



# ---------------------------------------------------------------------------
# Property-Based Tests
# ---------------------------------------------------------------------------

@st.composite
def clip_pair_strategy(draw):
    """Generate a pair of clips with transcript for property testing.
    
    Returns a tuple of (clip_a, clip_b, transcript) where both clips
    reference valid segments in the transcript.
    """
    # Generate 2-10 segments
    num_segments = draw(st.integers(min_value=2, max_value=10))
    
    segments = []
    current_time = 0.0
    
    for i in range(num_segments):
        # Each segment is 3-10 seconds long
        duration = draw(st.floats(min_value=3.0, max_value=10.0, allow_nan=False, allow_infinity=False))
        start = current_time
        end = current_time + duration
        
        # Generate text for the segment (shorter to reduce entropy)
        text = draw(st.text(
            alphabet='abcdefghijklmnopqrstuvwxyz ',
            min_size=5,
            max_size=30
        ))
        
        segments.append(Segment(start=start, end=end, text=text))
        current_time = end
    
    transcript = Transcript(segments=segments)
    
    # Generate two clips with different segment indices
    # Clip A: use first half of segments
    clip_a_indices = list(range(0, num_segments // 2 + 1))
    clip_a = Clip(
        start=segments[0].start,
        end=segments[clip_a_indices[-1]].end,
        score=0.8,
        rank=1,
        segment_indices=clip_a_indices
    )
    
    # Clip B: use second half of segments
    clip_b_indices = list(range(num_segments // 2, num_segments))
    clip_b = Clip(
        start=segments[clip_b_indices[0]].start,
        end=segments[clip_b_indices[-1]].end,
        score=0.7,
        rank=2,
        segment_indices=clip_b_indices
    )
    
    return clip_a, clip_b, transcript


# ---------------------------------------------------------------------------
# Property 8: Semantic Similarity Symmetry and Bounds
# Validates: Requirements 7.1
# ---------------------------------------------------------------------------

# Feature: clip-selection-improvements, Property 8: Semantic Similarity Symmetry and Bounds
@given(clip_pair=clip_pair_strategy())
@settings(max_examples=25, deadline=None)
def test_semantic_similarity_symmetry_and_bounds(clip_pair):
    """For any two clip transcripts A and B, the semantic similarity should be
    symmetric (sim(A,B) == sim(B,A)) and bounded in the range [0.0, 1.0].
    
    This test validates two critical properties of semantic similarity:
    
    1. **Symmetry**: The similarity between clip A and clip B should be the same
       as the similarity between clip B and clip A. This is a fundamental property
       of cosine similarity and ensures consistent behavior regardless of argument order.
    
    2. **Bounds**: The similarity score must always be in the range [0.0, 1.0], where:
       - 0.0 indicates completely different/orthogonal content
       - 1.0 indicates identical content
       - Values in between indicate varying degrees of semantic similarity
    
    The test uses property-based testing to verify these properties hold across
    a wide range of randomly generated clip pairs with different text content.
    
    **Validates: Requirements 7.1**
    """
    clip_a, clip_b, transcript = clip_pair
    
    # Skip if either clip has empty text (edge case handled separately in unit tests)
    text_a = _extract_clip_text(clip_a, transcript)
    text_b = _extract_clip_text(clip_b, transcript)
    
    # Skip empty or whitespace-only text (these return 0.0 by design)
    if not text_a.strip() or not text_b.strip():
        return
    
    # Compute similarity in both directions
    similarity_ab = compute_semantic_similarity(clip_a, clip_b, transcript)
    similarity_ba = compute_semantic_similarity(clip_b, clip_a, transcript)
    
    # Property 1: Symmetry - sim(A, B) should equal sim(B, A)
    # Allow small floating-point tolerance (1e-6) for numerical precision
    assert abs(similarity_ab - similarity_ba) < 1e-6, \
        f"Semantic similarity should be symmetric: " \
        f"sim(A, B) = {similarity_ab:.8f}, sim(B, A) = {similarity_ba:.8f}, " \
        f"difference = {abs(similarity_ab - similarity_ba):.8e}. " \
        f"Text A: '{text_a[:50]}...', Text B: '{text_b[:50]}...'"
    
    # Property 2: Bounds - similarity must be in [0.0, 1.0]
    assert 0.0 <= similarity_ab <= 1.0, \
        f"Semantic similarity must be in range [0.0, 1.0], got: {similarity_ab}. " \
        f"Text A: '{text_a[:50]}...', Text B: '{text_b[:50]}...'"
    
    assert 0.0 <= similarity_ba <= 1.0, \
        f"Semantic similarity must be in range [0.0, 1.0], got: {similarity_ba}. " \
        f"Text A: '{text_a[:50]}...', Text B: '{text_b[:50]}...'"


# Feature: clip-selection-improvements, Property 8: Semantic Similarity Symmetry and Bounds (identical clips)
@given(
    text=st.text(
        alphabet='abcdefghijklmnopqrstuvwxyz ',
        min_size=10,
        max_size=50
    )
)
@settings(max_examples=25, deadline=None)
def test_semantic_similarity_identical_clips(text):
    """For any clip compared with itself, the semantic similarity should be 1.0
    (or very close to 1.0 due to floating-point precision).
    
    This is a special case of the symmetry property where A == B, so sim(A, A)
    should equal 1.0 (perfect similarity).
    
    **Validates: Requirements 7.1**
    """
    # Skip empty or whitespace-only text
    if not text.strip():
        return
    
    # Create a transcript with one segment
    segment = Segment(start=0.0, end=5.0, text=text)
    transcript = Transcript(segments=[segment])
    
    # Create a clip referencing this segment
    clip = Clip(start=0.0, end=5.0, score=0.8, rank=1, segment_indices=[0])
    
    # Compute similarity with itself
    similarity = compute_semantic_similarity(clip, clip, transcript)
    
    # Should be very close to 1.0 (allow small floating-point tolerance)
    assert 0.99 <= similarity <= 1.0, \
        f"Semantic similarity of a clip with itself should be ~1.0, got: {similarity}. " \
        f"Text: '{text[:50]}...'"
    
    # Verify bounds
    assert 0.0 <= similarity <= 1.0, \
        f"Semantic similarity must be in range [0.0, 1.0], got: {similarity}"



# ---------------------------------------------------------------------------
# Unit Tests for deduplicate_semantic()
# ---------------------------------------------------------------------------

class TestDeduplicateSemantic:
    """Tests for deduplicate_semantic function."""
    
    def test_empty_list_returns_empty(self):
        """Empty clip list should return empty list."""
        transcript = Transcript(segments=[])
        clips = []
        
        result = deduplicate_semantic(clips, transcript, threshold=0.8)
        
        assert result == []
    
    def test_single_clip_returns_unchanged(self):
        """Single clip should be returned unchanged."""
        transcript = Transcript(segments=[
            Segment(start=0.0, end=5.0, text="Hello world"),
        ])
        clips = [
            Clip(start=0.0, end=5.0, score=0.8, rank=1, segment_indices=[0]),
        ]
        
        result = deduplicate_semantic(clips, transcript, threshold=0.8)
        
        assert len(result) == 1
        assert result[0] == clips[0]
    
    def test_two_dissimilar_clips_both_kept(self):
        """Two dissimilar clips should both be kept."""
        transcript = Transcript(segments=[
            Segment(start=0.0, end=5.0, text="Machine learning is a subset of artificial intelligence"),
            Segment(start=5.0, end=10.0, text="I love eating pizza and pasta for dinner"),
        ])
        clips = [
            Clip(start=0.0, end=5.0, score=0.8, rank=1, segment_indices=[0]),
            Clip(start=5.0, end=10.0, score=0.7, rank=2, segment_indices=[1]),
        ]
        
        result = deduplicate_semantic(clips, transcript, threshold=0.8)
        
        # Both clips should be kept (different topics)
        assert len(result) == 2
        assert result[0] == clips[0]
        assert result[1] == clips[1]
    
    def test_two_similar_clips_keeps_higher_score(self):
        """Two similar clips should keep only the higher-scoring one."""
        transcript = Transcript(segments=[
            Segment(start=0.0, end=5.0, text="The cat sat on the mat"),
            Segment(start=5.0, end=10.0, text="A feline rested on the rug"),
        ])
        clips = [
            Clip(start=0.0, end=5.0, score=0.8, rank=1, segment_indices=[0]),
            Clip(start=5.0, end=10.0, score=0.7, rank=2, segment_indices=[1]),
        ]
        
        result = deduplicate_semantic(clips, transcript, threshold=0.5)
        
        # Only the higher-scoring clip should be kept
        assert len(result) == 1
        assert result[0] == clips[0]  # Higher score (0.8)
    
    def test_two_similar_clips_keeps_higher_score_reversed(self):
        """Two similar clips should keep the higher-scoring one (reversed order)."""
        transcript = Transcript(segments=[
            Segment(start=0.0, end=5.0, text="The cat sat on the mat"),
            Segment(start=5.0, end=10.0, text="A feline rested on the rug"),
        ])
        clips = [
            Clip(start=0.0, end=5.0, score=0.6, rank=2, segment_indices=[0]),
            Clip(start=5.0, end=10.0, score=0.9, rank=1, segment_indices=[1]),
        ]
        
        result = deduplicate_semantic(clips, transcript, threshold=0.5)
        
        # Only the higher-scoring clip should be kept
        assert len(result) == 1
        assert result[0] == clips[1]  # Higher score (0.9)
    
    def test_three_clips_two_similar_one_different(self):
        """Three clips: two similar, one different. Should keep two clips."""
        transcript = Transcript(segments=[
            Segment(start=0.0, end=5.0, text="The cat sat on the mat"),
            Segment(start=5.0, end=10.0, text="A feline rested on the rug"),
            Segment(start=10.0, end=15.0, text="Machine learning algorithms are powerful"),
        ])
        clips = [
            Clip(start=0.0, end=5.0, score=0.8, rank=1, segment_indices=[0]),
            Clip(start=5.0, end=10.0, score=0.7, rank=2, segment_indices=[1]),
            Clip(start=10.0, end=15.0, score=0.9, rank=3, segment_indices=[2]),
        ]
        
        result = deduplicate_semantic(clips, transcript, threshold=0.5)
        
        # Should keep clips[0] (higher score than clips[1]) and clips[2] (different topic)
        assert len(result) == 2
        assert clips[0] in result
        assert clips[2] in result
        assert clips[1] not in result
    
    def test_identical_clips_keeps_higher_score(self):
        """Identical clips should keep only the higher-scoring one."""
        transcript = Transcript(segments=[
            Segment(start=0.0, end=5.0, text="This is a test sentence"),
        ])
        clips = [
            Clip(start=0.0, end=5.0, score=0.8, rank=1, segment_indices=[0]),
            Clip(start=0.0, end=5.0, score=0.6, rank=2, segment_indices=[0]),
        ]
        
        result = deduplicate_semantic(clips, transcript, threshold=0.8)
        
        # Only the higher-scoring clip should be kept
        assert len(result) == 1
        assert result[0] == clips[0]  # Higher score (0.8)
    
    def test_threshold_controls_deduplication(self):
        """Threshold parameter should control deduplication strictness."""
        transcript = Transcript(segments=[
            Segment(start=0.0, end=5.0, text="The cat sat on the mat"),
            Segment(start=5.0, end=10.0, text="A feline rested on the rug"),
        ])
        clips = [
            Clip(start=0.0, end=5.0, score=0.8, rank=1, segment_indices=[0]),
            Clip(start=5.0, end=10.0, score=0.7, rank=2, segment_indices=[1]),
        ]
        
        # With high threshold (0.95), clips should not be deduplicated
        result_high = deduplicate_semantic(clips, transcript, threshold=0.95)
        assert len(result_high) == 2
        
        # With low threshold (0.3), clips should be deduplicated
        result_low = deduplicate_semantic(clips, transcript, threshold=0.3)
        assert len(result_low) == 1
    
    def test_preserves_original_order(self):
        """Deduplicated list should preserve original order."""
        transcript = Transcript(segments=[
            Segment(start=0.0, end=5.0, text="First clip about cats"),
            Segment(start=5.0, end=10.0, text="Second clip about dogs"),
            Segment(start=10.0, end=15.0, text="Third clip about birds"),
        ])
        clips = [
            Clip(start=0.0, end=5.0, score=0.8, rank=1, segment_indices=[0]),
            Clip(start=5.0, end=10.0, score=0.7, rank=2, segment_indices=[1]),
            Clip(start=10.0, end=15.0, score=0.9, rank=3, segment_indices=[2]),
        ]
        
        result = deduplicate_semantic(clips, transcript, threshold=0.8)
        
        # All clips should be kept (different topics)
        assert len(result) == 3
        # Order should be preserved
        assert result[0] == clips[0]
        assert result[1] == clips[1]
        assert result[2] == clips[2]
    
    def test_multiple_similar_pairs(self):
        """Multiple similar pairs should all be deduplicated correctly."""
        transcript = Transcript(segments=[
            Segment(start=0.0, end=5.0, text="The cat sat on the mat"),
            Segment(start=5.0, end=10.0, text="A feline rested on the rug"),
            Segment(start=10.0, end=15.0, text="Dogs are loyal animals"),
            Segment(start=15.0, end=20.0, text="Canines are faithful creatures"),
        ])
        clips = [
            Clip(start=0.0, end=5.0, score=0.8, rank=1, segment_indices=[0]),
            Clip(start=5.0, end=10.0, score=0.7, rank=2, segment_indices=[1]),
            Clip(start=10.0, end=15.0, score=0.9, rank=3, segment_indices=[2]),
            Clip(start=15.0, end=20.0, score=0.6, rank=4, segment_indices=[3]),
        ]
        
        result = deduplicate_semantic(clips, transcript, threshold=0.5)
        
        # Should keep clips[0] (higher than clips[1]) and clips[2] (higher than clips[3])
        assert len(result) == 2
        assert clips[0] in result
        assert clips[2] in result
        assert clips[1] not in result
        assert clips[3] not in result
    
    def test_chain_of_similar_clips(self):
        """Chain of similar clips (A~B, B~C) should keep only highest-scoring."""
        transcript = Transcript(segments=[
            Segment(start=0.0, end=5.0, text="The cat sat on the mat"),
            Segment(start=5.0, end=10.0, text="A feline rested on the rug"),
            Segment(start=10.0, end=15.0, text="The kitty lay on the carpet"),
        ])
        clips = [
            Clip(start=0.0, end=5.0, score=0.8, rank=1, segment_indices=[0]),
            Clip(start=5.0, end=10.0, score=0.9, rank=2, segment_indices=[1]),
            Clip(start=10.0, end=15.0, score=0.7, rank=3, segment_indices=[2]),
        ]
        
        result = deduplicate_semantic(clips, transcript, threshold=0.5)
        
        # clips[1] has highest score and should be kept
        # clips[0] and clips[2] should be removed (similar to clips[1])
        assert len(result) == 1
        assert result[0] == clips[1]
