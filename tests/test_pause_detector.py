"""Tests for natural pause detection module.

This module tests the natural pause detection functionality for identifying
punctuation pauses, silence gaps, and breath pauses in transcripts and audio.

**Validates: Requirements 5.1, 5.3, 5.4, 5.6**
"""

import os
import tempfile
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pipeline.models import NaturalPause, Segment, Transcript, WordTimestamp
from pipeline.pause_detector import (
    detect_natural_pauses,
    snap_to_nearest_pause,
    _detect_punctuation_pauses,
    _detect_silence_pauses,
    _group_consecutive,
)


class TestDetectPunctuationPauses:
    """Unit tests for punctuation pause detection."""
    
    def test_single_period(self):
        """Test detection of a single period."""
        transcript = Transcript(segments=[
            Segment(start=0.0, end=2.0, text="This is a sentence.")
        ])
        pauses = _detect_punctuation_pauses(transcript)
        assert len(pauses) == 1
        assert pauses[0].time == 2.0
        assert pauses[0].type == "punctuation"
        assert pauses[0].confidence == 0.9
    
    def test_multiple_punctuation_types(self):
        """Test detection of different punctuation marks."""
        transcript = Transcript(segments=[
            Segment(start=0.0, end=2.0, text="This is a sentence."),
            Segment(start=2.0, end=4.0, text="Is this a question?"),
            Segment(start=4.0, end=6.0, text="This is exciting!"),
        ])
        pauses = _detect_punctuation_pauses(transcript)
        assert len(pauses) == 3
        assert all(p.type == "punctuation" for p in pauses)
        assert all(p.confidence == 0.9 for p in pauses)
    
    def test_multiple_sentences_in_segment(self):
        """Test detection of multiple sentences in one segment."""
        transcript = Transcript(segments=[
            Segment(start=0.0, end=5.0, text="First sentence. Second sentence! Third?")
        ])
        pauses = _detect_punctuation_pauses(transcript)
        # All punctuation in the same segment should create pauses at segment end
        assert len(pauses) == 3
        assert all(p.time == 5.0 for p in pauses)
    
    def test_no_punctuation(self):
        """Test segment with no sentence-ending punctuation."""
        transcript = Transcript(segments=[
            Segment(start=0.0, end=2.0, text="This has no ending punctuation")
        ])
        pauses = _detect_punctuation_pauses(transcript)
        assert len(pauses) == 0
    
    def test_punctuation_with_whitespace(self):
        """Test punctuation followed by whitespace."""
        transcript = Transcript(segments=[
            Segment(start=0.0, end=2.0, text="Sentence one. "),
            Segment(start=2.0, end=4.0, text="Question? "),
        ])
        pauses = _detect_punctuation_pauses(transcript)
        assert len(pauses) == 2
    
    def test_context_extraction(self):
        """Test that context is properly extracted around punctuation."""
        long_text = "This is a very long sentence that goes on and on and on and finally ends here."
        transcript = Transcript(segments=[
            Segment(start=0.0, end=5.0, text=long_text)
        ])
        pauses = _detect_punctuation_pauses(transcript)
        assert len(pauses) == 1
        # Context should be truncated to ~60 chars (30 before + 30 after)
        assert len(pauses[0].context) <= len(long_text)


class TestDetectSilencePauses:
    """Unit tests for silence gap detection."""
    
    def test_single_silence_gap(self):
        """Test detection of a single silence gap."""
        transcript = Transcript(segments=[
            Segment(start=0.0, end=2.0, text="First segment"),
            Segment(start=3.0, end=5.0, text="Second segment"),
        ])
        pauses = _detect_silence_pauses(transcript, silence_threshold=0.5)
        assert len(pauses) == 1
        assert pauses[0].time == 2.5  # Midpoint of gap
        assert pauses[0].type == "silence"
        assert pauses[0].confidence == 0.8
    
    def test_multiple_silence_gaps(self):
        """Test detection of multiple silence gaps."""
        transcript = Transcript(segments=[
            Segment(start=0.0, end=1.0, text="First"),
            Segment(start=2.0, end=3.0, text="Second"),
            Segment(start=4.5, end=5.5, text="Third"),
        ])
        pauses = _detect_silence_pauses(transcript, silence_threshold=0.5)
        assert len(pauses) == 2
        assert pauses[0].time == 1.5  # Midpoint of first gap
        assert pauses[1].time == 3.75  # Midpoint of second gap
    
    def test_no_silence_gaps(self):
        """Test when segments are continuous with no gaps."""
        transcript = Transcript(segments=[
            Segment(start=0.0, end=2.0, text="First"),
            Segment(start=2.0, end=4.0, text="Second"),
        ])
        pauses = _detect_silence_pauses(transcript, silence_threshold=0.5)
        assert len(pauses) == 0
    
    def test_gap_below_threshold(self):
        """Test that gaps below threshold are ignored."""
        transcript = Transcript(segments=[
            Segment(start=0.0, end=2.0, text="First"),
            Segment(start=2.3, end=4.0, text="Second"),  # 0.3s gap
        ])
        pauses = _detect_silence_pauses(transcript, silence_threshold=0.5)
        assert len(pauses) == 0
    
    def test_gap_exactly_at_threshold(self):
        """Test that gaps exactly at threshold are detected."""
        transcript = Transcript(segments=[
            Segment(start=0.0, end=2.0, text="First"),
            Segment(start=2.5, end=4.0, text="Second"),  # Exactly 0.5s gap
        ])
        pauses = _detect_silence_pauses(transcript, silence_threshold=0.5)
        assert len(pauses) == 1
    
    def test_context_includes_surrounding_text(self):
        """Test that context includes text from both segments."""
        transcript = Transcript(segments=[
            Segment(start=0.0, end=2.0, text="This is the first segment"),
            Segment(start=3.0, end=5.0, text="This is the second segment"),
        ])
        pauses = _detect_silence_pauses(transcript, silence_threshold=0.5)
        assert len(pauses) == 1
        # Context is truncated to last 20 chars of first and first 20 chars of second
        assert "first segment" in pauses[0].context
        assert "This is the second" in pauses[0].context
        assert "silence" in pauses[0].context


class TestSnapToNearestPause:
    """Unit tests for snapping timestamps to nearest pause."""
    
    def test_snap_to_nearby_pause(self):
        """Test snapping to a pause within max_distance."""
        pauses = [
            NaturalPause(time=10.0, type="punctuation", confidence=0.9, context="test"),
            NaturalPause(time=20.0, type="silence", confidence=0.8, context="test"),
        ]
        result = snap_to_nearest_pause(11.5, pauses, max_distance=3.0)
        assert result == 10.0
    
    def test_no_snap_when_too_far(self):
        """Test that timestamp is not changed when pause is too far."""
        pauses = [
            NaturalPause(time=10.0, type="punctuation", confidence=0.9, context="test"),
        ]
        result = snap_to_nearest_pause(15.0, pauses, max_distance=3.0)
        assert result == 15.0  # Original time unchanged
    
    def test_snap_to_closest_of_multiple(self):
        """Test snapping to the closest pause when multiple are nearby."""
        pauses = [
            NaturalPause(time=10.0, type="punctuation", confidence=0.9, context="test"),
            NaturalPause(time=12.0, type="silence", confidence=0.8, context="test"),
            NaturalPause(time=15.0, type="breath", confidence=0.6, context="test"),
        ]
        result = snap_to_nearest_pause(11.0, pauses, max_distance=3.0)
        assert result == 10.0  # Closest is 10.0 (1.0 away)
    
    def test_empty_pause_list(self):
        """Test with empty pause list."""
        result = snap_to_nearest_pause(10.0, [], max_distance=3.0)
        assert result == 10.0
    
    def test_exact_match(self):
        """Test when timestamp exactly matches a pause."""
        pauses = [
            NaturalPause(time=10.0, type="punctuation", confidence=0.9, context="test"),
        ]
        result = snap_to_nearest_pause(10.0, pauses, max_distance=3.0)
        assert result == 10.0
    
    def test_snap_at_max_distance_boundary(self):
        """Test snapping when pause is exactly at max_distance."""
        pauses = [
            NaturalPause(time=10.0, type="punctuation", confidence=0.9, context="test"),
        ]
        result = snap_to_nearest_pause(13.0, pauses, max_distance=3.0)
        assert result == 10.0  # Exactly 3.0 away, should snap


class TestGroupConsecutive:
    """Unit tests for grouping consecutive integers."""
    
    def test_single_group(self):
        """Test grouping consecutive integers."""
        result = _group_consecutive(np.array([1, 2, 3, 4]))
        assert result == [[1, 2, 3, 4]]
    
    def test_multiple_groups(self):
        """Test grouping with gaps."""
        result = _group_consecutive(np.array([1, 2, 3, 5, 6, 8]))
        assert result == [[1, 2, 3], [5, 6], [8]]
    
    def test_single_elements(self):
        """Test with all single-element groups."""
        result = _group_consecutive(np.array([1, 3, 5, 7]))
        assert result == [[1], [3], [5], [7]]
    
    def test_empty_array(self):
        """Test with empty array."""
        result = _group_consecutive(np.array([]))
        assert result == []
    
    def test_single_element(self):
        """Test with single element."""
        result = _group_consecutive(np.array([5]))
        assert result == [[5]]


class TestDetectNaturalPausesIntegration:
    """Integration tests for full pause detection pipeline."""
    
    def test_detect_with_mock_audio(self, tmp_path):
        """Test full detection with a mock audio file."""
        # Create a simple mock WAV file
        wav_path = tmp_path / "test.wav"
        
        # Generate 5 seconds of audio with librosa
        try:
            import librosa
            import soundfile as sf
        except ImportError:
            pytest.skip("librosa or soundfile not installed")
        
        # Create simple audio: 5 seconds at 16kHz
        sr = 16000
        duration = 5.0
        t = np.linspace(0, duration, int(sr * duration))
        # Simple sine wave
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        sf.write(str(wav_path), audio, sr)
        
        # Create transcript with punctuation and silence
        transcript = Transcript(segments=[
            Segment(start=0.0, end=2.0, text="This is a sentence."),
            Segment(start=3.0, end=5.0, text="Another sentence!"),
        ])
        
        pauses = detect_natural_pauses(transcript, str(wav_path), silence_threshold=0.5)
        
        # Should detect at least punctuation and silence pauses
        assert len(pauses) > 0
        
        # Check that pauses are sorted by time
        times = [p.time for p in pauses]
        assert times == sorted(times)
        
        # Check that we have different types
        types = {p.type for p in pauses}
        assert "punctuation" in types
        assert "silence" in types
    
    def test_detect_without_librosa(self, monkeypatch):
        """Test that detection works gracefully without librosa."""
        # Mock librosa import to fail
        import sys
        monkeypatch.setitem(sys.modules, 'librosa', None)
        
        transcript = Transcript(segments=[
            Segment(start=0.0, end=2.0, text="This is a sentence."),
            Segment(start=3.0, end=5.0, text="Another sentence!"),
        ])
        
        # Should still work, just without breath detection
        pauses = detect_natural_pauses(transcript, "dummy.wav", silence_threshold=0.5)
        
        # Should have punctuation and silence, but no breath
        assert len(pauses) > 0
        types = {p.type for p in pauses}
        assert "punctuation" in types
        assert "silence" in types
        assert "breath" not in types


# ---------------------------------------------------------------------------
# Property-Based Tests
# ---------------------------------------------------------------------------

@given(
    num_segments=st.integers(min_value=1, max_value=20),
    silence_threshold=st.floats(min_value=0.1, max_value=2.0),
)
@settings(max_examples=50)
def test_silence_detection_property(num_segments: int, silence_threshold: float):
    """Property: All detected silence pauses should have gaps >= threshold.
    
    **Validates: Requirement 5.4**
    """
    # Generate segments with random gaps
    segments = []
    current_time = 0.0
    
    for i in range(num_segments):
        start = current_time
        end = start + np.random.uniform(1.0, 3.0)
        segments.append(Segment(start=start, end=end, text=f"Segment {i}"))
        
        # Add random gap before next segment
        current_time = end + np.random.uniform(0.0, 3.0)
    
    transcript = Transcript(segments=segments)
    pauses = _detect_silence_pauses(transcript, silence_threshold)
    
    # Property: All detected pauses should correspond to gaps >= threshold
    for pause in pauses:
        # Find the segments around this pause
        found_gap = False
        for i in range(len(segments) - 1):
            gap = segments[i + 1].start - segments[i].end
            if abs(pause.time - (segments[i].end + gap / 2.0)) < 0.01:
                assert gap >= silence_threshold, f"Gap {gap} < threshold {silence_threshold}"
                found_gap = True
                break
        assert found_gap, f"Pause at {pause.time} doesn't correspond to any gap"


@given(
    time=st.floats(min_value=0.0, max_value=100.0),
    num_pauses=st.integers(min_value=0, max_value=10),
    max_distance=st.floats(min_value=0.5, max_value=5.0),
)
@settings(max_examples=50)
def test_snap_distance_property(time: float, num_pauses: int, max_distance: float):
    """Property: Snapped time should be within max_distance or unchanged.
    
    **Validates: Requirement 5.6**
    """
    # Generate random pauses
    pauses = [
        NaturalPause(
            time=np.random.uniform(0.0, 100.0),
            type="punctuation",
            confidence=0.9,
            context="test",
        )
        for _ in range(num_pauses)
    ]
    
    result = snap_to_nearest_pause(time, pauses, max_distance)
    
    # Property: Result should either be original time or within max_distance
    if result == time:
        # If unchanged, verify no pause is within max_distance
        for pause in pauses:
            distance = abs(pause.time - time)
            # Allow small floating point error
            assert distance > max_distance or abs(distance - max_distance) < 1e-6
    else:
        # If changed, verify it's within max_distance
        distance = abs(result - time)
        assert distance <= max_distance


@given(
    text=st.text(min_size=1, max_size=200),
)
@settings(max_examples=50)
def test_punctuation_confidence_property(text: str):
    """Property: All punctuation pauses should have confidence=0.9.
    
    **Validates: Requirement 5.1**
    """
    transcript = Transcript(segments=[
        Segment(start=0.0, end=5.0, text=text)
    ])
    
    pauses = _detect_punctuation_pauses(transcript)
    
    # Property: All punctuation pauses have confidence 0.9
    for pause in pauses:
        assert pause.type == "punctuation"
        assert pause.confidence == 0.9


@given(
    num_segments=st.integers(min_value=2, max_value=10),
)
@settings(max_examples=50)
def test_pause_ordering_property(num_segments: int):
    """Property: Detected pauses should be sorted by time.
    
    **Validates: Requirement 5.1**
    """
    # Generate segments with various gaps
    segments = []
    current_time = 0.0
    
    for i in range(num_segments):
        start = current_time
        end = start + np.random.uniform(1.0, 3.0)
        text = "This is a sentence." if i % 2 == 0 else "No punctuation"
        segments.append(Segment(start=start, end=end, text=text))
        current_time = end + np.random.uniform(0.0, 2.0)
    
    transcript = Transcript(segments=segments)
    
    # Create a dummy audio file for testing
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name
    
    try:
        # Generate simple audio
        try:
            import soundfile as sf
            sr = 16000
            duration = current_time
            audio = np.random.randn(int(sr * duration)) * 0.1
            sf.write(wav_path, audio, sr)
            
            pauses = detect_natural_pauses(transcript, wav_path, silence_threshold=0.5)
            
            # Property: Pauses should be sorted by time
            times = [p.time for p in pauses]
            assert times == sorted(times), "Pauses are not sorted by time"
        except ImportError:
            # Skip if soundfile not available
            pass
    finally:
        if os.path.exists(wav_path):
            os.unlink(wav_path)


@given(
    indices=st.lists(st.integers(min_value=0, max_value=100), min_size=0, max_size=50),
)
@settings(max_examples=100)
def test_group_consecutive_property(indices: list[int]):
    """Property: Grouped consecutive integers should preserve all elements.
    
    **Validates: Internal helper function correctness**
    """
    if not indices:
        result = _group_consecutive(np.array([]))
        assert result == []
        return
    
    # Sort and remove duplicates
    unique_sorted = sorted(set(indices))
    arr = np.array(unique_sorted)
    
    groups = _group_consecutive(arr)
    
    # Property 1: All elements should be preserved
    flattened = [item for group in groups for item in group]
    assert sorted(flattened) == unique_sorted
    
    # Property 2: Within each group, elements should be consecutive
    for group in groups:
        for i in range(len(group) - 1):
            assert group[i + 1] == group[i] + 1, f"Group {group} is not consecutive"
    
    # Property 3: Groups should not be mergeable (gap between groups > 1)
    for i in range(len(groups) - 1):
        gap = groups[i + 1][0] - groups[i][-1]
        assert gap > 1, f"Groups {groups[i]} and {groups[i+1]} should be merged"
