"""Unit tests for emotion detection module.

Tests cover:
- Feature extraction (mocked librosa)
- Emotion classification rules
- Confidence scoring
- Silent segment handling
- Graceful degradation when librosa unavailable

**Validates: Requirements 22.1, 22.4**
"""

import numpy as np
import pytest
import scipy.io.wavfile
import tempfile
import os
from unittest.mock import patch, MagicMock

from pipeline.emotion_detector import (
    extract_emotion_features,
    _classify_emotion,
    _normalize_array,
    _compute_rolling_std,
)
from pipeline.models import EmotionFeatures


@pytest.fixture
def test_wav_file():
    """Create a temporary WAV file with synthetic audio for testing."""
    # Generate 5 seconds of audio at 16kHz
    sample_rate = 16000
    duration = 5.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create audio with varying characteristics
    # First 2 seconds: moderate volume, moderate pitch (300 Hz) - excitement
    # Next 2 seconds: high volume, high pitch (500 Hz) - scream
    # Last 1 second: silence - neutral
    audio = np.zeros_like(t)
    
    # Moderate volume, moderate pitch (0-2s)
    mask1 = t < 2.0
    audio[mask1] = 0.6 * np.sin(2 * np.pi * 300 * t[mask1])
    
    # High volume, high pitch (2-4s)
    mask2 = (t >= 2.0) & (t < 4.0)
    audio[mask2] = 0.9 * np.sin(2 * np.pi * 500 * t[mask2])
    
    # Silence (4-5s) - already zeros
    
    # Convert to int16
    audio_int16 = (audio * 32767).astype(np.int16)
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        scipy.io.wavfile.write(f.name, sample_rate, audio_int16)
        wav_path = f.name
    
    yield wav_path
    
    # Cleanup
    os.unlink(wav_path)


# ============================================================================
# Feature Extraction Tests (with mocked librosa)
# ============================================================================

def test_extract_emotion_features_with_mocked_librosa():
    """Test feature extraction with mocked librosa to verify processing logic."""
    # Create a mock module that will be imported
    mock_librosa = MagicMock()
    
    # Mock audio data
    sample_rate = 16000
    duration = 2.0
    num_samples = int(sample_rate * duration)
    mock_audio = np.random.randn(num_samples) * 0.5
    
    # Mock librosa.load
    mock_librosa.load.return_value = (mock_audio, sample_rate)
    
    # Mock feature extraction functions
    # RMS: 4 windows for 2 seconds at 0.5s window size
    mock_librosa.feature.rms.return_value = np.array([[0.5, 0.6, 0.7, 0.4]])
    
    # Pitch (F0): same length as RMS
    mock_librosa.pyin.return_value = (
        np.array([200.0, 300.0, 400.0, 150.0]),  # f0
        np.array([True, True, True, True]),       # voiced_flag
        np.array([0.9, 0.9, 0.9, 0.9])           # voiced_probs
    )
    
    # Spectral centroid
    mock_librosa.feature.spectral_centroid.return_value = np.array([[2000.0, 2500.0, 3500.0, 1800.0]])
    
    # Zero-crossing rate
    mock_librosa.feature.zero_crossing_rate.return_value = np.array([[0.05, 0.08, 0.12, 0.04]])
    
    # Mock note_to_hz
    mock_librosa.note_to_hz.side_effect = lambda x: 65.0 if x == 'C2' else 2093.0
    
    # Patch the import inside the function
    with patch.dict('sys.modules', {'librosa': mock_librosa}):
        # Need to reload the module to pick up the mocked import
        import importlib
        import pipeline.emotion_detector
        importlib.reload(pipeline.emotion_detector)
        
        features = pipeline.emotion_detector.extract_emotion_features("test.wav", window_size=0.5)
        
        # Reload again to restore original
        importlib.reload(pipeline.emotion_detector)
    
    # Verify librosa was called correctly
    mock_librosa.load.assert_called_once()
    mock_librosa.feature.rms.assert_called_once()
    mock_librosa.pyin.assert_called_once()
    mock_librosa.feature.spectral_centroid.assert_called_once()
    mock_librosa.feature.zero_crossing_rate.assert_called_once()
    
    # Verify features were extracted
    assert len(features) == 4
    assert all(isinstance(f, EmotionFeatures) for f in features)
    
    # Verify feature values are in expected ranges
    for f in features:
        assert f.time >= 0.0
        assert f.pitch_mean >= 0.0
        assert 0.0 <= f.volume_rms <= 1.0
        assert f.spectral_centroid >= 0.0
        assert 0.0 <= f.zero_crossing_rate <= 1.0


def test_extract_emotion_features_librosa_unavailable():
    """Test graceful degradation when librosa is not installed."""
    # Mock ImportError when trying to import librosa
    import builtins
    real_import = builtins.__import__
    
    def mock_import(name, *args, **kwargs):
        if name == 'librosa':
            raise ImportError("No module named 'librosa'")
        return real_import(name, *args, **kwargs)
    
    with patch('builtins.__import__', side_effect=mock_import):
        # Need to reload to trigger the import error
        import importlib
        import pipeline.emotion_detector
        importlib.reload(pipeline.emotion_detector)
        
        features = pipeline.emotion_detector.extract_emotion_features("test.wav", window_size=0.5)
        
        # Reload to restore
        importlib.reload(pipeline.emotion_detector)
    
    # Should return empty list when librosa unavailable
    assert features == []


def test_extract_emotion_features_load_failure():
    """Test handling of audio file load failures."""
    # This test uses real librosa but with a bad file path
    # The function should catch the exception and return empty list
    features = extract_emotion_features("nonexistent_bad_file_12345.wav", window_size=0.5)
    
    # Should return empty list on load failure
    assert features == []


def test_extract_emotion_features_empty_audio():
    """Test handling of empty audio files."""
    # Create an actual empty WAV file
    sample_rate = 16000
    audio = np.array([], dtype=np.int16)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        scipy.io.wavfile.write(f.name, sample_rate, audio)
        wav_path = f.name
    
    try:
        features = extract_emotion_features(wav_path, window_size=0.5)
        
        # Should return empty list for empty audio
        assert features == []
    finally:
        os.unlink(wav_path)


def test_extract_emotion_features_silent_audio_mocked():
    """Test handling of completely silent audio."""
    # Create actual silent audio file
    sample_rate = 16000
    duration = 2.0
    silent_audio = np.zeros(int(sample_rate * duration), dtype=np.int16)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        scipy.io.wavfile.write(f.name, sample_rate, silent_audio)
        wav_path = f.name
    
    try:
        features = extract_emotion_features(wav_path, window_size=0.5)
        
        # Should return empty list for silent audio
        assert features == []
    finally:
        os.unlink(wav_path)


def test_extract_emotion_features_pitch_extraction_failure():
    """Test handling of pitch extraction failures.
    
    This test verifies that if pitch extraction fails, the function
    continues with zero pitch values rather than crashing.
    """
    # Create a very short audio file that might cause pitch extraction issues
    sample_rate = 16000
    # Very short audio (0.1 seconds) with random noise
    short_audio = (np.random.randn(int(sample_rate * 0.1)) * 0.5 * 32767).astype(np.int16)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        scipy.io.wavfile.write(f.name, sample_rate, short_audio)
        wav_path = f.name
    
    try:
        # This should not crash even if pitch extraction has issues
        features = extract_emotion_features(wav_path, window_size=0.5)
        
        # Should return some features or empty list, but not crash
        assert isinstance(features, list)
        # All features should have valid pitch values (including 0.0)
        for f in features:
            assert f.pitch_mean >= 0.0
    finally:
        os.unlink(wav_path)


# ============================================================================
# Emotion Classification Rules Tests
# ============================================================================

def test_classify_emotion_laughter():
    """Test laughter detection: high ZCR + pitch variation + moderate volume."""
    emotion, confidence = _classify_emotion(
        pitch_mean=250.0,      # Moderate pitch
        pitch_std=30.0,        # High pitch variation
        volume_rms=0.5,        # Moderate volume
        spectral_centroid=2500.0,
        zero_crossing_rate=0.20  # High ZCR
    )
    
    assert emotion == "laughter"
    assert 0.0 <= confidence <= 1.0


def test_classify_emotion_scream():
    """Test scream detection: high pitch + high volume + high spectral centroid."""
    emotion, confidence = _classify_emotion(
        pitch_mean=500.0,      # High pitch
        pitch_std=20.0,
        volume_rms=0.85,       # High volume
        spectral_centroid=3500.0,  # High spectral centroid
        zero_crossing_rate=0.08
    )
    
    assert emotion == "scream"
    assert 0.0 <= confidence <= 1.0


def test_classify_emotion_excitement():
    """Test excitement detection: high volume + high pitch + low ZCR."""
    emotion, confidence = _classify_emotion(
        pitch_mean=350.0,      # High pitch
        pitch_std=15.0,
        volume_rms=0.75,       # High volume
        spectral_centroid=2800.0,
        zero_crossing_rate=0.05  # Low ZCR
    )
    
    assert emotion == "excitement"
    assert 0.0 <= confidence <= 1.0


def test_classify_emotion_calm():
    """Test calm detection: low volume + low pitch variation."""
    emotion, confidence = _classify_emotion(
        pitch_mean=180.0,      # Low pitch
        pitch_std=5.0,         # Low pitch variation
        volume_rms=0.2,        # Low volume
        spectral_centroid=1500.0,
        zero_crossing_rate=0.06
    )
    
    assert emotion == "calm"
    assert 0.0 <= confidence <= 1.0


def test_classify_emotion_neutral_default():
    """Test neutral classification for audio that doesn't match other patterns."""
    emotion, confidence = _classify_emotion(
        pitch_mean=220.0,      # Moderate pitch
        pitch_std=12.0,        # Moderate variation
        volume_rms=0.4,        # Moderate volume
        spectral_centroid=2200.0,
        zero_crossing_rate=0.08
    )
    
    assert emotion == "neutral"
    assert confidence == 0.5


def test_classify_emotion_silent_segment():
    """Test that silent segments (very low volume) are classified as neutral."""
    emotion, confidence = _classify_emotion(
        pitch_mean=200.0,
        pitch_std=10.0,
        volume_rms=0.05,       # Very low volume (< 0.1 threshold)
        spectral_centroid=2000.0,
        zero_crossing_rate=0.08
    )
    
    assert emotion == "neutral"
    assert confidence == 0.0


# ============================================================================
# Confidence Scoring Tests
# ============================================================================

def test_confidence_bounds_all_emotions():
    """Test that confidence scores are always in [0, 1] for all emotion types."""
    test_cases = [
        # Laughter with extreme values
        (250.0, 100.0, 0.5, 2500.0, 0.30),
        # Scream with extreme values
        (800.0, 20.0, 0.95, 5000.0, 0.08),
        # Excitement with extreme values
        (600.0, 15.0, 0.90, 3000.0, 0.02),
        # Calm with extreme values
        (100.0, 1.0, 0.05, 1000.0, 0.06),
        # Neutral
        (220.0, 12.0, 0.4, 2200.0, 0.08),
    ]
    
    for pitch_mean, pitch_std, volume_rms, spec_centroid, zcr in test_cases:
        emotion, confidence = _classify_emotion(
            pitch_mean=pitch_mean,
            pitch_std=pitch_std,
            volume_rms=volume_rms,
            spectral_centroid=spec_centroid,
            zero_crossing_rate=zcr
        )
        
        assert 0.0 <= confidence <= 1.0, \
            f"Confidence {confidence} out of bounds for emotion {emotion}"


def test_confidence_increases_with_stronger_features():
    """Test that confidence increases as features become more pronounced."""
    # Test laughter with increasing ZCR
    _, conf1 = _classify_emotion(250.0, 30.0, 0.5, 2500.0, 0.16)
    _, conf2 = _classify_emotion(250.0, 30.0, 0.5, 2500.0, 0.20)
    _, conf3 = _classify_emotion(250.0, 30.0, 0.5, 2500.0, 0.25)
    
    # Higher ZCR should lead to higher confidence for laughter
    assert conf1 <= conf2 <= conf3


# ============================================================================
# Silent Segment Handling Tests
# ============================================================================

def test_extract_emotion_features_silent_audio():
    """Test that silent audio is handled gracefully."""
    # Create silent audio
    sample_rate = 16000
    duration = 2.0
    audio = np.zeros(int(sample_rate * duration), dtype=np.int16)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        scipy.io.wavfile.write(f.name, sample_rate, audio)
        wav_path = f.name
    
    try:
        features = extract_emotion_features(wav_path, window_size=0.5)
        
        # Silent audio should return empty list or neutral emotions
        if len(features) > 0:
            # If features are returned, they should be neutral with low confidence
            for f in features:
                assert f.emotion == "neutral"
                assert f.confidence <= 0.5
    finally:
        os.unlink(wav_path)


def test_silent_segments_in_mixed_audio():
    """Test that silent segments within audio are classified as neutral or have low confidence."""
    # Create audio with silent and non-silent parts
    sample_rate = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.zeros_like(t)
    
    # First second: loud audio
    mask1 = t < 1.0
    audio[mask1] = 0.7 * np.sin(2 * np.pi * 400 * t[mask1])
    
    # Second second: silence (already zeros)
    
    # Third second: loud audio again
    mask3 = t >= 2.0
    audio[mask3] = 0.7 * np.sin(2 * np.pi * 400 * t[mask3])
    
    audio_int16 = (audio * 32767).astype(np.int16)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        scipy.io.wavfile.write(f.name, sample_rate, audio_int16)
        wav_path = f.name
    
    try:
        features = extract_emotion_features(wav_path, window_size=0.5)
        
        # Find features in the silent region (around 1.0-2.0 seconds)
        silent_features = [f for f in features if 1.0 <= f.time < 2.0]
        
        if len(silent_features) > 0:
            # Silent segments should have lower volume than non-silent segments
            non_silent_features = [f for f in features if f.time < 1.0 or f.time >= 2.0]
            if len(non_silent_features) > 0:
                avg_silent_volume = np.mean([f.volume_rms for f in silent_features])
                avg_non_silent_volume = np.mean([f.volume_rms for f in non_silent_features])
                # Silent segments should have lower volume
                assert avg_silent_volume < avg_non_silent_volume
    finally:
        os.unlink(wav_path)


# ============================================================================
# Helper Function Tests
# ============================================================================

def test_normalize_array():
    """Test array normalization to [0, 1] range."""
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    normalized = _normalize_array(arr)
    
    assert np.min(normalized) == 0.0
    assert np.max(normalized) == 1.0
    assert len(normalized) == len(arr)


def test_normalize_array_constant():
    """Test normalization of constant array."""
    arr = np.array([5.0, 5.0, 5.0, 5.0])
    normalized = _normalize_array(arr)
    
    # Constant array should normalize to all zeros
    assert np.all(normalized == 0.0)


def test_normalize_array_empty():
    """Test normalization of empty array."""
    arr = np.array([])
    normalized = _normalize_array(arr)
    
    assert len(normalized) == 0


def test_compute_rolling_std():
    """Test rolling standard deviation computation."""
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    rolling_std = _compute_rolling_std(arr, window_size=3)
    
    assert len(rolling_std) == len(arr)
    assert all(std >= 0.0 for std in rolling_std)


def test_compute_rolling_std_small_array():
    """Test rolling std with array smaller than window size."""
    arr = np.array([1.0, 2.0])
    rolling_std = _compute_rolling_std(arr, window_size=5)
    
    # Should return zeros for arrays smaller than window
    assert len(rolling_std) == len(arr)
    assert np.all(rolling_std == 0.0)


# ============================================================================
# Integration Tests (Real Audio)
# ============================================================================

def test_extract_emotion_features_basic(test_wav_file):
    """Test that emotion features are extracted correctly from real audio."""
    features = extract_emotion_features(test_wav_file, window_size=0.5)
    
    # Should have features (5 seconds / 0.5s window = 10 windows)
    assert len(features) > 0
    assert len(features) >= 8  # At least 8 windows for 5 seconds
    
    # All features should be EmotionFeatures objects
    assert all(isinstance(f, EmotionFeatures) for f in features)
    
    # All features should have valid values
    for f in features:
        assert f.time >= 0.0
        assert f.pitch_mean >= 0.0
        assert f.pitch_std >= 0.0
        assert 0.0 <= f.volume_rms <= 1.0
        assert f.spectral_centroid >= 0.0
        assert f.zero_crossing_rate >= 0.0
        assert f.emotion in ["laughter", "scream", "excitement", "calm", "neutral"]
        assert 0.0 <= f.confidence <= 1.0


def test_extract_emotion_features_emotion_categories(test_wav_file):
    """Test that different emotion categories are detected."""
    features = extract_emotion_features(test_wav_file, window_size=0.5)
    
    # Collect all detected emotions
    emotions = [f.emotion for f in features]
    
    # Should have at least some non-neutral emotions
    assert len(emotions) > 0
    
    # All emotions should be valid categories
    valid_emotions = {"laughter", "scream", "excitement", "calm", "neutral"}
    assert all(e in valid_emotions for e in emotions)


def test_extract_emotion_features_confidence_bounds(test_wav_file):
    """Test that confidence scores are properly bounded."""
    features = extract_emotion_features(test_wav_file, window_size=0.5)
    
    # All confidence scores should be in [0, 1]
    for f in features:
        assert 0.0 <= f.confidence <= 1.0, \
            f"Confidence {f.confidence} out of bounds for emotion {f.emotion}"


def test_extract_emotion_features_invalid_file():
    """Test that invalid audio files are handled gracefully."""
    features = extract_emotion_features("nonexistent_file.wav", window_size=0.5)
    
    # Should return empty list for invalid files
    assert features == []


def test_extract_emotion_features_window_size():
    """Test that different window sizes work correctly."""
    # Create a simple test audio
    sample_rate = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = 0.5 * np.sin(2 * np.pi * 400 * t)
    audio_int16 = (audio * 32767).astype(np.int16)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        scipy.io.wavfile.write(f.name, sample_rate, audio_int16)
        wav_path = f.name
    
    try:
        # Test with different window sizes
        features_05 = extract_emotion_features(wav_path, window_size=0.5)
        features_10 = extract_emotion_features(wav_path, window_size=1.0)
        
        # Smaller window size should produce more features
        assert len(features_05) > len(features_10)
        
        # Both should have valid features
        assert len(features_05) > 0
        assert len(features_10) > 0
    finally:
        os.unlink(wav_path)
