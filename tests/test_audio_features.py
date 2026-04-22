"""Tests for enhanced audio feature extraction."""

import numpy as np
import pytest
import scipy.io.wavfile
import tempfile
import os

from config import Config
from pipeline.scorer import compute_audio_features
from pipeline.models import AudioFeatures


@pytest.fixture
def test_wav_file():
    """Create a temporary WAV file with synthetic audio."""
    # Generate 5 seconds of audio at 16kHz
    sample_rate = 16000
    duration = 5.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create audio with varying volume and pitch
    # First 2 seconds: quiet low pitch (200 Hz)
    # Next 2 seconds: loud high pitch (800 Hz)
    # Last 1 second: silence
    audio = np.zeros_like(t)
    
    # Quiet low pitch (0-2s)
    mask1 = t < 2.0
    audio[mask1] = 0.1 * np.sin(2 * np.pi * 200 * t[mask1])
    
    # Loud high pitch (2-4s)
    mask2 = (t >= 2.0) & (t < 4.0)
    audio[mask2] = 0.8 * np.sin(2 * np.pi * 800 * t[mask2])
    
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


def test_compute_audio_features_basic(test_wav_file):
    """Test that audio features are extracted correctly."""
    config = Config(work_dir="/tmp")
    
    features = compute_audio_features(config, test_wav_file)
    
    # Should have features (5 seconds / 0.5s window = 10 windows)
    assert len(features) > 0
    assert len(features) >= 8  # At least 8 windows for 5 seconds
    
    # All features should be AudioFeatures objects
    assert all(isinstance(f, AudioFeatures) for f in features)
    
    # All scores should be in [0, 1]
    for f in features:
        assert 0.0 <= f.volume_score <= 1.0
        assert 0.0 <= f.pitch_score <= 1.0
        assert 0.0 <= f.excitement_score <= 1.0
        assert 0.0 <= f.silence_score <= 1.0
        assert f.time >= 0.0


def test_compute_audio_features_excitement_pattern(test_wav_file):
    """Test that excitement score increases with volume and pitch."""
    config = Config(work_dir="/tmp")
    
    features = compute_audio_features(config, test_wav_file)
    
    # Find features in different time ranges
    quiet_features = [f for f in features if 0.0 <= f.time < 2.0]
    loud_features = [f for f in features if 2.0 <= f.time < 4.0]
    silent_features = [f for f in features if f.time >= 4.0]
    
    # Loud section should have higher excitement than quiet section
    if loud_features and quiet_features:
        avg_loud_excitement = np.mean([f.excitement_score for f in loud_features])
        avg_quiet_excitement = np.mean([f.excitement_score for f in quiet_features])
        assert avg_loud_excitement > avg_quiet_excitement
    
    # Silent section should have high silence score
    if silent_features:
        avg_silence = np.mean([f.silence_score for f in silent_features])
        assert avg_silence > 0.5  # Should be mostly silent


def test_compute_audio_features_silence_score(test_wav_file):
    """Test that silence score is inverse of volume."""
    config = Config(work_dir="/tmp")
    
    features = compute_audio_features(config, test_wav_file)
    
    # For each feature, silence_score should be approximately 1.0 - volume_score
    for f in features:
        # Allow small tolerance due to normalization
        expected_silence = 1.0 - f.volume_score
        assert abs(f.silence_score - expected_silence) < 0.1


def test_compute_audio_features_empty_audio():
    """Test handling of silent/empty audio file."""
    config = Config(work_dir="/tmp")
    
    # Create silent audio file
    sample_rate = 16000
    duration = 2.0
    audio = np.zeros(int(sample_rate * duration), dtype=np.int16)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        scipy.io.wavfile.write(f.name, sample_rate, audio)
        wav_path = f.name
    
    try:
        features = compute_audio_features(config, wav_path)
        
        # Should return empty list for silent audio
        assert features == []
    finally:
        os.unlink(wav_path)


def test_compute_audio_features_custom_window_size():
    """Test that custom window size is respected."""
    config = Config(work_dir="/tmp", audio_feature_window=1.0)  # 1 second windows
    
    # Create 5 second audio file
    sample_rate = 16000
    duration = 5.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
    audio_int16 = (audio * 32767).astype(np.int16)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        scipy.io.wavfile.write(f.name, sample_rate, audio_int16)
        wav_path = f.name
    
    try:
        features = compute_audio_features(config, wav_path)
        
        # Should have ~5 windows (5 seconds / 1.0s window)
        assert 4 <= len(features) <= 6
        
        # Time stamps should be spaced by 1.0 second
        if len(features) >= 2:
            time_diffs = [features[i+1].time - features[i].time for i in range(len(features)-1)]
            assert all(0.9 <= diff <= 1.1 for diff in time_diffs)
    finally:
        os.unlink(wav_path)


def test_compute_audio_features_percentile_clipping():
    """Test that percentile clipping prevents outlier dominance."""
    config = Config(work_dir="/tmp")
    
    # Create audio with one extreme outlier
    sample_rate = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Mostly quiet audio
    audio = 0.1 * np.sin(2 * np.pi * 440 * t)
    
    # Add one very loud spike at 1.5 seconds
    spike_idx = int(1.5 * sample_rate)
    audio[spike_idx:spike_idx+100] = 1.0
    
    audio_int16 = (audio * 32767).astype(np.int16)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        scipy.io.wavfile.write(f.name, sample_rate, audio_int16)
        wav_path = f.name
    
    try:
        features = compute_audio_features(config, wav_path)
        
        # Without percentile clipping, the spike would make everything else near 0
        # With clipping, we should see reasonable distribution
        volume_scores = [f.volume_score for f in features]
        
        # Should have some variation (not all near 0 or 1)
        assert np.std(volume_scores) > 0.1
        
        # The spike should be detected (at least one value near 1.0)
        assert max(volume_scores) > 0.8
        
        # But not all values should be at extremes
        # At least some values should be in the lower range (not all maxed out)
        low_range_count = sum(1 for v in volume_scores if v < 0.5)
        assert low_range_count >= 1  # At least one value in lower range
    finally:
        os.unlink(wav_path)
