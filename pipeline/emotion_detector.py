"""Emotion detection module for audio analysis.

This module extracts audio features using librosa and classifies emotions
using heuristic rules based on pitch, volume, spectral centroid, and
zero-crossing rate.

Emotion categories:
- laughter: high ZCR + pitch variation + moderate volume
- scream: high pitch + high volume + high spectral centroid
- excitement: high volume + high pitch + low ZCR
- calm: low volume + low pitch variation
- neutral: default (doesn't match other patterns)
"""

import logging
from typing import Optional

import numpy as np

from pipeline.models import EmotionFeatures

logger = logging.getLogger(__name__)


def extract_emotion_features(
    wav_path: str,
    window_size: float = 0.5,
) -> list[EmotionFeatures]:
    """Extract emotion features using librosa.
    
    Strategy:
    1. Load audio with librosa
    2. Extract features per window:
       - Pitch: librosa.pyin() for F0 tracking
       - Volume: librosa.feature.rms()
       - Spectral centroid: librosa.feature.spectral_centroid()
       - ZCR: librosa.feature.zero_crossing_rate()
    3. Classify emotion using heuristic rules
    4. Assign confidence based on feature strength
    
    Args:
        wav_path: Path to audio file
        window_size: Window size in seconds (default 0.5)
    
    Returns:
        List of EmotionFeatures objects
    """
    try:
        import librosa
    except ImportError:
        logger.warning(
            "librosa not installed. Emotion detection disabled. "
            "Install with: pip install librosa"
        )
        return []
    
    # Load audio
    try:
        y, sr = librosa.load(wav_path, sr=None, mono=True)
    except Exception as exc:
        logger.error("Failed to load audio file %s: %s", wav_path, exc)
        return []
    
    # Safety check for silent or empty audio
    if len(y) == 0 or np.max(np.abs(y)) == 0.0:
        logger.warning("Audio file %s is silent or empty", wav_path)
        return []
    
    hop_length = int(sr * window_size)
    
    # Extract volume (RMS energy)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    
    # Extract pitch (F0 fundamental frequency)
    # Use a much smaller hop_length for pitch to avoid frame_length issues
    # pitch_hop_length must be significantly smaller than frame_length
    pitch_hop_length = min(512, len(y) // 20)  # Use 512 or smaller
    pitch_hop_length = max(256, pitch_hop_length)  # But at least 256
    # frame_length should be at least 2048 but not larger than audio length
    frame_length = min(2048, len(y))
    
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),  # ~65 Hz
            fmax=librosa.note_to_hz('C7'),  # ~2093 Hz
            sr=sr,
            hop_length=pitch_hop_length,
            frame_length=frame_length,
        )
        # Replace NaN with 0
        f0 = np.nan_to_num(f0, nan=0.0)
        
        # Resample f0 to match rms length if needed
        if len(f0) != len(rms):
            from scipy import interpolate
            if len(f0) > 1:
                x_old = np.linspace(0, 1, len(f0))
                x_new = np.linspace(0, 1, len(rms))
                f_interp = interpolate.interp1d(x_old, f0, kind='linear', fill_value='extrapolate')
                f0 = f_interp(x_new)
            else:
                f0 = np.zeros(len(rms))
    except Exception as exc:
        logger.warning("Pitch extraction failed for %s: %s. Using zero pitch.", wav_path, exc)
        f0 = np.zeros(len(rms))
    
    # Extract spectral centroid (brightness)
    spectral_centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, hop_length=hop_length
    )[0]
    
    # Extract zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)[0]
    
    # Ensure all arrays have the same length
    min_len = min(len(rms), len(f0), len(spectral_centroid), len(zcr))
    rms = rms[:min_len]
    f0 = f0[:min_len]
    spectral_centroid = spectral_centroid[:min_len]
    zcr = zcr[:min_len]
    
    # Safety check
    if min_len == 0:
        logger.warning("No audio features extracted from %s", wav_path)
        return []
    
    # Normalize volume to [0, 1]
    volume_norm = _normalize_array(rms)
    
    # Compute pitch statistics for variation detection
    # Use rolling window to compute pitch std
    pitch_std = _compute_rolling_std(f0, window_size=5)
    
    # Build EmotionFeatures objects
    features: list[EmotionFeatures] = []
    for i in range(min_len):
        time = i * window_size
        
        # Extract raw features
        pitch_mean = float(f0[i])
        pitch_variation = float(pitch_std[i])
        volume_rms = float(volume_norm[i])
        spec_centroid = float(spectral_centroid[i])
        zero_cross_rate = float(zcr[i])
        
        # Classify emotion and compute confidence
        emotion, confidence = _classify_emotion(
            pitch_mean=pitch_mean,
            pitch_std=pitch_variation,
            volume_rms=volume_rms,
            spectral_centroid=spec_centroid,
            zero_crossing_rate=zero_cross_rate,
        )
        
        features.append(EmotionFeatures(
            time=time,
            pitch_mean=pitch_mean,
            pitch_std=pitch_variation,
            volume_rms=volume_rms,
            spectral_centroid=spec_centroid,
            zero_crossing_rate=zero_cross_rate,
            emotion=emotion,
            confidence=confidence,
        ))
    
    logger.info(
        "Extracted %d emotion feature windows (%.1fs resolution) from %s",
        len(features), window_size, wav_path
    )
    
    return features


def _normalize_array(arr: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 1] range."""
    if len(arr) == 0:
        return arr
    
    min_val = np.min(arr)
    max_val = np.max(arr)
    
    if max_val == min_val:
        return np.zeros_like(arr)
    
    return (arr - min_val) / (max_val - min_val)


def _compute_rolling_std(arr: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Compute rolling standard deviation for pitch variation detection."""
    if len(arr) < window_size:
        return np.zeros_like(arr)
    
    result = np.zeros_like(arr)
    for i in range(len(arr)):
        start = max(0, i - window_size // 2)
        end = min(len(arr), i + window_size // 2 + 1)
        result[i] = np.std(arr[start:end])
    
    return result


def _classify_emotion(
    pitch_mean: float,
    pitch_std: float,
    volume_rms: float,
    spectral_centroid: float,
    zero_crossing_rate: float,
) -> tuple[str, float]:
    """Classify emotion using heuristic rules.
    
    Classification rules (from design document):
    - Laughter: high ZCR + pitch variation + moderate volume
    - Scream: high pitch + high volume + high spectral centroid
    - Excitement: high volume + high pitch + low ZCR
    - Calm: low volume + low pitch variation
    - Neutral: default
    
    Args:
        pitch_mean: Mean pitch in Hz
        pitch_std: Pitch variation (std dev)
        volume_rms: Normalized volume [0, 1]
        spectral_centroid: Spectral centroid in Hz
        zero_crossing_rate: Zero-crossing rate
    
    Returns:
        Tuple of (emotion_label, confidence_score)
    """
    # Handle silent segments
    if volume_rms < 0.1:
        return ("neutral", 0.0)
    
    # Laughter detection: high ZCR + pitch variation + moderate volume
    if (zero_crossing_rate > 0.15 and 
        pitch_std > 20.0 and 
        0.3 <= volume_rms <= 0.7):
        confidence = min(1.0, (zero_crossing_rate - 0.15) / 0.1 + pitch_std / 50.0)
        return ("laughter", min(1.0, confidence))
    
    # Scream detection: high pitch + high volume + high spectral centroid
    if (pitch_mean > 400.0 and 
        volume_rms > 0.7 and 
        spectral_centroid > 3000.0):
        confidence = min(1.0, (pitch_mean - 400.0) / 200.0 + (volume_rms - 0.7) / 0.3)
        return ("scream", min(1.0, confidence))
    
    # Excitement detection: high volume + high pitch + low ZCR
    if (volume_rms > 0.6 and 
        pitch_mean > 300.0 and 
        zero_crossing_rate < 0.1):
        confidence = min(1.0, (volume_rms - 0.6) / 0.4 + (pitch_mean - 300.0) / 200.0)
        return ("excitement", min(1.0, confidence))
    
    # Calm detection: low volume + low pitch variation
    if volume_rms < 0.3 and pitch_std < 10.0:
        confidence = min(1.0, (0.3 - volume_rms) / 0.3 + (10.0 - pitch_std) / 10.0)
        return ("calm", min(1.0, confidence))
    
    # Default: neutral
    return ("neutral", 0.5)
