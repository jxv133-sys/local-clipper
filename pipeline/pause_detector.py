"""Natural pause detection for clip boundary refinement.

This module detects natural pause points in video transcripts and audio to help
refine clip boundaries. It identifies three types of pauses:
1. Punctuation pauses: sentence boundaries (., !, ?)
2. Silence pauses: gaps between transcript segments (>0.5s)
3. Breath pauses: low-energy moments within segments (RMS < 10% of mean)

**Validates: Requirements 5.1, 5.3, 5.4, 5.6**

Example usage:
    >>> from pipeline.models import Transcript
    >>> from pipeline.pause_detector import detect_natural_pauses, snap_to_nearest_pause
    >>> 
    >>> # Detect all natural pauses in a transcript
    >>> pauses = detect_natural_pauses(transcript, "audio.wav", silence_threshold=0.5)
    >>> 
    >>> # Snap a clip boundary to the nearest pause
    >>> refined_end = snap_to_nearest_pause(clip_end, pauses, max_distance=3.0)
"""

import logging
import re
from typing import Optional

import numpy as np

from pipeline.models import NaturalPause, Transcript

logger = logging.getLogger(__name__)


def detect_natural_pauses(
    transcript: Transcript,
    wav_path: str,
    silence_threshold: float = 0.5,
) -> list[NaturalPause]:
    """Detect natural pause points from transcript and audio.
    
    Strategy:
    1. Punctuation pauses: Find '.', '!', '?' in transcript
    2. Silence pauses: Detect gaps > silence_threshold between segments
    3. Breath pauses: Detect short silence within segments (RMS < 10% of mean)
    4. Assign confidence: punctuation=0.9, silence=0.8, breath=0.6
    
    Args:
        transcript: Whisper transcript with segments and timing
        wav_path: Path to audio WAV file for breath detection
        silence_threshold: Minimum gap duration (seconds) to consider as silence pause
    
    Returns:
        Sorted list of NaturalPause objects
        
    **Validates: Requirements 5.1, 5.3, 5.4**
    """
    pauses: list[NaturalPause] = []
    
    # 1. Detect punctuation pauses
    pauses.extend(_detect_punctuation_pauses(transcript))
    
    # 2. Detect silence gaps between segments
    pauses.extend(_detect_silence_pauses(transcript, silence_threshold))
    
    # 3. Detect breath pauses within segments
    pauses.extend(_detect_breath_pauses(transcript, wav_path))
    
    # Sort by time
    pauses.sort(key=lambda p: p.time)
    
    logger.debug(
        "Detected %d natural pauses: %d punctuation, %d silence, %d breath",
        len(pauses),
        sum(1 for p in pauses if p.type == "punctuation"),
        sum(1 for p in pauses if p.type == "silence"),
        sum(1 for p in pauses if p.type == "breath"),
    )
    
    return pauses


def snap_to_nearest_pause(
    time: float,
    pauses: list[NaturalPause],
    max_distance: float = 3.0,
) -> float:
    """Snap a timestamp to the nearest natural pause within max_distance.
    
    Args:
        time: Original timestamp (seconds)
        pauses: List of detected natural pauses
        max_distance: Maximum distance (seconds) to search for a pause
    
    Returns:
        Adjusted timestamp, or original if no pause found within max_distance
        
    **Validates: Requirement 5.6**
    """
    if not pauses:
        return time
    
    # Find nearest pause
    nearest_pause: Optional[NaturalPause] = None
    min_distance = float('inf')
    
    for pause in pauses:
        distance = abs(pause.time - time)
        if distance < min_distance:
            min_distance = distance
            nearest_pause = pause
    
    # Snap if within max_distance
    if nearest_pause and min_distance <= max_distance:
        logger.debug(
            "Snapped time %.2f to pause at %.2f (type=%s, distance=%.2f)",
            time,
            nearest_pause.time,
            nearest_pause.type,
            min_distance,
        )
        return nearest_pause.time
    
    return time


def _detect_punctuation_pauses(transcript: Transcript) -> list[NaturalPause]:
    """Detect pauses at sentence-ending punctuation marks.
    
    Searches for '.', '!', '?' in transcript text and creates pause markers
    at the end time of the segment containing the punctuation.
    
    Args:
        transcript: Whisper transcript with segments
    
    Returns:
        List of NaturalPause objects with type="punctuation" and confidence=0.9
    """
    pauses: list[NaturalPause] = []
    
    # Regex to find sentence-ending punctuation
    # Matches '.', '!', '?' not followed by more alphanumeric (to avoid abbreviations)
    punctuation_pattern = re.compile(r'[.!?](?=\s|$)')
    
    for segment in transcript.segments:
        text = segment.text.strip()
        
        # Find all punctuation matches
        for match in punctuation_pattern.finditer(text):
            # Get context around the punctuation (up to 30 chars before and after)
            start_idx = max(0, match.start() - 30)
            end_idx = min(len(text), match.end() + 30)
            context = text[start_idx:end_idx].strip()
            
            # Create pause at segment end time
            # (Whisper segments typically end at punctuation)
            pause = NaturalPause(
                time=segment.end,
                type="punctuation",
                confidence=0.9,
                context=context,
            )
            pauses.append(pause)
    
    return pauses


def _detect_silence_pauses(
    transcript: Transcript,
    silence_threshold: float,
) -> list[NaturalPause]:
    """Detect pauses at silence gaps between transcript segments.
    
    Identifies gaps between consecutive segments that exceed the silence_threshold.
    
    Args:
        transcript: Whisper transcript with segments
        silence_threshold: Minimum gap duration (seconds) to consider as silence
    
    Returns:
        List of NaturalPause objects with type="silence" and confidence=0.8
    """
    pauses: list[NaturalPause] = []
    
    segments = transcript.segments
    for i in range(len(segments) - 1):
        current_seg = segments[i]
        next_seg = segments[i + 1]
        
        # Calculate gap between segments
        gap = next_seg.start - current_seg.end
        
        if gap >= silence_threshold:
            # Place pause at midpoint of the gap
            pause_time = current_seg.end + (gap / 2.0)
            
            # Create context from surrounding segments
            context = f"...{current_seg.text[-20:]} [silence {gap:.1f}s] {next_seg.text[:20]}..."
            
            pause = NaturalPause(
                time=pause_time,
                type="silence",
                confidence=0.8,
                context=context.strip(),
            )
            pauses.append(pause)
    
    return pauses


def _detect_breath_pauses(
    transcript: Transcript,
    wav_path: str,
) -> list[NaturalPause]:
    """Detect breath pauses within segments using audio RMS energy.
    
    Analyzes audio energy within each segment to find brief low-energy moments
    that indicate natural breath pauses (RMS < 10% of segment mean).
    
    Args:
        transcript: Whisper transcript with segments
        wav_path: Path to audio WAV file
    
    Returns:
        List of NaturalPause objects with type="breath" and confidence=0.6
    """
    pauses: list[NaturalPause] = []
    
    try:
        import librosa
    except ImportError:
        logger.warning(
            "librosa not installed. Breath pause detection disabled. "
            "Install with: pip install librosa"
        )
        return pauses
    
    try:
        # Load audio
        y, sr = librosa.load(wav_path, sr=None, mono=True)
    except Exception as exc:
        logger.warning("Failed to load audio for breath detection: %s", exc)
        return pauses
    
    # Safety check
    if len(y) == 0 or np.max(np.abs(y)) == 0.0:
        logger.warning("Audio is silent or empty, skipping breath detection")
        return pauses
    
    # Analyze each segment
    for segment in transcript.segments:
        # Skip very short segments (< 2 seconds)
        if segment.end - segment.start < 2.0:
            continue
        
        # Extract audio for this segment
        start_sample = int(segment.start * sr)
        end_sample = int(segment.end * sr)
        
        # Bounds check
        if start_sample >= len(y) or end_sample > len(y):
            continue
        
        segment_audio = y[start_sample:end_sample]
        
        if len(segment_audio) == 0:
            continue
        
        # Compute RMS energy in small windows (0.1s)
        window_size = int(0.1 * sr)
        hop_length = window_size // 2
        
        rms = librosa.feature.rms(
            y=segment_audio,
            frame_length=window_size,
            hop_length=hop_length,
        )[0]
        
        if len(rms) == 0:
            continue
        
        # Calculate mean RMS for this segment
        mean_rms = np.mean(rms)
        
        if mean_rms == 0.0:
            continue
        
        # Find windows with RMS < 10% of mean
        threshold = 0.1 * mean_rms
        low_energy_indices = np.where(rms < threshold)[0]
        
        # Group consecutive low-energy windows
        if len(low_energy_indices) > 0:
            # Find groups of consecutive indices
            groups = _group_consecutive(low_energy_indices)
            
            for group in groups:
                # Only consider groups of at least 2 windows (0.1s duration)
                if len(group) >= 2:
                    # Calculate time of pause (midpoint of group)
                    first_idx = group[0]
                    last_idx = group[-1]
                    
                    # Convert frame indices to time
                    pause_frame = (first_idx + last_idx) // 2
                    pause_time_in_segment = (pause_frame * hop_length) / sr
                    pause_time = segment.start + pause_time_in_segment
                    
                    # Get context from segment text
                    context = segment.text[:50] + "..." if len(segment.text) > 50 else segment.text
                    
                    pause = NaturalPause(
                        time=pause_time,
                        type="breath",
                        confidence=0.6,
                        context=context,
                    )
                    pauses.append(pause)
    
    return pauses


def _group_consecutive(indices: np.ndarray) -> list[list[int]]:
    """Group consecutive integers into sublists.
    
    Args:
        indices: Array of integers
    
    Returns:
        List of lists, where each sublist contains consecutive integers
    
    Example:
        >>> _group_consecutive(np.array([1, 2, 3, 5, 6, 8]))
        [[1, 2, 3], [5, 6], [8]]
    """
    if len(indices) == 0:
        return []
    
    groups: list[list[int]] = []
    current_group = [int(indices[0])]
    
    for i in range(1, len(indices)):
        if indices[i] == indices[i - 1] + 1:
            current_group.append(int(indices[i]))
        else:
            groups.append(current_group)
            current_group = [int(indices[i])]
    
    # Add the last group
    groups.append(current_group)
    
    return groups
