"""Scorer module for the video highlight generator pipeline.

Two-phase scoring:
  Phase 1 (fast, local): text + audio scores on every segment.
  Phase 2 (LLM, optional): the top `llm_max_candidates` candidate *windows*
      (each ~30s of transcript) are sent to the LLM for quality refinement.
      The LLM never sees individual 3-second Whisper segments — it sees the
      full clip window so it can judge whether the moment is actually interesting.
"""

from __future__ import annotations

import logging
import re
import time
from typing import TYPE_CHECKING, Callable

import numpy as np
import requests
import scipy.io.wavfile

from config import Config
from pipeline.exceptions import LLMScoringError
from pipeline.models import LLMMetadata, ScoredSegment, Segment, Transcript

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Video summary generation (cached)
# ---------------------------------------------------------------------------

# Module-level cache for video summaries, keyed by video path
_video_summary_cache: dict[str, str] = {}


def generate_video_summary(config: Config, transcript: Transcript, video_path: str = "") -> str:
    """Generate a 2-3 sentence video summary for LLM context.
    
    Strategy:
    1. Sample transcript: take every Nth segment (N = len(segments) // sample_rate)
    2. Build condensed transcript (max 500 words)
    3. Send to LLM with summary prompt
    4. Cache result in module-level variable
    
    Args:
        config: Configuration object with LLM settings
        transcript: Full video transcript
        video_path: Optional path to video file (used as cache key)
    
    Returns:
        Summary string (e.g., "This is a Minecraft survival stream where the 
        creator builds a base while fighting mobs. High energy with frequent 
        reactions to unexpected events.")
    
    **Validates: Requirements 2.1, 2.2, 2.3, 2.6**
    """
    # Check cache first
    if video_path and video_path in _video_summary_cache:
        logger.debug("[Scorer] Using cached video summary for %s", video_path)
        return _video_summary_cache[video_path]
    
    segments = transcript.segments
    if not segments:
        summary = "Empty video with no transcript content."
        if video_path:
            _video_summary_cache[video_path] = summary
        return summary
    
    # Sample rate: aim for ~sample_rate samples across the video
    # Calculate step size: how many segments to skip between samples
    step_size = max(1, len(segments) // config.video_summary_sample_rate)
    
    # Sample every Nth segment
    sampled_segments = segments[::step_size]
    
    # Build condensed transcript (max 500 words)
    condensed_lines: list[str] = []
    word_count = 0
    max_words = 500
    
    for seg in sampled_segments:
        text = seg.text.strip()
        if not text:
            continue
        
        # Count words in this segment
        seg_words = len(text.split())
        if word_count + seg_words > max_words:
            # Add partial segment to reach word limit
            remaining_words = max_words - word_count
            if remaining_words > 0:
                words = text.split()[:remaining_words]
                condensed_lines.append(f"[{seg.start:.0f}s] {' '.join(words)}...")
            break
        
        condensed_lines.append(f"[{seg.start:.0f}s] {text}")
        word_count += seg_words
    
    condensed_transcript = "\n".join(condensed_lines)
    
    # Calculate video duration
    duration_minutes = segments[-1].end / 60.0 if segments else 0.0
    
    # Build LLM prompt
    prompt = (
        "You are analyzing a video transcript to provide context for clip selection.\n\n"
        f"Below is a condensed transcript (sampled segments from a {duration_minutes:.1f}-minute video):\n\n"
        f"{condensed_transcript}\n\n"
        "Provide a 2-3 sentence summary covering:\n"
        "1. Content type (gaming, podcast, comedy, vlog, educational, etc.)\n"
        "2. Main topics or activities\n"
        "3. Overall energy level (high-energy, calm, moderate)\n"
        "4. Key recurring themes or patterns\n\n"
        "Summary:"
    )
    
    # Call LLM
    try:
        raw_response = _call_llm(config, prompt)
        # Extract summary (remove any "Summary:" prefix if LLM echoes it)
        summary = raw_response.strip()
        if summary.lower().startswith("summary:"):
            summary = summary[8:].strip()
        
        # Fallback if LLM returns empty response
        if not summary:
            summary = (
                f"Video transcript with {len(segments)} segments spanning "
                f"{duration_minutes:.1f} minutes. Content type and themes unclear."
            )
            logger.warning("[Scorer] LLM returned empty video summary, using fallback")
        
        logger.info("[Scorer] Generated video summary: %r", summary[:100] + "..." if len(summary) > 100 else summary)
        
        # Cache the result
        if video_path:
            _video_summary_cache[video_path] = summary
        
        return summary
        
    except LLMScoringError as exc:
        logger.warning("[Scorer] Failed to generate video summary: %s", exc)
        summary = (
            f"Video transcript with {len(segments)} segments spanning "
            f"{duration_minutes:.1f} minutes. LLM summary unavailable."
        )
        if video_path:
            _video_summary_cache[video_path] = summary
        return summary


# ---------------------------------------------------------------------------
# Phase 1: text scoring
# ---------------------------------------------------------------------------

def compute_text_score(config: Config, segment: Segment) -> float:
    """Compute a normalized text score for a single segment.

    Scoring components:
    - +2.0 per keyword occurrence (case-insensitive)
    - +phrase_weight (default 4.0) per phrase keyword match (multi-word phrases)
    - +reaction_weight (default 3.0) per reaction keyword occurrence
      (whole-word match, case-insensitive)
    - +0.02 per character in segment.text (rewards longer speech)
    - +1.5 per '!' in segment.text
    - +1.0 per '?' in segment.text
    - pace bonus for fast speech (> 3 words/sec), indicating excitement/urgency

    Normalized to [0.0, 1.0] via soft-max: raw / (raw + 5.0).

    When config.text_pattern_weight > 0, the heuristic pattern score from
    text_patterns.analyze_text_patterns() is blended in:
        final = (1 - w) * keyword_score + w * pattern_score
    """
    from pipeline.text_patterns import analyze_text_patterns
    from pipeline.phrase_detector import detect_phrases

    text = segment.text
    text_lower = text.lower()
    raw_score = 0.0

    for keyword in config.keywords:
        kw = keyword.lower()
        pos = 0
        while True:
            idx = text_lower.find(kw, pos)
            if idx == -1:
                break
            raw_score += 2.0
            pos = idx + len(kw)

    # Phrase detection: multi-word keyword matching
    phrase_matches = detect_phrases(text, config.phrase_keywords)
    for phrase, start, end in phrase_matches:
        raw_score += config.phrase_weight  # Default: 4.0 (higher than single keyword)
        logger.debug(
            "[Scorer] Phrase detected at %.1fs: %r (weight=%.1f)",
            segment.start, phrase, config.phrase_weight
        )

    # Reaction keywords: whole-word, case-insensitive matching
    for rk in config.reaction_keywords:
        pattern = r'(?<!\w)' + re.escape(rk.lower()) + r'(?!\w)'
        count = len(re.findall(pattern, text_lower))
        raw_score += config.reaction_weight * count

    raw_score += len(text) * 0.02
    raw_score += text.count("!") * 1.5
    raw_score += text.count("?") * 1.0

    duration = segment.end - segment.start
    if duration > 0.0:
        wps = len(text.split()) / duration
        if wps > 3.0:
            raw_score += min(1.0, (wps - 3.0) / 3.0) * 2.0

    if raw_score <= 0.0:
        keyword_score = 0.0
    else:
        keyword_score = max(0.0, min(1.0, raw_score / (raw_score + 5.0)))

    # Repetition penalty: detect Whisper hallucinations and genuinely repetitive content.
    # Single-word segments cannot be repetitive, so skip them.
    words = text.lower().split()
    total_words = len(words)
    if total_words > 1:
        unique_words = len(set(words))
        repetition_ratio = unique_words / total_words
        if repetition_ratio < config.repetition_penalty_threshold:
            logger.debug(
                "[Scorer] Repetition penalty applied at %.1fs (ratio=%.2f)",
                segment.start, repetition_ratio,
            )
            keyword_score *= config.repetition_penalty_multiplier

    # Blend in heuristic pattern score (questions, laughter, emotional words, etc.)
    w = getattr(config, 'text_pattern_weight', 0.0)
    if w > 0.0:
        pattern_result = analyze_text_patterns(text)
        normalized = (1.0 - w) * keyword_score + w * pattern_result.score
        if pattern_result.signals:
            logger.debug(
                "[Scorer] Text patterns at %.1fs: %s (pattern=%.2f keyword=%.2f blended=%.2f)",
                segment.start, pattern_result.signals,
                pattern_result.score, keyword_score, normalized,
            )
        return normalized

    return keyword_score


# ---------------------------------------------------------------------------
# Phase 1: audio scoring
# ---------------------------------------------------------------------------

def compute_audio_score(segments: list[Segment], wav_path: str) -> list[float]:
    """Compute normalized RMS energy scores for all segments in one WAV pass."""
    _, raw_rms = compute_audio_score_with_raw(segments, wav_path)
    max_rms = max(raw_rms) if raw_rms else 0.0
    if max_rms == 0.0:
        return [0.0] * len(segments)
    return [v / max_rms for v in raw_rms]


def compute_audio_score_with_raw(
    segments: list[Segment],
    wav_path: str,
    wav_data: tuple[int, np.ndarray] | None = None,
) -> tuple[list[float], list[float]]:
    """Return (normalized_scores, raw_rms_values) for all segments.

    normalized_scores: each value in [0.0, 1.0] relative to the loudest segment.
    raw_rms_values: absolute RMS amplitude in [0.0, ~1.0] for float32 audio.

    If wav_data is provided as (sample_rate, data), it is used directly and
    wav_path is not read from disk (avoids duplicate I/O).
    """
    if wav_data is not None:
        sample_rate, data = wav_data
    else:
        sample_rate, data = scipy.io.wavfile.read(wav_path)

    if data.ndim == 2:
        data = data.mean(axis=1)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    else:
        data = data.astype(np.float32)

    total_samples = len(data)
    raw_rms: list[float] = []

    for seg in segments:
        s = max(0, min(int(seg.start * sample_rate), total_samples))
        e = max(0, min(int(seg.end * sample_rate), total_samples))
        chunk = data[s:e]
        raw_rms.append(float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) > 0 else 0.0)

    max_rms = max(raw_rms) if raw_rms else 0.0
    if max_rms == 0.0:
        return [0.0] * len(segments), raw_rms
    normalized = [v / max_rms for v in raw_rms]
    return normalized, raw_rms


# ---------------------------------------------------------------------------
# Enhanced audio feature extraction (new scoring system)
# ---------------------------------------------------------------------------

def compute_audio_features(config: Config, wav_path: str) -> list:
    """Extract enhanced audio features for viral clip detection.
    
    Extracts features at config.audio_feature_window (default 0.5s) resolution:
    - volume_score: RMS energy (normalized 0-1)
    - pitch_score: F0 fundamental frequency (normalized 0-1)
    - excitement_score: weighted combination (0.6 × volume + 0.4 × pitch)
    - silence_score: 1.0 - volume (detects pauses)
    
    Uses percentile clipping (5th-95th) to prevent outliers from skewing scores.
    Includes safety check for silent audio to prevent NaN values.
    
    Returns:
        List of AudioFeatures objects, one per temporal window
    """
    from pipeline.models import AudioFeatures
    
    try:
        import librosa
    except ImportError:
        logger.warning(
            "librosa not installed. Enhanced audio features disabled. "
            "Install with: pip install librosa"
        )
        return []
    
    # Load audio with librosa for pitch extraction
    try:
        y, sr = librosa.load(wav_path, sr=None, mono=True)
    except Exception as exc:
        logger.error("Failed to load audio file %s: %s", wav_path, exc)
        return []
    
    # Safety check for silent audio
    if len(y) == 0 or np.max(np.abs(y)) == 0.0:
        logger.warning("Audio file %s is silent or empty", wav_path)
        return []
    
    window_size = config.audio_feature_window
    hop_length = int(sr * window_size)
    
    # Extract volume (RMS energy) per window
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    
    # Extract pitch (F0 fundamental frequency) per window
    # Using pyin algorithm for better pitch tracking
    # Adjust parameters based on audio duration to avoid frame_length issues
    audio_duration = len(y) / sr
    
    # Use smaller hop_length for pitch to get better resolution
    pitch_hop_length = max(512, min(hop_length, len(y) // 10))
    
    # frame_length should be at least 2048 but not larger than audio length
    frame_length = min(2048, len(y))
    
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),  # ~65 Hz (low male voice)
            fmax=librosa.note_to_hz('C7'),  # ~2093 Hz (high female voice)
            sr=sr,
            hop_length=pitch_hop_length,
            frame_length=frame_length,
        )
        
        # Replace NaN pitch values (unvoiced segments) with 0
        f0 = np.nan_to_num(f0, nan=0.0)
        
        # If pitch was extracted at different resolution, resample to match RMS
        if len(f0) != len(rms):
            # Resample f0 to match rms length
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
    
    # Ensure both arrays have the same length
    min_len = min(len(rms), len(f0))
    rms = rms[:min_len]
    f0 = f0[:min_len]
    
    # Safety check: if all values are zero, return empty list
    if np.max(rms) == 0.0 and np.max(f0) == 0.0:
        logger.warning("All audio features are zero for %s", wav_path)
        return []
    
    # Normalize using percentile clipping to prevent outliers
    def normalize_with_percentiles(values: np.ndarray) -> np.ndarray:
        """Normalize array to [0, 1] using percentile clipping."""
        if len(values) == 0 or np.max(values) == np.min(values):
            return np.zeros_like(values)
        
        p_low = np.percentile(values, config.audio_percentile_low)
        p_high = np.percentile(values, config.audio_percentile_high)
        
        # Avoid division by zero
        if p_high == p_low:
            return np.zeros_like(values)
        
        clipped = np.clip(values, p_low, p_high)
        normalized = (clipped - p_low) / (p_high - p_low)
        return np.clip(normalized, 0.0, 1.0)
    
    # Normalize volume and pitch
    volume_norm = normalize_with_percentiles(rms)
    pitch_norm = normalize_with_percentiles(f0)
    
    # Compute excitement score: weighted combination, clamped to [0, 1]
    excitement = np.clip(
        config.excitement_volume_weight * volume_norm +
        config.excitement_pitch_weight * pitch_norm,
        0.0, 1.0,
    )
    
    # Compute silence score: inverse of volume
    silence = 1.0 - volume_norm
    
    # Build AudioFeatures objects
    features: list[AudioFeatures] = []
    for i in range(len(volume_norm)):
        time = i * window_size
        features.append(AudioFeatures(
            time=time,
            volume_score=float(volume_norm[i]),
            pitch_score=float(pitch_norm[i]),
            excitement_score=float(excitement[i]),
            silence_score=float(silence[i]),
        ))
    
    logger.info(
        "Extracted %d audio feature windows (%.1fs resolution) from %s",
        len(features), window_size, wav_path
    )
    
    return features


# ---------------------------------------------------------------------------
# Phase 1: spike score (audio burst detection)
# ---------------------------------------------------------------------------

_SPIKE_BASELINE_WINDOW_SECONDS = 30.0  # rolling baseline window before each segment
_SPIKE_STRONG_RATIO = 3.0              # ratio at which spike_score saturates near 1.0


def compute_spike_score(
    segments: list[Segment],
    wav_path: str,
    wav_data: tuple[int, np.ndarray] | None = None,
) -> list[float]:
    """Compute a spike score for each segment based on sudden audio energy bursts.

    Algorithm:
    - For each segment, compute the RMS of the audio within the segment.
    - Compute a rolling baseline RMS over the 30 seconds immediately before
      the segment start.
    - spike_ratio = segment_rms / baseline_rms
    - Normalize to [0.0, 1.0]: ratio >= 3x → score near 1.0; ratio <= 1x → near 0.0.
      Uses a linear ramp: score = clamp((ratio - 1.0) / (SPIKE_STRONG_RATIO - 1.0), 0, 1).
    - If the baseline is silent (baseline_rms == 0) and the segment has audio,
      the spike_score is 1.0 (silence-then-burst is a strong signal).
    - If both baseline and segment are silent, spike_score is 0.0.

    If wav_data is provided as (sample_rate, data), it is used directly and
    wav_path is not read from disk (avoids duplicate I/O).

    Returns a list of floats in [0.0, 1.0], one per segment.
    """
    if not segments:
        return []

    if wav_data is not None:
        sample_rate, data = wav_data
    else:
        sample_rate, data = scipy.io.wavfile.read(wav_path)

    if data.ndim == 2:
        data = data.mean(axis=1)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    else:
        data = data.astype(np.float32)

    total_samples = len(data)

    def _rms(start_sec: float, end_sec: float) -> float:
        s = max(0, min(int(start_sec * sample_rate), total_samples))
        e = max(0, min(int(end_sec * sample_rate), total_samples))
        chunk = data[s:e]
        if len(chunk) == 0:
            return 0.0
        return float(np.sqrt(np.mean(chunk ** 2)))

    spike_scores: list[float] = []
    for seg in segments:
        seg_rms = _rms(seg.start, seg.end)

        # Rolling baseline: the 30s window immediately before segment start
        baseline_start = max(0.0, seg.start - _SPIKE_BASELINE_WINDOW_SECONDS)
        baseline_rms = _rms(baseline_start, seg.start)

        if baseline_rms == 0.0:
            # Silence before the segment — any audio is a burst
            score = 1.0 if seg_rms > 0.0 else 0.0
        else:
            ratio = seg_rms / baseline_rms
            # Linear ramp: 1x → 0.0, 3x → 1.0, clamped
            score = max(0.0, min(1.0, (ratio - 1.0) / (_SPIKE_STRONG_RATIO - 1.0)))

        spike_scores.append(score)

    return spike_scores


# ---------------------------------------------------------------------------
# Phase 1: burst score (silence-then-burst detection)
# ---------------------------------------------------------------------------

_BURST_PRE_WINDOW_SECONDS = 5.0  # seconds of audio to examine before segment start


def compute_burst_score(
    segments: list[Segment],
    wav_path: str,
    wav_data: tuple[int, np.ndarray] | None = None,
) -> list[float]:
    """Detect the "silence → loud" transition pattern for each segment.

    Algorithm:
    1. First pass: compute global_rms_mean and global_rms_max across all segments.
    2. For each segment:
       a. Compute silence_before = avg RMS of the 5s window immediately before
          segment start.
       b. Compute segment_rms = RMS of the segment itself.
       c. If silence_before < 0.1 * global_rms_mean AND
             segment_rms > 0.5 * global_rms_max:
          burst_score = 1.0  (binary: either it's a burst or it's not)
       d. Otherwise: burst_score = 0.0
    3. Log each detected burst at INFO level.

    If wav_data is provided as (sample_rate, data), it is used directly and
    wav_path is not read from disk (avoids duplicate I/O).

    Returns a list of floats (each 0.0 or 1.0), one per segment.
    """
    if not segments:
        return []

    if wav_data is not None:
        sample_rate, data = wav_data
    else:
        sample_rate, data = scipy.io.wavfile.read(wav_path)

    if data.ndim == 2:
        data = data.mean(axis=1)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    else:
        data = data.astype(np.float32)

    total_samples = len(data)

    def _rms(start_sec: float, end_sec: float) -> float:
        s = max(0, min(int(start_sec * sample_rate), total_samples))
        e = max(0, min(int(end_sec * sample_rate), total_samples))
        chunk = data[s:e]
        if len(chunk) == 0:
            return 0.0
        return float(np.sqrt(np.mean(chunk ** 2)))

    # First pass: compute per-segment RMS and global stats
    seg_rms_values: list[float] = [_rms(seg.start, seg.end) for seg in segments]
    nonzero = [v for v in seg_rms_values if v > 0.0]
    global_rms_mean = sum(nonzero) / len(nonzero) if nonzero else 0.0
    global_rms_max = max(nonzero) if nonzero else 0.0

    # Second pass: classify each segment
    burst_scores: list[float] = []
    for seg, seg_rms in zip(segments, seg_rms_values):
        pre_start = max(0.0, seg.start - _BURST_PRE_WINDOW_SECONDS)
        silence_before = _rms(pre_start, seg.start)

        if (global_rms_mean > 0.0
                and silence_before < 0.1 * global_rms_mean
                and seg_rms > 0.5 * global_rms_max):
            burst_scores.append(1.0)
            logger.info("[Scorer] Burst detected at %.1fs (silence→loud)", seg.start)
        else:
            burst_scores.append(0.0)

    return burst_scores


# ---------------------------------------------------------------------------
# Score combination
# ---------------------------------------------------------------------------

def combine_scores(
    config: Config,
    text: float,
    audio: float,
    llm: float | None,
    spike: float = 0.0,
    burst: float = 0.0,
    excitement: float | None = None,
) -> float:
    """Multi-signal weighted combination of all scoring components.

    When excitement is provided (from AudioFeatures), it replaces the raw
    audio score in the audio component — excitement = 0.6×volume + 0.4×pitch
    is a richer signal than RMS alone.

    Genre-aware weighting (Task 4.3):
    - When config.genre is set (not "auto"), genre weights from config.genre_weights
      are used to split the budget between audio and semantic (text+llm) signals.
    - "auto" falls back to the original text_weight / audio_weight / llm_weight.

    LLM audio gate:
    - When config.llm_audio_gate is True, the LLM score is soft-capped by audio
      energy to prevent a high LLM score on a quiet moment from dominating.
    """
    # Use excitement score if available (richer than raw RMS)
    effective_audio = excitement if excitement is not None else audio

    # LLM audio gate
    if llm is not None and config.llm_audio_gate:
        effective_llm = llm * min(1.0, audio / 0.3)
    else:
        effective_llm = llm if llm is not None else 0.0

    genre = getattr(config, 'genre', 'auto')
    genre_weights = getattr(config, 'genre_weights', {})

    if genre != 'auto' and genre in genre_weights:
        # Genre-aware formula: split budget between audio and semantic signals
        gw = genre_weights[genre]
        audio_w = gw.get('audio', 0.5)
        semantic_w = gw.get('semantic', 0.5)
        total = audio_w + semantic_w
        if total > 0:
            audio_w /= total
            semantic_w /= total

        # Semantic = blended text + llm
        if effective_llm > 0.0:
            semantic = 0.5 * text + 0.5 * effective_llm
        else:
            semantic = text

        base = audio_w * effective_audio + semantic_w * semantic
    else:
        # Original formula
        base = (config.text_weight * text
                + config.audio_weight * effective_audio
                + config.llm_weight * effective_llm)

    # Spike and burst are always additive bonuses on top
    return max(0.0, base + config.spike_weight * spike + config.burst_weight * burst)


# ---------------------------------------------------------------------------
# Phase 2: LLM scoring on candidate windows
# ---------------------------------------------------------------------------

def _build_customized_rubric(config: Config) -> str:
    """Build a customized scoring rubric based on creator profile.
    
    Customizes the rubric emphasis based on content_type and energy_level:
    - High-energy content (gaming, comedy): emphasize audio energy and reactions
    - Calm content (podcast, educational): emphasize semantic interest and insights
    - Adjusts scale anchors to be relative to the creator's typical content
    
    Returns:
        Formatted rubric string for LLM prompt
    """
    profile = config.creator_profile
    
    # Default rubric (no profile or auto content type)
    if not profile or profile.content_type == "auto":
        return (
            "SCORING RUBRIC — be STRICT. The average clip should score 4-5. "
            "Only genuinely exciting moments score 7+:\n"
            "  1   Dead air, silence, nothing happening\n"
            "  2   Filler content, boring transition, no value\n"
            "  3   Routine commentary, flat delivery, low energy\n"
            "  4   Mildly interesting but forgettable — average content\n"
            "  5   Decent moment, some engagement, watchable\n"
            "  6   Good moment — clear reaction or interesting content\n"
            "  7   Strong clip — funny, exciting, or emotionally engaging\n"
            "  8   Very good — shareable, memorable, strong energy\n"
            "  9   Excellent — would stop a scroll, high viral potential\n"
            " 10   Perfect — unmissable, instant viral, once-in-a-stream moment\n\n"
            "IMPORTANT RULES:\n"
            "- A quiet/whispered moment (low volume %) should NEVER score above 5 unless "
            "the words themselves are extraordinary\n"
            "- Generic stream phrases ('follow the YouTube', 'w in the chat', 'here we go') "
            "score 1-3 regardless of energy\n"
            "- Score based on what a VIEWER would feel watching this cold, not the streamer's "
            "perspective\n"
            "- If you are unsure, score lower rather than higher"
        )
    
    # High-energy content: gaming, comedy
    if profile.content_type in ("gaming", "comedy") or profile.energy_level == "high":
        return (
            f"SCORING RUBRIC — calibrated for {profile.content_type.upper()} content. "
            "Be STRICT. The average clip should score 4-5. Only genuinely exciting moments score 7+:\n"
            "  1   Dead air, silence, nothing happening\n"
            "  2   Filler content, boring transition, no value\n"
            "  3   Routine gameplay/commentary, flat delivery, low energy\n"
            f"  4   Mildly interesting but forgettable — typical {profile.content_type} content\n"
            "  5   Decent moment, some engagement, watchable\n"
            "  6   Good moment — clear reaction, excitement, or funny moment\n"
            "  7   Strong clip — HIGH ENERGY, intense reaction, exciting gameplay/punchline\n"
            "  8   Very good — shareable, memorable, LOUD and engaging\n"
            "  9   Excellent — would stop a scroll, peak excitement/comedy\n"
            " 10   Perfect — unmissable, instant viral, legendary moment\n\n"
            "IMPORTANT RULES FOR HIGH-ENERGY CONTENT:\n"
            "- PRIORITIZE audio energy and reaction intensity — loud moments with strong reactions score higher\n"
            "- Look for sudden excitement, laughter, screaming, or intense emotional reactions\n"
            "- Quiet moments should score LOW (1-4) unless the words are extraordinary\n"
            "- Generic phrases score 1-3 regardless of energy\n"
            "- Score based on VIEWER excitement, not just content interest\n"
            "- If you are unsure, score lower rather than higher"
        )
    
    # Calm content: podcast, educational
    if profile.content_type in ("podcast", "educational") or profile.energy_level == "calm":
        return (
            f"SCORING RUBRIC — calibrated for {profile.content_type.upper()} content. "
            "Be STRICT. The average clip should score 4-5. Only genuinely insightful moments score 7+:\n"
            "  1   Dead air, silence, nothing happening\n"
            "  2   Filler content, boring transition, no value\n"
            "  3   Routine discussion, flat delivery, no insight\n"
            f"  4   Mildly interesting but forgettable — typical {profile.content_type} content\n"
            "  5   Decent moment, some engagement, watchable\n"
            "  6   Good moment — interesting insight, clear explanation, or engaging story\n"
            "  7   Strong clip — VALUABLE INSIGHT, memorable explanation, or compelling narrative\n"
            "  8   Very good — shareable, thought-provoking, high semantic value\n"
            "  9   Excellent — would stop a scroll, profound insight or fascinating story\n"
            " 10   Perfect — unmissable, instant viral, paradigm-shifting insight\n\n"
            "IMPORTANT RULES FOR CALM CONTENT:\n"
            "- PRIORITIZE semantic interest and insight quality — profound ideas score higher\n"
            "- Look for clear explanations, surprising insights, compelling stories, or valuable takeaways\n"
            "- Audio energy is LESS important — quiet moments can score high if content is strong\n"
            "- Generic discussion or filler scores 1-3\n"
            "- Score based on VIEWER learning/engagement, not just entertainment\n"
            "- If you are unsure, score lower rather than higher"
        )
    
    # Moderate energy: vlog, other content types
    return (
        f"SCORING RUBRIC — calibrated for {profile.content_type.upper()} content. "
        "Be STRICT. The average clip should score 4-5. Only genuinely engaging moments score 7+:\n"
        "  1   Dead air, silence, nothing happening\n"
        "  2   Filler content, boring transition, no value\n"
        "  3   Routine content, flat delivery, low engagement\n"
        f"  4   Mildly interesting but forgettable — typical {profile.content_type} content\n"
        "  5   Decent moment, some engagement, watchable\n"
        "  6   Good moment — clear emotion, interesting content, or engaging story\n"
        "  7   Strong clip — emotionally engaging, funny, or memorable\n"
        "  8   Very good — shareable, relatable, strong emotional connection\n"
        "  9   Excellent — would stop a scroll, highly relatable or moving\n"
        " 10   Perfect — unmissable, instant viral, universally compelling\n\n"
        "IMPORTANT RULES:\n"
        "- BALANCE audio energy and semantic interest — both matter\n"
        "- Look for authentic emotions, relatable moments, or interesting stories\n"
        "- Quiet moments can score moderately (4-6) if emotionally engaging\n"
        "- Generic content or filler scores 1-3\n"
        "- Score based on VIEWER emotional connection and engagement\n"
        "- If you are unsure, score lower rather than higher"
    )


def _call_llm(config: Config, prompt: str) -> str:
    payload = {"model": config.llm_model, "prompt": prompt, "stream": False}

    # Attempt 1: 60-second timeout (increased from 30s)
    try:
        response = requests.post(config.llm_endpoint, json=payload, timeout=60)
    except requests.ConnectionError as exc:
        raise LLMScoringError(
            f"LLM endpoint unreachable at {config.llm_endpoint!r}: {exc}"
        ) from exc
    except requests.Timeout:
        # Retry once with a 45-second timeout (increased from 15s)
        logger.warning("[Scorer] LLM call timed out after 60s, retrying with 45s timeout (attempt 2/2)…")
        try:
            response = requests.post(config.llm_endpoint, json=payload, timeout=45)
        except (requests.ConnectionError, requests.Timeout) as exc:
            raise LLMScoringError(
                f"LLM endpoint unreachable at {config.llm_endpoint!r}: {exc}"
            ) from exc

    try:
        response_data = response.json()
        raw_response = str(response_data.get("response", ""))
        if not raw_response.strip():
            # Log additional debugging info for empty responses
            logger.warning(
                "LLM returned empty response. Status: %d, full response: %r",
                response.status_code, response_data
            )
        return raw_response
    except (ValueError, KeyError, TypeError):
        return response.text


def _check_llm_model_available(config: Config) -> bool:
    """Check if the configured LLM model is available via a lightweight HTTP probe.

    Derives the base URL from config.llm_endpoint by stripping the ``/api/generate``
    path suffix, then issues a GET to ``<base_url>/api/tags`` (Ollama's model list
    endpoint) with a short 3-second timeout.

    - 200 response: parse the JSON and check whether config.llm_model appears in
      the model names list.  The tag suffix (e.g. ``:latest``) is stripped before
      comparing so ``llama3`` matches ``llama3:latest``.
    - Non-200 response or any request error (connection refused, timeout, etc.):
      assume the model is available (graceful fallback for non-Ollama endpoints).
    """
    # Derive base URL: strip /api/generate (or any trailing path) to get host:port
    endpoint = config.llm_endpoint.rstrip("/")
    if endpoint.endswith("/api/generate"):
        base_url = endpoint[: -len("/api/generate")]
    else:
        # Unknown endpoint format — fall back to assuming available
        logger.debug(
            "LLM endpoint %r does not end with /api/generate; skipping availability probe",
            config.llm_endpoint,
        )
        return True

    tags_url = f"{base_url}/api/tags"
    try:
        response = requests.get(tags_url, timeout=3)
    except Exception as exc:
        # Connection error, timeout, etc. — assume available (graceful fallback)
        logger.debug("LLM availability probe failed (%s); assuming model is available", exc)
        return True

    if response.status_code != 200:
        # Non-Ollama endpoint or server error — assume available
        logger.debug(
            "LLM availability probe returned HTTP %d; assuming model is available",
            response.status_code,
        )
        return True

    # Parse the model list and check for a match
    try:
        data = response.json()
        models = data.get("models", [])
        configured_model = config.llm_model  # e.g. "llama3.2:1b" or "llama3"
        configured_base = configured_model.split(":")[0]
        configured_tag  = configured_model.split(":")[1] if ":" in configured_model else None

        for entry in models:
            name = entry.get("name", "") if isinstance(entry, dict) else str(entry)
            entry_base = name.split(":")[0]
            entry_tag  = name.split(":")[1] if ":" in name else None

            if configured_tag:
                # Exact match required when a specific tag is requested
                if entry_base == configured_base and entry_tag == configured_tag:
                    logger.debug("LLM model %r found in Ollama tags", config.llm_model)
                    return True
            else:
                # No tag specified — match any variant of the base name
                if entry_base == configured_base:
                    logger.debug("LLM model %r found in Ollama tags", config.llm_model)
                    return True

        logger.warning(
            "LLM model %r not found in Ollama model list at %s. "
            "Run: ollama pull %s",
            config.llm_model, tags_url, config.llm_model,
        )
        return False
    except Exception as exc:
        # Malformed JSON or unexpected structure — assume available
        logger.debug("Failed to parse Ollama tags response (%s); assuming model is available", exc)
        return True


def _build_candidate_windows(
    segments: list[Segment],
    text_scores: list[float],
    audio_scores: list[float],
    config: Config,
    spike_scores: list[float] | None = None,
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Identify candidate seed indices for LLM scoring using two parallel tracks.

    Strategy:
    1. **Text+audio track**: rank by ``text_weight * text_score + audio_weight *
       audio_score`` and pick the top
       ``llm_max_candidates * (1 - llm_audio_spike_percentage)`` seeds (spaced at least
       ``min_clip_duration`` apart).
    2. **Audio spike track**: rank by ``spike_score`` alone and pick the top
       ``llm_max_candidates * llm_audio_spike_percentage`` seeds (same spacing constraint, applied
       independently of the text+audio track).
    3. Merge both shortlists and deduplicate: if two candidates have midpoints
       within ``min_clip_duration`` of each other, keep the one with the higher
       ``pre_score`` (text+audio combined score).

    This guarantees that high-energy silent moments (sudden loud sounds) always reach LLM scoring
    even when their text score is low.

    When ``llm_audio_spike_percentage`` is 0.0 (or ``spike_scores`` is not provided),
    the function falls back to the original single-track behaviour.

    Returns:
        Tuple of (llm_candidates, audio_spike_candidates):
        - llm_candidates: List of (segment_index, pre_score) for LLM scoring
        - audio_spike_candidates: List of (segment_index, pre_score) for pure audio spikes (bypass LLM)
    """
    n = len(segments)
    pre_scores = [
        config.llm_prefilter_text_weight * text_scores[i]
        + config.llm_prefilter_audio_weight * audio_scores[i]
        for i in range(n)
    ]

    min_spacing = config.min_clip_duration

    def _pick_top(
        ranked_indices: list[int],
        budget: int,
        already_used: list[float],
    ) -> list[tuple[int, float]]:
        """Greedily pick up to *budget* seeds from *ranked_indices*, respecting
        the minimum spacing constraint against *already_used* midpoints.

        Returns a list of (segment_index, pre_score) and appends the chosen
        midpoints to *already_used* in-place.
        """
        picked: list[tuple[int, float]] = []
        for idx in ranked_indices:
            if len(picked) >= budget:
                break
            seg_mid = (segments[idx].start + segments[idx].end) / 2.0
            if any(abs(seg_mid - t) < min_spacing for t in already_used):
                continue
            picked.append((idx, pre_scores[idx]))
            already_used.append(seg_mid)
        return picked

    # Calculate audio spike budget as a percentage of total candidates
    # Example: with llm_max_candidates=20 and llm_audio_spike_percentage=0.2,
    # audio_budget = int(20 * 0.2) = 4 slots for audio spikes
    audio_budget = int(config.llm_max_candidates * config.llm_audio_spike_percentage) if spike_scores else 0
    text_budget = config.llm_max_candidates - audio_budget

    # --- Track 1: text+audio ---
    ranked_text = sorted(range(n), key=lambda i: pre_scores[i], reverse=True)
    used_times: list[float] = []
    text_track = _pick_top(ranked_text, text_budget, used_times)

    # --- Track 2: audio spike (independent ranking, shared spacing pool) ---
    spike_track: list[tuple[int, float]] = []
    if audio_budget > 0 and spike_scores:
        ranked_spike = sorted(range(n), key=lambda i: spike_scores[i], reverse=True)
        # used_times already contains midpoints from the text track so we
        # don't place spike seeds on top of text+audio seeds.
        spike_track = _pick_top(ranked_spike, audio_budget, used_times)

    # Return both tracks separately:
    # - text_track goes to LLM scoring
    # - spike_track bypasses LLM (pure audio spike clips)
    text_track.sort(key=lambda x: x[1], reverse=True)
    spike_track.sort(key=lambda x: x[1], reverse=True)
    
    return (text_track, spike_track)


def _score_window_with_llm(
    config: Config,
    seed_idx: int,
    all_segments: list[Segment],
    all_audio_scores: list[float] | None = None,
    all_raw_rms: list[float] | None = None,
    global_rms_mean: float = 0.0,
    global_rms_max: float = 0.0,
    video_summary: str = "",
) -> tuple[float, LLMMetadata | None]:
    """Score a ~30s window centred on seed_idx using the LLM.

    Passes both relative (normalised) and absolute audio energy to the LLM
    so it can distinguish a genuinely loud moment from one that is merely
    louder than the rest of a quiet video.

    Scoring rubric (1–10, strict):
      1–2  Dead air, filler, nothing happening
      3–4  Routine commentary, flat delivery, no reaction
      5–6  Mildly interesting — some engagement but nothing standout
      7–8  Good clip — clear reaction, funny/exciting, strong energy or emotion
      9    Great clip — very shareable, strong hook, memorable moment
      10   Perfect — instant viral potential, would stop a scroll

    Args:
        video_summary: Optional video-level summary for context (prepended to prompt)

    Returns (llm_score_0_to_1, LLMMetadata | None).
    """
    from pipeline.models import LLMMetadata

    seed = all_segments[seed_idx]
    half = config.min_clip_duration / 2.0
    window_start = seed.start - half
    window_end = seed.end + half

    # Collect segments and their audio data within the window
    window_data: list[tuple[Segment, float, float]] = []  # (seg, norm_score, raw_rms)
    for i, seg in enumerate(all_segments):
        if seg.end > window_start and seg.start < window_end:
            norm = all_audio_scores[i] if all_audio_scores else 0.0
            raw = all_raw_rms[i] if all_raw_rms else 0.0
            window_data.append((seg, norm, raw))

    if not window_data:
        window_data = [(seed,
                        all_audio_scores[seed_idx] if all_audio_scores else 0.0,
                        all_raw_rms[seed_idx] if all_raw_rms else 0.0)]

    # Window-level energy stats
    raw_vals = [r for _, _, r in window_data]
    norm_vals = [n for _, n, _ in window_data]
    avg_raw = sum(raw_vals) / len(raw_vals) if raw_vals else 0.0
    peak_raw = max(raw_vals) if raw_vals else 0.0
    avg_norm = sum(norm_vals) / len(norm_vals) if norm_vals else 0.0
    peak_norm = max(norm_vals) if norm_vals else 0.0

    # Absolute energy level relative to the whole video
    # Classify how loud this window is in absolute terms
    if global_rms_max > 0:
        abs_ratio = avg_raw / global_rms_max  # 0–1 vs loudest moment in video
    else:
        abs_ratio = avg_norm

    if abs_ratio >= 0.6:
        abs_energy_desc = "LOUD — one of the loudest moments in the video"
    elif abs_ratio >= 0.35:
        abs_energy_desc = "moderate — average loudness for this video"
    elif abs_ratio >= 0.15:
        abs_energy_desc = "quiet — below average loudness"
    else:
        abs_energy_desc = "very quiet / near-silent — whisper or background noise"

    def _bar(val: float) -> str:
        filled = round(val * 5)
        return "█" * filled + "░" * (5 - filled)

    # Build transcript block with per-line energy
    lines: list[str] = []
    for seg, norm, raw in window_data:
        marker = " <<<HIGHLIGHT>>>" if seg is seed else ""
        # Show both relative bar and absolute % of video peak
        abs_pct = int((raw / global_rms_max * 100)) if global_rms_max > 0 else 0
        lines.append(
            f"[{seg.start:.1f}s-{seg.end:.1f}s] "
            f"[vol:{_bar(norm)} {abs_pct:3d}% of peak]{marker} "
            f"{seg.text.strip()}"
        )
    transcript_block = "\n".join(lines)

    # Build creator profile context if available
    profile_context = ""
    if config.creator_profile:
        profile = config.creator_profile
        profile_context = (
            f"CREATOR PROFILE:\n"
            f"Content Type: {profile.content_type}\n"
            f"Energy Level: {profile.energy_level}\n"
            f"Typical Clip Duration: {profile.typical_clip_duration:.0f}s\n\n"
        )

    # Customize rubric based on content_type and energy_level
    rubric = _build_customized_rubric(config)
    
    # Log customized rubric at DEBUG level
    logger.debug(
        "[Scorer] Customized rubric for creator_id=%s (content_type=%s, energy_level=%s):\n%s",
        config.creator_profile.creator_id if config.creator_profile else "none",
        config.creator_profile.content_type if config.creator_profile else "auto",
        config.creator_profile.energy_level if config.creator_profile else "moderate",
        rubric
    )

    # Build video context section (prepend video summary if available)
    video_context = ""
    if video_summary:
        video_context = f"VIDEO CONTEXT: {video_summary}\n\n"

    prompt = (
        "You are a strict YouTube Shorts editor. Your job is to find genuinely viral-worthy "
        "moments — not just anything that sounds vaguely interesting.\n\n"
        f"{video_context}"
        f"{profile_context}"
        "You will see a ~30-second transcript window. Each line shows:\n"
        "  [timestamp] [vol:█████ NNN% of peak] spoken text\n"
        "The volume bar and percentage show how loud that moment is relative to the "
        "LOUDEST moment in the entire video. 100% = maximum volume in the video.\n\n"
        f"WINDOW ENERGY: avg={int(abs_ratio*100)}% of video peak, "
        f"peak={int(peak_raw/global_rms_max*100) if global_rms_max > 0 else 0}% of video peak\n"
        f"ABSOLUTE LEVEL: {abs_energy_desc}\n\n"
        f"WINDOW TRANSCRIPT:\n{transcript_block}\n\n"
        "The segment marked <<<HIGHLIGHT>>> is the candidate clip moment.\n\n"
        "IMPORTANT: Score this moment relative to THIS VIDEO'S baseline, not generic content. "
        "A moment that is interesting FOR THIS VIDEO should score higher than a generically "
        "interesting moment. Use the VIDEO CONTEXT above to understand what is typical for this video.\n\n"
        f"{rubric}\n\n"
        "Respond in EXACTLY this format (no extra text, no preamble):\n"
        "SCORE: <integer 1-10>\n"
        "TITLE: <catchy YouTube Shorts title, max 60 chars, no quotes>\n"
        "DESCRIPTION: <1-2 sentences describing what actually happens in the clip>\n"
        "TAGS: <5-8 hashtags e.g. #shorts #viral #funny>"
    )

    raw_response = _call_llm(config, prompt)

    score_match = re.search(r'SCORE:\s*(10|[1-9])', raw_response)
    if score_match is None:
        logger.warning(
            "LLM returned no parseable SCORE for window at %.1fs; defaulting to 0.0. "
            "Response: %r. This may indicate the model '%s' is not responding correctly. "
            "Try: ollama pull %s",
            seed.start, raw_response[:300], config.llm_model, config.llm_model
        )
        return 0.0, None

    llm_score = float(score_match.group(1)) / 10.0

    title_match = re.search(r'TITLE:\s*(.+)', raw_response)
    desc_match = re.search(r'DESCRIPTION:\s*(.+)', raw_response)
    tags_match = re.search(r'TAGS:\s*(.+)', raw_response)

    title = title_match.group(1).strip() if title_match else ""
    description = desc_match.group(1).strip() if desc_match else ""
    tags_raw = tags_match.group(1).strip() if tags_match else ""
    tags = [t.strip() for t in tags_raw.split() if t.strip().startswith("#")]

    metadata = LLMMetadata(title=title, description=description, tags=tags) if title else None

    logger.info(
        "  LLM window at %.1fs: %.1f/10 (abs=%d%% of peak, %s) | %r",
        seed.start, float(score_match.group(1)),
        int(abs_ratio * 100), abs_energy_desc.split(" —")[0], title,
    )

    return llm_score, metadata


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

# Public alias kept for backward compatibility with tests
compute_llm_score_with_context = _score_window_with_llm


def score_segments(
    config: Config,
    transcript: Transcript,
    wav_path: str,
    llm_progress_callback: "Callable[[int, int], None] | None" = None,
) -> list[ScoredSegment]:
    """Score every segment in *transcript* and return a ScoredSegment list.

    Pipeline:
    1. Phase 1 — text + audio scores on ALL segments (fast, local).
       - text_patterns heuristics are blended into the text score.
    2. Hook detection (optional) — if LLM enabled and hook_detection_enabled,
       run a sliding-window hook scan over the full transcript and apply a
       multiplicative boost to text scores near detected hooks.
    3. Phase 2 — if LLM enabled, identify the top `llm_max_candidates`
       candidate *windows* (spaced at least min_clip_duration apart) and
       call the LLM once per window.  The LLM score is then applied to the
       seed segment and propagated to nearby segments in the same window.
    4. Combine scores and return.
    """
    from pipeline.hook_detector import detect_hooks, get_hook_score_for_window

    segments = transcript.segments
    logger.info("Scorer starting — %d segment(s) to score", len(segments))
    t0 = time.time()

    # Apply platform clip-length constraints if set
    platform = getattr(config, 'platform', 'none')
    platform_constraints = getattr(config, 'platform_constraints', {})
    if platform and platform != 'none' and platform in platform_constraints:
        pc = platform_constraints[platform]
        config.min_clip_duration = max(config.min_clip_duration, pc.get('min', config.min_clip_duration))
        config.max_clip_duration = min(config.max_clip_duration, pc.get('max', config.max_clip_duration))
        logger.info("Scorer: platform=%s → min_clip=%.0fs max_clip=%.0fs", platform, config.min_clip_duration, config.max_clip_duration)

    # Phase 1: text + audio + spike + burst
    text_scores = [compute_text_score(config, seg) for seg in segments]
    # Load WAV once and reuse across all audio scoring functions (avoids 3× disk reads)
    try:
        _wav_data: tuple[int, np.ndarray] | None = scipy.io.wavfile.read(wav_path)
    except Exception as exc:
        logger.warning("Failed to pre-load WAV file %s: %s — falling back to per-call reads", wav_path, exc)
        _wav_data = None
    audio_scores, raw_rms = compute_audio_score_with_raw(segments, wav_path, wav_data=_wav_data)
    spike_scores = compute_spike_score(segments, wav_path, wav_data=_wav_data) if config.spike_weight > 0.0 else [0.0] * len(segments)
    burst_scores = compute_burst_score(segments, wav_path, wav_data=_wav_data) if config.burst_weight > 0.0 else [0.0] * len(segments)

    # Compute global RMS stats for absolute energy context
    nonzero_rms = [v for v in raw_rms if v > 0.0]
    global_rms_mean = sum(nonzero_rms) / len(nonzero_rms) if nonzero_rms else 0.0
    global_rms_max = max(nonzero_rms) if nonzero_rms else 0.0

    # Enhanced audio features: excitement_score = 0.6×volume + 0.4×pitch
    # Map each segment to the nearest AudioFeatures window by timestamp.
    audio_features = compute_audio_features(config, wav_path)
    excitement_scores: list[float | None]
    if audio_features:
        window = config.audio_feature_window
        excitement_scores = []
        for seg in segments:
            mid = (seg.start + seg.end) / 2.0
            idx = min(int(mid / window), len(audio_features) - 1)
            excitement_scores.append(audio_features[idx].excitement_score)
        logger.info("Scorer: excitement scores computed from %d audio feature windows", len(audio_features))
    else:
        excitement_scores = [None] * len(segments)

    # Emotion detection: extract emotion features and boost audio scores for high-energy emotions
    # (laughter, scream, excitement)
    if config.emotion_detection_enabled:
        from pipeline.emotion_detector import extract_emotion_features
        
        emotion_features = extract_emotion_features(wav_path, window_size=config.audio_feature_window)
        
        if emotion_features:
            logger.info("Scorer: emotion detection enabled — processing %d emotion windows", len(emotion_features))
            
            # Map segments to emotion scores and apply boost
            for i, seg in enumerate(segments):
                mid = (seg.start + seg.end) / 2.0
                # Find the nearest emotion feature window
                idx = min(int(mid / config.audio_feature_window), len(emotion_features) - 1)
                emotion = emotion_features[idx]
                
                # Boost audio score for high-energy emotions
                if emotion.emotion in ["laughter", "scream", "excitement"]:
                    boost = 1.0 + config.emotion_boost_multiplier * emotion.confidence
                    audio_scores[i] *= boost
                    
                    logger.info(
                        "[Scorer] Emotion detected at %.1fs: %s (confidence=%.2f, boost=%.2fx)",
                        seg.start, emotion.emotion, emotion.confidence, boost
                    )
        else:
            logger.debug("Scorer: no emotion features extracted (audio may be silent or too short)")
    else:
        logger.debug("Scorer: emotion detection disabled")

    # Hook detection: run once over the full transcript, then boost text scores
    # multiplicatively where hooks are detected.
    # boost formula: text_score *= (1 + hook_boost_max * hook_score), capped at 1.0
    hook_boost_max = getattr(config, 'hook_boost_max', 0.4)
    hook_detection_enabled = getattr(config, 'hook_detection_enabled', True)
    if config.llm_enabled and hook_detection_enabled and segments:
        logger.info("Scorer: running hook detection over %d segments...", len(segments))
        hooks = detect_hooks(
            config,
            segments,
            window_size=getattr(config, 'hook_window_size', 3),
            stride=getattr(config, 'hook_stride', 2),
            min_words=getattr(config, 'hook_min_words', 5),
            score_threshold=getattr(config, 'hook_score_threshold', 0.6),
        )
        if hooks:
            logger.info("Scorer: %d hook(s) detected — applying multiplicative boost", len(hooks))
            for i, seg in enumerate(segments):
                hook_score = get_hook_score_for_window(hooks, seg.start, seg.end)
                if hook_score > 0.0:
                    boost = 1.0 + hook_boost_max * hook_score
                    text_scores[i] = min(1.0, text_scores[i] * boost)
    else:
        hooks = []

    # Phase 2: LLM on candidate windows only
    llm_scores: list[float] = [0.0] * len(segments)
    llm_metadatas: list[LLMMetadata | None] = [None] * len(segments)
    audio_spike_indices_set: set[int] = set()  # Initialize here to avoid UnboundLocalError

    if config.llm_enabled and segments:
        # Generate video summary for LLM context (cached)
        video_summary = ""
        if config.video_summary_enabled:
            video_summary = generate_video_summary(config, transcript, wav_path)
            logger.info("[Scorer] Video summary generated for LLM context")
        
        # Check if LLM model is available before attempting scoring
        if not _check_llm_model_available(config):
            logger.warning(
                "LLM model '%s' is not available at %s. "
                "Skipping LLM scoring. Make sure the model is pulled: ollama pull %s",
                config.llm_model, config.llm_endpoint, config.llm_model
            )
            config.llm_enabled = False  # Disable for this run
        else:
            llm_candidates, audio_spike_candidates = _build_candidate_windows(
                segments, text_scores, audio_scores, config, spike_scores
            )

            logger.info(
                "Scorer LLM: %d candidate window(s) for LLM scoring, %d audio spike clips (bypass LLM) "
                "from %d segments (spaced >= %.0fs apart, cap=%d)",
                len(llm_candidates), len(audio_spike_candidates), len(segments),
                config.min_clip_duration, config.llm_max_candidates,
            )

            # Track which segments are audio spikes (will bypass LLM)
            audio_spike_indices_set = {idx for idx, _ in audio_spike_candidates}

            # Score LLM candidates concurrently — HTTP I/O releases the GIL so
            # threads work well here and keep Ollama's inference queue full.
            total_llm = len(llm_candidates)
            completed_count = [0]
            results_lock = __import__("threading").Lock()

            def _score_one(args):
                llm_idx, (seed_idx, pre_score) = args
                try:
                    score, meta = _score_window_with_llm(
                        config, seed_idx, segments, audio_scores, raw_rms,
                        global_rms_mean, global_rms_max, video_summary
                    )
                except LLMScoringError as exc:
                    logger.warning("LLM scoring failed for window at %.1fs: %s",
                                   segments[seed_idx].start, exc)
                    score, meta = 0.0, None
                with results_lock:
                    completed_count[0] += 1
                    if llm_progress_callback is not None:
                        llm_progress_callback(completed_count[0], total_llm)
                return seed_idx, score, meta

            # Concurrency = number of parallel requests to Ollama.
            # Ollama queues requests internally so more than its thread count
            # doesn't help, but 4 keeps its pipeline full without overloading.
            llm_concurrency = min(total_llm, 4)
            import concurrent.futures as _cf
            with _cf.ThreadPoolExecutor(max_workers=llm_concurrency) as pool:
                scored_results = list(pool.map(
                    _score_one, enumerate(llm_candidates)
                ))

            for seed_idx, llm_score, metadata in scored_results:
                seed = segments[seed_idx]
                half = config.min_clip_duration / 2.0
                for i, seg in enumerate(segments):
                    if seg.end > seed.start - half and seg.start < seed.end + half:
                        if llm_score > llm_scores[i]:
                            llm_scores[i] = llm_score
                            llm_metadatas[i] = metadata

            # Signal completion after all candidates are scored
            if llm_progress_callback is not None and total_llm > 0:
                llm_progress_callback(total_llm, total_llm)

    # Assemble ScoredSegment objects
    scored: list[ScoredSegment] = []
    for i, (seg, text_s, audio_s, llm_s, spike_s, burst_s) in enumerate(
        zip(segments, text_scores, audio_scores, llm_scores, spike_scores, burst_scores)
    ):
        excitement_s = excitement_scores[i]
        clip_s = combine_scores(config, text_s, audio_s, llm_s, spike_s, burst_s, excitement_s)
        is_spike = i in audio_spike_indices_set
        scored.append(ScoredSegment(
            segment=seg,
            text_score=text_s,
            audio_score=audio_s,
            llm_score=llm_s,
            clip_score=clip_s,
            llm_metadata=llm_metadatas[i] if config.llm_enabled else None,
            is_audio_spike=is_spike,
        ))

    elapsed = time.time() - t0
    if scored:
        clip_scores = [s.clip_score for s in scored]
        logger.info(
            "Scorer complete — %d segment(s) in %.1fs | "
            "clip_score min=%.3f max=%.3f mean=%.3f",
            len(scored), elapsed,
            min(clip_scores), max(clip_scores),
            sum(clip_scores) / len(clip_scores),
        )
    else:
        logger.info("Scorer complete — 0 segments in %.1fs", elapsed)

    return scored
