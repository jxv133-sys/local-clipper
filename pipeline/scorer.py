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
from typing import TYPE_CHECKING

import numpy as np
import requests
import scipy.io.wavfile

from config import Config
from pipeline.exceptions import LLMScoringError
from pipeline.models import LLMMetadata, ScoredSegment, Segment, Transcript

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase 1: text scoring
# ---------------------------------------------------------------------------

def compute_text_score(config: Config, segment: Segment) -> float:
    """Compute a normalized text score for a single segment.

    Scoring components:
    - +2.0 per keyword occurrence (case-insensitive)
    - +reaction_weight (default 3.0) per reaction keyword occurrence
      (whole-word match, case-insensitive)
    - +0.02 per character in segment.text (rewards longer speech)
    - +1.5 per '!' in segment.text
    - +1.0 per '?' in segment.text
    - pace bonus for fast speech (> 3 words/sec), indicating excitement/urgency

    Normalized to [0.0, 1.0] via soft-max: raw / (raw + 5.0).
    """
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
        return 0.0
    normalized = max(0.0, min(1.0, raw_score / (raw_score + 5.0)))

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
            normalized *= config.repetition_penalty_multiplier

    return normalized


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
    segments: list[Segment], wav_path: str
) -> tuple[list[float], list[float]]:
    """Return (normalized_scores, raw_rms_values) for all segments.

    normalized_scores: each value in [0.0, 1.0] relative to the loudest segment.
    raw_rms_values: absolute RMS amplitude in [0.0, ~1.0] for float32 audio.
    """
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
# Phase 1: spike score (audio burst detection)
# ---------------------------------------------------------------------------

_SPIKE_BASELINE_WINDOW_SECONDS = 30.0  # rolling baseline window before each segment
_SPIKE_STRONG_RATIO = 3.0              # ratio at which spike_score saturates near 1.0


def compute_spike_score(segments: list[Segment], wav_path: str) -> list[float]:
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

    Returns a list of floats in [0.0, 1.0], one per segment.
    """
    if not segments:
        return []

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


def compute_burst_score(segments: list[Segment], wav_path: str) -> list[float]:
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

    Returns a list of floats (each 0.0 or 1.0), one per segment.
    """
    if not segments:
        return []

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
) -> float:
    """Weighted sum of text, audio, spike, burst, and optional LLM scores. Always >= 0.

    When config.llm_audio_gate is True and llm is not None, the LLM score
    contribution is soft-capped based on audio energy:
        effective_llm = llm * min(1.0, audio / 0.3)
    This prevents a high LLM score on a quiet moment from overriding strong
    audio signals.  At audio_score >= 0.3 the LLM score is used at full weight.
    """
    if llm is not None and config.llm_audio_gate:
        effective_llm = llm * min(1.0, audio / 0.3)
    else:
        effective_llm = llm if llm is not None else 0.0

    return max(0.0,
               config.text_weight * text
               + config.audio_weight * audio
               + config.llm_weight * effective_llm
               + config.spike_weight * spike
               + config.burst_weight * burst)


# ---------------------------------------------------------------------------
# Phase 2: LLM scoring on candidate windows
# ---------------------------------------------------------------------------

def _call_llm(config: Config, prompt: str) -> str:
    payload = {"model": config.llm_model, "prompt": prompt, "stream": False}
    try:
        response = requests.post(config.llm_endpoint, json=payload, timeout=60)
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
    """Check if the configured LLM model is available and responding."""
    try:
        # Try a simple prompt to test model availability
        test_prompt = "Respond with 'OK' if you can understand this message."
        payload = {"model": config.llm_model, "prompt": test_prompt, "stream": False}
        response = requests.post(config.llm_endpoint, json=payload, timeout=30)
        response_data = response.json()
        response_text = str(response_data.get("response", "")).strip()
        return bool(response_text and response_text.lower() != "failed")
    except Exception as exc:
        logger.warning("LLM model availability check failed: %s", exc)
        return False


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

    prompt = (
        "You are a strict YouTube Shorts editor. Your job is to find genuinely viral-worthy "
        "moments — not just anything that sounds vaguely interesting.\n\n"
        "You will see a ~30-second transcript window. Each line shows:\n"
        "  [timestamp] [vol:█████ NNN% of peak] spoken text\n"
        "The volume bar and percentage show how loud that moment is relative to the "
        "LOUDEST moment in the entire video. 100% = maximum volume in the video.\n\n"
        f"WINDOW ENERGY: avg={int(abs_ratio*100)}% of video peak, "
        f"peak={int(peak_raw/global_rms_max*100) if global_rms_max > 0 else 0}% of video peak\n"
        f"ABSOLUTE LEVEL: {abs_energy_desc}\n\n"
        f"TRANSCRIPT:\n{transcript_block}\n\n"
        "The segment marked <<<HIGHLIGHT>>> is the candidate clip moment.\n\n"
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
        "- If you are unsure, score lower rather than higher\n\n"
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

    raw = _call_llm(config, prompt)

    score_match = re.search(r'SCORE:\s*(10|[1-9])', raw)
    if score_match is None:
        logger.warning(
            "LLM returned no parseable SCORE for window at %.1fs; defaulting to 0.0. "
            "Response: %r",
            seed.start, raw[:300],
        )
        return 0.0, None

    llm_score = float(score_match.group(1)) / 10.0

    title_match = re.search(r'TITLE:\s*(.+)', raw)
    desc_match = re.search(r'DESCRIPTION:\s*(.+)', raw)
    tags_match = re.search(r'TAGS:\s*(.+)', raw)

    title = title_match.group(1).strip() if title_match else ""
    description = desc_match.group(1).strip() if desc_match else ""
    tags_raw = tags_match.group(1).strip() if tags_match else ""
    tags = [t.strip() for t in tags_raw.split() if t.strip().startswith("#")]

    metadata = LLMMetadata(title=title, description=description, tags=tags) if title else None

    logger.info(
        "  LLM window at %.1fs: %.1f/10 | %r",
        seed.start, float(score_match.group(1)), title,
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
) -> list[ScoredSegment]:
    """Score every segment in *transcript* and return a ScoredSegment list.

    Pipeline:
    1. Phase 1 — text + audio scores on ALL segments (fast, local).
    2. Phase 2 — if LLM enabled, identify the top `llm_max_candidates`
       candidate *windows* (spaced at least min_clip_duration apart) and
       call the LLM once per window.  The LLM score is then applied to the
       seed segment and propagated to nearby segments in the same window.
    3. Combine scores and return.
    """
    segments = transcript.segments
    logger.info("Scorer starting — %d segment(s) to score", len(segments))
    t0 = time.time()

    # Phase 1: text + audio + spike + burst
    text_scores = [compute_text_score(config, seg) for seg in segments]
    audio_scores, raw_rms = compute_audio_score_with_raw(segments, wav_path)
    spike_scores = compute_spike_score(segments, wav_path) if config.spike_weight > 0.0 else [0.0] * len(segments)
    burst_scores = compute_burst_score(segments, wav_path) if config.burst_weight > 0.0 else [0.0] * len(segments)

    # Compute global RMS stats for absolute energy context
    nonzero_rms = [v for v in raw_rms if v > 0.0]
    global_rms_mean = sum(nonzero_rms) / len(nonzero_rms) if nonzero_rms else 0.0
    global_rms_max = max(nonzero_rms) if nonzero_rms else 0.0

    # Phase 2: LLM on candidate windows only
    llm_scores: list[float] = [0.0] * len(segments)
    llm_metadatas: list[LLMMetadata | None] = [None] * len(segments)

    if config.llm_enabled and segments:
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

            # Score LLM candidates only (audio spikes bypass LLM)
            for seed_idx, pre_score in llm_candidates:
                try:
                    llm_score, metadata = _score_window_with_llm(
                        config, seed_idx, segments, audio_scores, raw_rms,
                        global_rms_mean, global_rms_max
                    )
                except LLMScoringError as exc:
                    logger.warning("LLM scoring failed for window at %.1fs: %s",
                                   segments[seed_idx].start, exc)
                    llm_score, metadata = 0.0, None

                # Apply the LLM score to the seed and all segments within the
                # same window (within half a clip-length).  This means nearby
                # segments benefit from the LLM's assessment of the moment.
                seed = segments[seed_idx]
                half = config.min_clip_duration / 2.0
                for i, seg in enumerate(segments):
                    if seg.end > seed.start - half and seg.start < seed.end + half:
                        # Take the max so a segment in two overlapping windows
                        # keeps the better LLM score.
                        if llm_score > llm_scores[i]:
                            llm_scores[i] = llm_score
                            llm_metadatas[i] = metadata
    else:
        audio_spike_indices_set = set()

    # Assemble ScoredSegment objects
    scored: list[ScoredSegment] = []
    for i, (seg, text_s, audio_s, llm_s, spike_s, burst_s) in enumerate(
        zip(segments, text_scores, audio_scores, llm_scores, spike_scores, burst_scores)
    ):
        clip_s = combine_scores(config, text_s, audio_s, llm_s, spike_s, burst_s)
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
