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
    return max(0.0, min(1.0, raw_score / (raw_score + 5.0)))


# ---------------------------------------------------------------------------
# Phase 1: audio scoring
# ---------------------------------------------------------------------------

def compute_audio_score(segments: list[Segment], wav_path: str) -> list[float]:
    """Compute normalized RMS energy scores for all segments in one WAV pass."""
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
    rms_values: list[float] = []

    for seg in segments:
        s = max(0, min(int(seg.start * sample_rate), total_samples))
        e = max(0, min(int(seg.end * sample_rate), total_samples))
        chunk = data[s:e]
        rms_values.append(float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) > 0 else 0.0)

    max_rms = max(rms_values) if rms_values else 0.0
    if max_rms == 0.0:
        return [0.0] * len(segments)
    return [v / max_rms for v in rms_values]


# ---------------------------------------------------------------------------
# Score combination
# ---------------------------------------------------------------------------

def combine_scores(
    config: Config,
    text: float,
    audio: float,
    llm: float | None,
) -> float:
    """Weighted sum of text, audio, and optional LLM scores. Always >= 0."""
    llm_value = llm if llm is not None else 0.0
    return max(0.0,
               config.text_weight * text
               + config.audio_weight * audio
               + config.llm_weight * llm_value)


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
        return str(response.json().get("response", ""))
    except (ValueError, KeyError, TypeError):
        return response.text


def _build_candidate_windows(
    segments: list[Segment],
    text_scores: list[float],
    audio_scores: list[float],
    config: Config,
) -> list[tuple[int, float]]:
    """Identify candidate seed indices for LLM scoring.

    Strategy:
    1. Compute a combined pre-score (text + audio) for every segment.
    2. Pick the top `llm_max_candidates` seeds, but enforce a minimum
       spacing of `min_clip_duration` seconds between seeds so we don't
       waste LLM calls on overlapping moments.

    Returns:
        List of (segment_index, pre_score) sorted by pre_score descending,
        length <= config.llm_max_candidates.
    """
    pre_scores = [
        config.text_weight * text_scores[i] + config.audio_weight * audio_scores[i]
        for i in range(len(segments))
    ]

    # Sort all segments by pre_score descending
    ranked = sorted(range(len(segments)), key=lambda i: pre_scores[i], reverse=True)

    selected: list[tuple[int, float]] = []
    used_times: list[float] = []
    min_spacing = config.min_clip_duration  # don't pick two seeds within one clip-length

    for idx in ranked:
        if len(selected) >= config.llm_max_candidates:
            break
        seg_mid = (segments[idx].start + segments[idx].end) / 2.0
        # Skip if too close to an already-selected seed
        if any(abs(seg_mid - t) < min_spacing for t in used_times):
            continue
        selected.append((idx, pre_scores[idx]))
        used_times.append(seg_mid)

    return selected  # already in descending pre_score order


def _score_window_with_llm(
    config: Config,
    seed_idx: int,
    all_segments: list[Segment],
    all_audio_scores: list[float] | None = None,
) -> tuple[float, LLMMetadata | None]:
    """Score a ~30s window centred on seed_idx using the LLM.

    Builds a transcript window spanning roughly `min_clip_duration` seconds
    around the seed segment.  Audio energy data for each line is included in
    the prompt so the LLM can factor in loudness/excitement alongside the text.

    Scoring rubric (1–10):
      1–2  Completely boring — filler, dead air, nothing happening
      3–4  Low interest — routine commentary, no reaction, flat delivery
      5–6  Mildly interesting — some engagement but nothing standout
      7–8  Good clip — clear reaction, funny/exciting moment, strong energy
      9    Great clip — very shareable, strong hook, high energy or emotion
      10   Perfect clip — instant viral potential, unmissable moment

    Returns (llm_score_0_to_1, LLMMetadata | None).
    """
    from pipeline.models import LLMMetadata

    seed = all_segments[seed_idx]
    half = config.min_clip_duration / 2.0
    window_start = seed.start - half
    window_end = seed.end + half

    # Collect segments and their audio scores within the window
    window_pairs: list[tuple[Segment, float]] = []
    for i, seg in enumerate(all_segments):
        if seg.end > window_start and seg.start < window_end:
            audio_val = all_audio_scores[i] if all_audio_scores else 0.0
            window_pairs.append((seg, audio_val))

    if not window_pairs:
        window_pairs = [(seed, all_audio_scores[seed_idx] if all_audio_scores else 0.0)]

    # Compute window-level energy stats for context
    energies = [e for _, e in window_pairs]
    avg_energy = sum(energies) / len(energies) if energies else 0.0
    peak_energy = max(energies) if energies else 0.0

    def _energy_bar(val: float) -> str:
        """Convert 0–1 energy to a visual bar: ░░░░░ to █████."""
        filled = round(val * 5)
        return "█" * filled + "░" * (5 - filled)

    # Build transcript block with per-line energy indicators
    lines: list[str] = []
    for seg, audio_val in window_pairs:
        marker = " <<<HIGHLIGHT>>>" if seg is seed else ""
        bar = _energy_bar(audio_val)
        lines.append(
            f"[{seg.start:.1f}s-{seg.end:.1f}s] [energy:{bar}]{marker} {seg.text.strip()}"
        )
    transcript_block = "\n".join(lines)

    # Describe overall window energy in plain English
    if avg_energy >= 0.75:
        energy_desc = "very high energy throughout"
    elif avg_energy >= 0.5:
        energy_desc = "high energy"
    elif avg_energy >= 0.3:
        energy_desc = "moderate energy"
    else:
        energy_desc = "low energy / quiet"

    prompt = (
        "You are a YouTube Shorts editor selecting the best highlight clips from a video.\n\n"
        "You will be given a ~30-second transcript window with audio energy indicators.\n"
        "Each line shows: [timestamp] [energy:█░░░░ to █████] followed by the spoken text.\n"
        "Energy bars show how loud/intense that moment is (5 filled = maximum energy).\n\n"
        f"WINDOW STATS: avg_energy={avg_energy:.2f}, peak_energy={peak_energy:.2f} ({energy_desc})\n\n"
        f"TRANSCRIPT:\n{transcript_block}\n\n"
        "The segment marked <<<HIGHLIGHT>>> is the candidate clip moment.\n\n"
        "SCORING RUBRIC (be strict — most clips should score 4–7):\n"
        "  1–2  Dead air, filler, nothing happening, completely boring\n"
        "  3–4  Routine commentary, flat delivery, low energy, no reaction\n"
        "  5–6  Mildly interesting — some engagement but nothing standout\n"
        "  7–8  Good clip — clear reaction, funny/exciting moment, strong energy or emotion\n"
        "  9    Great clip — very shareable, strong hook, high energy, memorable moment\n"
        " 10    Perfect clip — instant viral potential, unmissable, would stop a scroll\n\n"
        "Consider BOTH the spoken content AND the audio energy when scoring.\n"
        "A high-energy moment with weak text can still score well. "
        "A great quote with flat energy scores lower than the same quote delivered with excitement.\n\n"
        "Respond in EXACTLY this format (no extra text, no preamble):\n"
        "SCORE: <integer 1-10>\n"
        "TITLE: <catchy YouTube Shorts title, max 60 chars, no quotes>\n"
        "DESCRIPTION: <1-2 sentences, engaging, relevant to the moment>\n"
        "TAGS: <5-8 hashtags e.g. #shorts #viral #funny>"
    )

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
        "  LLM window at %.1fs: %.1f/10 (energy avg=%.2f peak=%.2f) | %r",
        seed.start, float(score_match.group(1)), avg_energy, peak_energy, title,
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

    # Phase 1: text + audio
    text_scores = [compute_text_score(config, seg) for seg in segments]
    audio_scores = compute_audio_score(segments, wav_path)

    # Phase 2: LLM on candidate windows only
    llm_scores: list[float] = [0.0] * len(segments)
    llm_metadatas: list[LLMMetadata | None] = [None] * len(segments)

    if config.llm_enabled and segments:
        candidates = _build_candidate_windows(segments, text_scores, audio_scores, config)

        logger.info(
            "Scorer LLM: %d candidate window(s) selected from %d segments "
            "(spaced >= %.0fs apart, cap=%d)",
            len(candidates), len(segments),
            config.min_clip_duration, config.llm_max_candidates,
        )

        for seed_idx, pre_score in candidates:
            try:
                llm_score, metadata = _score_window_with_llm(
                    config, seed_idx, segments, audio_scores
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

    # Assemble ScoredSegment objects
    scored: list[ScoredSegment] = []
    for i, (seg, text_s, audio_s, llm_s) in enumerate(
        zip(segments, text_scores, audio_scores, llm_scores)
    ):
        clip_s = combine_scores(config, text_s, audio_s, llm_s)
        scored.append(ScoredSegment(
            segment=seg,
            text_score=text_s,
            audio_score=audio_s,
            llm_score=llm_s,
            clip_score=clip_s,
            llm_metadata=llm_metadatas[i] if config.llm_enabled else None,
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
