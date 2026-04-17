"""Scorer module for the video highlight generator pipeline.

Computes text scores, audio scores, LLM scores, and combines them
into a final clip score for each transcript segment.
"""

from __future__ import annotations

import logging
import math
import re
from typing import TYPE_CHECKING

import numpy as np
import requests
import scipy.io.wavfile

if TYPE_CHECKING:
    pass

from config import Config
from pipeline.exceptions import LLMScoringError
from pipeline.models import ScoredSegment, Segment, Transcript


def compute_text_score(config: Config, segment: Segment) -> float:
    """Compute a normalized text score for a segment.

    Scoring components:
    - +1.0 per keyword occurrence (case-insensitive)
    - +0.01 per character in segment.text (rewards longer speech)
    - +1.0 per '!' or '?' in segment.text

    The raw score is normalized to [0.0, 1.0] using a shifted sigmoid:
        score = (2 / (1 + exp(-raw / 10.0))) - 1.0
    clamped to [0.0, 1.0].

    This function is deterministic: same input always produces same output.

    Args:
        config: Pipeline configuration (provides keywords list).
        segment: The transcript segment to score.

    Returns:
        A float in [0.0, 1.0].
    """
    text = segment.text
    text_lower = text.lower()

    raw_score = 0.0

    # Keyword occurrences (case-insensitive)
    for keyword in config.keywords:
        keyword_lower = keyword.lower()
        start = 0
        while True:
            idx = text_lower.find(keyword_lower, start)
            if idx == -1:
                break
            raw_score += 2.0          # raised from 1.0 — keywords are strong signals
            start = idx + len(keyword_lower)

    # Character length reward — longer speech = more content
    raw_score += len(text) * 0.02     # raised from 0.01

    # Punctuation reward
    raw_score += text.count("!") * 1.5   # raised from 1.0
    raw_score += text.count("?") * 1.0

    # Normalize using shifted sigmoid: (2 / (1 + exp(-raw/5))) - 1
    # Divisor lowered from 10.0 to 5.0 so scores spread more meaningfully
    score = (2.0 / (1.0 + math.exp(-raw_score / 5.0))) - 1.0

    # Clamp to [0.0, 1.0] for safety
    return max(0.0, min(1.0, score))


def compute_audio_score(segments: list[Segment], wav_path: str) -> list[float]:
    """Compute normalized audio RMS energy scores for a list of segments.

    For each segment, the RMS energy of the audio samples within the segment's
    time range is computed. All RMS values are then normalized by the maximum
    RMS across all segments so that the highest-energy segment receives a score
    of 1.0 and all others are proportionally scaled.

    Args:
        segments: List of transcript segments to score.
        wav_path: Path to the WAV file to read audio samples from.

    Returns:
        A list of floats in [0.0, 1.0], one per segment, in the same order as
        the input segments list.
    """
    sample_rate, data = scipy.io.wavfile.read(wav_path)

    # Convert multi-channel audio to mono by averaging channels
    if data.ndim == 2:
        data = data.mean(axis=1)

    # Normalize integer types to float32 in [-1.0, 1.0]
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    else:
        data = data.astype(np.float32)

    total_samples = len(data)
    rms_values: list[float] = []

    for segment in segments:
        start_idx = int(segment.start * sample_rate)
        end_idx = int(segment.end * sample_rate)

        # Clamp to valid range
        start_idx = max(0, min(start_idx, total_samples))
        end_idx = max(0, min(end_idx, total_samples))

        samples = data[start_idx:end_idx]

        if len(samples) == 0:
            rms_values.append(0.0)
        else:
            rms = float(np.sqrt(np.mean(samples ** 2)))
            rms_values.append(rms)

    # Normalize by the maximum RMS across all segments
    max_rms = max(rms_values) if rms_values else 0.0
    if max_rms == 0.0:
        return [0.0] * len(segments)

    return [rms / max_rms for rms in rms_values]


def combine_scores(
    config: Config,
    text: float,
    audio: float,
    llm: float | None,
) -> float:
    """Combine text, audio, and optional LLM scores into a final clip score.

    Returns:
        text_weight * text + audio_weight * audio + llm_weight * (llm or 0.0)
        Result is always >= 0.0.

    Args:
        config: Pipeline configuration (provides score weights).
        text: Text score in [0.0, 1.0].
        audio: Audio score in [0.0, 1.0].
        llm: LLM score in [0.0, 1.0], or None if LLM scoring is disabled.

    Returns:
        A float >= 0.0 representing the combined clip score.
    """
    llm_value = llm if llm is not None else 0.0
    result = (
        config.text_weight * text
        + config.audio_weight * audio
        + config.llm_weight * llm_value
    )
    return max(0.0, result)


def compute_llm_score(config: Config, segment: Segment) -> float:
    """Query a local LLM to rate a segment's clip-worthiness.

    Sends a POST request to ``config.llm_endpoint`` with the segment text and
    parses the numeric rating (1–10) from the response.  The rating is
    normalized to [0.0, 1.0] by dividing by 10.

    Args:
        config: Pipeline configuration (provides LLM endpoint and model name).
        segment: The transcript segment to score.

    Returns:
        A float in [0.0, 1.0].

    Raises:
        LLMScoringError: If the LLM endpoint is unreachable (ConnectionError or
            Timeout).  Callers should catch this and fall back to 0.0.
    """
    prompt = (
        "Rate this video clip segment for how clip-worthy it is on a scale of "
        "1-10. Respond with only a number.\n\nSegment: " + segment.text
    )
    payload = {
        "model": config.llm_model,
        "prompt": prompt,
        "stream": False,
    }

    try:
        response = requests.post(config.llm_endpoint, json=payload, timeout=30)
    except (requests.ConnectionError, requests.Timeout) as exc:
        raise LLMScoringError(
            f"LLM endpoint unreachable at {config.llm_endpoint!r}: {exc}"
        ) from exc

    # Try to extract the number from the structured JSON field first.
    raw_value: str | None = None
    try:
        raw_value = str(response.json()["response"])
    except (ValueError, KeyError, TypeError):
        raw_value = None

    # Fall back to searching the full response text with a regex.
    if raw_value is None:
        raw_value = response.text

    match = re.search(r'\b([1-9]|10)\b', raw_value)
    if match is None:
        logging.warning(
            "LLM response contained no parseable numeric score for segment "
            "starting at %.2fs; defaulting to 0.0. Response: %r",
            segment.start,
            response.text[:200],
        )
        return 0.0

    numeric = float(match.group(1))
    return numeric / 10.0


def score_segments(
    config: Config,
    transcript: Transcript,
    wav_path: str,
) -> list[ScoredSegment]:
    """Compute a clip score for every segment in *transcript*.

    Steps:
    1. Compute ``text_score`` for each segment via :func:`compute_text_score`.
    2. Compute ``audio_score`` for all segments at once via
       :func:`compute_audio_score`.
    3. If ``config.llm_enabled``, compute ``llm_score`` for each segment via
       :func:`compute_llm_score`; catch :exc:`LLMScoringError` and use 0.0.
       Otherwise ``llm_score`` is 0.0 for all segments.
    4. Combine the three sub-scores via :func:`combine_scores`.

    Args:
        config: Pipeline configuration.
        transcript: The transcript whose segments will be scored.
        wav_path: Path to the extracted WAV file for audio scoring.

    Returns:
        A list of :class:`~pipeline.models.ScoredSegment` objects, one per
        segment in *transcript*, in the same order.
    """
    segments = transcript.segments

    # --- Text scores (one per segment) ---
    text_scores = [compute_text_score(config, seg) for seg in segments]

    # --- Audio scores (computed in one pass over the WAV) ---
    audio_scores = compute_audio_score(segments, wav_path)

    # --- LLM scores ---
    if config.llm_enabled:
        llm_scores: list[float] = []
        for seg in segments:
            try:
                llm_scores.append(compute_llm_score(config, seg))
            except LLMScoringError:
                llm_scores.append(0.0)
    else:
        llm_scores = [0.0] * len(segments)

    # --- Combine and assemble ScoredSegment objects ---
    scored: list[ScoredSegment] = []
    for seg, text_s, audio_s, llm_s in zip(segments, text_scores, audio_scores, llm_scores):
        clip_s = combine_scores(config, text_s, audio_s, llm_s)
        scored.append(
            ScoredSegment(
                segment=seg,
                text_score=text_s,
                audio_score=audio_s,
                llm_score=llm_s,
                clip_score=clip_s,
            )
        )

    return scored
