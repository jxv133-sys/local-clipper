"""LLM-based hook detection for viral clip scoring.

Uses the existing Ollama LLM infrastructure to detect if text would stop
a scrolling user in 3 seconds. Analyzes transcript windows using a sliding
window approach.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config import Config
    from pipeline.models import Segment

logger = logging.getLogger(__name__)


@dataclass
class Hook:
    """A detected hook in the transcript."""
    start_time: float       # Start time of the hook window
    end_time: float         # End time of the hook window
    hook_score: float       # 0.0-1.0 score indicating hook strength
    hook_type: str          # Type: question, contrarian, reveal, emotional, none
    text: str               # The text that triggered the hook


def _call_llm_for_hook(llm_endpoint: str, llm_model: str, text: str) -> tuple[float, str]:
    """Call the LLM to detect hooks in text.

    Uses a minimal prompt optimised for small models (1B–3B parameters):
    asks only for a single integer score 1-10, then derives hook_type
    locally from heuristics. This avoids the JSON-copying problem where
    small models echo the example values instead of reasoning.

    Returns:
        (hook_score 0.0-1.0, hook_type str)
    """
    import requests

    # Minimal prompt — small models handle single-value output far better
    # than multi-field JSON. We ask for one integer and nothing else.
    prompt = (
        "Rate how likely this text would stop someone scrolling social media.\n"
        "Score 1 (boring) to 10 (would definitely stop scrolling).\n\n"
        f"Text: {text}\n\n"
        "Reply with ONLY a single integer from 1 to 10. Nothing else."
    )

    try:
        payload = {"model": llm_model, "prompt": prompt, "stream": False}
        response = requests.post(llm_endpoint, json=payload, timeout=30)
        response_data = response.json()
        raw_response = str(response_data.get("response", "")).strip()

        if not raw_response:
            logger.warning("LLM returned empty response for hook detection")
            return 0.0, "none"

        # Extract the first integer from the response
        int_match = re.search(r'\b(10|[1-9])\b', raw_response)
        if not int_match:
            logger.debug("No integer found in hook response: %r", raw_response[:100])
            return 0.0, "none"

        score_int = int(int_match.group(1))
        hook_score = (score_int - 1) / 9.0  # map 1-10 → 0.0-1.0

        # Derive hook type locally from text heuristics (no LLM needed)
        hook_type = _classify_hook_type(text)

        logger.debug("Hook score=%d/10 (%.2f) type=%s for: %r",
                     score_int, hook_score, hook_type, text[:60])
        return hook_score, hook_type

    except (requests.ConnectionError, requests.Timeout) as exc:
        logger.warning("LLM endpoint unreachable for hook detection: %s", exc)
        return 0.0, "none"
    except (ValueError, KeyError) as exc:
        logger.warning("Failed to parse LLM hook response: %s", exc)
        return 0.0, "none"


def _classify_hook_type(text: str) -> str:
    """Classify hook type from text heuristics — no LLM needed.

    Returns one of: question, contrarian, reveal, emotional, none
    """
    t = text.lower()

    # Question: ends with ? or contains question words
    if "?" in t:
        return "question"
    if any(w in t for w in ("what if", "how did", "why did", "who would", "can you", "do you")):
        return "question"

    # Contrarian: challenges common beliefs
    if any(p in t for p in ("everyone is wrong", "actually", "the truth is",
                             "nobody talks about", "stop doing", "you've been",
                             "most people don't", "unpopular opinion")):
        return "contrarian"

    # Reveal: surprising fact or outcome
    if any(p in t for p in ("turns out", "it turns out", "found out", "discovered",
                             "revealed", "the real reason", "secret", "never knew",
                             "you won't believe", "shocked", "plot twist")):
        return "reveal"

    # Emotional: strong reaction words
    if any(p in t for p in ("i can't believe", "oh my god", "no way", "insane",
                             "crazy", "unbelievable", "incredible", "amazing",
                             "terrible", "awful", "love", "hate", "scared",
                             "crying", "laughing", "hilarious", "devastating")):
        return "emotional"

    return "none"


def detect_hooks(
    config: Config,
    segments: list[Segment],
    window_size: int = 3,
    stride: int = 2,
    min_words: int = 5,
    score_threshold: float = 0.4,
) -> list[Hook]:
    """Detect hooks in transcript using sliding window approach.
    
    Args:
        config: Pipeline configuration with LLM settings
        segments: List of transcript segments
        window_size: Number of sentences per window (default: 3)
        stride: Number of sentences to slide forward (default: 2, 50% overlap)
        min_words: Minimum words required in window (default: 5)
        score_threshold: Minimum score to save hook (default: 0.4)
        
    Returns:
        List of detected hooks with score > threshold
    """
    if not config.llm_enabled:
        logger.info("LLM disabled, skipping hook detection")
        return []
    
    if not segments:
        return []
    
    hooks: list[Hook] = []
    
    # Slide window over segments
    for i in range(0, len(segments), stride):
        # Get window of segments
        window_segments = segments[i:i + window_size]
        if not window_segments:
            break
        
        # Combine text from window
        window_text = " ".join(seg.text.strip() for seg in window_segments)
        word_count = len(window_text.split())
        
        # Skip if too short
        if word_count < min_words:
            continue
        
        # Get time range
        start_time = window_segments[0].start
        end_time = window_segments[-1].end
        
        # Call LLM for hook detection
        hook_score, hook_type = _call_llm_for_hook(
            config.llm_endpoint,
            config.llm_model,
            window_text
        )
        
        # Only save if above threshold
        if hook_score >= score_threshold:
            hooks.append(Hook(
                start_time=start_time,
                end_time=end_time,
                hook_score=hook_score,
                hook_type=hook_type,
                text=window_text
            ))
            logger.info(
                "Hook detected at %.1fs-%.1fs: score=%.2f type=%s text=%r",
                start_time, end_time, hook_score, hook_type, window_text[:50]
            )
    
    logger.info("Hook detection complete: %d hooks found (threshold=%.2f)", len(hooks), score_threshold)
    return hooks


def get_hook_score_at_time(hooks: list[Hook], time: float) -> float:
    """Get the maximum hook score at a specific time.
    
    Args:
        hooks: List of detected hooks
        time: Time in seconds
        
    Returns:
        Maximum hook score at that time, or 0.0 if no hook
    """
    max_score = 0.0
    for hook in hooks:
        if hook.start_time <= time <= hook.end_time:
            max_score = max(max_score, hook.hook_score)
    return max_score


def get_hook_score_for_window(hooks: list[Hook], start_time: float, end_time: float) -> float:
    """Get the maximum hook score for a time window.
    
    Args:
        hooks: List of detected hooks
        start_time: Window start time in seconds
        end_time: Window end time in seconds
        
    Returns:
        Maximum hook score in that window, or 0.0 if no hook
    """
    max_score = 0.0
    for hook in hooks:
        # Check if hook overlaps with window
        if hook.start_time <= end_time and hook.end_time >= start_time:
            max_score = max(max_score, hook.hook_score)
    return max_score
