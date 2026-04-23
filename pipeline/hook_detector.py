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
    
    Args:
        llm_endpoint: Ollama API endpoint
        llm_model: Model name to use
        text: Text to analyze
        
    Returns:
        (hook_score, hook_type) where score is 0.0-1.0 and type is one of:
        question, contrarian, reveal, emotional, none
    """
    import requests
    
    prompt = (
        "You are a viral content classifier. Analyze the text below and determine "
        "how likely it would stop a scrolling user within 3 seconds.\n\n"
        "SCORING GUIDE:\n"
        "  0.0-0.2  Boring filler, greetings, generic commentary — no hook\n"
        "  0.3-0.4  Mildly interesting but forgettable\n"
        "  0.5-0.6  Decent hook — some curiosity or engagement\n"
        "  0.7-0.8  Strong hook — clear tension, surprise, or emotion\n"
        "  0.9-1.0  Exceptional — would definitely stop a scroll\n\n"
        "HOOK TYPES (pick the best fit):\n"
        "  question   — poses a question or creates curiosity (e.g. 'What if I told you...')\n"
        "  contrarian — challenges a common belief (e.g. 'Everyone is wrong about...')\n"
        "  reveal     — teases a surprising fact or outcome (e.g. 'Turns out...')\n"
        "  emotional  — strong emotion: shock, joy, anger, fear (e.g. 'I can't believe...')\n"
        "  none       — no meaningful hook present\n\n"
        f"TEXT TO ANALYZE:\n\"{text}\"\n\n"
        "Respond with ONLY a JSON object. No explanation, no preamble.\n"
        "Format: {\"hook_score\": <number 0.0-1.0>, \"hook_type\": <one of the types above>}\n"
        "Example for boring text: {\"hook_score\": 0.1, \"hook_type\": \"none\"}\n"
        "Example for strong reveal: {\"hook_score\": 0.8, \"hook_type\": \"reveal\"}"
    )

    try:
        payload = {"model": llm_model, "prompt": prompt, "stream": False}
        response = requests.post(llm_endpoint, json=payload, timeout=30)
        response_data = response.json()
        raw_response = str(response_data.get("response", ""))
        
        if not raw_response.strip():
            logger.warning("LLM returned empty response for hook detection")
            return 0.0, "none"
        
        # Extract JSON — try strict match first, then looser search
        json_match = re.search(r'\{[^{}]+\}', raw_response)
        if json_match:
            try:
                hook_data = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                # Try extracting score and type individually as fallback
                score_match = re.search(r'"hook_score"\s*:\s*([0-9.]+)', raw_response)
                type_match = re.search(r'"hook_type"\s*:\s*"([^"]+)"', raw_response)
                if score_match:
                    hook_score = max(0.0, min(1.0, float(score_match.group(1))))
                    hook_type = type_match.group(1) if type_match else "none"
                    valid_types = {"question", "contrarian", "reveal", "emotional", "none"}
                    if hook_type not in valid_types:
                        hook_type = "none"
                    return hook_score, hook_type
                logger.warning("Could not parse JSON from hook response: %r", raw_response[:200])
                return 0.0, "none"

            hook_score = float(hook_data.get("hook_score", 0.0))
            hook_type = str(hook_data.get("hook_type", "none")).lower().strip()
            
            # Validate
            hook_score = max(0.0, min(1.0, hook_score))
            valid_types = {"question", "contrarian", "reveal", "emotional", "none"}
            if hook_type not in valid_types:
                hook_type = "none"
            
            return hook_score, hook_type
        else:
            # Last resort: try to pull numbers and types from free text
            score_match = re.search(r'(?:score|hook_score)[^\d]*([0-9]\.[0-9]+)', raw_response, re.I)
            type_match = re.search(r'\b(question|contrarian|reveal|emotional|none)\b', raw_response, re.I)
            if score_match:
                hook_score = max(0.0, min(1.0, float(score_match.group(1))))
                hook_type = type_match.group(1).lower() if type_match else "none"
                logger.debug("Hook parsed from free text: score=%.2f type=%s", hook_score, hook_type)
                return hook_score, hook_type
            logger.warning("Could not parse hook response: %r", raw_response[:200])
            return 0.0, "none"
            
    except (requests.ConnectionError, requests.Timeout) as exc:
        logger.warning("LLM endpoint unreachable for hook detection: %s", exc)
        return 0.0, "none"
    except (json.JSONDecodeError, ValueError, KeyError) as exc:
        logger.warning("Failed to parse LLM hook response: %s", exc)
        return 0.0, "none"


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
