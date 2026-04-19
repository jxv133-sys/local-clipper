"""Clip Selector: ranks scored segments and selects top clips with expansion and overlap handling."""

from __future__ import annotations

import logging
import re
import time

import requests

from config import Config
from pipeline.exceptions import LLMScoringError
from pipeline.models import Clip, ScoredSegment, Transcript

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM boundary refinement
# ---------------------------------------------------------------------------

_BOUNDARY_CONTEXT_SECONDS = 30.0  # how far before/after the clip to show the LLM


def refine_clip_boundaries_with_llm(
    config: Config,
    clip: Clip,
    transcript: Transcript,
    video_duration: float,
) -> Clip:
    """Ask the LLM to pick exact start/end timestamps for a clip.

    Shows the LLM the transcript for the ±30s window around the current clip
    boundaries and asks it to choose the best start and end time from the
    available segment timestamps — or to extend in either direction if needed.

    The LLM must respond with:
        START_TIME: <seconds>
        END_TIME: <seconds>

    If parsing fails or the result is invalid, the original clip is returned
    unchanged.

    Args:
        config: Pipeline configuration (LLM endpoint, model, max_clip_duration).
        clip: The clip whose boundaries should be refined.
        transcript: Full transcript for context lookup.
        video_duration: Total video duration for clamping.

    Returns:
        A new Clip with refined boundaries, or the original clip on failure.
    """
    # Build the context window: ±30s around the current clip
    context_start = max(0.0, clip.start - _BOUNDARY_CONTEXT_SECONDS)
    context_end = min(video_duration, clip.end + _BOUNDARY_CONTEXT_SECONDS)

    # Collect all segments that fall within the context window
    context_segs = [
        seg for seg in transcript.segments
        if seg.end > context_start and seg.start < context_end
    ]

    if not context_segs:
        return clip

    # Format the transcript block.
    # Mark segments that are currently inside the clip with [IN CLIP].
    lines: list[str] = []
    for seg in context_segs:
        in_clip = seg.start >= clip.start - 0.5 and seg.end <= clip.end + 0.5
        marker = " [IN CLIP]" if in_clip else ""
        lines.append(
            f"[{seg.start:.1f}s] {seg.text.strip()}{marker}"
        )
    transcript_block = "\n".join(lines)

    # List of available start/end timestamps the LLM can choose from
    available_starts = sorted({f"{seg.start:.1f}" for seg in context_segs})
    available_ends = sorted({f"{seg.end:.1f}" for seg in context_segs})

    current_duration = clip.end - clip.start

    prompt = (
        "You are a video editor choosing the exact start and end of a highlight clip.\n\n"
        "Below is a transcript with timestamps. Lines marked [IN CLIP] are already "
        "selected. Your job is to choose the best start and end time to make a "
        f"compelling clip between {config.min_clip_duration:.0f}s and "
        f"{config.max_clip_duration:.0f}s long.\n\n"
        "RULES:\n"
        "- The clip must start at a natural beginning (before a reaction, punchline, "
        "or setup — not mid-sentence)\n"
        "- The clip must end at a natural conclusion (after the reaction lands, "
        "laughter dies down, or the moment resolves — not mid-sentence)\n"
        "- You may extend the clip earlier or later than the current [IN CLIP] range "
        "if it makes the clip more complete\n"
        f"- Minimum duration: {config.min_clip_duration:.0f}s, "
        f"Maximum: {config.max_clip_duration:.0f}s\n"
        f"- Current clip: {clip.start:.1f}s → {clip.end:.1f}s "
        f"({current_duration:.0f}s)\n\n"
        f"TRANSCRIPT (context window {context_start:.1f}s → {context_end:.1f}s):\n"
        f"{transcript_block}\n\n"
        f"Available start times: {', '.join(available_starts)}\n"
        f"Available end times: {', '.join(available_ends)}\n\n"
        "Respond in EXACTLY this format (no other text):\n"
        "START_TIME: <seconds from the available start times above>\n"
        "END_TIME: <seconds from the available end times above>\n"
        "REASON: <one sentence explaining why you chose these boundaries>"
    )

    try:
        payload = {"model": config.llm_model, "prompt": prompt, "stream": False}
        response = requests.post(config.llm_endpoint, json=payload, timeout=60)
        raw = str(response.json().get("response", ""))
    except (requests.ConnectionError, requests.Timeout, Exception) as exc:
        logger.warning("LLM boundary refinement failed for clip at %.1fs: %s", clip.start, exc)
        return clip

    start_match = re.search(r'START_TIME:\s*([\d.]+)', raw)
    end_match = re.search(r'END_TIME:\s*([\d.]+)', raw)
    reason_match = re.search(r'REASON:\s*(.+)', raw)

    if not start_match or not end_match:
        logger.warning(
            "LLM boundary refinement: could not parse START_TIME/END_TIME for clip "
            "at %.1fs. Response: %r", clip.start, raw[:300]
        )
        return clip

    try:
        new_start = float(start_match.group(1))
        new_end = float(end_match.group(1))
    except ValueError:
        return clip

    # Snap to the nearest actual segment boundary
    all_starts = [seg.start for seg in transcript.segments]
    all_ends = [seg.end for seg in transcript.segments]

    def _snap(value: float, candidates: list[float]) -> float:
        return min(candidates, key=lambda x: abs(x - value))

    new_start = _snap(new_start, all_starts)
    new_end = _snap(new_end, all_ends)

    # Validate: must be ordered, within video, within duration limits
    if new_start >= new_end:
        logger.warning(
            "LLM boundary refinement returned invalid range %.1fs → %.1fs; keeping original",
            new_start, new_end,
        )
        return clip

    new_start = max(0.0, new_start)
    new_end = min(video_duration, new_end)
    new_duration = new_end - new_start

    if new_duration > config.max_clip_duration:
        logger.warning(
            "LLM boundary refinement produced %.0fs clip (max %.0fs); keeping original",
            new_duration, config.max_clip_duration,
        )
        return clip

    reason = reason_match.group(1).strip() if reason_match else ""
    logger.info(
        "  Boundary refined: %.1fs→%.1fs (%.0fs) → %.1fs→%.1fs (%.0fs) | %s",
        clip.start, clip.end, current_duration,
        new_start, new_end, new_duration,
        reason,
    )

    return Clip(
        start=new_start,
        end=new_end,
        score=clip.score,
        rank=clip.rank,
        segment_indices=clip.segment_indices,
    )


def select_clips(
    config: Config,
    scored_segments: list[ScoredSegment],
    transcript: Transcript,
    video_duration: float,
) -> list[Clip]:
    """
    Rank segments by clip_score, select top N, expand to 20-45s,
    merge overlapping clips, and return the final Clip list sorted by rank.

    Args:
        config: Pipeline configuration (top_n_clips, min_clip_duration, max_clip_duration).
        scored_segments: Scored transcript segments to select from.
        transcript: Full transcript used for boundary-aligned expansion.
        video_duration: Total duration of the source video in seconds.

    Returns:
        List of Clip objects sorted by rank (1-based, descending score).
    """
    logger.info("ClipSelector starting — %d scored segment(s), video_duration=%.1fs",
                len(scored_segments), video_duration)
    t0 = time.time()

    if not scored_segments:
        return []

    # Step 1: Sort descending by clip_score
    sorted_segments = sorted(scored_segments, key=lambda s: s.clip_score, reverse=True)

    # Step 2: Apply minimum text score threshold to filter audio-only clips.
    # Segments whose text_score is below the threshold are deprioritised.
    # However, if filtering would leave fewer candidates than top_n_clips,
    # fall back to including the filtered-out segments so we always have
    # enough material to fill the requested clip count.
    threshold = config.min_text_score_for_selection
    above_threshold = [s for s in sorted_segments if s.text_score >= threshold]
    if len(above_threshold) >= config.top_n_clips:
        candidates = above_threshold
    else:
        # Not enough above-threshold candidates — use all segments as fallback
        candidates = sorted_segments

    # Step 3: Select top N
    top_segments = candidates[: config.top_n_clips]

    # Build a lookup: segment object -> index in transcript.segments
    # We match by identity first, then by (start, end, text) equality
    seg_to_idx: dict[int, int] = {}
    for i, ts in enumerate(transcript.segments):
        seg_to_idx[id(ts)] = i

    def find_segment_index(seg_obj) -> int:
        """Find the index of a segment in the transcript by identity or value equality."""
        # Try identity match first
        if id(seg_obj) in seg_to_idx:
            return seg_to_idx[id(seg_obj)]
        # Fall back to value equality
        for i, ts in enumerate(transcript.segments):
            if ts.start == seg_obj.start and ts.end == seg_obj.end and ts.text == seg_obj.text:
                return i
        return -1

    # Step 4: Expand each selected segment to reach min_clip_duration
    clips: list[Clip] = []
    for scored_seg in top_segments:
        seed_idx = find_segment_index(scored_seg.segment)

        # Start with the seed segment's own range
        clip_start = scored_seg.segment.start
        clip_end = scored_seg.segment.end
        included_indices: list[int] = []
        if seed_idx >= 0:
            included_indices = [seed_idx]

        # Expand outward using adjacent transcript segments.
        #
        # Gap handling: max_expansion_gap is a *hard boundary* — we will not
        # cross a gap larger than this value.  When a gap is too large we stop
        # expanding in that direction entirely (the silence represents a scene
        # or topic change we don't want to span).
        if seed_idx >= 0:
            left_idx = seed_idx - 1
            right_idx = seed_idx + 1
            left_blocked = False   # True once we hit a gap that is too large
            right_blocked = False

            while (clip_end - clip_start) < config.min_clip_duration:
                can_expand_left = (not left_blocked) and left_idx >= 0
                can_expand_right = (not right_blocked) and right_idx < len(transcript.segments)

                if not can_expand_left and not can_expand_right:
                    break

                left_seg = transcript.segments[left_idx] if can_expand_left else None
                right_seg = transcript.segments[right_idx] if can_expand_right else None

                expanded = False

                # --- Try left ---
                if can_expand_left and left_seg is not None:
                    left_gap = clip_start - left_seg.end
                    if left_gap > config.max_expansion_gap:
                        # Gap too large — permanently block this direction
                        left_blocked = True
                    else:
                        new_start = left_seg.start
                        new_duration = clip_end - new_start
                        if new_duration <= config.max_clip_duration:
                            clip_start = new_start
                            included_indices.insert(0, left_idx)
                            left_idx -= 1
                            expanded = True
                        else:
                            # Adding this segment would exceed max duration — block left
                            left_blocked = True

                # --- Try right (only if still short) ---
                if (clip_end - clip_start) < config.min_clip_duration and can_expand_right and right_seg is not None:
                    right_gap = right_seg.start - clip_end
                    if right_gap > config.max_expansion_gap:
                        # Gap too large — permanently block this direction
                        right_blocked = True
                    else:
                        new_end = right_seg.end
                        new_duration = new_end - clip_start
                        if new_duration <= config.max_clip_duration:
                            clip_end = new_end
                            included_indices.append(right_idx)
                            right_idx += 1
                            expanded = True
                        else:
                            # Adding this segment would exceed max duration — block right
                            right_blocked = True

                if not expanded:
                    break

        # Clamp to video bounds
        clip_start = max(0.0, clip_start)
        clip_end = min(video_duration, clip_end)

        clips.append(
            Clip(
                start=clip_start,
                end=clip_end,
                score=scored_seg.clip_score,
                rank=0,  # assigned later
                segment_indices=included_indices,
            )
        )

    # Step 5: Detect and handle overlaps
    clips = _resolve_overlaps(clips, config.max_clip_duration)

    # Step 6: LLM boundary refinement (if LLM enabled)
    # For each clip, ask the LLM to pick exact start/end timestamps using
    # the ±30s transcript context around the current clip boundaries.
    if config.llm_enabled:
        refined: list[Clip] = []
        for clip in clips:
            refined.append(
                refine_clip_boundaries_with_llm(config, clip, transcript, video_duration)
            )
        clips = refined
        # Re-resolve overlaps in case refinement caused new ones
        clips = _resolve_overlaps(clips, config.max_clip_duration)

    # Step 7: Assign 1-based rank by score (descending)
    clips_by_score = sorted(clips, key=lambda c: c.score, reverse=True)
    for rank, clip in enumerate(clips_by_score, start=1):
        clip.rank = rank

    # Step 8: Return sorted by rank
    result = sorted(clips_by_score, key=lambda c: c.rank)

    elapsed = time.time() - t0
    logger.info("ClipSelector complete — %d clip(s) selected in %.1fs", len(result), elapsed)
    for clip in result:
        logger.info(
            "  Clip #%d: %.1fs → %.1fs (duration=%.1fs, score=%.3f)",
            clip.rank, clip.start, clip.end, clip.end - clip.start, clip.score,
        )

    return result


def _resolve_overlaps(clips: list[Clip], max_clip_duration: float) -> list[Clip]:
    """
    Detect overlapping clips and either merge or discard the lower-scoring one.

    For each pair of overlapping clips (a.end > b.start when sorted by start):
    - If merged duration <= max_clip_duration: merge into one clip spanning both, keep higher score
    - Otherwise: discard the lower-scoring clip, keep the higher-scoring one

    Args:
        clips: List of Clip objects (may be in any order).
        max_clip_duration: Maximum allowed clip duration in seconds.

    Returns:
        List of non-overlapping Clip objects.
    """
    if not clips:
        return []

    # Sort by start time for overlap detection
    sorted_clips = sorted(clips, key=lambda c: c.start)

    result: list[Clip] = []
    current = sorted_clips[0]

    for next_clip in sorted_clips[1:]:
        if current.end > next_clip.start:
            # Overlap detected
            merged_start = min(current.start, next_clip.start)
            merged_end = max(current.end, next_clip.end)
            merged_duration = merged_end - merged_start

            if merged_duration <= max_clip_duration:
                # Merge: span both, keep higher score
                higher_score = max(current.score, next_clip.score)
                merged_indices = sorted(
                    set(current.segment_indices) | set(next_clip.segment_indices)
                )
                current = Clip(
                    start=merged_start,
                    end=merged_end,
                    score=higher_score,
                    rank=0,
                    segment_indices=merged_indices,
                )
            else:
                # Cannot merge: discard lower-scoring clip
                if current.score >= next_clip.score:
                    # Keep current, discard next_clip
                    pass
                else:
                    # Keep next_clip, discard current
                    current = next_clip
        else:
            # No overlap: emit current and move on
            result.append(current)
            current = next_clip

    result.append(current)
    return result
