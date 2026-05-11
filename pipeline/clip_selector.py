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

_BOUNDARY_CONTEXT_SECONDS = 45.0  # how far before/after the clip to show the LLM


def refine_clip_boundaries_with_llm(
    config: Config,
    clip: Clip,
    transcript: Transcript,
    video_duration: float,
    wav_path: str | None = None,
) -> Clip:
    """Ask the LLM to pick exact start/end timestamps for a clip using the Setup → Moment → Reaction arc.

    Shows the LLM the transcript for the ±45s window around the current clip
    boundaries and asks it to identify the three-part narrative arc:
    - Setup: context before the moment (5–10s)
    - Moment: the peak event (10–30s)
    - Reaction: the streamer's response (5–10s)

    The LLM must respond with:
        START_TIME: <seconds>
        END_TIME: <seconds>
        REASON: <explanation>

    If parsing fails or the result is invalid, the original clip is returned
    unchanged.

    Args:
        config: Pipeline configuration (LLM endpoint, model, min/max_clip_duration).
        clip: The clip whose boundaries should be refined.
        transcript: Full transcript for context lookup.
        video_duration: Total video duration for clamping.
        wav_path: Optional path to audio WAV file for natural pause detection.

    Returns:
        A new Clip with refined boundaries, or the original clip on failure.
    """
    # Build the context window: ±45s around the current clip
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
    # Filter to only include combinations that respect max_clip_duration
    available_starts = sorted({seg.start for seg in context_segs})
    available_ends = sorted({seg.end for seg in context_segs})
    
    # Filter available_ends to only include those that would create valid clips
    # For each potential start time, only allow end times that result in clips <= max_clip_duration
    valid_end_times = set()
    for start_time in available_starts:
        for end_time in available_ends:
            if end_time > start_time:
                duration = end_time - start_time
                if config.min_clip_duration <= duration <= config.max_clip_duration:
                    valid_end_times.add(end_time)
    
    # Convert to formatted strings for the prompt
    available_starts_str = sorted({f"{t:.1f}" for t in available_starts})
    available_ends_str = sorted({f"{t:.1f}" for t in valid_end_times})
    
    # If no valid end times exist, return original clip
    if not available_ends_str:
        logger.warning(
            "LLM boundary refinement: no valid end times found within duration constraints "
            "for clip at %.1fs", clip.start
        )
        return clip

    current_duration = clip.end - clip.start

    prompt = (
        "You are a video editor creating a YouTube Shorts clip from a gaming stream. "
        "Your goal is to identify and preserve the Setup → Moment → Reaction narrative arc.\n\n"
        "Below is a transcript with timestamps. Lines marked [IN CLIP] are currently selected. "
        "Your job is to identify the three-part arc and set boundaries that capture all three parts:\n\n"
        "1. SETUP (5–10s before the moment): What's happening in the stream — gives context\n"
        "2. MOMENT (10–30s): The funny/scary/impressive event — the reason the clip exists\n"
        "3. REACTION (5–10s after the moment): The streamer's response — what viewers share\n\n"
        "INSTRUCTIONS:\n"
        "- Identify where the SETUP begins (context before the moment)\n"
        "- Identify where the MOMENT peaks (the highlight event)\n"
        "- Identify where the REACTION ends (after the streamer responds)\n"
        "- Set START_TIME at the beginning of the setup\n"
        "- SET END_TIME after the reaction resolves\n"
        "- If the reaction is missing or cut off, extend END_TIME forward to capture it\n"
        "- Setup should be 5–10s before the moment. Reaction should be 5–10s after the moment.\n\n"
        "CONSTRAINTS:\n"
        f"- The clip MUST be at least {config.min_clip_duration:.0f}s long. Do not shrink below this.\n"
        f"- The clip MUST NOT exceed {config.max_clip_duration:.0f}s.\n"
        f"- Current clip: {clip.start:.1f}s → {clip.end:.1f}s ({current_duration:.0f}s)\n"
        f"- IMPORTANT: Only use END_TIME values from the 'Available end times' list below.\n"
        f"  These end times have been pre-filtered to ensure the clip stays within {config.max_clip_duration:.0f}s.\n\n"
        f"TRANSCRIPT (context window {context_start:.1f}s → {context_end:.1f}s):\n"
        f"{transcript_block}\n\n"
        f"Available start times: {', '.join(available_starts_str)}\n"
        f"Available end times: {', '.join(available_ends_str)}\n\n"
        "Respond in EXACTLY this format (no other text):\n"
        "START_TIME: <seconds from the available start times above>\n"
        "END_TIME: <seconds from the available end times above>\n"
        "REASON: <one sentence explaining the arc: where setup/moment/reaction are>"
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

    # Detect natural pauses and snap LLM-suggested boundaries to nearest pauses
    if wav_path:
        from pipeline.pause_detector import detect_natural_pauses, snap_to_nearest_pause  # noqa: PLC0415
        
        try:
            pauses = detect_natural_pauses(transcript, wav_path, silence_threshold=0.5)
            logger.info(
                "  Detected %d natural pauses for boundary refinement (clip at %.1fs)",
                len(pauses),
                clip.start,
            )
            
            # Snap both start and end to nearest pauses within 3.0 seconds
            original_start = new_start
            original_end = new_end
            new_start = snap_to_nearest_pause(new_start, pauses, max_distance=3.0)
            new_end = snap_to_nearest_pause(new_end, pauses, max_distance=3.0)
            
            # Log adjustments if boundaries were snapped
            if new_start != original_start or new_end != original_end:
                logger.info(
                    "  Snapped boundaries to natural pauses: start %.1fs→%.1fs, end %.1fs→%.1fs",
                    original_start, new_start, original_end, new_end,
                )
        except Exception as exc:
            logger.warning(
                "Natural pause detection failed for clip at %.1fs: %s. Continuing without pause snapping.",
                clip.start,
                exc,
            )

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
    video_path: str | None = None,
    wav_path: str | None = None,
) -> list[Clip]:
    """
    Rank segments by clip_score, select top N, expand to 20-45s,
    merge overlapping clips, and return the final Clip list sorted by rank.

    Args:
        config: Pipeline configuration (top_n_clips, min_clip_duration, max_clip_duration).
        scored_segments: Scored transcript segments to select from.
        transcript: Full transcript used for boundary-aligned expansion.
        video_duration: Total duration of the source video in seconds.
        video_path: Optional path to the source video file. Required when
            config.snap_to_scene_cuts is True; ignored otherwise.
        wav_path: Optional path to audio WAV file for natural pause detection
            during LLM boundary refinement.

    Returns:
        List of Clip objects sorted by rank (1-based, descending score).
    """
    logger.info("ClipSelector starting — %d scored segment(s), video_duration=%.1fs",
                len(scored_segments), video_duration)
    t0 = time.time()

    # Compute adaptive spacing constraint based on video duration and clip count
    from pipeline.adaptive_spacing import compute_adaptive_spacing  # noqa: PLC0415
    
    effective_spacing = compute_adaptive_spacing(
        video_duration, config.top_n_clips, config.min_clip_spacing
    )
    logger.info(
        "Adaptive spacing: video_duration=%.1fs, top_n_clips=%d, base_spacing=%.1fs → effective_spacing=%.1fs",
        video_duration, config.top_n_clips, config.min_clip_spacing, effective_spacing,
    )

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

        # Expand using adjacent transcript segments with a biased strategy:
        #
        # Phase 1 — Forward (reaction tail): expand right from the seed end until
        #   at least min_reaction_duration seconds of content after the seed end
        #   are included, or until blocked by a gap > max_expansion_gap, the
        #   max_clip_duration ceiling, or the end of the transcript.
        #
        # Phase 2 — Backward (setup): expand left from the seed start to fill
        #   the remaining duration budget up to min_clip_duration, subject to
        #   the same gap and max_duration constraints.
        #
        # Gap handling: max_expansion_gap is a *hard boundary* — we will not
        # cross a gap larger than this value.  When a gap is too large we stop
        # expanding in that direction entirely (the silence represents a scene
        # or topic change we don't want to span).
        if seed_idx >= 0:
            seed_end = scored_seg.segment.end  # original seed end (fixed reference)

            right_idx = seed_idx + 1
            right_blocked = False

            # --- Phase 1: expand forward to capture the reaction tail ---
            while True:
                # How much content after the seed end is already included?
                reaction_captured = clip_end - seed_end
                if reaction_captured >= config.min_reaction_duration:
                    break  # reaction tail satisfied

                can_expand_right = (not right_blocked) and right_idx < len(transcript.segments)
                if not can_expand_right:
                    break  # no more content to the right — use whatever we have

                right_seg = transcript.segments[right_idx]
                right_gap = right_seg.start - clip_end
                if right_gap > config.max_expansion_gap:
                    right_blocked = True
                    break
                new_end = right_seg.end
                new_duration = new_end - clip_start
                if new_duration > config.max_clip_duration:
                    right_blocked = True
                    break
                clip_end = new_end
                included_indices.append(right_idx)
                right_idx += 1

            # --- Phase 2: expand backward to fill remaining budget (setup) ---
            left_idx = seed_idx - 1
            left_blocked = False

            while (clip_end - clip_start) < config.min_clip_duration:
                can_expand_left = (not left_blocked) and left_idx >= 0
                can_expand_right_phase2 = (not right_blocked) and right_idx < len(transcript.segments)

                if not can_expand_left and not can_expand_right_phase2:
                    break

                expanded = False

                # --- Try left (primary in phase 2) ---
                if can_expand_left:
                    left_seg = transcript.segments[left_idx]
                    left_gap = clip_start - left_seg.end
                    if left_gap > config.max_expansion_gap:
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
                            left_blocked = True

                # --- Try right as fallback (if left is blocked and still short) ---
                if (clip_end - clip_start) < config.min_clip_duration and can_expand_right_phase2:
                    right_seg = transcript.segments[right_idx]
                    right_gap = right_seg.start - clip_end
                    if right_gap > config.max_expansion_gap:
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
                            right_blocked = True

                if not expanded:
                    break

        # Apply tail padding — add a short buffer after the last segment so the
        # clip doesn't feel cut off mid-breath.  Clamped by max_clip_duration.
        tail_padding = getattr(config, 'clip_tail_padding', 1.5)
        if tail_padding > 0.0:
            padded_end = clip_end + tail_padding
            if padded_end - clip_start <= config.max_clip_duration:
                clip_end = padded_end

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
                is_audio_spike=scored_seg.is_audio_spike,
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
            candidate = refine_clip_boundaries_with_llm(
                config, clip, transcript, video_duration, wav_path
            )
            new_duration = candidate.end - candidate.start
            if new_duration < config.min_clip_duration:
                logger.warning(
                    "[ClipSelector] LLM boundary refinement produced %.0fs clip "
                    "(min %.0fs); keeping original",
                    new_duration,
                    config.min_clip_duration,
                )
                refined.append(clip)
            else:
                refined.append(candidate)
        clips = refined
        # Re-resolve overlaps in case refinement caused new ones
        clips = _resolve_overlaps(clips, config.max_clip_duration)

    # Step 7: Greedy spacing pass — ensure clips are spread across the video.
    # (deduplication runs after spacing — see Step 7b below)
    # Sort by score descending; accept a clip only if its start time is at least
    # min_clip_spacing seconds away from every already-accepted clip.
    # If the strict pass yields fewer than top_n_clips, fill remaining slots from
    # the rejected candidates in score order (fallback).
    # This runs AFTER LLM boundary refinement so refined clips respect spacing.
    clips = _apply_spacing(clips, effective_spacing, config.top_n_clips)

    # Step 7b: Transcript deduplication pass — remove clips with near-identical
    # transcript content (Jaccard similarity > dedup_similarity_threshold).
    clips = _dedup_by_transcript(clips, transcript, config.dedup_similarity_threshold)

    # Step 7c: Scene-change aware boundary snapping.
    # If enabled and a video path is available, snap each clip's start and end
    # to the nearest I-frame (scene cut) within ±2 seconds.
    if config.snap_to_scene_cuts and video_path:
        from pipeline.scene_detector import _detect_scene_cuts, snap_to_nearest_cut  # noqa: PLC0415

        snapped: list[Clip] = []
        for clip in clips:
            start_cuts = _detect_scene_cuts(video_path, clip.start, clip.start)
            end_cuts = _detect_scene_cuts(video_path, clip.end, clip.end)
            new_start = snap_to_nearest_cut(clip.start, start_cuts)
            new_end = snap_to_nearest_cut(clip.end, end_cuts)
            # Clamp to video bounds and ensure ordering is preserved
            new_start = max(0.0, new_start)
            new_end = min(video_duration, new_end)
            if new_start >= new_end:
                # Snapping produced an invalid range — keep original
                snapped.append(clip)
            else:
                snapped.append(
                    Clip(
                        start=new_start,
                        end=new_end,
                        score=clip.score,
                        rank=clip.rank,
                        segment_indices=clip.segment_indices,
                        is_audio_spike=clip.is_audio_spike,
                    )
                )
            if new_start != clip.start or new_end != clip.end:
                logger.info(
                    "[ClipSelector] Scene-cut snap: %.1fs→%.1fs became %.1fs→%.1fs",
                    clip.start, clip.end, new_start, new_end,
                )
        clips = snapped

    # Step 8: Assign 1-based rank by score (descending)
    clips_by_score = sorted(clips, key=lambda c: c.score, reverse=True)
    for rank, clip in enumerate(clips_by_score, start=1):
        clip.rank = rank

    # Step 9: Return sorted by rank
    result = sorted(clips_by_score, key=lambda c: c.rank)

    elapsed = time.time() - t0
    logger.info("ClipSelector complete — %d clip(s) selected in %.1fs", len(result), elapsed)
    for clip in result:
        logger.info(
            "  Clip #%d: %.1fs → %.1fs (duration=%.1fs, score=%.3f)",
            clip.rank, clip.start, clip.end, clip.end - clip.start, clip.score,
        )

    return result


def _clip_word_set(clip: Clip, transcript: Transcript) -> set[str]:
    """Return the set of lowercase words from all transcript segments that overlap with *clip*."""
    words: set[str] = set()
    for seg in transcript.segments:
        if seg.end > clip.start and seg.start < clip.end:
            for word in seg.text.lower().split():
                words.add(word)
    return words


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two word sets. Returns 0.0 when both are empty."""
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _dedup_by_transcript(
    clips: list[Clip],
    transcript: Transcript,
    threshold: float,
) -> list[Clip]:
    """Remove clips whose transcript content is near-identical to a higher-scoring clip.

    For each pair of clips, compute Jaccard similarity on their transcript word
    sets.  If similarity > *threshold*, discard the lower-scoring clip.

    Args:
        clips: List of Clip objects (any order).
        transcript: Full transcript used to collect overlapping segment text.
        threshold: Jaccard similarity above which a clip is considered a duplicate.

    Returns:
        Deduplicated list of Clip objects (order preserved from input).
    """
    if not clips or threshold >= 1.0:
        return clips

    # Sort by score descending so we always keep the higher-scoring clip
    by_score = sorted(clips, key=lambda c: c.score, reverse=True)

    # Pre-compute word sets
    word_sets = [_clip_word_set(clip, transcript) for clip in by_score]

    kept_indices: list[int] = []
    removed: set[int] = set()

    for i, clip in enumerate(by_score):
        if i in removed:
            continue
        for j in kept_indices:
            sim = _jaccard(word_sets[i], word_sets[j])
            if sim > threshold:
                logger.info(
                    "[ClipSelector] Clip at %.1fs removed (transcript similarity %.2f to clip at %.1fs)",
                    clip.start,
                    sim,
                    by_score[j].start,
                )
                removed.add(i)
                break
        if i not in removed:
            kept_indices.append(i)

    return [by_score[i] for i in kept_indices]


def _apply_spacing(clips: list[Clip], min_spacing: float, top_n: int) -> list[Clip]:
    """Greedy spacing pass: ensure no two accepted clips start within min_spacing seconds.

    Algorithm:
    1. Sort candidates by score descending.
    2. Accept a clip only if its start time is at least min_spacing seconds away
       from every already-accepted clip's start time.
    3. If the strict pass yields fewer than top_n clips, fill remaining slots from
       the rejected candidates in score order (fallback).

    Args:
        clips: List of Clip objects (any order).
        min_spacing: Minimum required gap (seconds) between accepted clip start times.
        top_n: Desired number of clips; fallback kicks in if strict pass yields fewer.

    Returns:
        List of Clip objects that satisfy the spacing constraint (or the best
        available subset when there are not enough spaced candidates).
    """
    if not clips or min_spacing <= 0.0:
        return clips

    # Sort by score descending so higher-scoring clips are accepted first
    by_score = sorted(clips, key=lambda c: c.score, reverse=True)

    accepted: list[Clip] = []
    rejected: list[Clip] = []

    for clip in by_score:
        too_close = any(
            abs(clip.start - acc.start) <= min_spacing
            for acc in accepted
        )
        if too_close:
            rejected.append(clip)
            # Find which accepted clip is too close for logging
            closest = min(accepted, key=lambda acc: abs(clip.start - acc.start))
            logger.info(
                "[ClipSelector] Clip at %.1fs removed by spacing pass (too close to clip at %.1fs)",
                clip.start,
                closest.start,
            )
        else:
            accepted.append(clip)

    # Fallback: if we don't have enough clips, fill from rejected in score order
    if len(accepted) < top_n:
        needed = top_n - len(accepted)
        # rejected is already in score-descending order (same sort as by_score)
        accepted.extend(rejected[:needed])

    return accepted


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
                # Preserve is_audio_spike if either clip was an audio spike
                is_spike = current.is_audio_spike or next_clip.is_audio_spike
                current = Clip(
                    start=merged_start,
                    end=merged_end,
                    score=higher_score,
                    rank=0,
                    segment_indices=merged_indices,
                    is_audio_spike=is_spike,
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
