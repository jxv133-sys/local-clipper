"""Clip Selector: ranks scored segments and selects top clips with expansion and overlap handling."""

from __future__ import annotations

import logging
import time

from config import Config
from pipeline.models import Clip, ScoredSegment, Transcript

logger = logging.getLogger(__name__)


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

    # Step 6: Assign 1-based rank by score (descending)
    clips_by_score = sorted(clips, key=lambda c: c.score, reverse=True)
    for rank, clip in enumerate(clips_by_score, start=1):
        clip.rank = rank

    # Step 7: Return sorted by rank
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
