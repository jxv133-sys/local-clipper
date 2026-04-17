"""Clip Selector: ranks scored segments and selects top clips with expansion and overlap handling."""

from __future__ import annotations

from config import Config
from pipeline.models import Clip, ScoredSegment, Transcript


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
    if not scored_segments:
        return []

    # Step 1: Sort descending by clip_score
    sorted_segments = sorted(scored_segments, key=lambda s: s.clip_score, reverse=True)

    # Step 2: Select top N
    top_segments = sorted_segments[: config.top_n_clips]

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

    # Step 3: Expand each selected segment to reach min_clip_duration
    clips: list[Clip] = []
    for scored_seg in top_segments:
        seed_idx = find_segment_index(scored_seg.segment)

        # Start with the seed segment's own range
        clip_start = scored_seg.segment.start
        clip_end = scored_seg.segment.end
        included_indices: list[int] = []
        if seed_idx >= 0:
            included_indices = [seed_idx]

        # Expand outward using adjacent transcript segments
        if seed_idx >= 0:
            left_idx = seed_idx - 1
            right_idx = seed_idx + 1

            while (clip_end - clip_start) < config.min_clip_duration:
                can_expand_left = left_idx >= 0
                can_expand_right = right_idx < len(transcript.segments)

                if not can_expand_left and not can_expand_right:
                    break

                # Determine which direction to expand
                # Prefer the direction that adds more duration, or alternate
                left_seg = transcript.segments[left_idx] if can_expand_left else None
                right_seg = transcript.segments[right_idx] if can_expand_right else None

                # Try to expand in the direction that keeps us within max_clip_duration
                expanded = False

                # Try left first if it would not exceed max duration
                if can_expand_left and left_seg is not None:
                    new_start = left_seg.start
                    new_duration = clip_end - new_start
                    if new_duration <= config.max_clip_duration:
                        clip_start = new_start
                        included_indices.insert(0, left_idx)
                        left_idx -= 1
                        expanded = True

                # Try right if we still need more duration
                if (clip_end - clip_start) < config.min_clip_duration and can_expand_right and right_seg is not None:
                    new_end = right_seg.end
                    new_duration = new_end - clip_start
                    if new_duration <= config.max_clip_duration:
                        clip_end = new_end
                        included_indices.append(right_idx)
                        right_idx += 1
                        expanded = True

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

    # Step 4: Detect and handle overlaps
    clips = _resolve_overlaps(clips, config.max_clip_duration)

    # Step 5: Assign 1-based rank by score (descending)
    clips_by_score = sorted(clips, key=lambda c: c.score, reverse=True)
    for rank, clip in enumerate(clips_by_score, start=1):
        clip.rank = rank

    # Step 6: Return sorted by rank
    return sorted(clips_by_score, key=lambda c: c.rank)


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
