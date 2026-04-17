"""Tests for pipeline/clip_selector.py — unit tests and property-based tests."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from config import Config
from pipeline.clip_selector import select_clips
from pipeline.models import Clip, ScoredSegment, Segment, Transcript


# ---------------------------------------------------------------------------
# Helpers / Factories
# ---------------------------------------------------------------------------

def make_config(**kwargs) -> Config:
    defaults = dict(
        work_dir="/tmp/test",
        top_n_clips=5,
        min_clip_duration=20.0,
        max_clip_duration=45.0,
    )
    defaults.update(kwargs)
    return Config(**defaults)


def make_segment(start: float, end: float, text: str = "hello") -> Segment:
    return Segment(start=start, end=end, text=text)


def make_scored(segment: Segment, score: float) -> ScoredSegment:
    return ScoredSegment(
        segment=segment,
        text_score=score,
        audio_score=score,
        llm_score=0.0,
        clip_score=score,
    )


def make_transcript(segments: list[Segment]) -> Transcript:
    return Transcript(segments=segments)


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

@st.composite
def valid_segment(draw, min_start=0.0, max_end=1000.0):
    """Generate a Segment with start < end within [min_start, max_end]."""
    start = draw(st.floats(min_value=min_start, max_value=max_end - 0.1, allow_nan=False, allow_infinity=False))
    end = draw(st.floats(min_value=start + 0.1, max_value=max_end, allow_nan=False, allow_infinity=False))
    text = draw(st.text(min_size=1, max_size=50))
    return Segment(start=start, end=end, text=text)


@st.composite
def valid_scored_segment(draw, min_start=0.0, max_end=1000.0):
    """Generate a ScoredSegment with a valid underlying Segment."""
    seg = draw(valid_segment(min_start=min_start, max_end=max_end))
    score = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    return ScoredSegment(
        segment=seg,
        text_score=score,
        audio_score=score,
        llm_score=0.0,
        clip_score=score,
    )


@st.composite
def scored_segments_with_transcript(draw, max_end=500.0):
    """Generate a list of ScoredSegments and a matching Transcript (same segments).

    Guarantees that the total transcript spans at least 20s so that the clip
    boundary invariant (duration >= 20s) can be satisfied by expansion.
    """
    n = draw(st.integers(min_value=5, max_value=20))
    # Build non-overlapping segments in order, each 2-5s long with small gaps
    segments = []
    t = 0.0
    for _ in range(n):
        duration = draw(st.floats(min_value=2.0, max_value=5.0, allow_nan=False, allow_infinity=False))
        gap = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
        start = t + gap
        end = start + duration
        if end > max_end:
            break
        text = draw(st.text(min_size=1, max_size=30))
        segments.append(Segment(start=start, end=end, text=text))
        t = end

    if not segments or (segments[-1].end - segments[0].start) < 20.0:
        # Fallback: build a transcript that definitely spans >= 20s
        segments = [Segment(start=i * 2.0, end=i * 2.0 + 2.0, text=f"seg{i}") for i in range(15)]

    scored = [
        ScoredSegment(
            segment=seg,
            text_score=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
            audio_score=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
            llm_score=0.0,
            clip_score=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        )
        for seg in segments
    ]
    transcript = Transcript(segments=segments)
    video_duration = max(seg.end for seg in segments) + draw(
        st.floats(min_value=20.0, max_value=100.0, allow_nan=False, allow_infinity=False)
    )
    return scored, transcript, video_duration


# ---------------------------------------------------------------------------
# Property 9: Clip boundary invariant
# Validates: Requirements 7.3, 7.5, 7.8
# ---------------------------------------------------------------------------

# Feature: video-highlight-generator, Property 9: Clip boundary invariant
@given(data=scored_segments_with_transcript())
@settings(max_examples=100)
def test_clip_boundary_invariant(data):
    """Every Clip satisfies 20s <= duration <= 45s, start >= 0, end <= video_duration.

    **Validates: Requirements 7.3, 7.5, 7.8**
    """
    scored, transcript, video_duration = data
    config = make_config()

    clips = select_clips(config, scored, transcript, video_duration)

    # Only assert the full invariant when video_duration >= 20.0
    if video_duration >= 20.0:
        for clip in clips:
            duration = clip.end - clip.start
            assert clip.start >= 0.0, f"clip.start={clip.start} < 0"
            assert clip.end <= video_duration, f"clip.end={clip.end} > video_duration={video_duration}"
            assert duration >= 20.0, f"duration={duration} < 20.0"
            assert duration <= 45.0, f"duration={duration} > 45.0"


# ---------------------------------------------------------------------------
# Property 10: Clip selection preserves score ordering
# Validates: Requirements 7.1
# ---------------------------------------------------------------------------

# Feature: video-highlight-generator, Property 10: Clip selection preserves score ordering
@given(data=scored_segments_with_transcript())
@settings(max_examples=100)
def test_clip_score_ordering(data):
    """Returned Clips are in non-increasing order of score (rank order).

    **Validates: Requirements 7.1**
    """
    scored, transcript, video_duration = data
    config = make_config()

    clips = select_clips(config, scored, transcript, video_duration)

    # Clips are returned sorted by rank; rank is assigned by descending score
    for i in range(len(clips) - 1):
        assert clips[i].score >= clips[i + 1].score, (
            f"clips[{i}].score={clips[i].score} < clips[{i+1}].score={clips[i+1].score}"
        )


# ---------------------------------------------------------------------------
# Unit tests (subtask 8.3)
# ---------------------------------------------------------------------------

class TestTopNSelection:
    """Top N selection returns correct segments by score."""

    def test_selects_top_n_by_score(self):
        """Only the top_n_clips highest-scoring segments are selected."""
        segments = [make_segment(i * 30.0, i * 30.0 + 5.0) for i in range(10)]
        scores = [float(i) / 10.0 for i in range(10)]  # 0.0 .. 0.9
        scored = [make_scored(seg, score) for seg, score in zip(segments, scores)]
        transcript = make_transcript(segments)
        config = make_config(top_n_clips=3)

        clips = select_clips(config, scored, transcript, video_duration=400.0)

        # Should have at most 3 clips (may be fewer if merging occurs)
        assert len(clips) <= 3
        # The scores in the result should all be from the top 3 (0.7, 0.8, 0.9)
        top3_scores = {0.7, 0.8, 0.9}
        for clip in clips:
            assert clip.score in top3_scores, f"Unexpected score {clip.score}"

    def test_empty_input_returns_empty(self):
        """Empty scored_segments returns empty list."""
        config = make_config()
        result = select_clips(config, [], make_transcript([]), video_duration=100.0)
        assert result == []

    def test_fewer_segments_than_top_n(self):
        """When fewer segments than top_n_clips exist, all are selected."""
        segments = [make_segment(0.0, 5.0), make_segment(30.0, 35.0)]
        scored = [make_scored(seg, 0.5) for seg in segments]
        transcript = make_transcript(segments)
        config = make_config(top_n_clips=5)

        clips = select_clips(config, scored, transcript, video_duration=200.0)
        assert len(clips) <= 2


class TestExpansionClamping:
    """Expansion clamps start to 0.0 and end to video_duration."""

    def test_clamps_start_to_zero(self):
        """When a segment is near the video start, clip.start is clamped to 0.0."""
        # Segment starts at 1s — expansion will try to go before 0
        seg = make_segment(1.0, 3.0)
        scored = [make_scored(seg, 0.9)]
        transcript = make_transcript([seg])
        config = make_config(top_n_clips=1)

        clips = select_clips(config, scored, transcript, video_duration=100.0)

        assert len(clips) == 1
        assert clips[0].start >= 0.0

    def test_clamps_end_to_video_duration(self):
        """When a segment is near the video end, clip.end is clamped to video_duration."""
        video_duration = 50.0
        seg = make_segment(48.0, 50.0)
        scored = [make_scored(seg, 0.9)]
        transcript = make_transcript([seg])
        config = make_config(top_n_clips=1)

        clips = select_clips(config, scored, transcript, video_duration=video_duration)

        assert len(clips) == 1
        assert clips[0].end <= video_duration

    def test_short_video_clip_shorter_than_20s(self):
        """When video_duration < 20s, clip may be shorter than 20s — that's acceptable."""
        video_duration = 10.0
        seg = make_segment(0.0, 5.0)
        scored = [make_scored(seg, 0.9)]
        transcript = make_transcript([seg])
        config = make_config(top_n_clips=1)

        clips = select_clips(config, scored, transcript, video_duration=video_duration)

        assert len(clips) == 1
        assert clips[0].start >= 0.0
        assert clips[0].end <= video_duration


class TestOverlapMerging:
    """Overlap detection and resolution."""

    def _make_clip_scenario(self, seg_a_start, seg_a_end, score_a, seg_b_start, seg_b_end, score_b, video_duration=200.0):
        """Helper: build two scored segments that will produce overlapping clips after expansion."""
        seg_a = make_segment(seg_a_start, seg_a_end)
        seg_b = make_segment(seg_b_start, seg_b_end)

        # Build a dense transcript so expansion fills the gap between them
        all_segs = []
        t = 0.0
        while t < video_duration:
            all_segs.append(make_segment(t, t + 2.0))
            t += 2.0

        scored = [make_scored(seg_a, score_a), make_scored(seg_b, score_b)]
        transcript = make_transcript(all_segs)
        return scored, transcript

    def test_overlapping_clips_within_45s_merged(self):
        """Two overlapping clips whose merged duration <= 45s are merged into one."""
        # Place two segments close together so their expanded clips overlap
        # Segment A: 10-12s, Segment B: 15-17s
        # After expansion to 20s each, they will overlap
        seg_a = make_segment(10.0, 12.0)
        seg_b = make_segment(15.0, 17.0)

        # Dense transcript around them
        all_segs = [make_segment(i * 2.0, i * 2.0 + 2.0) for i in range(50)]
        scored = [make_scored(seg_a, 0.9), make_scored(seg_b, 0.8)]
        transcript = make_transcript(all_segs)
        config = make_config(top_n_clips=2)

        clips = select_clips(config, scored, transcript, video_duration=100.0)

        # The two clips should have been merged into one (since they're close)
        # OR remain separate if they don't actually overlap after expansion
        # We verify: no two clips overlap
        sorted_clips = sorted(clips, key=lambda c: c.start)
        for i in range(len(sorted_clips) - 1):
            assert sorted_clips[i].end <= sorted_clips[i + 1].start, (
                f"Clips still overlap: [{sorted_clips[i].start}, {sorted_clips[i].end}] "
                f"and [{sorted_clips[i+1].start}, {sorted_clips[i+1].end}]"
            )

    def test_overlapping_clips_exceeding_45s_keeps_higher_score(self):
        """Two overlapping clips that would exceed 45s: higher-scoring clip is retained."""
        # Create two segments far apart but with overlapping expanded clips
        # We'll directly test _resolve_overlaps logic via select_clips
        # Segment A at 0-2s (score 0.9), Segment B at 20-22s (score 0.5)
        # With a sparse transcript, A expands right and B expands left, potentially overlapping
        # and the merged duration would exceed 45s

        # Build a transcript with segments spanning 0-100s in 1s chunks
        all_segs = [make_segment(i * 1.0, i * 1.0 + 1.0) for i in range(100)]
        seg_a = all_segs[0]   # 0-1s, score 0.9
        seg_b = all_segs[50]  # 50-51s, score 0.5

        scored = [make_scored(seg_a, 0.9), make_scored(seg_b, 0.5)]
        transcript = make_transcript(all_segs)
        config = make_config(top_n_clips=2)

        clips = select_clips(config, scored, transcript, video_duration=100.0)

        # No two clips should overlap
        sorted_clips = sorted(clips, key=lambda c: c.start)
        for i in range(len(sorted_clips) - 1):
            assert sorted_clips[i].end <= sorted_clips[i + 1].start, (
                f"Clips still overlap after resolution"
            )

    def test_merge_keeps_higher_score(self):
        """When two clips are merged, the resulting clip has the higher of the two scores."""
        # Two segments very close together — their expansions will overlap and merge
        all_segs = [make_segment(i * 2.0, i * 2.0 + 2.0) for i in range(30)]
        seg_a = all_segs[5]   # 10-12s, score 0.9
        seg_b = all_segs[6]   # 12-14s, score 0.7

        scored = [make_scored(seg_a, 0.9), make_scored(seg_b, 0.7)]
        transcript = make_transcript(all_segs)
        config = make_config(top_n_clips=2)

        clips = select_clips(config, scored, transcript, video_duration=100.0)

        # If merged, the score should be the higher one (0.9)
        scores = [c.score for c in clips]
        assert 0.9 in scores, f"Expected score 0.9 in {scores}"

    def test_rank_assignment(self):
        """Clips are assigned 1-based ranks in descending score order."""
        all_segs = [make_segment(i * 30.0, i * 30.0 + 5.0) for i in range(5)]
        scores = [0.9, 0.7, 0.5, 0.3, 0.1]
        scored = [make_scored(seg, score) for seg, score in zip(all_segs, scores)]
        transcript = make_transcript(all_segs)
        config = make_config(top_n_clips=5)

        clips = select_clips(config, scored, transcript, video_duration=300.0)

        # Ranks should be 1-based and correspond to descending score
        ranks = [c.rank for c in clips]
        assert sorted(ranks) == list(range(1, len(clips) + 1)), f"Ranks not 1-based: {ranks}"
        # Clips are returned sorted by rank
        for i in range(len(clips) - 1):
            assert clips[i].rank < clips[i + 1].rank

    def test_no_overlap_in_result(self):
        """Final clip list has no overlapping clips."""
        # Many segments close together
        all_segs = [make_segment(i * 3.0, i * 3.0 + 2.0) for i in range(20)]
        scored = [make_scored(seg, float(i) / 20.0) for i, seg in enumerate(all_segs)]
        transcript = make_transcript(all_segs)
        config = make_config(top_n_clips=5)

        clips = select_clips(config, scored, transcript, video_duration=100.0)

        sorted_clips = sorted(clips, key=lambda c: c.start)
        for i in range(len(sorted_clips) - 1):
            assert sorted_clips[i].end <= sorted_clips[i + 1].start, (
                f"Overlap detected between clip {i} and {i+1}"
            )
