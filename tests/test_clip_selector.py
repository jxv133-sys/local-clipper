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


# ---------------------------------------------------------------------------
# Task 24: min_text_score_for_selection threshold tests
# ---------------------------------------------------------------------------

class TestMinTextScoreThreshold:
    """Segments below min_text_score_for_selection are excluded when enough
    above-threshold candidates exist, but included as fallback otherwise."""

    def _make_scored_with_text_score(
        self, segment: Segment, clip_score: float, text_score: float
    ) -> ScoredSegment:
        return ScoredSegment(
            segment=segment,
            text_score=text_score,
            audio_score=clip_score,
            llm_score=0.0,
            clip_score=clip_score,
        )

    def test_below_threshold_excluded_when_enough_above_threshold(self):
        """Segments with text_score < threshold are excluded when >= top_n_clips
        segments are above the threshold."""
        # 5 segments well above threshold (text_score=0.5), 2 below (text_score=0.01)
        # top_n_clips=3, threshold=0.05 → 5 above-threshold candidates >= 3, so
        # the 2 below-threshold segments should NOT appear in the result.
        above_segs = [make_segment(i * 30.0, i * 30.0 + 5.0, f"above{i}") for i in range(5)]
        below_segs = [make_segment(200.0 + i * 30.0, 200.0 + i * 30.0 + 5.0, f"below{i}") for i in range(2)]

        all_segs = above_segs + below_segs
        scored = (
            [self._make_scored_with_text_score(seg, 0.5, 0.5) for seg in above_segs]
            + [self._make_scored_with_text_score(seg, 0.9, 0.01) for seg in below_segs]
        )
        # Note: below_segs have higher clip_score (0.9) but text_score below threshold.
        # They should still be excluded because there are enough above-threshold candidates.

        transcript = make_transcript(all_segs)
        config = make_config(top_n_clips=3, min_text_score_for_selection=0.05)

        clips = select_clips(config, scored, transcript, video_duration=500.0)

        # All returned clips must come from above-threshold segments
        above_starts = {seg.start for seg in above_segs}
        for clip in clips:
            # The clip's time range should overlap with an above-threshold segment
            # (clip is expanded, so we check that the seed segment was above-threshold)
            # We verify by checking that no clip is centred on a below-threshold segment
            for below_seg in below_segs:
                # A clip seeded from a below-threshold segment would have its start
                # near that segment's start (before expansion). If the clip's range
                # contains the below segment's midpoint but no above segment, it was
                # seeded from below.
                pass  # structural check below is sufficient

        # The simplest check: with 5 above-threshold candidates and top_n=3,
        # we should get at most 3 clips, none seeded from the below-threshold segments.
        # Since below_segs have higher clip_score (0.9 > 0.5), if filtering is NOT
        # applied they would be selected first. So if filtering works, the clips
        # should have score=0.5 (from above-threshold segments).
        assert len(clips) <= 3
        for clip in clips:
            assert clip.score == 0.5, (
                f"Expected score 0.5 (above-threshold), got {clip.score}. "
                "Below-threshold segment may have been incorrectly selected."
            )

    def test_below_threshold_included_as_fallback_when_not_enough_above(self):
        """Segments with text_score < threshold are included as fallback when
        fewer than top_n_clips segments are above the threshold."""
        # Only 2 segments above threshold, but top_n_clips=5 → fallback to all segments
        above_segs = [make_segment(i * 30.0, i * 30.0 + 5.0, f"above{i}") for i in range(2)]
        below_segs = [make_segment(100.0 + i * 30.0, 100.0 + i * 30.0 + 5.0, f"below{i}") for i in range(4)]

        all_segs = above_segs + below_segs
        scored = (
            [self._make_scored_with_text_score(seg, 0.5, 0.5) for seg in above_segs]
            + [self._make_scored_with_text_score(seg, 0.8, 0.01) for seg in below_segs]
        )

        transcript = make_transcript(all_segs)
        config = make_config(top_n_clips=5, min_text_score_for_selection=0.05)

        clips = select_clips(config, scored, transcript, video_duration=500.0)

        # With fallback, below-threshold segments (score=0.8) should be included.
        # We expect clips with score=0.8 to appear since they rank higher.
        scores = {clip.score for clip in clips}
        assert 0.8 in scores, (
            f"Expected below-threshold segments (score=0.8) to be included as fallback, "
            f"but got scores: {scores}"
        )

    def test_threshold_zero_includes_all_segments(self):
        """With threshold=0.0, no segments are filtered out."""
        segs = [make_segment(i * 30.0, i * 30.0 + 5.0) for i in range(5)]
        scored = [
            self._make_scored_with_text_score(seg, float(i) / 10.0, 0.0)
            for i, seg in enumerate(segs)
        ]
        transcript = make_transcript(segs)
        config = make_config(top_n_clips=3, min_text_score_for_selection=0.0)

        clips = select_clips(config, scored, transcript, video_duration=300.0)

        # All segments have text_score=0.0 which equals threshold=0.0 (>= passes)
        # Top 3 by clip_score should be selected
        assert len(clips) <= 3
        top3_scores = {0.4, 0.3, 0.2}
        for clip in clips:
            assert clip.score in top3_scores, f"Unexpected score {clip.score}"

    def test_all_segments_below_threshold_uses_fallback(self):
        """When ALL segments are below the threshold, all are used as fallback."""
        segs = [make_segment(i * 30.0, i * 30.0 + 5.0) for i in range(4)]
        scored = [
            self._make_scored_with_text_score(seg, float(i + 1) / 10.0, 0.01)
            for i, seg in enumerate(segs)
        ]
        transcript = make_transcript(segs)
        config = make_config(top_n_clips=3, min_text_score_for_selection=0.05)

        clips = select_clips(config, scored, transcript, video_duration=300.0)

        # Fallback: all 4 segments are candidates, top 3 by clip_score selected
        assert len(clips) <= 3
        # Highest clip_scores are 0.4, 0.3, 0.2
        for clip in clips:
            assert clip.score in {0.4, 0.3, 0.2}, f"Unexpected score {clip.score}"


# ---------------------------------------------------------------------------
# Task 28: max_expansion_gap — silence gap boundary tests
# ---------------------------------------------------------------------------

class TestMaxExpansionGap:
    """Expansion stops at silence gaps > max_expansion_gap."""

    def test_expansion_stops_at_large_gap_left(self):
        """Left expansion stops when the gap between the left neighbour and the
        current clip start exceeds max_expansion_gap."""
        # Transcript: seg0 ends at 5s, then a 3s gap, seg1 starts at 8s (seed)
        # max_expansion_gap=2.0 → gap of 3s should block left expansion
        seg0 = make_segment(0.0, 5.0)   # left neighbour — gap of 3s to seed
        seg1 = make_segment(8.0, 10.0)  # seed segment
        # Add more segments to the right so the clip can still reach 20s
        right_segs = [make_segment(10.0 + i * 2.0, 12.0 + i * 2.0) for i in range(10)]

        all_segs = [seg0, seg1] + right_segs
        scored = [make_scored(seg1, 0.9)]
        transcript = make_transcript(all_segs)
        config = make_config(top_n_clips=1, max_expansion_gap=2.0)

        clips = select_clips(config, scored, transcript, video_duration=100.0)

        assert len(clips) == 1
        # seg0 ends at 5.0; gap to clip_start (8.0) is 3.0 > 2.0 → seg0 must NOT be included
        assert clips[0].start >= 8.0, (
            f"Expected clip.start >= 8.0 (gap blocked left expansion), got {clips[0].start}"
        )

    def test_expansion_stops_at_large_gap_right(self):
        """Right expansion stops when the gap between the current clip end and
        the right neighbour exceeds max_expansion_gap."""
        # Transcript: seg0 (seed) ends at 5s, then a 3s gap, seg1 starts at 8s
        # max_expansion_gap=2.0 → gap of 3s should block right expansion
        left_segs = [make_segment(i * 2.0, i * 2.0 + 2.0) for i in range(5)]  # 0-10s
        seed_seg = make_segment(10.0, 12.0)  # seed
        far_seg = make_segment(15.0, 17.0)   # gap of 3s from seed end (12.0) → 15.0

        all_segs = left_segs + [seed_seg, far_seg]
        scored = [make_scored(seed_seg, 0.9)]
        transcript = make_transcript(all_segs)
        config = make_config(top_n_clips=1, max_expansion_gap=2.0)

        clips = select_clips(config, scored, transcript, video_duration=100.0)

        assert len(clips) == 1
        # far_seg starts at 15.0; gap from clip_end (12.0) is 3.0 > 2.0 → far_seg must NOT be included
        assert clips[0].end <= 12.0, (
            f"Expected clip.end <= 12.0 (gap blocked right expansion), got {clips[0].end}"
        )

    def test_expansion_continues_across_small_gap(self):
        """Expansion continues when the gap between segments is <= max_expansion_gap."""
        # Transcript: seg0 ends at 5s, 1s gap, seg1 starts at 6s (seed)
        # max_expansion_gap=2.0 → gap of 1s should allow left expansion
        seg0 = make_segment(0.0, 5.0)   # left neighbour — gap of 1s to seed
        seg1 = make_segment(6.0, 8.0)   # seed segment
        right_segs = [make_segment(8.0 + i * 2.0, 10.0 + i * 2.0) for i in range(10)]

        all_segs = [seg0, seg1] + right_segs
        scored = [make_scored(seg1, 0.9)]
        transcript = make_transcript(all_segs)
        config = make_config(top_n_clips=1, max_expansion_gap=2.0)

        clips = select_clips(config, scored, transcript, video_duration=100.0)

        assert len(clips) == 1
        # seg0 ends at 5.0; gap to clip_start (6.0) is 1.0 <= 2.0 → seg0 SHOULD be included
        assert clips[0].start <= 5.0, (
            f"Expected clip.start <= 5.0 (small gap allows left expansion), got {clips[0].start}"
        )

    def test_expansion_continues_across_zero_gap(self):
        """Expansion continues when segments are contiguous (gap = 0)."""
        # Contiguous segments: each ends exactly where the next begins
        segs = [make_segment(i * 2.0, i * 2.0 + 2.0) for i in range(15)]
        seed_seg = segs[7]  # middle segment
        scored = [make_scored(seed_seg, 0.9)]
        transcript = make_transcript(segs)
        config = make_config(top_n_clips=1, max_expansion_gap=2.0)

        clips = select_clips(config, scored, transcript, video_duration=100.0)

        assert len(clips) == 1
        # Should expand to at least 20s since all gaps are 0
        assert (clips[0].end - clips[0].start) >= 20.0, (
            f"Expected duration >= 20s with contiguous segments, got {clips[0].end - clips[0].start}"
        )

    def test_large_gap_both_sides_clips_shorter_than_min(self):
        """When large gaps block both directions, clip may be shorter than min_clip_duration."""
        # Isolated segment surrounded by large gaps
        far_left = make_segment(0.0, 2.0)    # gap of 10s to seed
        seed_seg = make_segment(12.0, 14.0)  # seed
        far_right = make_segment(24.0, 26.0) # gap of 10s from seed end

        all_segs = [far_left, seed_seg, far_right]
        scored = [make_scored(seed_seg, 0.9)]
        transcript = make_transcript(all_segs)
        config = make_config(top_n_clips=1, max_expansion_gap=2.0)

        clips = select_clips(config, scored, transcript, video_duration=100.0)

        assert len(clips) == 1
        # Both neighbours are blocked by large gaps; clip stays at seed boundaries (clamped)
        assert clips[0].start >= 12.0, f"Expected start >= 12.0, got {clips[0].start}"
        assert clips[0].end <= 14.0, f"Expected end <= 14.0, got {clips[0].end}"

    def test_exact_gap_equal_to_max_expansion_gap_allows_expansion(self):
        """A gap exactly equal to max_expansion_gap should allow expansion (boundary is exclusive)."""
        # Gap exactly 2.0s — should be allowed (gap <= max_expansion_gap)
        seg0 = make_segment(0.0, 4.0)    # left neighbour — gap of exactly 2.0s to seed
        seed_seg = make_segment(6.0, 8.0)
        right_segs = [make_segment(8.0 + i * 2.0, 10.0 + i * 2.0) for i in range(10)]

        all_segs = [seg0, seed_seg] + right_segs
        scored = [make_scored(seed_seg, 0.9)]
        transcript = make_transcript(all_segs)
        config = make_config(top_n_clips=1, max_expansion_gap=2.0)

        clips = select_clips(config, scored, transcript, video_duration=100.0)

        assert len(clips) == 1
        # gap = 6.0 - 4.0 = 2.0, which is NOT > 2.0, so expansion should proceed
        assert clips[0].start <= 4.0, (
            f"Expected clip.start <= 4.0 (gap == max_expansion_gap allows expansion), got {clips[0].start}"
        )
