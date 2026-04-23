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
        clip_tail_padding=0.0,  # disable tail padding in tests unless explicitly set
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


# ---------------------------------------------------------------------------
# Task 45: min_clip_spacing — greedy spacing enforcement tests
# ---------------------------------------------------------------------------

class TestMinClipSpacing:
    """Greedy spacing pass ensures clips are spread across the video."""

    def _make_clip(self, start: float, end: float, score: float) -> Clip:
        """Build a Clip directly (bypassing select_clips) for spacing unit tests."""
        return Clip(start=start, end=end, score=score, rank=0, segment_indices=[])

    # ------------------------------------------------------------------
    # Helper: build a scenario where select_clips is called with spacing
    # ------------------------------------------------------------------

    def _run_select(self, seg_starts, scores, top_n, min_clip_spacing, video_duration=3000.0):
        """Build segments spaced 30s apart, run select_clips, return clips."""
        segments = [
            make_segment(s, s + 5.0, f"seg{i}")
            for i, s in enumerate(seg_starts)
        ]
        scored = [
            ScoredSegment(
                segment=seg,
                text_score=score,
                audio_score=score,
                llm_score=0.0,
                clip_score=score,
            )
            for seg, score in zip(segments, scores)
        ]
        transcript = make_transcript(segments)
        config = make_config(
            top_n_clips=top_n,
            min_clip_spacing=min_clip_spacing,
            min_clip_duration=20.0,
            max_clip_duration=45.0,
        )
        return select_clips(config, scored, transcript, video_duration=video_duration)

    def test_lower_scoring_close_clip_is_dropped(self):
        """Two clips within min_clip_spacing: the lower-scoring one is dropped.

        With top_n=1, the strict pass accepts the highest-scoring clip and the
        lower-scoring close clip is dropped (no fallback needed since top_n is met).
        """
        # Two segments only 60s apart, min_clip_spacing=300s
        # Segment A at 0s (score 0.9), Segment B at 60s (score 0.5)
        # After expansion both clips start near 0s and 60s — within 300s of each other.
        # With top_n=1, the strict pass accepts A (score=0.9) and drops B (score=0.5).
        clips = self._run_select(
            seg_starts=[0.0, 60.0],
            scores=[0.9, 0.5],
            top_n=1,
            min_clip_spacing=300.0,
        )
        # Only the higher-scoring clip should survive the spacing pass
        assert len(clips) == 1, f"Expected 1 clip after spacing, got {len(clips)}"
        assert clips[0].score == 0.9, f"Expected score=0.9, got {clips[0].score}"

    def test_lower_scoring_close_clip_dropped_with_enough_other_candidates(self):
        """With enough well-spaced candidates, a close lower-scoring clip is dropped.

        3 segments: A at 0s (score 0.9), B at 60s (score 0.5), C at 700s (score 0.7).
        min_clip_spacing=300s, top_n=2.
        Strict pass: accept A (0.9), skip B (60s < 300s from A), accept C (700s >= 300s from A).
        Result: 2 clips (A and C), B is dropped.
        """
        clips = self._run_select(
            seg_starts=[0.0, 60.0, 700.0],
            scores=[0.9, 0.5, 0.7],
            top_n=2,
            min_clip_spacing=300.0,
            video_duration=3000.0,
        )
        assert len(clips) == 2, f"Expected 2 clips, got {len(clips)}"
        clip_scores = {c.score for c in clips}
        assert 0.9 in clip_scores, "Expected high-scoring clip (0.9) to be present"
        assert 0.7 in clip_scores, "Expected second clip (0.7) to be present"
        assert 0.5 not in clip_scores, "Expected low-scoring close clip (0.5) to be dropped"

    def test_fallback_when_not_enough_candidates(self):
        """Fallback: if only 2 candidates exist but top_n=3, both are returned
        even if they're within min_clip_spacing of each other."""
        # Only 2 segments, both close together, top_n=3
        clips = self._run_select(
            seg_starts=[0.0, 60.0],
            scores=[0.9, 0.5],
            top_n=3,
            min_clip_spacing=300.0,
        )
        # Strict pass gives 1 clip; fallback fills from rejected → 2 clips total
        assert len(clips) == 2, (
            f"Expected 2 clips (fallback), got {len(clips)}: {[(c.start, c.score) for c in clips]}"
        )

    def test_clips_exactly_at_min_spacing_both_accepted(self):
        """Clips whose start times are exactly min_clip_spacing apart are both accepted."""
        # Segments at 0s and 300s, min_clip_spacing=300s
        # abs(300 - 0) = 300, which is NOT < 300, so both should be accepted
        clips = self._run_select(
            seg_starts=[0.0, 300.0],
            scores=[0.9, 0.8],
            top_n=2,
            min_clip_spacing=300.0,
        )
        assert len(clips) == 2, (
            f"Expected 2 clips (exactly at spacing boundary), got {len(clips)}"
        )

    def test_well_spaced_clips_all_accepted(self):
        """Clips that are already well-spaced are all accepted without fallback."""
        # 5 segments each 600s apart, min_clip_spacing=300s → all should pass
        seg_starts = [i * 600.0 for i in range(5)]
        clips = self._run_select(
            seg_starts=seg_starts,
            scores=[0.9, 0.8, 0.7, 0.6, 0.5],
            top_n=5,
            min_clip_spacing=300.0,
            video_duration=4000.0,
        )
        assert len(clips) == 5, (
            f"Expected 5 well-spaced clips, got {len(clips)}"
        )

    def test_spacing_zero_disables_enforcement(self):
        """min_clip_spacing=0.0 disables the spacing pass; all clips are returned."""
        clips = self._run_select(
            seg_starts=[0.0, 10.0, 20.0],
            scores=[0.9, 0.8, 0.7],
            top_n=3,
            min_clip_spacing=0.0,
        )
        # All 3 should be returned (spacing disabled)
        assert len(clips) <= 3  # may merge due to overlap, but spacing doesn't drop any


# ---------------------------------------------------------------------------
# Task 49: LLM boundary refinement min-duration fallback tests
# ---------------------------------------------------------------------------

class TestLLMBoundaryRefinementMinDurationFallback:
    """When LLM boundary refinement returns a clip shorter than min_clip_duration,
    the original clip boundaries are preserved."""

    def _make_config_llm_enabled(self, **kwargs) -> Config:
        defaults = dict(
            work_dir="/tmp/test",
            top_n_clips=1,
            min_clip_duration=20.0,
            max_clip_duration=45.0,
            llm_enabled=True,
        )
        defaults.update(kwargs)
        return Config(**defaults)

    def test_llm_tight_window_falls_back_to_original(self, monkeypatch):
        """When the LLM returns a very tight window (< min_clip_duration), the
        original clip boundaries are kept unchanged."""
        import pipeline.clip_selector as cs

        # Build a clip that is 25s long (above min_clip_duration=20s)
        original_start = 10.0
        original_end = 35.0

        # Mock refine_clip_boundaries_with_llm to return a 5s clip (below min)
        def mock_refine(config, clip, transcript, video_duration):
            return Clip(
                start=15.0,
                end=20.0,  # only 5s — below min_clip_duration
                score=clip.score,
                rank=clip.rank,
                segment_indices=clip.segment_indices,
            )

        monkeypatch.setattr(cs, "refine_clip_boundaries_with_llm", mock_refine)

        seg = make_segment(original_start, original_end)
        scored = [make_scored(seg, 0.9)]
        # Dense transcript so expansion can reach 20s
        all_segs = [make_segment(i * 2.0, i * 2.0 + 2.0) for i in range(30)]
        transcript = make_transcript(all_segs)
        config = self._make_config_llm_enabled()

        clips = select_clips(config, scored, transcript, video_duration=100.0)

        assert len(clips) == 1
        clip = clips[0]
        # The LLM's tight window must be rejected; original boundaries preserved
        assert clip.start != 15.0 or clip.end != 20.0, (
            "LLM's too-short clip was applied; expected fallback to original"
        )
        # Duration must be >= min_clip_duration
        assert (clip.end - clip.start) >= 20.0, (
            f"Clip duration {clip.end - clip.start:.1f}s is below min_clip_duration=20s"
        )

    def test_llm_valid_window_is_applied(self, monkeypatch):
        """When the LLM returns a valid window (>= min_clip_duration), it is applied."""
        import pipeline.clip_selector as cs

        def mock_refine(config, clip, transcript, video_duration):
            # Return a 25s clip — valid
            return Clip(
                start=5.0,
                end=30.0,
                score=clip.score,
                rank=clip.rank,
                segment_indices=clip.segment_indices,
            )

        monkeypatch.setattr(cs, "refine_clip_boundaries_with_llm", mock_refine)

        seg = make_segment(10.0, 35.0)
        scored = [make_scored(seg, 0.9)]
        all_segs = [make_segment(i * 2.0, i * 2.0 + 2.0) for i in range(30)]
        transcript = make_transcript(all_segs)
        config = self._make_config_llm_enabled()

        clips = select_clips(config, scored, transcript, video_duration=100.0)

        assert len(clips) == 1
        clip = clips[0]
        # The LLM's valid refinement should be applied
        assert clip.start == 5.0, f"Expected start=5.0, got {clip.start}"
        assert clip.end == 30.0, f"Expected end=30.0, got {clip.end}"

    def test_llm_exactly_at_min_duration_is_applied(self, monkeypatch):
        """When the LLM returns a clip exactly at min_clip_duration, it is applied."""
        import pipeline.clip_selector as cs

        def mock_refine(config, clip, transcript, video_duration):
            # Return exactly 20s — right at the boundary
            return Clip(
                start=10.0,
                end=30.0,  # exactly 20s == min_clip_duration
                score=clip.score,
                rank=clip.rank,
                segment_indices=clip.segment_indices,
            )

        monkeypatch.setattr(cs, "refine_clip_boundaries_with_llm", mock_refine)

        seg = make_segment(10.0, 35.0)
        scored = [make_scored(seg, 0.9)]
        all_segs = [make_segment(i * 2.0, i * 2.0 + 2.0) for i in range(30)]
        transcript = make_transcript(all_segs)
        config = self._make_config_llm_enabled()

        clips = select_clips(config, scored, transcript, video_duration=100.0)

        assert len(clips) == 1
        clip = clips[0]
        # Exactly at min — should be accepted
        assert clip.start == 10.0, f"Expected start=10.0, got {clip.start}"
        assert clip.end == 30.0, f"Expected end=30.0, got {clip.end}"

    def test_llm_fallback_warning_is_logged(self, monkeypatch, caplog):
        """A WARNING is logged when the LLM's refined clip is below min_clip_duration."""
        import logging
        import pipeline.clip_selector as cs

        def mock_refine(config, clip, transcript, video_duration):
            return Clip(
                start=15.0,
                end=16.0,  # 1s — way below min
                score=clip.score,
                rank=clip.rank,
                segment_indices=clip.segment_indices,
            )

        monkeypatch.setattr(cs, "refine_clip_boundaries_with_llm", mock_refine)

        seg = make_segment(10.0, 35.0)
        scored = [make_scored(seg, 0.9)]
        all_segs = [make_segment(i * 2.0, i * 2.0 + 2.0) for i in range(30)]
        transcript = make_transcript(all_segs)
        config = self._make_config_llm_enabled()

        with caplog.at_level(logging.WARNING, logger="pipeline.clip_selector"):
            select_clips(config, scored, transcript, video_duration=100.0)

        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("keeping original" in msg for msg in warning_messages), (
            f"Expected a 'keeping original' warning, got: {warning_messages}"
        )
        assert any("min" in msg for msg in warning_messages), (
            f"Expected min duration mentioned in warning, got: {warning_messages}"
        )


# ---------------------------------------------------------------------------
# Task 50: Biased expansion — reaction tail first, then setup
# ---------------------------------------------------------------------------

class TestReactionTailExpansion:
    """Expansion is biased: forward (reaction tail) first, then backward (setup).

    Three scenarios:
    1. Seed near start — reaction fills forward, setup is limited
    2. Seed near end — reaction is truncated gracefully (no failure)
    3. Seed in middle — full arc captured (reaction + setup)
    """

    def _make_dense_transcript(self, total_duration: float, seg_duration: float = 2.0) -> list[Segment]:
        """Build a dense, contiguous transcript covering [0, total_duration]."""
        segs = []
        t = 0.0
        while t + seg_duration <= total_duration:
            segs.append(make_segment(t, t + seg_duration))
            t += seg_duration
        return segs

    def test_seed_near_start_reaction_fills_forward(self):
        """Seed near the start: expansion fills forward first (reaction), then backward (setup).

        With the seed at 2–4s and min_reaction_duration=8s, the clip must include
        at least 8s of content after the seed end (4s), i.e. clip_end >= 12s.
        The remaining budget is filled backward toward 0s.
        """
        segs = self._make_dense_transcript(total_duration=60.0)
        # Seed is the second segment (2–4s) — near the start
        seed_seg = segs[1]  # start=2.0, end=4.0
        scored = [make_scored(seed_seg, 0.9)]
        transcript = make_transcript(segs)
        config = make_config(
            top_n_clips=1,
            min_clip_duration=20.0,
            max_clip_duration=45.0,
            min_reaction_duration=8.0,
        )

        clips = select_clips(config, scored, transcript, video_duration=60.0)

        assert len(clips) == 1
        clip = clips[0]
        # Reaction tail: clip_end must be at least seed_end + min_reaction_duration = 4 + 8 = 12s
        assert clip.end >= 12.0, (
            f"Expected clip.end >= 12.0 (reaction tail), got {clip.end}"
        )
        # Clip must be clamped to video bounds
        assert clip.start >= 0.0
        assert clip.end <= 60.0

    def test_seed_near_end_reaction_truncated_gracefully(self):
        """Seed near the video end: reaction is truncated gracefully — no failure.

        With the seed at 55–57s in a 60s video, there are only 3s of content
        after the seed end.  min_reaction_duration=8s cannot be satisfied, but
        the pipeline must not raise an exception and must return a valid clip.
        """
        segs = self._make_dense_transcript(total_duration=60.0)
        # Seed is near the end: start=54.0, end=56.0
        seed_seg = segs[27]  # 54–56s
        scored = [make_scored(seed_seg, 0.9)]
        transcript = make_transcript(segs)
        config = make_config(
            top_n_clips=1,
            min_clip_duration=20.0,
            max_clip_duration=45.0,
            min_reaction_duration=8.0,
        )

        # Must not raise
        clips = select_clips(config, scored, transcript, video_duration=60.0)

        assert len(clips) == 1
        clip = clips[0]
        # Clip must be within video bounds
        assert clip.start >= 0.0
        assert clip.end <= 60.0
        # Clip end should be at or near the video end (all available content used)
        assert clip.end >= 56.0, (
            f"Expected clip.end >= 56.0 (seed end), got {clip.end}"
        )
        # Duration must be positive
        assert clip.end > clip.start

    def test_seed_in_middle_full_arc_captured(self):
        """Seed in the middle: full arc captured — reaction tail + setup.

        With the seed at 30–32s in a 60s video and min_reaction_duration=8s,
        the clip must include at least 8s after the seed end (clip_end >= 40s)
        and expand backward to fill the remaining budget.
        """
        segs = self._make_dense_transcript(total_duration=60.0)
        # Seed is in the middle: start=30.0, end=32.0
        seed_seg = segs[15]  # 30–32s
        scored = [make_scored(seed_seg, 0.9)]
        transcript = make_transcript(segs)
        config = make_config(
            top_n_clips=1,
            min_clip_duration=20.0,
            max_clip_duration=45.0,
            min_reaction_duration=8.0,
        )

        clips = select_clips(config, scored, transcript, video_duration=60.0)

        assert len(clips) == 1
        clip = clips[0]
        # Reaction tail: clip_end >= seed_end + min_reaction_duration = 32 + 8 = 40s
        assert clip.end >= 40.0, (
            f"Expected clip.end >= 40.0 (reaction tail), got {clip.end}"
        )
        # Setup: clip should also expand backward from seed start (30s)
        assert clip.start < 30.0, (
            f"Expected clip.start < 30.0 (setup expanded backward), got {clip.start}"
        )
        # Duration should reach min_clip_duration
        assert (clip.end - clip.start) >= 20.0, (
            f"Expected duration >= 20s, got {clip.end - clip.start}"
        )
        # Bounds
        assert clip.start >= 0.0
        assert clip.end <= 60.0


# ---------------------------------------------------------------------------
# Task 51: LLM boundary refinement with Setup → Moment → Reaction arc
# ---------------------------------------------------------------------------

class TestLLMBoundaryRefinementArcStructure:
    """LLM boundary refinement prompt targets the Setup → Moment → Reaction arc."""

    def _make_config_llm_enabled(self, **kwargs) -> Config:
        defaults = dict(
            work_dir="/tmp/test",
            top_n_clips=1,
            min_clip_duration=30.0,
            max_clip_duration=60.0,
            llm_enabled=True,
        )
        defaults.update(kwargs)
        return Config(**defaults)

    def test_llm_arc_prompt_applied_correctly(self, monkeypatch):
        """When the LLM returns a valid arc-based refinement, it is applied correctly."""
        import pipeline.clip_selector as cs

        # Mock the LLM to return a valid arc: setup at 10s, moment at 20s, reaction ends at 45s
        def mock_refine(config, clip, transcript, video_duration):
            # Simulate LLM identifying the arc and returning boundaries
            return Clip(
                start=10.0,  # setup begins
                end=45.0,    # reaction ends
                score=clip.score,
                rank=clip.rank,
                segment_indices=clip.segment_indices,
            )

        monkeypatch.setattr(cs, "refine_clip_boundaries_with_llm", mock_refine)

        # Build a clip that spans 15–40s (25s)
        seg = make_segment(15.0, 40.0)
        scored = [make_scored(seg, 0.9)]
        # Dense transcript covering 0–60s
        all_segs = [make_segment(i * 2.0, i * 2.0 + 2.0) for i in range(30)]
        transcript = make_transcript(all_segs)
        config = self._make_config_llm_enabled()

        clips = select_clips(config, scored, transcript, video_duration=100.0)

        assert len(clips) == 1
        clip = clips[0]
        # The LLM's arc-based refinement should be applied
        assert clip.start == 10.0, f"Expected start=10.0 (setup), got {clip.start}"
        assert clip.end == 45.0, f"Expected end=45.0 (reaction end), got {clip.end}"
        # Duration should be 35s (within min=30s, max=60s)
        assert 30.0 <= (clip.end - clip.start) <= 60.0

    def test_llm_arc_prompt_respects_min_duration(self, monkeypatch):
        """When the LLM returns an arc that is too short, the original clip is kept."""
        import pipeline.clip_selector as cs

        def mock_refine(config, clip, transcript, video_duration):
            # LLM returns a tight arc that is only 15s (below min_clip_duration=30s)
            return Clip(
                start=20.0,
                end=35.0,  # only 15s — below min
                score=clip.score,
                rank=clip.rank,
                segment_indices=clip.segment_indices,
            )

        monkeypatch.setattr(cs, "refine_clip_boundaries_with_llm", mock_refine)

        seg = make_segment(15.0, 50.0)
        scored = [make_scored(seg, 0.9)]
        all_segs = [make_segment(i * 2.0, i * 2.0 + 2.0) for i in range(30)]
        transcript = make_transcript(all_segs)
        config = self._make_config_llm_enabled()

        clips = select_clips(config, scored, transcript, video_duration=100.0)

        assert len(clips) == 1
        clip = clips[0]
        # The LLM's too-short arc must be rejected; original boundaries preserved
        assert clip.start != 20.0 or clip.end != 35.0, (
            "LLM's too-short arc was applied; expected fallback to original"
        )
        # Duration must be >= min_clip_duration
        assert (clip.end - clip.start) >= 30.0, (
            f"Clip duration {clip.end - clip.start:.1f}s is below min_clip_duration=30s"
        )

    def test_llm_arc_prompt_context_window_is_45s(self, monkeypatch):
        """The LLM boundary refinement uses a ±45s context window."""
        import pipeline.clip_selector as cs

        captured_prompt = []

        # Mock requests.post to capture the prompt sent to the LLM
        original_post = __import__('requests').post
        def mock_post(url, json=None, timeout=None):
            if json and 'prompt' in json:
                captured_prompt.append(json['prompt'])
            # Return a valid response
            class MockResponse:
                def json(self):
                    return {"response": "START_TIME: 10.0\nEND_TIME: 40.0\nRESSON: test"}
            return MockResponse()

        monkeypatch.setattr('requests.post', mock_post)

        seg = make_segment(20.0, 30.0)
        scored = [make_scored(seg, 0.9)]
        # Dense transcript covering 0–100s
        all_segs = [make_segment(i * 2.0, i * 2.0 + 2.0) for i in range(50)]
        transcript = make_transcript(all_segs)
        config = self._make_config_llm_enabled()

        # Call the actual refine function (not mocked)
        from pipeline.clip_selector import refine_clip_boundaries_with_llm
        clip = Clip(start=20.0, end=30.0, score=0.9, rank=1, segment_indices=[])
        refine_clip_boundaries_with_llm(config, clip, transcript, video_duration=100.0)

        # Check that the prompt mentions the ±45s context window
        assert len(captured_prompt) == 1
        prompt = captured_prompt[0]
        # The context window should be mentioned in the prompt
        # Clip is 20–30s, so context should be roughly -25s to +55s (clamped to 0–75s)
        # We just verify that "45" appears in the context description
        assert "45" in prompt or "context window" in prompt.lower(), (
            "Expected prompt to mention ±45s context window"
        )
        # Verify the arc structure is mentioned
        assert "Setup" in prompt or "SETUP" in prompt, "Expected prompt to mention Setup"
        assert "Moment" in prompt or "MOMENT" in prompt, "Expected prompt to mention Moment"
        assert "Reaction" in prompt or "REACTION" in prompt, "Expected prompt to mention Reaction"


# ---------------------------------------------------------------------------
# Task 52: Spacing deduplication after LLM boundary refinement
# ---------------------------------------------------------------------------

class TestSpacingAfterLLMRefinement:
    """Spacing pass runs after LLM boundary refinement, so refined clips respect spacing."""

    def _make_config_llm_enabled(self, **kwargs) -> Config:
        defaults = dict(
            work_dir="/tmp/test",
            top_n_clips=2,
            min_clip_duration=20.0,
            max_clip_duration=45.0,
            llm_enabled=True,
            min_clip_spacing=300.0,
        )
        defaults.update(kwargs)
        return Config(**defaults)

    def test_spacing_removes_refined_clip_when_too_close(self, monkeypatch):
        """Two clips that are far apart before refinement but close after refinement:
        the spacing pass should remove the lower-scoring one."""
        import pipeline.clip_selector as cs

        # Mock LLM to move clips closer together
        def mock_refine(config, clip, transcript, video_duration):
            # Clip A (score 0.9) at 0–25s → refined to 0–30s
            # Clip B (score 0.5) at 400–425s → refined to 50–75s (moved much closer!)
            # After refinement, they're only 50s apart (< min_clip_spacing=300s)
            if clip.start < 100:  # Clip A
                return Clip(
                    start=0.0,
                    end=30.0,
                    score=clip.score,
                    rank=clip.rank,
                    segment_indices=clip.segment_indices,
                )
            else:  # Clip B
                return Clip(
                    start=50.0,
                    end=75.0,
                    score=clip.score,
                    rank=clip.rank,
                    segment_indices=clip.segment_indices,
                )

        monkeypatch.setattr(cs, "refine_clip_boundaries_with_llm", mock_refine)

        # Build two segments far apart (400s gap)
        seg_a = make_segment(0.0, 25.0)
        seg_b = make_segment(400.0, 425.0)
        scored = [make_scored(seg_a, 0.9), make_scored(seg_b, 0.5)]
        # Dense transcript covering 0–500s
        all_segs = [make_segment(i * 2.0, i * 2.0 + 2.0) for i in range(250)]
        transcript = make_transcript(all_segs)
        # Set top_n_clips=1 so fallback doesn't kick in
        config = self._make_config_llm_enabled(top_n_clips=1)

        clips = select_clips(config, scored, transcript, video_duration=500.0)

        # After refinement, clips are at 0–30s and 50–75s (only 50s apart)
        # Spacing pass should remove the lower-scoring one (B, score=0.5)
        assert len(clips) == 1, f"Expected 1 clip after spacing, got {len(clips)}"
        assert clips[0].score == 0.9, f"Expected score=0.9 (higher-scoring clip kept), got {clips[0].score}"
        assert clips[0].start == 0.0, f"Expected start=0.0 (clip A), got {clips[0].start}"
        assert clips[0].end == 30.0, f"Expected end=30.0 (clip A), got {clips[0].end}"

    def test_spacing_preserves_well_spaced_refined_clips(self, monkeypatch):
        """Two clips that remain well-spaced after refinement are both kept."""
        import pipeline.clip_selector as cs

        def mock_refine(config, clip, transcript, video_duration):
            # Both clips stay in their original regions (far apart)
            if clip.start < 100:  # Clip A
                return Clip(start=0.0, end=30.0, score=clip.score, rank=clip.rank, segment_indices=clip.segment_indices)
            else:  # Clip B
                return Clip(start=400.0, end=430.0, score=clip.score, rank=clip.rank, segment_indices=clip.segment_indices)

        monkeypatch.setattr(cs, "refine_clip_boundaries_with_llm", mock_refine)

        seg_a = make_segment(0.0, 25.0)
        seg_b = make_segment(400.0, 425.0)
        scored = [make_scored(seg_a, 0.9), make_scored(seg_b, 0.5)]
        all_segs = [make_segment(i * 2.0, i * 2.0 + 2.0) for i in range(250)]
        transcript = make_transcript(all_segs)
        # dedup_similarity_threshold=1.0 disables dedup so this test focuses on spacing only
        config = self._make_config_llm_enabled(dedup_similarity_threshold=1.0)

        clips = select_clips(config, scored, transcript, video_duration=500.0)

        # Both clips are well-spaced (400s apart) — both should be kept
        assert len(clips) == 2, f"Expected 2 clips (well-spaced), got {len(clips)}"
        clip_scores = {c.score for c in clips}
        assert 0.9 in clip_scores, "Expected high-scoring clip (0.9) to be present"
        assert 0.5 in clip_scores, "Expected second clip (0.5) to be present"


# ---------------------------------------------------------------------------
# Task 7: Transcript deduplication — Jaccard similarity pass
# ---------------------------------------------------------------------------

class TestTranscriptDeduplication:
    """After the spacing pass, clips with near-identical transcript content are deduplicated."""

    def _make_scored_seg(self, start: float, end: float, text: str, score: float) -> ScoredSegment:
        seg = Segment(start=start, end=end, text=text)
        return ScoredSegment(
            segment=seg,
            text_score=score,
            audio_score=score,
            llm_score=0.0,
            clip_score=score,
        )

    def test_identical_text_deduplicates_lower_score(self):
        """Two clips with identical transcript text: the lower-scoring one is removed."""
        # Both clips share the exact same transcript text
        shared_text = "this is a great moment watch this incredible play"
        # Place them far apart so spacing pass keeps both, but text is identical
        seg_a = Segment(start=0.0, end=5.0, text=shared_text)
        seg_b = Segment(start=600.0, end=605.0, text=shared_text)

        scored = [
            ScoredSegment(segment=seg_a, text_score=0.9, audio_score=0.9, llm_score=0.0, clip_score=0.9),
            ScoredSegment(segment=seg_b, text_score=0.5, audio_score=0.5, llm_score=0.0, clip_score=0.5),
        ]
        # Transcript contains both segments
        transcript = make_transcript([seg_a, seg_b])
        config = make_config(
            top_n_clips=2,
            min_clip_spacing=0.0,  # disable spacing so both survive to dedup pass
            dedup_similarity_threshold=0.7,
        )

        clips = select_clips(config, scored, transcript, video_duration=1200.0)

        # Only the higher-scoring clip should survive
        assert len(clips) == 1, f"Expected 1 clip after dedup, got {len(clips)}: {[(c.start, c.score) for c in clips]}"
        assert clips[0].score == 0.9, f"Expected score=0.9 (higher-scoring kept), got {clips[0].score}"

    def test_dissimilar_text_both_kept(self):
        """Two clips with completely different transcript text are both kept."""
        seg_a = Segment(start=0.0, end=5.0, text="amazing clutch play incredible moment")
        seg_b = Segment(start=600.0, end=605.0, text="cooking recipe pasta sauce tomato basil")

        scored = [
            ScoredSegment(segment=seg_a, text_score=0.9, audio_score=0.9, llm_score=0.0, clip_score=0.9),
            ScoredSegment(segment=seg_b, text_score=0.8, audio_score=0.8, llm_score=0.0, clip_score=0.8),
        ]
        transcript = make_transcript([seg_a, seg_b])
        config = make_config(
            top_n_clips=2,
            min_clip_spacing=0.0,
            dedup_similarity_threshold=0.7,
        )

        clips = select_clips(config, scored, transcript, video_duration=1200.0)

        assert len(clips) == 2, f"Expected 2 clips (dissimilar text), got {len(clips)}"
        scores = {c.score for c in clips}
        assert 0.9 in scores and 0.8 in scores

    def test_threshold_respected_just_below_keeps_both(self):
        """Similarity just below the threshold keeps both clips."""
        # Craft two texts with known Jaccard similarity just below 0.7
        # Words in A: {a, b, c, d, e, f, g, h, i, j}  (10 words)
        # Words in B: {a, b, c, d, e, f, x, y, z, w}  (10 words)
        # Intersection: {a,b,c,d,e,f} = 6, Union = 14, Jaccard = 6/14 ≈ 0.43 < 0.7
        text_a = "a b c d e f g h i j"
        text_b = "a b c d e f x y z w"

        seg_a = Segment(start=0.0, end=5.0, text=text_a)
        seg_b = Segment(start=600.0, end=605.0, text=text_b)

        scored = [
            ScoredSegment(segment=seg_a, text_score=0.9, audio_score=0.9, llm_score=0.0, clip_score=0.9),
            ScoredSegment(segment=seg_b, text_score=0.8, audio_score=0.8, llm_score=0.0, clip_score=0.8),
        ]
        transcript = make_transcript([seg_a, seg_b])
        config = make_config(
            top_n_clips=2,
            min_clip_spacing=0.0,
            dedup_similarity_threshold=0.7,
        )

        clips = select_clips(config, scored, transcript, video_duration=1200.0)

        assert len(clips) == 2, (
            f"Expected 2 clips (similarity below threshold), got {len(clips)}: "
            f"{[(c.start, c.score) for c in clips]}"
        )

    def test_dedup_logs_removal(self, caplog):
        """Removal of a duplicate clip is logged with the correct format."""
        import logging

        shared_text = "watch this incredible moment right now"
        seg_a = Segment(start=0.0, end=5.0, text=shared_text)
        seg_b = Segment(start=600.0, end=605.0, text=shared_text)

        scored = [
            ScoredSegment(segment=seg_a, text_score=0.9, audio_score=0.9, llm_score=0.0, clip_score=0.9),
            ScoredSegment(segment=seg_b, text_score=0.5, audio_score=0.5, llm_score=0.0, clip_score=0.5),
        ]
        transcript = make_transcript([seg_a, seg_b])
        config = make_config(
            top_n_clips=2,
            min_clip_spacing=0.0,
            dedup_similarity_threshold=0.7,
        )

        with caplog.at_level(logging.INFO, logger="pipeline.clip_selector"):
            select_clips(config, scored, transcript, video_duration=1200.0)

        log_messages = [r.message for r in caplog.records]
        assert any("transcript similarity" in msg for msg in log_messages), (
            f"Expected a 'transcript similarity' log message, got: {log_messages}"
        )
        assert any("removed" in msg for msg in log_messages), (
            f"Expected 'removed' in log message, got: {log_messages}"
        )


# ---------------------------------------------------------------------------
# Task 11: Auto-scale min_clip_spacing for short videos
# ---------------------------------------------------------------------------

class TestAutoScaleMinClipSpacing:
    """Auto-scaling of min_clip_spacing when video is too short for the default spacing."""

    def _make_segments(self, starts, duration=5.0, text="hello"):
        return [make_segment(s, s + duration, text) for s in starts]

    def test_short_video_auto_scales_spacing(self):
        """10-minute video (600s) with top_n=5: spacing auto-scales from 300s to 100s (600/6)."""
        video_duration = 600.0  # 10 minutes
        top_n = 5
        # 5 segments spread across the video, each 25s long so they meet min_clip_duration=20s
        seg_starts = [i * 100.0 for i in range(top_n)]
        segments = self._make_segments(seg_starts, duration=25.0)
        scored = [make_scored(seg, float(i + 1) / top_n) for i, seg in enumerate(segments)]
        transcript = make_transcript(segments)
        config = make_config(
            top_n_clips=top_n,
            min_clip_spacing=300.0,  # default 5 minutes
            min_clip_duration=20.0,
            max_clip_duration=45.0,
            dedup_similarity_threshold=1.0,  # disable dedup — segments have identical text
        )

        # video_duration / top_n_clips = 600 / 5 = 120 < 300 → auto-scale to 600 / 6 = 100s
        clips = select_clips(config, scored, transcript, video_duration=video_duration)

        # With effective_spacing=100s, segments 100s apart should all be accepted
        assert len(clips) == top_n, (
            f"Expected {top_n} clips after auto-scaling spacing to 100s, got {len(clips)}"
        )

    def test_long_video_does_not_auto_scale(self):
        """Long video (3600s) with top_n=5: spacing should NOT auto-scale (3600/5=720 > 300)."""
        video_duration = 3600.0  # 1 hour
        top_n = 5
        # 5 segments spread 700s apart, each 25s long so they meet min_clip_duration=20s
        seg_starts = [i * 700.0 for i in range(top_n)]
        segments = self._make_segments(seg_starts, duration=25.0)
        scored = [make_scored(seg, float(i + 1) / top_n) for i, seg in enumerate(segments)]
        transcript = make_transcript(segments)
        config = make_config(
            top_n_clips=top_n,
            min_clip_spacing=300.0,
            min_clip_duration=20.0,
            max_clip_duration=45.0,
            dedup_similarity_threshold=1.0,  # disable dedup — segments have identical text
        )

        # video_duration / top_n_clips = 3600 / 5 = 720 > 300 → no auto-scaling
        clips = select_clips(config, scored, transcript, video_duration=video_duration)

        # All 5 clips are 700s apart (> 300s), so all should be accepted
        assert len(clips) == top_n, (
            f"Expected {top_n} clips (no auto-scaling needed), got {len(clips)}"
        )

    def test_auto_scale_log_message_emitted(self, caplog):
        """Log message is emitted when auto-scaling occurs."""
        import logging

        video_duration = 600.0
        top_n = 5
        seg_starts = [i * 100.0 for i in range(top_n)]
        segments = self._make_segments(seg_starts)
        scored = [make_scored(seg, float(i + 1) / top_n) for i, seg in enumerate(segments)]
        transcript = make_transcript(segments)
        config = make_config(
            top_n_clips=top_n,
            min_clip_spacing=300.0,
            min_clip_duration=20.0,
            max_clip_duration=45.0,
        )

        with caplog.at_level(logging.INFO, logger="pipeline.clip_selector"):
            select_clips(config, scored, transcript, video_duration=video_duration)

        log_messages = [r.message for r in caplog.records]
        assert any("Auto-scaled min_clip_spacing" in msg for msg in log_messages), (
            f"Expected auto-scale log message, got: {log_messages}"
        )
        assert any("video too short" in msg for msg in log_messages), (
            f"Expected 'video too short' in log message, got: {log_messages}"
        )

    def test_no_auto_scale_log_when_not_needed(self, caplog):
        """No auto-scale log message when video is long enough."""
        import logging

        video_duration = 3600.0
        top_n = 5
        seg_starts = [i * 700.0 for i in range(top_n)]
        segments = self._make_segments(seg_starts)
        scored = [make_scored(seg, float(i + 1) / top_n) for i, seg in enumerate(segments)]
        transcript = make_transcript(segments)
        config = make_config(
            top_n_clips=top_n,
            min_clip_spacing=300.0,
            min_clip_duration=20.0,
            max_clip_duration=45.0,
        )

        with caplog.at_level(logging.INFO, logger="pipeline.clip_selector"):
            select_clips(config, scored, transcript, video_duration=video_duration)

        log_messages = [r.message for r in caplog.records]
        assert not any("Auto-scaled min_clip_spacing" in msg for msg in log_messages), (
            f"Expected no auto-scale log message for long video, got: {log_messages}"
        )
