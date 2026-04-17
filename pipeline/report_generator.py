"""Report generation: writes a human-readable 'why chosen' summary for each clip."""

from __future__ import annotations

import os

from pipeline.models import Clip, ScoredSegment, Transcript


def _format_time(seconds: float) -> str:
    """Format seconds as MM:SS."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def _score_bar(score: float, width: int = 20) -> str:
    """Render a simple ASCII progress bar for a score in [0, 1]."""
    filled = int(round(score * width))
    return "[" + "█" * filled + "░" * (width - filled) + f"] {score:.2f}"


def generate_report(
    clip: Clip,
    scored_segments: list[ScoredSegment],
    transcript: Transcript,
    clip_path: str,
) -> str:
    """Write a 'why chosen' report alongside a clip and return the report path.

    The report includes:
    - Clip timing and duration
    - Overall clip score and rank
    - Score breakdown (text, audio, LLM)
    - The transcript text covered by the clip
    - Which scoring signals fired (keywords, punctuation, energy)

    Args:
        clip: The Clip object.
        scored_segments: All scored segments from the pipeline.
        transcript: The full transcript.
        clip_path: Path to the exported clip .mp4 file.

    Returns:
        Path to the written .txt report file.
    """
    # Find the seed scored segment (highest score within the clip's segment_indices)
    seed_scored: ScoredSegment | None = None
    clip_scored_segs: list[ScoredSegment] = []

    for idx in clip.segment_indices:
        if 0 <= idx < len(scored_segments):
            clip_scored_segs.append(scored_segments[idx])

    if clip_scored_segs:
        seed_scored = max(clip_scored_segs, key=lambda s: s.clip_score)

    # Collect transcript text for the clip window
    clip_text_segments = [
        seg for seg in transcript.segments
        if seg.text.strip() and not (seg.end <= clip.start or seg.start >= clip.end)
    ]
    clip_text = " ".join(seg.text.strip() for seg in clip_text_segments)

    # Build the report
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append(f"  HIGHLIGHT CLIP #{clip.rank}")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"  Time:     {_format_time(clip.start)} → {_format_time(clip.end)}")
    lines.append(f"  Duration: {clip.end - clip.start:.1f}s")
    lines.append(f"  Rank:     #{clip.rank} (higher = better)")
    lines.append("")
    lines.append("WHY THIS CLIP WAS CHOSEN")
    lines.append("-" * 60)

    if seed_scored:
        lines.append("")
        lines.append(f"  Overall Score:  {_score_bar(seed_scored.clip_score)}")
        lines.append(f"  Audio Energy:   {_score_bar(seed_scored.audio_score)}")
        lines.append(f"  Text Interest:  {_score_bar(seed_scored.text_score)}")
        if seed_scored.llm_score > 0.0:
            lines.append(f"  LLM Rating:     {_score_bar(seed_scored.llm_score)}")
        lines.append("")

        # Explain what drove the score
        reasons: list[str] = []

        if seed_scored.audio_score >= 0.7:
            reasons.append("• High audio energy — loud or energetic moment")
        elif seed_scored.audio_score >= 0.4:
            reasons.append("• Moderate audio energy")

        if seed_scored.text_score >= 0.5:
            reasons.append("• Engaging speech detected")

        # Check for keywords in the clip text
        from config import Config  # avoid circular import at module level
        # We don't have config here, so scan for common highlight words
        highlight_words = [
            "crazy", "important", "watch this", "incredible", "unbelievable",
            "amazing", "insane", "wow", "no way", "seriously", "actually",
        ]
        found_keywords = [kw for kw in highlight_words if kw.lower() in clip_text.lower()]
        if found_keywords:
            reasons.append(f"• Keyword(s) detected: {', '.join(found_keywords)}")

        # Check for punctuation signals
        exclamations = clip_text.count("!")
        questions = clip_text.count("?")
        if exclamations > 0:
            reasons.append(f"• {exclamations} exclamation mark(s) — expressive speech")
        if questions > 0:
            reasons.append(f"• {questions} question mark(s) — engaging dialogue")

        if seed_scored.llm_score >= 0.6:
            reasons.append(f"• Local LLM rated this segment {seed_scored.llm_score * 10:.0f}/10")

        if not reasons:
            reasons.append("• Selected based on combined audio and text signals")

        for reason in reasons:
            lines.append(f"  {reason}")
    else:
        lines.append("")
        lines.append(f"  Overall Score:  {_score_bar(clip.score)}")
        lines.append("")
        lines.append("  • Selected based on combined audio and text signals")

    lines.append("")
    lines.append("TRANSCRIPT")
    lines.append("-" * 60)
    if clip_text:
        # Word-wrap at ~56 chars
        words = clip_text.split()
        line_buf: list[str] = []
        char_count = 0
        for word in words:
            if char_count + len(word) + 1 > 56 and line_buf:
                lines.append("  " + " ".join(line_buf))
                line_buf = [word]
                char_count = len(word)
            else:
                line_buf.append(word)
                char_count += len(word) + 1
        if line_buf:
            lines.append("  " + " ".join(line_buf))
    else:
        lines.append("  (no speech detected in this clip)")

    lines.append("")
    lines.append("=" * 60)
    lines.append("")

    report_content = "\n".join(lines)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(clip_path)), exist_ok=True)

    # Write alongside the clip
    base = os.path.splitext(clip_path)[0]
    report_path = base + "_why_chosen.txt"
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report_content)

    return report_path
