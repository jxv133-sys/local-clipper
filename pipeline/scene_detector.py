"""Scene Detector: detects scene cuts via ffprobe I-frame analysis."""

from __future__ import annotations

import logging
import subprocess

logger = logging.getLogger(__name__)


def _detect_scene_cuts(
    video_path: str,
    start: float,
    end: float,
    window: float = 2.0,
) -> list[float]:
    """Return timestamps of I-frames (scene cuts) within [start-window, end+window].

    Runs ffprobe to extract all video frame timestamps and picture types, then
    filters to I-frames (pict_type=I) that fall within the search window around
    the clip boundary.

    Args:
        video_path: Path to the source video file.
        start: Clip start time in seconds.
        end: Clip end time in seconds.
        window: Search radius in seconds around each boundary.

    Returns:
        Sorted list of I-frame timestamps within [start-window, end+window].
        Returns an empty list if ffprobe fails or produces no output.
    """
    range_start = max(0.0, start - window)
    range_end = end + window

    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-select_streams", "v",
        "-show_entries", "frame=pkt_pts_time,pict_type",
        "-of", "csv=p=0",
        video_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.warning("[SceneDetector] ffprobe failed for %s: %s", video_path, exc)
        return []

    if result.returncode != 0:
        logger.warning(
            "[SceneDetector] ffprobe returned non-zero exit code %d for %s",
            result.returncode,
            video_path,
        )
        return []

    cuts: list[float] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 2:
            continue
        pts_str, pict_type = parts[0].strip(), parts[1].strip()
        if pict_type != "I":
            continue
        try:
            pts = float(pts_str)
        except ValueError:
            continue
        if range_start <= pts <= range_end:
            cuts.append(pts)

    return sorted(cuts)


def snap_to_nearest_cut(
    boundary: float,
    cuts: list[float],
    window: float = 2.0,
) -> float:
    """Return the nearest scene cut within ±window seconds of boundary.

    If no cut falls within the window, returns boundary unchanged.

    Args:
        boundary: The original clip boundary timestamp in seconds.
        cuts: List of scene cut timestamps (I-frame positions).
        window: Maximum distance in seconds to search for a nearby cut.

    Returns:
        The nearest cut timestamp within ±window, or boundary if none found.
    """
    best: float | None = None
    best_dist = float("inf")

    for cut in cuts:
        dist = abs(cut - boundary)
        if dist <= window and dist < best_dist:
            best_dist = dist
            best = cut

    return best if best is not None else boundary
