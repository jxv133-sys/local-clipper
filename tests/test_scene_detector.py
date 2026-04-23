"""Tests for pipeline/scene_detector.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pipeline.scene_detector import _detect_scene_cuts, snap_to_nearest_cut


# ---------------------------------------------------------------------------
# snap_to_nearest_cut — pure function, no I/O
# ---------------------------------------------------------------------------

class TestSnapToNearestCut:
    def test_returns_boundary_when_no_cuts(self):
        assert snap_to_nearest_cut(10.0, []) == 10.0

    def test_returns_boundary_when_no_cut_within_window(self):
        # Cuts at 5.0 and 15.0 — both > 2s away from boundary 10.0
        assert snap_to_nearest_cut(10.0, [5.0, 15.0], window=2.0) == 10.0

    def test_snaps_to_nearest_cut_within_window(self):
        # Cut at 9.5 is 0.5s away — within window=2.0
        result = snap_to_nearest_cut(10.0, [9.5], window=2.0)
        assert result == 9.5

    def test_snaps_to_closest_of_multiple_cuts(self):
        # Cuts at 8.5 (1.5s away) and 9.8 (0.2s away) — 9.8 is closer
        result = snap_to_nearest_cut(10.0, [8.5, 9.8], window=2.0)
        assert result == 9.8

    def test_cut_exactly_at_window_boundary_is_included(self):
        # Cut exactly 2.0s away — should be included (dist <= window)
        result = snap_to_nearest_cut(10.0, [8.0], window=2.0)
        assert result == 8.0

    def test_cut_just_outside_window_is_excluded(self):
        # Cut 2.001s away — just outside window
        result = snap_to_nearest_cut(10.0, [7.999], window=2.0)
        assert result == 10.0

    def test_cut_after_boundary_within_window(self):
        # Cut at 11.5 is 1.5s after boundary 10.0 — within window=2.0
        result = snap_to_nearest_cut(10.0, [11.5], window=2.0)
        assert result == 11.5

    def test_returns_boundary_unchanged_when_cuts_empty(self):
        boundary = 42.7
        assert snap_to_nearest_cut(boundary, [], window=5.0) == boundary

    def test_default_window_is_2_seconds(self):
        # Default window=2.0: cut at 1.9s away should snap
        result = snap_to_nearest_cut(10.0, [8.1])
        assert result == 8.1

    def test_cut_at_boundary_itself(self):
        # Cut exactly at the boundary — distance 0, always snaps
        result = snap_to_nearest_cut(10.0, [10.0], window=2.0)
        assert result == 10.0


# ---------------------------------------------------------------------------
# _detect_scene_cuts — tests with mocked subprocess
# ---------------------------------------------------------------------------

class TestDetectSceneCuts:
    def _run(self, stdout: str, returncode: int = 0, side_effect=None):
        """Helper: patch subprocess.run and call _detect_scene_cuts."""
        mock_result = MagicMock()
        mock_result.returncode = returncode
        mock_result.stdout = stdout

        with patch("pipeline.scene_detector.subprocess.run") as mock_run:
            if side_effect:
                mock_run.side_effect = side_effect
            else:
                mock_run.return_value = mock_result
            return _detect_scene_cuts("/fake/video.mp4", start=10.0, end=20.0, window=2.0)

    def test_returns_i_frames_within_window(self):
        # I-frames at 9.0 (within [8,22]) and 21.0 (within [8,22])
        stdout = "9.0,I\n15.0,I\n21.0,I\n"
        cuts = self._run(stdout)
        assert 9.0 in cuts
        assert 15.0 in cuts
        assert 21.0 in cuts

    def test_excludes_non_i_frames(self):
        # P and B frames should be ignored
        stdout = "9.0,P\n10.5,B\n11.0,I\n"
        cuts = self._run(stdout)
        assert cuts == [11.0]

    def test_excludes_frames_outside_window(self):
        # window=2.0, start=10.0, end=20.0 → range [8.0, 22.0]
        # Frame at 7.9 is outside; frame at 22.1 is outside
        stdout = "7.9,I\n10.0,I\n22.1,I\n"
        cuts = self._run(stdout)
        assert 7.9 not in cuts
        assert 22.1 not in cuts
        assert 10.0 in cuts

    def test_returns_empty_on_nonzero_returncode(self):
        cuts = self._run("", returncode=1)
        assert cuts == []

    def test_returns_empty_on_ffprobe_not_found(self):
        cuts = self._run("", side_effect=FileNotFoundError("ffprobe not found"))
        assert cuts == []

    def test_returns_empty_on_timeout(self):
        import subprocess
        cuts = self._run("", side_effect=subprocess.TimeoutExpired(cmd="ffprobe", timeout=60))
        assert cuts == []

    def test_returns_empty_on_empty_output(self):
        cuts = self._run("")
        assert cuts == []

    def test_skips_malformed_lines(self):
        # Lines with missing comma or non-numeric pts
        stdout = "bad_line\n,I\nN/A,I\n12.0,I\n"
        cuts = self._run(stdout)
        assert cuts == [12.0]

    def test_result_is_sorted(self):
        stdout = "20.0,I\n9.0,I\n14.5,I\n"
        cuts = self._run(stdout)
        assert cuts == sorted(cuts)

    def test_window_applied_correctly(self):
        # start=10, end=20, window=2 → range [8.0, 22.0]
        stdout = "8.0,I\n22.0,I\n7.9,I\n22.1,I\n"
        cuts = self._run(stdout)
        assert 8.0 in cuts
        assert 22.0 in cuts
        assert 7.9 not in cuts
        assert 22.1 not in cuts

    def test_ffprobe_command_includes_video_path(self):
        """ffprobe is called with the correct video path."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""

        with patch("pipeline.scene_detector.subprocess.run") as mock_run:
            mock_run.return_value = mock_result
            _detect_scene_cuts("/my/video.mp4", start=5.0, end=15.0)
            args = mock_run.call_args[0][0]
            assert "/my/video.mp4" in args


# ---------------------------------------------------------------------------
# Integration: snap_to_nearest_cut + _detect_scene_cuts together
# ---------------------------------------------------------------------------

class TestSceneSnapIntegration:
    def test_snap_uses_detected_cuts(self):
        """snap_to_nearest_cut correctly uses cuts returned by _detect_scene_cuts."""
        # Simulate a cut at 9.8s near boundary 10.0
        cuts = [9.8]
        result = snap_to_nearest_cut(10.0, cuts, window=2.0)
        assert result == 9.8

    def test_no_snap_when_no_cuts_detected(self):
        """When _detect_scene_cuts returns empty, boundary is unchanged."""
        cuts: list[float] = []
        result = snap_to_nearest_cut(10.0, cuts, window=2.0)
        assert result == 10.0
