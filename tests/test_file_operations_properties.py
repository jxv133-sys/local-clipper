"""Property-based tests for file operations (Task 8.6).

Validates: Requirements 8.2, 8.3, 9.2, 9.3
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from pipeline.models import CanvasLayout, FacecamRegion, VerticalFormattingJob
from pipeline.vertical_formatter import (
    get_output_path,
    backup_clips,
    restore_from_backup,
    replace_clips_with_vertical,
    create_backup_directory,
    _build_vertical_filter,
)
from pipeline.frame_reformatter import compute_canvas_layout


# ---------------------------------------------------------------------------
# Helpers / Strategies
# ---------------------------------------------------------------------------

def make_config(
    shorts_width: int = 1080,
    shorts_height: int = 1920,
    facecam_top_fraction: float = 0.35,
) -> SimpleNamespace:
    return SimpleNamespace(
        shorts_width=shorts_width,
        shorts_height=shorts_height,
        facecam_top_fraction=facecam_top_fraction,
    )


DEFAULT_CONFIG = make_config()


def make_canvas_layout() -> CanvasLayout:
    return compute_canvas_layout(DEFAULT_CONFIG)


def make_facecam_region() -> FacecamRegion:
    return FacecamRegion(
        x=100, y=50, width=300, height=200,
        corner="top-left", confidence=0.9,
    )


@st.composite
def valid_filename_stem(draw):
    """Generate a valid filename stem (no path separators, no dots)."""
    chars = st.characters(
        whitelist_categories=("Lu", "Ll", "Nd"),
        whitelist_characters="_-",
    )
    stem = draw(st.text(alphabet=chars, min_size=1, max_size=20))
    assume(stem and not stem.startswith("-"))
    return stem


@st.composite
def settings_with_suffix(draw):
    """Generate settings dict with optional suffix/prefix."""
    use_suffix = draw(st.booleans())
    use_prefix = draw(st.booleans())

    settings_dict: dict = {}
    if use_suffix:
        suffix = draw(st.sampled_from(["_vertical", "_v", "_formatted", ""]))
        settings_dict["suffix"] = suffix
    if use_prefix:
        prefix = draw(st.sampled_from(["vertical_", "v_", ""]))
        settings_dict["prefix"] = prefix
    return settings_dict


# ---------------------------------------------------------------------------
# Property 16: Output naming
# **Validates: Requirements 8.2, 20.4**
# ---------------------------------------------------------------------------

@settings(max_examples=200)
@given(
    stem=valid_filename_stem(),
    ext=st.sampled_from([".mp4", ".mov", ".avi", ".mkv"]),
    settings_dict=settings_with_suffix(),
)
def test_property_16_output_filename_contains_original_stem(stem, ext, settings_dict):
    """
    Property 16: For any output file, its name must contain the original
    filename stem.

    **Validates: Requirements 8.2, 20.4**
    """
    input_path = f"/some/dir/{stem}{ext}"
    output_dir = "/output/dir"

    output = get_output_path(input_path, settings_dict, output_dir)
    output_name = Path(output).name

    assert stem in output_name, (
        f"Original stem '{stem}' not found in output name '{output_name}'"
    )


@settings(max_examples=200)
@given(
    stem=valid_filename_stem(),
    ext=st.sampled_from([".mp4", ".mov", ".avi", ".mkv"]),
    settings_dict=settings_with_suffix(),
)
def test_property_16_output_preserves_extension(stem, ext, settings_dict):
    """
    Property 16: For any output file, the extension must be preserved from
    the original file.

    **Validates: Requirements 8.2, 20.4**
    """
    input_path = f"/some/dir/{stem}{ext}"
    output_dir = "/output/dir"

    output = get_output_path(input_path, settings_dict, output_dir)

    assert output.endswith(ext), (
        f"Extension '{ext}' not preserved in output '{output}'"
    )


@settings(max_examples=200)
@given(stem=valid_filename_stem())
def test_property_16_default_suffix_appended(stem):
    """
    Property 16: When no suffix is specified in settings, the default
    '_vertical' suffix must be appended before the extension.

    **Validates: Requirements 8.2, 20.4**
    """
    input_path = f"/some/dir/{stem}.mp4"
    output_dir = "/output/dir"

    output = get_output_path(input_path, {}, output_dir)
    output_name = Path(output).name

    assert output_name == f"{stem}_vertical.mp4", (
        f"Expected '{stem}_vertical.mp4', got '{output_name}'"
    )


@settings(max_examples=200)
@given(
    stem=valid_filename_stem(),
    suffix=st.sampled_from(["_v", "_formatted", "_short", ""]),
)
def test_property_16_custom_suffix_used_when_specified(stem, suffix):
    """
    Property 16: When a custom suffix is specified in settings, it must be
    used instead of the default '_vertical' suffix.

    **Validates: Requirements 8.2, 20.4**
    """
    input_path = f"/some/dir/{stem}.mp4"
    output_dir = "/output/dir"
    settings_dict = {"suffix": suffix}

    output = get_output_path(input_path, settings_dict, output_dir)
    output_name = Path(output).name

    assert output_name == f"{stem}{suffix}.mp4", (
        f"Expected '{stem}{suffix}.mp4', got '{output_name}'"
    )


@settings(max_examples=200)
@given(
    stem=valid_filename_stem(),
    prefix=st.sampled_from(["vertical_", "v_", "fmt_"]),
)
def test_property_16_custom_prefix_prepended_when_specified(stem, prefix):
    """
    Property 16: When a custom prefix is specified in settings, it must be
    prepended to the filename.

    **Validates: Requirements 8.2, 20.4**
    """
    input_path = f"/some/dir/{stem}.mp4"
    output_dir = "/output/dir"
    settings_dict = {"prefix": prefix, "suffix": ""}

    output = get_output_path(input_path, settings_dict, output_dir)
    output_name = Path(output).name

    assert output_name.startswith(prefix), (
        f"Expected output name to start with '{prefix}', got '{output_name}'"
    )
    assert stem in output_name, (
        f"Original stem '{stem}' not found in output name '{output_name}'"
    )


# ---------------------------------------------------------------------------
# Property 17: Output location
# **Validates: Requirements 8.3**
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    stem=valid_filename_stem(),
    settings_dict=settings_with_suffix(),
)
def test_property_17_output_saved_to_configured_directory(stem, settings_dict):
    """
    Property 17: For any output file, it must be saved to the configured
    output directory.

    **Validates: Requirements 8.3**
    """
    input_path = f"/original/dir/{stem}.mp4"
    output_dir = "/configured/output/dir"

    output = get_output_path(input_path, settings_dict, output_dir)

    assert output.startswith(output_dir), (
        f"Output '{output}' not in configured directory '{output_dir}'"
    )


@settings(max_examples=100)
@given(
    stem=valid_filename_stem(),
    settings_dict=settings_with_suffix(),
)
def test_property_17_output_not_in_original_directory(stem, settings_dict):
    """
    Property 17: When output_dir differs from the original clip directory,
    the output must not be placed in the original directory.

    **Validates: Requirements 8.3**
    """
    original_dir = "/original/clips"
    output_dir = "/output/vertical"
    input_path = f"{original_dir}/{stem}.mp4"

    output = get_output_path(input_path, settings_dict, output_dir)

    assert not output.startswith(original_dir), (
        f"Output '{output}' was placed in original directory '{original_dir}'"
    )


# ---------------------------------------------------------------------------
# Property 18: Audio preservation
# **Validates: Requirements 8.5**
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    src_w=st.integers(min_value=320, max_value=1920),
    src_h=st.integers(min_value=240, max_value=1080),
    region=st.builds(
        FacecamRegion,
        x=st.integers(min_value=0, max_value=100),
        y=st.integers(min_value=0, max_value=100),
        width=st.integers(min_value=50, max_value=200),
        height=st.integers(min_value=50, max_value=200),
        corner=st.sampled_from(["top-left", "top-right", "bottom-left", "bottom-right"]),
        confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    ),
)
def test_property_18_filter_chain_includes_audio_copy(src_w, src_h, region):
    """
    Property 18: For any output file, the audio track must be preserved.

    The FFmpeg command built by VerticalFormatter must include '-c:a copy'
    to preserve the original audio track without re-encoding.

    This test verifies the filter chain is built correctly (the actual
    '-c:a copy' flag is in the FFmpeg command, not the filter_complex string).

    **Validates: Requirements 8.5**
    """
    from unittest.mock import patch, MagicMock
    import subprocess

    layout = make_canvas_layout()

    # Verify the filter chain is valid (contains all required stages)
    filter_str = _build_vertical_filter(src_w, src_h, region, layout)

    assert "[canvas]" in filter_str, "Filter missing canvas stage"
    assert "[facecam_scaled]" in filter_str, "Filter missing facecam_scaled stage"
    assert "[with_facecam]" in filter_str, "Filter missing overlay stage"


@settings(max_examples=50)
@given(
    stem=valid_filename_stem(),
    settings_dict=settings_with_suffix(),
)
def test_property_18_apply_placement_uses_audio_copy(stem, settings_dict):
    """
    Property 18: The apply_placement_to_clip method must use '-c:a copy'
    to preserve the original audio track.

    **Validates: Requirements 8.5**
    """
    import subprocess
    from unittest.mock import patch, MagicMock
    from pipeline.vertical_formatter import VerticalFormatter

    formatter = VerticalFormatter()
    layout = make_canvas_layout()
    region = make_facecam_region()

    captured_cmd: list[list[str]] = []

    def mock_run(cmd, *args, **kwargs):
        captured_cmd.append(cmd)
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        return result

    config = SimpleNamespace(output_crf=23, output_codec="libx264")

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, f"{stem}.mp4")
        output_path = os.path.join(tmpdir, f"{stem}_vertical.mp4")

        # Create a dummy input file
        Path(input_path).write_bytes(b"dummy")

        with patch("subprocess.run", side_effect=mock_run):
            formatter.apply_placement_to_clip(
                clip_path=input_path,
                facecam_region=region,
                canvas_layout=layout,
                output_path=output_path,
                config=config,
                clip_resolution=(1920, 1080),
            )

    assert len(captured_cmd) == 1, "Expected exactly one FFmpeg call"
    cmd = captured_cmd[0]

    # Must include '-c:a copy' to preserve audio
    assert "-c:a" in cmd, "FFmpeg command missing '-c:a' flag"
    ca_idx = cmd.index("-c:a")
    assert cmd[ca_idx + 1] == "copy", (
        f"Expected '-c:a copy', got '-c:a {cmd[ca_idx + 1]}'"
    )


# ---------------------------------------------------------------------------
# Property 19: Backup creation
# **Validates: Requirements 9.2, 9.3**
# ---------------------------------------------------------------------------

@settings(max_examples=50)
@given(n_clips=st.integers(min_value=1, max_value=5))
def test_property_19_backup_creates_copies_of_all_clips(n_clips):
    """
    Property 19: For any replacement operation with backup enabled, a backup
    copy must be created for every original clip before replacement.

    **Validates: Requirements 9.2, 9.3**
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy original clips
        original_paths = []
        for i in range(n_clips):
            clip_path = os.path.join(tmpdir, f"clip_{i}.mp4")
            Path(clip_path).write_bytes(f"original content {i}".encode())
            original_paths.append(clip_path)

        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir, exist_ok=True)

        # Create backup
        backup_dir = backup_clips(original_paths, output_dir)

        # Backup directory must exist
        assert os.path.isdir(backup_dir), (
            f"Backup directory not created: {backup_dir}"
        )

        # Every original clip must have a backup copy
        for original_path in original_paths:
            filename = os.path.basename(original_path)
            backup_path = os.path.join(backup_dir, filename)
            assert os.path.exists(backup_path), (
                f"Backup not created for '{filename}' in '{backup_dir}'"
            )


@settings(max_examples=50)
@given(n_clips=st.integers(min_value=1, max_value=5))
def test_property_19_backup_content_matches_original(n_clips):
    """
    Property 19: For any backup operation, the backup copies must have the
    same content as the originals.

    **Validates: Requirements 9.2, 9.3**
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        original_paths = []
        original_contents = []
        for i in range(n_clips):
            content = f"original content {i} - {uuid.uuid4()}".encode()
            clip_path = os.path.join(tmpdir, f"clip_{i}.mp4")
            Path(clip_path).write_bytes(content)
            original_paths.append(clip_path)
            original_contents.append(content)

        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir, exist_ok=True)

        backup_dir = backup_clips(original_paths, output_dir)

        for original_path, expected_content in zip(original_paths, original_contents):
            filename = os.path.basename(original_path)
            backup_path = os.path.join(backup_dir, filename)
            actual_content = Path(backup_path).read_bytes()
            assert actual_content == expected_content, (
                f"Backup content mismatch for '{filename}'"
            )


@settings(max_examples=30)
@given(n_clips=st.integers(min_value=1, max_value=4))
def test_property_19_backup_directory_has_timestamp_format(n_clips):
    """
    Property 19: The backup directory must use a timestamped naming convention
    (backup_YYYYMMDD_HHMMSS) for clear identification.

    **Validates: Requirements 9.2, 9.3**
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        original_paths = []
        for i in range(n_clips):
            clip_path = os.path.join(tmpdir, f"clip_{i}.mp4")
            Path(clip_path).write_bytes(b"content")
            original_paths.append(clip_path)

        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir, exist_ok=True)

        backup_dir = backup_clips(original_paths, output_dir)
        backup_name = os.path.basename(backup_dir)

        # Must start with "backup_"
        assert backup_name.startswith("backup_"), (
            f"Backup directory name '{backup_name}' does not start with 'backup_'"
        )

        # Must have timestamp format: backup_YYYYMMDD_HHMMSS
        parts = backup_name.split("_")
        assert len(parts) == 3, (
            f"Expected 'backup_YYYYMMDD_HHMMSS' format, got '{backup_name}'"
        )
        date_part, time_part = parts[1], parts[2]
        assert len(date_part) == 8 and date_part.isdigit(), (
            f"Date part '{date_part}' is not 8 digits"
        )
        assert len(time_part) == 6 and time_part.isdigit(), (
            f"Time part '{time_part}' is not 6 digits"
        )


@settings(max_examples=30)
@given(n_clips=st.integers(min_value=1, max_value=4))
def test_property_19_restore_from_backup_recovers_originals(n_clips):
    """
    Property 19: For any backup, restore_from_backup must recover the
    original files to their original paths.

    **Validates: Requirements 9.2, 9.3**
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        original_paths = []
        original_contents = []
        for i in range(n_clips):
            content = f"original content {i} - {uuid.uuid4()}".encode()
            clip_path = os.path.join(tmpdir, f"clip_{i}.mp4")
            Path(clip_path).write_bytes(content)
            original_paths.append(clip_path)
            original_contents.append(content)

        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir, exist_ok=True)

        # Create backup
        backup_dir = backup_clips(original_paths, output_dir)

        # Overwrite originals with different content
        for clip_path in original_paths:
            Path(clip_path).write_bytes(b"modified content")

        # Restore from backup
        restore_from_backup(backup_dir, original_paths)

        # Originals must be restored to their original content
        for clip_path, expected_content in zip(original_paths, original_contents):
            actual_content = Path(clip_path).read_bytes()
            assert actual_content == expected_content, (
                f"Restore failed for '{os.path.basename(clip_path)}': "
                f"content does not match original"
            )


@settings(max_examples=30)
@given(n_clips=st.integers(min_value=1, max_value=4))
def test_property_19_replace_with_backup_creates_backup_before_replacing(n_clips):
    """
    Property 19: When replace_clips_with_vertical is called with backup=True,
    backups must be created before the originals are replaced.

    **Validates: Requirements 9.2, 9.3**
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        original_paths = []
        vertical_paths = []
        original_contents = []

        for i in range(n_clips):
            orig_content = f"original {i}".encode()
            vert_content = f"vertical {i}".encode()

            orig_path = os.path.join(tmpdir, f"clip_{i}.mp4")
            vert_path = os.path.join(tmpdir, f"clip_{i}_vertical.mp4")

            Path(orig_path).write_bytes(orig_content)
            Path(vert_path).write_bytes(vert_content)

            original_paths.append(orig_path)
            vertical_paths.append(vert_path)
            original_contents.append(orig_content)

        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir, exist_ok=True)

        # Replace with backup enabled
        backup_dir = replace_clips_with_vertical(
            original_paths, vertical_paths, output_dir, backup=True
        )

        # Backup directory must have been created
        assert backup_dir is not None, "Expected backup_dir to be returned"
        assert os.path.isdir(backup_dir), (
            f"Backup directory not created: {backup_dir}"
        )

        # Backup must contain original content
        for orig_path, orig_content in zip(original_paths, original_contents):
            filename = os.path.basename(orig_path)
            backup_path = os.path.join(backup_dir, filename)
            assert os.path.exists(backup_path), (
                f"Backup not found for '{filename}'"
            )
            assert Path(backup_path).read_bytes() == orig_content, (
                f"Backup content mismatch for '{filename}'"
            )

        # Originals must now contain vertical content
        for orig_path, vert_content in zip(original_paths, [f"vertical {i}".encode() for i in range(n_clips)]):
            assert Path(orig_path).read_bytes() == vert_content, (
                f"Original not replaced with vertical content for '{os.path.basename(orig_path)}'"
            )
