"""Video Highlight Generator — pipeline orchestrator.

Usage:
    python3 main.py <input_video_path> [options]

Options:
    --output-dir DIR        Directory for output clips (default: <input_folder>/highlights)
    --whisper-model MODEL   Whisper model size: tiny/base/small/medium/large (default: base)
    --top-n N               Number of highlight clips to generate (default: 5)
    --llm                   Enable local LLM scoring via Ollama (default: disabled)
    --llm-endpoint URL      LLM endpoint URL (default: http://localhost:11434/api/generate)
    --llm-model MODEL       LLM model name (default: llama3)
    --keywords KW [KW ...]  Space-separated list of highlight keywords
"""

from __future__ import annotations

import argparse
import os
import shutil
import ssl
import sys
import tempfile
import time

# Ensure FFmpeg (Homebrew) is on PATH regardless of how the script is launched
_HOMEBREW_BIN = "/opt/homebrew/bin"
if _HOMEBREW_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _HOMEBREW_BIN + ":" + os.environ.get("PATH", "")

# Fix SSL certificate verification on macOS Python.org builds
# (required for Whisper model download on first run)
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except ImportError:
    os.environ.setdefault("PYTHONHTTPSVERIFY", "0")

from config import Config
from pipeline.audio_extractor import extract_audio
from pipeline.clip_extractor import extract_clips
from pipeline.clip_selector import select_clips
from pipeline.exceptions import PipelineError
from pipeline.report_generator import generate_report
from pipeline.scorer import score_segments
from pipeline.subtitle_generator import generate_subtitles
from pipeline.transcriber import transcribe


def _get_video_duration(video_path: str) -> float:
    """Use ffprobe to get the total duration of the video in seconds."""
    import subprocess
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        return float(result.stdout.strip())
    except (ValueError, AttributeError):
        # Fallback: assume a very long video so clip selection isn't clamped
        return 86400.0


def _run_stage(name: str, fn, *args, **kwargs):
    """Run a pipeline stage, logging start/end and elapsed time."""
    print(f"[{name}] Starting...", flush=True)
    t0 = time.time()
    result = fn(*args, **kwargs)
    elapsed = time.time() - t0
    print(f"[{name}] Done in {elapsed:.1f}s", flush=True)
    return result


def build_config(args: argparse.Namespace, work_dir: str) -> Config:
    """Construct a Config from parsed CLI arguments."""
    cfg = Config(work_dir=work_dir)
    cfg.output_dir = args.output_dir
    cfg.whisper_model = args.whisper_model
    cfg.top_n_clips = args.top_n
    cfg.llm_enabled = args.llm
    cfg.llm_endpoint = args.llm_endpoint
    cfg.llm_model = args.llm_model
    if args.keywords:
        cfg.keywords = args.keywords
    return cfg


def run_pipeline(video_path: str, config: Config) -> list[str]:
    """Execute the full highlight generation pipeline.

    Args:
        video_path: Path to the input video file.
        config: Pipeline configuration.

    Returns:
        List of paths to the final exported clip files.
    """
    # Stage 1: Audio extraction
    wav_path = _run_stage("AudioExtractor", extract_audio, config, video_path)

    # Stage 2: Transcription
    transcript = _run_stage("Transcriber", transcribe, config, wav_path)

    if not transcript.segments:
        print("Warning: No speech detected in video. Scoring will rely on audio energy only.",
              flush=True)

    # Stage 3: Scoring
    scored_segments = _run_stage("Scorer", score_segments, config, transcript, wav_path)

    if not scored_segments:
        raise PipelineError("No segments to score. The video may have no audio content.")

    # Stage 4: Clip selection
    video_duration = _get_video_duration(video_path)
    clips = _run_stage("ClipSelector", select_clips, config, scored_segments,
                       transcript, video_duration)

    if not clips:
        raise PipelineError("No clips were selected. Try lowering --top-n or check the video.")

    # Stage 5: Clip extraction
    clip_paths = _run_stage("ClipExtractor", extract_clips, config, clips, video_path)

    # Stage 6: Subtitle generation
    final_paths = _run_stage("SubtitleGenerator", generate_subtitles, config, clips,
                             transcript, clip_paths)

    # Stage 7: Why-chosen reports
    print("[ReportGenerator] Writing selection reports...", flush=True)
    report_paths: list[str] = []
    for clip, clip_path in zip(clips, final_paths):
        report_path = generate_report(clip, scored_segments, transcript, clip_path)
        report_paths.append(report_path)
    print(f"[ReportGenerator] Done — {len(report_paths)} report(s) written", flush=True)

    return final_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate highlight clips from a video file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python3 main.py input.mp4 --top-n 3 --whisper-model small",
    )
    parser.add_argument("input_video", help="Path to the input video file")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for output clips (default: <input_video_folder>/highlights)")
    parser.add_argument("--whisper-model", default="base",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: base)")
    parser.add_argument("--top-n", type=int, default=5,
                        help="Number of highlight clips to generate (default: 5)")
    parser.add_argument("--llm", action="store_true",
                        help="Enable local LLM scoring via Ollama")
    parser.add_argument("--llm-endpoint", default="http://localhost:11434/api/generate",
                        help="LLM endpoint URL")
    parser.add_argument("--llm-model", default="llama3",
                        help="LLM model name (default: llama3)")
    parser.add_argument("--keywords", nargs="+",
                        help="Keywords that boost segment scores")

    args = parser.parse_args()

    work_dir = tempfile.mkdtemp(prefix="highlight_")

    # Default output dir: a "highlights" folder next to the input video
    if args.output_dir is None:
        input_dir = os.path.dirname(os.path.abspath(args.input_video))
        args.output_dir = os.path.join(input_dir, "highlights")

    config = build_config(args, work_dir)

    print(f"Input:      {args.input_video}")
    print(f"Output dir: {config.output_dir}")
    print(f"Whisper:    {config.whisper_model}")
    print(f"Top N:      {config.top_n_clips}")
    print(f"LLM:        {'enabled (' + config.llm_model + ')' if config.llm_enabled else 'disabled'}")
    print()

    try:
        final_paths = run_pipeline(args.input_video, config)
    except PipelineError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        print(f"Temporary files kept for debugging: {work_dir}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"\nUnexpected error: {exc}", file=sys.stderr)
        print(f"Temporary files kept for debugging: {work_dir}", file=sys.stderr)
        sys.exit(1)

    # Success — clean up temp dir
    shutil.rmtree(work_dir, ignore_errors=True)

    print("\n✓ Done! Exported clips:")
    for path in final_paths:
        base = os.path.splitext(path)[0]
        report = base + "_why_chosen.txt"
        print(f"  {path}")
        if os.path.exists(report):
            print(f"    └─ {report}")


if __name__ == "__main__":
    main()
