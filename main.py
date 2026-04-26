"""Video Highlight Generator — pipeline orchestrator.

Usage:
    python3 main.py <input_video_path> [options]
    python3 main.py --url <youtube_url> [options]

Options:
    --url URL               Download a YouTube video and clip it automatically
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
import logging
import os
import shutil
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


def download_youtube_video(url: str, output_dir: str, max_height: int = 720) -> str:
    """Download a YouTube video using yt-dlp.

    Downloads up to *max_height*p resolution (default 720p) into *output_dir*
    and returns the path to the downloaded file. Lower resolution = much faster
    download and processing, which is fine for highlight clip generation.

    Args:
        url: YouTube URL (any format yt-dlp accepts).
        output_dir: Directory to save the downloaded file.
        max_height: Maximum video height in pixels (default: 720).

    Returns:
        Absolute path to the downloaded video file.

    Raises:
        PipelineError: If yt-dlp is not installed or the download fails.
    """
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        raise PipelineError(
            "yt-dlp is required for YouTube downloads. "
            "Install it with: pip install yt-dlp"
        )

    os.makedirs(output_dir, exist_ok=True)
    out_template = os.path.join(output_dir, "%(title)s.%(ext)s")

    # Cap resolution to max_height for faster downloads
    format_selector = (
        f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]"
        f"/bestvideo[height<={max_height}]+bestaudio"
        f"/best[height<={max_height}]"
        f"/best"
    )

    ydl_opts = {
        "format": format_selector,
        "outtmpl": out_template,
        "merge_output_format": "mp4",
        "quiet": False,
        "no_warnings": False,
        "noplaylist": True,   # Only download the single video, not a playlist
    }

    print(f"[YouTube] Downloading (max {max_height}p): {url}", flush=True)
    t0 = time.time()

    try:
        import yt_dlp
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            # Resolve the actual filename yt-dlp wrote
            filename = ydl.prepare_filename(info)
            # yt-dlp may change the extension after merging
            if not os.path.exists(filename):
                base = os.path.splitext(filename)[0]
                filename = base + ".mp4"
            if not os.path.exists(filename):
                raise PipelineError(f"Downloaded file not found at expected path: {filename}")
    except yt_dlp.utils.DownloadError as exc:
        raise PipelineError(f"YouTube download failed: {exc}") from exc

    elapsed = time.time() - t0
    size_mb = os.path.getsize(filename) / (1024 * 1024)
    print(f"[YouTube] Downloaded in {elapsed:.1f}s — {size_mb:.1f} MB → {filename}", flush=True)
    return filename


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
    
    # Validate output_dir - ensure it's not a URL
    if args.output_dir.startswith(('http://', 'https://', 'ftp://')):
        raise ValueError(f"Invalid output directory: '{args.output_dir}'. Output directory must be a local file path, not a URL.")
    
    cfg.output_dir = args.output_dir
    cfg.whisper_model = args.whisper_model
    cfg.top_n_clips = args.top_n
    cfg.llm_enabled = args.llm
    if args.llm_endpoint:
        cfg.llm_endpoint = args.llm_endpoint
    cfg.llm_model = args.llm_model
    if args.keywords:
        cfg.keywords = args.keywords
    cfg.burn_subtitles = not args.no_subtitles
    cfg.use_cache = not args.no_cache
    cfg.language = args.language
    cfg.trim_silence = not args.no_trim_silence
    cfg.clip_tail_padding = args.clip_tail_padding

    # When LLM is enabled, give it real weight and reduce text/audio proportionally
    if cfg.llm_enabled:
        cfg.llm_weight = 0.4
        cfg.text_weight = 0.35
        cfg.audio_weight = 0.25

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
    print("[Transcriber] Starting...", flush=True)
    t0 = time.time()
    last_pct = [0]  # mutable so the closure can update it

    def _transcription_progress(pct: int) -> None:
        if pct > last_pct[0]:
            filled = pct // 5          # 0-20 blocks
            bar = "█" * filled + "░" * (20 - filled)
            print(f"\r[Transcriber] [{bar}] {pct}%  ", end="", flush=True)
            last_pct[0] = pct

    transcript = transcribe(config, wav_path, progress_callback=_transcription_progress)
    print(f"\r[Transcriber] Done in {time.time() - t0:.1f}s — {len(transcript.segments)} segment(s)          ", flush=True)

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
        report_path = generate_report(clip, scored_segments, transcript, clip_path, config)
        report_paths.append(report_path)
    print(f"[ReportGenerator] Done — {len(report_paths)} report(s) written", flush=True)

    return final_paths, clips


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate highlight clips from a video file or YouTube URL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 main.py input.mp4 --top-n 3 --whisper-model small\n"
            "  python3 main.py --url https://youtu.be/gzL2xoZLg9I --top-n 5 --llm"
        ),
    )
    parser.add_argument("input_video", nargs="?", default=None,
                        help="Path to the input video file (omit when using --url)")
    parser.add_argument("--url", default=None,
                        help="YouTube URL to download and clip automatically")
    parser.add_argument("--quality", type=int, default=720,
                        choices=[360, 480, 720, 1080],
                        help="Max video resolution for YouTube downloads (default: 720)")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for output clips (default: <input_video_folder>/highlights)")
    parser.add_argument("--whisper-model", default="base",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: base)")
    parser.add_argument("--top-n", type=int, default=5,
                        help="Number of highlight clips to generate (default: 5)")
    parser.add_argument("--llm", action="store_true",
                        help="Enable local LLM scoring via Ollama")
    parser.add_argument("--llm-endpoint", default=None,
                        help="LLM endpoint URL (default: OLLAMA_HOST env or http://localhost:11434/api/generate)")
    parser.add_argument("--llm-model", default="llama3",
                        help="LLM model name (default: llama3)")
    parser.add_argument("--keywords", nargs="+",
                        help="Keywords that boost segment scores")
    parser.add_argument("--no-subtitles", action="store_true",
                        help="Skip burning subtitles into clips (SRT files are still written)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force re-transcription, ignoring any cached transcript")
    parser.add_argument("--no-trim-silence", action="store_true",
                        help="Skip trimming leading/trailing silence from extracted clips")
    parser.add_argument("--clip-tail-padding", type=float, default=1.5,
                        help="Seconds of video to keep after the last word in a clip (default: 1.5)")
    parser.add_argument("--language", default="auto",
                        help="Transcription language code (default: auto). E.g. 'en', 'es', 'fr'")

    args = parser.parse_args()

    # Validate: must have either input_video or --url
    if not args.input_video and not args.url:
        parser.error("Provide either a video file path or --url <youtube_url>")
    if args.input_video and args.url:
        parser.error("Provide either a video file path or --url, not both")

    # Configure logging so pipeline INFO messages appear on stdout
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )

    work_dir = tempfile.mkdtemp(prefix="highlight_")

    # Handle YouTube download
    if args.url:
        # Download into a dedicated subfolder inside the work dir
        download_dir = os.path.join(work_dir, "download")
        try:
            args.input_video = download_youtube_video(args.url, download_dir, max_height=args.quality)
        except PipelineError as exc:
            print(f"\nError: {exc}", file=sys.stderr)
            shutil.rmtree(work_dir, ignore_errors=True)
            sys.exit(1)

    # Default output dir: a "highlights" folder next to the input video
    if args.output_dir is None:
        input_dir = os.path.dirname(os.path.abspath(args.input_video))
        args.output_dir = os.path.join(input_dir, "highlights")

    config = build_config(args, work_dir)

    print(f"Input:      {args.input_video}" + (f"  (from {args.url})" if args.url else ""))
    print(f"Output dir: {config.output_dir}")
    print(f"Whisper:    {config.whisper_model}")
    print(f"Top N:      {config.top_n_clips}")
    print(f"LLM:        {'enabled (' + config.llm_model + ')' if config.llm_enabled else 'disabled'}")
    print(f"Subtitles:  {'enabled' if config.burn_subtitles else 'disabled (SRT only)'}")
    print(f"Weights:    text={config.text_weight:.2f}  audio={config.audio_weight:.2f}  llm={config.llm_weight:.2f}")
    print()

    try:
        final_paths, clips = run_pipeline(args.input_video, config)
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
    for path, clip in zip(final_paths, clips):
        filename = os.path.basename(path)
        duration_s = int(round(clip.end - clip.start))
        score_str = f"{clip.score:.2f}"
        print(f"  {filename}  {duration_s}s  score={score_str}")
        base = os.path.splitext(path)[0]
        report = base + "_why_chosen.txt"
        if os.path.exists(report):
            print(f"    └─ {report}")


if __name__ == "__main__":
    main()
