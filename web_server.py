"""Video Highlight Generator — Web Server

Serves a web UI at http://0.0.0.0:6800 that replaces the tkinter GUI.

Usage:
    python3 web_server.py [--uploads-dir DIR] [--output-dir DIR]

The server:
- Accepts video uploads (saved to uploads/)
- Runs the pipeline in background threads
- Streams progress via Server-Sent Events (SSE)
- Serves completed clips as static files for download
- Supports multiple concurrent jobs with a job queue
"""

from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterator

# Ensure FFmpeg (Homebrew) is on PATH regardless of how the script is launched
_HOMEBREW_BIN = "/opt/homebrew/bin"
if _HOMEBREW_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _HOMEBREW_BIN + ":" + os.environ.get("PATH", "")

# Fix SSL certificate verification on macOS Python.org builds
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except ImportError:
    os.environ.setdefault("PYTHONHTTPSVERIFY", "0")

import io
import zipfile

from flask import Flask, Response, jsonify, request, send_file, send_from_directory

from config import Config
from main import download_youtube_video
from pipeline.audio_extractor import extract_audio
from pipeline.clip_extractor import extract_clips, generate_thumbnail
from pipeline.clip_selector import select_clips
from pipeline.exceptions import PipelineError
from pipeline.facecam_relocator import FacecamRelocator
from pipeline.models import SessionStore, VerticalFormattingJob
from pipeline.report_generator import generate_report
from pipeline.scorer import score_segments
from pipeline.subtitle_generator import generate_subtitles
from pipeline.transcriber import transcribe

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"
WEB_DIR = BASE_DIR / "web"

UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# When True, uploaded source videos are deleted after the job finishes (done or failed).
# Can be disabled via --no-cleanup-uploads CLI flag.
CLEANUP_UPLOADS: bool = True


# ---------------------------------------------------------------------------
# Job state
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    job_id: str
    video_path: str
    config: Config
    status: JobStatus = JobStatus.QUEUED
    log_lines: list[str] = field(default_factory=list)
    result_clips: list[dict] = field(default_factory=list)  # [{path, name, why_chosen}]
    error: str = ""
    created_at: float = field(default_factory=time.time)
    job_config_options: dict = field(default_factory=dict)
    # Cancellation signal — set this event to request cancellation
    cancel_event: threading.Event = field(default_factory=threading.Event)
    # SSE subscribers: each is a queue.Queue that receives log-line strings
    _subscribers: list[queue.Queue] = field(default_factory=list)

    def add_log(self, line: str) -> None:
        self.log_lines.append(line)
        for q in list(self._subscribers):
            try:
                q.put_nowait(line)
            except queue.Full:
                pass

    def add_progress(self, stage: int, total: int, stage_name: str = "", percentage: int = 0) -> None:
        """Emit a progress event to all SSE subscribers (not stored in log_lines)."""
        msg = json.dumps({
            "type": "progress",
            "stage": stage,
            "total": total,
            "stage_name": stage_name,
            "percentage": percentage
        })
        sentinel = f"__PROGRESS__:{msg}"
        for q in list(self._subscribers):
            try:
                q.put_nowait(sentinel)
            except queue.Full:
                pass

    def add_llm_progress(self, current: int, total: int) -> None:
        """Emit an LLM scoring progress event to all SSE subscribers."""
        if current == total:
            msg = json.dumps({"type": "llm_done", "scored": total})
        else:
            msg = json.dumps({"type": "llm_progress", "current": current, "total": total})
        sentinel = f"__LLM_PROGRESS__:{msg}"
        for q in list(self._subscribers):
            try:
                q.put_nowait(sentinel)
            except queue.Full:
                pass

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=500)
        # Replay existing log lines so a late subscriber catches up
        for line in self.log_lines:
            try:
                q.put_nowait(line)
            except queue.Full:
                break
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        try:
            self._subscribers.remove(q)
        except ValueError:
            pass


# In-memory job registry (job_id -> Job)
_jobs: dict[str, Job] = {}
_jobs_lock = threading.Lock()

# In-memory session store for mini-editor
_session_store = SessionStore()

# In-memory detection cache for mini-editor
# Key: (clip_path, frame_width, frame_height, mtime)
# Value: dict with facecam_region data or None
_detection_cache: dict[tuple, dict | None] = {}
_detection_cache_lock = threading.Lock()

# In-memory preview cache for mini-editor
# Key: (clip_path, facecam_x, facecam_y, facecam_width, facecam_height, mtime)
# Value: preview image path
_preview_cache: dict[tuple, str] = {}
_preview_cache_lock = threading.Lock()

# In-memory store for VerticalFormattingJob objects
# Key: job_id (str)
# Value: VerticalFormattingJob
_formatting_jobs: dict[str, "VerticalFormattingJob"] = {}
_formatting_jobs_lock = threading.Lock()

# Queue for VerticalFormattingJob IDs waiting to be processed
_formatting_job_queue: queue.Queue = queue.Queue()


def _vertical_formatting_worker() -> None:
    """Background worker thread that processes queued VerticalFormattingJobs.

    Runs indefinitely as a daemon thread.  Picks job IDs from
    ``_formatting_job_queue``, looks them up in ``_formatting_jobs``, and
    calls ``process_vertical_formatting_job()`` from the vertical formatter.
    """
    from pipeline.vertical_formatter import process_vertical_formatting_job as _process_job

    _wlog = logging.getLogger(__name__)
    while True:
        try:
            job_id = _formatting_job_queue.get(timeout=5)
        except queue.Empty:
            continue

        try:
            with _formatting_jobs_lock:
                formatting_job = _formatting_jobs.get(job_id)

            if formatting_job is None:
                _wlog.warning("Vertical formatting worker: job %s not found", job_id)
                continue

            if formatting_job.status == "cancelled":
                _wlog.info("Vertical formatting worker: job %s already cancelled", job_id)
                continue

            _wlog.info("Vertical formatting worker: starting job %s", job_id)
            _process_job(formatting_job)
            _wlog.info(
                "Vertical formatting worker: finished job %s with status %s",
                job_id, formatting_job.status,
            )
        except Exception as exc:  # noqa: BLE001
            _wlog.exception("Vertical formatting worker: unexpected error for job %s: %s", job_id, exc)
        finally:
            _formatting_job_queue.task_done()


# Start the background vertical formatting worker thread
_vertical_formatting_thread = threading.Thread(
    target=_vertical_formatting_worker,
    daemon=True,
    name="vertical-formatting-worker",
)
_vertical_formatting_thread.start()


def _get_job(job_id: str) -> Job | None:
    with _jobs_lock:
        return _jobs.get(job_id)


def _register_job(job: Job) -> None:
    with _jobs_lock:
        _jobs[job.job_id] = job


# ---------------------------------------------------------------------------
# Logging handler — forwards pipeline log records to the active job's SSE stream
# ---------------------------------------------------------------------------

class JobLogHandler(logging.Handler):
    """A logging.Handler that forwards pipeline log records to a Job's add_log()."""

    def __init__(self, job: Job) -> None:
        super().__init__()
        self._job = job

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self._job.add_log(msg)
        except Exception:
            self.handleError(record)


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def _get_video_duration(video_path: str) -> float:
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
        return 86400.0


class _CancelledError(Exception):
    """Raised internally when a job's cancel_event is set."""


def _run_pipeline_for_job(job: Job) -> None:
    """Execute the full pipeline for a job, posting log lines to job.add_log()."""

    def log(text: str) -> None:
        job.add_log(text)

    job.status = JobStatus.RUNNING
    log(f"[Job {job.job_id[:8]}] Starting pipeline")
    log(f"[Job {job.job_id[:8]}] Input: {job.video_path}")
    log(f"[Job {job.job_id[:8]}] Output dir: {job.config.output_dir}")

    # Attach a logging handler so pipeline module log records stream to this job
    handler = JobLogHandler(job)
    handler.setFormatter(logging.Formatter("%(name)s %(levelname)s %(message)s"))
    pipeline_logger = logging.getLogger("pipeline")
    pipeline_logger.setLevel(logging.INFO)
    pipeline_logger.addHandler(handler)

    # Determine total stages (7 stages, no shorts)
    total_stages = 7

    try:
        # Stage 1: Audio extraction (0% → 5%)
        if job.cancel_event.is_set():
            raise _CancelledError()
        job.add_progress(1, total_stages, "Audio Extraction", 5)
        log("[AudioExtractor] Starting...")
        t0 = time.time()
        wav_path = extract_audio(job.config, job.video_path)
        log(f"[AudioExtractor] Done in {time.time() - t0:.1f}s")

        # Stage 2: Transcription (5% → 60%)
        if job.cancel_event.is_set():
            raise _CancelledError()
        job.add_progress(2, total_stages, "Transcription", 10)
        log("[Transcriber] Starting...")
        t0 = time.time()
        
        # Define progress callback for transcription
        def transcription_progress(percentage: int) -> None:
            job.add_progress(2, total_stages, "Transcription", percentage)
        
        transcript = transcribe(job.config, wav_path, progress_callback=transcription_progress)
        log(f"[Transcriber] Done in {time.time() - t0:.1f}s — {len(transcript.segments)} segment(s)")
        job.add_progress(2, total_stages, "Transcription", 60)

        if not transcript.segments:
            log("[Transcriber] Warning: No speech detected — scoring on audio energy only")

        # Stage 3: Scoring (60% → 70%)
        if job.cancel_event.is_set():
            raise _CancelledError()
        job.add_progress(3, total_stages, "Scoring Segments", 65)
        log("[Scorer] Starting...")
        t0 = time.time()

        def llm_progress_callback(current: int, total: int) -> None:
            job.add_llm_progress(current, total)

        scored_segments = score_segments(job.config, transcript, wav_path,
                                         llm_progress_callback=llm_progress_callback)
        log(f"[Scorer] Done in {time.time() - t0:.1f}s — {len(scored_segments)} segment(s) scored")
        job.add_progress(3, total_stages, "Scoring Segments", 70)

        if not scored_segments:
            raise PipelineError("No segments to score. The video may have no audio content.")

        # Stage 4: Clip selection (70% → 75%)
        if job.cancel_event.is_set():
            raise _CancelledError()
        job.add_progress(4, total_stages, "Selecting Clips", 72)
        log("[ClipSelector] Starting...")
        t0 = time.time()
        video_duration = _get_video_duration(job.video_path)
        clips = select_clips(job.config, scored_segments, transcript, video_duration)
        log(f"[ClipSelector] Done in {time.time() - t0:.1f}s — {len(clips)} clip(s) selected")
        job.add_progress(4, total_stages, "Selecting Clips", 75)

        if not clips:
            raise PipelineError("No clips selected. Try lowering Top N or check the video.")

        # Stage 5: Clip extraction (75% → 85%)
        if job.cancel_event.is_set():
            raise _CancelledError()
        job.add_progress(5, total_stages, "Extracting Clips", 77)
        log("[ClipExtractor] Starting...")
        t0 = time.time()
        clip_paths = extract_clips(job.config, clips, job.video_path)
        log(f"[ClipExtractor] Done in {time.time() - t0:.1f}s")
        job.add_progress(5, total_stages, "Extracting Clips", 85)

        # Stage 6: Subtitle generation (85% → 95%)
        job.add_progress(6, total_stages, "Generating Subtitles", 87)
        log("[SubtitleGenerator] Starting...")
        t0 = time.time()
        final_paths = generate_subtitles(job.config, clips, transcript, clip_paths)
        log(f"[SubtitleGenerator] Done in {time.time() - t0:.1f}s")
        job.add_progress(6, total_stages, "Generating Subtitles", 95)

        # Stage 7: Why-chosen reports (95% → 100% if no shorts, else 95% → 98%)
        job.add_progress(7, total_stages, "Generating Reports", 97)
        log("[ReportGenerator] Writing selection reports...")
        t0 = time.time()
        result_clips: list[dict] = []
        for clip, clip_path in zip(clips, final_paths):
            report_path = generate_report(clip, scored_segments, transcript, clip_path, job.config)
            why_text = ""
            try:
                with open(report_path, encoding="utf-8") as fh:
                    why_text = fh.read()
            except OSError:
                pass

            # Generate thumbnail (non-fatal)
            thumb_path = generate_thumbnail(clip_path)
            thumbnail_name = os.path.basename(thumb_path) if thumb_path else None

            result_clips.append({
                "path": clip_path,
                "name": os.path.basename(clip_path),
                "why_chosen": why_text,
                "report_path": report_path,
                "timestamp_range": _format_timestamp_range(clip.start, clip.end),
                "duration": f"{int(round(clip.end - clip.start))}s",
                "score": f"{clip.score:.2f}",
                "thumbnail_name": thumbnail_name,
                "start": clip.start,  # Add start time for subtitle generation
                "end": clip.end,      # Add end time for subtitle generation
            })
        log(f"[ReportGenerator] Done in {time.time() - t0:.1f}s — {len(result_clips)} report(s)")
        job.add_progress(7, total_stages, "Generating Reports", 100)

        # Save transcript to JSON for subtitle burning in vertical formatter
        transcript_path = Path(job.config.output_dir) / "transcript.json"
        try:
            with open(transcript_path, "w", encoding="utf-8") as fh:
                json.dump(transcript.to_dict(), fh, indent=2)
            log(f"[ReportGenerator] Saved transcript to {transcript_path}")
        except Exception as exc:
            log(f"[ReportGenerator] Warning: Failed to save transcript: {exc}")

        # Clean up temp dir
        shutil.rmtree(job.config.work_dir, ignore_errors=True)

        job.result_clips = result_clips
        job.status = JobStatus.DONE
        log(f"[Job {job.job_id[:8]}] ✓ Pipeline complete — {len(result_clips)} clip(s) exported")
    except _CancelledError:
        job.status = JobStatus.CANCELLED
        job.error = "Cancelled by user"
        log(f"[Job {job.job_id[:8]}] ✗ Job cancelled")
        # Clean up temp dir on cancellation
        shutil.rmtree(job.config.work_dir, ignore_errors=True)
    except PipelineError as exc:
        job.error = str(exc)
        job.status = JobStatus.FAILED
        log(f"[Job {job.job_id[:8]}] ✗ Pipeline error: {exc}")
    except Exception as exc:
        job.error = f"Unexpected error: {exc}"
        job.status = JobStatus.FAILED
        log(f"[Job {job.job_id[:8]}] ✗ Unexpected error: {exc}")
    finally:
        # Remove the per-job logging handler
        pipeline_logger.removeHandler(handler)

        # Delete the uploaded source video now that the job is terminal
        _logger = logging.getLogger(__name__)
        if CLEANUP_UPLOADS and os.path.exists(job.video_path):
            try:
                os.remove(job.video_path)
                _logger.info("[Cleanup] Deleted uploaded file: %s", job.video_path)
            except OSError as e:
                _logger.warning(
                    "[Cleanup] Failed to delete uploaded file %s: %s", job.video_path, e
                )

        # Signal all SSE subscribers that the stream is finished
        job.add_log("__DONE__")


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__, static_folder=None)


@app.route("/")
def index():
    """Serve the single-page frontend."""
    index_path = WEB_DIR / "index.html"
    if index_path.exists():
        return send_from_directory(str(WEB_DIR), "index.html")
    return Response("web/index.html not found", status=404)


@app.route("/mini-editor")
def serve_mini_editor():
    """Serve the Mini Video Editor single-page app."""
    mini_editor_path = WEB_DIR / "mini_editor.html"
    if mini_editor_path.exists():
        return send_from_directory(str(WEB_DIR), "mini_editor.html")
    return Response("web/mini_editor.html not found", status=404)


@app.route("/output/<path:filename>")
def serve_output(filename: str):
    """Serve completed clip files for download."""
    return send_from_directory(str(OUTPUT_DIR), filename)


# ---------------------------------------------------------------------------
# API: Jobs
# ---------------------------------------------------------------------------

@app.route("/api/jobs", methods=["GET"])
def list_jobs():
    """Return all jobs ordered by creation time (newest first)."""
    with _jobs_lock:
        jobs_snapshot = list(_jobs.values())
    jobs_snapshot.sort(key=lambda j: j.created_at, reverse=True)
    return jsonify([_job_summary(j) for j in jobs_snapshot])


@app.route("/api/jobs/<job_id>", methods=["GET"])
def get_job(job_id: str):
    job = _get_job(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(_job_detail(job))


@app.route("/api/jobs", methods=["POST"])
def create_job():
    """Accept a video upload (or YouTube URL) and enqueue a new pipeline job."""
    reuse_video_path = request.form.get("reuse_video_path", "").strip()
    youtube_url = request.form.get("youtube_url", "").strip()

    if reuse_video_path:
        # Re-run path: use an existing server-side file instead of uploading
        if not os.path.exists(reuse_video_path):
            return jsonify({"error": "Original video file no longer exists on the server"}), 400
        upload_path = Path(reuse_video_path)
    elif youtube_url:
        # YouTube download path
        try:
            quality = int(request.form.get("youtube_quality", "720"))
        except (ValueError, TypeError):
            quality = 720
        try:
            downloaded = download_youtube_video(youtube_url, str(UPLOADS_DIR), max_height=quality)
            upload_path = Path(downloaded)
        except PipelineError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            return jsonify({"error": f"YouTube download failed: {exc}"}), 400
    else:
        if "video" not in request.files:
            return jsonify({"error": "No video file provided"}), 400

        video_file = request.files["video"]
        if not video_file.filename:
            return jsonify({"error": "Empty filename"}), 400

        # Save uploaded video
        safe_name = _safe_filename(video_file.filename)
        upload_path = UPLOADS_DIR / safe_name
        video_file.save(str(upload_path))

    # Parse options from form data
    whisper_model = request.form.get("whisper_model", "base")
    top_n = int(request.form.get("top_n", "5"))
    keywords_raw = request.form.get("keywords", "")
    llm_enabled = request.form.get("llm_enabled", "false").lower() == "true"
    llm_model = request.form.get("llm_model", "llama3")
    output_dir = request.form.get("output_dir", "").strip() or str(OUTPUT_DIR)
    burn_subtitles = request.form.get("burn_subtitles", "true").lower() != "false"
    trim_silence = request.form.get("trim_silence", "true").lower() != "false"
    shorts_enabled = request.form.get("shorts_enabled", "false").lower() == "true"
    subtitle_style = request.form.get("subtitle_style", "bubble").strip() or "bubble"
    genre = request.form.get("genre", "auto").strip() or "auto"
    platform = request.form.get("platform", "none").strip() or "none"
    language = request.form.get("language", "auto").strip() or "auto"

    # Advanced settings (with safe float parsing)
    def _float(key: str, default: float) -> float:
        try:
            return float(request.form.get(key, default))
        except (ValueError, TypeError):
            return default

    def _bool_field(key: str, default: bool) -> bool:
        val = request.form.get(key, None)
        if val is None:
            return default
        return val.lower() not in ("false", "0", "no")

    adv_text_weight       = _float("text_weight", 0.5)
    adv_audio_weight      = _float("audio_weight", 0.5)
    adv_min_text_score    = _float("min_text_score", 0.05)
    adv_reaction_weight   = _float("reaction_weight", 3.0)
    adv_min_clip_duration = _float("min_clip_duration", 30.0)
    adv_max_clip_duration = _float("max_clip_duration", 100.0)
    adv_min_clip_spacing  = _float("min_clip_spacing", 300.0)
    adv_spike_pct         = _float("llm_audio_spike_percentage", 0.2)
    adv_llm_audio_gate    = _bool_field("llm_audio_gate", True)
    adv_rep_threshold     = _float("repetition_penalty_threshold", 0.4)
    adv_rep_multiplier    = _float("repetition_penalty_multiplier", 0.5)
    adv_tail_padding      = _float("clip_tail_padding", 1.5)

    # Build config
    work_dir = tempfile.mkdtemp(prefix="highlight_web_")
    cfg = Config(work_dir=work_dir)
    
    # Validate output_dir - ensure it's not a URL
    if output_dir.startswith(('http://', 'https://', 'ftp://')):
        return jsonify({"error": f"Invalid output directory: '{output_dir}'. Output directory must be a local file path, not a URL."}), 400
    
    cfg.output_dir = output_dir
    cfg.whisper_model = whisper_model
    cfg.top_n_clips = top_n
    cfg.llm_enabled = llm_enabled
    cfg.llm_model = llm_model
    cfg.burn_subtitles = burn_subtitles
    cfg.trim_silence = trim_silence
    cfg.shorts_enabled = shorts_enabled
    cfg.subtitle_style = subtitle_style
    cfg.genre = genre
    cfg.platform = platform
    cfg.language = language

    # Apply advanced settings
    cfg.text_weight = adv_text_weight
    cfg.audio_weight = adv_audio_weight
    cfg.min_text_score_for_selection = adv_min_text_score
    cfg.reaction_weight = adv_reaction_weight
    cfg.min_clip_duration = adv_min_clip_duration
    cfg.max_clip_duration = adv_max_clip_duration
    cfg.min_clip_spacing = adv_min_clip_spacing
    cfg.llm_audio_spike_percentage = adv_spike_pct
    cfg.llm_audio_gate = adv_llm_audio_gate
    cfg.repetition_penalty_threshold = adv_rep_threshold
    cfg.repetition_penalty_multiplier = adv_rep_multiplier
    cfg.clip_tail_padding = adv_tail_padding

    if keywords_raw.strip():
        cfg.keywords = [k.strip() for k in keywords_raw.split(",") if k.strip()]

    if llm_enabled:
        cfg.llm_weight = 0.4
        cfg.text_weight = 0.35
        cfg.audio_weight = 0.25

        # Validate LLM model availability
        from pipeline.scorer import _check_llm_model_available
        if not _check_llm_model_available(cfg):
            return jsonify({
                "error": f"LLM model '{llm_model}' is not available. "
                        f"Make sure Ollama is running and the model is pulled: ollama pull {llm_model}"
            }), 400

    # Create and register job
    job_id = str(uuid.uuid4())
    job = Job(job_id=job_id, video_path=str(upload_path), config=cfg)
    job.job_config_options = {
        "whisper_model": whisper_model,
        "top_n": top_n,
        "keywords": keywords_raw,
        "llm_enabled": llm_enabled,
        "llm_model": llm_model,
        "output_dir": output_dir,
        "burn_subtitles": burn_subtitles,
        "trim_silence": trim_silence,
        "shorts_enabled": shorts_enabled,
        "subtitle_style": subtitle_style,
        "genre": genre,
        "platform": platform,
        "language": language,
        "original_video_path": str(upload_path),
        "youtube_url": youtube_url,
    }
    _register_job(job)

    # Launch pipeline in background thread
    t = threading.Thread(target=_run_pipeline_for_job, args=(job,), daemon=True)
    t.start()

    return jsonify({"job_id": job_id, "status": JobStatus.RUNNING}), 201


@app.route("/api/jobs/<job_id>/log-stream")
def job_log_stream(job_id: str):
    """SSE endpoint — streams log lines for a job as they are produced."""
    job = _get_job(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404

    def generate() -> Iterator[str]:
        q = job.subscribe()
        try:
            while True:
                try:
                    line = q.get(timeout=30)
                except queue.Empty:
                    # Send a keep-alive comment so the connection stays open
                    yield ": keep-alive\n\n"
                    continue

                if line == "__DONE__":
                    yield f"data: {json.dumps({'type': 'done', 'status': job.status})}\n\n"
                    break

                if line.startswith("__PROGRESS__:"):
                    payload = line[len("__PROGRESS__:"):]
                    yield f"data: {payload}\n\n"
                    continue

                if line.startswith("__LLM_PROGRESS__:"):
                    payload = line[len("__LLM_PROGRESS__:"):]
                    yield f"data: {payload}\n\n"
                    continue

                yield f"data: {json.dumps({'type': 'log', 'line': line})}\n\n"
        finally:
            job.unsubscribe(q)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/jobs/<job_id>/clips/<int:clip_index>/download")
def download_clip(job_id: str, clip_index: int):
    """Download a specific clip from a completed job."""
    job = _get_job(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404
    if job.status != JobStatus.DONE:
        return jsonify({"error": "Job not complete"}), 400
    if clip_index < 0 or clip_index >= len(job.result_clips):
        return jsonify({"error": "Clip index out of range"}), 404

    clip_path = job.result_clips[clip_index]["path"]
    if not os.path.exists(clip_path):
        return jsonify({"error": "Clip file not found on disk"}), 404

    return send_file(
        clip_path,
        as_attachment=True,
        download_name=os.path.basename(clip_path),
        mimetype="video/mp4",
    )


@app.route("/api/jobs/<job_id>/download-all")
def download_all_clips(job_id: str):
    """Stream a zip archive of all clips for a completed job."""
    job = _get_job(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404
    if job.status != JobStatus.DONE:
        return jsonify({"error": "Job not complete"}), 400
    if len(job.result_clips) < 2:
        return jsonify({"error": "Job has fewer than 2 clips"}), 400

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for clip in job.result_clips:
            clip_path = clip["path"]
            if os.path.exists(clip_path):
                zf.write(clip_path, arcname=os.path.basename(clip_path))
    buf.seek(0)

    zip_name = f"clips_{job_id[:8]}.zip"
    return Response(
        buf.read(),
        mimetype="application/zip",
        headers={"Content-Disposition": f"attachment; filename=\"{zip_name}\""},
    )


@app.route("/api/jobs/<job_id>/cancel", methods=["POST"])
def cancel_job(job_id: str):
    """Signal a running job to stop at the next checkpoint."""
    job = _get_job(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404
    if job.status not in (JobStatus.QUEUED, JobStatus.RUNNING):
        return jsonify({"error": f"Job is already {job.status} and cannot be cancelled"}), 400
    job.cancel_event.set()
    return jsonify({"job_id": job_id, "status": "cancelling"}), 200


# ---------------------------------------------------------------------------
# API: Ollama models
# ---------------------------------------------------------------------------

@app.route("/api/ollama/models", methods=["GET"])
def list_ollama_models():
    """Return a list of locally available Ollama models via the Ollama HTTP API."""
    import requests as _requests

    # Build the tags URL from the configured LLM endpoint
    from config import Config as _Cfg
    _tmp_cfg = _Cfg(work_dir="/tmp")
    endpoint = _tmp_cfg.llm_endpoint.rstrip("/")
    if endpoint.endswith("/api/generate"):
        base_url = endpoint[: -len("/api/generate")]
    else:
        base_url = endpoint

    tags_url = f"{base_url}/api/tags"

    try:
        response = _requests.get(tags_url, timeout=5)
    except Exception as exc:
        return jsonify({"error": f"Could not reach Ollama at {tags_url}: {exc}", "models": []}), 200

    if response.status_code != 200:
        return jsonify({"error": f"Ollama returned HTTP {response.status_code}", "models": []}), 200

    try:
        data = response.json()
        raw_models = data.get("models", [])
        # Keep full name but strip ":latest" only (it's redundant noise).
        # Other tags like ":1b", ":7b-instruct" are meaningful and must be preserved.
        models = []
        seen = set()
        for entry in raw_models:
            name = entry.get("name", "") if isinstance(entry, dict) else str(entry)
            # Only strip the tag if it's literally ":latest"
            if name.endswith(":latest"):
                name = name[: -len(":latest")]
            if name and name not in seen:
                seen.add(name)
                models.append(name)
        return jsonify({"models": models, "error": None})
    except Exception as exc:
        return jsonify({"error": f"Failed to parse Ollama response: {exc}", "models": []}), 200


# ---------------------------------------------------------------------------
# API: Mini Video Editor
# ---------------------------------------------------------------------------

@app.route("/api/mini-editor/session", methods=["POST"])
def create_mini_editor_session():
    """Initialize a mini video editor session.
    
    Request JSON:
        {
            "clip_batch_id": str,           # Job ID or batch identifier
            "reference_clip_path": str      # Path to the reference clip (first clip)
        }
    
    Response JSON:
        {
            "session_id": str,              # UUID for the session
            "clips": [                      # List of clips in the batch
                {
                    "path": str,
                    "name": str,
                    "resolution": [int, int]  # [width, height]
                }
            ],
            "reference_clip": {             # Details of the reference clip
                "path": str,
                "name": str,
                "resolution": [int, int]
            },
            "error": str | null,
            "version": str                  # Backend version for debugging
        }
    """
    try:
        data = request.get_json()
    except Exception:
        return jsonify({"error": "Request body must be JSON"}), 400
    
    try:
        if not data:
            return jsonify({"error": "Request body must be JSON"}), 400
        
        clip_batch_id = data.get("clip_batch_id")
        reference_clip_path = data.get("reference_clip_path")
        
        # Validate required fields
        if not clip_batch_id:
            return jsonify({"error": "clip_batch_id is required"}), 400
        if not reference_clip_path:
            return jsonify({"error": "reference_clip_path is required"}), 400
        
        # Check if reference clip exists
        if not os.path.exists(reference_clip_path):
            return jsonify({"error": f"Reference clip not found: {reference_clip_path}"}), 404
        
        # Get the job to retrieve all clips in the batch
        job = _get_job(clip_batch_id)
        if job is None:
            return jsonify({"error": f"Job not found: {clip_batch_id}"}), 404
        
        if job.status != JobStatus.DONE:
            return jsonify({"error": "Job is not complete yet"}), 400
        
        if not job.result_clips:
            return jsonify({"error": "Job has no clips"}), 400
        
        # Get resolution of reference clip using FFprobe
        import subprocess
        ffprobe_cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "json",
            reference_clip_path,
        ]
        
        result = subprocess.run(
            ffprobe_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        
        if result.returncode != 0:
            return jsonify({
                "error": f"Failed to get video resolution: {result.stderr[:200]}"
            }), 500
        
        try:
            probe_data = json.loads(result.stdout)
            streams = probe_data.get("streams", [])
            if not streams:
                return jsonify({"error": "No video stream found in reference clip"}), 400
            
            frame_width = streams[0].get("width")
            frame_height = streams[0].get("height")
            
            if not frame_width or not frame_height:
                return jsonify({"error": "Could not determine video resolution"}), 400
        except (json.JSONDecodeError, KeyError, IndexError) as exc:
            return jsonify({"error": f"Failed to parse FFprobe output: {exc}"}), 500
        
        reference_resolution = (frame_width, frame_height)
        
        # Run facecam detection on reference clip
        config = Config(work_dir=tempfile.gettempdir())
        relocator = FacecamRelocator()
        facecam_region = relocator.detect_facecam(
            clip_path=reference_clip_path,
            frame_width=frame_width,
            frame_height=frame_height,
            config=config,
        )
        
        # If detection fails, create a default region (will be handled by frontend)
        if facecam_region is None:
            # Create a default facecam region in top-right corner (common placement)
            # Use 15% of frame area as default
            default_width = int(frame_width * 0.4)
            default_height = int(frame_height * 0.4)
            default_x = frame_width - default_width - 10
            default_y = 10
            
            from pipeline.models import FacecamRegion
            facecam_region = FacecamRegion(
                x=default_x,
                y=default_y,
                width=default_width,
                height=default_height,
                corner="top-right",
                confidence=0.0,  # Zero confidence indicates default/manual placement
            )
        
        # Compute canvas layout
        from pipeline.frame_reformatter import compute_canvas_layout
        canvas_layout = compute_canvas_layout(config)
        
        # Create session
        session = _session_store.create_session(
            clip_batch_id=clip_batch_id,
            reference_clip_path=reference_clip_path,
            reference_resolution=reference_resolution,
            facecam_region=facecam_region,
            canvas_layout=canvas_layout,
        )
        
        # Build clips list with resolution info
        clips = []
        for clip_data in job.result_clips:
            clip_path = clip_data["path"]
            
            # Get resolution for each clip
            ffprobe_cmd = [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "json",
                clip_path,
            ]
            
            result = subprocess.run(
                ffprobe_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            
            if result.returncode == 0:
                try:
                    probe_data = json.loads(result.stdout)
                    streams = probe_data.get("streams", [])
                    if streams:
                        clip_width = streams[0].get("width", frame_width)
                        clip_height = streams[0].get("height", frame_height)
                    else:
                        clip_width, clip_height = frame_width, frame_height
                except (json.JSONDecodeError, KeyError, IndexError):
                    clip_width, clip_height = frame_width, frame_height
            else:
                # Fallback to reference resolution if probe fails
                clip_width, clip_height = frame_width, frame_height
            
            clips.append({
                "path": clip_path,
                "name": clip_data["name"],
                "resolution": [clip_width, clip_height],
            })
        
        # Build reference clip info
        reference_clip = {
            "path": reference_clip_path,
            "name": os.path.basename(reference_clip_path),
            "resolution": [frame_width, frame_height],
        }
        
        return jsonify({
            "session_id": session.session_id,
            "clips": clips,
            "reference_clip": reference_clip,
            "error": None,
            "version": "2.0-crop",  # Version identifier to verify server restart
        }), 201
        
    except Exception as exc:
        _logger = logging.getLogger(__name__)
        _logger.exception("Error in create_mini_editor_session")
        return jsonify({
            "session_id": None,
            "clips": [],
            "reference_clip": None,
            "error": f"Internal error: {str(exc)}",
        }), 500


@app.route("/api/mini-editor/detect", methods=["POST"])
def detect_facecam_endpoint():
    """Detect facecam region in a clip using FacecamRelocator.
    
    Request JSON:
        {
            "clip_path": str,           # Path to the clip file
            "frame_width": int,         # Frame width in pixels
            "frame_height": int,        # Frame height in pixels
            "config": dict (optional)   # Config overrides
        }
    
    Response JSON:
        {
            "facecam_region": {
                "x": int,
                "y": int,
                "width": int,
                "height": int,
                "corner": str,
                "confidence": float
            } | null,
            "error": str | null,
            "cached": bool              # True if result was from cache
        }
    """
    try:
        data = request.get_json()
    except Exception:
        return jsonify({"error": "Request body must be JSON"}), 400
    
    try:
        if not data:
            return jsonify({"error": "Request body must be JSON"}), 400
        
        clip_path = data.get("clip_path")
        frame_width = data.get("frame_width")
        frame_height = data.get("frame_height")
        
        # Validate required fields
        if not clip_path:
            return jsonify({"error": "clip_path is required"}), 400
        if not frame_width or not isinstance(frame_width, int):
            return jsonify({"error": "frame_width must be an integer"}), 400
        if not frame_height or not isinstance(frame_height, int):
            return jsonify({"error": "frame_height must be an integer"}), 400
        
        # Check if clip exists
        if not os.path.exists(clip_path):
            return jsonify({"error": f"Clip not found: {clip_path}"}), 404
        
        # Get file modification time for cache invalidation
        try:
            mtime = os.path.getmtime(clip_path)
        except OSError:
            mtime = 0
        
        # Build cache key
        cache_key = (clip_path, frame_width, frame_height, mtime)
        
        # Check cache first
        with _detection_cache_lock:
            cached_result = _detection_cache.get(cache_key)
        
        if cached_result is not None:
            # Cache hit - return cached result
            return jsonify({
                **cached_result,
                "cached": True
            }), 200
        
        # Cache miss - run detection
        # Build config (use defaults, allow overrides)
        config_overrides = data.get("config", {})
        config = Config(work_dir=tempfile.gettempdir())
        
        # Apply config overrides if provided
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Run detection
        relocator = FacecamRelocator()
        facecam_region = relocator.detect_facecam(
            clip_path=clip_path,
            frame_width=frame_width,
            frame_height=frame_height,
            config=config,
        )
        
        # Build result
        if facecam_region is None:
            # Determine reason for detection failure
            # Check area fraction to give a more specific reason
            reason = "no_pip_detected"
            # We can't know the exact reason without more info, but we can
            # provide a structured error with fallback options
            result = {
                "facecam_region": None,
                "error": "No valid facecam region detected",
                "reason": reason,
                "offer_manual_selection": True,
                "offer_fallback": True,
            }
        else:
            result = {
                "facecam_region": {
                    "x": facecam_region.x,
                    "y": facecam_region.y,
                    "width": facecam_region.width,
                    "height": facecam_region.height,
                    "corner": facecam_region.corner,
                    "confidence": facecam_region.confidence,
                },
                "error": None,
                "reason": None,
                "offer_manual_selection": False,
                "offer_fallback": False,
            }
        
        # Store in cache
        with _detection_cache_lock:
            _detection_cache[cache_key] = result
        
        # Return result with cached=False
        return jsonify({
            **result,
            "cached": False
        }), 200
        
    except Exception as exc:
        _logger = logging.getLogger(__name__)
        _logger.exception("Error in detect_facecam_endpoint")
        return jsonify({
            "facecam_region": None,
            "error": f"Internal error: {str(exc)}",
            "cached": False
        }), 500


@app.route("/api/mini-editor/manual-region", methods=["POST"])
def set_manual_region_endpoint():
    """Accept a manually-drawn bounding box and store it in the session.

    Allows users to manually specify a facecam region when auto-detection fails.
    The region is validated against area fraction constraints before being stored.

    Request JSON:
        {
            "session_id": str,
            "facecam_region": {
                "x": int, "y": int, "width": int, "height": int,
                "corner": str (optional), "confidence": float (optional)
            }
        }

    Response JSON:
        { "facecam_region": { ... }, "error": str | null }
    """
    try:
        data = request.get_json()
    except Exception:
        return jsonify({"error": "Request body must be JSON"}), 400

    try:
        if not data:
            return jsonify({"error": "Request body must be JSON"}), 400

        session_id = data.get("session_id")
        facecam_region_data = data.get("facecam_region")

        if not session_id:
            return jsonify({"error": "session_id is required"}), 400
        if not facecam_region_data:
            return jsonify({"error": "facecam_region is required"}), 400

        for fld in ["x", "y", "width", "height"]:
            if fld not in facecam_region_data:
                return jsonify({"error": f"facecam_region.{fld} is required"}), 400

        session = _session_store.get_session(session_id)
        if session is None:
            return jsonify({"error": f"Session not found or expired: {session_id}"}), 404

        frame_width, frame_height = session.reference_resolution

        try:
            x = int(facecam_region_data["x"])
            y = int(facecam_region_data["y"])
            width = int(facecam_region_data["width"])
            height = int(facecam_region_data["height"])
        except (ValueError, TypeError) as exc:
            return jsonify({"error": f"Invalid facecam_region values: {exc}"}), 400

        if x < 0 or y < 0:
            return jsonify({"error": "facecam_region x and y must be >= 0"}), 400
        if width <= 0 or height <= 0:
            return jsonify({"error": "facecam_region width and height must be > 0"}), 400
        if x + width > frame_width:
            return jsonify({"error": f"facecam_region extends beyond frame width ({frame_width})"}), 400
        if y + height > frame_height:
            return jsonify({"error": f"facecam_region extends beyond frame height ({frame_height})"}), 400

        config = Config(work_dir=tempfile.gettempdir())
        frame_area = frame_width * frame_height
        region_area = width * height
        area_fraction = region_area / frame_area

        if area_fraction < config.facecam_min_area_fraction:
            return jsonify({
                "error": (
                    f"Region area fraction ({area_fraction:.3f}) is below minimum "
                    f"({config.facecam_min_area_fraction}). The region is too small."
                )
            }), 400
        if area_fraction > config.facecam_max_area_fraction:
            return jsonify({
                "error": (
                    f"Region area fraction ({area_fraction:.3f}) exceeds maximum "
                    f"({config.facecam_max_area_fraction}). The region is too large."
                )
            }), 400

        # Compute corner from position if not provided
        center_x = x + width / 2
        center_y = y + height / 2
        if center_x < frame_width / 2:
            computed_corner = "top-left" if center_y < frame_height / 2 else "bottom-left"
        else:
            computed_corner = "top-right" if center_y < frame_height / 2 else "bottom-right"

        corner = facecam_region_data.get("corner", computed_corner)
        confidence = float(facecam_region_data.get("confidence", 1.0))

        from pipeline.models import FacecamRegion as _FacecamRegion
        new_region = _FacecamRegion(
            x=x, y=y, width=width, height=height,
            corner=corner, confidence=confidence,
        )

        # Push current region to undo history before updating
        session.push_undo(session.facecam_region)
        session.facecam_region = new_region
        session.refresh_expiry()

        return jsonify({
            "facecam_region": {
                "x": new_region.x,
                "y": new_region.y,
                "width": new_region.width,
                "height": new_region.height,
                "corner": new_region.corner,
                "confidence": new_region.confidence,
            },
            "error": None,
        }), 200

    except Exception as exc:
        _logger = logging.getLogger(__name__)
        _logger.exception("Error in set_manual_region_endpoint")
        return jsonify({"error": f"Internal error: {str(exc)}"}), 500


@app.route("/api/mini-editor/fallback", methods=["POST"])
def set_fallback_fill_endpoint():
    """Set use_fallback_fill=True on the session.

    When fallback fill is enabled, the vertical formatter will use blurred
    gameplay fill in the top region instead of a facecam overlay.

    Request JSON:
        { "session_id": str }

    Response JSON:
        { "use_fallback_fill": bool, "error": str | null }
    """
    try:
        data = request.get_json()
    except Exception:
        return jsonify({"error": "Request body must be JSON"}), 400

    try:
        if not data:
            return jsonify({"error": "Request body must be JSON"}), 400

        session_id = data.get("session_id")
        if not session_id:
            return jsonify({"error": "session_id is required"}), 400

        session = _session_store.get_session(session_id)
        if session is None:
            return jsonify({"error": f"Session not found or expired: {session_id}"}), 404

        session.settings["use_fallback_fill"] = True
        session.refresh_expiry()

        return jsonify({
            "use_fallback_fill": True,
            "error": None,
        }), 200

    except Exception as exc:
        _logger = logging.getLogger(__name__)
        _logger.exception("Error in set_fallback_fill_endpoint")
        return jsonify({"error": f"Internal error: {str(exc)}"}), 500


@app.route("/api/mini-editor/preview", methods=["POST"])
def generate_preview_endpoint():
    """Generate a preview image showing the vertical canvas with facecam placement.


    Request JSON:
        {
            "clip_path": str,           # Path to the clip file
            "facecam_region": {         # Facecam region to preview
                "x": int,
                "y": int,
                "width": int,
                "height": int,
                "corner": str,
                "confidence": float
            },
            "frame_width": int,         # Source frame width
            "frame_height": int,        # Source frame height
            "config": dict (optional)   # Config overrides
        }
    
    Response JSON:
        {
            "preview_image_url": str,   # URL to the preview image
            "canvas_layout": {          # Canvas layout used
                "canvas_width": int,
                "canvas_height": int,
                "facecam_x": int,
                "facecam_y": int,
                "facecam_width": int,
                "facecam_height": int,
                "gameplay_x": int,
                "gameplay_y": int,
                "gameplay_width": int,
                "gameplay_height": int
            },
            "error": str | null,
            "cached": bool              # True if result was from cache
        }
    """
    try:
        data = request.get_json()
    except Exception:
        return jsonify({"error": "Request body must be JSON"}), 400
    
    try:
        if not data:
            return jsonify({"error": "Request body must be JSON"}), 400
        
        clip_path = data.get("clip_path")
        facecam_region_data = data.get("facecam_region")
        frame_width = data.get("frame_width")
        frame_height = data.get("frame_height")
        
        # Validate required fields
        if not clip_path:
            return jsonify({"error": "clip_path is required"}), 400
        if not facecam_region_data:
            return jsonify({"error": "facecam_region is required"}), 400
        if not frame_width or not isinstance(frame_width, int):
            return jsonify({"error": "frame_width must be an integer"}), 400
        if not frame_height or not isinstance(frame_height, int):
            return jsonify({"error": "frame_height must be an integer"}), 400
        
        # Validate facecam_region structure
        required_region_fields = ["x", "y", "width", "height", "corner", "confidence"]
        for field in required_region_fields:
            if field not in facecam_region_data:
                return jsonify({"error": f"facecam_region.{field} is required"}), 400
        
        # Check if clip exists
        if not os.path.exists(clip_path):
            return jsonify({"error": f"Clip not found: {clip_path}"}), 404
        
        # Get file modification time for cache invalidation
        try:
            mtime = os.path.getmtime(clip_path)
        except OSError:
            mtime = 0
        
        # Build cache key based on facecam region coordinates
        cache_key = (
            clip_path,
            facecam_region_data["x"],
            facecam_region_data["y"],
            facecam_region_data["width"],
            facecam_region_data["height"],
            mtime
        )
        
        # Check cache first
        with _preview_cache_lock:
            cached_preview_path = _preview_cache.get(cache_key)
        
        if cached_preview_path and os.path.exists(cached_preview_path):
            # Cache hit - return cached preview
            _logger = logging.getLogger(__name__)
            _logger.info(f"Preview cache HIT for {cache_key}")
            
            # Build config to get canvas layout
            config_overrides = data.get("config", {})
            config = Config(work_dir=tempfile.gettempdir())
            for key, value in config_overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            from pipeline.frame_reformatter import compute_canvas_layout
            canvas_layout = compute_canvas_layout(config)
            
            preview_filename = os.path.basename(cached_preview_path)
            return jsonify({
                "preview_image_url": f"/output/{preview_filename}",
                "canvas_layout": {
                    "canvas_width": canvas_layout.canvas_width,
                    "canvas_height": canvas_layout.canvas_height,
                    "facecam_x": canvas_layout.facecam_x,
                    "facecam_y": canvas_layout.facecam_y,
                    "facecam_width": canvas_layout.facecam_width,
                    "facecam_height": canvas_layout.facecam_height,
                    "gameplay_x": canvas_layout.gameplay_x,
                    "gameplay_y": canvas_layout.gameplay_y,
                    "gameplay_width": canvas_layout.gameplay_width,
                    "gameplay_height": canvas_layout.gameplay_height,
                },
                "error": None,
                "cached": True
            }), 200
        
        # Cache miss - generate preview
        _logger = logging.getLogger(__name__)
        _logger.info(f"Preview cache MISS for {cache_key} - generating new preview")
        
        # Build config (use defaults, allow overrides)
        config_overrides = data.get("config", {})
        config = Config(work_dir=tempfile.gettempdir())
        
        # Apply config overrides if provided
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Compute canvas layout
        from pipeline.frame_reformatter import compute_canvas_layout, FrameReformatter
        canvas_layout = compute_canvas_layout(config)
        
        # Reconstruct FacecamRegion from data
        from pipeline.models import FacecamRegion
        facecam_region = FacecamRegion(
            x=facecam_region_data["x"],
            y=facecam_region_data["y"],
            width=facecam_region_data["width"],
            height=facecam_region_data["height"],
            corner=facecam_region_data["corner"],
            confidence=facecam_region_data["confidence"],
        )
        
        # Generate preview image using FFmpeg
        # We'll extract a single frame from the clip and apply the vertical formatting
        import subprocess
        
        # Create output path for preview image (use lower resolution for performance)
        preview_width = 540  # Half of 1080 for faster generation
        preview_height = 960  # Half of 1920
        
        # Generate unique filename for preview
        preview_filename = f"preview_{uuid.uuid4().hex[:8]}.jpg"
        preview_path = OUTPUT_DIR / preview_filename
        
        # Build FFmpeg filter for preview
        # For vertical shorts, we want to CROP the gameplay region to 9:16, not letterbox it
        # 1. Crop the center of the source video to 9:16 aspect ratio for gameplay
        # 2. Scale it to fit the gameplay region height
        # 3. Pad to full canvas size with black (gameplay at bottom)
        # 4. Crop and scale facecam from source
        # 5. Overlay facecam on top
        
        reformatter = FrameReformatter()
        
        # Calculate crop dimensions for 9:16 gameplay region
        # Target aspect ratio for gameplay: 9:16
        gameplay_target_w = canvas_layout.gameplay_width
        gameplay_target_h = canvas_layout.gameplay_height
        gameplay_aspect = gameplay_target_w / gameplay_target_h  # 9/16 = 0.5625
        
        # Source aspect ratio
        src_aspect = frame_width / frame_height
        
        # Crop source to match gameplay aspect ratio (9:16)
        if src_aspect > gameplay_aspect:
            # Source is wider - crop width
            crop_h = frame_height
            crop_w = round(frame_height * gameplay_aspect)
            crop_x = (frame_width - crop_w) // 2  # Center horizontally
            crop_y = 0
        else:
            # Source is taller - crop height
            crop_w = frame_width
            crop_h = round(frame_width / gameplay_aspect)
            crop_x = 0
            crop_y = (frame_height - crop_h) // 2  # Center vertically
        
        # Build gameplay filter: crop to 9:16, scale to gameplay region, pad to canvas
        gameplay_filter = (
            f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y},"
            f"scale={gameplay_target_w}:{gameplay_target_h},"
            f"pad={canvas_layout.canvas_width}:{canvas_layout.canvas_height}:0:{canvas_layout.gameplay_y}:black"
        )
        
        # Build facecam crop and scale filter
        # Crop facecam from source, scale to fit facecam region on canvas
        facecam_crop = f"crop={facecam_region.width}:{facecam_region.height}:{facecam_region.x}:{facecam_region.y}"
        facecam_scale = f"scale={canvas_layout.facecam_width}:{canvas_layout.facecam_height}"
        
        _logger = logging.getLogger(__name__)
        _logger.info(f"Gameplay crop: {crop_w}x{crop_h} at ({crop_x},{crop_y}) from {frame_width}x{frame_height}")
        _logger.info(f"Gameplay filter: {gameplay_filter}")
        _logger.info(f"Facecam crop: {facecam_crop}, scale: {facecam_scale}")
        
        # Complete filter chain:
        # [0:v] -> split into two streams
        # Stream 1: crop and build canvas with gameplay in bottom region -> [canvas]
        # Stream 2: crop and scale facecam -> [facecam]
        # [canvas][facecam] -> overlay facecam on top -> scale to preview size -> output
        filter_complex = (
            f"[0:v]split=2[v1][v2];"
            f"[v1]{gameplay_filter}[canvas];"
            f"[v2]{facecam_crop},{facecam_scale}[facecam];"
            f"[canvas][facecam]overlay={canvas_layout.facecam_x}:{canvas_layout.facecam_y},"
            f"scale={preview_width}:{preview_height}[out]"
        )
        
        # Debug logging
        _logger = logging.getLogger(__name__)
        _logger.info(f"Preview filter_complex: {filter_complex}")
        _logger.info(f"Canvas layout: facecam=({canvas_layout.facecam_x},{canvas_layout.facecam_y},{canvas_layout.facecam_width}x{canvas_layout.facecam_height}), gameplay=({canvas_layout.gameplay_x},{canvas_layout.gameplay_y},{canvas_layout.gameplay_width}x{canvas_layout.gameplay_height})")
        
        # Run FFmpeg to generate preview image
        ffmpeg_cmd = [
            "ffmpeg",
            "-ss", "1",  # Seek to 1 second
            "-i", clip_path,
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-frames:v", "1",  # Extract single frame
            "-q:v", "2",  # High quality JPEG
            "-y",  # Overwrite output
            str(preview_path),
        ]
        
        result = subprocess.run(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        
        if result.returncode != 0:
            _logger = logging.getLogger(__name__)
            _logger.error(f"FFmpeg preview generation failed: {result.stderr}")
            return jsonify({
                "preview_image_url": None,
                "canvas_layout": None,
                "error": f"Failed to generate preview: {result.stderr[:200]}",
                "cached": False
            }), 500
        
        # Store in cache
        with _preview_cache_lock:
            _preview_cache[cache_key] = str(preview_path)
        
        # Return result
        return jsonify({
            "preview_image_url": f"/output/{preview_filename}",
            "canvas_layout": {
                "canvas_width": canvas_layout.canvas_width,
                "canvas_height": canvas_layout.canvas_height,
                "facecam_x": canvas_layout.facecam_x,
                "facecam_y": canvas_layout.facecam_y,
                "facecam_width": canvas_layout.facecam_width,
                "facecam_height": canvas_layout.facecam_height,
                "gameplay_x": canvas_layout.gameplay_x,
                "gameplay_y": canvas_layout.gameplay_y,
                "gameplay_width": canvas_layout.gameplay_width,
                "gameplay_height": canvas_layout.gameplay_height,
            },
            "error": None,
            "cached": False
        }), 200
        
    except Exception as exc:
        _logger = logging.getLogger(__name__)
        _logger.exception("Error in generate_preview_endpoint")
        return jsonify({
            "preview_image_url": None,
            "canvas_layout": None,
            "error": f"Internal error: {str(exc)}",
            "cached": False
        }), 500


@app.route("/api/mini-editor/confirm", methods=["POST"])
def confirm_placement_endpoint():
    """Confirm facecam placement and create a VerticalFormattingJob.

    Request JSON:
        {
            "session_id": str,          # Active editor session ID
            "facecam_region": {         # Confirmed facecam placement
                "x": int,
                "y": int,
                "width": int,
                "height": int,
                "corner": str,
                "confidence": float
            },
            "settings": dict (optional) # User settings (backup, naming, etc.)
        }

    Response JSON:
        {
            "job_id": str,              # UUID for the formatting job
            "status": str,              # "queued"
            "error": str | null
        }
    """
    try:
        data = request.get_json()
    except Exception:
        return jsonify({"error": "Request body must be JSON"}), 400

    try:
        if not data:
            return jsonify({"error": "Request body must be JSON"}), 400

        session_id = data.get("session_id")
        facecam_region_data = data.get("facecam_region")
        settings = data.get("settings", {})

        # Validate required fields
        if not session_id:
            return jsonify({"error": "session_id is required"}), 400
        if not facecam_region_data:
            return jsonify({"error": "facecam_region is required"}), 400

        # Validate facecam_region structure
        required_region_fields = ["x", "y", "width", "height", "corner", "confidence"]
        for fld in required_region_fields:
            if fld not in facecam_region_data:
                return jsonify({"error": f"facecam_region.{fld} is required"}), 400

        # Look up session
        session = _session_store.get_session(session_id)
        if session is None:
            return jsonify({"error": f"Session not found or expired: {session_id}"}), 404

        # Reconstruct FacecamRegion
        from pipeline.models import FacecamRegion as _FacecamRegion
        try:
            facecam_region = _FacecamRegion(
                x=int(facecam_region_data["x"]),
                y=int(facecam_region_data["y"]),
                width=int(facecam_region_data["width"]),
                height=int(facecam_region_data["height"]),
                corner=str(facecam_region_data["corner"]),
                confidence=float(facecam_region_data["confidence"]),
            )
        except (ValueError, TypeError) as exc:
            return jsonify({"error": f"Invalid facecam_region values: {exc}"}), 400

        # Validate facecam_region bounds against reference resolution
        frame_width, frame_height = session.reference_resolution

        # Auto-clamp region to frame bounds instead of rejecting
        # This allows users to confirm whatever they see in the preview
        _logger = logging.getLogger(__name__)
        
        # Clamp negative positions to 0
        if facecam_region.x < 0:
            _logger.warning(f"Clamping facecam x from {facecam_region.x} to 0")
            facecam_region.x = 0
        if facecam_region.y < 0:
            _logger.warning(f"Clamping facecam y from {facecam_region.y} to 0")
            facecam_region.y = 0
        
        # Ensure width and height are positive
        if facecam_region.width <= 0:
            return jsonify({"error": "facecam_region.width must be > 0"}), 400
        if facecam_region.height <= 0:
            return jsonify({"error": "facecam_region.height must be > 0"}), 400
        
        # Clamp region to stay within frame bounds
        if facecam_region.x + facecam_region.width > frame_width:
            old_width = facecam_region.width
            facecam_region.width = frame_width - facecam_region.x
            _logger.warning(
                f"Clamping facecam width from {old_width} to {facecam_region.width} "
                f"to fit within frame (x={facecam_region.x}, frame_width={frame_width})"
            )
        
        if facecam_region.y + facecam_region.height > frame_height:
            old_height = facecam_region.height
            facecam_region.height = frame_height - facecam_region.y
            _logger.warning(
                f"Clamping facecam height from {old_height} to {facecam_region.height} "
                f"to fit within frame (y={facecam_region.y}, frame_height={frame_height})"
            )

        # Validate area fraction (must be 4%–30% of frame area)
        # Relax this to just a warning instead of rejection
        frame_area = frame_width * frame_height
        region_area = facecam_region.width * facecam_region.height
        area_fraction = region_area / frame_area

        config = Config(work_dir=tempfile.gettempdir())
        min_fraction = config.facecam_min_area_fraction  # 0.04
        max_fraction = config.facecam_max_area_fraction  # 0.30

        if area_fraction < min_fraction:
            _logger.warning(
                f"Facecam area fraction ({area_fraction:.3f}) is below minimum ({min_fraction}), "
                f"but allowing it anyway"
            )
        if area_fraction > max_fraction:
            _logger.warning(
                f"Facecam area fraction ({area_fraction:.3f}) exceeds maximum ({max_fraction}), "
                f"but allowing it anyway"
            )

        # Get the job to retrieve clips
        job = _get_job(session.clip_batch_id)
        if job is None:
            return jsonify({"error": f"Job not found: {session.clip_batch_id}"}), 404

        # Build clips list for the formatting job
        clips = []
        for clip_data in job.result_clips:
            clips.append({
                "path": clip_data["path"],
                "name": clip_data["name"],
                "resolution": [frame_width, frame_height],
                "start": clip_data.get("start", 0.0),  # Clip start time for subtitle generation
                "end": clip_data.get("end", 0.0),      # Clip end time for subtitle generation
            })

        # Create VerticalFormattingJob
        from pipeline.models import VerticalFormattingJob as _VerticalFormattingJob
        job_id = str(uuid.uuid4())
        formatting_job = _VerticalFormattingJob(
            job_id=job_id,
            session_id=session_id,
            clip_batch_id=session.clip_batch_id,
            facecam_region=facecam_region,
            canvas_layout=session.canvas_layout,
            settings=settings or {},
            clips=clips,
            output_dir=str(OUTPUT_DIR),
            status="queued",
        )

        # Update session with confirmed region and clear undo/redo history
        session.facecam_region = facecam_region
        session.clear_history()
        session.refresh_expiry()

        # Store the formatting job in the module-level dict for progress polling
        with _formatting_jobs_lock:
            _formatting_jobs[job_id] = formatting_job

        # Enqueue the job for background processing
        _formatting_job_queue.put(job_id)

        return jsonify({
            "job_id": job_id,
            "status": "queued",
            "error": None,
        }), 201

    except Exception as exc:
        _logger = logging.getLogger(__name__)
        _logger.exception("Error in confirm_placement_endpoint")
        return jsonify({
            "job_id": None,
            "status": None,
            "error": f"Internal error: {str(exc)}",
        }), 500


@app.route("/api/mini-editor/cancel", methods=["POST"])
def cancel_editor_session_endpoint():
    """Cancel an editor session without processing any clips.

    Request JSON:
        {
            "session_id": str           # Active editor session ID
        }

    Response JSON:
        {
            "status": str,              # "cancelled"
            "error": str | null
        }
    """
    try:
        data = request.get_json()
    except Exception:
        return jsonify({"error": "Request body must be JSON"}), 400

    try:
        if data is None:
            return jsonify({"error": "Request body must be JSON"}), 400

        session_id = data.get("session_id")

        if not session_id:
            return jsonify({"error": "session_id is required"}), 400

        # Look up session (allow expired sessions to be "cancelled" gracefully)
        session = _session_store.get_session(session_id)
        if session is None:
            # Session may have already expired — treat as already cancelled
            return jsonify({
                "status": "cancelled",
                "error": None,
            }), 200

        # Clear undo/redo history and delete the session
        session.clear_history()
        _session_store.delete_session(session_id)

        return jsonify({
            "status": "cancelled",
            "error": None,
        }), 200

    except Exception as exc:
        _logger = logging.getLogger(__name__)
        _logger.exception("Error in cancel_editor_session_endpoint")
        return jsonify({
            "status": None,
            "error": f"Internal error: {str(exc)}",
        }), 500


@app.route("/api/mini-editor/job/<job_id>/cancel", methods=["GET"])
def cancel_formatting_job_endpoint(job_id: str):
    """Cancel a running VerticalFormattingJob.

    Sets the job status to "cancelled" so that the processing loop stops
    before the next clip.  Already-processed clips are preserved.

    URL parameter:
        job_id: UUID of the VerticalFormattingJob to cancel.

    Response JSON:
        {
            "job_id": str,
            "status": str,      # "cancelled" or current status
            "error": str | null
        }
    """
    try:
        with _formatting_jobs_lock:
            formatting_job = _formatting_jobs.get(job_id)

        if formatting_job is None:
            return jsonify({
                "job_id": job_id,
                "status": None,
                "error": f"Formatting job not found: {job_id}",
            }), 404

        # Mark as cancelled — the processing loop checks this flag
        if formatting_job.status in ("queued", "running"):
            formatting_job.status = "cancelled"

        return jsonify({
            "job_id": job_id,
            "status": formatting_job.status,
            "error": None,
        }), 200

    except Exception as exc:
        _logger = logging.getLogger(__name__)
        _logger.exception("Error in cancel_formatting_job_endpoint")
        return jsonify({
            "job_id": job_id,
            "status": None,
            "error": f"Internal error: {str(exc)}",
        }), 500


@app.route("/api/mini-editor/job/<job_id>/progress", methods=["GET"])
def get_formatting_job_progress(job_id: str):
    """Poll progress of a VerticalFormattingJob.

    Response JSON:
        {
            "job_id": str,
            "status": str,
            "clips_processed": int,
            "clips_total": int,
            "current_clip": str,
            "progress_pct": float,
            "eta_seconds": float,
            "elapsed_seconds": float,
            "errors": list[str],
            "error": str | null
        }
    """
    try:
        with _formatting_jobs_lock:
            formatting_job = _formatting_jobs.get(job_id)

        if formatting_job is None:
            return jsonify({
                "job_id": job_id,
                "status": None,
                "error": f"Formatting job not found: {job_id}",
            }), 404

        return jsonify({
            "job_id": job_id,
            "status": formatting_job.status,
            "clips_processed": formatting_job.clips_processed,
            "clips_total": formatting_job.clips_total,
            "current_clip": formatting_job.current_clip,
            "progress_pct": formatting_job.get_progress_percentage(),
            "eta_seconds": formatting_job.estimate_remaining_time(),
            "elapsed_seconds": formatting_job.get_elapsed_time(),
            "errors": formatting_job.errors,
            "error": None,
        }), 200

    except Exception as exc:
        _logger = logging.getLogger(__name__)
        _logger.exception("Error in get_formatting_job_progress")
        return jsonify({
            "job_id": job_id,
            "status": None,
            "error": f"Internal error: {str(exc)}",
        }), 500


@app.route("/api/mini-editor/jobs", methods=["GET"])
def list_formatting_jobs():
    """Return all vertical formatting jobs ordered by creation time.
    
    Response JSON:
        [
            {
                "job_id": str,
                "status": str,
                "clips_processed": int,
                "clips_total": int,
                "progress_pct": float,
                "created_at": float,
                "elapsed_seconds": float,
                "eta_seconds": float,
                "type": str,
                "name": str
            },
            ...
        ]
    """
    try:
        with _formatting_jobs_lock:
            jobs_snapshot = list(_formatting_jobs.values())
        
        jobs_snapshot.sort(key=lambda j: j.created_at, reverse=True)
        
        return jsonify([{
            "job_id": j.job_id,
            "status": j.status,
            "clips_processed": j.clips_processed,
            "clips_total": j.clips_total,
            "progress_pct": j.get_progress_percentage(),
            "created_at": j.created_at,
            "elapsed_seconds": j.get_elapsed_time(),
            "eta_seconds": j.estimate_remaining_time(),
            "type": "formatting",
            "name": f"Vertical Formatting ({j.clips_total} clips)",
        } for j in jobs_snapshot]), 200
        
    except Exception as exc:
        _logger = logging.getLogger(__name__)
        _logger.exception("Error in list_formatting_jobs")
        return jsonify({"error": f"Internal error: {str(exc)}"}), 500


@app.route("/api/mini-editor/undo", methods=["POST"])
def undo_adjustment_endpoint():
    """Undo the last facecam region adjustment.

    Pops from undo_history, pushes current region to redo_history,
    and returns the previous facecam_region.

    Request JSON:
        {
            "session_id": str           # Active editor session ID
        }

    Response JSON:
        {
            "facecam_region": {         # Restored previous placement (or null if nothing to undo)
                "x": int,
                "y": int,
                "width": int,
                "height": int,
                "corner": str,
                "confidence": float
            } | null,
            "can_undo": bool,           # True if more undo steps are available
            "can_redo": bool,           # True if redo is available
            "error": str | null
        }
    """
    try:
        data = request.get_json()
    except Exception:
        return jsonify({"error": "Request body must be JSON"}), 400

    try:
        if data is None:
            return jsonify({"error": "Request body must be JSON"}), 400

        session_id = data.get("session_id")

        if not session_id:
            return jsonify({"error": "session_id is required"}), 400

        session = _session_store.get_session(session_id)
        if session is None:
            return jsonify({"error": f"Session not found or expired: {session_id}"}), 404

        # Pop from undo history
        previous_region = session.pop_undo()
        if previous_region is None:
            return jsonify({
                "facecam_region": None,
                "can_undo": False,
                "can_redo": len(session.redo_history) > 0,
                "error": "Nothing to undo",
            }), 400

        # Push current region to redo history
        session.push_redo(session.facecam_region)

        # Restore previous region
        session.facecam_region = previous_region
        session.refresh_expiry()

        return jsonify({
            "facecam_region": {
                "x": previous_region.x,
                "y": previous_region.y,
                "width": previous_region.width,
                "height": previous_region.height,
                "corner": previous_region.corner,
                "confidence": previous_region.confidence,
            },
            "can_undo": len(session.undo_history) > 0,
            "can_redo": len(session.redo_history) > 0,
            "error": None,
        }), 200

    except Exception as exc:
        _logger = logging.getLogger(__name__)
        _logger.exception("Error in undo_adjustment_endpoint")
        return jsonify({
            "facecam_region": None,
            "can_undo": False,
            "can_redo": False,
            "error": f"Internal error: {str(exc)}",
        }), 500


@app.route("/api/mini-editor/redo", methods=["POST"])
def redo_adjustment_endpoint():
    """Redo the last undone facecam region adjustment.

    Pops from redo_history, pushes current region to undo_history,
    and returns the reapplied facecam_region.

    Request JSON:
        {
            "session_id": str           # Active editor session ID
        }

    Response JSON:
        {
            "facecam_region": {         # Reapplied placement (or null if nothing to redo)
                "x": int,
                "y": int,
                "width": int,
                "height": int,
                "corner": str,
                "confidence": float
            } | null,
            "can_undo": bool,           # True if undo is available
            "can_redo": bool,           # True if more redo steps are available
            "error": str | null
        }
    """
    try:
        data = request.get_json()
    except Exception:
        return jsonify({"error": "Request body must be JSON"}), 400

    try:
        if data is None:
            return jsonify({"error": "Request body must be JSON"}), 400

        session_id = data.get("session_id")

        if not session_id:
            return jsonify({"error": "session_id is required"}), 400

        session = _session_store.get_session(session_id)
        if session is None:
            return jsonify({"error": f"Session not found or expired: {session_id}"}), 404

        # Pop from redo history
        next_region = session.pop_redo()
        if next_region is None:
            return jsonify({
                "facecam_region": None,
                "can_undo": len(session.undo_history) > 0,
                "can_redo": False,
                "error": "Nothing to redo",
            }), 400

        # Push current region to undo history (without clearing redo — we're redoing)
        session.undo_history.append(session.facecam_region)

        # Restore the redo region
        session.facecam_region = next_region
        session.refresh_expiry()

        return jsonify({
            "facecam_region": {
                "x": next_region.x,
                "y": next_region.y,
                "width": next_region.width,
                "height": next_region.height,
                "corner": next_region.corner,
                "confidence": next_region.confidence,
            },
            "can_undo": len(session.undo_history) > 0,
            "can_redo": len(session.redo_history) > 0,
            "error": None,
        }), 200

    except Exception as exc:
        _logger = logging.getLogger(__name__)
        _logger.exception("Error in redo_adjustment_endpoint")
        return jsonify({
            "facecam_region": None,
            "can_undo": False,
            "can_redo": False,
            "error": f"Internal error: {str(exc)}",
        }), 500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_timestamp_range(start: float, end: float) -> str:
    """Format a start/end time pair as 'M:SS – M:SS'."""
    def fmt(t: float) -> str:
        m = int(t) // 60
        s = int(t) % 60
        return f"{m}:{s:02d}"
    return f"{fmt(start)} – {fmt(end)}"


def _job_summary(job: Job) -> dict:
    return {
        "job_id": job.job_id,
        "status": job.status,
        "video_name": os.path.basename(job.video_path),
        "created_at": job.created_at,
        "clip_count": len(job.result_clips),
        "error": job.error,
    }


def _job_detail(job: Job) -> dict:
    clips = []
    for i, c in enumerate(job.result_clips):
        thumbnail_name = c.get("thumbnail_name")
        clips.append({
            "index": i,
            "name": c["name"],
            "path": c["path"],  # Add actual file path for mini-editor
            "download_url": f"/api/jobs/{job.job_id}/clips/{i}/download",
            "why_chosen": c.get("why_chosen", ""),
            "timestamp_range": c.get("timestamp_range", ""),
            "duration": c.get("duration", ""),
            "score": c.get("score", ""),
            "thumbnail_url": f"/output/{thumbnail_name}" if thumbnail_name else None,
        })
    # Signal to the frontend that vertical formatting is available
    format_to_vertical_available = (
        job.status == JobStatus.DONE and len(job.result_clips) > 0
    )
    return {
        **_job_summary(job),
        "log_lines": job.log_lines,
        "clips": clips,
        "config_options": job.job_config_options,
        "format_to_vertical_available": format_to_vertical_available,
    }


def _safe_filename(filename: str) -> str:
    """Sanitize an uploaded filename, preserving extension."""
    from pathlib import PurePosixPath
    stem = PurePosixPath(filename).stem
    suffix = PurePosixPath(filename).suffix or ".mp4"
    # Keep only alphanumeric, dash, underscore, dot
    safe_stem = "".join(c if c.isalnum() or c in "-_." else "_" for c in stem)
    safe_stem = safe_stem[:80] or "video"
    # Prepend timestamp to avoid collisions
    ts = int(time.time())
    return f"{ts}_{safe_stem}{suffix}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    global UPLOADS_DIR, OUTPUT_DIR, CLEANUP_UPLOADS

    parser = argparse.ArgumentParser(description="Video Highlight Generator Web Server")
    parser.add_argument("--uploads-dir", default=str(UPLOADS_DIR),
                        help="Directory for uploaded videos (default: ./uploads)")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR),
                        help="Default output directory for clips (default: ./output)")
    parser.add_argument("--port", type=int, default=6800,
                        help="Port to listen on (default: 6800)")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--no-cleanup-uploads", action="store_true",
                        help="Keep uploaded videos after job completion (default: delete them)")
    args = parser.parse_args()

    UPLOADS_DIR = Path(args.uploads_dir)
    OUTPUT_DIR = Path(args.output_dir)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.no_cleanup_uploads:
        CLEANUP_UPLOADS = False

    print(f"Video Highlight Generator Web UI")
    print(f"  Listening on: http://{args.host}:{args.port}")
    print(f"  Uploads dir:  {UPLOADS_DIR}")
    print(f"  Output dir:   {OUTPUT_DIR}")
    print()

    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
