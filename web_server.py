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
from pipeline.audio_extractor import extract_audio
from pipeline.clip_extractor import extract_clips, generate_thumbnail
from pipeline.clip_selector import select_clips
from pipeline.exceptions import PipelineError
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

    try:
        # Stage 1: Audio extraction (0% → 5%)
        if job.cancel_event.is_set():
            raise _CancelledError()
        job.add_progress(1, 7, "Audio Extraction", 5)
        log("[AudioExtractor] Starting...")
        t0 = time.time()
        wav_path = extract_audio(job.config, job.video_path)
        log(f"[AudioExtractor] Done in {time.time() - t0:.1f}s")

        # Stage 2: Transcription (5% → 60%)
        if job.cancel_event.is_set():
            raise _CancelledError()
        job.add_progress(2, 7, "Transcription", 10)
        log("[Transcriber] Starting...")
        t0 = time.time()
        
        # Define progress callback for transcription
        def transcription_progress(percentage: int) -> None:
            job.add_progress(2, 7, "Transcription", percentage)
        
        transcript = transcribe(job.config, wav_path, progress_callback=transcription_progress)
        log(f"[Transcriber] Done in {time.time() - t0:.1f}s — {len(transcript.segments)} segment(s)")
        job.add_progress(2, 7, "Transcription", 60)

        if not transcript.segments:
            log("[Transcriber] Warning: No speech detected — scoring on audio energy only")

        # Stage 3: Scoring (60% → 70%)
        if job.cancel_event.is_set():
            raise _CancelledError()
        job.add_progress(3, 7, "Scoring Segments", 65)
        log("[Scorer] Starting...")
        t0 = time.time()

        def llm_progress_callback(current: int, total: int) -> None:
            job.add_llm_progress(current, total)

        scored_segments = score_segments(job.config, transcript, wav_path,
                                         llm_progress_callback=llm_progress_callback)
        log(f"[Scorer] Done in {time.time() - t0:.1f}s — {len(scored_segments)} segment(s) scored")
        job.add_progress(3, 7, "Scoring Segments", 70)

        if not scored_segments:
            raise PipelineError("No segments to score. The video may have no audio content.")

        # Stage 4: Clip selection (70% → 75%)
        if job.cancel_event.is_set():
            raise _CancelledError()
        job.add_progress(4, 7, "Selecting Clips", 72)
        log("[ClipSelector] Starting...")
        t0 = time.time()
        video_duration = _get_video_duration(job.video_path)
        clips = select_clips(job.config, scored_segments, transcript, video_duration)
        log(f"[ClipSelector] Done in {time.time() - t0:.1f}s — {len(clips)} clip(s) selected")
        job.add_progress(4, 7, "Selecting Clips", 75)

        if not clips:
            raise PipelineError("No clips selected. Try lowering Top N or check the video.")

        # Stage 5: Clip extraction (75% → 85%)
        if job.cancel_event.is_set():
            raise _CancelledError()
        job.add_progress(5, 7, "Extracting Clips", 77)
        log("[ClipExtractor] Starting...")
        t0 = time.time()
        clip_paths = extract_clips(job.config, clips, job.video_path)
        log(f"[ClipExtractor] Done in {time.time() - t0:.1f}s")
        job.add_progress(5, 7, "Extracting Clips", 85)

        # Stage 6: Subtitle generation (85% → 95%)
        job.add_progress(6, 7, "Generating Subtitles", 87)
        log("[SubtitleGenerator] Starting...")
        t0 = time.time()
        final_paths = generate_subtitles(job.config, clips, transcript, clip_paths)
        log(f"[SubtitleGenerator] Done in {time.time() - t0:.1f}s")
        job.add_progress(6, 7, "Generating Subtitles", 95)

        # Stage 7: Why-chosen reports (95% → 100%)
        job.add_progress(7, 7, "Generating Reports", 97)
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
            })
        log(f"[ReportGenerator] Done in {time.time() - t0:.1f}s — {len(result_clips)} report(s)")
        job.add_progress(7, 7, "Complete", 100)

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
    """Accept a video upload and enqueue a new pipeline job."""
    reuse_video_path = request.form.get("reuse_video_path", "").strip()

    if reuse_video_path:
        # Re-run path: use an existing server-side file instead of uploading
        if not os.path.exists(reuse_video_path):
            return jsonify({"error": "Original video file no longer exists on the server"}), 400
        upload_path = Path(reuse_video_path)
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

    # Build config
    work_dir = tempfile.mkdtemp(prefix="highlight_web_")
    cfg = Config(work_dir=work_dir)
    cfg.output_dir = output_dir
    cfg.whisper_model = whisper_model
    cfg.top_n_clips = top_n
    cfg.llm_enabled = llm_enabled
    cfg.llm_model = llm_model
    cfg.burn_subtitles = burn_subtitles
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
        "genre": genre,
        "platform": platform,
        "language": language,
        "original_video_path": str(upload_path),
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
    """Return a list of locally available Ollama models."""
    import subprocess
    try:
        result = subprocess.run(
            ["ollama", "list"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return jsonify({"error": "Ollama is not running or not installed", "models": []}), 200

        # Parse output: skip header line, extract model names from first column
        lines = result.stdout.strip().split("\n")
        models = []
        for line in lines[1:]:  # skip header
            if line.strip():
                # Model name is the first column (e.g., "llama3:latest" or "mistral")
                parts = line.split()
                if parts:
                    model_name = parts[0]
                    # Strip tag suffix (e.g. :latest, :8b, :7b-instruct) for cleaner display
                    if ":" in model_name:
                        model_name = model_name.split(":")[0]
                    models.append(model_name)

        return jsonify({"models": models, "error": None})

    except FileNotFoundError:
        return jsonify({"error": "Ollama command not found. Install from https://ollama.ai", "models": []}), 200
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Ollama command timed out", "models": []}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to list models: {e}", "models": []}), 200


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
            "download_url": f"/api/jobs/{job.job_id}/clips/{i}/download",
            "why_chosen": c.get("why_chosen", ""),
            "timestamp_range": c.get("timestamp_range", ""),
            "duration": c.get("duration", ""),
            "score": c.get("score", ""),
            "thumbnail_url": f"/output/{thumbnail_name}" if thumbnail_name else None,
        })
    return {
        **_job_summary(job),
        "log_lines": job.log_lines,
        "clips": clips,
        "config_options": job.job_config_options,
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
