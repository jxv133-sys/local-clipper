# Implementation Plan: Video Highlight Generator

## Overview

Implement a local, offline Python pipeline that extracts audio from a video, transcribes it with Whisper, scores segments, selects highlights, extracts clips, and burns in subtitles. Each stage is a separate module under `pipeline/`. The implementation follows the staged architecture defined in the design document.

## Tasks

- [x] 1. Set up project structure, data models, and configuration
  - _Requirements: 11.1, 11.2, 11.3, 11.4_

- [x] 2. Implement Audio Extractor (`pipeline/audio_extractor.py`)
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6_

- [x] 3. Implement Transcriber (`pipeline/transcriber.py`)
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

- [x] 4. Implement Scorer — text and combination logic (`pipeline/scorer.py`)
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 6.1, 6.2, 6.4, 6.5_

- [x] 5. Implement Scorer — audio score (`pipeline/scorer.py`, continued)
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [x] 6. Implement Scorer — LLM score and full `score_segments` assembly (`pipeline/scorer.py`, continued)
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 6.1, 6.2, 6.3_

- [x] 7. Checkpoint — All scorer and transcriber tests pass (128 tests)

- [x] 8. Implement Clip Selector (`pipeline/clip_selector.py`)
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8_

- [x] 9. Implement Clip Extractor (`pipeline/clip_extractor.py`)
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7_

- [x] 10. Implement Subtitle Generator (`pipeline/subtitle_generator.py`)
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.7_

- [x] 11. Checkpoint — All pipeline tests pass

- [x] 12. Implement Pipeline Orchestrator (`main.py`)
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 11.1, 11.2, 11.3, 11.4_

- [x] 13. Add why-chosen report per clip (`pipeline/report_generator.py`)
  - Generates a `.txt` file alongside each clip explaining scores, keywords, energy, and transcript

- [x] 14. Add tkinter GUI (`gui.py`)
  - File picker, options panel, live progress log, results panel with open/why-chosen buttons

- [x] 15. Verify subtitle burn works end-to-end on real footage

- [x] 16. Clean up stray files and fix SSL workaround

- [x] 17. Write README.md with setup and usage instructions

- [x] 18. Final end-to-end test on real footage

- [x] 19. Replace tkinter GUI with web UI hosted at localhost:6800 (`web/`)
  - Replace `gui.py` with a Flask/FastAPI web server
  - Frontend: single HTML page with vanilla JS (no build step required)
  - Features:
    - Video file upload (drag-and-drop + browse button)
    - Options panel: Whisper model, top N, keywords, LLM toggle + model name, output dir
    - Live progress log streamed via Server-Sent Events (SSE) — one line per stage event
    - Results panel: list of completed clips with download links and why-chosen text inline
    - Job queue: support multiple jobs, show status (queued / running / done / failed)
  - Server runs on `0.0.0.0:6800` so it's accessible from other machines on the network
  - Uploaded videos saved to a configurable `uploads/` directory on the server
  - Completed clips served as static files for download
  - Keep `main.py` CLI working unchanged — web server calls the same pipeline functions

- [x] 20. Add verbose logging throughout the pipeline
  - Add a `logging` call at the start and end of every pipeline stage with timing
  - Log segment count after transcription, score distribution after scoring (min/max/mean)
  - Log clip timestamps and scores after clip selection
  - Log file sizes of extracted clips after extraction
  - Log subtitle entry count per clip after subtitle generation
  - All log messages use Python's `logging` module at `INFO` level
  - Web UI streams these log lines in real time via SSE
  - CLI prints them to stdout (already partially done — extend coverage)

- [x] 21. Write Docker Compose deployment for headless Ubuntu server (`docker-compose.yml`)
  - Single `docker-compose.yml` at project root with two services:
    - `app`: the web UI + pipeline (Python, FFmpeg, faster-whisper)
    - `ollama`: the Ollama LLM server (optional, can be disabled)
  - `app` service:
    - Base image: `python:3.11-slim` with FFmpeg installed via apt
    - Mounts `./uploads` and `./output` as volumes so files persist across restarts
    - Exposes port `6800`
    - Sets `OLLAMA_HOST=http://ollama:11434` env var
  - `ollama` service:
    - Image: `ollama/ollama:latest`
    - Mounts `./ollama_models` volume for model persistence
    - Exposes port `11434` internally
  - `Dockerfile` for the app service:
    - Install system deps: `ffmpeg`, `libgomp1`, `fonts-liberation` (for Pillow text)
    - Copy requirements and install Python deps
    - Copy source code
    - `CMD ["python3", "web_server.py"]`
  - `.dockerignore` to exclude test footage, output, venv, `__pycache__`

- [x] 22. Update README.md with Docker deployment and Ollama setup guide
  - Add "Server Deployment" section with step-by-step Docker Compose instructions:
    - `git clone`, `docker compose up -d`, open `http://<server-ip>:6800`
  - Add detailed Ollama setup section:
    - Installing Ollama on Ubuntu (curl install script)
    - Pulling models: `ollama pull llama3`, `ollama pull llama3.2:1b`
    - Running as a systemd service so it starts on boot
    - Connecting the pipeline to a remote Ollama instance via `--llm-endpoint`
    - Model size guide: which model to use based on available RAM
  - Add "Accessing from another machine" section explaining the network setup
  - Add troubleshooting section: common errors and fixes

## Notes

- Tasks 1–22 are complete
- Tasks 23–42 are improvements identified from end-to-end testing on real footage
- Priority order: scoring quality (23–27) → clip selection (28–30) → output (31–35) → web UI (36–39) → infrastructure (40–42)
- The tkinter `gui.py` can be kept for local macOS use but is no longer the primary UI
- The LLM scoring path is optional at runtime via `config.llm_enabled`
- All pushes go to `kiro/main` (plain `git push` works)

---

## Improvement Tasks (from end-to-end test analysis)

### Scoring Quality

- [x] 23. Recalibrate text score normalization to use the full 0–1 range
  - Current sigmoid is too aggressive — max observed clip score was 0.51 on real footage
  - Tune the sigmoid divisor or switch to min-max normalization across all segments
  - Add a score distribution assertion in tests: mean score should be > 0.2 on typical input

- [x] 24. Add minimum text score threshold to prevent audio-only clips
  - Clips with text_score < 0.05 and no keywords should be penalized or filtered
  - Add a `min_text_score_for_selection: float = 0.05` config field
  - Clip selector should skip segments below this threshold unless no better candidates exist

- [x] 25. Add speech density (pace) signal to text scoring
  - Compute words-per-second for each segment
  - Fast speech (> 3 words/sec) gets a bonus — indicates excitement or urgency
  - Add as a fourth component to `compute_text_score`, normalized to [0.0, 1.0]

- [x] 26. Log VAD-removed time ranges during transcription
  - faster-whisper's VAD filter silently drops audio — log which time ranges were removed
  - Format: `[Transcriber] VAD removed 32:24 of audio across N silent sections`
  - Helps users diagnose over-aggressive VAD on music/ambient content

### Clip Selection

- [x] 28. Treat silence gaps > 2s as hard clip boundaries during expansion
  - When expanding a clip, stop at any gap between segments > 2 seconds
  - Add `max_expansion_gap: float = 2.0` config field
  - Prevents clips from spanning topic changes or scene cuts

### Output

- [x] 33. Fix subtitle font fallback for non-macOS systems
  - Current font search list only covers macOS paths
  - Add Linux paths: `/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf`
  - Add Windows paths: `C:/Windows/Fonts/arial.ttf`
  - Log which font was selected at DEBUG level

- [x] 35. Add clip duration and score to output summary
  - CLI final summary should show: `clip_1_05m53s.mp4  28s  score=0.51`
  - Web UI results panel should show timestamp range, duration, and score under each clip name

### Web UI / UX

- [x] 36. Add per-stage progress indicator to the web UI
  - Track current stage (1–7) server-side and emit a `{"type": "progress", "stage": N, "total": 7}` SSE event at each stage start
  - Frontend renders a simple `Stage 3 / 7` label and a progress bar above the log

- [x] 37. Clean up uploaded files after job completion
  - Delete the uploaded source video from `uploads/` when job status becomes `done` or `failed`
  - Add a `cleanup_uploads: bool = True` server config option to disable this
  - Log the deletion at INFO level

- [x] 38. Add re-run button to completed jobs in the web UI
  - Store the job's config options (whisper model, top N, keywords, etc.) in the Job object
  - "Re-run" button pre-fills the options panel with those values and re-uses the uploaded file path if it still exists

---

## Silent Reaction Detection (from end-to-end test analysis)

The pipeline currently misses purely visual/audio moments — e.g. a streamer getting hit by a car in-game and reacting with laughter or screaming but no words. These are often the funniest clips. The root cause is that candidate selection is text-dominated: a silent reaction with a massive energy spike scores lower than a quiet segment with keywords.

### Problem breakdown

- `text_weight=0.35, audio_weight=0.25` in the pre-filter means audio energy is underweighted
- RMS energy per segment is a flat average — it doesn't detect *sudden spikes* (the moment of impact)
- Single-word reactions ("oh!", "whoa", "no!") are in the transcript but not in the keyword list
- There is no "silence-then-burst" pattern detector

### Proposed solutions

- [x] 39. Add audio spike detection to candidate selection
  - Compute a rolling baseline RMS over a 30s window before each segment
  - Spike score = `segment_rms / baseline_rms` — a sudden burst scores much higher than sustained loudness
  - Add `spike_score` as a fourth signal alongside text, audio, and pace
  - A spike ratio > 3x baseline should strongly boost the candidate pre-score
  - This catches "something just happened" moments regardless of whether anyone spoke

- [x] 40. Add a separate audio-only candidate track
  - Run two candidate selection passes before LLM scoring:
    1. **Text+audio track** (current): top N/2 candidates by combined text+audio score
    2. **Audio spike track**: top N/2 candidates by spike score alone, ignoring text entirely
  - Merge the two shortlists (deduplicate by proximity), then send the combined list to the LLM
  - This guarantees high-energy silent moments always get a shot at LLM scoring
  - Config: `llm_audio_candidates: int = 10` (how many pure-audio slots to reserve)

- [x] 41. Expand reaction keyword list with single-word exclamations
  - Add short reaction words that Whisper reliably transcribes even from brief outbursts:
    `"oh", "wow", "whoa", "no", "yes", "what", "ahhh", "omg", "noo", "yoo", "bro",
     "wait", "stop", "go", "run", "help", "dead", "gone", "hit", "fly", "fall"`
  - Weight these higher than multi-word keywords (they indicate a pure reaction moment)
  - Add a `reaction_keywords` list to Config separate from the main `keywords` list
  - `reaction_weight: float = 3.0` per occurrence (vs 2.0 for regular keywords)

- [x] 42. Detect silence-then-burst transition pattern
  - For each segment, compute: `silence_before = avg RMS of the 5s immediately before segment start`
  - If `silence_before < 0.1 * global_rms_mean` AND `segment_rms > 0.5 * global_rms_max`,
    classify as a "burst after silence" — strong highlight signal
  - Add `burst_score: float` to the pre-filter scoring (weight: 0.3)
  - Log detected burst moments at INFO level: `[Scorer] Burst detected at 1234.5s (silence→loud)`

- [x] 43. Decouple pre-filter weights from final scoring weights
  - Currently the same `text_weight`/`audio_weight` config values are used for both
    candidate pre-filtering AND final clip score combination
  - Add separate `llm_prefilter_text_weight: float = 0.2` and
    `llm_prefilter_audio_weight: float = 0.8` config fields
  - Pre-filter uses these (audio-heavy) to surface energetic moments
  - Final scoring still uses `text_weight`/`audio_weight` (balanced) for clip ranking
  - This is the simplest fix and should be implemented first
