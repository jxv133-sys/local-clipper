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

---

## Selection Quality Improvements (from why-chosen report analysis)

Analysis of the short-footage test run (output/short-test-3) revealed four concrete problems:
1. The why-chosen report transcript only shows the seed segment, not the full clip
2. No minimum time gap is enforced between selected clips — two clips could be from the same 5-minute stretch
3. The LLM can give a 9/10 to a quiet/repetitive moment and override strong audio signals
4. Whisper hallucinations (garbled transcription) are not detected or penalized

### Fixes

- [x] 44. Fix why-chosen report to show the full clip transcript
  - `pipeline/report_generator.py` currently pulls only the seed segment's text
  - Change it to collect all `transcript.segments` whose time range falls within `clip.start → clip.end`
  - Display them in order with timestamps, same as the existing format
  - This also means the LLM boundary refinement prompt already has the right context — no scorer change needed

- [x] 45. Enforce minimum time gap between selected clips
  - After ranking and selecting top N clips, apply a greedy deduplication pass:
    - Sort clips by score descending
    - Accept a clip only if its start time is at least `min_clip_spacing` seconds away from all already-accepted clips
    - If a clip is too close to a higher-scoring one, skip it and try the next candidate
  - Add `min_clip_spacing: float = 300.0` config field (default 5 minutes)
  - This ensures clips are spread across the video rather than clustered around one moment
  - Fall back to closer clips if there are not enough candidates to fill `top_n_clips`

- [x] 46. Cap LLM score contribution when audio energy is low
  - A high LLM score on a quiet moment should not override strong audio signals
  - Apply a soft cap: effective LLM contribution = `llm_score * min(1.0, audio_score / 0.3)`
  - This means: if `audio_score >= 0.3`, LLM score is used at full weight; below 0.3 it scales down linearly
  - Implement in `combine_scores` as an optional behaviour gated on a new config flag `llm_audio_gate: bool = True`
  - Add unit tests: a 9/10 LLM score with audio_score=0.1 should produce a lower clip_score than a 7/10 LLM score with audio_score=0.8

- [x] 47. Detect and penalize repetitive transcripts
  - Compute a repetition ratio for each segment: `unique_words / total_words`
  - If `repetition_ratio < 0.4` (more than 60% of words are duplicates), apply a penalty multiplier of 0.5 to the text score
  - This catches Whisper hallucinations (repeated phrases) and genuinely repetitive content (e.g. "Hallelujah" x4)
  - Add `repetition_penalty_threshold: float = 0.4` and `repetition_penalty_multiplier: float = 0.5` config fields
  - Log penalized segments at DEBUG level: `[Scorer] Repetition penalty applied at 345.6s (ratio=0.25)`
  - Add unit tests covering: normal text (no penalty), highly repetitive text (penalty applied), single-word segment (edge case)

---

## Full-Footage Evaluation

- [x] 48. Run full end-to-end test on full footage with LLM, generate 10 clips, and evaluate
  - Run: `python3 main.py ~/Desktop/test-footage/full-footage.mp4 --llm --llm-model llama3 --whisper-model base --top-n 10 --output-dir output/full-test-1`
  - Read all 10 why-chosen reports from `output/full-test-1/`
  - Evaluate each clip on:
    - Does the transcript make sense (no obvious Whisper hallucinations)?
    - Is the clip score distribution healthy (spread across the video, not clustered)?
    - Do the LLM titles/descriptions match the transcript content?
    - Are any clips clearly wrong picks (low energy, boring content, mid-sentence cuts)?
  - Document findings and propose any further improvements to scoring, selection, or boundary refinement

---

## Shorts Format: Setup → Moment → Reaction Arc

Analysis of `output/full-test-2` revealed that clips are selected around the peak moment but don't follow the YouTube Shorts narrative arc. The target structure is:

- **Setup (5–10s):** what's happening in the stream — gives the viewer context
- **Moment (10–30s):** the funny/scary/impressive thing — the reason the clip exists
- **Reaction (5–10s):** the streamer's response — this is what viewers share

Current problems:
1. LLM boundary refinement collapses clips below `min_clip_duration` (clips #7 at 15s and #9 at 10s in full-test-2)
2. Expansion is symmetric — it grabs equally left and right of the seed, so the moment ends up in the middle with no guaranteed reaction tail
3. The LLM boundary prompt asks "pick good start/end times" with no awareness of the arc structure
4. The spacing deduplication pass runs before LLM boundary refinement, so refined clips can violate spacing

- [x] 49. Fix LLM boundary refinement collapsing clips below minimum duration
  - After `refine_clip_boundaries_with_llm` returns, check if the new duration is below `config.min_clip_duration`
  - If it is, fall back to the pre-refinement clip boundaries (do not apply the LLM's suggestion)
  - Log a warning: `[ClipSelector] LLM boundary refinement produced {new_duration:.0f}s clip (min {min_clip_duration:.0f}s); keeping original`
  - Add unit tests: mock the LLM to return a very tight window; assert the original clip is returned unchanged

- [x] 50. Bias clip expansion to guarantee a reaction tail after the seed segment
  - After identifying the seed segment (the peak moment), change the expansion strategy:
    - First, always expand **forward** from the seed end until at least `min_reaction_duration` seconds of content are included (default 8s)
    - Then expand **backward** from the seed start to fill the remaining duration budget up to `min_clip_duration`
    - This ensures the reaction is always captured; setup fills whatever space is left
  - Add `min_reaction_duration: float = 8.0` config field
  - If there is not enough content after the seed to reach `min_reaction_duration` (e.g. seed is near the video end), use whatever is available — do not fail
  - Add unit tests: seed near start of transcript (reaction fills forward), seed near end (reaction truncated gracefully), seed in middle (full arc captured)

- [x] 51. Rewrite LLM boundary refinement prompt to target Setup → Moment → Reaction structure
  - Replace the current generic "pick good start/end times" prompt with one that explicitly asks the LLM to identify and preserve the three-part arc
  - New prompt structure:
    - Show the ±45s transcript context window (expanded from ±30s to give more arc visibility)
    - Ask the LLM to identify: (a) where the setup begins, (b) where the moment peaks, (c) where the reaction ends
    - Instruct it to set `START_TIME` at the setup start and `END_TIME` after the reaction resolves
    - Add explicit constraints: "The clip MUST be at least {min_clip_duration}s long. Do not shrink the clip below this."
    - Add arc guidance: "Setup should be 5–10s before the moment. Reaction should be 5–10s after the moment. If the reaction is missing, extend END_TIME forward to capture it."
  - Keep the same `START_TIME: / END_TIME: / REASON:` response format so parsing is unchanged
  - Add a unit test: mock LLM response with a valid arc; assert boundaries are applied correctly

- [x] 52. Re-run spacing deduplication after LLM boundary refinement
  - Currently `_apply_spacing` runs before LLM boundary refinement, so refined clips can end up closer than `min_clip_spacing`
  - Move the `_apply_spacing` call to after the LLM refinement + re-resolve-overlaps step
  - When spacing removes a refined clip, log: `[ClipSelector] Clip at {start:.1f}s removed by spacing pass (too close to clip at {other:.1f}s)`
  - Add a unit test: two clips that are far apart before refinement but close after; assert spacing pass removes the lower-scoring one

- [x] 53. Evaluate Shorts arc quality on full footage after tasks 49–52
  - Re-run: `python3 main.py ~/Desktop/test-footage/full-footage.mp4 --llm --llm-model llama3 --whisper-model small --top-n 10 --output-dir output/full-test-3`
  - For each clip, read the why-chosen report and evaluate:
    - Does the clip have a recognisable setup (5–10s of context before the moment)?
    - Is the moment clearly identifiable in the transcript?
    - Does the clip end with a reaction (streamer response after the moment)?
    - Is the duration between 30s and 60s?
  - Document pass/fail for each clip and note any remaining structural issues


---

## Audio Spike Priority Improvements (from user feedback on full-test-3)

User feedback: Clip #4 (the outro) was actually the best clip. The evaluation was wrong. We need to prioritize sudden loud sounds (audio spikes) and reserve 20% of clips for pure audio spike moments.

- [ ] 54. Increase audio spike candidate reservation to 20% of clips
  - Currently `llm_audio_candidates: int = 10` is a fixed number
  - Change to a percentage: `llm_audio_spike_percentage: float = 0.2` (20% of top_n_clips)
  - Update `_build_candidate_windows` to calculate audio_budget as `int(config.top_n_clips * config.llm_audio_spike_percentage)`
  - This ensures audio spike moments always get 20% of the clip slots regardless of top_n setting
  - Add unit tests: with top_n=10, audio spike track should get 2 slots; with top_n=5, should get 1 slot

- [ ] 55. Add `--no-subs` CLI flag to disable subtitle burn-in during testing
  - Add `--no-subs` argument to main.py that sets `config.burn_subtitles = False`
  - When disabled, skip the subtitle generation step entirely (don't call `generate_subtitles`)
  - Log: `[Pipeline] Subtitle burn-in disabled (--no-subs flag)`
  - This speeds up testing runs significantly (subtitle burn-in is the slowest step)
  - Update README.md with the new flag

- [ ] 56. Re-run full-test-3 with audio spike priority and no subtitle burn-in
  - Run: `python3 main.py ~/Desktop/test-footage/full-footage.mp4 --llm --llm-model llama3 --whisper-model small --top-n 10 --output-dir output/full-test-4 --no-subs`
  - Compare clip selection to full-test-3: are more audio spike moments selected?
  - Verify that 2 out of 10 clips (20%) are pure audio spike moments (high spike_score, low text_score)
  - Document findings in output/full-test-4/COMPARISON_REPORT.md
