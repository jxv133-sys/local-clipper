# Improvement Tasks

## Scoring & Performance

- [x] 1. Transcript caching / resume
  - Cache the transcript to a stable path keyed on `(video_path, whisper_model, file_mtime)`
  - On re-run, load from cache instead of re-transcribing — Whisper is the slowest stage
  - Cache file: `~/.cache/local-clipper/<hash>.json` or a configurable cache dir
  - Add `--no-cache` CLI flag to force re-transcription
  - Log: `[Transcriber] Loaded transcript from cache (skipping Whisper)`

- [x] 2. Config validation in `__post_init__`
  - Add a `__post_init__` method to `Config` that validates:
    - `text_weight + audio_weight + llm_weight` approximately equals 1.0 when LLM is enabled
    - `min_clip_duration <= max_clip_duration`
    - `0.0 <= llm_audio_spike_percentage <= 1.0`
    - `audio_percentile_low < audio_percentile_high`
    - `excitement_volume_weight + excitement_pitch_weight == 1.0`
  - Raise `ValueError` with a clear message on invalid config
  - Add unit tests covering each validation rule

- [x] 3. Load WAV file once — eliminate duplicate reads
  - `compute_spike_score`, `compute_burst_score`, and `compute_audio_score_with_raw` each call `scipy.io.wavfile.read(wav_path)` independently
  - Load the WAV data once in `score_segments` and pass the `(sample_rate, data)` tuple into each function
  - For a 1-hour stream this avoids reading a large file 3× per run
  - Update function signatures: add `wav_data: tuple[int, np.ndarray] | None = None` parameter with fallback to reading from path

- [x] 4. Add language detection config field
  - Add `language: str = "auto"` to `Config`
  - Pass it through to `model.transcribe()` in both `_transcribe_faster_whisper` and `_transcribe_openai_whisper`
  - When `"auto"`, use Whisper's built-in detection (current behaviour)
  - When set (e.g. `"en"`, `"es"`), skip auto-detection for a meaningful speedup
  - Expose as `--language` CLI flag and web UI dropdown

- [x] 5. Replace LLM availability check with a lightweight HTTP probe
  - `_check_llm_model_available` currently sends a full inference request just to test connectivity — slow and wasteful
  - Replace with a GET to `http://<host>:<port>/api/tags` (Ollama's model list endpoint)
  - Check that the configured model name appears in the response
  - Falls back gracefully if the endpoint is not Ollama (non-200 → assume available)
  - Reduces startup latency when LLM is enabled

## Output Quality

- [x] 6. Word-level subtitle timing
  - `faster-whisper` already returns `word_timestamps=True` in `transcriber.py` but word-level data is discarded
  - Store word timestamps in `Segment` (add `words: list[WordTimestamp]` field to the model)
  - Use word boundaries in `subtitle_generator.py` to split subtitles at the word level
  - Enables karaoke-style word-by-word highlighting — standard for Shorts/TikTok
  - Fall back to segment-level timing when word timestamps are unavailable (openai-whisper path)

- [x] 7. Deduplicate clips with near-identical transcript content
  - The spacing pass only checks start-time proximity — two clips with very different start times can have nearly identical transcript text
  - After the spacing pass, compute pairwise Jaccard similarity on clip transcript word sets
  - If two clips have similarity > 0.7, discard the lower-scoring one
  - Add `dedup_similarity_threshold: float = 0.7` to `Config`
  - Log: `[ClipSelector] Clip at {start:.1f}s removed (transcript similarity {sim:.2f} to clip at {other:.1f}s)`

- [x] 8. Fix `report_generator.py` using hardcoded keyword list
  - The report scans for a hardcoded list of highlight words instead of `config.keywords`
  - Pass `config` into `generate_report` and use `config.keywords` for keyword detection
  - This ensures the report accurately reflects what actually drove the score
  - Update all call sites to pass `config`

## Web UI

- [x] 9. Add clip thumbnail preview to web UI results panel
  - After clip extraction, generate a JPEG thumbnail at the clip midpoint:
    `ffmpeg -ss <mid> -i <clip.mp4> -frames:v 1 -q:v 2 <clip_thumb.jpg>`
  - Serve thumbnails as static files alongside clips
  - Display as a small preview image in the results panel above the filename
  - Fall back gracefully if ffmpeg thumbnail generation fails

- [x] 10. Add LLM scoring progress to web UI
  - The LLM phase can take several minutes (20 candidates × ~10s each) with no feedback
  - Emit a structured SSE event per candidate: `{"type": "llm_progress", "current": 3, "total": 20}`
  - Frontend renders: `LLM scoring: candidate 3 / 20` with a mini progress bar
  - Also emit a final `{"type": "llm_done", "scored": 20}` event when complete

- [x] 11. Auto-scale `min_clip_spacing` for short videos
  - Default `min_clip_spacing = 300s` (5 minutes) is too aggressive for videos shorter than `top_n_clips * 5` minutes
  - At the start of `select_clips`, if `video_duration / top_n_clips < min_clip_spacing`, auto-scale to `video_duration / (top_n_clips + 1)`
  - Log: `[ClipSelector] Auto-scaled min_clip_spacing from {old:.0f}s to {new:.0f}s (video too short)`
  - Add unit tests: 10-minute video with top_n=5 should auto-scale spacing to ~100s

- [x] 12. Add advanced settings panel to web UI
  - Expand the options panel to include all major config fields grouped into collapsible sections:
    - **Basic**: Whisper model, Top N clips, Keywords, Output dir
    - **Scoring**: Text/audio/LLM weights (sliders), Min text score threshold, Reaction weight
    - **Clips**: Min/max clip duration, Min clip spacing, Audio spike percentage, Burn subtitles toggle
    - **Advanced**: Genre dropdown, Platform dropdown, LLM audio gate toggle, Repetition penalty settings
  - Add tooltips explaining each setting
  - Store settings in `localStorage` so they persist across page reloads

- [x] 13. Add job cancellation to web UI
  - Add "Cancel" button next to running jobs in the job list
  - New API endpoint: `POST /api/jobs/<job_id>/cancel` that signals the background thread to stop
  - Update job status to `"cancelled"` and clean up temp files
  - Show a confirmation dialog before cancelling

## Performance

- [x] 14. Parallel clip extraction
  - Clips are currently extracted sequentially; each FFmpeg call is independent
  - Use `concurrent.futures.ThreadPoolExecutor` in `extract_clips` to run FFmpeg calls concurrently
  - Limit concurrency to `min(len(clips), 4)` to avoid saturating I/O
  - Preserve rank-ordered output list regardless of completion order

- [x] 15. Cache Whisper model in memory for web server
  - The web server reloads the Whisper model from disk on every job, even when the model name hasn't changed
  - Cache the loaded model in a module-level dict keyed on `(model_name, backend)` in `transcriber.py`
  - Invalidate the cache only when the model name changes between jobs
  - Log: `[Transcriber] Using cached model (skipping reload)`

- [x] 16. Skip subtitle burning when no segments overlap the clip
  - `generate_subtitles` runs an FFmpeg subtitle-burn pass even when a clip has zero overlapping transcript segments
  - Check segment overlap before invoking FFmpeg; if no segments overlap, copy the clip file as-is
  - Log: `[SubtitleGenerator] Clip #N has no transcript segments — skipping subtitle burn`

## Output Quality


- [x] 18. Silence trimming at clip boundaries
  - Clips often start or end on dead air because boundaries are snapped to segment timestamps
  - After extraction, run `ffmpeg -af silenceremove` to trim leading/trailing silence > 0.5s
  - Only trim if the resulting duration stays above `min_clip_duration`
  - Add `trim_silence: bool = True` to `Config`; expose as `--no-trim-silence` CLI flag

- [x] 19. Scene-change aware clip boundaries
  - Clip start/end times are snapped to Whisper segment boundaries, which may land mid-scene
  - Use `ffprobe -show_frames -select_streams v -show_entries frame=pkt_pts_time,pict_type` to detect scene changes near each boundary
  - Snap the boundary to the nearest scene cut within ±2 seconds if one exists
  - Fall back to the original boundary when no scene cut is found nearby

## Web UI

- [x] 20. Inline clip preview player
  - Users must download a clip to watch it; add an inline `<video>` element in each clip card
  - Show a collapsed `▶ Preview` button below the download button; clicking it expands the video player
  - Use the existing `/output/<filename>` static route to stream the clip
  - Auto-pause other players when a new one is opened



- [~] 22. Batch download as zip
  - Users must download clips one at a time; add a "Download all" button to the results panel
  - New API endpoint: `GET /api/jobs/<job_id>/download-all` — streams a zip of all clips for that job
  - Use Python's `zipfile` module to build the archive in memory and stream it as `application/zip`
  - Show the button only when a job has 2+ clips

## Reliability

- [~] 23. Retry logic for LLM calls
  - A single timeout currently drops the LLM score to 0.0 with no retry
  - Retry once with a 15-second timeout after an initial 30-second timeout failure
  - Log: `[Scorer] LLM call timed out, retrying (attempt 2/2)…`
  - Only fall back to 0.0 after both attempts fail


- [~] 25. FFmpeg version detection at startup
  - Some FFmpeg flags differ between versions (e.g. `-c:s` subtitle codec options changed in 5.x)
  - Run `ffmpeg -version` once at import time in `audio_extractor.py` and parse the major version
  - Store it in a module-level `FFMPEG_VERSION: int` constant
  - Use it to select the correct flag variants in `clip_extractor.py` and `subtitle_generator.py`

- [~] 26. Run full test using short footage
-anylise output for improvments and add and to this tasks.md file
-use short footage at /Users/jonahvaira/Desktop/test-footage/short-footage.mov