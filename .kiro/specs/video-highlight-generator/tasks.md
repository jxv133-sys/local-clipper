# Implementation Plan: Video Highlight Generator

## Overview

Implement a local, offline Python pipeline that extracts audio from a video, transcribes it with Whisper, scores segments, selects highlights, extracts clips, and burns in subtitles. Each stage is a separate module under `pipeline/`. The implementation follows the staged architecture defined in the design document.

## Tasks

- [~] 1. Set up project structure, data models, and configuration
  - Create the `pipeline/` package directory with `__init__.py`
  - Create `config.py` with the `Config` dataclass and all default values (weights, keywords, paths, Whisper model, LLM settings, clip count, durations)
  - Create `pipeline/models.py` (or inline in each module) with `Segment`, `Transcript`, `ScoredSegment`, `Clip`, and `SRTEntry` dataclasses
  - Implement `Transcript.to_dict()` and `Transcript.from_dict()` serialization methods
  - Define the `PipelineError` base exception and all module-specific subclasses (`AudioExtractionError`, `TranscriptionError`, `ClipExtractionError`, `SubtitleError`, `LLMScoringError`)
  - Create `tests/` directory with empty `__init__.py`
  - Add `requirements.txt` (or `pyproject.toml`) pinning `openai-whisper`, `hypothesis`, `pytest`, `numpy`, `scipy`
  - _Requirements: 11.1, 11.2, 11.3, 11.4_

  - [ ]* 1.1 Write property test for Transcript serialization round-trip
    - **Property 1: Transcript serialization round-trip**
    - Use `st.builds(Transcript, segments=st.lists(st.builds(Segment, start=st.floats(min_value=0.0, max_value=10000.0, allow_nan=False), end=st.floats(min_value=0.0, max_value=10000.0, allow_nan=False), text=st.text())))` 
    - Assert `Transcript.from_dict(t.to_dict()) == t` for all generated transcripts
    - `@settings(max_examples=100)`
    - **Validates: Requirements 2.7**

- [~] 2. Implement Audio Extractor (`pipeline/audio_extractor.py`)
  - Implement `extract_audio(config: Config, video_path: str) -> str`
  - Check that `video_path` exists; raise `FileNotFoundError` with a descriptive message if not
  - Detect missing FFmpeg on PATH; raise `AudioExtractionError` if `ffmpeg` is not found
  - Invoke FFmpeg with `-ac 1 -ar 16000 -vn` to produce a mono 16 kHz WAV in `config.work_dir`
  - Parse FFmpeg stderr to detect "no audio" conditions; raise `AudioExtractionError` with a descriptive message
  - Raise `AudioExtractionError` on any non-zero FFmpeg exit code, including stderr in the message
  - Return the path to the extracted `.wav` file
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6_

  - [ ]* 2.1 Write unit tests for Audio Extractor
    - Mock `subprocess.run` to simulate FFmpeg success, non-zero exit, and missing executable
    - Test: valid video → returns `.wav` path in `work_dir`
    - Test: missing video file → `FileNotFoundError`
    - Test: FFmpeg not on PATH → `AudioExtractionError`
    - Test: FFmpeg stderr contains no-audio indicator → `AudioExtractionError`
    - Test: FFmpeg non-zero exit → `AudioExtractionError` with stderr in message
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [~] 3. Implement Transcriber (`pipeline/transcriber.py`)
  - Implement `transcribe(config: Config, wav_path: str) -> Transcript`
  - Check that `wav_path` exists; raise `FileNotFoundError` with a descriptive message if not
  - Load the Whisper model via `whisper.load_model(config.whisper_model)` and call `.transcribe(wav_path, word_timestamps=True)`
  - Map Whisper output segments to `Segment` dataclass instances (start, end, text)
  - Return a `Transcript` with an empty segment list if Whisper detects no speech
  - Serialize the `Transcript` to a JSON file in `config.work_dir` using `Transcript.to_dict()`
  - Raise `TranscriptionError` on Whisper model load failure
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

  - [ ]* 3.1 Write unit tests for Transcriber
    - Mock `whisper.load_model` to return a fake model with a controlled `.transcribe()` response
    - Test: valid WAV → returns `Transcript` with correct segments and writes JSON to `work_dir`
    - Test: missing WAV file → `FileNotFoundError`
    - Test: Whisper returns no segments → `Transcript` with empty list
    - Test: JSON file round-trips correctly (deserialize and compare)
    - _Requirements: 2.1, 2.3, 2.4, 2.5, 2.6_

- [~] 4. Implement Scorer — text and combination logic (`pipeline/scorer.py`)
  - Implement `compute_text_score(config: Config, segment: Segment) -> float`
    - Add score for each keyword occurrence in `segment.text` (case-insensitive)
    - Add score proportional to character length of `segment.text`
    - Add score for each `!` or `?` in `segment.text`
    - Normalize the raw score to [0.0, 1.0] using a fixed maximum (e.g., sigmoid or min-max with a cap)
  - Implement `combine_scores(config: Config, text: float, audio: float, llm: float | None) -> float`
    - Return `text_weight * text + audio_weight * audio + llm_weight * (llm or 0.0)`
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 6.1, 6.2, 6.4, 6.5_

  - [ ]* 4.1 Write property test for text score determinism
    - **Property 2: Text score determinism**
    - Generate arbitrary `Segment` with `st.builds(Segment, text=st.text(), start=st.just(0.0), end=st.just(1.0))`
    - Assert `compute_text_score(config, seg) == compute_text_score(config, seg)` for all inputs
    - `@settings(max_examples=100)`
    - **Validates: Requirements 3.6**

  - [ ]* 4.2 Write property test for text score normalization
    - **Property 3: Text score is normalized**
    - Generate arbitrary `Segment` with arbitrary text
    - Assert `0.0 <= compute_text_score(config, seg) <= 1.0` for all inputs
    - `@settings(max_examples=100)`
    - **Validates: Requirements 3.5**

  - [ ]* 4.3 Write property test for text score monotonicity
    - **Property 4: Text score monotonicity**
    - Generate a base text string; create enriched variants by appending a keyword, `!`, `?`, or extra non-whitespace characters
    - Assert `compute_text_score(config, enriched) >= compute_text_score(config, base)` for all variants
    - `@settings(max_examples=100)`
    - **Validates: Requirements 3.2, 3.3, 3.4**

  - [ ]* 4.4 Write property test for clip score weighted sum
    - **Property 6: Clip score equals weighted sum**
    - Generate `st.floats(0.0, 1.0)` for text, audio, llm scores and `st.floats(min_value=0.0)` for weights
    - Assert `combine_scores` returns exactly `text_w * text + audio_w * audio + llm_w * llm`
    - `@settings(max_examples=100)`
    - **Validates: Requirements 6.1, 6.4**

  - [ ]* 4.5 Write property test for clip score monotonicity with text score
    - **Property 7: Clip score monotonicity with text score**
    - Generate two floats `text_a >= text_b` in [0.0, 1.0] with equal audio and llm scores
    - Assert `combine_scores(..., text_a, ...) >= combine_scores(..., text_b, ...)`
    - `@settings(max_examples=100)`
    - **Validates: Requirements 6.5**

  - [ ]* 4.6 Write unit tests for Scorer text and combination logic
    - Test: segment with known keyword → text score increases vs. segment without keyword
    - Test: segment with `!` → text score increases vs. segment without
    - Test: longer segment text → text score increases vs. shorter
    - Test: `combine_scores` with known weights and scores → expected weighted sum
    - Test: `combine_scores` with `llm=None` and `llm_weight=0.0` → same as text+audio only
    - _Requirements: 3.2, 3.3, 3.4, 6.1, 6.2_

- [~] 5. Implement Scorer — audio score (`pipeline/scorer.py`, continued)
  - Implement `compute_audio_score(segments: list[Segment], wav_path: str) -> list[float]`
    - Load the WAV file with `scipy.io.wavfile` or `wave` + `numpy`
    - For each segment, slice the audio samples corresponding to `[start, end]` seconds
    - Compute RMS energy for each slice; assign 0.0 if the slice is empty
    - Normalize all RMS values by the maximum observed RMS across all segments (avoid divide-by-zero)
    - Return a list of floats in [0.0, 1.0], one per segment
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

  - [ ]* 5.1 Write property test for audio score normalization
    - **Property 5: Audio score is normalized**
    - Generate a list of `Segment` objects and a synthetic in-memory WAV (mocked or written to a temp file)
    - Assert every value in `compute_audio_score(segments, wav_path)` is in [0.0, 1.0]
    - `@settings(max_examples=100)`
    - **Validates: Requirements 4.3, 4.5**

  - [ ]* 5.2 Write unit tests for audio score
    - Test: segment with known audio samples → expected normalized RMS value
    - Test: segment with silent audio (all zeros) → audio score 0.0
    - Test: segment time range outside WAV duration → audio score 0.0
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [ ] 6. Implement Scorer — LLM score and full `score_segments` assembly (`pipeline/scorer.py`, continued)
  - Implement `compute_llm_score(config: Config, segment: Segment) -> float`
    - POST segment text to `config.llm_endpoint` with `config.llm_model`
    - Parse response to extract a numeric value in [1, 10]; normalize to [0.0, 1.0] by dividing by 10
    - If response contains no parseable number, log a warning and return 0.0
    - Raise `LLMScoringError` (non-fatal) if the endpoint is unreachable; caller falls back to 0.0
  - Implement `score_segments(config: Config, transcript: Transcript, wav_path: str) -> list[ScoredSegment]`
    - Call `compute_text_score` for each segment
    - Call `compute_audio_score` for all segments at once
    - If `config.llm_enabled`, call `compute_llm_score` for each segment; catch `LLMScoringError` and use 0.0
    - Call `combine_scores` for each segment to produce `clip_score`
    - Return list of `ScoredSegment` objects
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 6.1, 6.2, 6.3_

  - [ ]* 6.1 Write property test for score weights sum to 1.0
    - **Property 8: Score weights sum to 1.0**
    - Generate a `Config` with `llm_enabled=True` and an arbitrary `llm_weight` in [0.0, 1.0]; derive `text_weight` and `audio_weight` proportionally
    - Assert `abs(text_weight + audio_weight + llm_weight - 1.0) < 1e-9`
    - `@settings(max_examples=100)`
    - **Validates: Requirements 6.3**

  - [ ]* 6.2 Write unit tests for LLM score and score assembly
    - Mock HTTP call to LLM endpoint; test: valid numeric response → correct normalized score
    - Test: LLM response with no parseable number → score 0.0 and warning logged
    - Test: LLM endpoint unreachable → `LLMScoringError` caught, score 0.0
    - Test: `llm_enabled=False` → `compute_llm_score` not called; `clip_score` uses only text + audio
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [ ] 7. Checkpoint — Ensure all scorer and transcriber tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 8. Implement Clip Selector (`pipeline/clip_selector.py`)
  - Implement `select_clips(config: Config, scored_segments: list[ScoredSegment], transcript: Transcript, video_duration: float) -> list[Clip]`
  - Sort `scored_segments` in descending order by `clip_score`
  - Select the top `config.top_n_clips` segments
  - For each selected segment, expand the time range to reach `min_clip_duration` (20 s) by pulling in adjacent transcript segments at sentence boundaries; cap at `max_clip_duration` (45 s)
  - Clamp `start` to `>= 0.0` and `end` to `<= video_duration`
  - Detect overlapping clips; merge if merged duration `<= 45 s`, otherwise discard the lower-scoring clip
  - Assign 1-based `rank` by score to each final `Clip`
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8_

  - [ ]* 8.1 Write property test for clip boundary invariant
    - **Property 9: Clip boundary invariant**
    - Generate `st.lists(st.builds(ScoredSegment, ...))` with a positive `video_duration`
    - Assert every returned `Clip` satisfies: `20.0 <= (clip.end - clip.start) <= 45.0`, `clip.start >= 0.0`, `clip.end <= video_duration`
    - `@settings(max_examples=100)`
    - **Validates: Requirements 7.3, 7.5, 7.8**

  - [ ]* 8.2 Write property test for clip selection score ordering
    - **Property 10: Clip selection preserves score ordering**
    - Generate a list of `ScoredSegment` objects with arbitrary scores
    - Assert the returned `Clip` list is in non-increasing order of seed score
    - `@settings(max_examples=100)`
    - **Validates: Requirements 7.1**

  - [ ]* 8.3 Write unit tests for Clip Selector
    - Test: top N selection returns correct segments by score
    - Test: expansion clamps start to 0.0 when segment is near video start
    - Test: expansion clamps end to `video_duration` when segment is near video end
    - Test: two overlapping clips within 45 s → merged into one clip
    - Test: two overlapping clips that would exceed 45 s → higher-scoring clip retained, lower discarded
    - _Requirements: 7.2, 7.3, 7.4, 7.5, 7.6, 7.7_

- [ ] 9. Implement Clip Extractor (`pipeline/clip_extractor.py`)
  - Implement `extract_clips(config: Config, clips: list[Clip], video_path: str) -> list[str]`
  - Create `config.output_dir` if it does not exist
  - For each clip, attempt stream-copy extraction with FFmpeg (`-c copy`)
  - Name each output file `clip_<rank>_<start_seconds>s.mp4`
  - After stream-copy, probe the output duration (via `ffprobe` or FFmpeg stderr); if it differs from the requested duration by more than 1 second, re-extract with re-encoding
  - Raise `ClipExtractionError` on any non-zero FFmpeg exit code, including stderr in the message
  - Return the list of output `.mp4` file paths
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7_

  - [ ]* 9.1 Write unit tests for Clip Extractor
    - Mock `subprocess.run` for FFmpeg and `ffprobe`
    - Test: successful stream-copy → correct output filename and path returned
    - Test: stream-copy duration mismatch > 1 s → re-encode invoked
    - Test: output directory does not exist → directory created before writing
    - Test: FFmpeg non-zero exit → `ClipExtractionError` with stderr in message
    - _Requirements: 8.2, 8.3, 8.5, 8.6, 8.7_

- [ ] 10. Implement Subtitle Generator (`pipeline/subtitle_generator.py`)
  - Implement SRT serialization: given a list of `SRTEntry` objects, produce a valid SRT string
  - Implement SRT parsing: given an SRT string, return a list of `SRTEntry` objects
  - Implement `generate_subtitles(config: Config, clips: list[Clip], transcript: Transcript, clip_paths: list[str]) -> list[str]`
    - For each clip, collect all non-empty `Segment` objects whose time range falls within `[clip.start, clip.end]`
    - Adjust each segment's timestamps to be relative to `clip.start`
    - Serialize to SRT and write the `.srt` file alongside the clip in `config.output_dir`
    - Invoke FFmpeg to burn the SRT into the clip video, producing the final `.mp4`
    - Raise `SubtitleError` on any non-zero FFmpeg exit code, including stderr in the message
  - Return the list of final subtitle-burned `.mp4` file paths
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.7_

  - [ ]* 10.1 Write property test for SRT timestamp offset
    - **Property 11: SRT timestamp offset**
    - Generate a `Segment` with arbitrary start/end and a clip start time `st.floats(min_value=0.0)`
    - Assert the produced `SRTEntry` has `start == segment.start - clip_start` and `end == segment.end - clip_start`
    - `@settings(max_examples=100)`
    - **Validates: Requirements 9.2**

  - [ ]* 10.2 Write property test for SRT entry count
    - **Property 12: SRT entry count matches in-range segments**
    - Generate a `Clip` and a list of `Segment` objects with arbitrary time ranges and text
    - Assert the number of SRT entries equals the count of non-empty segments whose range falls within the clip
    - `@settings(max_examples=100)`
    - **Validates: Requirements 9.1, 9.5**

  - [ ]* 10.3 Write property test for SRT serialization round-trip
    - **Property 13: SRT serialization round-trip**
    - Generate `st.lists(st.builds(SRTEntry, index=st.integers(min_value=1), start=st.floats(min_value=0.0, allow_nan=False), end=st.floats(min_value=0.0, allow_nan=False), text=st.text(min_size=1)))`
    - Serialize to SRT string, parse back, assert equivalent list
    - `@settings(max_examples=100)`
    - **Validates: Requirements 9.6**

  - [ ]* 10.4 Write unit tests for Subtitle Generator
    - Test: segment with empty text → omitted from SRT output
    - Test: timestamps adjusted correctly relative to clip start
    - Test: SRT file written to correct path alongside clip
    - Test: FFmpeg non-zero exit during subtitle burn → `SubtitleError` with stderr in message
    - _Requirements: 9.2, 9.4, 9.5, 9.7_

- [ ] 11. Checkpoint — Ensure all clip selector, extractor, and subtitle tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 12. Implement Pipeline Orchestrator (`main.py`)
  - Parse CLI arguments; if `<input_video_path>` is missing, print usage to stderr and exit with code 1
  - Load `Config` with defaults; allow overrides via CLI flags or environment variables as appropriate
  - Create a temporary working directory and set `config.work_dir`
  - Wrap the full pipeline in a `try/except PipelineError` block
  - Call each stage in sequence: `extract_audio` → `transcribe` → `score_segments` → `select_clips` → `extract_clips` → `generate_subtitles`
  - Log stage name and elapsed time to stdout before and after each stage call
  - On success: print paths of all exported clips to stdout, then delete the temp directory
  - On failure: log error message to stderr, exit with code 1, leave temp directory intact for debugging
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 11.1, 11.2, 11.3, 11.4_

  - [ ]* 12.1 Write unit tests for Pipeline Orchestrator
    - Mock all stage functions; test: all stages called in correct order with correct arguments
    - Test: missing CLI argument → usage message on stderr, exit code 1
    - Test: stage raises `PipelineError` → error logged to stderr, exit code 1, temp dir not deleted
    - Test: successful run → temp dir deleted, clip paths printed to stdout
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7_

- [ ] 13. Final checkpoint — Ensure all tests pass
  - Run `pytest tests/` and confirm all unit and property-based tests pass.
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Each task references specific requirements for traceability
- Property tests use Hypothesis with `@settings(max_examples=100)` and a comment referencing the design property number
- Checkpoints at tasks 7, 11, and 13 ensure incremental validation before moving to the next stage
- The LLM scoring path (task 6) is optional at runtime via `config.llm_enabled`; the implementation must still be present
