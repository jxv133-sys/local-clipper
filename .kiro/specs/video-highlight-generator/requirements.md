# Requirements Document

## Introduction

The Video Highlight Generator is a local, self-hosted Python pipeline that takes a video file as input, automatically identifies "clip-worthy" segments using audio and text analysis, and exports multiple short clips (20–45 seconds each) with burned-in subtitles. The pipeline runs entirely offline with no external API dependencies, using Whisper for transcription and FFmpeg for media processing. Each pipeline stage is implemented as a separate module to ensure maintainability and testability.

## Glossary

- **Pipeline**: The end-to-end sequence of processing stages that transforms an input video into highlight clips.
- **Audio_Extractor**: The module responsible for extracting audio from the input video file.
- **Transcriber**: The module responsible for converting extracted audio to timestamped text using Whisper.
- **Transcript**: A JSON structure containing a list of segments, each with a start time, end time, and text string.
- **Segment**: A single unit of the transcript with a start time (seconds), end time (seconds), and associated text.
- **Scorer**: The module responsible for computing a clip score for each transcript segment.
- **Clip_Score**: A numeric value representing how "clip-worthy" a segment is, computed as a weighted combination of text score, audio score, and optional LLM score.
- **Text_Score**: A sub-score derived from keyword detection, sentence length, and punctuation analysis of a segment's text.
- **Audio_Score**: A sub-score derived from the RMS energy (volume) of the audio corresponding to a segment's time range.
- **LLM_Score**: An optional sub-score (1–10) produced by a local LLM rating a segment's clip-worthiness.
- **Clip_Selector**: The module responsible for ranking segments by score and selecting the top candidates for export.
- **Clip**: A selected video segment expanded to 20–45 seconds, aligned to sentence boundaries.
- **Clip_Extractor**: The module responsible for extracting clip video segments from the original video file using FFmpeg.
- **Subtitle_Generator**: The module responsible for converting transcript segments into SRT format and burning subtitles into exported clips.
- **FFmpeg**: The external command-line tool used for all media processing operations.
- **Whisper**: The local speech-to-text model used for transcription.
- **RMS_Energy**: Root Mean Square energy of an audio signal, used as a proxy for loudness.
- **SRT**: SubRip subtitle file format containing timed text entries.

## Requirements

### Requirement 1: Audio Extraction

**User Story:** As a user, I want the pipeline to extract audio from my input video, so that downstream stages can analyze speech and sound without processing the full video file.

#### Acceptance Criteria

1. WHEN a valid video file path is provided, THE Audio_Extractor SHALL extract the audio track and save it as a `.wav` file in a temporary working directory.
2. WHEN audio extraction completes, THE Audio_Extractor SHALL return the file path of the extracted `.wav` file.
3. IF the input video file does not exist, THEN THE Audio_Extractor SHALL raise a descriptive error identifying the missing file path.
4. IF the input video file contains no audio track, THEN THE Audio_Extractor SHALL raise a descriptive error indicating no audio was found.
5. IF FFmpeg is not installed or not accessible on the system PATH, THEN THE Audio_Extractor SHALL raise a descriptive error indicating the missing dependency.
6. THE Audio_Extractor SHALL invoke FFmpeg with parameters that produce a mono, 16kHz WAV file suitable for Whisper transcription.

### Requirement 2: Transcription

**User Story:** As a user, I want the pipeline to transcribe the extracted audio with word-level timestamps, so that segments can be scored and aligned to specific moments in the video.

#### Acceptance Criteria

1. WHEN a valid `.wav` file path is provided, THE Transcriber SHALL produce a Transcript containing one or more Segments.
2. THE Transcriber SHALL use the locally installed Whisper model to perform transcription without making any external network requests.
3. WHEN transcription completes, THE Transcriber SHALL return a Transcript where each Segment contains a `start` time in seconds, an `end` time in seconds, and a `text` string.
4. WHEN transcription completes, THE Transcriber SHALL serialize the Transcript to a JSON file in the working directory.
5. IF the provided `.wav` file does not exist, THEN THE Transcriber SHALL raise a descriptive error identifying the missing file path.
6. IF the audio contains no detectable speech, THEN THE Transcriber SHALL return a Transcript with an empty segment list.
7. FOR ALL valid Transcripts, deserializing the JSON file produced by the Transcriber SHALL produce a Transcript equivalent to the original (round-trip property).

### Requirement 3: Segment Scoring — Text Score

**User Story:** As a user, I want each transcript segment to receive a text-based score, so that segments with engaging language are ranked higher as clip candidates.

#### Acceptance Criteria

1. WHEN a Segment is provided, THE Scorer SHALL compute a Text_Score for that Segment.
2. THE Scorer SHALL increase the Text_Score for each occurrence of a configurable keyword (e.g., "crazy", "important", "watch this") found in the Segment's text.
3. THE Scorer SHALL increase the Text_Score based on the character length of the Segment's text, rewarding longer, more substantive speech.
4. THE Scorer SHALL increase the Text_Score for each exclamation mark (`!`) or question mark (`?`) found in the Segment's text.
5. THE Scorer SHALL normalize the Text_Score to a value in the range [0.0, 1.0].
6. FOR ALL Segments with identical text content, THE Scorer SHALL produce identical Text_Score values (deterministic property).

### Requirement 4: Segment Scoring — Audio Score

**User Story:** As a user, I want each transcript segment to receive an audio energy score, so that louder, more energetic moments are ranked higher as clip candidates.

#### Acceptance Criteria

1. WHEN a Segment and the extracted `.wav` file are provided, THE Scorer SHALL compute an Audio_Score for that Segment based on the RMS_Energy of the audio within the Segment's time range.
2. THE Scorer SHALL read only the audio samples corresponding to the Segment's `start` and `end` times when computing RMS_Energy.
3. THE Scorer SHALL normalize the Audio_Score to a value in the range [0.0, 1.0] relative to the maximum RMS_Energy observed across all Segments.
4. IF a Segment's time range contains no audio samples, THEN THE Scorer SHALL assign an Audio_Score of 0.0 for that Segment.
5. FOR ALL Segments, the Audio_Score SHALL be greater than or equal to 0.0 and less than or equal to 1.0 (invariant property).

### Requirement 5: Segment Scoring — LLM Score (Optional)

**User Story:** As a user, I want the option to use a local LLM to rate segment clip-worthiness, so that I can improve scoring quality when a local model is available.

#### Acceptance Criteria

1. WHERE LLM scoring is enabled, THE Scorer SHALL send each Segment's text to a locally running LLM and request a clip-worthiness rating.
2. WHERE LLM scoring is enabled, THE Scorer SHALL parse the LLM response to extract a numeric score in the range [1, 10].
3. WHERE LLM scoring is enabled, THE Scorer SHALL normalize the LLM_Score to a value in the range [0.0, 1.0] by dividing by 10.
4. WHERE LLM scoring is enabled, IF the LLM returns a response that does not contain a parseable numeric score, THEN THE Scorer SHALL fall back to a LLM_Score of 0.0 for that Segment and log a warning.
5. WHERE LLM scoring is disabled, THE Scorer SHALL compute the Clip_Score using only Text_Score and Audio_Score.

### Requirement 6: Segment Scoring — Final Score Combination

**User Story:** As a user, I want each segment's final clip score to combine all sub-scores with configurable weights, so that I can tune the balance between text, audio, and LLM signals.

#### Acceptance Criteria

1. THE Scorer SHALL compute the Clip_Score for each Segment as a weighted sum of Text_Score, Audio_Score, and (where enabled) LLM_Score.
2. THE Scorer SHALL read score weights from a configuration source, with default weights of 0.4 for Text_Score, 0.6 for Audio_Score, and 0.0 for LLM_Score.
3. WHERE LLM scoring is enabled, THE Scorer SHALL apply a default LLM weight of 0.3 and reduce Text_Score and Audio_Score weights proportionally so that all weights sum to 1.0.
4. FOR ALL Segments, the Clip_Score SHALL be greater than or equal to 0.0 (invariant property).
5. FOR ALL Segments, a Segment with a higher Text_Score and equal Audio_Score SHALL receive a higher or equal Clip_Score compared to a Segment with a lower Text_Score and equal Audio_Score (monotonicity property).

### Requirement 7: Clip Selection

**User Story:** As a user, I want the pipeline to select the top-scoring segments and expand them to valid clip durations, so that the exported clips are long enough to be watchable and do not cut off mid-sentence.

#### Acceptance Criteria

1. WHEN a list of scored Segments is provided, THE Clip_Selector SHALL rank all Segments in descending order by Clip_Score.
2. THE Clip_Selector SHALL select the top N Segments by Clip_Score, where N is a configurable parameter with a default value of 5.
3. WHEN expanding a selected Segment, THE Clip_Selector SHALL extend the Segment's time range so that the resulting Clip duration is between 20 and 45 seconds.
4. WHEN expanding a Segment, THE Clip_Selector SHALL align the Clip boundaries to the nearest Segment boundaries in the Transcript to avoid cutting mid-sentence.
5. WHEN expanding a Segment, THE Clip_Selector SHALL not extend the Clip start time before 0.0 seconds or the Clip end time beyond the total video duration.
6. IF two selected Segments overlap after expansion, THE Clip_Selector SHALL merge them into a single Clip spanning both time ranges, provided the merged duration does not exceed 45 seconds.
7. IF a merged Clip would exceed 45 seconds, THE Clip_Selector SHALL retain the higher-scoring Segment as a standalone Clip and discard the lower-scoring overlapping Segment.
8. FOR ALL selected Clips, the Clip duration SHALL be between 20 and 45 seconds (invariant property).

### Requirement 8: Clip Extraction

**User Story:** As a user, I want the pipeline to extract each selected clip from the original video at full quality, so that the output clips are visually identical to the source footage.

#### Acceptance Criteria

1. WHEN a Clip time range and the original video file path are provided, THE Clip_Extractor SHALL use FFmpeg to extract the corresponding segment from the original video.
2. THE Clip_Extractor SHALL attempt stream-copy extraction (no re-encoding) to preserve the original video and audio quality.
3. IF stream-copy extraction produces a file with a duration that differs from the requested Clip duration by more than 1 second, THEN THE Clip_Extractor SHALL re-extract the Clip using re-encoding to ensure accurate cut points.
4. THE Clip_Extractor SHALL save each extracted Clip as an `.mp4` file in the configured output directory.
5. THE Clip_Extractor SHALL name each output file using the pattern `clip_<index>_<start_seconds>s.mp4`, where index is the 1-based rank of the Clip by score.
6. IF the output directory does not exist, THE Clip_Extractor SHALL create it before writing output files.
7. IF FFmpeg exits with a non-zero return code during extraction, THEN THE Clip_Extractor SHALL raise a descriptive error including the FFmpeg stderr output.

### Requirement 9: Subtitle Generation

**User Story:** As a user, I want each exported clip to have subtitles burned in, so that the spoken content is readable when watching the clips.

#### Acceptance Criteria

1. WHEN a Clip and its corresponding Transcript Segments are provided, THE Subtitle_Generator SHALL produce an SRT file containing one entry per Segment within the Clip's time range.
2. THE Subtitle_Generator SHALL adjust each SRT entry's timestamps to be relative to the Clip's start time, so that subtitles are correctly synchronized within the clip.
3. THE Subtitle_Generator SHALL use FFmpeg to burn the SRT subtitles into the extracted Clip video, producing a final `.mp4` output file.
4. THE Subtitle_Generator SHALL save the intermediate SRT file alongside the clip in the output directory.
5. IF a Segment's text is empty, THE Subtitle_Generator SHALL omit that Segment from the SRT file.
6. FOR ALL valid SRT files produced, parsing then re-serializing the SRT content SHALL produce an equivalent SRT file (round-trip property).
7. IF FFmpeg exits with a non-zero return code during subtitle burning, THEN THE Subtitle_Generator SHALL raise a descriptive error including the FFmpeg stderr output.

### Requirement 10: Pipeline Orchestration

**User Story:** As a user, I want to run the entire pipeline with a single command, so that I can generate highlight clips without manually invoking each stage.

#### Acceptance Criteria

1. WHEN the user executes `python main.py <input_video_path>`, THE Pipeline SHALL execute all stages in sequence: audio extraction, transcription, scoring, clip selection, clip extraction, and subtitle generation.
2. THE Pipeline SHALL log the start and completion of each stage to standard output, including the stage name and elapsed time in seconds.
3. IF any stage raises an error, THEN THE Pipeline SHALL log the error message to standard error and exit with a non-zero exit code.
4. WHEN the pipeline completes successfully, THE Pipeline SHALL print a summary to standard output listing the file paths of all exported clips.
5. THE Pipeline SHALL store all intermediate files (extracted audio, transcript JSON, SRT files) in a temporary working directory that is created at pipeline start.
6. WHEN the pipeline completes successfully, THE Pipeline SHALL delete the temporary working directory and its contents.
7. IF the input video file path argument is not provided, THEN THE Pipeline SHALL print a usage message to standard error and exit with a non-zero exit code.

### Requirement 11: Modular Design

**User Story:** As a developer, I want each pipeline stage implemented as a separate Python module, so that individual stages can be tested, replaced, or extended independently.

#### Acceptance Criteria

1. THE Pipeline SHALL implement each stage (Audio_Extractor, Transcriber, Scorer, Clip_Selector, Clip_Extractor, Subtitle_Generator) as a separate Python module within a `pipeline/` package directory.
2. THE Pipeline SHALL define a clear function interface for each module that accepts only the inputs it requires and returns only the outputs it produces.
3. THE Pipeline SHALL not share mutable global state between modules.
4. WHERE configuration values (weights, keyword lists, clip count, output directory) are required, THE Pipeline SHALL read them from a single configuration object passed to each module at invocation time.
