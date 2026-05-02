# Requirements Document

## Introduction

The Vertical Shorts Formatter transforms the pipeline's horizontal highlight clips into platform-ready vertical short-form videos (9:16 aspect ratio) suitable for YouTube Shorts, TikTok, and Instagram Reels. It runs as Stage 8 in the existing pipeline, consuming the final clip paths produced by Stage 7 (SubtitleGenerator) and producing a parallel set of `_shorts.mp4` files alongside the originals. The feature is opt-in via a `--shorts` CLI flag and a `shorts_enabled` config field so existing workflows are unaffected.

The formatter handles three concerns: reformatting the frame from 16:9 to 9:16, intelligently repositioning the facecam overlay from a corner pip to the top third of the vertical canvas, and replacing the existing basic subtitle burn with an animated, visually engaging subtitle system featuring bubble letters, pop-up word reveals, highlight effects, and karaoke-style word colouring.

---

## Glossary

- **ShortsFormatter**: The Stage 8 orchestrator class that coordinates all sub-components and produces `_shorts.mp4` output files.
- **FrameReformatter**: Sub-component responsible for building the FFmpeg filter fragment that creates a 9:16 canvas from a 16:9 source.
- **FacecamRelocator**: Sub-component responsible for detecting the facecam pip region and building the FFmpeg filter fragment to reposition it to the top third of the vertical canvas.
- **AnimatedSubtitleRenderer**: Sub-component responsible for generating ASS subtitle files with animated visual styles and building the corresponding FFmpeg filter fragment.
- **ShortsConfig**: Dataclass holding all configuration fields for the shorts formatting stage.
- **FacecamRegion**: Dataclass representing the detected facecam pip region with pixel coordinates, corner classification, and confidence score.
- **CanvasLayout**: Dataclass representing the computed layout of the 9:16 canvas, including facecam and gameplay region boundaries.
- **FilterFragment**: Dataclass representing a composable FFmpeg filter-graph fragment with input/output labels.
- **SubtitleStyle**: Enum with values `BUBBLE`, `POPUP`, `HIGHLIGHT`, and `KARAOKE` representing the four animated subtitle visual styles.
- **ASS**: Advanced SubStation Alpha subtitle format, which supports per-event animation override tags (`\t`, `\fad`, `\k`, `\fscx`, `\fscy`) not available in SRT.
- **Pip**: Picture-in-picture; the small facecam overlay typically placed in a corner of the gameplay video.
- **Cropdetect**: An FFmpeg filter that analyses video frames to detect the bounding box of non-black content, used here to locate the facecam pip.
- **9:16**: Vertical aspect ratio (1080×1920) used by YouTube Shorts, TikTok, and Instagram Reels.
- **16:9**: Standard horizontal aspect ratio of the source gameplay clips.
- **SRTEntry**: Dataclass representing a single subtitle entry with index, start time, end time, and text.
- **Clip**: Dataclass representing a selected highlight clip with start/end times and rank.
- **Transcript**: Dataclass holding all transcript segments produced by the Whisper transcription stage.

---

## Requirements

### Requirement 1: Opt-In Shorts Formatting Stage

**User Story:** As a content creator, I want to opt in to vertical shorts formatting via a CLI flag, so that my existing horizontal clip workflow is unaffected by default.

#### Acceptance Criteria

1. THE `Config` dataclass SHALL include a `shorts_enabled` boolean field that defaults to `False`.
2. THE `main.py` CLI SHALL accept a `--shorts` flag that sets `shorts_enabled` to `True`.
3. WHEN `shorts_enabled` is `False`, THE Pipeline SHALL skip Stage 8 entirely and produce no `_shorts.mp4` files.
4. WHEN `shorts_enabled` is `True`, THE Pipeline SHALL execute Stage 8 after Stage 7 and pass the final clip paths to the `ShortsFormatter`.
5. THE Pipeline SHALL preserve all original clip files regardless of whether `shorts_enabled` is `True` or `False`.

---

### Requirement 2: Canvas Reformatting (16:9 → 9:16)

**User Story:** As a content creator, I want my horizontal gameplay clips reformatted to a vertical 9:16 canvas, so that they display correctly on YouTube Shorts, TikTok, and Instagram Reels without cropping or distortion.

#### Acceptance Criteria

1. THE `FrameReformatter` SHALL produce an FFmpeg filter fragment that scales the source video to fit within the gameplay region of the vertical canvas while preserving the source aspect ratio.
2. THE `FrameReformatter` SHALL pad the scaled gameplay video to exactly `shorts_width × shorts_height` pixels (default 1080×1920) using black fill.
3. THE `FrameReformatter` SHALL position the gameplay video in the bottom `(1 - facecam_top_fraction)` fraction of the canvas (default bottom 65%).
4. THE `FrameReformatter` SHALL centre the gameplay video horizontally within the gameplay region.
5. THE `FrameReformatter` SHALL leave the top `facecam_top_fraction` fraction of the canvas (default top 35%) empty for the facecam overlay.
6. WHEN the source video has a non-standard aspect ratio (e.g. 4:3, 1:1), THE `FrameReformatter` SHALL still produce a canvas of exactly `shorts_width × shorts_height` with black letterboxing.

---

### Requirement 3: Facecam Detection

**User Story:** As a content creator, I want the formatter to automatically detect my facecam pip in the source clip, so that it can be repositioned without manual configuration.

#### Acceptance Criteria

1. THE `FacecamRelocator` SHALL use FFmpeg's `cropdetect` filter on the first `facecam_sample_duration` seconds (default 10s) of the clip to locate the facecam pip region.
2. THE `FacecamRelocator` SHALL select the most frequently reported crop region from the `cropdetect` output as the candidate facecam region.
3. WHEN the candidate region's area fraction is less than `facecam_min_area_fraction` (default 0.04), THE `FacecamRelocator` SHALL return `None` (region too small, likely noise).
4. WHEN the candidate region's area fraction is greater than `facecam_max_area_fraction` (default 0.30), THE `FacecamRelocator` SHALL return `None` (region too large, likely gameplay area).
5. WHEN a valid facecam region is detected, THE `FacecamRelocator` SHALL classify it into one of four corners: `top-left`, `top-right`, `bottom-left`, or `bottom-right` based on the region's centre coordinates relative to the frame midpoint.
6. THE `FacecamRelocator` SHALL compute a `confidence` score as the fraction of `cropdetect` frames that reported the selected region, in the range `[0.0, 1.0]`.
7. THE `FacecamRelocator` SHALL return a `FacecamRegion` dataclass with pixel coordinates, corner classification, and confidence score when detection succeeds.

---

### Requirement 4: Facecam Relocation to Top Third

**User Story:** As a content creator, I want my facecam moved from its corner pip position to the top third of the vertical canvas, so that viewers can see my reactions prominently while watching the gameplay below.

#### Acceptance Criteria

1. WHEN a `FacecamRegion` is provided, THE `FacecamRelocator` SHALL build an FFmpeg filter fragment that crops the facecam from its source position in the original video.
2. THE `FacecamRelocator` SHALL scale the cropped facecam to fill the top `facecam_top_fraction` region of the canvas (default top 35%, 1080×672 px) while preserving the facecam's aspect ratio.
3. THE `FacecamRelocator` SHALL centre the scaled facecam horizontally within the top region.
4. THE `FacecamRelocator` SHALL overlay the facecam at position `(0, 0)` on the canvas (top-left corner of the vertical frame).
5. WHEN `detect_facecam` returns `None`, THE `FacecamRelocator` SHALL fall back to filling the top third with a blurred and zoomed version of the gameplay video.
6. THE fallback blur fill SHALL use FFmpeg's `crop`, `scale`, and `boxblur` filters to produce an aesthetically pleasing background.

---

### Requirement 5: Animated Subtitle Generation

**User Story:** As a content creator, I want animated subtitles with engaging visual styles burned into my shorts clips, so that viewers are more likely to watch to the end and engage with the content.

#### Acceptance Criteria

1. THE `AnimatedSubtitleRenderer` SHALL convert the `SRTEntry` list for each clip into an ASS subtitle file with animation override tags.
2. THE `AnimatedSubtitleRenderer` SHALL support four subtitle styles selectable via `ShortsConfig.subtitle_style`: `bubble`, `popup`, `highlight`, and `karaoke`.
3. WHEN `subtitle_style` is `bubble`, THE `AnimatedSubtitleRenderer` SHALL render subtitles with a bold thick outline and a brief scale-pop animation on entry (scale from 110% to 100% over 80ms).
4. WHEN `subtitle_style` is `popup`, THE `AnimatedSubtitleRenderer` SHALL render each word group with a fade-in and scale-in animation from 0% to 100% over 100ms.
5. WHEN `subtitle_style` is `highlight`, THE `AnimatedSubtitleRenderer` SHALL render the active word group with a coloured background box using the `subtitle_highlight_color` setting.
6. WHEN `subtitle_style` is `karaoke`, THE `AnimatedSubtitleRenderer` SHALL render word-by-word colour changes using ASS `\k` timing tags so each word changes colour as it is spoken.
7. THE `AnimatedSubtitleRenderer` SHALL position subtitles in the lower portion of the gameplay region, `subtitle_margin_bottom` pixels from the bottom of the canvas.
8. THE `AnimatedSubtitleRenderer` SHALL render subtitle text in uppercase.
9. THE `AnimatedSubtitleRenderer` SHALL write the ASS file to disk and return its path.
10. WHEN an `SRTEntry` has `start >= end` or a negative timestamp, THE `AnimatedSubtitleRenderer` SHALL skip that entry and log a warning.

---

### Requirement 6: Single FFmpeg Pass Composition

**User Story:** As a developer, I want all three transformations composed into a single FFmpeg invocation per clip, so that the shorts conversion is fast and avoids repeated encode/decode cycles.

#### Acceptance Criteria

1. THE `ShortsFormatter` SHALL assemble the `FilterFragment` outputs from `FrameReformatter`, `FacecamRelocator`, and `AnimatedSubtitleRenderer` into a single `filter_complex` string.
2. THE `ShortsFormatter` SHALL invoke FFmpeg exactly once per clip to produce the `_shorts.mp4` output.
3. THE `ShortsFormatter` SHALL use `libx264 -preset fast -crf 23` for the shorts video encoding.
4. THE `ShortsFormatter` SHALL copy the audio stream from the source clip without re-encoding.
5. THE output `_shorts.mp4` file SHALL have a video stream of exactly `shorts_width × shorts_height` pixels.
6. THE output `_shorts.mp4` file SHALL have audio.
7. WHEN FFmpeg exits with a non-zero return code, THE `ShortsFormatter` SHALL raise a `ShortsFormattingError` containing the FFmpeg stderr output.

---

### Requirement 7: Shorts Output File Naming and Preservation

**User Story:** As a content creator, I want the shorts output files named predictably alongside the originals, so that I can easily identify and upload them.

#### Acceptance Criteria

1. THE `ShortsFormatter` SHALL derive the shorts output path by appending `_shorts` to the stem of the original clip path (e.g. `clip_1_30s.mp4` → `clip_1_30s_shorts.mp4`).
2. THE `ShortsFormatter` SHALL write shorts output files to the same directory as the original clips.
3. THE `ShortsFormatter` SHALL NOT modify or overwrite the original clip files.
4. THE `ShortsFormatter` SHALL return the list of shorts paths in rank order, matching the order of the input `clip_paths`.
5. WHEN a shorts conversion fails for one clip, THE `ShortsFormatter` SHALL log the error and continue processing remaining clips, returning paths only for successfully converted clips.

---

### Requirement 8: Canvas Layout Computation

**User Story:** As a developer, I want a well-defined canvas layout dataclass computed from config, so that all sub-components share consistent region boundaries without duplicating arithmetic.

#### Acceptance Criteria

1. THE `ShortsFormatter` SHALL compute a `CanvasLayout` from `ShortsConfig` before processing each clip.
2. THE `CanvasLayout` SHALL satisfy: `facecam_height + gameplay_height == canvas_height`.
3. THE `CanvasLayout` SHALL satisfy: `facecam_width == gameplay_width == canvas_width`.
4. THE `CanvasLayout` SHALL satisfy: `gameplay_y == facecam_height`.
5. THE `CanvasLayout` SHALL set `facecam_x == 0` and `facecam_y == 0`.

---

### Requirement 9: SRT-to-Clip Subtitle Collection

**User Story:** As a developer, I want subtitle entries collected and time-adjusted per clip from the full transcript, so that each shorts clip has correctly timed subtitles relative to its own start time.

#### Acceptance Criteria

1. THE `ShortsFormatter` SHALL collect `SRTEntry` objects for each clip by filtering transcript segments that overlap the clip's `[start, end]` window.
2. THE `ShortsFormatter` SHALL adjust all subtitle timestamps to be relative to the clip's `start` time.
3. ALL returned `SRTEntry` objects SHALL have `start >= 0.0` and `start < end`.
4. WHEN no transcript segments overlap the clip window, THE `ShortsFormatter` SHALL return an empty list and produce a shorts clip with no subtitles.

---

### Requirement 10: Parallel Clip Processing

**User Story:** As a content creator, I want multiple clips converted to shorts format concurrently, so that the total formatting time scales with available CPU cores rather than clip count.

#### Acceptance Criteria

1. THE `ShortsFormatter` SHALL process multiple clips concurrently using `concurrent.futures.ThreadPoolExecutor`.
2. THE `ShortsFormatter` SHALL use at most `min(len(clips), cpu_count)` worker threads.
3. THE `ShortsFormatter` SHALL return results in rank order regardless of the order in which threads complete.

---

### Requirement 11: Error Handling and Graceful Degradation

**User Story:** As a content creator, I want the pipeline to continue producing original clips even if shorts conversion fails, so that a formatting error never blocks my main workflow.

#### Acceptance Criteria

1. WHEN `detect_facecam` returns `None`, THE `ShortsFormatter` SHALL log an INFO-level warning and proceed with the blur fallback for the top third.
2. WHEN the composite FFmpeg command fails, THE `ShortsFormatter` SHALL raise `ShortsFormattingError` and the original clip SHALL remain unmodified.
3. WHEN FFmpeg is not compiled with `libass`, THE `AnimatedSubtitleRenderer` SHALL fall back to the existing SRT-based `subtitles` filter with an enhanced style string and log a warning.
4. WHEN an `SRTEntry` has invalid timing (`start >= end` or negative values), THE `AnimatedSubtitleRenderer` SHALL skip that entry, log a warning, and continue processing remaining entries.
5. THE `ShortsFormattingError` SHALL be caught by the pipeline orchestrator so that a shorts failure does not terminate the main pipeline run.

---

### Requirement 12: ShortsConfig Dataclass

**User Story:** As a developer, I want all shorts-related settings in a dedicated `ShortsConfig` dataclass with sensible defaults, so that the feature is self-contained and easy to configure.

#### Acceptance Criteria

1. THE `ShortsConfig` dataclass SHALL include `shorts_enabled` (default `False`), `shorts_width` (default `1080`), and `shorts_height` (default `1920`).
2. THE `ShortsConfig` dataclass SHALL include `facecam_top_fraction` (default `0.35`) controlling the fraction of canvas height reserved for the facecam.
3. THE `ShortsConfig` dataclass SHALL include `facecam_detection_enabled` (default `True`), `facecam_sample_duration` (default `10.0`), `facecam_min_area_fraction` (default `0.04`), and `facecam_max_area_fraction` (default `0.30`).
4. THE `ShortsConfig` dataclass SHALL include `subtitle_style` (default `"bubble"`), `subtitle_font_size` (default `72`), `subtitle_font_name` (default `"Impact"`), `subtitle_primary_color`, `subtitle_outline_color`, `subtitle_highlight_color`, `subtitle_outline_width`, `subtitle_shadow_depth`, `subtitle_margin_bottom` (default `80`), and `subtitle_words_per_group` (default `3`).
5. THE `Config` dataclass SHALL incorporate all `ShortsConfig` fields, either by composition or by direct field inclusion.

---

### Requirement 13: ASS Subtitle File Validity

**User Story:** As a developer, I want the generated ASS subtitle files to be structurally valid, so that FFmpeg can render them without errors.

#### Acceptance Criteria

1. THE `AnimatedSubtitleRenderer` SHALL write a valid ASS file header including `[Script Info]`, `[V4+ Styles]`, and `[Events]` sections.
2. EVERY `Dialogue` line in the ASS file SHALL have `start_time < end_time`.
3. THE `AnimatedSubtitleRenderer` SHALL escape special characters in subtitle text before inserting them into ASS override tags.
4. THE `AnimatedSubtitleRenderer` SHALL produce subtitle positions within the canvas bounds (`0 ≤ x ≤ canvas_width`, `0 ≤ y ≤ canvas_height`).
5. WHEN `srt_entries` is empty, THE `AnimatedSubtitleRenderer` SHALL produce a valid ASS file with no `Dialogue` lines.

---

### Requirement 14: Security — Subprocess Safety

**User Story:** As a developer, I want all FFmpeg and ffprobe subprocess calls to use list-form arguments, so that file paths with special characters cannot cause shell injection.

#### Acceptance Criteria

1. ALL FFmpeg and ffprobe subprocess calls in the `ShortsFormatter`, `FrameReformatter`, `FacecamRelocator`, and `AnimatedSubtitleRenderer` SHALL pass arguments as Python lists, not as shell strings.
2. THE `ShortsFormatter` SHALL derive the shorts output path using `os.path.splitext` on the existing clip path, ensuring the output stays within `config.output_dir`.
3. THE `AnimatedSubtitleRenderer` SHALL escape special characters in subtitle text before inserting them into ASS override tag strings.
