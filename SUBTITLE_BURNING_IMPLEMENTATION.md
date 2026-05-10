# Subtitle Burning Implementation for Vertical Editor

## Overview
Added animated subtitle burning capability to the vertical video editor. Users can now burn TikTok-style animated subtitles into vertical videos with 1-4 words per group.

## Changes Made

### 1. Frontend (web/index.html)

#### Subtitle Options UI
- Added subtitle options section in the vertical editor controls
- Checkbox to enable/disable subtitle burning (checked by default)
- Dropdown to select subtitle style (bubble/popup/highlight/karaoke)
- JavaScript to toggle style dropdown visibility based on checkbox state

#### confirmPlacement() Function
- Modified to extract subtitle settings from UI:
  - `burn_subtitles`: boolean from checkbox
  - `subtitle_style`: string from dropdown
- Passes subtitle settings in the request body to backend

### 2. Backend (web_server.py)

#### Pipeline Execution (_run_pipeline_for_job)
- Added clip start/end times to result_clips dict
- Saves transcript.json to output directory after pipeline completes
- Transcript is saved for later use by vertical formatter

#### confirm_placement_endpoint
- Extracts subtitle settings from request body
- Passes settings to VerticalFormattingJob
- Includes clip timing information (start/end) in clips list

### 3. Vertical Formatter (pipeline/vertical_formatter.py)

#### apply_placement_to_clip() Method
- Added new parameters:
  - `transcript`: Full transcript with all segments (optional)
  - `clip_start`: Start time of clip in source video
  - `clip_end`: End time of clip in source video
  - `settings`: User settings dict (includes burn_subtitles, subtitle_style)

- Subtitle burning logic:
  1. Checks if `settings["burn_subtitles"]` is True
  2. Extracts SRT entries for clip's time range from transcript
  3. Uses word-level splitting (1-4 words per group) via `_word_level_entries()`
  4. Creates AnimatedSubtitleRenderer instance
  5. Generates ASS subtitle filter with specified style
  6. Appends subtitle filter to FFmpeg filter chain
  7. Updates output label to include subtitles

#### process_vertical_formatting_job() Function
- Loads transcript.json from output directory if subtitle burning is enabled
- Extracts clip timing information (start/end) from clip data
- Passes transcript, timing, and settings to apply_placement_to_clip()

## Subtitle Styles Available

1. **Bubble** (TikTok style): Bold thick outline with scale-pop animation
2. **Popup**: Fade-in + scale-in from 0% → 100%
3. **Highlight**: Active word group with highlight color and border box
4. **Karaoke**: Word-by-word color change using \k timing tags

## Technical Details

### Word Grouping
- Uses existing `_word_level_entries()` function from subtitle_generator.py
- Groups words into phrases of 1-4 words each
- Timestamps are adjusted to be relative to clip start

### Filter Chain
- Base filter: `[0:v]split=2[v1][v2];[v1]{gameplay_filter}[canvas];[v2]{facecam_filter}[facecam];[canvas][facecam]overlay={x}:{y}[with_facecam]`
- With subtitles: Appends `;[with_facecam]ass={ass_path}[final]`
- FFmpeg maps the final output label (either `[with_facecam]` or `[final]`)

### Transcript Storage
- Transcript is saved as `transcript.json` in the output directory
- Format: JSON serialization of Transcript.to_dict()
- Loaded by vertical formatter when subtitle burning is enabled

## Error Handling

- If transcript file is not found, logs warning and skips subtitles
- If transcript loading fails, logs warning and continues without subtitles
- If no SRT entries found for clip time range, logs info and skips subtitles
- Invalid subtitle style falls back to BUBBLE with warning

## Testing Recommendations

1. **Basic functionality**: Enable subtitles, select a style, confirm placement
2. **Style variations**: Test all 4 subtitle styles (bubble/popup/highlight/karaoke)
3. **Toggle behavior**: Disable subtitles checkbox, verify no subtitles burned
4. **Edge cases**: 
   - Clips with no overlapping transcript segments
   - Clips at the very start/end of video
   - Very short clips (< 1 second)
5. **Performance**: Test with multiple clips to verify progress tracking

## Files Modified

1. `web/index.html` - Added subtitle UI and modified confirmPlacement()
2. `web_server.py` - Added transcript saving and clip timing data
3. `pipeline/vertical_formatter.py` - Added subtitle burning logic

## Dependencies

- Existing AnimatedSubtitleRenderer class (pipeline/animated_subtitle_renderer.py)
- Existing subtitle_generator._word_level_entries() function
- FFmpeg with libass support (for ASS subtitle rendering)
