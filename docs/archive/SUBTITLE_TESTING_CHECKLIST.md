# Subtitle Burning Testing Checklist

## Prerequisites
- Server must be restarted after Python backend changes: `python3 web_server.py`
- Browser must be hard-refreshed after JavaScript changes: Cmd+Shift+R (Mac) or Ctrl+Shift+R (Windows/Linux)

## Test Cases

### 1. Basic Functionality
- [ ] Run pipeline to generate clips with transcript
- [ ] Click "Format to Vertical (9:16)" button
- [ ] Verify subtitle options section is visible
- [ ] Verify "Burn subtitles into video" checkbox is checked by default
- [ ] Verify subtitle style dropdown is visible and set to "Bubble (TikTok style)"

### 2. Checkbox Toggle
- [ ] Uncheck "Burn subtitles into video" checkbox
- [ ] Verify subtitle style dropdown is hidden
- [ ] Check the checkbox again
- [ ] Verify subtitle style dropdown is visible again

### 3. Style Selection
Test each style:
- [ ] Bubble (TikTok style) - Bold thick outline with scale-pop animation
- [ ] Popup (Zoom effect) - Fade-in + scale-in from 0% → 100%
- [ ] Highlight (Background) - Active word group with highlight color
- [ ] Karaoke (Word-by-word) - Word-by-word color change

### 4. Processing with Subtitles Enabled
- [ ] Enable subtitles, select "Bubble" style
- [ ] Adjust facecam placement
- [ ] Click "Confirm & Process All Clips"
- [ ] Verify progress UI shows
- [ ] Wait for processing to complete
- [ ] Download a processed clip
- [ ] Play the clip and verify:
  - [ ] Video is in 9:16 vertical format
  - [ ] Facecam is positioned correctly
  - [ ] Gameplay is cropped to 9:16 (not letterboxed)
  - [ ] Subtitles appear at the bottom
  - [ ] Subtitles are animated (bubble style)
  - [ ] Subtitles show 1-4 words per group
  - [ ] Subtitles are synchronized with audio

### 5. Processing with Subtitles Disabled
- [ ] Disable subtitles checkbox
- [ ] Adjust facecam placement
- [ ] Click "Confirm & Process All Clips"
- [ ] Wait for processing to complete
- [ ] Download a processed clip
- [ ] Play the clip and verify:
  - [ ] Video is in 9:16 vertical format
  - [ ] Facecam is positioned correctly
  - [ ] Gameplay is cropped to 9:16
  - [ ] NO subtitles appear

### 6. Different Subtitle Styles
For each style (bubble, popup, highlight, karaoke):
- [ ] Select the style
- [ ] Process clips
- [ ] Verify the style is applied correctly:
  - Bubble: Scale-pop animation on entry
  - Popup: Fade-in + scale-in effect
  - Highlight: Colored background box
  - Karaoke: Word-by-word color change

### 7. Edge Cases
- [ ] Test with a clip that has no overlapping transcript segments
  - Expected: No subtitles, but video processes successfully
- [ ] Test with a very short clip (< 2 seconds)
  - Expected: Subtitles appear but may be brief
- [ ] Test with a clip at the very start of the video
  - Expected: Subtitles start from 0:00
- [ ] Test with a clip at the very end of the video
  - Expected: Subtitles end at clip end time

### 8. Error Handling
- [ ] Delete transcript.json from output directory
- [ ] Try to process clips with subtitles enabled
- [ ] Expected: Warning logged, clips process without subtitles
- [ ] Verify clips are still formatted correctly (just no subtitles)

### 9. Console Logs
Check browser console for:
- [ ] "[Confirm] Subtitle settings: { burnSubtitles: true, subtitleStyle: 'bubble' }"
- [ ] No JavaScript errors

Check server logs for:
- [ ] "Loaded transcript with X segments for subtitle burning"
- [ ] "Burning X subtitle entries into clip with style bubble"
- [ ] "Encoded vertical clip successfully"
- [ ] No Python exceptions

### 10. Performance
- [ ] Process 5+ clips with subtitles enabled
- [ ] Verify progress tracking works correctly
- [ ] Verify ETA is reasonable
- [ ] Verify all clips complete successfully

## Known Limitations
- Requires transcript.json to exist in output directory
- Requires FFmpeg with libass support for ASS subtitle rendering
- Falls back to subtitles filter if libass not available
- Word grouping is fixed at 1-4 words per group

## Troubleshooting

### Subtitles not appearing
1. Check if transcript.json exists in output directory
2. Check server logs for "Loaded transcript" message
3. Check server logs for "Burning X subtitle entries" message
4. Verify FFmpeg has libass support: `ffmpeg -filters | grep ass`

### Subtitle style not working
1. Check browser console for subtitle settings log
2. Check server logs for subtitle style being used
3. Verify style name matches enum values (bubble/popup/highlight/karaoke)

### Processing fails
1. Check server logs for Python exceptions
2. Verify transcript.json is valid JSON
3. Verify clip timing information (start/end) is present
4. Check FFmpeg stderr output in logs
