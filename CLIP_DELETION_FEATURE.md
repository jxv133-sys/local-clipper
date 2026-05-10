# Clip Deletion Feature

## Overview
Added the ability to delete individual clips from the web UI after they've been rendered. This helps users manage their output and remove unwanted clips.

## Features

### Delete Button
- Each clip now has a red "🗑️ Delete" button in the clip actions area
- Button appears alongside Download, Shorts, Preview, and "Why chosen" buttons
- Styled with error color (red) to indicate destructive action

### Confirmation Dialog
- Shows a confirmation dialog before deletion
- Warns user that deletion is permanent
- Lists what will be deleted: clip file, subtitles, report, thumbnail

### Files Deleted
When a clip is deleted, the following files are removed:
1. **Main clip file** (e.g., `clip_1_10s.mp4` or `volume_spike_2_45s.mp4`)
2. **Why chosen report** (e.g., `clip_1_10s_why_chosen.txt`)
3. **SRT subtitle file** (e.g., `clip_1_10s.srt`)
4. **Thumbnail** (e.g., `clip_1_10s_thumb.jpg`)
5. **Vertical shorts version** (if exists, e.g., `clip_1_10s_vertical.mp4`)

### UI Behavior
- Smooth fade-out animation when clip is deleted
- Clip card disappears with opacity and scale transition
- Clips list automatically refreshes to update indices
- Remaining clips are renumbered automatically

### Backend API

#### Endpoint
```
DELETE /api/jobs/<job_id>/clips/<clip_index>/delete
POST   /api/jobs/<job_id>/clips/<clip_index>/delete  (alternative)
```

#### Request
- `job_id`: The job ID containing the clip
- `clip_index`: Zero-based index of the clip to delete

#### Response
```json
{
  "success": true,
  "deleted_files": [
    "/path/to/clip_1_10s.mp4",
    "/path/to/clip_1_10s_why_chosen.txt",
    "/path/to/clip_1_10s.srt",
    "/path/to/clip_1_10s_thumb.jpg"
  ],
  "errors": [],
  "remaining_clips": 4
}
```

#### Error Handling
- Returns 404 if job not found
- Returns 400 if clip index is invalid
- Continues deletion even if some files fail
- Reports errors in the `errors` array
- Successfully deleted files are listed in `deleted_files`

### Implementation Details

#### Backend (`web_server.py`)
```python
@app.route("/api/jobs/<job_id>/clips/<int:clip_index>/delete", methods=["DELETE", "POST"])
def delete_clip(job_id: str, clip_index: int):
    # Validates job and clip index
    # Deletes all associated files
    # Removes clip from job.result_clips list
    # Returns summary of deleted files and errors
```

#### Frontend (`web/index.html`)
```javascript
async function deleteClip(jobId, clipIndex) {
    // Shows confirmation dialog
    // Calls DELETE endpoint
    // Animates clip removal
    // Refreshes clips list
}
```

## Use Cases

1. **Remove unwanted clips**: Delete clips that don't meet quality standards
2. **Free up disk space**: Remove clips after downloading the ones you want
3. **Curate results**: Keep only the best clips from a batch
4. **Iterative refinement**: Delete and re-run with different settings

## Safety Features

1. **Confirmation required**: User must confirm before deletion
2. **Clear warning**: Dialog explains that deletion is permanent
3. **Visual distinction**: Red delete button stands out from other actions
4. **Error reporting**: Shows which files failed to delete (if any)
5. **Graceful degradation**: Continues even if some files are missing

## Files Modified

- `web_server.py` - Added `/api/jobs/<job_id>/clips/<clip_index>/delete` endpoint
- `web/index.html` - Added delete button and `deleteClip()` JavaScript function

## Testing

To test the deletion feature:
1. Run the pipeline and generate some clips
2. Click the "🗑️ Delete" button on a clip
3. Confirm the deletion dialog
4. Verify the clip disappears from the UI
5. Check the output directory to confirm files were deleted
6. Verify remaining clips are renumbered correctly
