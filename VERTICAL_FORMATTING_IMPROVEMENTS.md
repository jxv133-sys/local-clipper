# Vertical Formatting - User Experience Improvements

## Issues Identified

### Issue 1: Facecam Auto-Detection Failure Not Clear
**Problem:** When facecam detection fails, the system creates a default region but doesn't clearly communicate this to the user.

**Current Behavior:**
- Detection fails silently
- Default 25%×25% region created in top-right corner
- Warning message shown but easy to miss
- User might not realize they need to adjust manually

**Proposed Solution:**
1. **Clearer visual feedback** when using default region
2. **Highlight the default region** with different color/style
3. **Show prominent message** explaining manual adjustment is needed
4. **Add "Skip Detection" button** for users who want to place manually from the start

### Issue 2: No Progress Visibility for Vertical Formatting Jobs
**Problem:** After confirming vertical formatting, users have no way to track progress.

**Current Behavior:**
- Job runs in background
- No progress indicator
- No way to see which clip is being processed
- No ETA shown
- Editor closes immediately after confirmation
- User doesn't know when clips are ready

**Proposed Solution:**
1. **Add formatting jobs to main job queue** with visual indicator
2. **Show progress bar** with clip count (e.g., "3/5 clips processed")
3. **Display current clip** being processed
4. **Show ETA** based on average processing time
5. **Keep editor open** with progress display until job completes
6. **Add notification** when job finishes

---

## Implementation Plan

### Part 1: Improve Facecam Detection Feedback

#### Frontend Changes (`web/index.html`)

**1. Add visual distinction for default vs detected regions:**
```javascript
// In detectFacecam() function
if (!data.facecam_region) {
  // Create default region
  editorState.facecamRegion = {
    x: editorState.sourceWidth - defaultWidth - 10,
    y: 10,
    width: defaultWidth,
    height: defaultHeight,
    corner: 'top-right',
    confidence: 0.0,
    isDefault: true  // NEW: Flag to indicate this is a default region
  };
  
  showEditorStatus(
    '⚠️ No facecam detected. Using default placement. Please adjust manually to match your video.',
    'warning'
  );
  
  // Highlight the facecam box with warning color
  document.getElementById('facecam-box-horizontal').style.borderColor = 'var(--warning)';
}
```

**2. Add "Skip Detection" button:**
```html
<div class="editor-actions">
  <button id="skip-detect-btn" class="editor-btn editor-btn-secondary" onclick="skipDetection()">
    ⏭️ Skip Detection & Place Manually
  </button>
  <button id="detect-btn" class="editor-btn editor-btn-primary" onclick="detectFacecam()">
    🔍 Auto-Detect Facecam
  </button>
</div>
```

**3. Add manual placement mode:**
```javascript
function skipDetection() {
  // Create default region without trying detection
  const defaultWidth = Math.floor(editorState.sourceWidth * 0.25);
  const defaultHeight = Math.floor(editorState.sourceHeight * 0.25);
  
  editorState.facecamRegion = {
    x: editorState.sourceWidth - defaultWidth - 10,
    y: 10,
    width: defaultWidth,
    height: defaultHeight,
    corner: 'top-right',
    confidence: 0.0,
    isDefault: true
  };
  
  updateFacecamControls();
  updateFacecamBox();
  generatePreview();
  
  document.getElementById('confirm-btn').disabled = false;
  
  showEditorStatus(
    'Manual placement mode. Adjust the facecam region using the sliders below.',
    'info'
  );
}
```

### Part 2: Add Progress Tracking for Vertical Formatting

#### Frontend Changes (`web/index.html`)

**1. Modify confirmPlacement() to show progress instead of closing:**
```javascript
async function confirmPlacement() {
  if (!editorState.sessionId) return;
  
  const confirmBtn = document.getElementById('confirm-btn');
  confirmBtn.disabled = true;
  confirmBtn.textContent = '⏳ Processing...';
  
  try {
    const res = await fetch('/api/mini-editor/confirm', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: editorState.sessionId,
        facecam_region: editorState.facecamRegion,
        settings: {
          create_backup: true,
          replace_originals: true,
        },
      }),
    });
    const data = await res.json();
    
    if (!res.ok) {
      throw new Error(data.error || 'Confirmation failed');
    }
    
    // Store job ID and start polling for progress
    editorState.formattingJobId = data.job_id;
    
    // Show progress UI
    showFormattingProgress();
    
    // Start polling
    pollFormattingProgress(data.job_id);
    
  } catch (err) {
    showEditorStatus(`Error: ${err.message}`, 'error');
    confirmBtn.disabled = false;
    confirmBtn.textContent = '✓ Confirm & Process All Clips';
  }
}
```

**2. Add progress display UI:**
```javascript
function showFormattingProgress() {
  // Hide controls section
  document.querySelector('.controls-section').style.display = 'none';
  
  // Show progress section
  const progressHTML = `
    <div id="formatting-progress" class="formatting-progress">
      <h3>Processing Clips...</h3>
      <div class="progress-bar-container">
        <div id="formatting-progress-bar" class="progress-bar-fill"></div>
      </div>
      <div id="formatting-status" class="formatting-status">
        <span id="formatting-clip-count">0 / 0 clips processed</span>
        <span id="formatting-current-clip"></span>
        <span id="formatting-eta"></span>
      </div>
      <div id="formatting-errors" class="formatting-errors"></div>
    </div>
  `;
  
  document.querySelector('.editor-body').insertAdjacentHTML('beforeend', progressHTML);
}
```

**3. Add progress polling:**
```javascript
async function pollFormattingProgress(jobId) {
  const pollInterval = 1000; // Poll every second
  
  const poll = async () => {
    try {
      const res = await fetch(`/api/mini-editor/job/${jobId}/progress`);
      const data = await res.json();
      
      if (!res.ok) {
        throw new Error(data.error || 'Failed to get progress');
      }
      
      // Update progress UI
      updateFormattingProgressUI(data);
      
      // Check if done
      if (data.status === 'done' || data.status === 'failed' || data.status === 'cancelled') {
        handleFormattingComplete(data);
        return;
      }
      
      // Continue polling
      setTimeout(poll, pollInterval);
      
    } catch (err) {
      console.error('[Formatting Progress] Error:', err);
      showEditorStatus(`Progress error: ${err.message}`, 'error');
    }
  };
  
  poll();
}
```

**4. Update progress UI:**
```javascript
function updateFormattingProgressUI(data) {
  // Update progress bar
  const progressBar = document.getElementById('formatting-progress-bar');
  progressBar.style.width = `${data.progress_pct}%`;
  
  // Update clip count
  document.getElementById('formatting-clip-count').textContent = 
    `${data.clips_processed} / ${data.clips_total} clips processed`;
  
  // Update current clip
  if (data.current_clip) {
    document.getElementById('formatting-current-clip').textContent = 
      `Processing: ${data.current_clip}`;
  }
  
  // Update ETA
  if (data.eta_seconds > 0) {
    const minutes = Math.floor(data.eta_seconds / 60);
    const seconds = Math.floor(data.eta_seconds % 60);
    document.getElementById('formatting-eta').textContent = 
      `ETA: ${minutes}m ${seconds}s`;
  }
  
  // Show errors if any
  if (data.errors && data.errors.length > 0) {
    const errorsDiv = document.getElementById('formatting-errors');
    errorsDiv.innerHTML = data.errors.map(err => 
      `<div class="error-message">⚠️ ${err}</div>`
    ).join('');
  }
}
```

**5. Handle completion:**
```javascript
function handleFormattingComplete(data) {
  if (data.status === 'done') {
    showEditorStatus(
      `✅ All clips processed successfully! ${data.clips_processed} clips reformatted to vertical.`,
      'success'
    );
    
    // Show close button
    document.getElementById('formatting-progress').innerHTML += `
      <button class="editor-btn editor-btn-primary" onclick="closeVerticalEditor(); refreshJobs();">
        ✓ Done - Close Editor
      </button>
    `;
  } else if (data.status === 'failed') {
    showEditorStatus(
      `❌ Processing failed. ${data.clips_processed} / ${data.clips_total} clips completed.`,
      'error'
    );
  } else if (data.status === 'cancelled') {
    showEditorStatus(
      `⚠️ Processing cancelled. ${data.clips_processed} / ${data.clips_total} clips completed.`,
      'warning'
    );
  }
}
```

**6. Add CSS for progress UI:**
```css
.formatting-progress {
  padding: 20px;
  text-align: center;
}

.formatting-progress h3 {
  font-size: 1.1rem;
  color: var(--text);
  margin-bottom: 20px;
}

.progress-bar-container {
  width: 100%;
  height: 30px;
  background: var(--border);
  border-radius: var(--radius);
  overflow: hidden;
  margin-bottom: 16px;
}

.progress-bar-fill {
  height: 100%;
  width: 0%;
  background: var(--accent);
  transition: width 0.3s ease;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #fff;
  font-weight: 600;
  font-size: 0.85rem;
}

.formatting-status {
  display: flex;
  flex-direction: column;
  gap: 8px;
  font-size: 0.85rem;
  color: var(--dim);
}

.formatting-status span {
  display: block;
}

.formatting-errors {
  margin-top: 16px;
}

.error-message {
  background: rgba(243, 139, 168, 0.15);
  color: var(--error);
  padding: 8px 12px;
  border-radius: var(--radius);
  font-size: 0.8rem;
  margin-bottom: 8px;
}
```

### Part 3: Add Formatting Jobs to Main Job Queue

**1. Modify job list to show formatting jobs:**
```javascript
async function refreshJobs() {
  try {
    // Fetch regular jobs
    const res = await fetch('/api/jobs');
    const jobs = await res.json();
    
    // Fetch formatting jobs
    const formattingRes = await fetch('/api/mini-editor/jobs');
    const formattingJobs = await formattingRes.json();
    
    // Combine and sort by creation time
    const allJobs = [
      ...jobs.map(j => ({ ...j, type: 'regular' })),
      ...formattingJobs.map(j => ({ ...j, type: 'formatting' }))
    ].sort((a, b) => b.created_at - a.created_at);
    
    renderJobList(allJobs);
    
  } catch (err) {
    console.error('[Jobs] Error refreshing:', err);
  }
}
```

**2. Add backend endpoint to list formatting jobs:**
```python
@app.route("/api/mini-editor/jobs", methods=["GET"])
def list_formatting_jobs():
    """Return all vertical formatting jobs."""
    with _formatting_jobs_lock:
        jobs_snapshot = list(_formatting_jobs.values())
    
    jobs_snapshot.sort(key=lambda j: j.created_at, reverse=True)
    
    return jsonify([{
        "job_id": j.job_id,
        "status": j.status,
        "clips_processed": j.clips_processed,
        "clips_total": j.clips_total,
        "progress_pct": j.get_progress_percentage(),
        "created_at": j.created_at,
        "type": "formatting",
    } for j in jobs_snapshot])
```

---

## Summary of Changes

### Files to Modify:

1. **`web/index.html`**:
   - Improve facecam detection feedback
   - Add "Skip Detection" button
   - Add progress tracking UI
   - Add progress polling logic
   - Add CSS for progress display

2. **`web_server.py`**:
   - Add `/api/mini-editor/jobs` endpoint to list formatting jobs
   - Ensure progress endpoint is accessible

### User Experience Improvements:

**Before:**
- ❌ Facecam detection failure unclear
- ❌ No progress visibility
- ❌ Editor closes immediately
- ❌ No way to know when clips are ready

**After:**
- ✅ Clear visual feedback when detection fails
- ✅ Option to skip detection and place manually
- ✅ Real-time progress tracking
- ✅ Clip-by-clip status updates
- ✅ ETA display
- ✅ Error reporting
- ✅ Completion notification
- ✅ Formatting jobs visible in main job queue

---

## Testing Checklist

- [ ] Test facecam detection failure scenario
- [ ] Verify default region is clearly indicated
- [ ] Test "Skip Detection" button
- [ ] Test progress tracking during formatting
- [ ] Verify progress bar updates correctly
- [ ] Test ETA calculation accuracy
- [ ] Test error display when clip fails
- [ ] Test completion notification
- [ ] Verify formatting jobs appear in job queue
- [ ] Test cancellation of formatting job
- [ ] Test with 1 clip, 5 clips, and 10+ clips

