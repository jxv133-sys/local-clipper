# Vertical Formatting - UX Issues & Solutions

## Issue Summary

You've identified two critical UX problems:

1. **Facecam auto-detection fails** - Users don't get clear feedback
2. **No progress visibility** - Users can't see when vertical formatting is complete

---

## Issue 1: Facecam Auto-Detection Failure

### Current Behavior
When facecam detection fails (returns `null`):
- System creates a **default 25%×25% region** in top-right corner
- Shows warning: "No facecam detected. Using default placement in top-right corner. Adjust manually as needed."
- Facecam box appears with **same purple color** as detected regions
- User might not realize this is a fallback, not a detection

### Problems
- ❌ Warning message easy to miss
- ❌ No visual distinction between detected vs default regions
- ❌ Users might think detection worked when it didn't
- ❌ No option to skip detection and place manually from start

### Proposed Solutions

#### Solution 1A: Visual Distinction for Default Regions
Change the facecam box color when using default region:

```javascript
// In detectFacecam() function, when no facecam detected:
if (!data.facecam_region) {
  // ... create default region ...
  editorState.facecamRegion.isDefault = true;  // Flag it
  
  showEditorStatus(
    '⚠️ AUTO-DETECTION FAILED: No facecam found. Using default placement (top-right corner). Please adjust manually to match your video.',
    'warning'
  );
  
  // Update facecam box with warning color
  updateFacecamBox();
  const box = document.getElementById('facecam-box-horizontal');
  box.style.borderColor = 'var(--warning)';  // Orange instead of purple
  box.style.boxShadow = '0 0 0 2px rgba(250, 179, 135, 0.3)';
}
```

#### Solution 1B: Add "Skip Detection" Button
Give users option to skip detection entirely:

```html
<!-- In editor controls section -->
<div class="editor-actions">
  <button id="skip-detect-btn" class="editor-btn editor-btn-secondary" onclick="skipDetection()">
    ⏭️ Skip & Place Manually
  </button>
  <button id="detect-btn" class="editor-btn editor-btn-primary" onclick="detectFacecam()">
    🔍 Auto-Detect Facecam
  </button>
</div>
```

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
    '📍 Manual placement mode. Adjust the facecam region using the sliders below to match your video.',
    'info'
  );
}
```

#### Solution 1C: Improved Status Messages
Make status messages more prominent and actionable:

```javascript
// When detection fails:
showEditorStatus(
  '⚠️ <strong>AUTO-DETECTION FAILED</strong><br>' +
  'No facecam found in the video. A default region has been placed in the top-right corner.<br>' +
  '<strong>Action Required:</strong> Use the sliders below to adjust the facecam position and size to match your video.',
  'warning'
);
```

---

## Issue 2: No Progress Visibility for Vertical Formatting

### Current Behavior
After clicking "Confirm & Process All Clips":
- Editor closes immediately
- Job runs in background
- **No progress indicator anywhere**
- **No way to see which clip is being processed**
- **No ETA shown**
- **No notification when complete**
- User has to manually check if clips are ready

### Problems
- ❌ User doesn't know if processing started
- ❌ Can't see progress (e.g., "3/5 clips done")
- ❌ Can't see which clip is currently processing
- ❌ No ETA for completion
- ❌ No notification when done
- ❌ Formatting jobs don't appear in main job queue

### Proposed Solutions

#### Solution 2A: Keep Editor Open with Progress Display

**Step 1: Modify confirmPlacement() to show progress instead of closing:**

```javascript
async function confirmPlacement() {
  if (!editorState.sessionId) return;
  
  const confirmBtn = document.getElementById('confirm-btn');
  confirmBtn.disabled = true;
  confirmBtn.textContent = '⏳ Starting...';
  
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
    
    // Store job ID
    editorState.formattingJobId = data.job_id;
    
    // Hide controls, show progress UI
    showFormattingProgressUI();
    
    // Start polling for progress
    pollFormattingProgress(data.job_id);
    
  } catch (err) {
    showEditorStatus(`Error: ${err.message}`, 'error');
    confirmBtn.disabled = false;
    confirmBtn.textContent = '✓ Confirm & Process All Clips';
  }
}
```

**Step 2: Add progress UI:**

```javascript
function showFormattingProgressUI() {
  // Hide controls section
  document.querySelector('.controls-section').style.display = 'none';
  
  // Add progress section
  const progressHTML = `
    <div id="formatting-progress-section" class="formatting-progress-section">
      <h3 style="text-align:center; color:var(--text); margin-bottom:20px;">
        🎬 Processing Clips to Vertical Format
      </h3>
      
      <div class="progress-bar-container">
        <div id="formatting-progress-bar" class="progress-bar-fill">0%</div>
      </div>
      
      <div class="formatting-status">
        <div class="status-row">
          <span class="status-label">Progress:</span>
          <span id="formatting-clip-count" class="status-value">0 / 0 clips</span>
        </div>
        <div class="status-row">
          <span class="status-label">Current:</span>
          <span id="formatting-current-clip" class="status-value">Starting...</span>
        </div>
        <div class="status-row">
          <span class="status-label">Time Remaining:</span>
          <span id="formatting-eta" class="status-value">Calculating...</span>
        </div>
        <div class="status-row">
          <span class="status-label">Elapsed:</span>
          <span id="formatting-elapsed" class="status-value">0s</span>
        </div>
      </div>
      
      <div id="formatting-errors" class="formatting-errors"></div>
      
      <div id="formatting-complete-actions" style="display:none; text-align:center; margin-top:20px;">
        <button class="editor-btn editor-btn-primary" onclick="closeVerticalEditor(); refreshJobs();">
          ✓ Done - Close Editor
        </button>
      </div>
    </div>
  `;
  
  document.querySelector('.controls-section').insertAdjacentHTML('afterend', progressHTML);
}
```

**Step 3: Add progress polling:**

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
        return; // Stop polling
      }
      
      // Continue polling
      setTimeout(poll, pollInterval);
      
    } catch (err) {
      console.error('[Formatting Progress] Error:', err);
      showEditorStatus(`Progress error: ${err.message}`, 'error');
      // Retry after delay
      setTimeout(poll, pollInterval * 2);
    }
  };
  
  poll(); // Start polling
}
```

**Step 4: Update progress UI:**

```javascript
function updateFormattingProgressUI(data) {
  // Update progress bar
  const progressBar = document.getElementById('formatting-progress-bar');
  const progressPct = Math.round(data.progress_pct);
  progressBar.style.width = `${progressPct}%`;
  progressBar.textContent = `${progressPct}%`;
  
  // Update clip count
  document.getElementById('formatting-clip-count').textContent = 
    `${data.clips_processed} / ${data.clips_total} clips`;
  
  // Update current clip
  if (data.current_clip) {
    document.getElementById('formatting-current-clip').textContent = data.current_clip;
  } else if (data.status === 'running') {
    document.getElementById('formatting-current-clip').textContent = 'Processing...';
  }
  
  // Update ETA
  if (data.eta_seconds > 0) {
    const minutes = Math.floor(data.eta_seconds / 60);
    const seconds = Math.floor(data.eta_seconds % 60);
    document.getElementById('formatting-eta').textContent = 
      `${minutes}m ${seconds}s`;
  } else {
    document.getElementById('formatting-eta').textContent = 'Calculating...';
  }
  
  // Update elapsed time
  const elapsed = Math.floor(data.elapsed_seconds);
  const elapsedMin = Math.floor(elapsed / 60);
  const elapsedSec = elapsed % 60;
  document.getElementById('formatting-elapsed').textContent = 
    `${elapsedMin}m ${elapsedSec}s`;
  
  // Show errors if any
  if (data.errors && data.errors.length > 0) {
    const errorsDiv = document.getElementById('formatting-errors');
    errorsDiv.innerHTML = '<h4 style="color:var(--error); margin-bottom:8px;">⚠️ Errors:</h4>' +
      data.errors.map(err => 
        `<div class="error-message">${esc(err)}</div>`
      ).join('');
  }
}
```

**Step 5: Handle completion:**

```javascript
function handleFormattingComplete(data) {
  const progressBar = document.getElementById('formatting-progress-bar');
  
  if (data.status === 'done') {
    progressBar.style.background = 'var(--success)';
    progressBar.style.width = '100%';
    progressBar.textContent = '100%';
    
    showEditorStatus(
      `✅ <strong>All clips processed successfully!</strong><br>` +
      `${data.clips_processed} clips reformatted to vertical (9:16).<br>` +
      `You can now download the clips from the results panel.`,
      'success'
    );
    
    // Show close button
    document.getElementById('formatting-complete-actions').style.display = '';
    
  } else if (data.status === 'failed') {
    progressBar.style.background = 'var(--error)';
    
    showEditorStatus(
      `❌ <strong>Processing failed.</strong><br>` +
      `${data.clips_processed} / ${data.clips_total} clips completed before failure.`,
      'error'
    );
    
    document.getElementById('formatting-complete-actions').style.display = '';
    
  } else if (data.status === 'cancelled') {
    progressBar.style.background = 'var(--warning)';
    
    showEditorStatus(
      `⚠️ <strong>Processing cancelled.</strong><br>` +
      `${data.clips_processed} / ${data.clips_total} clips completed.`,
      'warning'
    );
    
    document.getElementById('formatting-complete-actions').style.display = '';
  }
}
```

**Step 6: Add CSS:**

```css
.formatting-progress-section {
  padding: 20px;
  border-top: 1px solid var(--border);
  margin-top: 20px;
}

.progress-bar-container {
  width: 100%;
  height: 40px;
  background: var(--border);
  border-radius: var(--radius);
  overflow: hidden;
  margin-bottom: 20px;
}

.progress-bar-fill {
  height: 100%;
  width: 0%;
  background: var(--accent);
  transition: width 0.5s ease, background 0.3s ease;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #fff;
  font-weight: 700;
  font-size: 0.9rem;
}

.formatting-status {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-bottom: 20px;
}

.status-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 12px;
  background: var(--bg);
  border-radius: 5px;
}

.status-label {
  font-size: 0.85rem;
  color: var(--dim);
  font-weight: 600;
}

.status-value {
  font-size: 0.85rem;
  color: var(--text);
  font-family: "Cascadia Code", "Fira Code", monospace;
}

.formatting-errors {
  margin-top: 16px;
}

.formatting-errors h4 {
  font-size: 0.9rem;
  margin-bottom: 8px;
}

.error-message {
  background: rgba(243, 139, 168, 0.15);
  color: var(--error);
  padding: 8px 12px;
  border-radius: 5px;
  font-size: 0.8rem;
  margin-bottom: 6px;
  font-family: "Cascadia Code", "Fira Code", monospace;
}
```

#### Solution 2B: Add Formatting Jobs to Main Job Queue

**Backend: Add endpoint to list formatting jobs:**

```python
@app.route("/api/mini-editor/jobs", methods=["GET"])
def list_formatting_jobs():
    """Return all vertical formatting jobs ordered by creation time."""
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
        "elapsed_seconds": j.get_elapsed_time(),
        "eta_seconds": j.estimate_remaining_time(),
        "type": "formatting",
        "name": f"Vertical Formatting ({j.clips_total} clips)",
    } for j in jobs_snapshot]), 200
```

**Frontend: Merge formatting jobs into job list:**

```javascript
async function refreshJobs() {
  try {
    // Fetch regular jobs
    const res = await fetch('/api/jobs');
    const regularJobs = await res.json();
    
    // Fetch formatting jobs
    const formattingRes = await fetch('/api/mini-editor/jobs');
    const formattingJobs = await formattingRes.json();
    
    // Combine and sort by creation time
    const allJobs = [
      ...regularJobs.map(j => ({ ...j, type: 'regular' })),
      ...formattingJobs
    ].sort((a, b) => b.created_at - a.created_at);
    
    renderJobList(allJobs);
    
  } catch (err) {
    console.error('[Jobs] Error refreshing:', err);
  }
}
```

**Frontend: Render formatting jobs with special styling:**

```javascript
function renderJobItem(job) {
  if (job.type === 'formatting') {
    return `
      <div class="job-item ${activeJobId === job.job_id ? 'active' : ''}" 
           onclick="selectJob('${job.job_id}', 'formatting')">
        <div class="job-status-dot dot-${job.status}"></div>
        <div class="job-name">
          🎬 ${job.name}
          ${job.status === 'running' ? `(${job.clips_processed}/${job.clips_total})` : ''}
        </div>
        <span class="job-badge badge-${job.status}">${job.status}</span>
      </div>
    `;
  } else {
    // Regular job rendering...
  }
}
```

---

## Implementation Priority

### High Priority (Implement First)
1. ✅ **Solution 2A: Progress tracking in editor** - Most critical UX issue
2. ✅ **Solution 1A: Visual distinction for default regions** - Quick win

### Medium Priority
3. ✅ **Solution 1B: Skip detection button** - Nice to have
4. ✅ **Solution 2B: Formatting jobs in main queue** - Better visibility

### Low Priority
5. ✅ **Solution 1C: Improved status messages** - Polish

---

## Quick Implementation Guide

### Minimal Fix (30 minutes)
1. Add progress UI to editor (Solution 2A steps 1-6)
2. Change facecam box color for default regions (Solution 1A)

### Complete Fix (2 hours)
1. Implement all of Solution 2A (progress tracking)
2. Implement all of Solution 1A (visual distinction)
3. Add Solution 1B (skip detection button)
4. Add Solution 2B (formatting jobs in queue)

---

## Testing Checklist

### Facecam Detection
- [ ] Test with video that has facecam (should detect)
- [ ] Test with video without facecam (should show default with orange box)
- [ ] Test "Skip Detection" button
- [ ] Verify default region is clearly indicated
- [ ] Verify detected region shows confidence badge

### Progress Tracking
- [ ] Test with 1 clip (verify progress updates)
- [ ] Test with 5 clips (verify ETA calculation)
- [ ] Test with 10+ clips (verify performance)
- [ ] Verify progress bar animates smoothly
- [ ] Verify current clip name updates
- [ ] Verify ETA updates correctly
- [ ] Verify elapsed time updates
- [ ] Test error display when clip fails
- [ ] Test completion notification
- [ ] Verify "Done" button appears and works
- [ ] Verify formatting jobs appear in main queue

