# Design Document: Mini Video Editor

## Overview

The Mini Video Editor is a web-based GUI component that enables users to format horizontally-recorded video clips into vertical (9:16) format with repositionable webcam overlays. It integrates seamlessly with the existing clip generation pipeline, appearing as an optional post-processing step after clips are generated.

**Key Design Principles:**
- **Reuse existing components**: Leverage FacecamRelocator, FrameReformatter, and FFmpeg filter generation from the pipeline
- **Real-time feedback**: Provide instant preview updates as users adjust facecam placement
- **Graceful degradation**: Offer fallback options (manual selection, blurred fill) when auto-detection fails
- **Batch consistency**: Apply the same placement uniformly across all clips in a batch
- **Non-invasive**: Optional feature that doesn't interfere with the existing pipeline

---

## Architecture

### System Context

```
┌─────────────────────────────────────────────────────────────┐
│                    Existing Pipeline                         │
│  (Transcription → Scoring → Clip Selection → Extraction)    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
            ┌────────────────────────┐
            │  Format to Vertical?   │
            │  (Prompt to User)      │
            └────────┬───────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
    ┌─────────┐          ┌──────────┐
    │Mini     │          │Skip      │
    │Editor   │          │(Original)│
    └────┬────┘          └──────────┘
         │
         ├─ Detect Facecam (FacecamRelocator)
         ├─ Display Preview (Canvas Layout)
         ├─ Allow Adjustment (UI Controls)
         ├─ Confirm Placement
         │
         ▼
    ┌──────────────────────┐
    │ Vertical Formatter   │
    │ (Batch Processing)   │
    └────┬─────────────────┘
         │
         ├─ Apply placement to all clips
         ├─ Generate FFmpeg filters
         ├─ Encode output files
         ├─ Replace originals (optional backup)
         │
         ▼
    ┌──────────────────┐
    │ Vertical Clips   │
    │ (9:16 format)    │
    └──────────────────┘
```

### Component Layers

**Frontend (Web GUI)**
- HTML/CSS/JavaScript interface
- Real-time preview rendering
- Adjustment controls (sliders, input fields)
- Progress indicators and status displays

**Backend API**
- REST endpoints for detection, preview, confirmation
- WebSocket for long-running operations (optional)
- Job queue for batch processing

**Processing Layer**
- FacecamRelocator: Detects facecam region using FFmpeg cropdetect
- FrameReformatter: Computes canvas layout and builds FFmpeg filters
- VerticalFormatter: Applies placement to all clips and generates output

**Data Layer**
- EditorSession: Tracks user state during editing
- FacecamRegion: Detected or manually-selected facecam coordinates
- CanvasLayout: Computed layout for 9:16 canvas

---

## Components and Interfaces

### 1. Frontend Components

#### Mini Editor Interface
- **Horizontal Preview Pane**: Displays source clip at original aspect ratio with facecam region highlighted
- **Vertical Preview Pane**: Shows 9:16 canvas with facecam in top region and gameplay in bottom region
- **Adjustment Controls**: Sliders/inputs for X, Y, width, height of facecam region
- **Confidence Display**: Shows auto-detection confidence score (0.0–1.0)
- **Action Buttons**: Confirm, Cancel, Undo, Redo
- **Settings Panel**: Optional backup, naming convention, replacement options
- **Progress Indicator**: Shows detection/processing status

#### Responsive Design
- Desktop-first layout (minimum 1024×768)
- Horizontal layout: preview panes side-by-side, controls below
- Vertical layout: preview panes stacked, controls below (for smaller screens)
- Touch-friendly controls for tablet support

### 2. Backend API Endpoints

```
POST /api/mini-editor/session
  Request: { clip_batch_id, reference_clip_path }
  Response: { session_id, clips: [...] }
  Purpose: Initialize editor session

POST /api/mini-editor/detect
  Request: { session_id, clip_path }
  Response: { facecam_region, confidence, canvas_layout }
  Purpose: Run facecam detection on a clip

POST /api/mini-editor/preview
  Request: { session_id, facecam_region }
  Response: { preview_image_url }
  Purpose: Generate preview image with current placement

POST /api/mini-editor/confirm
  Request: { session_id, facecam_region, settings }
  Response: { job_id, status }
  Purpose: Confirm placement and start batch processing

POST /api/mini-editor/cancel
  Request: { session_id }
  Response: { status }
  Purpose: Cancel editor session without processing

GET /api/mini-editor/progress/:job_id
  Response: { status, clips_processed, clips_total, current_clip, eta_seconds }
  Purpose: Poll for batch processing progress

POST /api/mini-editor/undo
  Request: { session_id }
  Response: { facecam_region }
  Purpose: Undo last adjustment

POST /api/mini-editor/redo
  Request: { session_id }
  Response: { facecam_region }
  Purpose: Redo last undone adjustment
```

### 3. Data Models

#### EditorSession
```python
@dataclass
class EditorSession:
    session_id: str                    # UUID
    clip_batch_id: str                 # Reference to batch
    reference_clip_path: str           # Path to first clip
    reference_resolution: tuple        # (width, height)
    facecam_region: FacecamRegion      # Current placement
    canvas_layout: CanvasLayout        # Computed layout
    undo_history: list[FacecamRegion]  # For undo/redo
    redo_history: list[FacecamRegion]
    settings: dict                     # User preferences
    created_at: float                  # Timestamp
    expires_at: float                  # Session expiry
```

#### FacecamRegion (reused from pipeline.models)
```python
@dataclass
class FacecamRegion:
    x: int                  # Left edge in source frame pixels
    y: int                  # Top edge in source frame pixels
    width: int              # Crop width
    height: int             # Crop height
    corner: str             # "top-left" | "top-right" | "bottom-left" | "bottom-right"
    confidence: float       # 0.0–1.0, detection confidence
```

#### CanvasLayout (reused from pipeline.models)
```python
@dataclass
class CanvasLayout:
    canvas_width: int       # 1080
    canvas_height: int      # 1920
    facecam_x: int          # 0
    facecam_y: int          # 0
    facecam_width: int      # 1080
    facecam_height: int     # ~672 (35% of 1920)
    gameplay_x: int         # 0
    gameplay_y: int         # ~672
    gameplay_width: int     # 1080
    gameplay_height: int    # ~1248 (65% of 1920)
```

#### VerticalFormattingJob
```python
@dataclass
class VerticalFormattingJob:
    job_id: str                        # UUID
    session_id: str                    # Reference to editor session
    clip_batch_id: str                 # Batch being processed
    facecam_region: FacecamRegion      # Confirmed placement
    canvas_layout: CanvasLayout        # Layout to apply
    settings: dict                     # User settings
    clips: list[dict]                  # [{path, name, resolution}]
    status: str                        # "queued" | "running" | "done" | "failed"
    clips_processed: int               # Progress counter
    clips_total: int                   # Total clips
    current_clip: str                  # Currently processing
    errors: list[str]                  # Error messages
    output_dir: str                    # Where to save results
    created_at: float                  # Timestamp
    started_at: float                  # When processing began
    completed_at: float                # When processing finished
```

---

## Data Models

### Core Data Structures

**FacecamRegion** (from pipeline.models)
- Represents the detected or manually-selected facecam region in source frame coordinates
- Includes corner classification and confidence score
- Validated against area fraction constraints (4%–30% of frame)

**CanvasLayout** (from pipeline.models)
- Describes the 9:16 vertical canvas layout
- Computed from config settings (canvas dimensions, facecam_top_fraction)
- Defines regions for facecam (top) and gameplay (bottom)

**EditorSession**
- Tracks user state during editing session
- Maintains undo/redo history for adjustments
- Stores user preferences and settings
- Expires after inactivity (configurable, default 30 minutes)

**VerticalFormattingJob**
- Represents a batch processing job
- Tracks progress and errors
- Stores confirmed placement and settings
- Enables progress polling and cancellation

---

## UI/UX Design

### Layout

**Main Editor View**
```
┌─────────────────────────────────────────────────────────────┐
│ Mini Video Editor                                    [×]     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐  ┌──────────────────┐                 │
│  │  Horizontal      │  │  Vertical        │                 │
│  │  Preview         │  │  Preview         │                 │
│  │  (16:9)          │  │  (9:16)          │                 │
│  │                  │  │                  │                 │
│  │  [Facecam Box]   │  │  ┌────────────┐  │                 │
│  │                  │  │  │ Facecam    │  │                 │
│  │                  │  │  │ (Top)      │  │                 │
│  │                  │  │  ├────────────┤  │                 │
│  │                  │  │  │ Gameplay   │  │                 │
│  │                  │  │  │ (Bottom)   │  │                 │
│  │                  │  │  └────────────┘  │                 │
│  └──────────────────┘  └──────────────────┘                 │
│                                                               │
│  Confidence: 0.87 ████████░                                  │
│                                                               │
│  Adjustment Controls:                                        │
│  X Position:  [─────●─────] 150 px                          │
│  Y Position:  [────●──────] 100 px                          │
│  Width:       [──────●────] 400 px                          │
│  Height:      [─────●─────] 300 px                          │
│                                                               │
│  [⟲ Undo] [⟳ Redo] [⚙ Settings] [✓ Confirm] [✕ Cancel]    │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

**Settings Panel**
```
┌─────────────────────────────────────┐
│ Settings                        [×]  │
├─────────────────────────────────────┤
│                                     │
│ ☑ Create backup of original clips   │
│                                     │
│ Output naming:                      │
│ ○ Append "_vertical" suffix         │
│ ○ Prepend "vertical_" prefix        │
│ ○ Custom: [________________]        │
│                                     │
│ ☑ Replace original clips            │
│ ○ Save to separate directory        │
│                                     │
│ [Cancel] [Save Settings]            │
│                                     │
└─────────────────────────────────────┘
```

**Progress View**
```
┌─────────────────────────────────────────────────────────────┐
│ Processing Vertical Clips                                   │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│ Progress: 7 of 10 clips                                      │
│ ███████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
│                                                               │
│ Current: clip_5_highlight.mp4                               │
│ Status: Encoding...                                         │
│                                                               │
│ Estimated time remaining: 2 minutes 15 seconds              │
│                                                               │
│ [⏸ Pause] [✕ Cancel]                                        │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Interaction Patterns

**Adjustment Workflow**
1. User opens Mini Editor → Auto-detection runs in background
2. Detection completes → Facecam region displayed in both previews
3. User adjusts sliders → Preview updates in real-time (< 500ms)
4. User clicks Confirm → Validation runs, batch processing starts
5. Progress displayed → User can pause/cancel at any time
6. Processing completes → Summary shown, clips replaced

**Error Handling Workflow**
1. Detection fails → Error message displayed with reason
2. User offered options:
   - Manual selection (draw bounding box)
   - Fallback fill (blurred gameplay)
   - Cancel and skip formatting
3. User chooses option → Editor switches to appropriate mode
4. User confirms → Processing proceeds with chosen option

**Undo/Redo Workflow**
1. User adjusts facecam region → Adjustment added to undo history
2. User clicks Undo → Previous state restored, adjustment moved to redo history
3. User clicks Redo → Adjustment reapplied, moved back to undo history
4. User confirms placement → Undo/redo history cleared

---

## Integration Points

### Reused Components

**FacecamRelocator** (pipeline.facecam_relocator)
- `detect_facecam(clip_path, frame_width, frame_height, config)` → FacecamRegion
- Uses FFmpeg cropdetect to find facecam region
- Validates area fraction and classifies corner
- Computes confidence score

**FrameReformatter** (pipeline.frame_reformatter)
- `compute_canvas_layout(config)` → CanvasLayout
- `build_canvas_filter(src_width, src_height, layout)` → FilterFragment
- Computes 9:16 canvas layout from config
- Builds FFmpeg filter for gameplay region scaling/padding

**Config Object** (config.py)
- `shorts_width`, `shorts_height`: Canvas dimensions (1080×1920)
- `facecam_top_fraction`: Facecam region height fraction (0.35)
- `facecam_min_area_fraction`, `facecam_max_area_fraction`: Validation bounds
- `facecam_sample_duration`: Duration to sample for detection (10s)

### FFmpeg Filter Generation

The Mini Editor reuses FFmpeg filter generation from the pipeline:

```
# Canvas filter (from FrameReformatter)
[0:v]scale=1080:1248,pad=1080:1920:0:672:black[canvas]

# Facecam filter (from FacecamRelocator)
[0:v]crop=400:300:150:100,scale=1080:672[facecam_scaled];
[canvas][facecam_scaled]overlay=0:0[with_facecam]

# Final output
[with_facecam][0:a]concat=n=1:v=1:a=1[v][a]
```

### Web Server Integration

**Route Registration**
```python
# In web_server.py
app.route('/mini-editor', methods=['GET'])(serve_mini_editor)
app.route('/api/mini-editor/session', methods=['POST'])(create_session)
app.route('/api/mini-editor/detect', methods=['POST'])(detect_facecam)
app.route('/api/mini-editor/preview', methods=['POST'])(generate_preview)
app.route('/api/mini-editor/confirm', methods=['POST'])(confirm_placement)
# ... additional routes
```

**Job Queue Integration**
- Reuse existing job queue from web_server.py
- VerticalFormattingJob added to queue after confirmation
- Progress polling via existing job status endpoint

---

## Error Handling

### Detection Failures

**Scenario 1: No facecam detected**
- Reason: Facecam area outside 4%–30% bounds
- User options:
  - Manual selection (draw bounding box)
  - Fallback fill (blurred gameplay)
  - Cancel formatting

**Scenario 2: Multiple candidate regions**
- Reason: Ambiguous detection (multiple regions with similar frequency)
- Behavior: Use most-common region, display lower confidence score
- User can manually adjust if needed

**Scenario 3: Detection timeout**
- Reason: FFmpeg cropdetect takes > 30 seconds
- Behavior: Cancel detection, offer manual selection or fallback
- Log timeout for debugging

### Processing Errors

**Scenario 1: Clip encoding fails**
- Reason: Codec error, disk full, permission denied
- Behavior: Log error, skip clip, continue with next
- Display error in progress view with clip name and reason

**Scenario 2: Batch processing cancelled**
- Reason: User clicks Cancel button
- Behavior: Stop processing, preserve already-processed clips
- Offer option to resume or discard

**Scenario 3: Insufficient disk space**
- Reason: Output directory full
- Behavior: Detect before processing, display error
- Suggest freeing space or changing output directory

### Network Errors

**Scenario 1: API request timeout**
- Reason: Backend unresponsive
- Behavior: Display error, offer retry
- Implement exponential backoff for retries

**Scenario 2: WebSocket disconnection**
- Reason: Network interruption
- Behavior: Attempt reconnection, buffer updates
- Display connection status to user

### User Input Validation

**Invalid facecam region**
- Region extends beyond frame bounds
- Region area outside 4%–30% bounds
- Behavior: Prevent confirmation, display error message

**Invalid settings**
- Output directory doesn't exist or not writable
- Invalid naming convention
- Behavior: Validate before applying, display error

---

## Performance Considerations

### Real-Time Preview

**Target**: Update preview within 500ms of adjustment

**Implementation**:
- Debounce slider inputs (100ms)
- Generate preview image server-side (cached)
- Use canvas rendering for client-side preview
- Lazy-load preview images

**Optimization**:
- Cache preview images for common placements
- Use lower resolution for preview (e.g., 540×960)
- Render preview asynchronously

### Facecam Detection

**Target**: Complete detection within 30 seconds for typical 30-second clip

**Implementation**:
- Sample first 10 seconds of clip (configurable)
- Run FFmpeg cropdetect with reasonable parameters
- Parse output incrementally
- Cache detection results per clip

**Optimization**:
- Use hardware acceleration if available
- Parallelize detection for multiple clips (if batch)
- Implement timeout with fallback

### Batch Processing

**Target**: Process each clip within 2–5 minutes

**Implementation**:
- Queue clips for sequential processing
- Display progress in real-time
- Allow pause/cancel at any time
- Implement checkpointing for resume capability

**Optimization**:
- Use hardware encoding (H.264, H.265) if available
- Parallelize encoding for multiple clips (if resources allow)
- Implement adaptive bitrate based on source quality

### Memory Management

**Considerations**:
- Keep only current clip in memory during processing
- Stream video data rather than loading entire file
- Limit undo/redo history to last 20 adjustments
- Expire old editor sessions after 30 minutes

---

## Sequence Diagrams

### Opening Editor and Detecting Facecam

```
User                Mini Editor         Backend              Pipeline
  │                     │                  │                    │
  ├─ Click "Format"────>│                  │                    │
  │                     ├─ POST /session──>│                    │
  │                     │                  ├─ Create session    │
  │                     │<─ session_id ────┤                    │
  │                     │                  │                    │
  │                     ├─ POST /detect───>│                    │
  │                     │                  ├─ FacecamRelocator──>
  │                     │                  │  .detect_facecam() │
  │                     │                  │<─ FacecamRegion ───┤
  │                     │<─ region + conf ─┤                    │
  │                     │                  │                    │
  │<─ Display preview ──┤                  │                    │
  │   with facecam      │                  │                    │
  │                     │                  │                    │
```

### Adjusting Placement and Previewing

```
User                Mini Editor         Backend              Pipeline
  │                     │                  │                    │
  ├─ Adjust slider ────>│                  │                    │
  │                     ├─ Debounce (100ms)                     │
  │                     ├─ POST /preview──>│                    │
  │                     │                  ├─ Generate preview  │
  │                     │<─ preview_url ───┤                    │
  │                     │                  │                    │
  │<─ Update preview ───┤                  │                    │
  │   (< 500ms)         │                  │                    │
  │                     │                  │                    │
  ├─ Adjust again ─────>│                  │                    │
  │                     ├─ Debounce (100ms)                     │
  │                     ├─ POST /preview──>│                    │
  │                     │<─ preview_url ───┤                    │
  │<─ Update preview ───┤                  │                    │
  │                     │                  │                    │
```

### Confirming and Processing Clips

```
User                Mini Editor         Backend              Pipeline
  │                     │                  │                    │
  ├─ Click Confirm ────>│                  │                    │
  │                     ├─ Validate region │                    │
  │                     ├─ POST /confirm──>│                    │
  │                     │                  ├─ Create job       │
  │                     │                  ├─ Queue job        │
  │                     │<─ job_id ────────┤                    │
  │                     │                  │                    │
  │<─ Show progress ────┤                  │                    │
  │                     ├─ Poll /progress ─┤                    │
  │                     │<─ status ────────┤                    │
  │                     │                  │                    │
  │                     │                  ├─ Process clip 1 ──>
  │                     │                  │  (encode, etc.)    │
  │                     │                  │<─ Done ────────────┤
  │                     │                  │                    │
  │<─ Update progress ──┤                  │                    │
  │   (1 of 10)         │                  │                    │
  │                     │                  │                    │
  │                     │                  ├─ Process clip 2 ──>
  │                     │                  │  ...               │
  │                     │                  │                    │
  │<─ Final summary ────┤                  │                    │
  │   (10 clips done)   │                  │                    │
  │                     │                  │                    │
```

---

## Testing Strategy

### Unit Tests

**FacecamRegion Validation**
- Test area fraction validation (4%–30% bounds)
- Test corner classification (top-left, top-right, bottom-left, bottom-right)
- Test edge cases (region at frame boundary, minimum/maximum sizes)

**CanvasLayout Computation**
- Test layout computation from config
- Test facecam and gameplay region dimensions
- Test layout consistency (facecam_height + gameplay_height == canvas_height)

**EditorSession Management**
- Test session creation and expiry
- Test undo/redo history management
- Test settings persistence

**VerticalFormattingJob**
- Test job creation and status tracking
- Test progress updates
- Test error handling and logging

### Integration Tests

**Detection Pipeline**
- Test FacecamRelocator integration
- Test detection with various clip formats (16:9, 4:3, 1:1, 9:16)
- Test detection failure scenarios (no facecam, ambiguous regions)

**Preview Generation**
- Test preview image generation with various placements
- Test preview caching
- Test preview updates on adjustment

**Batch Processing**
- Test applying placement to multiple clips
- Test resolution scaling for different source resolutions
- Test output file generation and naming
- Test backup creation and original replacement

**API Endpoints**
- Test session creation and management
- Test detection endpoint with various inputs
- Test preview endpoint with various placements
- Test confirmation endpoint with valid/invalid placements
- Test progress polling
- Test cancellation

### Example-Based Tests

**UI Interaction**
- Test dismiss prompt without formatting
- Test opening mini editor
- Test manual facecam selection
- Test fallback fill option
- Test undo/redo buttons
- Test keyboard shortcuts (Ctrl+Z, Ctrl+Y)

**Error Scenarios**
- Test detection failure messaging
- Test invalid region error handling
- Test network error retry logic
- Test batch processing cancellation

**Settings**
- Test backup enable/disable
- Test output naming conventions
- Test replacement vs. separate directory
- Test settings persistence across sessions

### Property-Based Tests

**Facecam Region Validation**
- *For any* detected region, if its area fraction is outside [4%, 30%], it should be rejected
- *For any* detected region, it should be classified into exactly one of four corners
- *For any* set of detected crops, confidence should be in [0.0, 1.0]

**Preview Updates**
- *For any* adjustment to facecam region, preview should update within 500ms
- *For any* facecam placement, aspect ratios should be preserved in preview
- *For any* adjustment, preview should reflect the new placement accurately

**Batch Consistency**
- *For any* batch of clips, the same facecam placement should be applied to all
- *For any* clip with different resolution, coordinates should be proportionally adjusted
- *For any* batch, all output files should have consistent naming

**Validation**
- *For any* facecam placement, if it extends beyond frame bounds, confirmation should be prevented
- *For any* invalid placement, an error message should be displayed
- *For any* valid placement, confirmation should proceed to batch processing

**Undo/Redo**
- *For any* adjustment, undo should restore the previous state
- *For any* undone adjustment, redo should reapply it
- *For any* confirmation or close, undo/redo history should be cleared

**File Operations**
- *For any* output file, its name should match the original with optional suffix
- *For any* output file, it should be saved to the configured output directory
- *For any* output file, it should preserve the original audio track
- *For any* replacement operation with backup enabled, backups should be created

**Configuration Respect**
- *For any* configuration setting, it should be honored by the mini editor
- *For any* canvas dimension setting, it should be used in layout computation
- *For any* facecam area fraction constraint, it should be enforced in validation

**Cancellation**
- *For any* long-running operation, cancellation should be available
- *For any* cancelled operation, no clips should be modified
- *For any* cancelled batch, already-processed clips should be preserved

---

## Accessibility

The Mini Editor follows WCAG 2.1 Level AA guidelines:

- **Keyboard Navigation**: All controls accessible via Tab, Enter, Arrow keys
- **Screen Reader Support**: ARIA labels, status announcements, progress updates
- **Color Contrast**: Minimum 4.5:1 for text, sufficient for UI elements
- **Alt Text**: All images and icons have descriptive alt text
- **Font Sizing**: Users can adjust font size and spacing
- **Focus Indicators**: Clear visual focus indicators on all interactive elements
- **Error Messages**: Clear, descriptive error messages with suggestions

---

## Future Enhancements

1. **Manual Facecam Selection**: Draw bounding box on source clip to manually select facecam
2. **Fallback Fill Options**: Choose between blurred gameplay, solid color, or custom image
3. **Multi-Clip Preview**: Preview multiple clips simultaneously to verify consistency
4. **Batch Undo**: Undo entire batch processing operation
5. **Custom Canvas Layouts**: Allow users to define custom facecam/gameplay splits
6. **Facecam Animations**: Smooth transitions when facecam moves between clips
7. **Audio Normalization**: Normalize audio levels across clips
8. **Subtitle Positioning**: Adjust subtitle position relative to facecam/gameplay regions
9. **Export Presets**: Save and reuse placement presets for future batches
10. **Comparison View**: Side-by-side comparison of original vs. vertical format

---

## Conclusion

The Mini Video Editor provides a user-friendly interface for formatting horizontal clips to vertical format with repositionable facecam overlays. By reusing existing pipeline components and following established design patterns, it integrates seamlessly with the existing workflow while providing powerful customization options for users who want to fine-tune their vertical content.

The design prioritizes real-time feedback, graceful error handling, and batch consistency, ensuring a smooth user experience from detection through final output.
