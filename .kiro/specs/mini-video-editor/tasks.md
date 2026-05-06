# Implementation Plan: Mini Video Editor

## Overview

The Mini Video Editor is a web-based GUI component that enables users to format horizontally-recorded video clips into vertical (9:16) format with repositionable webcam overlays. The implementation follows a phased approach: backend API and data models, frontend UI components, integration with the existing pipeline, and comprehensive testing.

This plan reuses existing pipeline components (FacecamRelocator, FrameReformatter) and integrates with the existing web server infrastructure.

---

## Phase 1: Backend API and Data Models

- [x] 1. Create data models and session management
  - [x] 1.1 Define EditorSession dataclass with session tracking
    - Create EditorSession with session_id, clip_batch_id, reference_clip_path, facecam_region, canvas_layout, undo/redo history
    - Implement session expiry logic (30-minute timeout)
    - _Requirements: 2.1, 11.1, 11.5_
  
  - [x] 1.2 Define VerticalFormattingJob dataclass for batch processing
    - Create VerticalFormattingJob with job tracking, progress counters, error logging
    - Implement job status tracking (queued, running, done, failed)
    - _Requirements: 7.1, 19.1, 19.2_
  
  - [x] 1.3 Create session storage and retrieval layer
    - Implement in-memory session store with expiry cleanup
    - Add session lookup by session_id
    - _Requirements: 2.1, 11.1_

- [x] 2. Implement facecam detection backend
  - [x] 2.1 Create detection endpoint wrapper
    - Implement POST /api/mini-editor/detect endpoint
    - Call FacecamRelocator.detect_facecam() from existing pipeline
    - Return FacecamRegion with confidence score
    - _Requirements: 3.1, 3.2, 3.5, 3.6, 11.2_
  
  - [x] 2.2 Implement detection result caching
    - Cache detection results per clip to avoid redundant processing
    - Implement cache invalidation on clip update
    - _Requirements: 12.3_
  
  - [x] 2.3 Write property tests for facecam detection
    - **Property 1: Area fraction validation** - For any detected region, if its area fraction is outside [4%, 30%], it should be rejected
    - **Property 2: Confidence bounds** - For any set of detected crops, confidence should be in [0.0, 1.0]
    - **Property 3: Corner classification** - For any detected region, it should be classified into exactly one of four corners
    - **Validates: Requirements 3.3, 3.4, 3.5_

- [x] 3. Implement preview generation backend
  - [x] 3.1 Create preview endpoint
    - Implement POST /api/mini-editor/preview endpoint
    - Accept facecam_region and generate preview image
    - Return preview_image_url
    - _Requirements: 5.1, 5.2, 5.3, 5.4_
  
  - [x] 3.2 Implement canvas layout computation
    - Call FrameReformatter.compute_canvas_layout() from existing pipeline
    - Compute facecam and gameplay region dimensions
    - _Requirements: 5.1, 5.2, 5.3, 11.3, 11.5_
  
  - [x] 3.3 Implement preview image rendering
    - Generate preview image showing vertical canvas with facecam and gameplay regions
    - Use lower resolution for performance (540×960)
    - _Requirements: 5.1, 5.2, 5.3, 5.5, 5.6_
  
  - [x] 3.4 Implement preview caching
    - Cache preview images for common placements
    - Implement cache key based on facecam_region coordinates
    - _Requirements: 12.2_
  
  - [x] 3.5 Write property tests for preview generation
    - **Property 4: Aspect ratio preservation** - For any facecam placement, aspect ratios should be preserved in preview
    - **Property 5: Preview accuracy** - For any adjustment, preview should reflect the new placement accurately
    - **Validates: Requirements 5.5, 5.6_

- [x] 4. Implement session and confirmation endpoints
  - [x] 4.1 Create session initialization endpoint
    - Implement POST /api/mini-editor/session endpoint
    - Accept clip_batch_id and reference_clip_path
    - Return session_id and clip list
    - _Requirements: 2.1, 11.1_
  
  - [x] 4.2 Create confirmation endpoint
    - Implement POST /api/mini-editor/confirm endpoint
    - Validate facecam_region bounds
    - Create VerticalFormattingJob and queue for processing
    - Return job_id
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 7.1_
  
  - [x] 4.3 Create cancellation endpoint
    - Implement POST /api/mini-editor/cancel endpoint
    - Close editor session without processing
    - _Requirements: 6.5, 6.6_
  
  - [x] 4.4 Write property tests for validation
    - **Property 6: Bounds validation** - For any facecam placement, if it extends beyond frame bounds, confirmation should be prevented
    - **Property 7: Validation consistency** - For any invalid placement, an error message should be displayed
    - **Validates: Requirements 6.2, 6.3, 6.4_

- [x] 5. Implement undo/redo backend
  - [x] 5.1 Create undo endpoint
    - Implement POST /api/mini-editor/undo endpoint
    - Pop from undo_history, push to redo_history
    - Return previous facecam_region
    - _Requirements: 18.1, 18.2_
  
  - [x] 5.2 Create redo endpoint
    - Implement POST /api/mini-editor/redo endpoint
    - Pop from redo_history, push to undo_history
    - Return reapplied facecam_region
    - _Requirements: 18.1, 18.3_
  
  - [x] 5.3 Write property tests for undo/redo
    - **Property 8: Undo restoration** - For any adjustment, undo should restore the previous state
    - **Property 9: Redo reapplication** - For any undone adjustment, redo should reapply it
    - **Property 10: History clearing** - For any confirmation or close, undo/redo history should be cleared
    - **Validates: Requirements 18.1, 18.2, 18.3, 18.5_

---

## Phase 2: Vertical Formatting and Batch Processing

- [x] 6. Implement vertical formatter core
  - [x] 6.1 Create VerticalFormatter class
    - Implement apply_placement_to_clip() method
    - Build FFmpeg filter chain for facecam overlay and gameplay scaling
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 11.4_
  
  - [x] 6.2 Implement resolution scaling for different source formats
    - Handle proportional coordinate adjustment for different resolutions
    - Support horizontal (16:9), vertical (9:16), and square (1:1) source videos
    - _Requirements: 7.5, 14.1, 14.2, 14.3, 14.4, 14.5_
  
  - [x] 6.3 Implement FFmpeg filter generation
    - Reuse FrameReformatter.build_canvas_filter() from existing pipeline
    - Build facecam crop and scale filters
    - Build overlay filter to composite facecam on canvas
    - _Requirements: 7.1, 11.4, 16.3_
  
  - [x] 6.4 Write property tests for vertical formatting
    - **Property 11: Consistent placement** - For any batch of clips, the same facecam placement should be applied to all
    - **Property 12: Resolution scaling** - For any clip with different resolution, coordinates should be proportionally adjusted
    - **Property 13: Aspect ratio preservation** - For any source format, aspect ratio should be preserved in output
    - **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5, 14.1, 14.4_

- [x] 7. Implement batch processing and job queue
  - [x] 7.1 Create batch processing job handler
    - Implement process_vertical_formatting_job() function
    - Iterate through clips in batch, apply placement to each
    - Track progress and update job status
    - _Requirements: 7.1, 7.6, 19.1, 19.2, 19.3_
  
  - [x] 7.2 Implement progress tracking and reporting
    - Update clips_processed counter after each clip
    - Calculate estimated time remaining
    - Store current_clip name for display
    - _Requirements: 19.1, 19.2, 19.3, 19.4_
  
  - [x] 7.3 Implement error handling and logging
    - Catch encoding errors, log with clip name and reason
    - Continue processing remaining clips on error
    - Store errors in job.errors list
    - _Requirements: 10.7, 19.5_
  
  - [x] 7.4 Implement job cancellation
    - Allow user to cancel batch processing at any time
    - Preserve already-processed clips
    - Update job status to "cancelled"
    - _Requirements: 12.7, 19.6_
  
  - [x] 7.5 Write property tests for batch processing
    - **Property 14: Cancellation safety** - For any cancelled operation, no clips should be modified
    - **Property 15: Progress accuracy** - For any batch, progress should accurately reflect clips processed
    - **Validates: Requirements 7.6, 12.7, 19.1, 19.2_

- [x] 8. Implement output file generation and replacement
  - [x] 8.1 Create output file naming logic
    - Implement naming convention (append "_vertical" suffix by default)
    - Support custom naming from user settings
    - _Requirements: 8.2, 20.4_
  
  - [x] 8.2 Implement output file encoding
    - Encode vertical canvas with facecam and gameplay regions
    - Preserve audio track from original clip
    - Use same codec and quality settings as original
    - _Requirements: 8.1, 8.4, 8.5, 8.6_
  
  - [x] 8.3 Implement original clip backup
    - Create backup directory with timestamp
    - Copy original clips to backup before replacement
    - Use clear naming convention for backups
    - _Requirements: 9.2, 9.3_
  
  - [x] 8.4 Implement original clip replacement
    - Replace original clips with vertical versions
    - Handle backup creation if enabled
    - _Requirements: 9.1, 9.2, 9.3, 9.4_
  
  - [x] 8.5 Implement undo replacement (optional)
    - Restore original clips from backup if available
    - Provide undo option within same session
    - _Requirements: 9.5_
  
  - [x] 8.6 Write property tests for file operations
    - **Property 16: Output naming** - For any output file, its name should match the original with optional suffix
    - **Property 17: Output location** - For any output file, it should be saved to the configured output directory
    - **Property 18: Audio preservation** - For any output file, it should preserve the original audio track
    - **Property 19: Backup creation** - For any replacement operation with backup enabled, backups should be created
    - **Validates: Requirements 8.2, 8.3, 9.2, 9.3_

---

## Phase 3: Frontend UI Components

- [x] 9. Create HTML structure and layout
  - [x] 9.1 Create main editor HTML template
    - Build responsive layout with horizontal and vertical preview panes
    - Add adjustment controls (sliders for X, Y, width, height)
    - Add action buttons (Confirm, Cancel, Undo, Redo, Settings)
    - _Requirements: 2.2, 2.3, 2.4, 2.5,   2.6, 4.1, 4.2, 4.3, 4.4, 4.5_
  
  - [x] 9.2 Create settings panel HTML
    - Add backup enable/disable checkbox
    - Add output naming convention options
    - Add replacement vs. separate directory options
    - _Requirements: 20.2, 20.3, 20.4, 20.5_
  
  - [x] 9.3 Create progress view HTML
    - Add progress bar showing clips processed
    - Add current clip name display
    - Add estimated time remaining
    - Add pause/cancel buttons
    - _Requirements: 12.6, 19.1, 19.2, 19.3, 19.4_
  
  - [x] 9.4 Create error message templates
    - Add detection failure message with suggestions
    - Add invalid region error message
    - Add network error message with retry option
    - _Requirements: 3.7, 10.1, 10.2, 10.3_

- [x] 10. Implement CSS styling and responsive design
  - [x] 10.1 Create base styles and layout
    - Implement responsive grid layout for preview panes
    - Style adjustment controls (sliders, inputs)
    - Style action buttons with clear visual hierarchy
    - _Requirements: 2.6, 4.1, 4.2, 4.3, 4.4_
  
  - [x] 10.2 Implement preview pane styling
    - Style horizontal preview pane (16:9 aspect ratio)
    - Style vertical preview pane (9:16 aspect ratio)
    - Add facecam region highlight/bounding box
    - _Requirements: 2.2, 2.3, 2.4, 2.5_
  
  - [x] 10.3 Implement responsive breakpoints
    - Desktop layout (1024×768+): side-by-side preview panes
    - Tablet layout (768×1024): stacked preview panes
    - Mobile layout: single preview pane with toggle
    - _Requirements: 2.6, 13.4_
  
  - [x] 10.4 Implement accessibility styling
    - Ensure 4.5:1 color contrast for text
    - Add clear focus indicators on interactive elements
    - Support font size adjustment
    - _Requirements: 17.2, 17.4, 17.5, 17.7_

- [x] 11. Implement JavaScript preview rendering
  - [x] 11.1 Create canvas rendering for horizontal preview
    - Render source clip preview at original aspect ratio
    - Draw facecam region bounding box/highlight
    - Update on facecam_region changes
    - _Requirements: 2.2, 2.3, 4.6_
  
  - [x] 11.2 Create canvas rendering for vertical preview
    - Render 9:16 vertical canvas
    - Draw facecam region in top portion
    - Draw gameplay region in bottom portion
    - Apply black background fill
    - _Requirements: 2.4, 2.5, 5.1, 5.2, 5.3, 5.6_
  
  - [x] 11.3 Implement real-time preview updates
    - Debounce slider inputs (100ms)
    - Update preview within 500ms of adjustment
    - Fetch preview image from backend
    - _Requirements: 4.6, 5.4, 12.1_
  
  - [x] 11.4 Write property tests for preview rendering
    - **Property 20: Update responsiveness** - For any adjustment to facecam region, preview should update within 500ms
    - **Validates: Requirements 4.6, 5.4, 12.1_

- [x] 12. Implement adjustment controls
  - [x] 12.1 Create slider controls for X, Y, width, height
    - Implement range sliders with min/max bounds
    - Display current value next to each slider
    - Bind to facecam_region state
    - _Requirements: 4.2, 4.3, 4.4, 4.5_
  
  - [x] 12.2 Implement bounds validation
    - Prevent region from extending beyond frame bounds
    - Enforce minimum and maximum sizes
    - Display error message on invalid adjustment
    - _Requirements: 4.7, 4.8_
  
  - [x] 12.3 Implement visual feedback for invalid regions
    - Highlight invalid slider positions
    - Show error message below slider
    - Disable Confirm button if region is invalid
    - _Requirements: 4.8_
  
  - [x] 12.4 Implement keyboard support for adjustments
    - Allow arrow keys to adjust values
    - Support Tab navigation between controls
    - _Requirements: 17.2_

- [x] 13. Implement undo/redo UI
  - [x] 13.1 Create undo/redo buttons
    - Add Undo button that calls POST /api/mini-editor/undo
    - Add Redo button that calls POST /api/mini-editor/redo
    - Disable buttons when history is empty
    - _Requirements: 18.2, 18.3_
  
  - [x] 13.2 Implement keyboard shortcuts
    - Ctrl+Z for undo
    - Ctrl+Y for redo
    - _Requirements: 18.4_
  
  - [x] 13.3 Display undo/redo state
    - Show "Undo (3 steps)" or "Redo disabled" text
    - Update state display after each action
    - _Requirements: 18.6_

- [x] 14. Implement settings panel
  - [x] 14.1 Create settings panel UI
    - Add backup enable/disable checkbox
    - Add output naming convention radio buttons
    - Add replacement vs. separate directory options
    - _Requirements: 20.2, 20.3, 20.4, 20.5_
  
  - [x] 14.2 Implement settings persistence
    - Save settings to localStorage
    - Load settings on page load
    - _Requirements: 20.6_
  
  - [x] 14.3 Implement settings validation
    - Validate output directory exists and is writable
    - Validate naming convention format
    - Display error message on invalid settings
    - _Requirements: 20.7_

---

## Phase 4: Integration and Testing

- [x] 15. Integrate with existing web server
  - [x] 15.1 Register mini-editor routes
    - Add GET /mini-editor route to serve HTML
    - Register all API endpoints (/api/mini-editor/*)
    - _Requirements: 13.2, 11.1_
  
  - [x] 15.2 Integrate with existing job queue
    - Add VerticalFormattingJob to existing queue
    - Reuse job status endpoint for progress polling
    - _Requirements: 7.1, 19.1, 19.2_
  
  - [x] 15.3 Integrate with existing config
    - Load canvas dimensions from Config object
    - Load facecam area constraints from Config object
    - Load facecam_sample_duration from Config object
    - _Requirements: 11.5, 15.6, 16.5_

- [x] 16. Integrate with existing pipeline components
  - [x] 16.1 Integrate FacecamRelocator
    - Import FacecamRelocator from pipeline.facecam_relocator
    - Call detect_facecam() method in detection endpoint
    - Reuse FacecamRegion data model
    - _Requirements: 11.2, 15.1, 15.2, 15.3, 15.4, 15.5, 15.6, 15.7_
  
  - [x] 16.2 Integrate FrameReformatter
    - Import FrameReformatter from pipeline.frame_reformatter
    - Call compute_canvas_layout() in preview endpoint
    - Call build_canvas_filter() in vertical formatter
    - Reuse CanvasLayout data model
    - _Requirements: 11.3, 16.1, 16.2, 16.3, 16.4, 16.5, 16.6_
  
  - [x] 16.3 Integrate FFmpeg filter generation
    - Reuse FFmpeg filter generation logic from pipeline
    - Build complete filter chain for vertical output
    - _Requirements: 11.4, 16.3_

- [x] 17. Implement error handling and fallback options
  - [x] 17.1 Implement detection failure handling
    - Display error message with reason (area outside bounds, no clear pip detected)
    - Offer manual selection option
    - Offer fallback fill option
    - _Requirements: 3.7, 10.1, 10.2, 10.3, 10.4_
  
  - [x] 17.2 Implement manual facecam selection mode
    - Switch UI to drawing mode on user request
    - Allow user to draw bounding box on source clip
    - Validate drawn region against area constraints
    - _Requirements: 10.3, 10.5_
  
  - [x] 17.3 Implement fallback fill option
    - Generate blurred and cropped gameplay fill for top region
    - Apply fallback to all clips in batch
    - _Requirements: 10.4, 10.6, 5.7_
  
  - [x] 17.4 Implement network error handling
    - Catch API request timeouts
    - Display error message with retry option
    - Implement exponential backoff for retries
    - _Requirements: 13.7_

- [x] 18. Implement format-to-vertical prompt
  - [x] 18.1 Create prompt UI component
    - Display after clip generation completes
    - Show "Format to Vertical" call-to-action button
    - Allow user to dismiss without formatting
    - _Requirements: 1.1, 1.2, 1.3_
  
  - [x] 18.2 Implement prompt trigger logic
    - Detect when clip generation completes
    - Display prompt only after all clips are ready
    - _Requirements: 1.4, 1.5_
  
  - [x] 18.3 Implement prompt action handlers
    - Open mini-editor on "Format to Vertical" click
    - Close prompt on dismiss
    - _Requirements: 1.4_

- [x] 19. Implement progress and status displays
  - [x] 19.1 Create progress indicator for detection
    - Show loading spinner while FacecamRelocator runs
    - Display "Detecting facecam..." message
    - _Requirements: 12.5, 3.1_
  
  - [x] 19.2 Create progress indicator for batch processing
    - Show progress bar with clips processed/total
    - Display current clip name
    - Display estimated time remaining
    - _Requirements: 12.6, 19.1, 19.2, 19.3, 19.4_
  
  - [x] 19.3 Implement batch processing summary
    - Display "X clips formatted successfully, Y failed" message
    - Show list of failed clips with reasons
    - _Requirements: 19.7_

- [ ] 20. Implement accessibility features
  - [x] 20.1 Add ARIA labels and descriptions
    - Add aria-label to all interactive elements
    - Add aria-describedby for error messages
    - Add aria-live for status updates
    - _Requirements: 17.1, 17.3, 17.6_
  
  - [x] 20.2 Implement keyboard navigation
    - Ensure all controls accessible via Tab
    - Implement Enter for button activation
    - Implement Arrow keys for slider adjustment
    - _Requirements: 17.2_
  
  - [x] 20.3 Implement screen reader announcements
    - Announce detection completion
    - Announce preview updates
    - Announce batch processing progress
    - Announce errors and validation messages
    - _Requirements: 17.6_
  
  - [x] 20.4 Implement color contrast and visual indicators
    - Ensure 4.5:1 contrast for all text
    - Add clear focus indicators on all interactive elements
    - Use icons + text for all buttons
    - _Requirements: 17.4, 17.5_
  
  - [x] 20.5 Implement font sizing and spacing adjustment
    - Allow users to increase font size
    - Allow users to increase spacing
    - Persist preferences across sessions
    - _Requirements: 17.7_

- [x] 21. Checkpoint - Ensure all backend and frontend components are integrated
  - Verify all API endpoints are registered and functional
  - Verify all UI components render correctly
  - Verify integration with existing pipeline components
  - Ensure all tests pass, ask the user if questions arise.

---

## Phase 5: Comprehensive Testing

- [-] 22. Write integration tests for API endpoints
  - [x] 22.1 Test session creation and management
    - Test POST /api/mini-editor/session with valid batch_id
    - Test session expiry after 30 minutes
    - Test session lookup by session_id
    - _Requirements: 2.1, 11.1_
  
  - [x] 22.2 Test detection endpoint
    - Test POST /api/mini-editor/detect with various clip formats
    - Test detection success with confidence score
    - Test detection failure with error message
    - _Requirements: 3.1, 3.2, 3.5, 3.6, 3.7_
  
  - [x] 22.3 Test preview endpoint
    - Test POST /api/mini-editor/preview with various placements
    - Test preview image generation and caching
    - Test preview updates on adjustment
    - _Requirements: 5.1, 5.2, 5.3, 5.4_
  
  - [x] 22.4 Test confirmation endpoint
    - Test POST /api/mini-editor/confirm with valid placement
    - Test validation of invalid placements
    - Test job creation and queuing
    - _Requirements: 6.1, 6.2, 6.3, 6.4_
  
  - [x] 22.5 Test undo/redo endpoints
    - Test POST /api/mini-editor/undo and redo
    - Test history management
    - Test history clearing on confirmation
    - _Requirements: 18.1, 18.2, 18.3, 18.5_

- [-] 23. Write integration tests for batch processing
  - [x] 23.1 Test vertical formatting on single clip
    - Test apply_placement_to_clip() with various source formats
    - Test output file generation with correct naming
    - Test audio preservation
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 8.1, 8.2, 8.5_
  
  - [x] 23.2 Test batch processing with multiple clips
    - Test applying same placement to all clips
    - Test progress tracking and updates
    - Test error handling for individual clip failures
    - _Requirements: 7.1, 7.6, 19.1, 19.2, 19.3_
  
  - [x] 23.3 Test resolution scaling
    - Test proportional coordinate adjustment for different resolutions
    - Test handling of horizontal, vertical, and square source videos
    - _Requirements: 7.5, 14.1, 14.2, 14.3, 14.4, 14.5_
  
  - [x] 23.4 Test backup and replacement
    - Test backup creation before replacement
    - Test original clip replacement
    - Test undo replacement from backup
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_
  
  - [x] 23.5 Test batch processing cancellation
    - Test cancellation during processing
    - Test preservation of already-processed clips
    - Test job status update to "cancelled"
    - _Requirements: 12.7, 19.6_

- [-] 24. Write UI interaction tests
  - [x] 24.1 Test adjustment control interactions
    - Test slider adjustments update preview
    - Test bounds validation prevents invalid adjustments
    - Test error messages display on invalid input
    - _Requirements: 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8_
  
  - [x] 24.2 Test undo/redo button interactions
    - Test undo button reverts last adjustment
    - Test redo button reapplies undone adjustment
    - Test keyboard shortcuts (Ctrl+Z, Ctrl+Y)
    - Test button disable state when history empty
    - _Requirements: 18.1, 18.2, 18.3, 18.4, 18.6_
  
  - [x] 24.3 Test settings panel interactions
    - Test backup checkbox enable/disable
    - Test output naming convention selection
    - Test replacement vs. separate directory options
    - Test settings persistence across sessions
    - _Requirements: 20.2, 20.3, 20.4, 20.5, 20.6_
  
  - [x] 24.4 Test error handling and recovery
    - Test detection failure messaging and options
    - Test manual selection mode
    - Test fallback fill option
    - Test network error retry logic
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 13.7_

- [-] 25. Write accessibility tests
  - [x] 25.1 Test keyboard navigation
    - Test Tab navigation through all controls
    - Test Enter activation of buttons
    - Test Arrow keys for slider adjustment
    - _Requirements: 17.2_
  
  - [x] 25.2 Test screen reader compatibility
    - Test ARIA labels on all interactive elements
    - Test status announcements for detection/processing
    - Test error message announcements
    - _Requirements: 17.3, 17.6_
  
  - [x] 25.3 Test color contrast and visual indicators
    - Test 4.5:1 contrast ratio for all text
    - Test focus indicators on interactive elements
    - Test alt text on images and icons
    - _Requirements: 17.4, 17.5_

- [x] 26. Checkpoint - Ensure all tests pass
  - Run all unit tests for data models and validation
  - Run all integration tests for API endpoints and batch processing
  - Run all UI interaction tests
  - Run all accessibility tests
  - Ensure all tests pass, ask the user if questions arise.

- [x] 27. Performance testing and optimization
  - [x] 27.1 Test detection performance
    - Verify detection completes within 30 seconds for typical 30-second clip
    - Profile FFmpeg cropdetect performance
    - _Requirements: 12.3_
  
  - [x] 27.2 Test preview update responsiveness
    - Verify preview updates within 500ms of adjustment
    - Profile canvas rendering performance
    - _Requirements: 12.1_
  
  - [x] 27.3 Test batch processing performance
    - Verify each clip processes within 2-5 minutes
    - Profile encoding performance
    - _Requirements: 12.4_
  
  - [x] 27.4 Test memory usage
    - Verify memory usage stays reasonable during batch processing
    - Verify session cleanup on expiry
    - _Requirements: 12.1, 12.2_

- [x] 28. Final checkpoint - Ensure all requirements are met
  - Verify all 20 requirements are implemented and tested
  - Verify integration with existing pipeline is complete
  - Verify web GUI is responsive and accessible
  - Verify batch processing works end-to-end
  - Ensure all tests pass, ask the user if questions arise.

---

## Notes

- Tasks marked with `*` are optional property-based tests and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation and integration
- Property tests validate universal correctness properties
- Unit and integration tests validate specific examples and edge cases
- All implementation uses Python with Flask for backend and vanilla JavaScript for frontend
- Reuses existing pipeline components (FacecamRelocator, FrameReformatter) to maintain consistency
- Integrates with existing web server infrastructure and job queue
