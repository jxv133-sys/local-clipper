# Task 3.1 Implementation Summary: Create Preview Endpoint

## Overview
Successfully implemented the POST /api/mini-editor/preview endpoint for the mini video editor feature. This endpoint generates preview images showing how the vertical (9:16) output will look with the current facecam placement.

## Implementation Details

### Endpoint: POST /api/mini-editor/preview

**Location:** `web_server.py` (lines ~890-1100)

**Request JSON:**
```json
{
    "clip_path": "string",           // Path to the clip file
    "facecam_region": {              // Facecam region to preview
        "x": 100,
        "y": 50,
        "width": 400,
        "height": 300,
        "corner": "top-right",
        "confidence": 0.85
    },
    "frame_width": 1920,             // Source frame width
    "frame_height": 1080,            // Source frame height
    "config": {                      // Optional config overrides
        "shorts_width": 1080,
        "shorts_height": 1920,
        "facecam_top_fraction": 0.35
    }
}
```

**Response JSON:**
```json
{
    "preview_image_url": "/output/preview_abc123.jpg",
    "canvas_layout": {
        "canvas_width": 1080,
        "canvas_height": 1920,
        "facecam_x": 0,
        "facecam_y": 0,
        "facecam_width": 1080,
        "facecam_height": 672,
        "gameplay_x": 0,
        "gameplay_y": 672,
        "gameplay_width": 1080,
        "gameplay_height": 1248
    },
    "error": null,
    "cached": false
}
```

### Key Features Implemented

1. **Preview Image Generation**
   - Uses FFmpeg to extract a frame at 1 second from the clip
   - Applies the vertical canvas layout with facecam overlay
   - Generates lower resolution preview (540×960) for performance
   - Saves as JPEG with high quality settings

2. **Canvas Layout Computation**
   - Reuses `FrameReformatter.compute_canvas_layout()` from existing pipeline
   - Computes 9:16 vertical canvas dimensions
   - Calculates facecam and gameplay region positions

3. **FFmpeg Filter Chain**
   - Builds canvas filter for gameplay region (scales and pads source video)
   - Crops facecam region from source
   - Scales facecam to fit top portion of canvas
   - Overlays facecam on canvas at correct position
   - Scales final result to preview resolution

4. **Preview Caching**
   - Caches preview images based on clip path and facecam region coordinates
   - Cache key: `(clip_path, x, y, width, height, mtime)`
   - Invalidates cache when source file is modified (mtime check)
   - Returns cached preview if available, avoiding redundant FFmpeg calls

5. **Error Handling**
   - Validates all required fields (clip_path, facecam_region, frame dimensions)
   - Checks if clip file exists
   - Handles FFmpeg failures gracefully
   - Returns detailed error messages

6. **Config Overrides**
   - Accepts optional config overrides for canvas dimensions
   - Allows customization of facecam_top_fraction
   - Maintains backward compatibility with default config

### Files Modified

1. **web_server.py**
   - Added `_preview_cache` dictionary for caching preview images
   - Added `_preview_cache_lock` for thread-safe cache access
   - Implemented `generate_preview_endpoint()` function

### Tests Created

**File:** `tests/test_mini_editor_preview.py` (12 tests, all passing)

Test coverage includes:
- ✅ Successful preview generation
- ✅ Missing required fields (clip_path, facecam_region, frame dimensions)
- ✅ Invalid facecam_region structure
- ✅ Non-existent clip file
- ✅ FFmpeg failure handling
- ✅ Preview caching (cache hit on second request)
- ✅ Cache invalidation on file modification
- ✅ Different facecam regions generate different previews
- ✅ Invalid JSON body
- ✅ Custom config overrides

### Requirements Satisfied

✅ **Requirement 5.1:** Preview pane displays 9:16 vertical canvas representation  
✅ **Requirement 5.2:** Preview shows facecam region in top portion  
✅ **Requirement 5.3:** Preview shows gameplay region in bottom portion  
✅ **Requirement 5.4:** Preview updates when facecam region changes  
✅ **Requirement 12.2:** Preview caching for performance  

### Integration Points

- **FrameReformatter:** Reuses `compute_canvas_layout()` and `build_canvas_filter()`
- **Config:** Respects canvas dimensions and facecam constraints
- **FacecamRegion:** Uses existing data model from `pipeline.models`
- **CanvasLayout:** Uses existing data model from `pipeline.models`

### Performance Considerations

- Preview images generated at 540×960 (half resolution) for faster generation
- Preview caching reduces redundant FFmpeg calls
- Cache invalidation based on file mtime ensures fresh previews when clips change
- FFmpeg extracts single frame at 1 second (fast operation)

### Next Steps

This completes task 3.1. The preview endpoint is now ready for:
- Task 3.2: Canvas layout computation (already integrated)
- Task 3.3: Preview image rendering (already implemented)
- Task 3.4: Preview caching (already implemented)
- Frontend integration for real-time preview updates

## Testing

All tests pass:
```bash
$ python3 -m pytest tests/test_mini_editor_preview.py -v
============================== 12 passed in 0.57s ==============================

$ python3 -m pytest tests/test_mini_editor_api.py -v
============================== 14 passed in 0.56s ==============================
```

No regressions in existing mini-editor API tests.
