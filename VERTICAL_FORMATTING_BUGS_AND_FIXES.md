# Vertical Formatting - Comprehensive Bug Analysis

## Critical Bugs Found

### 🔴 BUG #1: Actual Processing Uses Letterboxing Instead of Cropping
**Location:** `pipeline/vertical_formatter.py` - `_build_vertical_filter()`
**Status:** NOT FIXED
**Severity:** CRITICAL

**Problem:**
- The preview endpoint (`web_server.py`) uses crop-based gameplay region (correct)
- The actual formatter (`vertical_formatter.py`) uses `build_canvas_filter()` which does letterboxing (wrong)
- User sees cropped preview but gets letterboxed output

**Evidence:**
```python
# In vertical_formatter.py line 130:
canvas_fragment = reformatter.build_canvas_filter(src_width, src_height, layout)
# This scales and pads (letterbox), doesn't crop!
```

**Fix Needed:**
Replace `build_canvas_filter()` call with the same crop logic used in preview endpoint:
1. Calculate crop dimensions for 9:16 aspect ratio
2. Crop center of source video
3. Scale to gameplay region
4. Pad to canvas

---

### 🟡 BUG #2: Inconsistency Between Preview and Processing
**Location:** `web_server.py` (preview) vs `pipeline/vertical_formatter.py` (processing)
**Status:** NOT FIXED
**Severity:** HIGH

**Problem:**
- Preview shows one thing (cropped)
- Processing produces another (letterboxed)
- User confirmation is meaningless if output doesn't match preview

**Fix Needed:**
Extract the crop-based filter logic into a shared function and use it in both places.

---

### 🟢 BUG #3: Facecam Region Bounds Validation
**Location:** `web_server.py` - confirm endpoint
**Status:** FIXED ✅

**Problem:**
- Strict validation rejected regions extending beyond frame
- User couldn't confirm even if preview looked good

**Fix Applied:**
- Auto-clamp regions to frame bounds
- Log warnings instead of rejecting
- Allow users to confirm what they see

---

### 🟢 BUG #4: Default Facecam Size Too Small
**Location:** `web/index.html` - detectFacecam()
**Status:** FIXED ✅

**Problem:**
- Default was 40%×40% = 16% area
- Sometimes resulted in regions below 4% minimum after adjustment

**Fix Applied:**
- Changed to 25%×25% = 6.25% area
- Added real-time validation warnings

---

### 🟢 BUG #5: Source Clip Preview Not Loading
**Location:** `web/index.html` - loadHorizontalPreview()
**Status:** FIXED ✅

**Problem:**
- Only showed placeholder text
- User couldn't see source video

**Fix Applied:**
- Implemented HTML5 video element loading
- Extracts frame at 1 second
- Draws to canvas
- Falls back to placeholder on error

---

### 🟢 BUG #6: Clips Not Being Replaced
**Location:** `web/index.html` - confirmPlacement()
**Status:** FIXED ✅

**Problem:**
- `replace_originals: false` meant clips weren't replaced

**Fix Applied:**
- Changed to `replace_originals: true`
- Clips now replaced after processing

---

## Recommended Actions

### Immediate (Critical)
1. **Fix `_build_vertical_filter()` to use crop-based approach**
   - Copy crop logic from preview endpoint
   - Replace `build_canvas_filter()` usage
   - Test that output matches preview

### High Priority
2. **Extract shared filter building function**
   - Create `build_vertical_crop_filter()` function
   - Use in both preview and processing
   - Ensures consistency

### Testing Checklist
- [ ] Preview shows cropped gameplay (no letterboxing)
- [ ] Processed clips match preview exactly
- [ ] Facecam overlay positioned correctly
- [ ] Clips are replaced after processing
- [ ] Backup is created before replacement
- [ ] Source preview loads actual video frame
- [ ] Bounds validation allows user confirmation
- [ ] Area fraction warnings show but don't block

## Files That Need Changes

### Critical
- `pipeline/vertical_formatter.py` - Fix `_build_vertical_filter()`

### Optional (Refactoring)
- `pipeline/frame_reformatter.py` - Add `build_crop_filter()` method
- `web_server.py` - Use shared filter function
- `pipeline/vertical_formatter.py` - Use shared filter function

## Testing Strategy

1. **Generate clips** from 16:9 source video
2. **Open vertical editor**
3. **Check preview** - should show cropped gameplay filling 9:16
4. **Confirm and process**
5. **Download processed clips**
6. **Verify** - clips should match preview (cropped, not letterboxed)
7. **Check** - original clips should be replaced
8. **Verify backup** - originals should be in backup directory
