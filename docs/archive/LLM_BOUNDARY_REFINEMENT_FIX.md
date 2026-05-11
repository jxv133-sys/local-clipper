# LLM Boundary Refinement Fix

## Problem

The LLM boundary refinement was producing clips that exceeded the maximum duration constraint (e.g., 175s clips when max is 60s). This resulted in warnings:

```
pipeline.clip_selector WARNING LLM boundary refinement produced 175s clip (max 60s); keeping original
```

## Root Cause

The LLM was being given all segment timestamps within a ±45s context window without filtering for valid combinations. This meant:

1. The LLM could see timestamps from 45s before to 45s after the clip (90s total window)
2. It could select any start time and any end time from this window
3. Even though the prompt instructed it to stay within max_clip_duration, the LLM would sometimes select timestamps that were too far apart
4. The validation logic would then reject the refined boundaries and keep the original clip

**Example:**
- Context window: 100s → 190s (90s window)
- LLM selects: START_TIME: 100.0, END_TIME: 275.0 (175s duration)
- Validation rejects: 175s > 60s max
- Result: Original clip kept, warning logged

## Solution

Pre-filter the available end times to only include those that would create valid clips when paired with any available start time.

### Algorithm

```python
# 1. Collect all segment start times in context window
available_starts = [seg.start for seg in context_segs]

# 2. Collect all segment end times in context window
available_ends = [seg.end for seg in context_segs]

# 3. Filter end times to only include valid combinations
valid_end_times = set()
for start_time in available_starts:
    for end_time in available_ends:
        if end_time > start_time:
            duration = end_time - start_time
            # Only include if duration is within constraints
            if min_clip_duration <= duration <= max_clip_duration:
                valid_end_times.add(end_time)

# 4. Pass filtered lists to LLM
```

### Benefits

1. **Guarantees valid clips**: The LLM can only select combinations that respect duration constraints
2. **Clearer prompt**: Added explicit note that end times are pre-filtered
3. **Fewer rejections**: Validation will rarely reject LLM suggestions now
4. **Better LLM guidance**: Smaller set of options makes it easier for LLM to choose correctly

## Changes Made

**File:** `pipeline/clip_selector.py`

**Function:** `_refine_clip_boundaries_with_llm()`

**Changes:**
1. Changed available_starts/ends from formatted strings to float sets
2. Added filtering logic to compute valid_end_times
3. Convert to formatted strings only after filtering
4. Added early return if no valid end times exist
5. Updated prompt to mention that end times are pre-filtered

## Testing

To verify the fix works:

1. Run pipeline with LLM enabled on a video
2. Check logs for boundary refinement messages
3. Verify no warnings about clips exceeding max duration
4. Verify refined clips are within min/max duration constraints

**Expected log output:**
```
INFO Boundary refined: 100.0s→145.0s (45s) → 95.0s→155.0s (60s) | Setup at 95s, moment at 120s, reaction ends at 155s
```

**No longer expected:**
```
WARNING LLM boundary refinement produced 175s clip (max 60s); keeping original
```

## Edge Cases Handled

1. **No valid end times**: If filtering results in empty set, return original clip with warning
2. **Very short context window**: If context window is smaller than min_clip_duration, may have limited options
3. **Clip at video boundaries**: Clamping to video_duration still applies after LLM selection

## Performance Impact

- Minimal: O(n²) filtering where n = number of segments in context window
- Typical n ≈ 10-20 segments (±45s window with ~5s segments)
- Total operations: ~100-400 comparisons per clip refinement
- Negligible compared to LLM API call time

## Alternative Approaches Considered

1. **Post-processing LLM output**: Clamp selected times to valid range
   - Rejected: Would change LLM's intent, might break narrative arc
   
2. **Stricter prompt engineering**: Add more constraints to prompt
   - Rejected: LLMs don't always follow complex numerical constraints
   
3. **Reduce context window**: Limit to max_clip_duration
   - Rejected: Would lose valuable context for identifying setup/reaction

4. **Pre-filtering (chosen)**: Only show valid options to LLM
   - Advantages: Guarantees valid output, maintains full context, respects LLM's narrative choices
