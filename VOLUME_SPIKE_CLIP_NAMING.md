# Volume Spike Clip Naming

## Overview
Clips that bypass LLM scoring due to sudden volume spikes now have a special naming convention to make them easily identifiable.

## Problem
When audio spike detection is enabled, some clips are selected purely based on dramatic volume increases, bypassing the normal LLM scoring process. These clips were previously named the same as regular clips (e.g., `clip_1_10s.mp4`), making it difficult to distinguish them from LLM-scored clips.

## Solution
Clips selected for volume spikes now use a special naming prefix:

### Naming Convention
- **Regular clips** (LLM-scored): `clip_{rank}_{start_time}s.mp4`
  - Example: `clip_1_10s.mp4`, `clip_2_45s.mp4`
  
- **Volume spike clips** (bypassed LLM): `volume_spike_{rank}_{start_time}s.mp4`
  - Example: `volume_spike_3_120s.mp4`, `volume_spike_5_180s.mp4`

### Implementation
The filename is determined in `pipeline/clip_extractor.py` based on the `clip.is_audio_spike` flag:

```python
def _extract_single_clip(config: Config, clip: Clip, video_path: str) -> tuple[int, str]:
    """Extract a single clip and return ``(clip.rank, output_path)``."""
    t0 = time.time()
    
    # Use special naming for audio spike clips (bypassed LLM scoring)
    if clip.is_audio_spike:
        filename = f"volume_spike_{clip.rank}_{int(clip.start)}s.mp4"
    else:
        filename = f"clip_{clip.rank}_{int(clip.start)}s.mp4"
    
    output_path = os.path.join(config.output_dir, filename)
    # ... rest of extraction logic
```

## Benefits
1. **Easy identification**: Users can immediately see which clips were selected for volume spikes
2. **Better organization**: Volume spike clips are clearly distinguished from content-based clips
3. **Debugging**: Makes it easier to analyze the clip selection algorithm's behavior
4. **User understanding**: Clear naming helps users understand why certain clips were chosen

## Related Features
- The `_why_chosen.txt` report already includes a special "⚡ AUDIO SPIKE CLIP" section for these clips
- The `is_audio_spike` flag is preserved through clip merging and boundary refinement
- Volume spike clips are selected based on the `llm_audio_spike_percentage` configuration parameter

## Files Modified
- `pipeline/clip_extractor.py` - Updated filename generation logic

## Testing
To verify the naming works correctly:
1. Run the pipeline with LLM enabled and `llm_audio_spike_percentage > 0`
2. Check the output directory for clips with the `volume_spike_` prefix
3. Verify the corresponding `_why_chosen.txt` reports show "⚡ AUDIO SPIKE CLIP"
4. Confirm regular clips still use the `clip_` prefix
