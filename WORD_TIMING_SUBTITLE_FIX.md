# Word-by-Word Subtitle Timing Fix

## Problem
Subtitles were not appearing as words were spoken. Instead, all words in a group appeared at once and highlighted at equal intervals, regardless of when each word was actually said. This made the subtitles feel like captions rather than true TikTok-style word-by-word highlighting.

## Root Cause
The `_build_word_by_word_line()` function in `animated_subtitle_renderer.py` was dividing the total duration of each subtitle entry equally among all words:

```python
total_cs = round((entry.end - entry.start) * 100)
word_cs = round(total_cs / max(len(words), 1))
```

This meant that if a phrase took 2 seconds and had 4 words, each word would get 0.5 seconds, regardless of how long each word actually took to say.

## Solution
We now pass actual word-level timestamps from the transcript through to the subtitle renderer:

### 1. Extended `SRTEntry` Model
Added `word_timings` field to store individual word timestamps:

```python
@dataclass
class SRTEntry:
    index: int
    start: float
    end: float
    text: str
    word_timings: list[tuple[str, float, float]] | None = None  # [(word, start, end), ...]
```

### 2. Updated `_word_level_entries()` in `subtitle_generator.py`
Now extracts and passes word-level timings when creating SRT entries:

```python
word_timings = [
    (w.word.strip(), max(0.0, w.start - clip_start), max(0.0, w.end - clip_start))
    for w in group
]
entries.append(
    SRTEntry(
        index=idx,
        start=rel_start,
        end=rel_end,
        text=group_text,
        word_timings=word_timings,
    )
)
```

### 3. Updated `_build_word_by_word_line()` in `animated_subtitle_renderer.py`
Now uses actual word timings instead of dividing duration equally:

```python
# If we have word-level timings, use them for precise highlighting
if entry.word_timings:
    words_with_timing = entry.word_timings
else:
    # Fallback: split text and divide duration equally
    words = entry.text.upper().split()
    total_cs = round((entry.end - entry.start) * 100)
    word_cs = round(total_cs / max(len(words), 1))
    words_with_timing = [
        (word, entry.start + (i * word_cs / 100), entry.start + ((i + 1) * word_cs / 100))
        for i, word in enumerate(words)
    ]
```

Then uses these timings for all subtitle styles:

```python
for word, word_start, word_end in words_with_timing:
    escaped_word = escape_ass_text(word.upper().strip())
    start_cs = round((word_start - entry.start) * 100)
    end_cs = round((word_end - entry.start) * 100)
    # Apply style-specific effects at exact word timing
```

## Result
Now each word highlights at the exact moment it's spoken in the audio, creating true TikTok-style word-by-word subtitles where:
- All words in a group (1-4 words) appear at once
- Each word highlights individually as it's spoken
- Timing is based on actual speech, not equal division

## Files Modified
- `pipeline/models.py` - Added `word_timings` field to `SRTEntry`
- `pipeline/subtitle_generator.py` - Updated `_word_level_entries()` to extract and pass word timings
- `pipeline/animated_subtitle_renderer.py` - Updated `_build_word_by_word_line()` to use actual word timings

## Testing
Test by:
1. Running the pipeline with subtitle burning enabled
2. Checking that words highlight as they're spoken, not at equal intervals
3. Verifying all 4 subtitle styles (bubble, popup, highlight, karaoke) work correctly
