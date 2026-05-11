# Word-by-Word Subtitle Timing Fix

## Problem (Initial)
Subtitles were not appearing as words were spoken. Instead, all words in a group appeared at once and highlighted at equal intervals, regardless of when each word was actually said. This made the subtitles feel like captions rather than true TikTok-style word-by-word highlighting.

## Problem (Follow-up)
After the initial fix, words that hadn't been spoken yet were staying visible on screen for too long. Words would appear dimmed/faded before they were actually spoken, which looked unnatural.

## Root Cause
1. **Initial issue**: The `_build_word_by_word_line()` function was dividing the total duration equally among all words instead of using actual speech timestamps
2. **Follow-up issue**: Words were starting in a visible state (dimmed/faded) rather than completely invisible, and weren't disappearing after being spoken

## Solution

### Phase 1: Use Actual Word Timestamps
We pass actual word-level timestamps from the transcript through to the subtitle renderer:

#### 1. Extended `SRTEntry` Model
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

#### 2. Updated `_word_level_entries()` in `subtitle_generator.py`
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

### Phase 2: Make Words Appear/Disappear at Exact Times
Updated `_build_word_by_word_line()` to make words truly invisible until spoken, then disappear after:

#### BUBBLE Style
```python
# Word starts invisible, appears and scales up when spoken, then disappears
f"{{\\alpha&HFF&"  # Start invisible
f"\\t({start_cs},{start_cs + 50},\\1c{highlight_color}\\alpha&H00&\\fscx110\\fscy110)"  # Appear + highlight + scale
f"\\t({end_cs},{end_cs + 50},\\alpha&HFF&)}}"  # Disappear
```

#### POPUP Style
```python
# Word starts invisible, pops in when spoken, then disappears
f"{{\\alpha&HFF&"  # Start invisible
f"\\t({start_cs},{start_cs + 80},\\alpha&H00&\\fscx100\\fscy100)"  # Pop in
f"\\t({end_cs},{end_cs + 50},\\alpha&HFF&)}}"  # Disappear
```

#### HIGHLIGHT Style
```python
# Word starts invisible, appears with highlight, then disappears
f"{{\\alpha&HFF&"  # Start invisible
f"\\t({start_cs},{start_cs + 50},\\1c{highlight_color}\\alpha&H00&\\bord6)"  # Appear + highlight
f"\\t({end_cs},{end_cs + 50},\\alpha&HFF&)}}"  # Disappear
```

#### KARAOKE Style
Uses native `\k` tags which already handle timing correctly.

## Result
Now each word:
1. **Starts completely invisible** (`\alpha&HFF&`)
2. **Appears at the exact moment it's spoken** (using actual word timestamps)
3. **Highlights/animates while being spoken** (style-specific effects)
4. **Disappears shortly after being spoken** (50-80ms fade out)

This creates true TikTok-style word-by-word subtitles where only the currently spoken word is visible.

## Files Modified
- `pipeline/models.py` - Added `word_timings` field to `SRTEntry`
- `pipeline/subtitle_generator.py` - Updated `_word_level_entries()` to extract and pass word timings
- `pipeline/animated_subtitle_renderer.py` - Updated `_build_word_by_word_line()` to:
  - Use actual word timings instead of equal division
  - Make words invisible until spoken
  - Make words disappear after being spoken

## Testing
Test by:
1. Running the pipeline with subtitle burning enabled
2. Checking that words only appear when they're being spoken
3. Verifying words disappear shortly after being spoken
4. Confirming all 4 subtitle styles (bubble, popup, highlight, karaoke) work correctly
5. Ensuring no words are visible before or long after they're spoken
