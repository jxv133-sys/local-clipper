# Word-by-Word Subtitle Highlighting Fix

## Problem

Subtitles were appearing as complete phrases (1-4 words) that popped up and disappeared all at once, like traditional captions. This is not the TikTok/YouTube Shorts style.

**User feedback:** "the words should come up as they are said. right now its more like captions in a way you know? like once the word is said the subtitel pops up and goes away after being said"

## Expected Behavior (TikTok Style)

In TikTok/YouTube Shorts:
1. **All words in a phrase appear at once** (1-4 words visible)
2. **Each word is highlighted individually** as it's spoken
3. **Words stay on screen** until the entire phrase is done
4. **Next phrase replaces** the previous one

**Example timeline:**
```
0.0s: "WATCH THIS MOMENT" appears (all 3 words, dimmed)
0.0s: "WATCH" highlights (bright/scaled)
0.5s: "THIS" highlights (bright/scaled), "WATCH" dims
1.0s: "MOMENT" highlights (bright/scaled), "THIS" dims
1.5s: All words disappear, next phrase appears
```

## Root Cause

The old implementation treated each phrase as a single unit:
- Entire phrase appeared with one animation
- No individual word timing
- All words had the same visual state at any given time

## Solution

Created `_build_word_by_word_line()` function that:
1. Shows all words in the phrase at once
2. Applies individual timing to each word
3. Uses ASS `\t()` (transition) tags for word-by-word effects
4. Implements style-specific highlighting for each word

### Style-Specific Implementations

#### BUBBLE Style
```
- All words start dimmed (50% opacity)
- Each word: brightens + scales to 110% when spoken
- After spoken: dims back to 50% opacity + scales to 100%
- Creates a "wave" effect across the phrase
```

#### POPUP Style
```
- All words start invisible
- Each word: pops in (fade + scale) when spoken
- Once visible, stays visible until phrase ends
- Creates a "building" effect
```

#### HIGHLIGHT Style
```
- All words start with normal color
- Each word: gets highlight color + thick border when spoken
- After spoken: returns to normal color
- Creates a "spotlight" effect
```

#### KARAOKE Style
```
- Uses native ASS \k tags for color change
- Each word changes color as it's spoken
- Classic karaoke effect
```

## Technical Implementation

### ASS Animation Tags Used

**`\t(start,end,tags)`** - Transition
- Animates from current state to new state over time
- `start` and `end` are in centiseconds relative to subtitle start
- Example: `\t(0,50,\1c&H00FFFF&)` - transition to yellow over 50cs

**`\1c&HBBGGRR&`** - Primary color
- Changes text color
- Format: &H + alpha + blue + green + red (hex)

**`\alpha&HXX&`** - Transparency
- &H00& = fully opaque
- &HFF& = fully transparent
- &H80& = 50% transparent

**`\fscx` / `\fscy`** - Scale
- 100 = normal size
- 110 = 110% size
- 0 = invisible (for popup effect)

**`\bord`** - Border width
- Thickness of text outline in pixels

### Word Timing Calculation

```python
total_cs = round((entry.end - entry.start) * 100)  # Total duration in centiseconds
word_cs = round(total_cs / len(words))  # Duration per word

for i, word in enumerate(words):
    start_time = i * word_cs  # When this word starts highlighting
    end_time = (i + 1) * word_cs  # When this word stops highlighting
```

## Code Changes

**File:** `pipeline/animated_subtitle_renderer.py`

### Added Function
```python
def _build_word_by_word_line(
    entry: SRTEntry,
    cx: int,
    subtitle_y: int,
    style: SubtitleStyle,
    config,
) -> str:
    """Build ASS Dialogue with word-by-word highlighting (TikTok style)."""
```

### Modified Function
```python
def generate_ass_file(...):
    # OLD: Different code for each style
    if style == SubtitleStyle.BUBBLE:
        text = f"{{...}}{escaped_text}"  # Entire phrase at once
    
    # NEW: Unified word-by-word highlighting
    text = _build_word_by_word_line(entry, cx, subtitle_y, style, config)
```

## Visual Comparison

### Before (Caption Style)
```
Frame 1: [empty]
Frame 2: "WATCH THIS MOMENT" (all words appear, all bright)
Frame 3: "WATCH THIS MOMENT" (all words visible, all bright)
Frame 4: [empty] (all words disappear)
```

### After (TikTok Style)
```
Frame 1: [empty]
Frame 2: "WATCH THIS MOMENT" (all visible, "WATCH" bright, others dim)
Frame 3: "WATCH THIS MOMENT" (all visible, "THIS" bright, others dim)
Frame 4: "WATCH THIS MOMENT" (all visible, "MOMENT" bright, others dim)
Frame 5: [empty] (all words disappear, next phrase appears)
```

## Testing

To verify the fix:

1. **Restart server:** `python3 web_server.py`
2. **Hard refresh browser:** Cmd+Shift+R
3. **Process clips with subtitles enabled**
4. **Play the output video**
5. **Verify:** Each word highlights individually as it's spoken

### Expected Behavior

**BUBBLE:** Words "pop" one by one with scale animation
**POPUP:** Words appear one by one, building the phrase
**HIGHLIGHT:** Words get highlighted one by one like a spotlight
**KARAOKE:** Words change color one by one (classic karaoke)

## Performance Impact

- **Minimal:** ASS rendering is handled by libass (highly optimized)
- **File size:** Slightly larger ASS files due to more animation tags
- **Encoding time:** No change (same FFmpeg process)

## Edge Cases Handled

1. **Single word phrases:** Works correctly (highlights the one word)
2. **Empty phrases:** Returns basic positioning tag
3. **Special characters:** Properly escaped before animation tags applied
4. **Very short durations:** Each word gets minimum highlight time

## Future Enhancements

Possible improvements:
1. **Variable word timing:** Use actual word timestamps instead of equal distribution
2. **Overlap prevention:** Ensure word highlights don't overlap
3. **Custom colors:** Allow per-style color customization
4. **Easing functions:** Add smooth acceleration/deceleration to animations
