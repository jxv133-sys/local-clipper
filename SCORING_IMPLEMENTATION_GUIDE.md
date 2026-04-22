# Viral Clip Scoring System - Implementation Guide

## Overview

This document explains how the multi-signal scoring system works and how to implement it correctly. The system uses a **hybrid approach** combining traditional signal processing, semantic analysis, and LLM-based hook detection to identify viral-worthy clips.

---

## 🏗️ Architecture: 5-Stage Pipeline

```
Video Input
    ↓
[1] Signal Extraction → Audio, Visual, Transcript
    ↓
[2] Semantic Analysis → Heuristics + Embeddings + LLM Hooks
    ↓
[3] Signal Fusion → Merge all signals into time-series windows
    ↓
[4] Clip Candidate Generation → Spike detection + scoring
    ↓
[5] Ranking & Selection → Genre/Platform-specific weighting
    ↓
Final Clips
```

---

## 📊 Stage 1: Signal Extraction

### 1.1 Audio Signals (`audio_analysis.py`)

**Extracted Features:**
- `excitement_score` = 0.6 × volume + 0.4 × pitch
- `silence_score` = 1.0 - volume (detects pauses)
- `volume_score` = RMS energy (normalized 0-1)
- `pitch_score` = F0 fundamental frequency (normalized 0-1)

**Temporal Resolution:** 0.5 second windows

**Implementation:**
```python
# Normalize using percentile clipping (prevents outliers)
p_low, p_high = np.percentile(pitch, [5, 95])
pitch_clipped = np.clip(pitch, p_low, p_high)
pitch_norm = (pitch_clipped - min) / (max - min + 1e-6)

# Weighted combination
excitement = 0.6 * volume_norm + 0.4 * pitch_norm
```

**Critical Fix Needed:**
- Add safety check for silent audio (all zeros) to prevent NaN

---

### 1.2 Visual Signals (`visual_analysis.py`)

**Extracted Features:**
- `face_score` = Face detection confidence × size
- `motion_score` = Frame difference intensity
- `visual_score` = 0.6 × face + 0.4 × motion

**Temporal Resolution:** 1 FPS (per frame)

**Implementation:**
```python
# Blended scoring (not max)
visual_score = (0.6 * face_score) + (0.4 * motion_score)

# Penalty for no-face clips
if face_score == 0.0:
    visual_score *= 0.6
```

**Events Detected:**
- `FACE_CHANGE` - Face appears/disappears (Δ > 0.15)
- `MOTION_SPIKE` - Sudden movement (Δ > 0.15)

---

### 1.3 Transcription (`audio_transcription.py`)

**Output:**
- Word-level timestamps (critical for precise cuts)
- Sentence segments
- Language detection

**Model:** Whisper `large-v3` (faster-whisper)

**No scoring** - Just provides temporal anchors

---

## 🧠 Stage 2: Semantic Analysis

### 2.1 Heuristic Scoring (`text_analysis.py`)

**Rule-Based Signals:**

| Pattern | Score Boost | Signal Name |
|---------|-------------|-------------|
| Contains "?" | +0.3 | Question |
| Word repetition | +0.2 | Repetition |
| "(laughter)" marker | +0.5 | Laughter |
| Hinglish keywords | +0.25 | Question (HI) |
| Story phrases | +0.2 | Story (HI) |
| Emotional words | +0.15 | Emotional |

**Implementation:**
```python
score = 0.0
signals = []

if "?" in text:
    score += 0.3
    signals.append("Question")

if any(word in text for word in HINGLISH_QUESTIONS):
    score += 0.25
    signals.append("Question (HI)")

# Cap at 1.0
return min(score, 1.0), signals
```

---

### 2.2 Semantic Novelty (Embeddings)

**Model:** `sentence-transformers/all-MiniLM-L6-v2`

**Calculation:**
```python
# 1. Encode all segments
embeddings = model.encode(texts)

# 2. Calculate centroid (average embedding)
centroid = embeddings.mean(dim=0)

# 3. Novelty = 1 - cosine_similarity(segment, centroid)
novelty_score = 1.0 - cos_sim(embedding, centroid)
```

**Interpretation:**
- High novelty (0.7-1.0) = Unique/surprising content
- Low novelty (0.0-0.3) = Repetitive/common content

---

### 2.3 LLM Hook Detection (`hook_detector.py`)

**Model:** `Qwen/Qwen2.5-1.5B-Instruct`

**Prompt Template:**
```
System: You are a viral content classifier.
Task: Determine if this text would stop a scrolling user in 3 seconds.
Rules:
- Score high (0.8-1.0) for curiosity, tension, controversy, surprise
- Score low (0.0-0.4) for greetings, filler, boring facts
- hook_type: ['question', 'contrarian', 'reveal', 'emotional', 'none']

User: Text: "{text_chunk}"
JSON:
```

**Output:**
```json
{
  "hook_score": 0.85,
  "hook_type": "question"
}
```

**Sliding Window:**
- Window size: 3 sentences
- Stride: 2 sentences (50% overlap)
- Minimum words: 5

**Threshold:** Only save hooks with score > 0.4

---

### 2.4 Hook Fusion Logic (`run_pipeline.py`)

**Critical Implementation:**
```python
# Get LLM hook score for time window
llm_hook = get_hook_score_at(t_start, t_end)

# Get base semantic score from heuristics + embeddings
base_semantic = segment.get('semantic_score', 0.0)

# Apply multiplicative boost (40% max boost)
hook_boost = 1.0 + (0.4 * llm_hook)
final_semantic = min(base_semantic * hook_boost, 1.0)

# Also track best hook phrase score
heuristic_hook = 0.8 if "Question" in signals else 0.0
final_hook_phrase = max(heuristic_hook, llm_hook)
```

**Why Multiplicative?**
- Additive would over-boost weak segments
- Multiplicative amplifies already-good content
- Cap at 1.0 prevents runaway scores

---

## 🔗 Stage 3: Signal Fusion

### 3.1 Smart Window Generation (`run_pipeline.py`)

**Strategy:** Generate overlapping candidate windows

```python
MIN_DURATION = 5.0   # Minimum clip length
MAX_DURATION = 90.0  # Maximum clip length

for i in range(len(segments)):
    for j in range(i, len(segments)):
        seg_start = segments[i]['start']
        seg_end = segments[j]['end']
        duration = seg_end - seg_start
        
        if duration > MAX_DURATION:
            break  # Stop expanding this window
            
        if duration >= MIN_DURATION:
            # Calculate aggregated signals for this window
            # Save as candidate
```

**Aggregation Methods:**

| Signal | Aggregation |
|--------|-------------|
| Audio excitement | Mean |
| Silence breaks | Ratio (count / total) |
| Semantic novelty | Weighted max (60% max + 40% mean) |
| Hook phrase | Max in window |
| Face presence | Mean of 3 samples (start, mid, end) |
| Motion | Mean of 3 samples |

---

### 3.2 Output Format (`signals.csv`)

**Required Columns:**
```csv
window_start,window_end,audio_excitation,speech_rate_change,silence_breaks,
semantic_novelty,sentiment_intensity,hook_phrase_score,face_presence,
face_motion,scene_change_rate,laughter_or_reaction
```

**All values normalized 0.0-1.0**

---

## ✂️ Stage 4: Clip Candidate Generation

### 4.1 Spike Detection (`clip_selector.py`)

**Algorithm:**
```python
spikes = []
for i in range(1, len(audio_analysis)):
    curr = audio_analysis[i]
    prev = audio_analysis[i-1]
    delta = curr['excitement_score'] - prev['excitement_score']
    
    # Detect spike
    if delta >= 0.08 or curr['excitement_score'] >= 0.62:
        spikes.append(curr)
```

**Thresholds:**
- Delta spike: ≥ 0.08 (8% increase)
- Absolute spike: ≥ 0.62 (62% excitement)

---

### 4.2 Clip Expansion

**Strategy:** Capture setup + punchline

```python
center_time = spike['time']

# Expand window
raw_start = max(0, center_time - 10)  # 10s before (setup)
raw_end = min(duration, center_time + 5)  # 5s after (reaction)

# Snap to sentence boundaries (clean cuts)
clip_start = find_sentence_start(raw_start)
clip_end = find_sentence_end(raw_end)
```

**Length Constraints:**
```python
MIN_CLIP_LEN = 10.0  # Reject if < 10s
MAX_CLIP_LEN = 45.0  # Truncate if > 45s
ABS_MAX_CLIP_LEN = 60.0  # Hard cap
```

---

### 4.3 Multi-Signal Scoring

**Formula:**
```python
# 1. Aggregate signals for clip window
avg_audio = mean(excitement_scores in window)
avg_semantic = 0.6 * max(semantic_scores) + 0.4 * mean(semantic_scores)
avg_visual = mean(visual_scores in window)

# 2. Weighted combination
base_score = (0.4 * avg_audio) + (0.3 * avg_semantic) + (0.3 * avg_visual)

# 3. Scale to 0-10
final_score = base_score * 10

# 4. Apply entropy penalty (audio validator)
if entropy < 2.8:
    final_score *= 0.75

# 5. Visual event bonus
if event_count > 2:
    final_score += 0.5
```

**Component Weights:**
- Audio: 40% (most reliable signal)
- Semantic: 30% (captures content quality)
- Visual: 30% (engagement factor)

---

### 4.4 Audio Validation (`audio_validator.py`)

**Purpose:** Detect low-quality audio (silence, noise)

**Features Extracted:**
- Energy entropy (dynamics)
- Zero-crossing rate (voice activity)

**Scoring:**
```python
entropy_score = min(avg_entropy / 4.0, 1.0)
zcr_score = min(avg_zcr / 0.15, 1.0)

confidence = (0.6 * entropy_score) + (0.4 * zcr_score)
```

**Interpretation:**
- Confidence < 0.5 → Likely silence/noise
- Confidence > 0.7 → Good audio quality

---

### 4.5 Filtering & Merging

**Two-Tier Threshold:**
```python
HARD_REJECT = 2.3  # Discard immediately
SOFT_ACCEPT = 3.0  # Preferred quality
```

**Merge Overlapping Clips:**
```python
# Merge if:
# 1. Gap < 6 seconds OR overlap exists
# 2. Resulting duration ≤ 60 seconds

if gap < MAX_MERGE_GAP and potential_duration <= ABS_MAX_CLIP_LEN:
    merge_clips()
    keep_max_score()
```

**Temporal NMS (Non-Maximum Suppression):**
```python
# IoU threshold: 0.35
# Keep highest scoring clips
# Remove overlaps > 35%

for clip in sorted_by_score:
    if all(iou(clip, selected) < 0.35 for selected in final):
        final.append(clip)
```

---

## 🎯 Stage 5: Genre & Platform Ranking

### 5.1 Genre-Specific Weights (`weights.json`)

**Structure:**
```json
{
  "signal_order": [
    "audio_excitation",
    "speech_rate_change",
    "silence_breaks",
    "semantic_novelty",
    "sentiment_intensity",
    "hook_phrase_score",
    "face_presence",
    "face_motion",
    "scene_change_rate",
    "laughter_or_reaction"
  ],
  "genres": {
    "podcast": [0.35, 0.05, 0.10, 0.25, 0.10, 0.05, 0.02, 0.03, 0.03, 0.02],
    "gaming": [0.40, 0.08, 0.05, 0.15, 0.12, 0.05, 0.05, 0.05, 0.03, 0.02],
    "comedy": [0.30, 0.05, 0.12, 0.20, 0.15, 0.08, 0.03, 0.02, 0.03, 0.02]
  },
  "default_cutoffs": {
    "podcast": 0.70,
    "gaming": 0.75,
    "comedy": 0.65
  },
  "min_spacing_seconds": 20
}
```

**Weight Design Principles:**
- Weights should sum to ~1.0 per genre
- Emphasize genre-defining signals
- De-emphasize irrelevant signals

**Example Reasoning:**
- **Podcast:** High semantic_novelty (0.25), low face_presence (0.02)
- **Gaming:** High audio_excitation (0.40), moderate sentiment (0.12)
- **Comedy:** High silence_breaks (0.12), high sentiment (0.15)

---

### 5.2 Platform-Specific Boosts (`rank_clips.py`)

**Profiles:**
```python
PLATFORM_PROFILES = {
    "tiktok": {
        "boost": {
            "audio_excitation": 1.25,      # +25% if strong
            "hook_phrase_score": 1.30,     # +30% if strong
            "face_motion": 1.20            # +20% if strong
        },
        "penalty": {
            "silence_breaks": 0.70         # -30% if high
        },
        "constraints": {
            "min_length": 7.0,
            "max_length": 45.0
        }
    },
    "shorts": {
        "boost": {
            "semantic_novelty": 1.25
        },
        "penalty": {
            "scene_change_rate": 0.80
        },
        "constraints": {
            "min_length": 10.0,
            "max_length": 60.0
        }
    }
}
```

**Application Logic:**
```python
# 1. Base genre score
base_score = dot_product(signals, genre_weights)

# 2. Apply platform boosts
bias_multiplier = 1.0

for signal_name, boost_factor in profile["boost"].items():
    signal_value = window[signal_name]
    if signal_value > 0.6:  # Only boost if strong
        bias_multiplier *= (1.0 + (boost_factor - 1.0) * signal_value)

# 3. Apply platform penalties
for signal_name, penalty_factor in profile["penalty"].items():
    signal_value = window[signal_name]
    if signal_value > 0.6:  # Only penalize if strong
        bias_multiplier *= penalty_factor

# 4. Final score (0-10 scale)
final_score = base_score * bias_multiplier * 10.0
final_score = min(final_score, 10.0)
```

---

### 5.3 Cutoff & Spacing

**Score Cutoff:**
```python
# Cutoff is 0-1 in config, scale to 0-10
scaled_cutoff = cutoff * 10.0

# Relax by 20% (empirical tuning)
threshold = scaled_cutoff * 0.8

filtered = [clip for clip in clips if clip['score'] >= threshold]
```

**Minimum Spacing:**
```python
# Prevent clips too close together
MIN_SPACING = 20  # seconds

# Greedy selection (highest score first)
selected = []
for clip in sorted_by_score_desc:
    if all(abs(clip['start'] - s['start']) >= MIN_SPACING for s in selected):
        selected.append(clip)
```

---

## 🔧 Implementation Checklist

### Critical Files to Create

1. **`weights.json`** (MISSING - CRITICAL)
   - Define genre weights
   - Set cutoff thresholds
   - Configure spacing

2. **Fix `video_ingestion.py`**
   - Handle yt-dlp format codes in filenames
   - Consistent error handling (exceptions vs sys.exit)

3. **Fix `run_pipeline.py`**
   - Only pass `--workspace` to scripts that support it
   - Or update all scripts to accept and ignore unknown args

4. **Add to `requirements.txt`**
   - `pyAudioAnalysis` (missing dependency)

---

## 📈 Tuning Guidelines

### Adjusting Sensitivity

**More Clips (Higher Recall):**
- Lower `HARD_REJECT` threshold (2.3 → 2.0)
- Lower spike detection thresholds (0.08 → 0.06)
- Reduce `min_spacing` (20s → 15s)

**Fewer, Better Clips (Higher Precision):**
- Raise `SOFT_ACCEPT` threshold (3.0 → 3.5)
- Raise spike detection thresholds (0.08 → 0.10)
- Increase `min_spacing` (20s → 30s)

### Genre-Specific Tuning

**Run Ablation Study:**
```bash
python evaluate_pipeline.py --genre podcast
```

**Interpret Results:**
- If precision low → Increase cutoff or weights on quality signals
- If recall low → Decrease cutoff or add more signal types
- If F1 drops when removing signal → That signal is important

### Platform Optimization

**Test on Real Data:**
1. Export 10-20 clips per platform
2. Manually label quality (1-5 stars)
3. Correlate scores with labels
4. Adjust boost/penalty factors

---

## 🚨 Common Pitfalls

### 1. **Overfitting to LLM Hooks**
- **Problem:** LLM hook score dominates, ignores audio/visual
- **Fix:** Cap hook boost at 40% (current implementation)

### 2. **Silent Audio False Positives**
- **Problem:** Silence gets high scores due to NaN/Inf
- **Fix:** Add `if rms.max() == rms.min(): return default_score`

### 3. **Overlapping Clips**
- **Problem:** Same moment selected multiple times
- **Fix:** Temporal NMS with IoU threshold 0.35

### 4. **Genre Mismatch**
- **Problem:** Using "podcast" weights on gaming video
- **Fix:** Implement auto-detection or require manual genre input

### 5. **Platform Constraint Violations**
- **Problem:** 90s clip sent to TikTok (max 45s)
- **Fix:** Apply `filter_by_constraints()` BEFORE scoring

---

## 📊 Expected Performance

### Typical Output (60-min podcast)

| Stage | Count |
|-------|-------|
| Audio spikes detected | 150-300 |
| Clip candidates generated | 80-150 |
| After scoring (>2.3) | 30-60 |
| After NMS | 15-30 |
| After spacing (20s) | 8-15 |

### Score Distribution

- **8.0-10.0:** Viral-worthy (top 5%)
- **6.0-7.9:** Good quality (top 20%)
- **4.0-5.9:** Acceptable (top 50%)
- **2.3-3.9:** Marginal (borderline)
- **0.0-2.2:** Rejected

---

## 🎓 Key Takeaways

1. **Multi-signal is essential** - No single signal predicts virality
2. **LLM is strategic** - Only used for hook detection (10% of pipeline)
3. **Genre matters** - Podcast ≠ Gaming ≠ Comedy
4. **Platform matters** - TikTok ≠ Shorts ≠ Reels
5. **Thresholds are tunable** - Start conservative, relax based on data
6. **Validation prevents garbage** - Audio entropy catches silent clips
7. **Spacing prevents redundancy** - 20s minimum between clips

---

## 📚 Further Reading

- **Audio Analysis:** librosa documentation
- **Semantic Embeddings:** sentence-transformers docs
- **LLM Prompting:** Qwen model card
- **Computer Vision:** MediaPipe face detection guide
- **Ranking Theory:** Learning to Rank (LTR) algorithms

---

**Last Updated:** 2026-04-22  
**Version:** 1.0  
**Status:** Implementation guide based on codebase analysis
