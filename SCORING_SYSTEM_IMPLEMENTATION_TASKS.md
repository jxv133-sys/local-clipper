# Scoring System Implementation Tasks

## Overview
This document outlines the tasks needed to implement the new viral clip scoring system based on the SCORING_IMPLEMENTATION_GUIDE.md. The new system uses a 5-stage pipeline with multi-signal fusion, semantic analysis, and LLM hook detection.

---

## Phase 1: Foundation & Signal Extraction

### Task 1.1: Audio Signal Extraction Enhancement
**Current State:** Basic RMS energy calculation exists in `pipeline/scorer.py`
**New Requirements:**
- [ ] Extract `excitement_score` = 0.6 × volume + 0.4 × pitch
- [ ] Extract `silence_score` = 1.0 - volume
- [ ] Extract `pitch_score` using F0 fundamental frequency
- [ ] Use 0.5 second temporal windows (currently using segment-based)
- [ ] Add percentile clipping (5th-95th) to prevent outlier distortion
- [ ] Add safety check for silent audio (all zeros) to prevent NaN
- [ ] Store raw audio features for later validation

**Files to Modify:**
- `pipeline/scorer.py` - Add new `compute_audio_features()` function
- `pipeline/models.py` - Add `AudioFeatures` dataclass

**Dependencies:**
- `librosa` for pitch extraction (F0)
- `numpy` for percentile calculations

---

### Task 1.2: Visual Signal Extraction (NEW)
**Current State:** No visual analysis exists
**New Requirements:**
- [ ] Implement face detection using MediaPipe
- [ ] Calculate `face_score` = confidence × size
- [ ] Calculate `motion_score` from frame differences
- [ ] Calculate `visual_score` = 0.6 × face + 0.4 × motion
- [ ] Sample at 1 FPS (not every frame)
- [ ] Detect events: `FACE_CHANGE`, `MOTION_SPIKE` (Δ > 0.15)
- [ ] Apply 0.6 penalty for no-face clips

**Files to Create:**
- `pipeline/visual_analyzer.py` - New module for visual analysis

**Files to Modify:**
- `pipeline/models.py` - Add `VisualFeatures` dataclass
- `web_server.py` - Add visual analysis to pipeline

**Dependencies:**
- `mediapipe` for face detection
- `opencv-python` (cv2) for frame processing

---

### Task 1.3: Transcription Enhancement
**Current State:** Whisper transcription exists with word timestamps
**New Requirements:**
- [ ] Verify word-level timestamps are being captured
- [ ] Add sentence segmentation (currently only has segments)
- [ ] Add language detection output
- [ ] Ensure timestamps are precise for clean cuts

**Files to Modify:**
- `pipeline/transcriber.py` - Add sentence boundary detection
- `pipeline/models.py` - Add `sentence_boundaries` to Transcript

**Status:** Mostly complete, needs sentence boundary detection

---

## Phase 2: Semantic Analysis

### Task 2.1: Heuristic Text Scoring Enhancement
**Current State:** Basic keyword matching exists
**New Requirements:**
- [ ] Add question detection ("?" pattern) → +0.3 score
- [ ] Add word repetition detection → +0.2 score
- [ ] Add laughter marker detection "(laughter)" → +0.5 score
- [ ] Add Hinglish keyword support → +0.25 score
- [ ] Add story phrase detection → +0.2 score
- [ ] Add emotional word detection → +0.15 score
- [ ] Return both score AND list of detected signals
- [ ] Cap final score at 1.0

**Files to Modify:**
- `pipeline/scorer.py` - Enhance `compute_text_score()` function
- `config.py` - Add new keyword lists (Hinglish, story phrases, emotional words)

**Files to Create:**
- `pipeline/text_patterns.py` - Pattern matching utilities

---

### Task 2.2: Semantic Novelty with Embeddings (NEW)
**Current State:** No embedding-based analysis exists
**New Requirements:**
- [ ] Integrate `sentence-transformers/all-MiniLM-L6-v2` model
- [ ] Encode all transcript segments
- [ ] Calculate centroid (average embedding)
- [ ] Calculate novelty = 1 - cosine_similarity(segment, centroid)
- [ ] High novelty (0.7-1.0) = unique content
- [ ] Low novelty (0.0-0.3) = repetitive content

**Files to Create:**
- `pipeline/semantic_analyzer.py` - New module for embeddings

**Files to Modify:**
- `pipeline/models.py` - Add `novelty_score` to ScoredSegment
- `requirements.txt` - Add `sentence-transformers`

**Dependencies:**
- `sentence-transformers`
- `torch` (if not already present)

---

### Task 2.3: LLM Hook Detection (NEW)
**Current State:** LLM is used for clip boundary refinement only
**New Requirements:**
- [ ] Integrate `Qwen/Qwen2.5-1.5B-Instruct` model (or keep Ollama)
- [ ] Implement sliding window (3 sentences, stride=2)
- [ ] Create hook detection prompt template
- [ ] Parse JSON response: `{hook_score, hook_type}`
- [ ] Only save hooks with score > 0.4
- [ ] Support hook types: question, contrarian, reveal, emotional, none
- [ ] Minimum 5 words per window

**Files to Create:**
- `pipeline/hook_detector.py` - New module for LLM hook detection

**Files to Modify:**
- `pipeline/scorer.py` - Integrate hook scores
- `config.py` - Add hook detection settings

**Decision Needed:**
- Use Ollama (current) or switch to Qwen model?
- If Ollama: Create new prompt for hook detection
- If Qwen: Add new model dependency

---

### Task 2.4: Hook Fusion Logic
**Current State:** No hook fusion exists
**New Requirements:**
- [ ] Get LLM hook score for time window
- [ ] Get base semantic score from heuristics + embeddings
- [ ] Apply multiplicative boost: `hook_boost = 1.0 + (0.4 * llm_hook)`
- [ ] Calculate: `final_semantic = min(base_semantic * hook_boost, 1.0)`
- [ ] Track best hook phrase score: `max(heuristic_hook, llm_hook)`
- [ ] Use multiplicative (not additive) to avoid over-boosting weak segments

**Files to Modify:**
- `pipeline/scorer.py` - Add `fuse_hook_scores()` function

---

## Phase 3: Signal Fusion & Window Generation

### Task 3.1: Smart Window Generation
**Current State:** Clips are expanded from seed segments
**New Requirements:**
- [ ] Generate overlapping candidate windows (not just seed-based)
- [ ] MIN_DURATION = 5.0s (currently 30s)
- [ ] MAX_DURATION = 90.0s (currently 100s)
- [ ] For each segment pair (i, j), create window if duration in range
- [ ] Stop expanding when duration > MAX_DURATION
- [ ] Calculate aggregated signals for each window

**Files to Modify:**
- `pipeline/clip_selector.py` - Replace current expansion logic
- `config.py` - Update min/max duration defaults

---

### Task 3.2: Signal Aggregation Methods
**Current State:** Simple averaging exists
**New Requirements:**
- [ ] Audio excitement → Mean
- [ ] Silence breaks → Ratio (count / total)
- [ ] Semantic novelty → Weighted max (60% max + 40% mean)
- [ ] Hook phrase → Max in window
- [ ] Face presence → Mean of 3 samples (start, mid, end)
- [ ] Motion → Mean of 3 samples
- [ ] Implement different aggregation strategies per signal type

**Files to Create:**
- `pipeline/signal_aggregator.py` - Aggregation utilities

---

### Task 3.3: Signals CSV Export (Optional)
**Current State:** No CSV export exists
**New Requirements:**
- [ ] Export all signals to `signals.csv` for analysis
- [ ] Columns: window_start, window_end, audio_excitation, speech_rate_change, silence_breaks, semantic_novelty, sentiment_intensity, hook_phrase_score, face_presence, face_motion, scene_change_rate, laughter_or_reaction
- [ ] All values normalized 0.0-1.0
- [ ] Useful for debugging and tuning

**Files to Create:**
- `pipeline/signal_exporter.py` - CSV export utilities

---

## Phase 4: Clip Candidate Generation

### Task 4.1: Spike Detection Enhancement
**Current State:** Spike detection exists based on RMS
**New Requirements:**
- [ ] Detect delta spikes: Δ ≥ 0.08 (8% increase)
- [ ] Detect absolute spikes: excitement ≥ 0.62 (62% threshold)
- [ ] Use excitement_score (not just RMS)
- [ ] Store spike metadata for later analysis

**Files to Modify:**
- `pipeline/scorer.py` - Update spike detection thresholds
- `config.py` - Make thresholds configurable

---

### Task 4.2: Clip Expansion Strategy
**Current State:** Biased expansion (reaction-first) exists
**New Requirements:**
- [ ] Expand 10s before spike (setup)
- [ ] Expand 5s after spike (reaction)
- [ ] Snap to sentence boundaries for clean cuts
- [ ] MIN_CLIP_LEN = 10.0s (currently 30s)
- [ ] MAX_CLIP_LEN = 45.0s (currently 100s)
- [ ] ABS_MAX_CLIP_LEN = 60.0s (hard cap)

**Files to Modify:**
- `pipeline/clip_selector.py` - Update expansion logic
- `config.py` - Update duration constraints

---

### Task 4.3: Multi-Signal Scoring Formula
**Current State:** Weighted combination exists (text + audio + LLM)
**New Requirements:**
- [ ] Aggregate signals for clip window
- [ ] avg_audio = mean(excitement_scores)
- [ ] avg_semantic = 0.6 × max + 0.4 × mean (semantic scores)
- [ ] avg_visual = mean(visual_scores)
- [ ] base_score = 0.4 × audio + 0.3 × semantic + 0.3 × visual
- [ ] final_score = base_score × 10 (scale to 0-10)
- [ ] Apply entropy penalty if entropy < 2.8: score × 0.75
- [ ] Apply visual event bonus if event_count > 2: score + 0.5

**Files to Modify:**
- `pipeline/scorer.py` - Replace `combine_scores()` function
- `pipeline/models.py` - Update ScoredSegment to store component scores

---

### Task 4.4: Audio Validation (NEW)
**Current State:** No audio quality validation exists
**New Requirements:**
- [ ] Calculate energy entropy (dynamics)
- [ ] Calculate zero-crossing rate (voice activity)
- [ ] entropy_score = min(avg_entropy / 4.0, 1.0)
- [ ] zcr_score = min(avg_zcr / 0.15, 1.0)
- [ ] confidence = 0.6 × entropy + 0.4 × zcr
- [ ] Confidence < 0.5 → Likely silence/noise
- [ ] Confidence > 0.7 → Good audio quality

**Files to Create:**
- `pipeline/audio_validator.py` - Audio quality validation

**Dependencies:**
- `librosa` for entropy and ZCR calculation

---

### Task 4.5: Filtering & Merging Enhancement
**Current State:** Basic overlap resolution exists
**New Requirements:**
- [ ] HARD_REJECT = 2.3 (discard immediately)
- [ ] SOFT_ACCEPT = 3.0 (preferred quality)
- [ ] Merge if gap < 6 seconds AND duration ≤ 60s
- [ ] Temporal NMS with IoU threshold 0.35
- [ ] Keep highest scoring clips
- [ ] Remove overlaps > 35%

**Files to Modify:**
- `pipeline/clip_selector.py` - Update filtering thresholds
- `config.py` - Add HARD_REJECT and SOFT_ACCEPT thresholds

---

## Phase 5: Genre & Platform Ranking

### Task 5.1: Genre-Specific Weights (NEW)
**Current State:** No genre-specific weighting exists
**New Requirements:**
- [ ] Create `weights.json` configuration file
- [ ] Define signal order (10 signals)
- [ ] Define genre weights: podcast, gaming, comedy
- [ ] Weights should sum to ~1.0 per genre
- [ ] Set default cutoffs per genre
- [ ] Set min_spacing_seconds = 20

**Files to Create:**
- `weights.json` - Genre weight configuration

**Example Structure:**
```json
{
  "signal_order": [
    "audio_excitation", "speech_rate_change", "silence_breaks",
    "semantic_novelty", "sentiment_intensity", "hook_phrase_score",
    "face_presence", "face_motion", "scene_change_rate",
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

---

### Task 5.2: Platform-Specific Boosts (NEW)
**Current State:** No platform-specific logic exists
**New Requirements:**
- [ ] Define platform profiles: tiktok, shorts, reels
- [ ] Implement boost factors (1.25-1.30 for strong signals)
- [ ] Implement penalty factors (0.70-0.80 for weak signals)
- [ ] Apply boosts only if signal > 0.6
- [ ] Apply penalties only if signal > 0.6
- [ ] Set platform constraints (min/max length)

**Files to Create:**
- `pipeline/platform_ranker.py` - Platform-specific ranking

**Files to Modify:**
- `config.py` - Add platform profiles

---

### Task 5.3: Cutoff & Spacing Logic
**Current State:** Basic spacing exists (min_clip_spacing)
**New Requirements:**
- [ ] Scale cutoff from 0-1 to 0-10
- [ ] Relax cutoff by 20% (empirical tuning)
- [ ] Filter clips below threshold
- [ ] Apply minimum spacing (20s default)
- [ ] Greedy selection (highest score first)
- [ ] Prevent clips too close together

**Files to Modify:**
- `pipeline/clip_selector.py` - Update spacing logic
- `config.py` - Add cutoff_relaxation_factor = 0.8

---

## Phase 6: Integration & Configuration

### Task 6.1: Configuration Updates
**Current State:** Config has basic settings
**New Requirements:**
- [ ] Add genre selection (podcast/gaming/comedy)
- [ ] Add platform selection (tiktok/shorts/reels)
- [ ] Add signal weights per genre
- [ ] Add spike detection thresholds
- [ ] Add clip length constraints per platform
- [ ] Add audio validation thresholds
- [ ] Add hook detection settings

**Files to Modify:**
- `config.py` - Add all new configuration options

---

### Task 6.2: Web UI Updates
**Current State:** Basic web UI exists
**New Requirements:**
- [ ] Add genre dropdown (podcast/gaming/comedy)
- [ ] Add platform dropdown (tiktok/shorts/reels)
- [ ] Add advanced settings for signal weights
- [ ] Add visual analysis toggle (optional, slow)
- [ ] Add hook detection toggle
- [ ] Show multi-signal breakdown in results

**Files to Modify:**
- `web/index.html` - Add new UI controls
- `web_server.py` - Handle new parameters

---

### Task 6.3: Pipeline Orchestration
**Current State:** Pipeline runs sequentially
**New Requirements:**
- [ ] Stage 1: Extract audio, visual, transcript signals
- [ ] Stage 2: Run semantic analysis (heuristics + embeddings + hooks)
- [ ] Stage 3: Fuse signals into time-series windows
- [ ] Stage 4: Generate clip candidates with multi-signal scoring
- [ ] Stage 5: Apply genre/platform ranking
- [ ] Add progress tracking for each stage
- [ ] Add error handling for optional stages (visual, hooks)

**Files to Modify:**
- `web_server.py` - Update `_run_pipeline_for_job()` function
- `main.py` - Update CLI pipeline

---

## Phase 7: Dependencies & Requirements

### Task 7.1: Update Requirements
**Current State:** Basic dependencies exist
**New Requirements:**
- [ ] Add `sentence-transformers` for embeddings
- [ ] Add `mediapipe` for face detection
- [ ] Add `opencv-python` for video processing
- [ ] Add `librosa` for advanced audio analysis (if not present)
- [ ] Verify `torch` is included (for transformers)

**Files to Modify:**
- `requirements.txt`

---

### Task 7.2: Model Downloads
**Current State:** Whisper models auto-download
**New Requirements:**
- [ ] Auto-download sentence-transformers model on first run
- [ ] Auto-download MediaPipe face detection model
- [ ] Add model caching to avoid re-downloads
- [ ] Add model size warnings (embeddings ~80MB, face detection ~10MB)

**Files to Create:**
- `pipeline/model_manager.py` - Model download utilities

---

## Phase 8: Testing & Validation

### Task 8.1: Unit Tests
**New Requirements:**
- [ ] Test audio feature extraction (excitement, pitch, silence)
- [ ] Test visual feature extraction (face, motion)
- [ ] Test semantic novelty calculation
- [ ] Test hook detection (mock LLM responses)
- [ ] Test signal aggregation methods
- [ ] Test multi-signal scoring formula
- [ ] Test genre weight application
- [ ] Test platform boost/penalty logic

**Files to Create:**
- `tests/test_audio_features.py`
- `tests/test_visual_analyzer.py`
- `tests/test_semantic_analyzer.py`
- `tests/test_hook_detector.py`
- `tests/test_signal_aggregator.py`
- `tests/test_platform_ranker.py`

---

### Task 8.2: Integration Tests
**New Requirements:**
- [ ] Test full pipeline with sample video
- [ ] Test genre switching (podcast vs gaming)
- [ ] Test platform switching (tiktok vs shorts)
- [ ] Test with/without visual analysis
- [ ] Test with/without hook detection
- [ ] Verify output format matches expectations

**Files to Create:**
- `tests/test_full_pipeline.py`

---

### Task 8.3: Performance Benchmarks
**New Requirements:**
- [ ] Measure processing time per stage
- [ ] Measure memory usage
- [ ] Test on videos of different lengths (5min, 30min, 60min)
- [ ] Identify bottlenecks (likely: visual analysis, embeddings)
- [ ] Add caching for expensive operations

**Files to Create:**
- `tests/benchmark_pipeline.py`

---

## Phase 9: Documentation & Tuning

### Task 9.1: Update Documentation
**New Requirements:**
- [ ] Update README.md with new scoring system
- [ ] Document genre selection
- [ ] Document platform selection
- [ ] Document signal weights and tuning
- [ ] Add examples of good vs bad clips
- [ ] Add troubleshooting guide

**Files to Modify:**
- `README.md`

---

### Task 9.2: Tuning Guide
**New Requirements:**
- [ ] Create tuning guide for adjusting sensitivity
- [ ] Document how to increase recall (more clips)
- [ ] Document how to increase precision (better clips)
- [ ] Add ablation study instructions
- [ ] Add manual labeling workflow

**Files to Create:**
- `TUNING_GUIDE.md`

---

## Phase 10: Migration & Backward Compatibility

### Task 10.1: Backward Compatibility
**Current State:** Existing pipeline works
**New Requirements:**
- [ ] Add feature flag: `use_new_scoring_system = False` (default)
- [ ] Keep old scoring system as fallback
- [ ] Allow gradual migration
- [ ] Test both systems side-by-side

**Files to Modify:**
- `config.py` - Add feature flag
- `pipeline/scorer.py` - Add conditional logic

---

### Task 10.2: Migration Script
**New Requirements:**
- [ ] Create script to convert old config to new format
- [ ] Migrate existing clips to new scoring format
- [ ] Add validation to ensure no data loss

**Files to Create:**
- `scripts/migrate_to_new_scoring.py`

---

## Summary

### Total Tasks: 60+

### Estimated Effort:
- **Phase 1-2 (Signal Extraction & Semantic Analysis):** 2-3 weeks
- **Phase 3-4 (Fusion & Candidate Generation):** 1-2 weeks
- **Phase 5 (Genre & Platform Ranking):** 1 week
- **Phase 6-7 (Integration & Dependencies):** 1 week
- **Phase 8-9 (Testing & Documentation):** 1-2 weeks
- **Phase 10 (Migration):** 1 week

**Total: 7-10 weeks** (for full implementation)

### Priority Order:
1. **High Priority:** Phase 1-4 (Core scoring system)
2. **Medium Priority:** Phase 5-6 (Genre/Platform ranking)
3. **Low Priority:** Phase 7-10 (Polish, testing, migration)

### Quick Wins (Can implement immediately):
- Task 1.1: Audio signal enhancement (excitement, pitch)
- Task 2.1: Heuristic text scoring enhancement
- Task 4.2: Clip expansion strategy updates
- Task 5.1: Create weights.json for genre-specific weights

### Optional Features (Can skip initially):
- Task 1.2: Visual signal extraction (slow, requires video processing)
- Task 2.3: LLM hook detection (can use existing LLM for now)
- Task 3.3: Signals CSV export (debugging only)
- Task 5.2: Platform-specific boosts (nice-to-have)

---

## Next Steps

1. **Review this task list** with the team
2. **Prioritize phases** based on business needs
3. **Start with Phase 1** (audio signal extraction)
4. **Implement incrementally** - test each phase before moving on
5. **Gather feedback** from real clips to tune weights

---

**Created:** 2026-04-22  
**Based on:** SCORING_IMPLEMENTATION_GUIDE.md  
**Status:** Ready for implementation
