# Scoring System Implementation Tasks

## Overview
This document outlines the tasks needed to implement the new viral clip scoring system based on the SCORING_IMPLEMENTATION_GUIDE.md. The new system uses a 5-stage pipeline with multi-signal fusion, semantic analysis, and LLM hook detection.

---

## Phase 1: Foundation & Signal Extraction

### Task 1.1: Audio Signal Extraction Enhancement ✅ COMPLETE
**Files modified:** `pipeline/scorer.py`, `pipeline/models.py`, `config.py`, `requirements.txt`

- [x] Extract `excitement_score` = 0.6 × volume + 0.4 × pitch
- [x] Extract `silence_score` = 1.0 - volume
- [x] Extract `pitch_score` using F0 fundamental frequency (librosa pyin)
- [x] Use 0.5 second temporal windows (configurable via `audio_feature_window`)
- [x] Add percentile clipping (5th-95th) to prevent outlier distortion
- [x] Add safety check for silent audio (all zeros) to prevent NaN
- [x] `AudioFeatures` dataclass added to `pipeline/models.py`
- [x] `compute_audio_features()` function added to `pipeline/scorer.py`
- [x] `librosa==0.10.1` added to `requirements.txt`
- [x] Tests: `tests/test_audio_features.py` (6/6 passing)

---

### Task 1.2: Visual Signal Extraction ⏭️ SKIPPED
Skipped per user request. No visual analysis will be implemented.

---

### Task 1.3: Transcription Enhancement ✅ MOSTLY COMPLETE
Word-level timestamps already captured by faster-whisper. Sentence boundary detection not added — Whisper segments are already sentence-like and sufficient for clean cuts.

---

## Phase 2: Semantic Analysis

### Task 2.1: Heuristic Text Scoring Enhancement ✅ COMPLETE
**Files created:** `pipeline/text_patterns.py`
**Files modified:** `pipeline/scorer.py`, `config.py`

- [x] Question detection ("?" pattern) → +0.3 score
- [x] Word repetition detection → +0.2 score
- [x] Laughter marker detection "(laughter)", "haha", "lol" → +0.5 score
- [x] Hinglish keyword support (kya, yaar, bhai, etc.) → +0.25 score
- [x] Story phrase detection ("so basically", "one time", etc.) → +0.2 score
- [x] Emotional word detection (love, hate, amazing, etc.) → +0.15 score
- [x] Returns both score AND list of detected signals
- [x] Score capped at 1.0
- [x] **Wired into `compute_text_score()`** via `config.text_pattern_weight` (default 0.3)
  - Formula: `final = (1 - w) * keyword_score + w * pattern_score`
  - Set `text_pattern_weight=0.0` to disable
- [x] Tests: `tests/test_text_patterns.py` (33/33 passing)

---

### Task 2.2: Semantic Novelty with Embeddings ❌ NOT STARTED
- [ ] Integrate `sentence-transformers/all-MiniLM-L6-v2` model
- [ ] Encode all transcript segments
- [ ] Calculate centroid (average embedding)
- [ ] Calculate novelty = 1 - cosine_similarity(segment, centroid)

**Files to create:** `pipeline/semantic_analyzer.py`
**Dependencies:** `sentence-transformers`, `torch`

---

### Task 2.3: LLM Hook Detection ✅ COMPLETE
**Files created:** `pipeline/hook_detector.py`
**Files modified:** `config.py`

- [x] Uses existing Ollama LLM infrastructure (no new model dependency)
- [x] Sliding window approach (3 sentences, stride=2, configurable)
- [x] Hook detection prompt returns JSON `{hook_score, hook_type}`
- [x] Saves hooks with score > 0.4 (configurable threshold)
- [x] Supports hook types: question, contrarian, reveal, emotional, none
- [x] Minimum 5 words per window (configurable)
- [x] Helper functions: `get_hook_score_at_time()`, `get_hook_score_for_window()`
- [x] Tests: `tests/test_hook_detector.py` (21/21 passing)

---

### Task 2.4: Hook Fusion Logic ✅ COMPLETE
**Files modified:** `pipeline/scorer.py`

- [x] Hook detection runs over full transcript before LLM window scoring
- [x] Multiplicative boost applied to text scores near detected hooks
  - Formula: `text_score *= (1 + hook_boost_max * hook_score)`, capped at 1.0
  - Default `hook_boost_max=0.4` (max 40% boost from a perfect hook)
- [x] Only runs when `llm_enabled=True` and `hook_detection_enabled=True`
- [x] No cost when LLM is disabled

---

## Phase 3: Signal Fusion & Window Generation

### Task 3.1: Smart Window Generation ❌ NOT STARTED
Current seed-based expansion remains. Overlapping candidate window generation not yet implemented.

- [ ] Generate overlapping candidate windows (not just seed-based)
- [ ] MIN_DURATION = 5.0s, MAX_DURATION = 90.0s
- [ ] Calculate aggregated signals for each window

---

### Task 3.2: Signal Aggregation Methods ❌ NOT STARTED
- [ ] Audio excitement → Mean
- [ ] Silence breaks → Ratio (count / total)
- [ ] Semantic novelty → Weighted max (60% max + 40% mean)
- [ ] Hook phrase → Max in window

**Files to create:** `pipeline/signal_aggregator.py`

---

### Task 3.3: Signals CSV Export ⏭️ OPTIONAL
Debugging utility — low priority.

---

## Phase 4: Clip Candidate Generation

### Task 4.1: Spike Detection Enhancement ⚠️ PARTIAL
Spike detection exists based on RMS. Not yet updated to use `excitement_score` from the new audio features.

- [ ] Use `excitement_score` (not just RMS) for spike detection
- [ ] Detect delta spikes: Δ ≥ 0.08 (8% increase in excitement)
- [ ] Detect absolute spikes: excitement ≥ 0.62

---

### Task 4.2: Clip Expansion Strategy ⚠️ PARTIAL
Biased expansion (reaction-first) already implemented. Duration constraints differ from guide targets.

- [ ] Consider reducing MIN_CLIP_LEN from 30s toward 10s for shorter viral clips
- [ ] Consider reducing MAX_CLIP_LEN from 100s toward 45-60s

---

### Task 4.3: Multi-Signal Scoring Formula ❌ NOT STARTED
Current formula: `text_weight * text + audio_weight * audio + llm_weight * llm + spike + burst`

Guide target:
- [ ] avg_audio = mean(excitement_scores)
- [ ] avg_semantic = 0.6 × max + 0.4 × mean (semantic scores)
- [ ] base_score = 0.4 × audio + 0.3 × semantic + 0.3 × visual
- [ ] Apply entropy penalty if entropy < 2.8

---

### Task 4.4: Audio Validation ❌ NOT STARTED
- [ ] Calculate energy entropy (dynamics)
- [ ] Calculate zero-crossing rate (voice activity)
- [ ] confidence = 0.6 × entropy + 0.4 × zcr
- [ ] Confidence < 0.5 → Likely silence/noise

**Files to create:** `pipeline/audio_validator.py`

---

### Task 4.5: Filtering & Merging Enhancement ❌ NOT STARTED
- [ ] HARD_REJECT = 2.3 threshold
- [ ] Temporal NMS with IoU threshold 0.35

---

## Phase 5: Genre & Platform Ranking

### Task 5.1: Genre-Specific Weights ❌ NOT STARTED
- [ ] Create `weights.json` with podcast/gaming/comedy weights
- [ ] Wire genre selection into scoring pipeline

---

### Task 5.2: Platform-Specific Boosts ❌ NOT STARTED
- [ ] TikTok, Shorts, Reels profiles with boost/penalty factors

---

### Task 5.3: Cutoff & Spacing Logic ⚠️ PARTIAL
Basic `min_clip_spacing` exists. Guide-style score cutoff with 20% relaxation not implemented.

---

## Phase 6: Integration & Configuration

### Task 6.1: Configuration Updates ⚠️ PARTIAL
New config fields added for audio features and hook detection. Genre/platform fields not yet added.

- [x] `text_pattern_weight: float = 0.3`
- [x] `hook_detection_enabled: bool = True`
- [x] `hook_boost_max: float = 0.4`
- [x] `hook_window_size: int = 3`
- [x] `hook_stride: int = 2`
- [x] `hook_min_words: int = 5`
- [x] `hook_score_threshold: float = 0.4`
- [x] `audio_feature_window: float = 0.5`
- [x] `audio_percentile_low/high`
- [x] `excitement_volume_weight / excitement_pitch_weight`
- [ ] Genre selection (podcast/gaming/comedy)
- [ ] Platform selection (tiktok/shorts/reels)

---

### Task 6.2: Web UI Updates ❌ NOT STARTED
- [ ] Genre dropdown
- [ ] Platform dropdown
- [ ] Hook detection toggle
- [ ] Multi-signal breakdown in results

---

### Task 6.3: Pipeline Orchestration ⚠️ PARTIAL
Hook detection and text patterns are now called inside `score_segments()`. Audio features extracted but not yet fed into the main scoring formula.

---

## Phase 7: Dependencies & Requirements

### Task 7.1: Update Requirements ⚠️ PARTIAL
- [x] `librosa==0.10.1` added
- [x] `yt-dlp>=2024.1.0` added (YouTube download feature)
- [ ] `sentence-transformers` (for Task 2.2)
- [ ] `torch` (for sentence-transformers)

---

## Phase 8: Testing & Validation

### Task 8.1: Unit Tests ⚠️ PARTIAL
- [x] `tests/test_audio_features.py` (6 tests)
- [x] `tests/test_text_patterns.py` (33 tests)
- [x] `tests/test_hook_detector.py` (21 tests)
- [ ] `tests/test_semantic_analyzer.py`
- [ ] `tests/test_signal_aggregator.py`
- [ ] `tests/test_platform_ranker.py`

**Total new tests: 60 — all passing ✅**

---

## Phases 9–10: Documentation, Migration ❌ NOT STARTED
Low priority. Will address after core scoring system is complete.

---

## Summary

| Phase | Task | Status |
|-------|------|--------|
| 1 | 1.1 Audio Signal Extraction | ✅ Complete |
| 1 | 1.2 Visual Signal Extraction | ⏭️ Skipped |
| 1 | 1.3 Transcription Enhancement | ✅ Mostly complete |
| 2 | 2.1 Heuristic Text Scoring | ✅ Complete + wired |
| 2 | 2.2 Semantic Novelty (Embeddings) | ❌ Not started |
| 2 | 2.3 LLM Hook Detection | ✅ Complete + wired |
| 2 | 2.4 Hook Fusion Logic | ✅ Complete |
| 3 | 3.1 Smart Window Generation | ❌ Not started |
| 3 | 3.2 Signal Aggregation | ❌ Not started |
| 3 | 3.3 Signals CSV Export | ⏭️ Optional |
| 4 | 4.1 Spike Detection Enhancement | ⚠️ Partial |
| 4 | 4.2 Clip Expansion Strategy | ⚠️ Partial |
| 4 | 4.3 Multi-Signal Scoring Formula | ❌ Not started |
| 4 | 4.4 Audio Validation | ❌ Not started |
| 4 | 4.5 Filtering & Merging | ❌ Not started |
| 5 | 5.1 Genre-Specific Weights | ❌ Not started |
| 5 | 5.2 Platform-Specific Boosts | ❌ Not started |
| 5 | 5.3 Cutoff & Spacing | ⚠️ Partial |
| 6 | 6.1 Config Updates | ⚠️ Partial |
| 6 | 6.2 Web UI Updates | ❌ Not started |
| 6 | 6.3 Pipeline Orchestration | ⚠️ Partial |
| 7 | 7.1 Requirements | ⚠️ Partial |
| 8 | 8.1 Unit Tests | ⚠️ Partial (60 new tests passing) |
| 9–10 | Docs, Migration | ❌ Not started |

### Recommended Next Steps (priority order)
1. **Task 4.3** — Multi-signal scoring formula (connect audio features to final score)
2. **Task 4.4** — Audio validation (entropy + ZCR quality check)
4. **Task 6.2** — Web UI genre/platform dropdowns
5. **Task 2.2** — Semantic novelty embeddings (optional, ~80MB model)

---

**Last Updated:** 2026-04-22
**Status:** Phase 1 complete, Phase 2 complete and wired, Phases 3–6 in progress
