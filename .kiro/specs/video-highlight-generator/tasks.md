# Implementation Plan: Video Highlight Generator

## Overview

Implement a local, offline Python pipeline that extracts audio from a video, transcribes it with Whisper, scores segments, selects highlights, extracts clips, and burns in subtitles. Each stage is a separate module under `pipeline/`. The implementation follows the staged architecture defined in the design document.

## Tasks

- [x] 1. Set up project structure, data models, and configuration
  - _Requirements: 11.1, 11.2, 11.3, 11.4_

- [x] 2. Implement Audio Extractor (`pipeline/audio_extractor.py`)
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6_

- [x] 3. Implement Transcriber (`pipeline/transcriber.py`)
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

- [x] 4. Implement Scorer — text and combination logic (`pipeline/scorer.py`)
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 6.1, 6.2, 6.4, 6.5_

- [x] 5. Implement Scorer — audio score (`pipeline/scorer.py`, continued)
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [x] 6. Implement Scorer — LLM score and full `score_segments` assembly (`pipeline/scorer.py`, continued)
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 6.1, 6.2, 6.3_

- [x] 7. Checkpoint — All scorer and transcriber tests pass (128 tests)

- [x] 8. Implement Clip Selector (`pipeline/clip_selector.py`)
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8_

- [x] 9. Implement Clip Extractor (`pipeline/clip_extractor.py`)
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7_

- [x] 10. Implement Subtitle Generator (`pipeline/subtitle_generator.py`)
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.7_

- [x] 11. Checkpoint — All pipeline tests pass

- [x] 12. Implement Pipeline Orchestrator (`main.py`)
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 11.1, 11.2, 11.3, 11.4_

- [x] 13. Add why-chosen report per clip (`pipeline/report_generator.py`)
  - Generates a `.txt` file alongside each clip explaining scores, keywords, energy, and transcript

- [x] 14. Add tkinter GUI (`gui.py`)
  - File picker, options panel, live progress log, results panel with open/why-chosen buttons

- [x] 15. Verify subtitle burn works end-to-end on real footage
  - Run pipeline on test footage and confirm subtitles are visible in output clips
  - The temp-path fix in `_burn_subtitles` must be confirmed working
  - If still failing, investigate FFmpeg `libass` availability on this system

- [x] 16. Clean up stray files and fix SSL workaround
  - Delete `clip_why_chosen.txt` from project root (stray file from a test run)
  - Bake `PYTHONHTTPSVERIFY=0` into `gui.py` and `main.py` so users don't need to set it manually
  - Add `output/` directory cleanup (currently has a stray `.DS_Store`)

- [x] 17. Write README.md with setup and usage instructions
  - Installation steps: FFmpeg via Homebrew, pip dependencies, SSL cert fix
  - How to run CLI: `python3 main.py input.mp4 --top-n 3`
  - How to run GUI: `python3 gui.py`
  - Description of output files (clips, SRT, why-chosen reports)
  - Known limitations (CPU-only Whisper speed, subtitle font requires libass)

- [x] 18. Final end-to-end test on real footage
  - Run full pipeline via GUI on `WATERPARK-SIMULATOR-DAY-4.mp4`
  - Confirm 3 clips exported to `~/Desktop/test-footage/highlights/`
  - Confirm subtitles are burned in and readable
  - Confirm why-chosen `.txt` files are accurate
  - Run `pytest tests/` — all 128 tests must pass

## Notes

- Tasks 1–14 are complete and pushed to GitHub
- Tasks 15–18 are the remaining work, prioritised by impact:
  - Task 15 is the highest priority — subtitle burn is the last known bug
  - Task 16 is quick cleanup that improves usability
  - Task 17 makes the project usable by anyone cloning the repo
  - Task 18 is the final sign-off
- The LLM scoring path is optional at runtime via `config.llm_enabled`
- All pushes go to `kiro/main` (plain `git push` works)
