# Changelog

## Documentation Cleanup - May 2026

### Archived Implementation Notes
The following implementation-specific documentation has been moved to `docs/archive/`:

- **Vertical Formatting**: Implementation details, bug fixes, performance improvements, UX enhancements
- **Subtitle System**: Implementation notes, testing checklists, word timing fixes
- **LLM Improvements**: Boundary refinement fixes, performance optimizations
- **Feature Additions**: Clip deletion, volume spike naming, session endpoints

These files are preserved for historical reference but are no longer actively maintained.

### Active Documentation
- `README.md` - Main project documentation and setup guide
- `CLI_CHEATSHEET.md` - Command-line reference
- `.kiro/specs/` - Active feature specifications and task lists

---

## Project Status

### Current System
The video highlight generator uses a multi-phase scoring system:
1. **Phase 1 (Local)**: Text + audio scoring on all segments
2. **Phase 2 (LLM)**: Quality refinement on top candidates
3. **Clip Selection**: Expansion, overlap handling, boundary refinement
4. **Vertical Formatting**: 9:16 conversion with facecam detection
5. **Subtitle Burning**: Animated captions with word-level timing

### Known Areas for Improvement
See `.kiro/specs/clip-selection-improvements/` for the comprehensive improvement plan.
