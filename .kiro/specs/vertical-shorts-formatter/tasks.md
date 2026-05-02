# Tasks

## Task List

- [x] 1. Add ShortsConfig fields to Config and wire --shorts CLI flag
  - [x] 1.1 Add all ShortsConfig fields to the Config dataclass in config.py with correct defaults
  - [x] 1.2 Add --shorts CLI flag to the argparse parser in main.py that sets shorts_enabled = True
  - [x] 1.3 Update build_config() in main.py to map the --shorts flag to config.shorts_enabled
  - [x] 1.4 Write unit tests for the new Config fields (defaults, validation) in tests/test_config.py

- [x] 2. Implement data models for the shorts formatter
  - [x] 2.1 Add FacecamRegion dataclass to pipeline/models.py
  - [x] 2.2 Add CanvasLayout dataclass to pipeline/models.py
  - [x] 2.3 Add FilterFragment dataclass to pipeline/models.py
  - [x] 2.4 Add SubtitleStyle enum to pipeline/models.py
  - [x] 2.5 Add ShortsFormattingError exception class to pipeline/exceptions.py
  - [x] 2.6 Write unit tests for the new data models in tests/test_models.py

- [x] 3. Implement FrameReformatter
  - [x] 3.1 Create pipeline/frame_reformatter.py with the FrameReformatter class
  - [x] 3.2 Implement compute_canvas_layout(config) → CanvasLayout satisfying all layout invariants
  - [x] 3.3 Implement build_canvas_filter(src_width, src_height, layout) → FilterFragment that scales gameplay to the bottom region and pads to the full canvas with black
  - [x] 3.4 Write unit tests for build_canvas_filter with 16:9, 4:3, 1:1, and 9:16 source aspect ratios
  - [x] 3.5 Write property-based tests using hypothesis for Property 1 (canvas dimensions always exact) and Property 2 (gameplay region positioning) and Property 3 (CanvasLayout invariants)

- [x] 4. Implement FacecamRelocator
  - [x] 4.1 Create pipeline/facecam_relocator.py with the FacecamRelocator class
  - [x] 4.2 Implement detect_facecam(clip_path, frame_width, frame_height) → FacecamRegion | None using ffmpeg cropdetect on the first facecam_sample_duration seconds
  - [x] 4.3 Implement classify_region(x, y, w, h, frame_w, frame_h) → str | None with area fraction validation and corner classification
  - [x] 4.4 Implement build_facecam_filter(region, canvas_width, canvas_height, top_third_height) → FilterFragment that crops, scales, and overlays the facecam at (0, 0)
  - [x] 4.5 Implement the blur fallback filter for when detect_facecam returns None (crop + scale + boxblur of gameplay video filling the top third)
  - [x] 4.6 Write unit tests for classify_region with known corner coordinates and area fraction boundary cases
  - [x] 4.7 Write unit tests for detect_facecam with mocked ffmpeg cropdetect output
  - [x] 4.8 Write property-based tests using hypothesis for Property 4 (area fraction filtering), Property 5 (corner classification consistency), Property 6 (confidence in [0.0, 1.0]), and Property 7 (overlay coordinates within canvas bounds)

- [x] 5. Implement AnimatedSubtitleRenderer
  - [x] 5.1 Create pipeline/animated_subtitle_renderer.py with the AnimatedSubtitleRenderer class
  - [x] 5.2 Implement generate_ass_file(srt_entries, style, output_path, canvas_width, canvas_height, gameplay_region_top) → str that writes a valid ASS file with the correct header sections
  - [x] 5.3 Implement the BUBBLE style: bold thick outline with \fscx110\fscy110\t(0,80,\fscx100\fscy100) scale-pop animation
  - [x] 5.4 Implement the POPUP style: \fad(80,0) fade-in with \fscx0\fscy0 → \fscx100\fscy100 scale-in animation
  - [x] 5.5 Implement the HIGHLIGHT style: active word group with \3c highlight colour and \bord6 background box
  - [x] 5.6 Implement the KARAOKE style: word-by-word colour change using ASS \k timing tags
  - [x] 5.7 Implement invalid entry filtering: skip SRTEntry objects with start >= end or negative timestamps and log a warning
  - [x] 5.8 Implement special character escaping for subtitle text inserted into ASS override tags
  - [x] 5.9 Implement build_subtitle_filter(srt_entries, style, canvas_width, canvas_height, gameplay_region_top) → FilterFragment that references the generated ASS file
  - [x] 5.10 Write unit tests for generate_ass_file for each SubtitleStyle, empty entry list, and invalid timing entries
  - [x] 5.11 Write property-based tests using hypothesis for Property 10 (ASS file always has valid structure), Property 11 (invalid entries excluded), Property 12 (text always uppercased), and Property 13 (subtitle positions within canvas bounds)

- [x] 6. Implement ShortsFormatter orchestrator
  - [x] 6.1 Create pipeline/shorts_formatter.py with the ShortsFormatter class
  - [x] 6.2 Implement derive_shorts_path(clip_path) → str that appends _shorts to the clip stem
  - [x] 6.3 Implement collect_srt_entries(clip, transcript) → list[SRTEntry] that filters and time-adjusts transcript segments overlapping the clip window
  - [x] 6.4 Implement format_single_clip(config, clip, clip_path, srt_entries) → str that assembles the filter_complex from all three sub-components and runs a single FFmpeg invocation with libx264 -preset fast -crf 23 and audio copy
  - [x] 6.5 Implement format_clips(config, clips, clip_paths, transcript) → list[str] using ThreadPoolExecutor for parallel processing, returning results in rank order
  - [x] 6.6 Ensure format_single_clip raises ShortsFormattingError (not a generic exception) on non-zero FFmpeg exit, preserving the original clip file
  - [x] 6.7 Write unit tests for derive_shorts_path, collect_srt_entries (overlap logic, time adjustment, empty transcript), and format_clips error handling
  - [x] 6.8 Write property-based tests using hypothesis for Property 8 (SRT entries time-adjusted and non-negative), Property 9 (entries only from clip window), and Property 14 (shorts path derived from clip stem)

- [x] 7. Integrate Stage 8 into the main pipeline
  - [x] 7.1 Import ShortsFormatter in main.py and add Stage 8 call in run_pipeline() after Stage 7, guarded by config.shorts_enabled
  - [x] 7.2 Pass the shorts_paths to the ReportGenerator so reports reference both the original and shorts clip paths
  - [x] 7.3 Print the shorts output paths in the final summary in main()
  - [x] 7.4 Catch ShortsFormattingError in run_pipeline() so a shorts failure is logged but does not terminate the main pipeline
  - [x] 7.5 Write integration tests in tests/test_shorts_formatter.py using a synthetic FFmpeg testsrc video to verify the end-to-end output is 1080×1920 with audio

- [x] 8. Write remaining property-based tests
  - [x] 8.1 Verify Property 1 with hypothesis: for any (src_w, src_h) in [1, 7680] × [1, 4320], build_canvas_filter filter_str specifies canvas_width × canvas_height
  - [x] 8.2 Verify Property 3 with hypothesis: for any shorts_width, shorts_height > 0 and facecam_top_fraction in (0.0, 1.0), all five CanvasLayout invariants hold
  - [x] 8.3 Verify Property 10 with hypothesis: for any list of SRTEntry objects (including empty and mixed valid/invalid), generate_ass_file output contains all three required ASS sections and all Dialogue lines have start < end
