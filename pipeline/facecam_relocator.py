"""FacecamRelocator: detects the facecam pip and builds FFmpeg filter fragments
to reposition it to the top third of the vertical 9:16 canvas."""

from __future__ import annotations

import logging
import re
import subprocess
from collections import Counter

from pipeline.models import FacecamRegion, FilterFragment

logger = logging.getLogger(__name__)


def classify_region(
    x: int,
    y: int,
    w: int,
    h: int,
    frame_w: int,
    frame_h: int,
    min_area_fraction: float = 0.04,
    max_area_fraction: float = 0.30,
) -> str | None:
    """Classify a crop region into a corner label, or return None if invalid.

    Validates that the region's area fraction is within [min_area_fraction,
    max_area_fraction], then classifies the region's centre into one of four
    corners relative to the frame midpoint.

    Args:
        x: Left edge of the region in source frame pixels.
        y: Top edge of the region in source frame pixels.
        w: Width of the region in pixels.
        h: Height of the region in pixels.
        frame_w: Source frame width in pixels.
        frame_h: Source frame height in pixels.
        min_area_fraction: Minimum area fraction (inclusive) for a valid pip.
        max_area_fraction: Maximum area fraction (inclusive) for a valid pip.

    Returns:
        One of "top-left", "top-right", "bottom-left", "bottom-right",
        or None if the area fraction is outside the valid range.
    """
    area_fraction = (w * h) / (frame_w * frame_h)

    if area_fraction < min_area_fraction or area_fraction > max_area_fraction:
        return None

    center_x = x + w / 2
    center_y = y + h / 2

    if center_x < frame_w / 2 and center_y < frame_h / 2:
        return "top-left"
    elif center_x >= frame_w / 2 and center_y < frame_h / 2:
        return "top-right"
    elif center_x < frame_w / 2 and center_y >= frame_h / 2:
        return "bottom-left"
    else:
        return "bottom-right"


class FacecamRelocator:
    """Detects the facecam pip region and builds FFmpeg filter fragments to
    reposition it to the top third of the vertical 9:16 canvas.

    Responsibilities:
    - Use ffmpeg cropdetect to locate the facecam pip in the source clip.
    - Classify which corner the pip occupies.
    - Build crop + scale + overlay filter chain for the detected facecam.
    - Provide a blur fallback when no facecam is detected.
    """

    def detect_facecam(
        self,
        clip_path: str,
        frame_width: int,
        frame_height: int,
        config,
    ) -> FacecamRegion | None:
        """Probe the clip to find the facecam pip region using ffmpeg cropdetect.

        Runs ffmpeg cropdetect on the first config.facecam_sample_duration seconds
        of the clip, parses all "crop=W:H:X:Y" patterns from stderr, finds the
        most frequently reported crop region, validates its area fraction, and
        classifies its corner.

        Args:
            clip_path:    Path to the source clip.
            frame_width:  Source frame width in pixels.
            frame_height: Source frame height in pixels.
            config:       Config object with facecam detection settings.

        Returns:
            FacecamRegion if a valid pip is detected, None otherwise.
        """
        cmd = [
            "ffmpeg",
            "-t", str(config.facecam_sample_duration),
            "-i", clip_path,
            "-vf", "cropdetect=24:16:0",
            "-f", "null",
            "-",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )
            stderr = result.stderr
        except FileNotFoundError:
            logger.warning("ffmpeg not found; cannot detect facecam")
            return None
        except OSError as exc:
            logger.warning("ffmpeg subprocess error during cropdetect: %s", exc)
            return None

        # Parse all "crop=W:H:X:Y" occurrences from stderr
        crop_pattern = re.compile(r"crop=(\d+):(\d+):(\d+):(\d+)")
        matches = crop_pattern.findall(stderr)

        if not matches:
            logger.info("No cropdetect output found; skipping facecam detection")
            return None

        # Count occurrences of each (W, H, X, Y) tuple
        crop_counter: Counter = Counter(
            (int(w), int(h), int(x), int(y)) for w, h, x, y in matches
        )

        # Find the most frequent crop region
        best_crop, best_count = crop_counter.most_common(1)[0]
        best_w, best_h, best_x, best_y = best_crop

        # Validate area fraction and classify corner
        corner = classify_region(
            x=best_x,
            y=best_y,
            w=best_w,
            h=best_h,
            frame_w=frame_width,
            frame_h=frame_height,
            min_area_fraction=config.facecam_min_area_fraction,
            max_area_fraction=config.facecam_max_area_fraction,
        )

        if corner is None:
            area_fraction = (best_w * best_h) / (frame_width * frame_height)
            logger.info(
                "Detected crop region area fraction %.3f is outside valid range "
                "[%.3f, %.3f]; no facecam detected",
                area_fraction,
                config.facecam_min_area_fraction,
                config.facecam_max_area_fraction,
            )
            return None

        total_crops = len(matches)
        confidence = best_count / total_crops

        logger.info(
            "Detected facecam at (%d, %d) size %dx%d corner=%s confidence=%.2f",
            best_x, best_y, best_w, best_h, corner, confidence,
        )

        return FacecamRegion(
            x=best_x,
            y=best_y,
            width=best_w,
            height=best_h,
            corner=corner,
            confidence=confidence,
        )

    def build_facecam_filter(
        self,
        region: FacecamRegion,
        canvas_width: int,
        canvas_height: int,
        top_third_height: int,
    ) -> FilterFragment:
        """Build an FFmpeg filter fragment that crops, scales, and overlays the facecam.

        The facecam is:
        1. Cropped from its source position in the original video.
        2. Scaled to fit within canvas_width × top_third_height while preserving
           the facecam's aspect ratio.
        3. Centred horizontally within the top region.
        4. Overlaid at (overlay_x, 0) on the canvas.

        Args:
            region:           Detected facecam region with source coordinates.
            canvas_width:     Width of the 9:16 canvas in pixels.
            canvas_height:    Height of the 9:16 canvas in pixels.
            top_third_height: Height of the top facecam region in pixels.

        Returns:
            FilterFragment with input_label "[0:v]" and output_label "[with_facecam]".
        """
        # Step 1: Crop the facecam from its source position
        crop_w = region.width
        crop_h = region.height
        crop_x = region.x
        crop_y = region.y

        # Step 2: Scale to fit within canvas_width × top_third_height
        # Preserve aspect ratio — fit by width first, then check height
        scale_w = canvas_width
        scale_h = round(crop_h * canvas_width / crop_w)

        if scale_h > top_third_height:
            scale_h = top_third_height
            scale_w = round(crop_w * top_third_height / crop_h)

        # Step 3: Centre horizontally
        overlay_x = (canvas_width - scale_w) // 2

        # Build the filter string:
        # [0:v] → crop → scale → [facecam_scaled]
        # [canvas][facecam_scaled] → overlay at (overlay_x, 0) → [with_facecam]
        filter_str = (
            f"[0:v]crop={crop_w}:{crop_h}:{crop_x}:{crop_y},"
            f"scale={scale_w}:{scale_h}[facecam_scaled];"
            f"[canvas][facecam_scaled]overlay={overlay_x}:0[with_facecam]"
        )

        return FilterFragment(
            filter_str=filter_str,
            input_label="[0:v]",
            output_label="[with_facecam]",
        )

    def build_blur_fallback_filter(
        self,
        canvas_width: int,
        canvas_height: int,
        top_third_height: int,
    ) -> FilterFragment:
        """Build an FFmpeg filter fragment that fills the top third with blurred gameplay.

        When no facecam is detected, the top third of the canvas is filled with a
        blurred and cropped version of the gameplay video, creating an aesthetically
        pleasing background fill.

        The gameplay video is:
        1. Cropped to a canvas_width × top_third_height centre crop.
        2. Scaled to exactly canvas_width × top_third_height.
        3. Blurred with boxblur=20:5.
        4. Overlaid at (0, 0) on the canvas.

        Args:
            canvas_width:     Width of the 9:16 canvas in pixels.
            canvas_height:    Height of the 9:16 canvas in pixels.
            top_third_height: Height of the top region to fill in pixels.

        Returns:
            FilterFragment with input_label "[0:v]" and output_label "[with_blur_fill]".
        """
        # Scale the gameplay video to cover canvas_width × top_third_height
        # (scale up to cover, preserving aspect ratio), then crop to exact dimensions,
        # then blur.  Using scale2ref / force_original_aspect_ratio=increase ensures
        # the scaled frame is always at least as large as the target, so the subsequent
        # crop never requests more pixels than are available.
        #
        # Strategy:
        #   1. scale=canvas_width:-1  → fit width, height proportional
        #   2. If scaled height < top_third_height, scale=-1:top_third_height instead
        #   We use FFmpeg's scale filter with force_original_aspect_ratio=increase
        #   to guarantee the output covers the target area, then crop to exact size.
        filter_str = (
            f"[0:v]scale={canvas_width}:{top_third_height}:force_original_aspect_ratio=increase,"
            f"crop={canvas_width}:{top_third_height},"
            f"boxblur=20:5[blur_fill];"
            f"[canvas][blur_fill]overlay=0:0[with_blur_fill]"
        )

        return FilterFragment(
            filter_str=filter_str,
            input_label="[0:v]",
            output_label="[with_blur_fill]",
        )
