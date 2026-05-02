"""FrameReformatter: builds FFmpeg filter fragments for 16:9 → 9:16 canvas conversion."""

from __future__ import annotations

from pipeline.models import CanvasLayout, FilterFragment


def compute_canvas_layout(config) -> CanvasLayout:
    """Compute the 9:16 canvas layout from a ShortsConfig (or Config with shorts fields).

    Postconditions:
    - facecam_height + gameplay_height == canvas_height
    - facecam_width == gameplay_width == canvas_width
    - gameplay_y == facecam_height
    - facecam_x == 0, facecam_y == 0
    """
    canvas_width: int = config.shorts_width
    canvas_height: int = config.shorts_height

    facecam_height: int = round(canvas_height * config.facecam_top_fraction)
    # Exact complement ensures facecam_height + gameplay_height == canvas_height
    gameplay_height: int = canvas_height - facecam_height

    facecam_width: int = canvas_width
    gameplay_width: int = canvas_width

    facecam_x: int = 0
    facecam_y: int = 0
    gameplay_x: int = 0
    gameplay_y: int = facecam_height

    return CanvasLayout(
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        facecam_x=facecam_x,
        facecam_y=facecam_y,
        facecam_width=facecam_width,
        facecam_height=facecam_height,
        gameplay_x=gameplay_x,
        gameplay_y=gameplay_y,
        gameplay_width=gameplay_width,
        gameplay_height=gameplay_height,
    )


class FrameReformatter:
    """Builds the FFmpeg filter fragment that creates a 9:16 canvas from a 16:9 source.

    Responsibilities:
    - Scale the gameplay video to fit within the bottom region of the canvas
      while preserving the source aspect ratio.
    - Pad the full canvas to canvas_width × canvas_height with black fill.
    - Leave the top facecam region empty for the FacecamRelocator overlay.
    """

    def compute_canvas_layout(self, config) -> CanvasLayout:
        """Delegate to the module-level compute_canvas_layout function."""
        return compute_canvas_layout(config)

    def build_canvas_filter(
        self,
        src_width: int,
        src_height: int,
        layout: CanvasLayout,
    ) -> FilterFragment:
        """Build an FFmpeg filter fragment that scales and pads the source video.

        The source video is scaled to fit within the gameplay region
        (layout.gameplay_width × layout.gameplay_height) while preserving
        the source aspect ratio, then padded to the full canvas size with black.

        Args:
            src_width:  Source video width in pixels (must be > 0).
            src_height: Source video height in pixels (must be > 0).
            layout:     CanvasLayout computed from compute_canvas_layout().

        Returns:
            FilterFragment with input_label "[0:v]" and output_label "[canvas]".
        """
        gameplay_width = layout.gameplay_width
        gameplay_height = layout.gameplay_height
        canvas_w = layout.canvas_width
        canvas_h = layout.canvas_height
        gameplay_y = layout.gameplay_y

        # Try fitting to gameplay_width first (scale height proportionally)
        scale_w: int = gameplay_width
        scale_h: int = round(src_height * gameplay_width / src_width)

        # If the scaled height exceeds the gameplay region, fit to gameplay_height instead
        if scale_h > gameplay_height:
            scale_h = gameplay_height
            scale_w = round(src_width * gameplay_height / src_height)

        # Centre horizontally within the gameplay region
        pad_x: int = (gameplay_width - scale_w) // 2

        # Centre vertically within the gameplay region, offset by gameplay_y
        pad_y: int = gameplay_y + (gameplay_height - scale_h) // 2

        filter_str = (
            f"[0:v]scale={scale_w}:{scale_h},"
            f"pad={canvas_w}:{canvas_h}:{pad_x}:{pad_y}:black[canvas]"
        )

        return FilterFragment(
            filter_str=filter_str,
            input_label="[0:v]",
            output_label="[canvas]",
        )
