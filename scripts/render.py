#!/usr/bin/env python3
"""
Pillow-based text compositor for rendering bilingual (English/Arabic) text
overlays on images.

Usage:
    python3 scripts/render.py /path/to/image.jpg /path/to/decisions.json

Output is written as JSON to stdout:
    {"output_image": "/path/to/output.jpg", "state_file": "/path/to/state.json"}

All progress/debug messages go to stderr.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ---------------------------------------------------------------------------
# RAW image support
# ---------------------------------------------------------------------------

RAW_EXTENSIONS = {".arw", ".cr2", ".cr3", ".nef", ".orf", ".raf", ".dng", ".rw2", ".pef", ".srw"}


def _convert_raw_to_tiff(raw_path: str) -> str:
    """Convert a RAW image to a temporary TIFF file using macOS sips."""
    tmp = tempfile.NamedTemporaryFile(suffix=".tiff", delete=False)
    tmp.close()
    try:
        subprocess.run(
            ["sips", "-s", "format", "tiff", raw_path, "--out", tmp.name],
            check=True, capture_output=True, text=True,
        )
        _log(f"Converted RAW to TIFF: {tmp.name}")
        return tmp.name
    except subprocess.CalledProcessError as exc:
        os.unlink(tmp.name)
        _fatal(f"Failed to convert RAW file: {exc.stderr}")

# ---------------------------------------------------------------------------
# Project root resolution (script lives in <root>/scripts/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Arabic text helpers
# ---------------------------------------------------------------------------

def _reshape_arabic(text: str) -> str:
    """Reshape and apply BiDi algorithm to Arabic text."""
    import arabic_reshaper
    from bidi.algorithm import get_display

    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)


# ---------------------------------------------------------------------------
# Font helpers
# ---------------------------------------------------------------------------

def _load_font(font_path: str, size: int, weight: str = "bold") -> ImageFont.FreeTypeFont:
    """Load a TrueType font, optionally handling variable-font weight axes.

    *weight* controls which weight to select for variable fonts:
      - "bold" / "black" → maximum weight (700-900)
      - "light"          → 300
      - "regular"        → 400
    For non-variable fonts this parameter is ignored.
    """
    font = ImageFont.truetype(font_path, size)

    # Attempt variable-font weight axis setup for *-Variable.ttf files
    if font_path.endswith("-Variable.ttf"):
        # Map weight names to numeric values
        weight_map = {
            "bold": 700, "black": 900,
            "light": 300, "regular": 400,
        }
        target_weight = weight_map.get(weight.lower(), 700)

        try:
            axes = font.get_variation_axes()
            for axis in axes:
                # Pillow returns axis names as bytes (b'Weight') — handle both
                axis_name = axis.get("name", b"")
                if isinstance(axis_name, bytes):
                    axis_name = axis_name.decode("utf-8", errors="ignore")
                if axis_name.lower() == "weight":
                    lo = axis.get("minimum", 100)
                    hi = axis.get("maximum", 900)
                    clamped = max(lo, min(hi, target_weight))
                    defaults = [float(a.get("default", a.get("minimum", 0))) for a in axes]
                    idx = axes.index(axis)
                    defaults[idx] = float(clamped)
                    font.set_variation_by_axes(defaults)
                    break
        except Exception:
            pass

    return font


def _resolve_font_path(relative_path: str) -> str:
    """Resolve a font path relative to the project root and verify it exists."""
    full_path = os.path.join(PROJECT_ROOT, relative_path)
    if not os.path.isfile(full_path):
        _fatal(f"Font file not found: {full_path}")
    return full_path


# ---------------------------------------------------------------------------
# Font-size binary search
# ---------------------------------------------------------------------------

def find_font_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    font_path: str,
    target_width: int,
    min_size: int = 10,
    max_size: int = 500,
    weight: str = "bold",
) -> int:
    """Return the largest font size whose rendered width <= *target_width*."""
    lo, hi = min_size, max_size
    while lo < hi:
        mid = (lo + hi + 1) // 2
        font = _load_font(font_path, mid, weight=weight)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        if text_width <= target_width:
            lo = mid
        else:
            hi = mid - 1
    return lo


# ---------------------------------------------------------------------------
# Readability enhancements
# ---------------------------------------------------------------------------

def _apply_drop_shadow(
    base_image: Image.Image,
    text: str,
    position: tuple[int, int],
    font: ImageFont.FreeTypeFont,
    fill: str,
    main_font_size: int,
    anchor: str | None = None,
) -> Image.Image:
    """Render a blurred drop-shadow behind the text, composite onto *base_image*."""
    shadow_offset = max(2, main_font_size // 25)
    shadow_color = (0, 0, 0, 128)  # semi-transparent black

    # Create a transparent layer for the shadow
    shadow_layer = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow_layer)

    sx = position[0] + shadow_offset
    sy = position[1] + shadow_offset
    shadow_draw.text((sx, sy), text, font=font, fill=shadow_color, anchor=anchor)

    # Slight blur
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=shadow_offset))

    # Composite shadow onto base
    base_image = Image.alpha_composite(base_image, shadow_layer)
    return base_image


def _apply_gradient_overlay(
    base_image: Image.Image,
    block_y: int,
    text_block_height: int,
    image_height: int,
) -> Image.Image:
    """Draw a semi-transparent dark gradient over the text region."""
    gradient_height = int(image_height * 0.4)
    # Center the gradient region around the text block
    grad_top = max(0, block_y - (gradient_height - text_block_height) // 2)
    grad_bottom = min(image_height, grad_top + gradient_height)
    actual_grad_height = grad_bottom - grad_top

    overlay = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
    for y in range(actual_grad_height):
        # Alpha ramps from 0 at top to 160 at bottom
        alpha = int(160 * (y / max(actual_grad_height - 1, 1)))
        for x in range(base_image.width):
            overlay.putpixel((x, grad_top + y), (0, 0, 0, alpha))

    # Using a more efficient approach: draw horizontal lines
    overlay = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for y in range(actual_grad_height):
        alpha = int(160 * (y / max(actual_grad_height - 1, 1)))
        draw.line([(0, grad_top + y), (base_image.width, grad_top + y)], fill=(0, 0, 0, alpha))

    base_image = Image.alpha_composite(base_image, overlay)
    return base_image


# ---------------------------------------------------------------------------
# Main rendering logic
# ---------------------------------------------------------------------------

def _render_text_element(
    base_image: Image.Image,
    draw: ImageDraw.ImageDraw,
    text: str,
    position: tuple[int, int],
    font: ImageFont.FreeTypeFont,
    fill: str,
    readability: dict,
    main_font_size: int,
    anchor: str | None = None,
) -> Image.Image:
    """Render a single text element with the appropriate readability enhancement."""
    rtype = readability.get("type", "none")

    if rtype == "drop-shadow":
        base_image = _apply_drop_shadow(
            base_image, text, position, font, fill, main_font_size, anchor=anchor,
        )
        # Re-acquire draw context after compositing
        draw = ImageDraw.Draw(base_image)
        draw.text(position, text, font=font, fill=fill, anchor=anchor)

    elif rtype == "stroke":
        stroke_width = max(1, main_font_size // 50)
        stroke_fill = "#000000"
        draw.text(
            position, text, font=font, fill=fill,
            stroke_width=stroke_width, stroke_fill=stroke_fill,
            anchor=anchor,
        )

    else:
        # "none" or unrecognised — plain rendering
        draw.text(position, text, font=font, fill=fill, anchor=anchor)

    return base_image


def _save_output(
    img: Image.Image,
    image_path: str,
    src_ext: str,
    output_dir: str | None,
    decisions: dict,
    tmp_tiff: str | None = None,
) -> tuple[str, str]:
    """Save the rendered overlay image and state JSON.

    Returns (output_image_path, state_file_path).
    """
    src = Path(image_path)
    stem = src.stem

    # Always export as JPEG at quality=95 (visually lossless, sane file sizes)
    # No compounding: each render loads the original source, not a previous overlay
    out_ext = ".jpg"

    out_dir = Path(output_dir) if output_dir else src.parent

    output_image_path = str(out_dir / f"{stem}_overlay{out_ext}")
    state_file_path = str(Path("/tmp") / f"{stem}_overlay_state.json")

    out_img = img.convert("RGB") if img.mode == "RGBA" else img
    out_img.save(output_image_path, quality=95)

    _log(f"Saved overlay image: {output_image_path}")

    # Clean up temporary TIFF if we converted from RAW
    if tmp_tiff and os.path.isfile(tmp_tiff):
        os.unlink(tmp_tiff)
        _log(f"Cleaned up temp TIFF: {tmp_tiff}")

    # Save state JSON (copy of decisions)
    with open(state_file_path, "w", encoding="utf-8") as f:
        json.dump(decisions, f, indent=2, ensure_ascii=False)
    _log(f"Saved state file: {state_file_path}")

    return output_image_path, state_file_path


# ---------------------------------------------------------------------------
# Rectangle-based rendering
# ---------------------------------------------------------------------------

def _render_rectangle(
    img: Image.Image,
    image_width: int,
    image_height: int,
    safe_margin: int,
    decisions: dict,
    image_path: str,
    src_ext: str,
    output_dir: str | None,
    tmp_tiff: str | None,
) -> tuple[str, str]:
    """Render text inside a bounding rectangle with equal-volume sizing.

    Both main texts are independently sized to fill the rectangle width,
    giving equal visual weight regardless of text length.
    """
    texts = decisions.get("texts", {})
    font_cfg = decisions.get("font", {})
    color = decisions.get("color", "#FFFFFF")
    rect_cfg = decisions.get("rectangle", {})
    row_order = decisions.get("row_order", ["subtext_en", "main_en", "main_ar", "subtext_ar"])
    sizing = decisions.get("sizing", {})
    readability = decisions.get("readability", {"type": "none"})

    main_text = texts.get("main", "")
    subtext = texts.get("subtext", "")
    main_ar = texts.get("main_ar", "")
    subtext_ar = texts.get("subtext_ar", "")

    center_x_pct = rect_cfg.get("center_x_pct", 0.5)
    center_y_pct = rect_cfg.get("center_y_pct", 0.5)
    width_pct = rect_cfg.get("width_pct", 0.40)
    alignment = rect_cfg.get("alignment", "center")

    subtext_ratio = sizing.get("subtext_ratio", 0.30)
    line_spacing = sizing.get("line_spacing", 1.2)

    MIN_FONT_SIZE = 16

    # ------------------------------------------------------------------
    # Process Arabic text
    # ------------------------------------------------------------------
    bidi_main_ar = _reshape_arabic(main_ar) if main_ar else ""
    bidi_sub_ar = _reshape_arabic(subtext_ar) if subtext_ar else ""

    # ------------------------------------------------------------------
    # Resolve font paths
    # ------------------------------------------------------------------
    main_en_font_path = _resolve_font_path(font_cfg["main_en"]) if font_cfg.get("main_en") else None
    sub_en_font_path = _resolve_font_path(font_cfg["sub_en"]) if font_cfg.get("sub_en") else None
    main_ar_font_path = _resolve_font_path(font_cfg["main_ar"]) if font_cfg.get("main_ar") else None
    sub_ar_font_path = _resolve_font_path(font_cfg["sub_ar"]) if font_cfg.get("sub_ar") else None

    # ------------------------------------------------------------------
    # Compute rectangle pixel dimensions
    # ------------------------------------------------------------------
    rect_w = int(image_width * width_pct)
    _log(f"Rectangle width: {rect_w}px ({width_pct*100:.0f}% of {image_width})")

    tmp_draw = ImageDraw.Draw(img)

    # ------------------------------------------------------------------
    # Size main texts: each independently fills rect_w
    # ------------------------------------------------------------------
    main_en_size = 0
    if main_text and main_en_font_path:
        main_en_size = max(MIN_FONT_SIZE, find_font_size(
            tmp_draw, main_text, main_en_font_path, rect_w, weight="bold",
        ))
        _log(f"Main EN font size: {main_en_size}")

    main_ar_size = 0
    if bidi_main_ar and main_ar_font_path:
        main_ar_size = max(MIN_FONT_SIZE, find_font_size(
            tmp_draw, bidi_main_ar, main_ar_font_path, rect_w, weight="bold",
        ))
        _log(f"Main AR font size: {main_ar_size}")

    # Reference main size for subtext ratio
    ref_main_size = max(main_en_size, main_ar_size) if (main_en_size or main_ar_size) else 80
    sub_en_size = max(MIN_FONT_SIZE, int(ref_main_size * subtext_ratio))
    sub_ar_size = max(MIN_FONT_SIZE, int(ref_main_size * subtext_ratio))

    _log(f"Font sizes — main_en: {main_en_size}, main_ar: {main_ar_size}, "
         f"sub_en: {sub_en_size}, sub_ar: {sub_ar_size}")

    # ------------------------------------------------------------------
    # Load fonts
    # ------------------------------------------------------------------
    font_main_en = _load_font(main_en_font_path, main_en_size, weight="bold") if main_en_size and main_en_font_path else None
    font_main_ar = _load_font(main_ar_font_path, main_ar_size, weight="bold") if main_ar_size and main_ar_font_path else None
    font_sub_en = _load_font(sub_en_font_path, sub_en_size, weight="light") if sub_en_font_path and subtext else None
    font_sub_ar = _load_font(sub_ar_font_path, sub_ar_size, weight="light") if sub_ar_font_path and bidi_sub_ar else None

    # ------------------------------------------------------------------
    # Build rows from row_order
    # ------------------------------------------------------------------
    def _text_height(text, font):
        if not text or not font:
            return 0
        bbox = tmp_draw.textbbox((0, 0), text, font=font)
        return bbox[3] - bbox[1]

    def _text_width(text, font):
        if not text or not font:
            return 0
        bbox = tmp_draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0]

    rows = []  # list of dicts: {type, text, font, height, width, is_arabic}
    for row_type in row_order:
        if row_type == "main_en" and font_main_en and main_text:
            rows.append({
                "type": "main_en", "text": main_text, "font": font_main_en,
                "height": _text_height(main_text, font_main_en),
                "width": _text_width(main_text, font_main_en),
                "is_arabic": False,
            })
        elif row_type == "main_ar" and font_main_ar and bidi_main_ar:
            rows.append({
                "type": "main_ar", "text": bidi_main_ar, "font": font_main_ar,
                "height": _text_height(bidi_main_ar, font_main_ar),
                "width": _text_width(bidi_main_ar, font_main_ar),
                "is_arabic": True,
            })
        elif row_type == "subtext_en" and font_sub_en and subtext:
            rows.append({
                "type": "subtext_en", "text": subtext, "font": font_sub_en,
                "height": _text_height(subtext, font_sub_en),
                "width": _text_width(subtext, font_sub_en),
                "is_arabic": False,
            })
        elif row_type == "subtext_ar" and font_sub_ar and bidi_sub_ar:
            rows.append({
                "type": "subtext_ar", "text": bidi_sub_ar, "font": font_sub_ar,
                "height": _text_height(bidi_sub_ar, font_sub_ar),
                "width": _text_width(bidi_sub_ar, font_sub_ar),
                "is_arabic": True,
            })
        elif row_type == "subtext_line":
            # Both subtexts on same line: EN left, AR right
            row_h = 0
            if font_sub_en and subtext:
                row_h = max(row_h, _text_height(subtext, font_sub_en))
            if font_sub_ar and bidi_sub_ar:
                row_h = max(row_h, _text_height(bidi_sub_ar, font_sub_ar))
            if row_h > 0:
                rows.append({
                    "type": "subtext_line",
                    "text": None,  # composite row
                    "font": None,
                    "height": row_h,
                    "width": rect_w,  # spans full rectangle
                    "is_arabic": False,
                    "sub_en": (subtext, font_sub_en) if (font_sub_en and subtext) else None,
                    "sub_ar": (bidi_sub_ar, font_sub_ar) if (font_sub_ar and bidi_sub_ar) else None,
                })

    if not rows:
        _log("WARNING: No renderable rows — nothing to draw")
        return _save_output(img, image_path, src_ext, output_dir, decisions, tmp_tiff)

    # ------------------------------------------------------------------
    # Compute total height with per-element gaps
    # ------------------------------------------------------------------
    def _get_line_gap(element_height: int) -> int:
        raw_gap = int(element_height * (line_spacing - 1.0))
        min_gap = max(int(element_height * 0.25), 8)
        return max(raw_gap, min_gap)

    row_gaps = []
    for i, row in enumerate(rows):
        if i < len(rows) - 1:
            row_gaps.append(_get_line_gap(row["height"]))
        else:
            row_gaps.append(0)  # no gap after last row
    total_height = sum(r["height"] for r in rows) + sum(row_gaps)

    # ------------------------------------------------------------------
    # Convert center position → top-left corner, clamp to bounds
    # ------------------------------------------------------------------
    rect_x = int(image_width * center_x_pct - rect_w / 2)
    rect_y = int(image_height * center_y_pct - total_height / 2)

    # Clamp
    rect_x = max(safe_margin, min(rect_x, image_width - rect_w - safe_margin))
    rect_y = max(safe_margin, min(rect_y, image_height - total_height - safe_margin))
    _log(f"Rectangle top-left: ({rect_x}, {rect_y}), size: {rect_w}x{total_height}")

    # ------------------------------------------------------------------
    # Apply gradient readability overlay if requested
    # ------------------------------------------------------------------
    if readability.get("type") == "gradient":
        _log("Applying gradient overlay")
        img = _apply_gradient_overlay(img, rect_y, total_height, image_height)

    # ------------------------------------------------------------------
    # Render each row
    # ------------------------------------------------------------------
    draw = ImageDraw.Draw(img)
    cursor_y = rect_y

    # Reference main size for readability effects (drop shadow offset, stroke width)
    ref_font_size = ref_main_size

    for row_idx, row in enumerate(rows):
        if row["type"] == "subtext_line":
            # Composite row: EN left, AR right within rectangle
            sub_en_data = row.get("sub_en")
            sub_ar_data = row.get("sub_ar")

            if alignment == "center":
                # Center the pair
                parts = []
                if sub_en_data:
                    parts.append(sub_en_data)
                if sub_ar_data:
                    parts.append(sub_ar_data)

                if len(parts) == 2:
                    w_en = _text_width(parts[0][0], parts[0][1])
                    w_ar = _text_width(parts[1][0], parts[1][1])
                    gap_between = int(rect_w * 0.05)
                    total_w = w_en + gap_between + w_ar
                    start_x = rect_x + (rect_w - total_w) // 2
                    img = _render_text_element(
                        img, ImageDraw.Draw(img), parts[0][0],
                        (start_x, cursor_y), parts[0][1], color,
                        readability, ref_font_size,
                    )
                    img = _render_text_element(
                        img, ImageDraw.Draw(img), parts[1][0],
                        (start_x + w_en + gap_between, cursor_y), parts[1][1], color,
                        readability, ref_font_size,
                    )
                elif len(parts) == 1:
                    w_p = _text_width(parts[0][0], parts[0][1])
                    px = rect_x + (rect_w - w_p) // 2
                    img = _render_text_element(
                        img, ImageDraw.Draw(img), parts[0][0],
                        (px, cursor_y), parts[0][1], color,
                        readability, ref_font_size,
                    )
            else:
                # Left/right: EN on left side, AR on right side
                if sub_en_data:
                    if alignment == "left":
                        sx = rect_x
                    else:  # right
                        w_en = _text_width(sub_en_data[0], sub_en_data[1])
                        sx = rect_x + rect_w - w_en
                    img = _render_text_element(
                        img, ImageDraw.Draw(img), sub_en_data[0],
                        (sx, cursor_y), sub_en_data[1], color,
                        readability, ref_font_size,
                    )
                if sub_ar_data:
                    w_ar = _text_width(sub_ar_data[0], sub_ar_data[1])
                    if alignment == "left":
                        sx = rect_x + rect_w - w_ar
                    else:  # right
                        sx = rect_x
                    img = _render_text_element(
                        img, ImageDraw.Draw(img), sub_ar_data[0],
                        (sx, cursor_y), sub_ar_data[1], color,
                        readability, ref_font_size,
                    )
        else:
            # Single text row
            text = row["text"]
            font = row["font"]
            w = row["width"]
            is_ar = row["is_arabic"]

            if alignment == "center":
                tx = rect_x + (rect_w - w) // 2
            elif alignment == "left":
                tx = rect_x if not is_ar else rect_x + rect_w - w
            else:  # right
                tx = rect_x + rect_w - w if not is_ar else rect_x

            # Clamp x to safe bounds
            tx = max(safe_margin, min(tx, image_width - w - safe_margin))

            img = _render_text_element(
                img, ImageDraw.Draw(img), text,
                (tx, cursor_y), font, color,
                readability, ref_font_size,
            )

        cursor_y += row["height"] + row_gaps[row_idx]

    # ------------------------------------------------------------------
    # Save output
    # ------------------------------------------------------------------
    return _save_output(img, image_path, src_ext, output_dir, decisions, tmp_tiff)


def render(image_path: str, decisions: dict, output_dir: str | None = None) -> tuple[str, str]:
    """
    Render text overlay onto the image according to *decisions*.

    If *output_dir* is provided, output files are saved there instead of
    next to the source image.

    Returns (output_image_path, state_file_path).
    """
    # ------------------------------------------------------------------
    # 1. Load image at original resolution (with RAW support)
    # ------------------------------------------------------------------
    src_ext = os.path.splitext(image_path)[1].lower()
    tmp_tiff = None
    if src_ext in RAW_EXTENSIONS:
        _log(f"RAW file detected ({src_ext}), converting via sips ...")
        tmp_tiff = _convert_raw_to_tiff(image_path)
        load_path = tmp_tiff
    else:
        load_path = image_path

    _log(f"Loading image: {load_path}")
    img = Image.open(load_path).convert("RGBA")
    image_width, image_height = img.size
    safe_margin = int(min(image_width, image_height) * 0.03)
    _log(f"Image size: {image_width}x{image_height} (safe_margin={safe_margin})")

    # ------------------------------------------------------------------
    # Rectangle path: if decisions contain "rectangle", use new renderer
    # ------------------------------------------------------------------
    if "rectangle" in decisions:
        _log("Rectangle layout detected — using rectangle renderer")
        return _render_rectangle(
            img, image_width, image_height, safe_margin, decisions,
            image_path, src_ext, output_dir, tmp_tiff,
        )

    # ------------------------------------------------------------------
    # 2. Parse decisions (LEGACY path)
    # ------------------------------------------------------------------
    texts = decisions.get("texts", {})
    font_cfg = decisions.get("font", {})
    color = decisions.get("color", "#FFFFFF")
    placement = decisions.get("placement", {})
    sizing = decisions.get("sizing", {})
    readability = decisions.get("readability", {"type": "none"})

    main_text = texts.get("main", "")
    subtext = texts.get("subtext", "")
    main_ar = texts.get("main_ar", "")
    subtext_ar = texts.get("subtext_ar", "")

    anchor_mode = placement.get("anchor", "bottom-left")
    x_pct = placement.get("x_pct", 0.08)
    y_pct = placement.get("y_pct", 0.55)

    main_width_ratio = sizing.get("main_width_ratio", 0.70)
    subtext_ratio = sizing.get("subtext_ratio", 0.30)
    line_spacing = sizing.get("line_spacing", 1.2)
    arabic_scale_factor = sizing.get("arabic_scale_factor", 1.15)

    # ------------------------------------------------------------------
    # 3. Process Arabic text
    # ------------------------------------------------------------------
    bidi_main_ar = ""
    bidi_sub_ar = ""
    if main_ar:
        bidi_main_ar = _reshape_arabic(main_ar)
        _log(f"Arabic main reshaped: {main_ar!r} -> rendered")
    if subtext_ar:
        bidi_sub_ar = _reshape_arabic(subtext_ar)
        _log(f"Arabic subtext reshaped: {subtext_ar!r} -> rendered")

    # ------------------------------------------------------------------
    # 4. Resolve font paths
    # ------------------------------------------------------------------
    main_en_font_path = _resolve_font_path(font_cfg["main_en"]) if font_cfg.get("main_en") else None
    sub_en_font_path = _resolve_font_path(font_cfg["sub_en"]) if font_cfg.get("sub_en") else None
    main_ar_font_path = _resolve_font_path(font_cfg["main_ar"]) if font_cfg.get("main_ar") else None
    sub_ar_font_path = _resolve_font_path(font_cfg["sub_ar"]) if font_cfg.get("sub_ar") else None

    # ------------------------------------------------------------------
    # 5. Calculate font sizes via binary search
    # ------------------------------------------------------------------
    target_width = int(image_width * main_width_ratio)
    _log(f"Target main text width: {target_width}px ({main_width_ratio*100:.0f}% of {image_width})")

    # We need a temporary draw context for measurement
    tmp_draw = ImageDraw.Draw(img)

    # Determine the reference text for the binary search — use main_text
    if main_text and main_en_font_path:
        main_font_size = find_font_size(tmp_draw, main_text, main_en_font_path, target_width)
    elif bidi_main_ar and main_ar_font_path:
        # Fall back to Arabic main if no English main
        raw_size = find_font_size(tmp_draw, bidi_main_ar, main_ar_font_path, target_width)
        main_font_size = int(raw_size / arabic_scale_factor)
    else:
        main_font_size = 80  # sensible fallback

    sub_font_size = int(main_font_size * subtext_ratio)
    arabic_main_size = int(main_font_size * arabic_scale_factor)
    arabic_sub_size = int(sub_font_size * arabic_scale_factor)

    # If Arabic main text overflows available width, shrink it to fit
    available_width = image_width - 2 * safe_margin
    if bidi_main_ar and main_ar_font_path:
        test_font = _load_font(main_ar_font_path, arabic_main_size, weight="bold")
        bbox_test = tmp_draw.textbbox((0, 0), bidi_main_ar, font=test_font)
        ar_test_w = bbox_test[2] - bbox_test[0]
        if ar_test_w > available_width:
            arabic_main_size = find_font_size(
                tmp_draw, bidi_main_ar, main_ar_font_path,
                available_width, weight="bold",
            )
            _log(f"Arabic main text too wide ({ar_test_w}px), shrunk to size {arabic_main_size}")

    # Same check for Arabic subtext
    if bidi_sub_ar and sub_ar_font_path:
        test_font = _load_font(sub_ar_font_path, arabic_sub_size, weight="light")
        bbox_test = tmp_draw.textbbox((0, 0), bidi_sub_ar, font=test_font)
        ar_test_w = bbox_test[2] - bbox_test[0]
        if ar_test_w > available_width:
            arabic_sub_size = find_font_size(
                tmp_draw, bidi_sub_ar, sub_ar_font_path,
                available_width, weight="light",
            )
            _log(f"Arabic sub text too wide, shrunk to size {arabic_sub_size}")

    _log(f"Font sizes — main: {main_font_size}, sub: {sub_font_size}, "
         f"ar_main: {arabic_main_size}, ar_sub: {arabic_sub_size}")

    # Pre-load fonts (with appropriate weight for variable fonts)
    font_main_en = _load_font(main_en_font_path, main_font_size, weight="bold") if main_en_font_path and main_text else None
    font_sub_en = _load_font(sub_en_font_path, sub_font_size, weight="light") if sub_en_font_path and subtext else None
    font_main_ar = _load_font(main_ar_font_path, arabic_main_size, weight="bold") if main_ar_font_path and bidi_main_ar else None
    font_sub_ar = _load_font(sub_ar_font_path, arabic_sub_size, weight="light") if sub_ar_font_path and bidi_sub_ar else None

    # ------------------------------------------------------------------
    # 6. Compute the main text rendered width (used for right-alignment)
    # ------------------------------------------------------------------
    if font_main_en and main_text:
        bbox = tmp_draw.textbbox((0, 0), main_text, font=font_main_en)
        main_text_rendered_width = bbox[2] - bbox[0]
    elif font_main_ar and bidi_main_ar:
        bbox = tmp_draw.textbbox((0, 0), bidi_main_ar, font=font_main_ar)
        main_text_rendered_width = bbox[2] - bbox[0]
    else:
        main_text_rendered_width = target_width

    # ------------------------------------------------------------------
    # 7. Determine block position
    # ------------------------------------------------------------------
    if anchor_mode == "center":
        block_x = image_width // 2  # will centre each line individually
        block_y = int(image_height * y_pct)
    else:
        block_x = int(image_width * x_pct)
        block_y = int(image_height * y_pct)

    _log(f"Block position: ({block_x}, {block_y})  anchor={anchor_mode}")

    # ------------------------------------------------------------------
    # 8. Pre-calculate full text block height (needed for gradient)
    # ------------------------------------------------------------------
    def _text_height(draw, text, font):
        if not text or not font:
            return 0
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[3] - bbox[1]

    elements_heights = []
    # Subtext line (English sub + Arabic sub on same line)
    sub_line_h = 0
    if font_sub_en and subtext:
        sub_line_h = max(sub_line_h, _text_height(tmp_draw, subtext, font_sub_en))
    if font_sub_ar and bidi_sub_ar:
        sub_line_h = max(sub_line_h, _text_height(tmp_draw, bidi_sub_ar, font_sub_ar))
    if sub_line_h > 0:
        elements_heights.append(sub_line_h)

    # Main English text
    main_en_h = _text_height(tmp_draw, main_text, font_main_en) if font_main_en and main_text else 0
    if main_en_h > 0:
        elements_heights.append(main_en_h)

    # Arabic main text
    main_ar_h = _text_height(tmp_draw, bidi_main_ar, font_main_ar) if font_main_ar and bidi_main_ar else 0
    if main_ar_h > 0:
        elements_heights.append(main_ar_h)

    # Total height with spacing gaps between elements
    gap_count = max(0, len(elements_heights) - 1)
    avg_h = sum(elements_heights) / len(elements_heights) if elements_heights else 0
    gap_pixels = int(avg_h * (line_spacing - 1.0)) if avg_h else 0
    text_block_height = sum(elements_heights) + gap_count * gap_pixels

    # Clamp block position to keep text within image bounds
    max_block_y = image_height - text_block_height - safe_margin
    block_y = max(safe_margin, min(block_y, max_block_y))
    if anchor_mode != "center":
        max_block_x = image_width - safe_margin
        block_x = max(safe_margin, min(block_x, max_block_x))
    _log(f"Clamped block position: ({block_x}, {block_y})")

    # ------------------------------------------------------------------
    # 9. Apply gradient readability overlay (before text rendering)
    # ------------------------------------------------------------------
    if readability.get("type") == "gradient":
        _log("Applying gradient overlay")
        img = _apply_gradient_overlay(img, block_y, text_block_height, image_height)

    # ------------------------------------------------------------------
    # 10. Render text elements
    # ------------------------------------------------------------------
    draw = ImageDraw.Draw(img)
    cursor_y = block_y

    def _get_line_gap(element_height: int) -> int:
        raw_gap = int(element_height * (line_spacing - 1.0))
        # Minimum gap scales with the element's own height (25%).
        # This prevents text crowding between different scripts
        # while keeping proportional spacing for subtexts.
        min_gap = max(int(element_height * 0.25), 8)
        return max(raw_gap, min_gap)

    # --- Row 1: Subtexts (English left, Arabic right on same line) ----
    has_subtext_line = (font_sub_en and subtext) or (font_sub_ar and bidi_sub_ar)
    if has_subtext_line:
        if anchor_mode == "center":
            # In center mode, render subtexts centered together
            # Build a combined string: "subtext_en   subtext_ar"
            parts = []
            if font_sub_en and subtext:
                parts.append((subtext, font_sub_en))
            if font_sub_ar and bidi_sub_ar:
                parts.append((bidi_sub_ar, font_sub_ar))

            if len(parts) == 2:
                # Measure both
                bbox_en = tmp_draw.textbbox((0, 0), parts[0][0], font=parts[0][1])
                w_en = bbox_en[2] - bbox_en[0]
                bbox_ar = tmp_draw.textbbox((0, 0), parts[1][0], font=parts[1][1])
                w_ar = bbox_ar[2] - bbox_ar[0]
                spacing_between = int(main_text_rendered_width * 0.05)  # small gap
                total_w = w_en + spacing_between + w_ar
                start_x = (image_width - total_w) // 2

                img = _render_text_element(
                    img, ImageDraw.Draw(img), parts[0][0],
                    (start_x, cursor_y), parts[0][1], color,
                    readability, main_font_size,
                )
                img = _render_text_element(
                    img, ImageDraw.Draw(img), parts[1][0],
                    (start_x + w_en + spacing_between, cursor_y), parts[1][1], color,
                    readability, main_font_size,
                )
            elif len(parts) == 1:
                bbox_p = tmp_draw.textbbox((0, 0), parts[0][0], font=parts[0][1])
                w_p = bbox_p[2] - bbox_p[0]
                cx = (image_width - w_p) // 2
                img = _render_text_element(
                    img, ImageDraw.Draw(img), parts[0][0],
                    (cx, cursor_y), parts[0][1], color,
                    readability, main_font_size,
                )
        else:
            # Left-aligned layout
            if font_sub_en and subtext:
                img = _render_text_element(
                    img, ImageDraw.Draw(img), subtext,
                    (block_x, cursor_y), font_sub_en, color,
                    readability, main_font_size,
                )
            if font_sub_ar and bidi_sub_ar:
                # Right-align: right edge = block_x + main_text_rendered_width
                bbox_ar = tmp_draw.textbbox((0, 0), bidi_sub_ar, font=font_sub_ar)
                ar_w = bbox_ar[2] - bbox_ar[0]
                ar_x = block_x + main_text_rendered_width - ar_w
                # Clamp to keep within image bounds
                ar_x = max(safe_margin, min(ar_x, image_width - ar_w - safe_margin))
                img = _render_text_element(
                    img, ImageDraw.Draw(img), bidi_sub_ar,
                    (ar_x, cursor_y), font_sub_ar, color,
                    readability, main_font_size,
                )

        cursor_y += sub_line_h + _get_line_gap(sub_line_h)

    # --- Row 2: English main text ---------------------------------
    if font_main_en and main_text:
        if anchor_mode == "center":
            bbox_m = tmp_draw.textbbox((0, 0), main_text, font=font_main_en)
            w_m = bbox_m[2] - bbox_m[0]
            mx = (image_width - w_m) // 2
            img = _render_text_element(
                img, ImageDraw.Draw(img), main_text,
                (mx, cursor_y), font_main_en, color,
                readability, main_font_size,
            )
        else:
            img = _render_text_element(
                img, ImageDraw.Draw(img), main_text,
                (block_x, cursor_y), font_main_en, color,
                readability, main_font_size,
            )
        cursor_y += main_en_h + _get_line_gap(main_en_h)

    # --- Row 3: Arabic main text ----------------------------------
    if font_main_ar and bidi_main_ar:
        if anchor_mode == "center":
            bbox_a = tmp_draw.textbbox((0, 0), bidi_main_ar, font=font_main_ar)
            w_a = bbox_a[2] - bbox_a[0]
            ax = (image_width - w_a) // 2
            img = _render_text_element(
                img, ImageDraw.Draw(img), bidi_main_ar,
                (ax, cursor_y), font_main_ar, color,
                readability, main_font_size,
            )
        else:
            # Right-align: right edge = block_x + main_text_rendered_width
            bbox_a = tmp_draw.textbbox((0, 0), bidi_main_ar, font=font_main_ar)
            ar_w = bbox_a[2] - bbox_a[0]
            ar_x = block_x + main_text_rendered_width - ar_w
            # Clamp to keep within image bounds
            ar_x = max(safe_margin, min(ar_x, image_width - ar_w - safe_margin))
            img = _render_text_element(
                img, ImageDraw.Draw(img), bidi_main_ar,
                (ar_x, cursor_y), font_main_ar, color,
                readability, main_font_size,
            )

    # ------------------------------------------------------------------
    # 11. Save output
    # ------------------------------------------------------------------
    return _save_output(img, image_path, src_ext, output_dir, decisions, tmp_tiff)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    """Print a debug/progress message to stderr."""
    print(f"[render] {msg}", file=sys.stderr)


def _fatal(msg: str) -> None:
    """Print an error to stderr and exit with code 1."""
    print(f"[render] ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        _fatal(f"Usage: {sys.argv[0]} <image_path> <decisions_json_path> [output_dir]")

    image_path = sys.argv[1]
    decisions_path = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) == 4 else None

    # Validate image
    if not os.path.isfile(image_path):
        _fatal(f"Image file not found: {image_path}")

    # Validate and parse decisions JSON
    if not os.path.isfile(decisions_path):
        _fatal(f"Decisions JSON file not found: {decisions_path}")

    try:
        with open(decisions_path, "r", encoding="utf-8") as f:
            decisions = json.load(f)
    except json.JSONDecodeError as exc:
        _fatal(f"Malformed decisions JSON: {exc}")

    # Validate that we can open the image (handle RAW files)
    src_ext = os.path.splitext(image_path)[1].lower()
    if src_ext not in RAW_EXTENSIONS:
        try:
            Image.open(image_path).verify()
        except Exception as exc:
            _fatal(f"Cannot load image {image_path}: {exc}")
    else:
        _log(f"Skipping Pillow verify for RAW file ({src_ext})")

    # Validate output dir
    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        _log(f"Created output directory: {output_dir}")

    # Render
    output_image, state_file = render(image_path, decisions, output_dir=output_dir)

    # Output result JSON to stdout
    result = {"output_image": output_image, "state_file": state_file}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
