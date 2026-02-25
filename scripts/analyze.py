#!/usr/bin/env python3
"""
Image analysis script for the text overlay system.

Analyzes an image to extract:
  - Dominant color palette (via k-means clustering)
  - Candidate text colors with WCAG contrast ratios
  - 4x4 region grid with edge density, luminance variance, and text friendliness
  - Best text placement positions scored by text friendliness and rule-of-thirds

Usage:
    python3 scripts/analyze.py /path/to/image.jpg

Output:
    JSON object printed to stdout.
    Debug/progress info printed to stderr.
"""

import colorsys
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image
from scipy.cluster.vq import kmeans2
from scipy.ndimage import sobel

# ---------------------------------------------------------------------------
# RAW image support
# ---------------------------------------------------------------------------

RAW_EXTENSIONS = {".arw", ".cr2", ".cr3", ".nef", ".orf", ".raf", ".dng", ".rw2", ".pef", ".srw"}


def convert_raw_to_tiff(raw_path: str) -> str:
    """Convert a RAW image to a temporary TIFF file using macOS sips.

    Returns the path to the temporary TIFF file.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".tiff", delete=False)
    tmp.close()
    try:
        subprocess.run(
            ["sips", "-s", "format", "tiff", raw_path, "--out", tmp.name],
            check=True, capture_output=True, text=True,
        )
        print(f"Converted RAW to TIFF: {tmp.name}", file=sys.stderr)
        return tmp.name
    except subprocess.CalledProcessError as exc:
        os.unlink(tmp.name)
        print(f"ERROR: Failed to convert RAW file: {exc.stderr}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def rgb_to_hex(r, g, b):
    """Convert an (R, G, B) tuple (0-255 ints) to a hex string like '#A1B2C3'."""
    return "#{:02X}{:02X}{:02X}".format(int(round(r)), int(round(g)), int(round(b)))


def hex_to_rgb(hex_str):
    """Convert '#RRGGBB' to (R, G, B) tuple with 0-255 ints."""
    hex_str = hex_str.lstrip("#")
    return tuple(int(hex_str[i:i + 2], 16) for i in (0, 2, 4))


def linearize_channel(c):
    """Linearize a single sRGB channel value (0-1 float) for luminance calc."""
    if c <= 0.03928:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def relative_luminance(r, g, b):
    """Compute WCAG relative luminance from 0-255 RGB values."""
    r_lin = linearize_channel(r / 255.0)
    g_lin = linearize_channel(g / 255.0)
    b_lin = linearize_channel(b / 255.0)
    return 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin


def contrast_ratio(color1_rgb, color2_rgb):
    """Compute WCAG contrast ratio between two (R, G, B) tuples (0-255)."""
    l1 = relative_luminance(*color1_rgb)
    l2 = relative_luminance(*color2_rgb)
    if l1 < l2:
        l1, l2 = l2, l1
    return (l1 + 0.05) / (l2 + 0.05)


def rgb_to_hsl(r, g, b):
    """Convert 0-255 RGB to (H 0-360, S 0-1, L 0-1)."""
    h, l, s = colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)
    return h * 360.0, s, l


def hsl_to_rgb(h, s, l):
    """Convert (H 0-360, S 0-1, L 0-1) to 0-255 RGB tuple."""
    r, g, b = colorsys.hls_to_rgb(h / 360.0, l, s)
    return (
        int(round(r * 255)),
        int(round(g * 255)),
        int(round(b * 255)),
    )


def rotate_hue(r, g, b, degrees):
    """Rotate the hue of an RGB color by the given degrees."""
    h, s, l = rgb_to_hsl(r, g, b)
    h = (h + degrees) % 360.0
    return hsl_to_rgb(h, s, l)


def adjust_lightness(r, g, b, delta):
    """Shift lightness of an RGB color by *delta* (clamped to 0-1)."""
    h, s, l = rgb_to_hsl(r, g, b)
    l = max(0.0, min(1.0, l + delta))
    return hsl_to_rgb(h, s, l)


# ---------------------------------------------------------------------------
# Core analysis functions
# ---------------------------------------------------------------------------

def extract_palette(image, k=5):
    """Extract *k* dominant colors via k-means on a downsampled image.

    Returns a list of hex strings sorted by cluster size (most dominant first).
    """
    # Downsample for speed
    thumb = image.copy()
    thumb.thumbnail((150, 150))
    arr = np.asarray(thumb.convert("RGB"), dtype=np.float64).reshape(-1, 3)

    centroids, labels = kmeans2(arr, k, minit="points", iter=20)

    # Sort by cluster population (largest first)
    counts = np.bincount(labels, minlength=k)
    order = np.argsort(-counts)
    centroids = centroids[order]

    palette = []
    for c in centroids:
        r, g, b = np.clip(c, 0, 255).astype(int)
        palette.append(rgb_to_hex(r, g, b))
    return palette


def generate_candidates(palette, center_avg_rgb):
    """Generate candidate text colors from palette + safe fallbacks.

    Each candidate has: hex, source, contrast_vs_center.
    Candidates with contrast < 2.0 are discarded.
    Result is sorted by contrast descending.
    """
    candidates = []

    for hex_color in palette:
        r, g, b = hex_to_rgb(hex_color)

        derived = [
            (rgb_to_hex(r, g, b), "palette"),
            (rgb_to_hex(*rotate_hue(r, g, b, 180)), "complementary"),
            (rgb_to_hex(*rotate_hue(r, g, b, 30)), "analogous_warm"),
            (rgb_to_hex(*rotate_hue(r, g, b, -30)), "analogous_cool"),
            (rgb_to_hex(*adjust_lightness(r, g, b, 0.40)), "tinted_light"),
            (rgb_to_hex(*adjust_lightness(r, g, b, -0.40)), "tinted_dark"),
        ]

        for hex_val, source in derived:
            cr = contrast_ratio(hex_to_rgb(hex_val), center_avg_rgb)
            candidates.append({
                "hex": hex_val,
                "source": source,
                "contrast_vs_center": round(cr, 2),
            })

    # Filter: contrast must be >= 2.0
    candidates = [c for c in candidates if c["contrast_vs_center"] >= 2.0]

    # Deduplicate by hex (keep the entry with highest contrast if dupes exist)
    seen = {}
    for c in candidates:
        key = c["hex"]
        if key not in seen or c["contrast_vs_center"] > seen[key]["contrast_vs_center"]:
            seen[key] = c
    candidates = list(seen.values())

    # Sort by contrast descending
    candidates.sort(key=lambda c: c["contrast_vs_center"], reverse=True)

    return candidates


def analyze_regions(image):
    """Divide the image into a 4x4 grid and compute per-cell metrics.

    Returns a list of 16 cell dicts with:
        row, col, edge_density, luminance_variance, avg_brightness, avg_color,
        text_friendliness
    """
    arr = np.asarray(image.convert("RGB"), dtype=np.float64)
    height, width, _ = arr.shape

    # Grayscale for edge / luminance analysis
    gray = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]

    # Full-image Sobel edge magnitude
    sobel_x = sobel(gray, axis=1)
    sobel_y = sobel(gray, axis=0)
    edge_mag = np.hypot(sobel_x, sobel_y)

    # Normalize edge magnitudes to 0-1 globally
    emax = edge_mag.max()
    if emax > 0:
        edge_mag_norm = edge_mag / emax
    else:
        edge_mag_norm = edge_mag

    row_edges = np.linspace(0, height, 5).astype(int)
    col_edges = np.linspace(0, width, 5).astype(int)

    cells = []
    for r in range(4):
        for c in range(4):
            y0, y1 = row_edges[r], row_edges[r + 1]
            x0, x1 = col_edges[c], col_edges[c + 1]

            cell_gray = gray[y0:y1, x0:x1]
            cell_rgb = arr[y0:y1, x0:x1]
            cell_edge = edge_mag_norm[y0:y1, x0:x1]

            ed = float(cell_edge.mean())
            lv = float(cell_gray.var() / (255.0 ** 2))  # normalize to 0-1 range
            ab = float(cell_gray.mean())
            avg_r = int(round(cell_rgb[:, :, 0].mean()))
            avg_g = int(round(cell_rgb[:, :, 1].mean()))
            avg_b = int(round(cell_rgb[:, :, 2].mean()))

            cells.append({
                "row": r,
                "col": c,
                "edge_density": round(ed, 4),
                "luminance_variance": round(lv, 4),
                "avg_brightness": round(ab, 1),
                "avg_color": rgb_to_hex(avg_r, avg_g, avg_b),
            })

    # Compute text_friendliness with cross-cell normalization
    eds = np.array([c["edge_density"] for c in cells])
    lvs = np.array([c["luminance_variance"] for c in cells])

    ed_min, ed_max = eds.min(), eds.max()
    lv_min, lv_max = lvs.min(), lvs.max()

    for i, cell in enumerate(cells):
        if ed_max > ed_min:
            norm_ed = (cell["edge_density"] - ed_min) / (ed_max - ed_min)
        else:
            norm_ed = 0.0

        if lv_max > lv_min:
            norm_lv = (cell["luminance_variance"] - lv_min) / (lv_max - lv_min)
        else:
            norm_lv = 0.0

        tf = 1.0 - (0.5 * norm_ed + 0.5 * norm_lv)
        cell["text_friendliness"] = round(tf, 4)

    return cells


def compute_best_placements(cells):
    """Score 6 named positions using text_friendliness + rule-of-thirds bonus.

    Returns a sorted list of placement dicts (best first).
    """
    # Build a lookup: (row, col) -> cell
    cell_map = {}
    for c in cells:
        cell_map[(c["row"], c["col"])] = c

    positions = {
        "top-left":      [(0, 0), (0, 1), (1, 0), (1, 1)],
        "top-center":    [(0, 1), (0, 2), (1, 1), (1, 2)],
        "center":        [(1, 1), (1, 2), (2, 1), (2, 2)],
        "bottom-left":   [(2, 0), (2, 1), (3, 0), (3, 1)],
        "bottom-center": [(2, 1), (2, 2), (3, 1), (3, 2)],
        "bottom-right":  [(2, 2), (2, 3), (3, 2), (3, 3)],
    }

    # Rule-of-thirds bonus positions
    thirds_bonus = {"top-left", "bottom-left", "bottom-right"}

    placements = []
    for name, coords in positions.items():
        covered = [cell_map[rc] for rc in coords]
        avg_tf = np.mean([c["text_friendliness"] for c in covered])

        score = avg_tf
        if name in thirds_bonus:
            score += 0.1

        # Average color of covered region
        avg_r = int(round(np.mean([hex_to_rgb(c["avg_color"])[0] for c in covered])))
        avg_g = int(round(np.mean([hex_to_rgb(c["avg_color"])[1] for c in covered])))
        avg_b = int(round(np.mean([hex_to_rgb(c["avg_color"])[2] for c in covered])))

        placements.append({
            "position": name,
            "score": round(float(score), 4),
            "region_avg_color": rgb_to_hex(avg_r, avg_g, avg_b),
        })

    placements.sort(key=lambda p: p["score"], reverse=True)
    return placements


def get_center_avg_rgb(cells):
    """Return the average RGB of the four center cells (1,1), (1,2), (2,1), (2,2)."""
    center_coords = [(1, 1), (1, 2), (2, 1), (2, 2)]
    cell_map = {(c["row"], c["col"]): c for c in cells}
    rs, gs, bs = [], [], []
    for rc in center_coords:
        r, g, b = hex_to_rgb(cell_map[rc]["avg_color"])
        rs.append(r)
        gs.append(g)
        bs.append(b)
    return (int(round(np.mean(rs))), int(round(np.mean(gs))), int(round(np.mean(bs))))


# ---------------------------------------------------------------------------
# Rectangle candidate scanning
# ---------------------------------------------------------------------------

def _build_cost_map(image, work_size=100):
    """Downsample image and compute a cost map (0=clean, 1=busy).

    Combines Sobel edge magnitude and local luminance variance.
    Returns (cost_map, work_w, work_h).
    """
    thumb = image.copy()
    w, h = thumb.size
    scale = work_size / max(w, h)
    work_w = max(1, int(w * scale))
    work_h = max(1, int(h * scale))
    thumb = thumb.resize((work_w, work_h), Image.LANCZOS)

    arr = np.asarray(thumb.convert("RGB"), dtype=np.float64)
    gray = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]

    # Edge magnitude
    sx = sobel(gray, axis=1)
    sy = sobel(gray, axis=0)
    edge_mag = np.hypot(sx, sy)
    emax = edge_mag.max()
    norm_edge = edge_mag / emax if emax > 0 else edge_mag

    # Local luminance variance (5x5 window via uniform filter)
    from scipy.ndimage import uniform_filter
    local_mean = uniform_filter(gray, size=5)
    local_sq_mean = uniform_filter(gray ** 2, size=5)
    local_var = np.maximum(local_sq_mean - local_mean ** 2, 0)
    vmax = local_var.max()
    norm_var = local_var / vmax if vmax > 0 else local_var

    cost = 0.5 * norm_edge + 0.5 * norm_var
    return cost, work_w, work_h


def _integral_image(arr):
    """Compute the integral (summed-area) image of a 2D array."""
    return arr.cumsum(axis=0).cumsum(axis=1)


def _rect_sum(integral, y0, x0, y1, x1):
    """O(1) sum over a rectangle using the integral image.

    Region is [y0:y1, x0:x1] (exclusive y1, x1).
    """
    # Shift indices for the inclusive integral image lookup
    s = integral[y1 - 1, x1 - 1]
    if y0 > 0:
        s -= integral[y0 - 1, x1 - 1]
    if x0 > 0:
        s -= integral[y1 - 1, x0 - 1]
    if y0 > 0 and x0 > 0:
        s += integral[y0 - 1, x0 - 1]
    return s


def compute_rectangle_candidates(image, layout_mode, width, height):
    """Scan the image for low-busyness rectangles suitable for text placement.

    Returns a list of candidate dicts sorted by score (best first), each with:
        center_x_pct, center_y_pct, width_pct, score, avg_color, region_luminance
    """
    cost, work_w, work_h = _build_cost_map(image)
    integral = _integral_image(cost)

    # Also build integral images for color extraction
    arr = np.asarray(
        image.copy().resize((work_w, work_h), Image.LANCZOS).convert("RGB"),
        dtype=np.float64,
    )
    gray = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
    int_r = _integral_image(arr[:, :, 0])
    int_g = _integral_image(arr[:, :, 1])
    int_b = _integral_image(arr[:, :, 2])
    int_gray = _integral_image(gray)

    # Width steps depend on layout mode
    if layout_mode == "center":
        width_steps = [0.30, 0.40, 0.50]
    else:  # auto / story
        width_steps = [0.15, 0.22, 0.30]

    safe_frac = 0.03  # 3% safe margin
    step_frac = 0.05  # 5% sliding step

    # Rule-of-thirds lines (for bonus scoring)
    thirds_y = [1.0 / 3.0, 2.0 / 3.0]
    thirds_x = [1.0 / 3.0, 2.0 / 3.0]

    all_candidates = []

    for wp in width_steps:
        rect_pw = int(work_w * wp)
        rect_ph = int(rect_pw * 0.6)  # height = 60% of width
        if rect_pw < 2 or rect_ph < 2:
            continue

        # Sliding window with 5% steps
        step_x = max(1, int(work_w * step_frac))
        step_y = max(1, int(work_h * step_frac))
        margin_x = int(work_w * safe_frac)
        margin_y = int(work_h * safe_frac)

        candidates_for_width = []

        for y0 in range(margin_y, work_h - rect_ph - margin_y + 1, step_y):
            for x0 in range(margin_x, work_w - rect_pw - margin_x + 1, step_x):
                y1 = y0 + rect_ph
                x1 = x0 + rect_pw
                area = rect_pw * rect_ph

                avg_cost = _rect_sum(integral, y0, x0, y1, x1) / area
                score = 1.0 - avg_cost  # higher = cleaner

                # Rule-of-thirds bonus: center of rectangle near thirds lines
                cx_frac = (x0 + rect_pw / 2.0) / work_w
                cy_frac = (y0 + rect_ph / 2.0) / work_h
                min_dist_x = min(abs(cx_frac - t) for t in thirds_x)
                min_dist_y = min(abs(cy_frac - t) for t in thirds_y)
                if min_dist_x < 0.08 or min_dist_y < 0.08:
                    score += 0.05

                # Average color and luminance of the region
                avg_r = _rect_sum(int_r, y0, x0, y1, x1) / area
                avg_g = _rect_sum(int_g, y0, x0, y1, x1) / area
                avg_b = _rect_sum(int_b, y0, x0, y1, x1) / area
                avg_lum = _rect_sum(int_gray, y0, x0, y1, x1) / area

                candidates_for_width.append({
                    "center_x_pct": round((x0 + rect_pw / 2.0) / work_w, 3),
                    "center_y_pct": round((y0 + rect_ph / 2.0) / work_h, 3),
                    "width_pct": wp,
                    "score": round(float(score), 4),
                    "avg_color": rgb_to_hex(
                        int(round(np.clip(avg_r, 0, 255))),
                        int(round(np.clip(avg_g, 0, 255))),
                        int(round(np.clip(avg_b, 0, 255))),
                    ),
                    "region_luminance": round(float(avg_lum), 1),
                })

        # Keep top 3 per width step
        candidates_for_width.sort(key=lambda c: c["score"], reverse=True)
        all_candidates.extend(candidates_for_width[:3])

    # Sort all candidates by score descending
    all_candidates.sort(key=lambda c: c["score"], reverse=True)
    return all_candidates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyze(image_path, layout_mode=None):
    """Run full analysis on the image at *image_path* and return the result dict.

    If *layout_mode* is provided ('auto' or 'center'), rectangle candidates
    are computed and included in the output.
    """
    # Handle RAW files by converting to TIFF first
    ext = os.path.splitext(image_path)[1].lower()
    tmp_tiff = None
    if ext in RAW_EXTENSIONS:
        print(f"RAW file detected ({ext}), converting via sips ...", file=sys.stderr)
        tmp_tiff = convert_raw_to_tiff(image_path)
        load_path = tmp_tiff
    else:
        load_path = image_path

    print(f"Loading image: {load_path}", file=sys.stderr)
    image = Image.open(load_path).convert("RGB")
    width, height = image.size
    print(f"Image dimensions: {width}x{height}", file=sys.stderr)

    # --- Palette ---
    print("Extracting palette ...", file=sys.stderr)
    palette = extract_palette(image, k=5)
    print(f"Palette: {palette}", file=sys.stderr)

    # --- Regions ---
    print("Analyzing regions (4x4 grid) ...", file=sys.stderr)
    cells = analyze_regions(image)

    # --- Center average color ---
    center_avg_rgb = get_center_avg_rgb(cells)
    print(f"Center average color: {rgb_to_hex(*center_avg_rgb)}", file=sys.stderr)

    # --- Candidate colors ---
    print("Generating candidate colors ...", file=sys.stderr)
    candidates = generate_candidates(palette, center_avg_rgb)
    print(f"Candidates after filtering: {len(candidates)}", file=sys.stderr)

    # --- Best placements ---
    print("Computing best placements ...", file=sys.stderr)
    placements = compute_best_placements(cells)

    # --- Rectangle candidates (if layout mode specified) ---
    rect_candidates = None
    if layout_mode in ("auto", "center"):
        print(f"Computing rectangle candidates (layout={layout_mode}) ...", file=sys.stderr)
        rect_candidates = compute_rectangle_candidates(image, layout_mode, width, height)
        print(f"Rectangle candidates: {len(rect_candidates)}", file=sys.stderr)

    result = {
        "image_dimensions": {"width": width, "height": height},
        "palette": palette,
        "color_candidates": candidates,
        "regions": {
            "grid": "4x4",
            "cells": cells,
        },
        "best_placements": placements,
    }
    if rect_candidates is not None:
        result["rectangle_candidates"] = rect_candidates

    # Clean up temporary TIFF if we converted from RAW
    if tmp_tiff and os.path.isfile(tmp_tiff):
        os.unlink(tmp_tiff)
        print(f"Cleaned up temp TIFF: {tmp_tiff}", file=sys.stderr)

    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/analyze.py /path/to/image.jpg [--layout auto|center]",
              file=sys.stderr)
        sys.exit(1)

    image_path = sys.argv[1]

    # Parse optional --layout flag
    layout_mode = None
    args = sys.argv[2:]
    for i, arg in enumerate(args):
        if arg == "--layout" and i + 1 < len(args):
            layout_mode = args[i + 1]
            break

    result = analyze(image_path, layout_mode=layout_mode)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
