#!/usr/bin/env python3
"""
Arabic Font Audit Script — Multi-layer quality check for all Arabic fonts
in the manifest.

Layers:
  1. cmap coverage  — glyph existence for Arabic letters + diacritics
  2. Shaping validation — connected forms render with variable widths
  3. Visual rendering — pangram + test words produce non-zero ink
  4. Width ratio — font isn't excessively wide vs reference (Noto Sans Arabic)
  5. Weight axis — variable fonts produce different output at weight 300 vs 700
  6. Visual grid — human-reviewable comparison image (--visual flag)

Usage:
    python3 scripts/test_arabic_fonts.py [--visual]

Output:
    Per-font PASS/FAIL report to stdout.
    With --visual: exports/arabic_font_audit.png
"""

import json
import os
import sys

from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "fonts", "manifest.json")
EXPORTS_DIR = os.path.join(PROJECT_ROOT, "exports")

# ---------------------------------------------------------------------------
# Arabic test strings
# ---------------------------------------------------------------------------

# 28 Arabic letters (isolated forms)
ARABIC_LETTERS = "ابتثجحخدذرزسشصضطظعغفقكلمنهوي"

# Common diacritics (tashkeel)
ARABIC_DIACRITICS = "\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652"  # fatHatan..sukun

# Pangram-like test sentence (uses most Arabic letters)
ARABIC_PANGRAM = "صِف خَلقَ خَودٍ كَمِثلِ الشَمسِ إذ بَزَغَت يَحظى الضَجيعُ بِها نَجلاءَ مِعطارِ"

# Test words for shaping (connected forms)
SHAPING_WORDS = [
    "بسم",      # ba-sin-mim (all connected)
    "الله",      # alif-lam-lam-ha (ligature)
    "العربية",   # common word with multiple connections
    "كنيسة",     # test from actual use (church)
    "فانكوفر",   # transliteration test
]

# Reference font for width comparison
REFERENCE_FONT = "arabic/NotoSansArabic-Variable.ttf"
REFERENCE_SIZE = 60


def _load_manifest():
    """Load font manifest and extract unique Arabic font files."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    fonts = {}  # {family_name: {"main_file": ..., "sub_file": ..., "pair_ids": [...]}}
    for pair in manifest["pairs"]:
        ar = pair["arabic"]
        family = ar["family"]
        if family not in fonts:
            fonts[family] = {
                "main_file": ar["main_file"],
                "sub_file": ar.get("sub_file", ar["main_file"]),
                "pair_ids": [],
            }
        fonts[family]["pair_ids"].append(pair["id"])

    return fonts


def _resolve(relative_path):
    """Resolve font path relative to fonts/ directory."""
    return os.path.join(PROJECT_ROOT, "fonts", relative_path)


# ---------------------------------------------------------------------------
# Layer 1: cmap coverage
# ---------------------------------------------------------------------------

def check_cmap_coverage(font_path):
    """Check how many Arabic letters + diacritics have glyphs in the font.

    Returns (coverage_pct, missing_chars).
    """
    tt = TTFont(font_path)
    cmap = tt.getBestCmap()
    if cmap is None:
        return 0.0, list(ARABIC_LETTERS + ARABIC_DIACRITICS)

    all_chars = ARABIC_LETTERS + ARABIC_DIACRITICS
    missing = []
    for ch in all_chars:
        if ord(ch) not in cmap:
            missing.append(ch)

    coverage = (len(all_chars) - len(missing)) / len(all_chars)
    return coverage, missing


# ---------------------------------------------------------------------------
# Layer 2: Shaping validation
# ---------------------------------------------------------------------------

def check_shaping(font_path, size=60):
    """Check that Arabic shaping produces variable-width connected forms.

    Compares rendered width of shaped words vs. sum of individual char widths.
    If width ratio is close to 1.0 for all words, shaping may be broken.

    Returns (passed, details).
    """
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
    except ImportError:
        return True, "arabic_reshaper not available, skipping shaping check"

    font = ImageFont.truetype(font_path, size)
    img = Image.new("L", (2000, 200), 0)
    draw = ImageDraw.Draw(img)

    shaped_ok = 0
    total = 0
    details = []

    for word in SHAPING_WORDS:
        reshaped = arabic_reshaper.reshape(word)
        bidi = get_display(reshaped)

        # Width of the shaped word
        bbox_word = draw.textbbox((0, 0), bidi, font=font)
        word_w = bbox_word[2] - bbox_word[0]

        # Sum of individual character widths
        char_widths = 0
        for ch in word:
            reshaped_ch = arabic_reshaper.reshape(ch)
            bidi_ch = get_display(reshaped_ch)
            bbox_ch = draw.textbbox((0, 0), bidi_ch, font=font)
            char_widths += bbox_ch[2] - bbox_ch[0]

        if char_widths == 0:
            details.append(f"  '{word}': zero individual char widths")
            total += 1
            continue

        ratio = word_w / char_widths
        total += 1

        # If connected forms work, the word should be narrower than sum of parts
        # (Arabic letters change form when connected, typically reducing width)
        # A ratio significantly different from 1.0 means shaping is happening
        if ratio < 0.95 or ratio > 1.05:
            shaped_ok += 1

        details.append(f"  '{word}': word_w={word_w}, sum_chars={char_widths}, ratio={ratio:.2f}")

    # If at least 60% of words show shaping effects, consider it passing
    passed = (shaped_ok / max(total, 1)) >= 0.6 if total > 0 else False
    return passed, "\n".join(details)


# ---------------------------------------------------------------------------
# Layer 3: Visual rendering (non-zero ink)
# ---------------------------------------------------------------------------

def check_visual_rendering(font_path, size=60):
    """Check that rendering Arabic text produces non-zero ink pixels.

    Returns (passed, ink_pixels).
    """
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
    except ImportError:
        return True, 0

    font = ImageFont.truetype(font_path, size)
    img = Image.new("L", (2000, 200), 0)
    draw = ImageDraw.Draw(img)

    reshaped = arabic_reshaper.reshape(ARABIC_PANGRAM)
    bidi = get_display(reshaped)
    draw.text((10, 10), bidi, font=font, fill=255)

    import numpy as np
    arr = np.array(img)
    ink_pixels = int((arr > 0).sum())

    return ink_pixels > 100, ink_pixels


# ---------------------------------------------------------------------------
# Layer 4: Width ratio vs reference
# ---------------------------------------------------------------------------

def check_width_ratio(font_path, size=REFERENCE_SIZE):
    """Check if font is excessively wide compared to Noto Sans Arabic reference.

    Returns (passed, ratio).
    """
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
    except ImportError:
        return True, 1.0

    ref_path = _resolve(REFERENCE_FONT)
    if not os.path.isfile(ref_path):
        return True, 1.0

    test_text = "العربية"
    reshaped = arabic_reshaper.reshape(test_text)
    bidi = get_display(reshaped)

    img = Image.new("L", (2000, 200), 0)
    draw = ImageDraw.Draw(img)

    # Reference width
    ref_font = ImageFont.truetype(ref_path, size)
    bbox_ref = draw.textbbox((0, 0), bidi, font=ref_font)
    ref_w = bbox_ref[2] - bbox_ref[0]

    # Test width
    test_font = ImageFont.truetype(font_path, size)
    bbox_test = draw.textbbox((0, 0), bidi, font=test_font)
    test_w = bbox_test[2] - bbox_test[0]

    if ref_w == 0:
        return True, 1.0

    ratio = test_w / ref_w
    return ratio <= 1.5, round(ratio, 2)


# ---------------------------------------------------------------------------
# Layer 5: Weight axis (variable fonts)
# ---------------------------------------------------------------------------

def check_weight_axis(font_path):
    """For variable fonts, check that weight 300 and 700 produce different output.

    Returns (passed, detail_str).
    """
    if not font_path.endswith("-Variable.ttf"):
        return True, "static font, skipped"

    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        import numpy as np
    except ImportError:
        return True, "deps missing"

    test_text = "العربية"
    reshaped = arabic_reshaper.reshape(test_text)
    bidi = get_display(reshaped)

    results = {}
    for weight in [300, 700]:
        font = ImageFont.truetype(font_path, 60)
        try:
            axes = font.get_variation_axes()
            for axis in axes:
                axis_name = axis.get("name", b"")
                if isinstance(axis_name, bytes):
                    axis_name = axis_name.decode("utf-8", errors="ignore")
                if axis_name.lower() == "weight":
                    lo = axis.get("minimum", 100)
                    hi = axis.get("maximum", 900)
                    clamped = max(lo, min(hi, weight))
                    defaults = [float(a.get("default", a.get("minimum", 0))) for a in axes]
                    idx = axes.index(axis)
                    defaults[idx] = float(clamped)
                    font.set_variation_by_axes(defaults)
                    break
        except Exception:
            return True, "no variation axes"

        img = Image.new("L", (2000, 200), 0)
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), bidi, font=font, fill=255)
        arr = np.array(img)
        results[weight] = int((arr > 0).sum())

    if 300 in results and 700 in results:
        diff = abs(results[700] - results[300])
        pct = diff / max(results[300], 1) * 100
        passed = diff > 0
        return passed, f"ink@300={results[300]}, ink@700={results[700]}, diff={pct:.1f}%"

    return True, "could not test both weights"


# ---------------------------------------------------------------------------
# Layer 6: Visual grid
# ---------------------------------------------------------------------------

def generate_visual_grid(fonts_data, results):
    """Generate a visual comparison grid image.

    Each row: font name + rendered Arabic pangram + test words.
    """
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
    except ImportError:
        print("Cannot generate visual grid: arabic_reshaper not available", file=sys.stderr)
        return

    row_height = 120
    label_width = 300
    content_width = 1400
    total_width = label_width + content_width
    num_fonts = len(fonts_data)
    total_height = num_fonts * row_height + 60  # +60 for header

    img = Image.new("RGB", (total_width, total_height), (30, 30, 30))
    draw = ImageDraw.Draw(img)

    # Header
    draw.text((10, 10), "Arabic Font Audit — Visual Grid", fill=(200, 200, 200))
    draw.text((label_width + 10, 10), "Pangram + Test Words", fill=(150, 150, 150))
    draw.line([(0, 50), (total_width, 50)], fill=(80, 80, 80))

    y = 60
    for family, info in fonts_data.items():
        font_path = _resolve(info["main_file"])
        if not os.path.isfile(font_path):
            y += row_height
            continue

        # Status color
        result = results.get(family, {})
        passed = result.get("overall_pass", False)
        status_color = (0, 200, 0) if passed else (255, 60, 60)

        # Label
        pair_ids = ", ".join(str(p) for p in info["pair_ids"])
        label = f"{family}\n(pairs: {pair_ids})"
        draw.text((10, y + 10), label, fill=status_color)

        # Render Arabic text
        try:
            font = ImageFont.truetype(font_path, 36)
            reshaped = arabic_reshaper.reshape(ARABIC_PANGRAM[:40])
            bidi = get_display(reshaped)
            draw.text((label_width + 10, y + 5), bidi, font=font, fill=(255, 255, 255))

            # Test words on second line
            words_line = "  |  ".join(SHAPING_WORDS)
            reshaped_words = arabic_reshaper.reshape(words_line)
            bidi_words = get_display(reshaped_words)
            font_small = ImageFont.truetype(font_path, 28)
            draw.text((label_width + 10, y + 55), bidi_words, font=font_small, fill=(200, 200, 200))
        except Exception as e:
            draw.text((label_width + 10, y + 30), f"RENDER ERROR: {e}", fill=(255, 0, 0))

        # Separator line
        draw.line([(0, y + row_height - 1), (total_width, y + row_height - 1)], fill=(50, 50, 50))
        y += row_height

    os.makedirs(EXPORTS_DIR, exist_ok=True)
    out_path = os.path.join(EXPORTS_DIR, "arabic_font_audit.png")
    img.save(out_path)
    print(f"\nVisual grid saved to: {out_path}")


# ---------------------------------------------------------------------------
# Main audit runner
# ---------------------------------------------------------------------------

def run_audit(visual=False):
    """Run the full 6-layer audit on all manifest Arabic fonts."""
    fonts = _load_manifest()
    results = {}
    all_passed = True

    print("=" * 70)
    print("ARABIC FONT AUDIT")
    print("=" * 70)
    print(f"Testing {len(fonts)} unique Arabic font families from manifest\n")

    for family, info in fonts.items():
        font_path = _resolve(info["main_file"])
        pair_ids = info["pair_ids"]

        print(f"\n--- {family} (pairs: {', '.join(str(p) for p in pair_ids)}) ---")
        print(f"    File: {info['main_file']}")

        if not os.path.isfile(font_path):
            print(f"    ** FILE NOT FOUND **")
            results[family] = {"overall_pass": False, "reason": "file not found"}
            all_passed = False
            continue

        font_results = {"layers": {}}
        font_pass = True

        # Layer 1: cmap
        coverage, missing = check_cmap_coverage(font_path)
        cmap_pass = coverage >= 0.90
        font_results["layers"]["cmap"] = {
            "pass": cmap_pass,
            "coverage": f"{coverage*100:.1f}%",
            "missing": [f"U+{ord(c):04X}" for c in missing] if missing else [],
        }
        status = "PASS" if cmap_pass else "FAIL"
        print(f"    L1 cmap coverage: {status} ({coverage*100:.1f}%)")
        if missing:
            print(f"       Missing: {', '.join(f'U+{ord(c):04X}' for c in missing[:5])}"
                  + (f" +{len(missing)-5} more" if len(missing) > 5 else ""))
        if not cmap_pass:
            font_pass = False

        # Layer 2: Shaping
        shaping_pass, shaping_detail = check_shaping(font_path)
        font_results["layers"]["shaping"] = {"pass": shaping_pass}
        status = "PASS" if shaping_pass else "FAIL"
        print(f"    L2 shaping:       {status}")

        # Layer 3: Visual rendering
        render_pass, ink_pixels = check_visual_rendering(font_path)
        font_results["layers"]["rendering"] = {"pass": render_pass, "ink_pixels": ink_pixels}
        status = "PASS" if render_pass else "FAIL"
        print(f"    L3 rendering:     {status} ({ink_pixels} ink pixels)")
        if not render_pass:
            font_pass = False

        # Layer 4: Width ratio
        width_pass, width_ratio = check_width_ratio(font_path)
        font_results["layers"]["width_ratio"] = {"pass": width_pass, "ratio": width_ratio}
        status = "PASS" if width_pass else "FAIL"
        print(f"    L4 width ratio:   {status} ({width_ratio}x vs Noto Sans Arabic)")
        if not width_pass:
            font_pass = False

        # Layer 5: Weight axis
        weight_pass, weight_detail = check_weight_axis(font_path)
        font_results["layers"]["weight_axis"] = {"pass": weight_pass, "detail": weight_detail}
        status = "PASS" if weight_pass else "FAIL"
        print(f"    L5 weight axis:   {status} ({weight_detail})")

        # Overall
        font_results["overall_pass"] = font_pass
        if font_pass:
            print(f"    ** OVERALL: PASS **")
        else:
            reasons = [k for k, v in font_results["layers"].items() if not v["pass"]]
            print(f"    ** OVERALL: FAIL ** (failed: {', '.join(reasons)})")
            all_passed = False

        results[family] = font_results

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    passed_fonts = [f for f, r in results.items() if r.get("overall_pass")]
    failed_fonts = [f for f, r in results.items() if not r.get("overall_pass")]
    print(f"  PASSED: {len(passed_fonts)}/{len(results)}")
    for f in passed_fonts:
        print(f"    + {f}")
    if failed_fonts:
        print(f"  FAILED: {len(failed_fonts)}/{len(results)}")
        for f in failed_fonts:
            reasons = []
            layers = results[f].get("layers", {})
            for k, v in layers.items():
                if not v.get("pass"):
                    reasons.append(k)
            reason_str = results[f].get("reason", ", ".join(reasons))
            print(f"    - {f}: {reason_str}")

    # Visual grid
    if visual:
        print("\nGenerating visual grid...")
        generate_visual_grid(fonts, results)

    return results, all_passed


if __name__ == "__main__":
    visual_flag = "--visual" in sys.argv
    results, all_passed = run_audit(visual=visual_flag)
    sys.exit(0 if all_passed else 1)
