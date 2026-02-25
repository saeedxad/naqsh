# Naqsh (نقش) — Bilingual Text Overlay System

AI-driven bilingual (English + Arabic) text overlay generator for Instagram-quality photo overlays.

## Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Download fonts (all 29 families from Google Fonts + Fontshare)
python3 fonts/download_fonts.py
```

## Usage

Use the `/text-overlay` slash command in Claude Code:

```
/text-overlay "VANCOUVER" --text-ar "فانكوفر" --subtext "Canada" --subtext-ar "كندا" --image tests/photo.jpg --layout auto
```

### Arguments
- `--text` / first quoted string: English headline text
- `--text-ar`: Arabic headline text
- `--subtext`: English subtext
- `--subtext-ar`: Arabic subtext
- `--image`: Path to image (JPEG, PNG, or RAW: ARW/CR2/NEF/DNG/etc.)
- `--layout`: `auto` (side placement) or `center` (centered)

After generating, you can iterate with natural language feedback like "move it up", "try a different font", "make the color more golden", etc.

## Project Structure

- `scripts/analyze.py` — Image analysis: palette extraction, region scoring, placement candidates
- `scripts/render.py` — Text rendering engine with full Arabic BiDi support
- `scripts/test_arabic_fonts.py` — Font glyph coverage testing utility
- `fonts/` — Font manifest + download script (run `download_fonts.py` to fetch .ttf files)
- `fonts/manifest.json` — 15 bilingual font pairs with mood/scene tags
- `.claude/commands/text-overlay.md` — The slash command that orchestrates the full pipeline

## Architecture

The pipeline has 6 phases:
1. **Parse** user arguments
2. **Check** for existing state (iteration detection)
3. **Analyze** the image (palette, regions, placement candidates)
4. **Creative Director** sub-agent (Opus) makes aesthetic decisions: color, font, placement, sizing
5. **Render** the overlay using the decisions JSON
6. **Present** result and enable iteration

## Platform Notes

- RAW file conversion uses macOS `sips` command (macOS only)
- All other functionality is cross-platform
