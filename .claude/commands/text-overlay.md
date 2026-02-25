# /text-overlay — Bilingual Text Overlay Generator

Generate Instagram-quality bilingual (English + Arabic) text overlays on images.

## Arguments
$ARGUMENTS

## Phase 1: Parse Arguments

Parse `$ARGUMENTS` to extract these fields:
- **main_text**: The first quoted string (positional), or the value after `--text`. This is the large headline text.
- **subtext**: Value after `--subtext`. The smaller text above the main text.
- **text_ar** (Arabic main): Value after `--text-ar`.
- **subtext_ar** (Arabic subtext): Value after `--subtext-ar`.
- **image_path**: Value after `--image`. REQUIRED — fail if missing.
- **layout_mode**: Value after `--layout`. Either `auto` (default) or `center`.

Verify the image file exists at the resolved path. If the path is relative, resolve it relative to the current working directory.

At least one of `main_text` or `text_ar` must be provided. Fail with a helpful message if neither is given.

## Phase 2: Check for Existing State (Iteration Detection)

Before running analysis, check if a state file exists for this image:
- Look for `/tmp/{image_stem}_overlay_state.json`.
- If found AND the user's message sounds like feedback/iteration (e.g., "make it more yellow", "try a different font", "move it up", "add a shadow") — this is an **iteration request**. Jump to the **Iteration** section below.
- If not found, or if the user is providing fresh text + image — proceed with a fresh run.

## Phase 3: Analyze Image

**Check if the image is a RAW file** (extensions: `.ARW`, `.CR2`, `.CR3`, `.NEF`, `.ORF`, `.RAF`, `.DNG`, `.RW2`, `.PEF`, `.SRW` — case-insensitive):

**If RAW file:**
1. First, create a preview JPEG for fast analysis. Spawn a **Haiku sub-agent** (Task tool, model: **haiku**, subagent_type: Bash):
   ```
   Determine if the image is landscape or portrait using sips:
   sips -g pixelWidth -g pixelHeight "<resolved_image_path>"

   If landscape (width > height), resize by width:
   sips -s format jpeg -s formatOptions 85 --resampleWidth 2000 "<resolved_image_path>" --out /tmp/<image_stem>_preview.jpg

   If portrait (height >= width), resize by height:
   sips -s format jpeg -s formatOptions 85 --resampleHeight 2000 "<resolved_image_path>" --out /tmp/<image_stem>_preview.jpg

   Return the preview path: /tmp/<image_stem>_preview.jpg
   ```
2. Run `analyze.py` on the **preview** (not the original RAW):
   ```bash
   python3 scripts/analyze.py /tmp/<image_stem>_preview.jpg --layout <layout_mode>
   ```
3. Save the preview path — the Creative Director sub-agent in Phase 4 will READ the **preview** to see the image.
4. But remember the **original RAW path** — Phase 5 render must use the original for full-resolution output.

**If NOT a RAW file:**
Run the analysis script as before:
```bash
python3 scripts/analyze.py <resolved_image_path> --layout <layout_mode>
```

Capture the JSON output. This gives: `palette`, `color_candidates`, `regions`, `best_placements`, and `rectangle_candidates`.

## Phase 4: Creative Decisions

Spawn a **sub-agent** (Task tool, model: **opus**, subagent_type: general-purpose) with the following prompt. The sub-agent will act as the **Creative Director** — it sees the image, understands the analysis data, and makes all aesthetic decisions.

### Sub-agent prompt (send ALL of the following):

---

**You are the Creative Director for a bilingual text overlay system.** You will receive an image, analysis data, a font manifest, and the user's text. Your job is to make all creative decisions and output a structured JSON.

**READ the image** at: `<preview_path if RAW file, otherwise resolved_image_path>` (use the Read tool to view it)

**READ the font manifest** at: `fonts/manifest.json` (relative to the project root)

**Analysis data from the image:**
```json
<paste the full analysis JSON output here>
```

**User's text inputs:**
- Main text (EN): `<main_text or empty>`
- Subtext (EN): `<subtext or empty>`
- Main text (AR): `<text_ar or empty>`
- Subtext (AR): `<subtext_ar or empty>`

**Layout mode:** `<layout_mode>`

---

### Decision Rules (include these in the sub-agent prompt):

**COLOR SELECTION (CRITICAL — read carefully):**
1. **NEVER use generic white, off-white, cream, or grey.** No #FFFFFF, #FFF8E7, #F5F0E8, #CDCFCD, #E0E0E0, or any near-white/near-grey. These are boring and defeat the purpose of palette analysis.
2. **ALWAYS pick a color that feels INTENTIONAL and tied to the image.** The text color should look like it belongs in the photo's world.
3. Pick from the `color_candidates` list. Every candidate is derived from the image's actual palette. Prioritize:
   - **"palette" source**: direct palette colors — use these when they have decent contrast (>=2.5)
   - **"tinted_light" or "tinted_dark"**: lightened/darkened palette colors — great for readability
   - **"complementary"**: opposite hue — bold and intentional, use for energetic/dramatic images
   - **"analogous_warm" or "analogous_cool"**: harmonious neighbors — subtle and sophisticated
4. Warm images (sunset, desert, golden hour) -> pick warm-toned candidates (golds, ambers, warm browns, terracotta)
5. Cool images (ocean, snow, night, blue hour) -> pick cool-toned candidates (steel blue, teal, slate, cool greens)
6. **Contrast requirements**: minimum 2.5 against the **rectangle region's avg_color**. Ideal >=3.5. But contrast is NOT the only factor — a 3.0 contrast color that matches the mood beats a 5.0 contrast generic white.
7. Be bold. A deep amber on a sunset photo, a forest green on a nature shot, a dusty blue on an ocean scene — these are the kind of choices that make overlays feel premium.

**FONT SELECTION:**
1. Look at the image and classify it: What's the scene? (travel, food, architecture, nature, urban, fashion, etc.)
2. What's the mood? (energetic, calm, luxurious, casual, dramatic, minimal, etc.)
3. Filter the font manifest `pairs` array — find pairs whose `scene` and `mood` tags overlap with your classification.
4. From the shortlist, pick the pair that best matches the image's energy level and the text's content/tone.
5. Consider text length: very short text (1-2 words) works with bold/impact fonts; longer text works better with clean/modern fonts.
6. **Arabic rendering quality**: If Arabic text is present, prefer pairs that use these top-rendering Arabic fonts: **Noto Sans Arabic** (pairs 3, 14), **Noto Kufi Arabic** (pairs 4, 10), **IBM Plex Sans Arabic** (pairs 7, 8), **Amiri** (pair 11). Other Arabic fonts work but may have minor glyph gaps with certain letter combinations.

**RECTANGLE SELECTION (NEW — this is how text is placed):**

All text lives inside a single bounding rectangle. The `rectangle_candidates` array from the analysis contains pre-scored positions. Each candidate has: `center_x_pct`, `center_y_pct`, `width_pct`, `score`, `avg_color`, `region_luminance`.

1. **Pick a rectangle** from `rectangle_candidates`. Prefer the highest-scored candidate, but you may override if a lower-scored position makes more creative sense (e.g., avoids covering an important subject you can see in the image).
2. You may **nudge** the chosen candidate's `center_x_pct` and `center_y_pct` by up to +-0.05 to fine-tune placement.
3. You may **adjust** `width_pct` within the allowed range for the layout mode:
   - `center` mode: 0.30 - 0.50
   - `auto` mode: 0.15 - 0.30
4. Use the candidate's `avg_color` and `region_luminance` to validate your color choice has sufficient contrast.

**ALIGNMENT:**
- **center placement** (rectangle center near image center) -> alignment: `"center"`
- **left-side placement** (rectangle center_x_pct < 0.4) -> alignment: `"left"` (EN left-aligned, AR right-aligned within rect)
- **right-side placement** (rectangle center_x_pct > 0.6) -> alignment: `"right"` (EN right-aligned, AR left-aligned within rect)
- When in doubt, use `"center"`.

**ROW ORDER:**
You decide the vertical stacking order of text rows. Available row types:
- `"subtext_en"` — English subtext (small, light weight)
- `"main_en"` — English main text (large, bold)
- `"main_ar"` — Arabic main text (large, bold)
- `"subtext_ar"` — Arabic subtext (small, light weight)
- `"subtext_line"` — Both subtexts on the same line (EN left, AR right)

Common patterns:
- `["subtext_en", "main_en", "main_ar", "subtext_ar"]` — classic bilingual stack
- `["main_en", "main_ar", "subtext_line"]` — mains stacked, subtexts together below
- `["subtext_line", "main_en", "main_ar"]` — subtexts above as a header line
- `["main_en", "subtext_en", "main_ar", "subtext_ar"]` — each language grouped

Pick the order that makes visual sense for the placement and text content. Only include row types that have actual text content.

**SIZING:**
- `subtext_ratio`: Size of subtext relative to the main text. Default 0.30. Range: 0.25-0.40.
- `line_spacing`: Multiplier for gap between lines. Default 1.2. Range: 1.0-1.5.
- Note: Both main texts are automatically sized independently to fill the rectangle width. No `main_width_ratio` or `arabic_scale_factor` needed — the renderer handles equal visual volume.

**READABILITY ENHANCEMENTS:**
- **Default is ALWAYS "none"** — no shadow, no border, no gradient.
- The correct color choice should handle readability. Do NOT add enhancements unless absolutely necessary.
- Only if contrast is critically low (below 2.5:1 against the rectangle region AND no better color exists), you may choose ONE of:
  - `drop-shadow`: subtle, for busy backgrounds
  - `stroke`: thin dark outline, for maximum contrast
  - `gradient`: dark gradient behind text, for photos with highly variable backgrounds
- Explain your reasoning for enhancement or lack thereof.

---

### Sub-agent output format (must be EXACTLY this JSON structure):

```json
{
  "texts": {
    "main": "<main_text>",
    "subtext": "<subtext>",
    "main_ar": "<text_ar>",
    "subtext_ar": "<subtext_ar>"
  },
  "font": {
    "pair_id": <number>,
    "main_en": "fonts/english/<filename>.ttf",
    "sub_en": "fonts/english/<filename>.ttf",
    "main_ar": "fonts/arabic/<filename>.ttf",
    "sub_ar": "fonts/arabic/<filename>.ttf"
  },
  "color": "#HEXCOLOR",
  "rectangle": {
    "center_x_pct": <0.0-1.0>,
    "center_y_pct": <0.0-1.0>,
    "width_pct": <0.15-0.50>,
    "alignment": "<left|center|right>"
  },
  "row_order": ["subtext_en", "main_en", "main_ar", "subtext_ar"],
  "sizing": {
    "subtext_ratio": <0.2-0.5>,
    "line_spacing": <1.0-1.5>
  },
  "readability": {
    "type": "none",
    "reason": "<brief explanation>"
  }
}
```

The sub-agent MUST output this JSON block. Extract it from the sub-agent's response.

## Phase 5: Render

1. Save the decisions JSON to a temporary file (e.g., `/tmp/text_overlay_decisions.json`).
2. Run the render script:
```bash
python3 scripts/render.py <resolved_image_path> /tmp/text_overlay_decisions.json
```
   - For RAW files: pass the **original RAW path** (not the preview). The renderer handles RAW->TIFF conversion internally.
   - Output is always **PNG** (lossless quality). The renderer produces `{stem}_overlay.png`.
3. Capture the output JSON which contains `output_image` and `state_file` paths.
4. **If RAW file**: clean up the preview JPEG: `rm /tmp/<image_stem>_preview.jpg`

## Phase 6: Present Result

1. Show the output image to the user by reading it with the Read tool.
2. Display a summary of key decisions:
   - Font pair chosen and why
   - Color chosen and its contrast ratio
   - Rectangle position and size
   - Row order and alignment
   - Whether any readability enhancement was used
3. Tell the user the output file path.
4. Mention they can iterate: "You can ask me to adjust the color, size, position, font, or add effects like shadows."

---

## Iteration Mode

When the user provides feedback after an overlay has been generated:

### Step 1: Load the existing state
Read the state file at `/tmp/{image_stem}_overlay_state.json`.

### Step 2: Detect state format
Check if the state JSON contains a `"rectangle"` key:
- **If yes**: this is a rectangle-layout state. Use rectangle iteration rules below.
- **If no**: this is a **legacy** state (old placement-based format). Use legacy iteration rules.

### Step 3: Classify the feedback

**Simple tweaks (handle directly — NO sub-agent needed):**

For **rectangle-layout** states:
- **Color changes**: "more yellow" -> shift hue toward yellow; "darker" -> reduce lightness by 15%; "lighter" -> increase lightness by 15%; "make it red/blue/green/gold" -> pick a hue in that family. Use Python color manipulation logic or just pick a reasonable hex value.
- **Size changes**: "bigger" -> increase `rectangle.width_pct` by 0.05; "smaller" -> decrease by 0.05; "bigger subtext" -> increase `sizing.subtext_ratio` by 0.05.
- **Position changes**: "move up" -> decrease `rectangle.center_y_pct` by 0.08; "move down" -> increase by 0.08; "move left" -> decrease `rectangle.center_x_pct` by 0.05; "move right" -> increase by 0.05; "center it" -> set center_x_pct to 0.5, alignment to "center".
- **Readability toggles**: "add a shadow" -> set `readability.type` to "drop-shadow"; "add a border/stroke" -> "stroke"; "add a gradient" -> "gradient"; "remove the shadow/effect" -> "none".
- **Spacing**: "more space between lines" -> increase `sizing.line_spacing` by 0.1; "tighter" -> decrease by 0.1.
- **Alignment**: "left align" -> set `rectangle.alignment` to "left"; "right align" -> "right"; "center align" -> "center".

For **legacy** states (no `"rectangle"` key):
- **Color changes**: same as above.
- **Size changes**: "bigger" -> increase `sizing.main_width_ratio` by 0.10; "smaller" -> decrease by 0.10; "bigger subtext" -> increase `sizing.subtext_ratio` by 0.05.
- **Position changes**: "move up" -> decrease `placement.y_pct` by 0.10; "move down" -> increase by 0.10; "move left" -> decrease `placement.x_pct` by 0.05; "move right" -> increase by 0.05; "center it" -> set anchor to "center", x_pct to 0.5.
- **Readability toggles**: same as above.
- **Spacing**: same as above.

For simple tweaks: modify the state JSON directly, save the updated decisions to `/tmp/text_overlay_decisions.json`, and re-run ONLY the render script. Do NOT re-run analysis or the creative sub-agent.

**Complex tweaks (re-spawn sub-agent):**
- "try a different font" -> re-run Phase 4 with a note to EXCLUDE the previously used pair_id
- "completely different style" -> re-run Phase 4 from scratch with fresh creative direction
- "try again" / "I don't like it" -> re-run Phase 4 with instruction to avoid the previous choices
- "different color that's more [description]" -> re-run Phase 4 with that color preference as a constraint

For complex tweaks: re-run the analysis (Phase 3) if not cached, then re-run the creative sub-agent (Phase 4) with the additional constraints, then render (Phase 5).
