"""Centralized image compositing for render outputs.

All image layout (grids, filmstrips, ROM bars) flows through this module.
Replaces duplicated PIL compositing code across renders.py, validation.py,
and component_renders.py.

Usage:
    from botcad.emit.composite import grid, filmstrip

    img = grid("Title", legends, views, cols=3, cell_w=800, cell_h=800)
    img = filmstrip("ROM sweep", frames, angles_deg, collisions, cell_w=600, cell_h=600)
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PNG_DPI = (150, 150)

# ── Fonts ──

# Title: large bold-ish text for sheet headers
# Label: medium text for view labels, legend entries, angle labels
# Use DejaVu Sans which ships with Pillow on most systems; fall back to default.
FONT_TITLE: ImageFont.FreeTypeFont | ImageFont.ImageFont
FONT_LABEL: ImageFont.FreeTypeFont | ImageFont.ImageFont
_FONT_PATHS = [
    # Preferred: Input Sans Narrow (user font dir)
    (
        str(Path.home() / "Library/Fonts/InputSansNarrow-Bold.ttf"),
        str(Path.home() / "Library/Fonts/InputSansNarrow-Regular.ttf"),
    ),
    # Fallbacks
    ("DejaVuSans-Bold", "DejaVuSans"),
    ("Arial Bold", "Arial"),
]
FONT_TITLE = ImageFont.load_default()
FONT_LABEL = ImageFont.load_default()
for _bold_path, _regular_path in _FONT_PATHS:
    try:
        FONT_TITLE = ImageFont.truetype(_bold_path, 18)
        FONT_LABEL = ImageFont.truetype(_regular_path, 14)
        break
    except OSError:
        continue

# Legend swatch size (pixels)
_SWATCH = 14


def grid(
    title: str,
    legends: list[tuple[str, tuple[int, int, int]]],
    views: list[tuple[Image.Image, str]],
    cols: int = 3,
    cell_w: int = 800,
    cell_h: int = 800,
) -> Image.Image:
    """Composite view images into an NxM grid with title and legend.

    Args:
        title: Header text.
        legends: [(label, (r,g,b)), ...] for color swatches.
        views: [(Image, label), ...] one per grid cell.
        cols: Number of columns.
        cell_w: Width of each cell.
        cell_h: Height of each cell.
    """
    n_rows = math.ceil(len(views) / cols)
    margin = 5
    label_h = 28
    title_h = 50

    grid_w = cell_w * cols + margin * (cols + 1)
    grid_h = title_h + (cell_h + label_h + margin) * n_rows + margin

    canvas = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    draw.text((margin, margin), title, fill=(0, 0, 0), font=FONT_TITLE)

    # Deduplicated legend
    seen = set()
    unique = []
    for entry in legends:
        if entry[0] not in seen:
            seen.add(entry[0])
            unique.append(entry)

    legend_y = margin + 24
    x_off = margin
    for text, color in unique:
        draw.rectangle(
            [x_off, legend_y, x_off + _SWATCH, legend_y + _SWATCH], fill=color
        )
        draw.text(
            (x_off + _SWATCH + 4, legend_y - 1), text, fill=(0, 0, 0), font=FONT_LABEL
        )
        # Measure actual text width for proper spacing
        bbox = draw.textbbox((0, 0), text, font=FONT_LABEL)
        text_w = bbox[2] - bbox[0]
        x_off += _SWATCH + 4 + text_w + 12

    for idx, (img, label) in enumerate(views):
        col = idx % cols
        row = idx // cols
        x = margin + col * (cell_w + margin)
        y = title_h + margin + row * (cell_h + label_h + margin)
        draw.text((x, y), label, fill=(0, 0, 0), font=FONT_LABEL)
        img_y = y + label_h
        draw.rectangle(
            [x - 1, img_y - 1, x + cell_w, img_y + cell_h],
            outline=(0, 0, 0),
            width=1,
        )
        canvas.paste(img, (x, img_y))

    return canvas


def filmstrip(
    title: str,
    frames: list[Image.Image],
    angles_deg: list[float],
    collisions: list[bool] | None = None,
    cell_w: int = 600,
    cell_h: int = 600,
    elapsed: float = 0.0,
) -> Image.Image:
    """Composite a horizontal filmstrip with optional collision indicators.

    Args:
        title: Sweep name for the header.
        frames: Rendered frame images.
        angles_deg: Angle label for each frame.
        collisions: Per-frame collision flag (None = no collision checking).
        cell_w: Width of each frame cell.
        cell_h: Height of each frame cell.
        elapsed: Render time for the header.
    """
    n = len(frames)
    if n == 0:
        return Image.new("RGB", (100, 100), (255, 255, 255))

    if collisions is None:
        collisions = [False] * n

    has_collisions = any(collisions)

    margin = 8
    header_h = 38
    label_h = 24
    border_w = 3

    canvas_w = cell_w * n + margin * (n + 1)
    canvas_h = header_h + cell_h + label_h + margin * 2

    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Header
    lo = min(angles_deg) if angles_deg else 0
    hi = max(angles_deg) if angles_deg else 0
    col_text = "  COLLISIONS DETECTED" if has_collisions else ""
    time_text = f"  ({elapsed:.1f}s)" if elapsed > 0 else ""
    header = f"ROM sweep: {title}  [{lo:+.0f}° .. {hi:+.0f}°]{time_text}{col_text}"
    draw.text((margin, margin), header, fill=(0, 0, 0), font=FONT_LABEL)

    # ROM bar
    _draw_rom_bar(
        draw,
        margin,
        margin + 20,
        min(300, canvas_w - 2 * margin),
        10,
        lo,
        hi,
        has_collisions,
    )

    # Frames
    for i, (frame, angle, has_col) in enumerate(zip(frames, angles_deg, collisions)):
        x = margin + i * (cell_w + margin)
        y = header_h

        if has_col:
            draw.rectangle(
                [
                    x - border_w,
                    y - border_w,
                    x + cell_w + border_w,
                    y + cell_h + border_w,
                ],
                fill=(220, 40, 40),
            )
        else:
            draw.rectangle(
                [x - 1, y - 1, x + cell_w, y + cell_h],
                outline=(0, 0, 0),
                width=1,
            )

        canvas.paste(frame, (x, y))
        color = (220, 40, 40) if has_col else (80, 80, 80)
        draw.text((x, y + cell_h + 2), f"{angle:+.0f}°", fill=color, font=FONT_LABEL)

    return canvas


def _draw_rom_bar(
    draw: ImageDraw.ImageDraw,
    x0: int,
    y: int,
    width: int,
    height: int,
    lo_deg: float,
    hi_deg: float,
    has_collisions: bool,
) -> None:
    """Draw a ROM range bar with center tick."""
    draw.rectangle(
        [x0, y, x0 + width, y + height],
        fill=(230, 230, 230),
        outline=(180, 180, 180),
    )
    px_lo = int((lo_deg + 180) / 360 * width)
    px_hi = int((hi_deg + 180) / 360 * width)
    px_lo = max(0, min(width, px_lo))
    px_hi = max(px_lo + 1, min(width, px_hi))
    bar_color = (255, 180, 50) if has_collisions else (80, 180, 80)
    draw.rectangle(
        [x0 + px_lo, y, x0 + px_hi, y + height],
        fill=bar_color,
    )
    # Center tick at 0°
    center_px = int(180 / 360 * width)
    draw.line(
        [x0 + center_px, y, x0 + center_px, y + height],
        fill=(100, 100, 100),
        width=1,
    )
