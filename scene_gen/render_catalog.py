"""Render named variations of each concept into labeled PNG grids.

Usage:
    uv run python scene_gen/render_catalog.py              # all concepts
    uv run python scene_gen/render_catalog.py chair         # single concept
    uv run python scene_gen/render_catalog.py --out docs/   # custom output dir
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scene_gen import SceneComposer, concepts
from scene_gen.composer import PlacedObject
from scene_gen.primitives import Prim

ROOM_XML = str(Path(__file__).resolve().parent.parent / "worlds" / "room.xml")

# Render settings
CELL_W = 400
CELL_H = 400
LABEL_H = 36
BG_COLOR = (40, 42, 48)
LABEL_BG = (30, 32, 36)
LABEL_FG = (220, 220, 220)


def _setup_scene() -> tuple[mujoco.MjModel, mujoco.MjData, SceneComposer]:
    """Load room.xml with obstacle slots, return (model, data, composer)."""
    spec = mujoco.MjSpec.from_file(ROOM_XML)
    SceneComposer.prepare_spec(spec, max_objects=1, geoms_per_object=8)
    model = spec.compile()
    data = mujoco.MjData(model)

    # Hide the target and distractor cubes so they don't appear in renders
    for name in [
        "target_cube",
        "distractor_0_cube",
        "distractor_1_cube",
        "distractor_2_cube",
        "distractor_3_cube",
    ]:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid >= 0:
            model.geom_rgba[gid] = [0, 0, 0, 0]

    mujoco.mj_forward(model, data)
    composer = SceneComposer(model, data)
    return model, data, composer


def _camera_for_prims(prims: tuple[Prim, ...]) -> mujoco.MjvCamera:
    """Compute a camera that frames the given prims nicely.

    Uses a fixed 3/4 overhead angle (good for all furniture) and adapts
    only the lookat height and distance based on the bounding box.
    """
    z_min = float("inf")
    z_max = float("-inf")
    xy_max = 0.0
    for p in prims:
        px, py, pz = p.pos
        sx, sy, sz = p.size
        z_min = min(z_min, pz - sz)
        z_max = max(z_max, pz + sz)
        xy_max = max(xy_max, abs(px) + sx, abs(py) + sy)

    obj_height = z_max - z_min
    obj_width = xy_max * 2
    extent = max(obj_height, obj_width, 0.3)

    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, (z_min + z_max) / 2]
    cam.azimuth = 145
    cam.elevation = -25
    cam.distance = extent * 2.0
    return cam


def _render_variation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    composer: SceneComposer,
    renderer: mujoco.Renderer,
    concept_name: str,
    params: object,
) -> np.ndarray:
    """Render a single concept variation, return RGB array."""
    concept_mod = concepts.get(concept_name)
    prims = concept_mod.generate(params)

    obj = PlacedObject(
        concept=concept_name, params=params, pos=(0.0, 0.0), rotation=0.0
    )
    composer.apply([obj])

    cam = _camera_for_prims(prims)
    renderer.update_scene(data, cam)
    return renderer.render().copy()


def _try_load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try to load a nice font, fall back to default."""
    candidates = [
        "/System/Library/Fonts/SFNSMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def render_concept(
    concept_name: str,
    out_dir: Path,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    composer: SceneComposer,
) -> Path | None:
    """Render all variations of a concept into a labeled grid PNG."""
    concept_mod = concepts.get(concept_name)
    variations: dict[str, object] | None = getattr(concept_mod, "VARIATIONS", None)
    if not variations:
        print(f"  {concept_name}: no VARIATIONS defined, skipping")
        return None

    names = list(variations.keys())
    n = len(names)
    cols = min(4, n)
    rows = math.ceil(n / cols)

    cell_total_h = CELL_H + LABEL_H
    grid_w = cols * CELL_W
    grid_h = rows * cell_total_h

    grid = Image.new("RGB", (grid_w, grid_h), BG_COLOR)
    draw = ImageDraw.Draw(grid)
    font = _try_load_font(18)

    renderer = mujoco.Renderer(model, height=CELL_H, width=CELL_W)

    for idx, name in enumerate(names):
        params = variations[name]
        pixels = _render_variation(
            model, data, composer, renderer, concept_name, params
        )

        cell_img = Image.fromarray(pixels)
        col = idx % cols
        row = idx // cols
        x = col * CELL_W
        y = row * cell_total_h

        grid.paste(cell_img, (x, y))

        # Label below the render
        label_y = y + CELL_H
        draw.rectangle([x, label_y, x + CELL_W, label_y + LABEL_H], fill=LABEL_BG)
        bbox = font.getbbox(name)
        tw = bbox[2] - bbox[0]
        tx = x + (CELL_W - tw) // 2
        ty = label_y + (LABEL_H - (bbox[3] - bbox[1])) // 2
        draw.text((tx, ty), name, fill=LABEL_FG, font=font)

    renderer.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{concept_name}.png"
    grid.save(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Render concept variation catalogs")
    parser.add_argument("concepts", nargs="*", help="Concept names (default: all)")
    parser.add_argument("--out", default="docs/concepts", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    target_concepts = args.concepts or concepts.list_concepts()

    # Validate concept names
    available = concepts.list_concepts()
    for name in target_concepts:
        if name not in available:
            print(f"Error: unknown concept '{name}'. Available: {', '.join(available)}")
            sys.exit(1)

    print("Setting up MuJoCo scene...")
    model, data, composer = _setup_scene()

    for name in target_concepts:
        print(f"Rendering {name}...")
        path = render_concept(name, out_dir, model, data, composer)
        if path:
            print(f"  -> {path}")

    print("Done.")


if __name__ == "__main__":
    main()
