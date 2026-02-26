"""Render random scenes as a top-down grid for visual validation.

Usage:
    uv run python -m scene_gen.render_scenes                    # 16 random seeds
    uv run python -m scene_gen.render_scenes --seeds 42 43 44   # specific seeds
    uv run python -m scene_gen.render_scenes --count 25         # 25 random scenes
    uv run python -m scene_gen.render_scenes --out docs/scenes  # custom output dir
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scene_gen import SceneComposer
from scene_gen.archetypes import list_archetypes
from scene_gen.composer import describe_scene, scene_id

ROOM_XML = str(Path(__file__).resolve().parent.parent / "worlds" / "room.xml")

# Render settings
CELL_W = 500
CELL_H = 500
LABEL_H = 32
BG_COLOR = (40, 42, 48)
LABEL_BG = (30, 32, 36)
LABEL_FG = (220, 220, 220)


def _setup_scene() -> tuple[mujoco.MjModel, mujoco.MjData, SceneComposer]:
    """Load room.xml with full obstacle slots for scene generation."""
    spec = mujoco.MjSpec.from_file(ROOM_XML)
    # Increase offscreen framebuffer for high-res renders
    spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, CELL_W)
    spec.visual.global_.offheight = max(spec.visual.global_.offheight, CELL_H)
    SceneComposer.prepare_spec(spec)
    model = spec.compile()
    data = mujoco.MjData(model)

    # Hide distractors (keep target visible — it's part of the scene)
    for name in [
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


def _make_topdown_camera() -> mujoco.MjvCamera:
    """Top-down camera covering the full arena."""
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, 0]
    cam.azimuth = 0
    cam.elevation = -90
    # Arena is ±4m. MuJoCo default FOV is 45°.
    # distance = half_extent / tan(FOV/2) = 4 / tan(22.5°) ≈ 9.66
    cam.distance = 10.0
    return cam


def _render_scene(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    composer: SceneComposer,
    renderer: mujoco.Renderer,
    cam: mujoco.MjvCamera,
    seed: int,
    archetype: str | None = None,
) -> tuple[np.ndarray, str]:
    """Generate and render a random scene. Returns (pixels, description)."""
    scene = composer.random_scene(seed=seed, archetype=archetype)
    composer.apply(scene)

    renderer.update_scene(data, cam)
    pixels = renderer.render().copy()

    desc = describe_scene(scene, seed=seed)
    return pixels, desc


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


def render_scene_grid(
    seeds: list[int],
    out_dir: Path,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    composer: SceneComposer,
    archetype: str | None = None,
) -> Path:
    """Render a grid of scenes, one per seed. Returns output path."""
    n = len(seeds)
    cols = min(4, n)
    rows = math.ceil(n / cols)

    cell_total_h = CELL_H + LABEL_H
    grid_w = cols * CELL_W
    grid_h = rows * cell_total_h

    grid = Image.new("RGB", (grid_w, grid_h), BG_COLOR)
    draw = ImageDraw.Draw(grid)
    font = _try_load_font(14)

    renderer = mujoco.Renderer(model, height=CELL_H, width=CELL_W)
    cam = _make_topdown_camera()

    for idx, seed in enumerate(seeds):
        pixels, desc = _render_scene(
            model, data, composer, renderer, cam, seed, archetype
        )

        cell_img = Image.fromarray(pixels)
        col = idx % cols
        row = idx // cols
        x = col * CELL_W
        y = row * cell_total_h

        grid.paste(cell_img, (x, y))

        # Label: seed + object count
        sid = scene_id(seed)
        # Count objects from the description (first line has "N objects")
        first_line = desc.split("\n")[0]
        label = f"#{sid}  {first_line.split('  ')[-1]}"

        label_y = y + CELL_H
        draw.rectangle([x, label_y, x + CELL_W, label_y + LABEL_H], fill=LABEL_BG)
        bbox = font.getbbox(label)
        tw = bbox[2] - bbox[0]
        tx = x + (CELL_W - tw) // 2
        ty = label_y + (LABEL_H - (bbox[3] - bbox[1])) // 2
        draw.text((tx, ty), label, fill=LABEL_FG, font=font)

        # Print full description to terminal
        print(f"  [{idx + 1}/{n}] {desc.split(chr(10))[0]}")

    renderer.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "scenes.png"
    grid.save(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Render random scene grids")
    parser.add_argument("--seeds", nargs="*", type=int, help="Specific seeds to render")
    parser.add_argument(
        "--count", type=int, default=16, help="Number of random scenes (default: 16)"
    )
    parser.add_argument("--out", default="docs/scenes", help="Output directory")
    parser.add_argument(
        "--archetype",
        choices=[*list_archetypes(), "random"],
        default="random",
        help="Room archetype (default: random = picks randomly each scene)",
    )
    args = parser.parse_args()

    if args.seeds:
        seeds = args.seeds
    else:
        rng = np.random.default_rng()
        seeds = [int(rng.integers(0, 2**32)) for _ in range(args.count)]

    out_dir = Path(args.out)

    print("Setting up MuJoCo scene...")
    model, data, composer = _setup_scene()

    archetype = args.archetype
    print(f"Rendering {len(seeds)} scenes (archetype={archetype})...")
    path = render_scene_grid(seeds, out_dir, model, data, composer, archetype=archetype)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
