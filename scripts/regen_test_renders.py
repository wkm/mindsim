#!/usr/bin/env python3
"""Regenerate all test render images and technical drawings.

Produces visual reference outputs at every design scale — component tear
sheets, bracket section drawings, ROM validation, and full bot builds.
These serve as screenshot tests: review them in git diff after any
geometry change to catch regressions visually.

Usage:
    uv run python scripts/regen_test_renders.py              # everything
    uv run python scripts/regen_test_renders.py --components  # component level only
    uv run python scripts/regen_test_renders.py --bots        # bot builds only
    uv run python scripts/regen_test_renders.py --rom         # ROM validation only

Outputs:
    botcad/components/test_*.png        — component & bracket 3D tear sheets
    botcad/components/drawing_*.svg     — component & bracket 2D section drawings
    botcad/components/test_rom_*.png    — subassembly ROM validation filmstrips
    bots/*/test_*.png                   — per-bot validation renders
    bots/*/test_*.pdf                   — per-bot assembly instruction PDFs
    bots/*/drawings/*.svg               — per-bot joint section drawings
    bots/*/*.step                       — per-bot STEP assemblies
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def regen_component_renders() -> None:
    """Regenerate component/bracket test images and technical drawings."""
    from botcad.component import Component
    from botcad.components import (
        OV5647,
        STS3215,
        LiPo2S,
        PiCamera2,
        PololuWheel90mm,
        RaspberryPiZero2W,
    )
    from botcad.emit.component_renders import (
        PNG_DPI,
        render_bracket_views,
        render_component_views,
        render_coupler_assembly_views,
        render_coupler_views,
        render_cradle_views,
    )
    from botcad.emit.drawings import emit_component_drawings

    out_dir = Path("botcad/components")

    # Component tear sheets (3D renders)
    components: list[tuple[str, str, Component]] = [
        ("servo", "STS3215 (position)", STS3215(continuous=False)),
        ("servo", "STS3215 (continuous)", STS3215(continuous=True)),
        ("wheel", "Pololu 90x10mm Wheel", PololuWheel90mm()),
        ("camera", "OV5647", OV5647()),
        ("camera", "PiCamera2", PiCamera2()),
        ("battery", "LiPo2S-1000", LiPo2S(1000)),
        ("compute", "RaspberryPiZero2W", RaspberryPiZero2W()),
    ]

    for category, name, comp in components:
        img = render_component_views(comp, name)
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        out_path = out_dir / f"test_{category}_{safe_name}.png"
        img.save(out_path, optimize=True, dpi=PNG_DPI)
        print(f"  component: {out_path}")

    # Bracket tear sheets (3D renders)
    servo = STS3215()
    for render_fn, label in [
        (render_bracket_views, "bracket"),
        (render_cradle_views, "cradle"),
        (render_coupler_views, "coupler"),
        (render_coupler_assembly_views, "coupler_assembly"),
    ]:
        img = render_fn(servo, "STS3215")
        out_path = out_dir / f"test_{label}_sts3215.png"
        img.save(out_path, optimize=True, dpi=PNG_DPI)
        print(f"  bracket: {out_path}")

    # Bracket technical drawings (2D section SVGs)
    svgs = emit_component_drawings("STS3215", out_dir)
    for svg_path in svgs:
        print(f"  drawing: {svg_path}")


def regen_rom_renders() -> None:
    """Regenerate subassembly ROM validation renders."""
    from botcad.bracket import BracketSpec, coupler_solid, cradle_solid, servo_solid
    from botcad.components import STS3215
    from botcad.validation import SolidPart, validate_subassembly

    servo = STS3215()
    spec = BracketSpec()
    out_dir = Path("botcad/components")

    result = validate_subassembly(
        name="coupler_sts3215",
        fixed=[
            SolidPart(
                "cradle",
                cradle_solid(servo, spec),
                rgba=(0.35, 0.55, 0.75, 1.0),
            ),
            SolidPart(
                "servo",
                servo_solid(servo),
                rgba=(0.15, 0.15, 0.15, 1.0),
                check_collision=False,
            ),
        ],
        moving=[
            SolidPart(
                "coupler",
                coupler_solid(servo, spec),
                rgba=(0.85, 0.35, 0.2, 1.0),
            ),
        ],
        pivot=servo.shaft_offset,
        axis=(0, 0, 1),
        range_rad=servo.range_rad,
        camera_distance=0.12,
        camera_azimuth=135,
        camera_elevation=-25,
    )
    out_path = out_dir / "test_rom_coupler_sts3215.png"
    result.save(out_path)
    print(f"  rom: {out_path}")


def regen_bot_outputs() -> None:
    """Rebuild all bots from their design.py files.

    Each design.py calls bot.solve() + bot.emit(), which produces
    STEP, STL, MuJoCo XML, renders, PDFs, and joint section drawings.
    """
    import os
    import subprocess

    bots_dir = Path("bots")
    env = os.environ.copy()
    # Ensure project root is on path for child processes
    env["PYTHONPATH"] = f".{os.pathsep}{env.get('PYTHONPATH', '')}"

    for design_py in sorted(bots_dir.glob("*/design.py")):
        bot_name = design_py.parent.name
        print(f"  bot: {bot_name} ...")
        result = subprocess.run(
            [sys.executable, str(design_py)],
            capture_output=True,
            text=True,
            env=env,
        )
        if result.returncode != 0:
            print(f"  FAILED: {bot_name}")
            print(result.stderr)
            sys.exit(1)
        for line in result.stdout.strip().splitlines():
            print(f"    {line}")
        print(f"  bot: {bot_name} done")


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate all test renders and technical drawings"
    )
    parser.add_argument(
        "--components", action="store_true", help="Component renders + drawings only"
    )
    parser.add_argument("--bots", action="store_true", help="Bot builds only")
    parser.add_argument(
        "--rom", action="store_true", help="ROM validation renders only"
    )
    args = parser.parse_args()

    run_all = not (args.components or args.bots or args.rom)

    t0 = time.perf_counter()

    if run_all or args.components:
        print("Component renders + drawings:")
        regen_component_renders()

    if run_all or args.rom:
        print("ROM validation renders:")
        regen_rom_renders()

    if run_all or args.bots:
        print("Bot builds (includes bot-level drawings):")
        regen_bot_outputs()

    elapsed = time.perf_counter() - t0
    print(f"\nDone ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
