"""CAD emitter — generates STEP assembly + per-body STLs using build123d.

Requires: pip install build123d

Creates:
- Per-body solid geometry (structural shells with component pockets)
- Full assembly with RevoluteJoint connections at servo locations
- STEP export (viewable in FreeCAD, Fusion 360)
- Per-body STL export (for MuJoCo visuals + 3D printing)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from botcad.skeleton import Body, Bot


def emit_cad(bot: Bot, output_dir: Path) -> None:
    """Generate STEP assembly and per-body STL files."""
    # Import build123d — raises ImportError if not installed
    from build123d import (
        Compound,
        export_step,
        export_stl,
    )

    meshes_dir = output_dir / "meshes"
    meshes_dir.mkdir(parents=True, exist_ok=True)

    parts: dict[str, object] = {}  # body_name -> Solid

    for body in bot.all_bodies:
        solid = _make_body_solid(body)
        if solid is not None:
            parts[body.name] = solid

            # Export per-body STL (overwrites the primitive ones from mujoco emitter)
            stl_path = meshes_dir / f"{body.name}.stl"
            export_stl(solid, str(stl_path))

    # Build assembly with joints
    if parts:
        assembly = Compound(children=list(parts.values()))
        step_path = output_dir / "assembly.step"
        export_step(assembly, str(step_path))
        print(f"CAD: wrote assembly.step + {len(parts)} STLs to {output_dir}")


def _make_body_solid(body: Body):
    """Create a build123d solid for a body.

    Returns a Solid with component pockets cut out of a structural shell.
    """
    from build123d import Align, Box, Cylinder, Location, Sphere

    dims = body.dimensions
    wall = 0.002  # 2mm wall thickness

    if body.shape == "cylinder":
        r = body.radius or dims[0] / 2
        h = dims[2]
        outer = Cylinder(r, h, align=(Align.CENTER, Align.CENTER, Align.CENTER))
        if r > wall and h > wall * 2:
            inner = Cylinder(
                r - wall, h - wall * 2, align=(Align.CENTER, Align.CENTER, Align.CENTER)
            )
            shell = outer - inner
        else:
            shell = outer

    elif body.shape == "tube":
        r = body.outer_r or dims[0] / 2
        length = body.length or dims[2]
        outer = Cylinder(r, length, align=(Align.CENTER, Align.CENTER, Align.CENTER))
        inner_r = max(r - wall, r * 0.6)
        inner = Cylinder(
            inner_r, length + 0.001, align=(Align.CENTER, Align.CENTER, Align.CENTER)
        )
        shell = outer - inner

    elif body.shape == "sphere":
        r = body.radius or dims[0] / 2
        shell = Sphere(r)

    else:
        # Box shell
        outer = Box(
            dims[0], dims[1], dims[2], align=(Align.CENTER, Align.CENTER, Align.CENTER)
        )
        if all(d > wall * 2 for d in dims):
            inner = Box(
                dims[0] - wall * 2,
                dims[1] - wall * 2,
                dims[2] - wall * 2,
                align=(Align.CENTER, Align.CENTER, Align.CENTER),
            )
            shell = outer - inner
        else:
            shell = outer

    # Cut component pockets
    for mount in body.mounts:
        cd = mount.component.dimensions
        pocket = Box(
            cd[0] + 0.0005,
            cd[1] + 0.0005,
            cd[2] + 0.0005,
            align=(Align.CENTER, Align.CENTER, Align.CENTER),
        )
        pocket = pocket.locate(Location(mount.resolved_pos))
        shell = shell - pocket

    return shell
