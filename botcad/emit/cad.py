"""CAD emitter — generates STEP assembly + per-body STLs using build123d.

Requires: pip install build123d

Creates:
- Per-body solid geometry (structural shells with component pockets)
- Full assembly with RevoluteJoint connections at servo locations
- STEP export (viewable in FreeCAD, Fusion 360)
- Per-body STL export (for MuJoCo visuals + 3D printing)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

from botcad.geometry import rotate_vec, servo_placement

if TYPE_CHECKING:
    from botcad.geometry import Quat
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

    # Cut servo pockets + add mounting standoffs at each joint
    tolerance = 0.001  # 1mm clearance around servo
    standoff_wall = 0.001  # 1mm wall around screw hole
    standoff_height = 0.003  # 3mm tall standoffs

    for joint in body.joints:
        servo = joint.servo
        center, quat = servo_placement(
            servo.shaft_offset, servo.shaft_axis, joint.axis, joint.pos
        )
        euler = _quat_to_euler(quat)
        sd = servo.dimensions

        # Cut servo pocket (servo dims + tolerance)
        pocket = Box(
            sd[0] + tolerance,
            sd[1] + tolerance,
            sd[2] + tolerance,
            align=(Align.CENTER, Align.CENTER, Align.CENTER),
        )
        pocket = pocket.locate(Location(center, euler))
        shell = shell - pocket

        # Add mounting standoffs at each screw hole
        for mp in servo.mounting_points:
            # Rotate mounting point position into parent body frame
            rotated_mp = rotate_vec(quat, mp.pos)
            standoff_pos = (
                center[0] + rotated_mp[0],
                center[1] + rotated_mp[1],
                center[2] + rotated_mp[2],
            )

            standoff_od = mp.diameter + standoff_wall * 2
            standoff = Cylinder(
                standoff_od / 2,
                standoff_height,
                align=(Align.CENTER, Align.CENTER, Align.CENTER),
            )
            standoff = standoff.locate(Location(standoff_pos, euler))
            shell = shell + standoff

            # Cut through-hole for screw
            hole = Cylinder(
                mp.diameter / 2,
                standoff_height + 0.002,
                align=(Align.CENTER, Align.CENTER, Align.CENTER),
            )
            hole = hole.locate(Location(standoff_pos, euler))
            shell = shell - hole

    return shell


def _quat_to_euler(q: Quat) -> tuple[float, float, float]:
    """Convert quaternion (w, x, y, z) to Euler angles (rx, ry, rz) in degrees.

    Uses intrinsic XYZ convention matching build123d's Location(pos, euler).
    """
    w, x, y, z = q

    # Roll (X)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    rx = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (Y)
    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    ry = math.asin(sinp)

    # Yaw (Z)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    rz = math.atan2(siny_cosp, cosy_cosp)

    return (math.degrees(rx), math.degrees(ry), math.degrees(rz))
