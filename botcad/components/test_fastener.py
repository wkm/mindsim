"""Test fastener component — stress-tests fastener placement at various angles and sizes."""

from __future__ import annotations

import math

from botcad.component import Component, MountPoint
from botcad.materials import MAT_BRASS

# Pre-computed unit vectors for angled fasteners
_INV_SQRT2 = math.sqrt(2) / 2  # ~0.7071


def TestFastenerPrism() -> Component:
    """A synthetic prism with mounting holes at varied angles and sizes.

    Exercises the fastener rotation pipeline across cardinal axes,
    diagonal angles, and multiple screw diameters. Not a real part —
    exists purely for visual validation that screws orient correctly.

    Layout (viewed from +Z, prism extends along X):
        Row 1 (Y=+10mm): Cardinal axes — +Z, -Z, +X, -X, +Y, -Y
        Row 2 (Y=-10mm): Diagonals — 45° in XZ, YZ, XY planes
    """
    # 60x30x15 mm prism.  Half-extents for placing points on faces.
    w, d, h = 0.060, 0.030, 0.015
    hw, hd, hh = w / 2, d / 2, h / 2  # 30, 15, 7.5 mm

    return Component(
        name="TestFastenerPrism",
        dimensions=(w, d, h),
        mass=0.050,
        mounting_points=(
            # --- Cardinal axes, M2: each point sits on the face it points out of ---
            # Top face (Z = +7.5mm)
            MountPoint(
                "up",
                pos=(-0.015, 0.005, hh),
                diameter=0.002,
                axis=(0.0, 0.0, 1.0),
                fastener_type="M2",
            ),
            # Bottom face (Z = -7.5mm)
            MountPoint(
                "down",
                pos=(-0.005, 0.005, -hh),
                diameter=0.002,
                axis=(0.0, 0.0, -1.0),
                fastener_type="M2",
            ),
            # Right face (X = +30mm)
            MountPoint(
                "right",
                pos=(hw, 0.005, 0.0),
                diameter=0.002,
                axis=(1.0, 0.0, 0.0),
                fastener_type="M2",
            ),
            # Left face (X = -30mm)
            MountPoint(
                "left",
                pos=(-hw, -0.005, 0.0),
                diameter=0.002,
                axis=(-1.0, 0.0, 0.0),
                fastener_type="M2",
            ),
            # Front face (Y = +15mm)
            MountPoint(
                "front",
                pos=(0.010, hd, 0.0),
                diameter=0.002,
                axis=(0.0, 1.0, 0.0),
                fastener_type="M2",
            ),
            # Back face (Y = -15mm)
            MountPoint(
                "back",
                pos=(0.010, -hd, 0.0),
                diameter=0.002,
                axis=(0.0, -1.0, 0.0),
                fastener_type="M2",
            ),
            # --- 45° diagonals, M2.5: on top face, angled outward ---
            MountPoint(
                "diag_xz",
                pos=(0.015, 0.005, hh),
                diameter=0.0025,
                axis=(_INV_SQRT2, 0.0, _INV_SQRT2),
                fastener_type="M2.5",
            ),
            MountPoint(
                "diag_yz",
                pos=(0.005, 0.005, hh),
                diameter=0.0025,
                axis=(0.0, _INV_SQRT2, _INV_SQRT2),
                fastener_type="M2.5",
            ),
            MountPoint(
                "diag_xy",
                pos=(hw, hd, 0.0),
                diameter=0.0025,
                axis=(_INV_SQRT2, _INV_SQRT2, 0.0),
                fastener_type="M2.5",
            ),
            MountPoint(
                "diag_neg_xz",
                pos=(-0.015, -0.005, hh),
                diameter=0.0025,
                axis=(-_INV_SQRT2, 0.0, _INV_SQRT2),
                fastener_type="M2.5",
            ),
            # --- M3 at 30° from vertical, on top face ---
            MountPoint(
                "steep_30",
                pos=(-0.005, -0.005, hh),
                diameter=0.003,
                axis=(math.sin(math.radians(30)), 0.0, math.cos(math.radians(30))),
                fastener_type="M3",
            ),
        ),
        default_material=MAT_BRASS,
    )
