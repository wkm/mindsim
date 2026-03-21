"""Clearance validation -- compute minimum distances between body solids.

Two-step approach:
1. Boolean intersection (a & b) detects overlapping/contained bodies
2. BRepExtrema_DistShapeShape measures gap for non-intersecting bodies

Intersection volume > 0 → negative distance (penetration).
No intersection → BRepExtrema distance (positive gap).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from botcad.skeleton import Bot


@dataclass
class ClearanceResult:
    """Result of checking one clearance constraint against actual geometry."""

    body_a: str
    body_b: str
    distance: float  # meters -- negative means intersection
    min_distance: float  # meters -- required minimum gap
    label: str
    satisfied: bool  # distance >= min_distance
    intersection_volume: float = 0.0  # m³ -- volume of overlap (0 = no overlap)


def _compute_distance(placed_a, placed_b) -> tuple[float, float]:
    """Compute the signed distance between two positioned solids.

    Returns (distance, intersection_volume):
    - distance > 0: gap between surfaces
    - distance == 0: surfaces touching
    - distance < 0: bodies overlap (magnitude is approximate penetration)
    - intersection_volume: volume of the overlapping region (0 if no overlap)
    """
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape

    # Step 1: Check for intersection via boolean common
    try:
        common = placed_a & placed_b
        solids = common.solids() if hasattr(common, "solids") else []
        int_vol = sum(abs(s.volume) for s in solids) if solids else 0.0
        if hasattr(common, "volume") and not solids:
            int_vol = abs(common.volume) if common.volume else 0.0
    except Exception:
        int_vol = 0.0

    if int_vol > 1e-15:
        # Bodies intersect — report negative distance proportional to overlap
        # Approximate penetration depth from intersection volume
        # (cube root gives a length scale)
        penetration = -(int_vol ** (1.0 / 3.0))
        return penetration, int_vol

    # Step 2: No intersection — measure gap via BRepExtrema
    dist_calc = BRepExtrema_DistShapeShape(placed_a.wrapped, placed_b.wrapped)
    if dist_calc.IsDone() and dist_calc.NbSolution() > 0:
        return dist_calc.Value(), 0.0

    return float("inf"), 0.0


def validate_clearances(bot: Bot, body_solids: dict) -> list[ClearanceResult]:
    """Check all clearance constraints against actual geometry.

    Args:
        bot: Solved bot with clearance constraints and world transforms.
        body_solids: Mapping of body name to build123d Solid (local frame).

    Returns:
        List of ClearanceResult for each constraint that could be evaluated.
    """
    from build123d import Location

    from botcad.geometry import quat_to_euler

    results: list[ClearanceResult] = []

    for constraint in bot._clearance_constraints:
        solid_a = body_solids.get(constraint.body_a)
        solid_b = body_solids.get(constraint.body_b)

        if solid_a is None or solid_b is None:
            continue  # body not built

        body_a = next((b for b in bot.all_bodies if b.name == constraint.body_a), None)
        body_b = next((b for b in bot.all_bodies if b.name == constraint.body_b), None)
        if not body_a or not body_b:
            continue

        # Position solids in world frame
        placed_a = solid_a.moved(
            Location(body_a.world_pos, quat_to_euler(body_a.world_quat))
        )
        placed_b = solid_b.moved(
            Location(body_b.world_pos, quat_to_euler(body_b.world_quat))
        )

        distance, int_vol = _compute_distance(placed_a, placed_b)

        results.append(
            ClearanceResult(
                body_a=constraint.body_a,
                body_b=constraint.body_b,
                distance=distance,
                min_distance=constraint.min_distance,
                label=constraint.label,
                satisfied=distance >= constraint.min_distance,
                intersection_volume=int_vol,
            )
        )

    return results
