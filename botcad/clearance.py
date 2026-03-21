"""Clearance validation -- compute minimum distances between body solids.

Uses OCCT BRepExtrema_DistShapeShape to measure the actual gap between
positioned body solids.  Negative distance means intersection.
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


def validate_clearances(bot: Bot, body_solids: dict) -> list[ClearanceResult]:
    """Check all clearance constraints against actual geometry.

    Uses OCCT BRepExtrema_DistShapeShape to compute minimum distance
    between positioned solids.

    Args:
        bot: Solved bot with clearance constraints and world transforms.
        body_solids: Mapping of body name to build123d Solid (local frame).

    Returns:
        List of ClearanceResult for each constraint that could be evaluated.
    """
    from build123d import Location
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape

    from botcad.geometry import quat_to_euler

    results: list[ClearanceResult] = []

    for constraint in bot._clearance_constraints:
        solid_a = body_solids.get(constraint.body_a)
        solid_b = body_solids.get(constraint.body_b)

        if solid_a is None or solid_b is None:
            continue  # body not built (e.g. purchased parts without solids yet)

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

        # Compute minimum distance via OCCT
        dist_calc = BRepExtrema_DistShapeShape(placed_a.wrapped, placed_b.wrapped)
        if dist_calc.IsDone() and dist_calc.NbSolution() > 0:
            distance = dist_calc.Value()
        else:
            distance = float("inf")  # couldn't compute

        results.append(
            ClearanceResult(
                body_a=constraint.body_a,
                body_b=constraint.body_b,
                distance=distance,
                min_distance=constraint.min_distance,
                label=constraint.label,
                satisfied=distance >= constraint.min_distance,
            )
        )

    return results
