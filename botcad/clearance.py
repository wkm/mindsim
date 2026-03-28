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

from botcad.ids import BodyId

if TYPE_CHECKING:
    from botcad.skeleton import Bot


@dataclass(frozen=True)
class ClearanceResult:
    """Result of checking one clearance constraint against actual geometry."""

    body_a: BodyId
    body_b: BodyId
    intersects: bool  # do the bodies overlap?
    intersection_volume: float  # m³ -- volume of overlap (0 = no overlap)
    distance: float  # meters -- surface-to-surface gap (0 if intersecting)
    min_distance: float  # meters -- required minimum gap
    label: str

    @property
    def satisfied(self) -> bool:
        if self.intersects:
            return False  # any intersection is a violation
        return self.distance >= self.min_distance


def _check_clearance(placed_a, placed_b) -> tuple[bool, float, float]:
    """Check clearance between two positioned solids.

    Returns (intersects, intersection_volume, distance):
    - intersects: True if bodies overlap
    - intersection_volume: m³ of overlap (0 if no overlap)
    - distance: surface-to-surface gap in meters (0 if intersecting, >= 0 always)
    """
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape

    # Step 1: Check for intersection via boolean common
    int_vol = 0.0
    try:
        common = placed_a & placed_b
        solids = common.solids() if hasattr(common, "solids") else []
        int_vol = sum(abs(s.volume) for s in solids) if solids else 0.0
        if hasattr(common, "volume") and not solids:
            int_vol = abs(common.volume) if common.volume else 0.0
    except Exception as e:
        import logging

        logging.getLogger(__name__).warning("Boolean intersection failed: %s", e)

    if int_vol > 1e-15:
        return True, int_vol, 0.0

    # Step 2: No intersection — measure gap via BRepExtrema
    dist_calc = BRepExtrema_DistShapeShape(placed_a.wrapped, placed_b.wrapped)
    if dist_calc.IsDone() and dist_calc.NbSolution() > 0:
        return False, 0.0, dist_calc.Value()

    return False, 0.0, float("inf")


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
            Location(body_a.world_pose.pos, quat_to_euler(body_a.world_pose.quat))
        )
        placed_b = solid_b.moved(
            Location(body_b.world_pose.pos, quat_to_euler(body_b.world_pose.quat))
        )

        intersects, int_vol, distance = _check_clearance(placed_a, placed_b)

        results.append(
            ClearanceResult(
                body_a=constraint.body_a,
                body_b=constraint.body_b,
                intersects=intersects,
                intersection_volume=int_vol,
                distance=distance,
                min_distance=constraint.min_distance,
                label=constraint.label,
            )
        )

    return results
