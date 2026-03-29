"""Wire bend radius DFM check.

Detects wire route junctions where the bend is too tight for the cable
to be pushed through during assembly. Wire routes are piecewise-linear
(list of WireSegments with start/end points); curvature occurs at
segment junctions where consecutive segment directions change.

Effective bend radius at a junction:
    radius = min(seg1_length, seg2_length) / (2 * sin(angle / 2))

Thresholds:
    - Static segment (no joint crossing): min radius = 5 x cable OD
    - Joint-crossing segment (joint_name set): min radius = 10 x cable OD
    - Default cable OD: 1.5mm (AWG 26 servo wire)
"""

from __future__ import annotations

import math

from botcad.assembly.refs import WireRef
from botcad.assembly.sequence import AssemblySequence
from botcad.dfm.check import DFMCheck, DFMFinding, DFMSeverity
from botcad.dfm.utils import build_wire_steps
from botcad.routing import WireRoute, WireSegment, solve_routing
from botcad.skeleton import Bot

# Default cable outer diameter (meters) — AWG 26 servo wire
_DEFAULT_CABLE_OD: float = 0.0015  # 1.5mm

# Multipliers for minimum bend radius
_STATIC_MULTIPLIER: float = 5.0
_DYNAMIC_MULTIPLIER: float = 10.0

# Minimum angle (radians) to consider a bend worth checking.
# Below this the segments are nearly collinear and bend radius is huge.
_MIN_ANGLE_RAD: float = 0.01  # ~0.6 degrees


def _compute_bend_radius(angle_rad: float, seg_length: float) -> float:
    """Compute effective bend radius at a junction.

    Parameters
    ----------
    angle_rad : float
        Angle between consecutive segment directions (0 = collinear,
        pi = U-turn). This is the deflection angle, NOT the interior angle.
    seg_length : float
        The shorter of the two adjacent segment lengths (meters).

    Returns
    -------
    float
        Effective bend radius in meters.
    """
    half_angle = angle_rad / 2.0
    sin_half = math.sin(half_angle)
    if sin_half < 1e-9:
        return float("inf")
    return seg_length / (2.0 * sin_half)


def _segment_direction(seg: WireSegment) -> tuple[float, float, float] | None:
    """Unit direction vector from start to end. None if degenerate."""
    dx = seg.end[0] - seg.start[0]
    dy = seg.end[1] - seg.start[1]
    dz = seg.end[2] - seg.start[2]
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if length < 1e-9:
        return None
    return (dx / length, dy / length, dz / length)


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


class WireBendRadius(DFMCheck):
    """Check that wire route bends are not too tight for cable routing."""

    @property
    def name(self) -> str:
        return "wire_bend_radius"

    def run(
        self,
        bot: Bot,
        sequence: AssemblySequence,
    ) -> list[DFMFinding]:
        findings: list[DFMFinding] = []

        if bot.root is None:
            return findings

        routes = solve_routing(bot)
        wire_steps = build_wire_steps(sequence)

        for route in routes:
            findings.extend(self._check_route(route, wire_steps.get(route.label, 0)))

        return findings

    def _check_route(self, route: WireRoute, step: int) -> list[DFMFinding]:
        findings: list[DFMFinding] = []
        segments = route.segments

        if len(segments) < 2:
            return findings

        for i in range(len(segments) - 1):
            seg_a = segments[i]
            seg_b = segments[i + 1]

            # Only check junctions within the same body — cross-body
            # transitions have a physical flex zone, not a printed bend.
            if seg_a.body_name != seg_b.body_name:
                continue

            dir_a = _segment_direction(seg_a)
            dir_b = _segment_direction(seg_b)
            if dir_a is None or dir_b is None:
                continue

            # Angle between directions (deflection angle)
            cos_angle = max(-1.0, min(1.0, _dot(dir_a, dir_b)))
            angle_rad = math.acos(cos_angle)

            if angle_rad < _MIN_ANGLE_RAD:
                continue

            shorter_seg = min(seg_a.straight_length, seg_b.straight_length)
            radius = _compute_bend_radius(angle_rad, shorter_seg)

            # Determine threshold based on whether either segment
            # crosses a joint (dynamic) or is static
            is_dynamic = seg_a.joint_name is not None or seg_b.joint_name is not None
            multiplier = _DYNAMIC_MULTIPLIER if is_dynamic else _STATIC_MULTIPLIER
            min_radius = multiplier * _DEFAULT_CABLE_OD

            if radius >= min_radius:
                continue

            # Junction position is the end of seg_a / start of seg_b
            junction_pos = seg_a.end

            angle_deg = math.degrees(angle_rad)
            severity = (
                DFMSeverity.ERROR if radius < min_radius * 0.5 else DFMSeverity.WARNING
            )

            kind = "dynamic (joint-crossing)" if is_dynamic else "static"
            findings.append(
                DFMFinding(
                    check_name=self.name,
                    severity=severity,
                    body=seg_a.body_name,
                    target=WireRef(label=route.label),
                    assembly_step=step,
                    title=(f"Wire bend too tight on {route.label} ({kind})"),
                    description=(
                        f"Bend angle {angle_deg:.0f} deg with "
                        f"segment length {shorter_seg * 1000:.1f}mm "
                        f"gives effective radius {radius * 1000:.1f}mm, "
                        f"below {kind} minimum of "
                        f"{min_radius * 1000:.1f}mm "
                        f"({multiplier:.0f}x cable OD). "
                        f"Segments {i} -> {i + 1} in route '{route.label}'."
                    ),
                    pos=junction_pos,
                    direction=dir_b,
                    measured=radius,
                    threshold=min_radius,
                    has_overlay=False,
                )
            )

        return findings
