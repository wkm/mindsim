"""Fastener tool clearance DFM check.

Detects fasteners that cannot be reached by the required tool — e.g.
servo mounting ear screws trapped between the servo body and a nearby
body wall with insufficient space for a hex key.

Uses bounding-box analysis (not full ray-casting) for speed and
robustness.  This catches the common case where fasteners are near
or beyond body walls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from botcad.assembly.refs import FastenerRef
from botcad.assembly.sequence import AssemblyAction, AssemblyOp
from botcad.assembly.tools import TOOL_LIBRARY
from botcad.dfm.check import DFMCheck, DFMFinding, DFMSeverity
from botcad.dfm.utils import build_body_map

if TYPE_CHECKING:
    from botcad.assembly.sequence import AssemblySequence
    from botcad.component import MountPoint, Vec3
    from botcad.skeleton import Body, Bot, Joint


class FastenerToolClearance(DFMCheck):
    """Check that every fastener has sufficient clearance for the required tool."""

    @property
    def name(self) -> str:
        return "fastener_tool_clearance"

    def run(
        self,
        bot: Bot,
        sequence: AssemblySequence,
    ) -> list[DFMFinding]:
        findings: list[DFMFinding] = []

        # Index bodies by name for lookup
        body_map = build_body_map(bot)

        for op in sequence.ops:
            if op.action != AssemblyAction.FASTEN:
                continue
            if not isinstance(op.target, FastenerRef):
                continue

            body = body_map.get(op.target.body)
            if body is None:
                continue

            # Find which mounting point this fastener corresponds to
            mp_info = _resolve_fastener_to_mount_point(body, op.target.index)
            if mp_info is None:
                continue

            joint, mount_point, fastener_pos_body = mp_info

            # Get the body's bounding box half-extents
            dims = body.dimensions
            half = (dims[0] / 2, dims[1] / 2, dims[2] / 2)

            tool_spec = TOOL_LIBRARY.get(op.tool) if op.tool else None
            if tool_spec is None:
                continue

            approach = op.approach_axis
            if approach is None:
                continue

            # Check axial clearance (distance from fastener to body surface
            # in the approach direction) and lateral clearance (distance
            # from fastener to nearest wall perpendicular to approach)
            finding = _check_clearance(
                op=op,
                body=body,
                joint=joint,
                mount_point=mount_point,
                fastener_pos=fastener_pos_body,
                approach=approach,
                tool_spec=tool_spec,
                half_extents=half,
            )
            if finding is not None:
                findings.append(finding)

        return findings


def _resolve_fastener_to_mount_point(
    body: Body,
    fastener_index: int,
) -> tuple[Joint | None, MountPoint, Vec3] | None:
    """Map a fastener index to its MountPoint and body-frame position.

    Returns (joint_or_none, mount_point, pos_in_body_frame) or None.

    The fastener index matches the order in build_assembly_sequence:
    first all servo mounting ears (across joints), then all component
    mounting points (across mounts).
    """
    idx = 0

    # Servo mounting ears first (matches build order)
    for joint in body.joints:
        for mp in joint.servo.mounting_ears:
            if idx == fastener_index:
                pos = _ear_pos_in_body_frame(joint, mp)
                return joint, mp, pos
            idx += 1

    # Component mounting points
    for mount in body.mounts:
        for mp in mount.component.mounting_points:
            if idx == fastener_index:
                # Transform mount point position to body frame
                rotated = mount.rotate_point(mp.pos)
                pos = (
                    mount.resolved_pos[0] + rotated[0],
                    mount.resolved_pos[1] + rotated[1],
                    mount.resolved_pos[2] + rotated[2],
                )
                return None, mp, pos
            idx += 1

    return None


def _ear_pos_in_body_frame(
    joint: Joint,
    ear: MountPoint,
) -> Vec3:
    """Transform a servo mounting ear position into the parent body frame.

    The ear position is in servo-local frame.  The servo is placed at the
    joint's solved position and orientation within the parent body.
    """
    # Use solved servo center if available (computed by packing solver),
    # otherwise fall back to the joint position.
    center = joint.solved_servo_center
    quat = joint.solved_servo_quat

    # Rotate ear position by the servo's orientation quaternion
    rotated = _quat_rotate(quat, ear.pos)

    return (
        center[0] + rotated[0],
        center[1] + rotated[1],
        center[2] + rotated[2],
    )


def _quat_rotate(
    q: tuple[float, float, float, float],
    v: Vec3,
) -> Vec3:
    """Rotate vector v by quaternion q = (w, x, y, z)."""
    w, qx, qy, qz = q
    vx, vy, vz = v

    # q * v * q^-1 via the efficient formula
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)

    return (
        vx + w * tx + (qy * tz - qz * ty),
        vy + w * ty + (qz * tx - qx * tz),
        vz + w * tz + (qx * ty - qy * tx),
    )


def _check_clearance(
    *,
    op: AssemblyOp,
    body: Body,
    joint: Joint | None,
    mount_point: MountPoint,
    fastener_pos: Vec3,
    approach: tuple[float, float, float],
    tool_spec: object,
    half_extents: tuple[float, float, float],
) -> DFMFinding | None:
    """Check axial and lateral clearance for a single fastener.

    Returns a DFMFinding if clearance is insufficient, else None.

    Clearance model:
    - **Axial:** distance from the fastener head along the approach axis
      to the nearest body surface.  The tool shaft must fit in this gap.
    - **Lateral:** distance from the fastener to the nearest body wall
      perpendicular to the approach axis.  The tool head / grip must fit.

    For servo mounting ears the critical case is lateral clearance: the
    ear screws sit between the servo and the body wall with very little
    room for the hex key.
    """
    from botcad.assembly.tools import ToolSpec

    assert isinstance(tool_spec, ToolSpec)

    fx, fy, fz = fastener_pos
    hx, hy, hz = half_extents
    ax, ay, az = approach

    # --- Lateral clearance ---
    # Find the minimum distance from the fastener to the nearest body wall
    # in directions perpendicular to the approach axis.
    # We check all 6 walls and find the minimum distance to any wall
    # that is roughly perpendicular to the approach.

    # Distances from fastener to each face of the bounding box
    wall_distances = [
        (hx - fx, (1, 0, 0)),  # +X wall
        (hx + fx, (-1, 0, 0)),  # -X wall
        (hy - fy, (0, 1, 0)),  # +Y wall
        (hy + fy, (0, -1, 0)),  # -Y wall
        (hz - fz, (0, 0, 1)),  # +Z wall
        (hz + fz, (0, 0, -1)),  # -Z wall
    ]

    # Lateral clearance: minimum distance to walls perpendicular to approach
    min_lateral = float("inf")
    for dist, wall_normal in wall_distances:
        # How aligned is this wall normal with the approach axis?
        dot = abs(wall_normal[0] * ax + wall_normal[1] * ay + wall_normal[2] * az)
        if dot < 0.5:
            # Wall is roughly perpendicular to approach — lateral clearance
            min_lateral = min(min_lateral, dist)

    # Axial clearance: distance to walls along approach direction
    min_axial = float("inf")
    for dist, wall_normal in wall_distances:
        dot = wall_normal[0] * ax + wall_normal[1] * ay + wall_normal[2] * az
        if dot < -0.5:
            # Wall that the tool approaches from (dot < 0 means the wall
            # normal points opposite to approach — the tool comes through
            # this wall).
            min_axial = min(min_axial, dist)

    # Also account for the servo body itself blocking lateral access.
    # If this is a servo ear fastener, compute distance from fastener
    # to the servo body edge.
    servo_lateral_clearance = float("inf")
    if joint is not None:
        servo_lateral_clearance = _servo_body_clearance(joint, fastener_pos, approach)

    effective_lateral = min(min_lateral, servo_lateral_clearance)

    # --- Check tool fit ---
    tool_head_radius = tool_spec.head_diameter / 2
    tool_shaft_length = tool_spec.shaft_length
    grip_clearance = tool_spec.grip_clearance

    # Determine severity
    # Lateral: the tool head must fit, and ideally fingers/grip too
    lateral_needed = tool_head_radius  # minimum: tool tip must fit
    grip_needed = grip_clearance  # ideal: hand/fingers must fit too

    # Axial: tool shaft must reach the fastener
    # If the fastener is outside the bounding box on the approach side,
    # axial clearance is effectively infinite.

    severity = None
    title = ""
    description = ""
    measured = None
    threshold = None

    if effective_lateral < lateral_needed:
        severity = DFMSeverity.ERROR
        measured = effective_lateral * 1000  # mm
        threshold = lateral_needed * 1000
        title = f"No tool access: {mount_point.label}"
        description = (
            f"Lateral clearance {measured:.1f}mm < tool head radius "
            f"{threshold:.1f}mm ({tool_spec.kind.value}). "
            f"The tool cannot physically reach this fastener."
        )
    elif effective_lateral < grip_needed:
        severity = DFMSeverity.WARNING
        measured = effective_lateral * 1000
        threshold = grip_needed * 1000
        title = f"Tight tool access: {mount_point.label}"
        description = (
            f"Lateral clearance {measured:.1f}mm < grip clearance "
            f"{threshold:.1f}mm ({tool_spec.kind.value}). "
            f"The tool may fit but fingers cannot grip it properly."
        )
    elif min_axial < tool_shaft_length and min_axial != float("inf"):
        # Fastener is deep inside the body but accessible laterally.
        # Check if the tool shaft is long enough to reach.
        # (This is less common than lateral issues but still worth flagging.)
        severity = DFMSeverity.WARNING
        measured = min_axial * 1000
        threshold = tool_shaft_length * 1000
        title = f"Deep fastener: {mount_point.label}"
        description = (
            f"Axial depth {measured:.1f}mm but tool shaft is "
            f"{threshold:.1f}mm ({tool_spec.kind.value})."
        )

    if severity is None:
        return None

    return DFMFinding(
        check_name="fastener_tool_clearance",
        severity=severity,
        body=op.body,
        target=op.target,
        assembly_step=op.step,
        title=title,
        description=description,
        pos=fastener_pos,
        direction=approach,
        measured=measured,
        threshold=threshold,
        has_overlay=False,
    )


def _servo_body_clearance(
    joint: Joint,
    fastener_pos: Vec3,
    approach: tuple[float, float, float],
) -> float:
    """Compute distance from fastener to the nearest servo body edge.

    The servo body is a rectangular block centered at the solved_servo_center.
    The mounting ears extend below it.  We compute the gap between the
    fastener position and the edge of the servo body in directions
    perpendicular to the approach axis.

    This catches the classic case: ear screws are just a few mm from the
    servo body wall, leaving no room for a hex key shaft.
    """
    center = joint.solved_servo_center
    quat = joint.solved_servo_quat
    servo = joint.servo
    body_dims = servo.effective_body_dims

    # Servo body half-extents in servo-local frame
    half_servo = (body_dims[0] / 2, body_dims[1] / 2, body_dims[2] / 2)

    # Transform fastener position into servo-local frame
    # (inverse rotation of the displacement from servo center)
    dx = fastener_pos[0] - center[0]
    dy = fastener_pos[1] - center[1]
    dz = fastener_pos[2] - center[2]

    inv_quat = (quat[0], -quat[1], -quat[2], -quat[3])
    local = _quat_rotate(inv_quat, (dx, dy, dz))

    # Distance from fastener to each servo body face in local frame
    # Only consider faces perpendicular to the approach axis (after
    # rotating approach into servo-local frame too).
    approach_local = _quat_rotate(inv_quat, approach)

    min_dist = float("inf")
    for axis_idx in range(3):
        # Check if this axis is roughly perpendicular to approach
        if abs(approach_local[axis_idx]) > 0.5:
            continue  # This axis is along approach, skip

        # Distance from fastener to the nearest servo body face along this axis
        dist_pos = half_servo[axis_idx] - local[axis_idx]
        dist_neg = half_servo[axis_idx] + local[axis_idx]

        # The fastener is outside the servo body (ears extend beyond),
        # so one of these should be negative (fastener is beyond the face).
        # The relevant clearance is the distance to the closest face that
        # the fastener is outside of.
        if dist_pos < 0:
            # Fastener is beyond the +face; distance is |dist_pos|
            min_dist = min(min_dist, abs(dist_pos))
        elif dist_neg < 0:
            min_dist = min(min_dist, abs(dist_neg))
        else:
            # Fastener is inside the servo body extent on this axis —
            # clearance is the smaller of the two distances
            min_dist = min(min_dist, dist_pos, dist_neg)

    return min_dist
