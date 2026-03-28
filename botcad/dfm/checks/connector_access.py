"""Connector mating access DFM check.

Detects connector ports on mounted components that lack sufficient
clearance for plug insertion or finger access.  A servo connector
facing a body wall with 2mm clearance is impossible to plug in by hand.

Uses bounding-box analysis (same approach as fastener clearance) — fast,
robust, catches the common case of connectors near body walls.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from botcad.assembly.refs import ComponentRef
from botcad.connectors import connector_spec
from botcad.dfm.check import DFMCheck, DFMFinding, DFMSeverity

if TYPE_CHECKING:
    from botcad.assembly.sequence import AssemblySequence
    from botcad.component import Vec3, WirePort
    from botcad.ids import BodyId
    from botcad.skeleton import Body, Bot, Joint, Mount

# Clearance thresholds (meters)
MATING_AXIAL_CLEARANCE = 0.015  # 15mm along plug insertion direction
LATERAL_FINGER_CLEARANCE = 0.010  # 10mm perpendicular for finger access


class ConnectorMatingAccess(DFMCheck):
    """Check that every connector port has clearance for plug insertion."""

    @property
    def name(self) -> str:
        return "connector_mating_access"

    def run(
        self,
        bot: Bot,
        sequence: AssemblySequence,
        body_solids: dict[BodyId, object],
    ) -> list[DFMFinding]:
        findings: list[DFMFinding] = []

        body_map = _build_body_map(bot)

        # Build a step index: which assembly step installs each component
        step_index = _build_step_index(sequence)

        for body in body_map.values():
            half = _half_extents(body)

            # Check wire ports on mounted components
            for mount in body.mounts:
                for port in mount.component.wire_ports:
                    finding = _check_mount_port(
                        body=body,
                        mount=mount,
                        port=port,
                        half_extents=half,
                        step_index=step_index,
                    )
                    if finding is not None:
                        findings.append(finding)

            # Check wire ports on servos at joints
            for joint in body.joints:
                for port in joint.servo.wire_ports:
                    finding = _check_servo_port(
                        body=body,
                        joint=joint,
                        port=port,
                        half_extents=half,
                        step_index=step_index,
                    )
                    if finding is not None:
                        findings.append(finding)

        return findings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_body_map(bot: Bot) -> dict[BodyId, Body]:
    """Walk the kinematic tree and collect all bodies by id."""
    result: dict[BodyId, Body] = {}
    if bot.root is None:
        return result

    queue: deque[Body] = deque([bot.root])
    while queue:
        body = queue.popleft()
        if body.name in result:
            continue
        result[body.name] = body
        for joint in body.joints:
            if joint.child is not None:
                queue.append(joint.child)
    return result


def _half_extents(body: Body) -> tuple[float, float, float]:
    dims = body.dimensions
    return (dims[0] / 2, dims[1] / 2, dims[2] / 2)


def _build_step_index(sequence: AssemblySequence) -> dict[tuple[BodyId, str], int]:
    """Map (body_id, mount_label) -> assembly step for INSERT ops."""
    from botcad.assembly.sequence import AssemblyAction

    index: dict[tuple[BodyId, str], int] = {}
    for op in sequence.ops:
        if op.action == AssemblyAction.INSERT and isinstance(op.target, ComponentRef):
            index[(op.target.body, op.target.mount_label)] = op.step
    return index


def _port_pos_in_body_frame(
    mount: Mount,
    port: WirePort,
) -> Vec3:
    """Transform a wire port position from component-local to body frame."""
    rotated = mount.rotate_point(port.pos)
    rp = mount.resolved_pos
    return (
        rp[0] + rotated[0],
        rp[1] + rotated[1],
        rp[2] + rotated[2],
    )


def _servo_port_pos_in_body_frame(
    joint: Joint,
    port: WirePort,
) -> Vec3:
    """Transform a servo wire port position into the parent body frame."""
    center = joint.solved_servo_center
    quat = joint.solved_servo_quat
    rotated = _quat_rotate(quat, port.pos)
    return (
        center[0] + rotated[0],
        center[1] + rotated[1],
        center[2] + rotated[2],
    )


def _mating_dir_in_body_frame(
    mount: Mount,
    mating_direction: Vec3,
) -> Vec3:
    """Rotate the mating direction from component-local to body frame."""
    return mount.rotate_point(mating_direction)


def _servo_mating_dir_in_body_frame(
    joint: Joint,
    mating_direction: Vec3,
) -> Vec3:
    """Rotate the mating direction from servo-local to body frame."""
    return _quat_rotate(joint.solved_servo_quat, mating_direction)


def _quat_rotate(
    q: tuple[float, float, float, float],
    v: Vec3,
) -> Vec3:
    """Rotate vector v by quaternion q = (w, x, y, z)."""
    w, qx, qy, qz = q
    vx, vy, vz = v
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + w * tx + (qy * tz - qz * ty),
        vy + w * ty + (qz * tx - qx * tz),
        vz + w * tz + (qx * ty - qy * tx),
    )


def _check_clearance_at(
    *,
    connector_pos: Vec3,
    mating_dir: Vec3,
    half_extents: tuple[float, float, float],
) -> tuple[float, float]:
    """Compute axial and lateral clearance for a connector.

    Returns (axial_clearance, lateral_clearance) in meters.
    Axial = distance along mating direction to nearest wall.
    Lateral = minimum distance to walls perpendicular to mating direction.
    """
    fx, fy, fz = connector_pos
    hx, hy, hz = half_extents
    mx, my, mz = mating_dir

    # Distances from connector to each face of the bounding box
    wall_distances = [
        (hx - fx, (1, 0, 0)),  # +X wall
        (hx + fx, (-1, 0, 0)),  # -X wall
        (hy - fy, (0, 1, 0)),  # +Y wall
        (hy + fy, (0, -1, 0)),  # -Y wall
        (hz - fz, (0, 0, 1)),  # +Z wall
        (hz + fz, (0, 0, -1)),  # -Z wall
    ]

    # Axial clearance: distance to the wall the plug mates towards.
    # The mating direction points from the port towards the plug — we need
    # clearance in that direction for the plug body + fingers.
    min_axial = float("inf")
    for dist, wall_normal in wall_distances:
        # Wall normal aligned with mating direction means the plug
        # approaches this wall.  dot > 0.5 means roughly aligned.
        dot = wall_normal[0] * mx + wall_normal[1] * my + wall_normal[2] * mz
        if dot > 0.5:
            min_axial = min(min_axial, dist)

    # Lateral clearance: distance to walls perpendicular to mating direction
    min_lateral = float("inf")
    for dist, wall_normal in wall_distances:
        dot = abs(wall_normal[0] * mx + wall_normal[1] * my + wall_normal[2] * mz)
        if dot < 0.5:
            min_lateral = min(min_lateral, dist)

    return min_axial, min_lateral


def _make_finding(
    *,
    body: Body,
    target: ComponentRef,
    assembly_step: int,
    port: WirePort,
    connector_label: str,
    pos: Vec3,
    mating_dir: Vec3,
    axial: float,
    lateral: float,
) -> DFMFinding | None:
    """Create a DFMFinding if clearance is insufficient, else None."""
    severity = None
    title = ""
    description = ""
    measured: float | None = None
    threshold: float | None = None

    if axial < MATING_AXIAL_CLEARANCE:
        measured = axial * 1000  # mm
        threshold = MATING_AXIAL_CLEARANCE * 1000
        if axial < MATING_AXIAL_CLEARANCE * 0.5:
            severity = DFMSeverity.ERROR
            title = f"Blocked connector: {connector_label} on {port.label}"
            description = (
                f"Axial clearance {measured:.1f}mm < {threshold:.1f}mm "
                f"(need {threshold:.1f}mm for plug insertion). "
                f"Cannot insert plug into {connector_label}."
            )
        else:
            severity = DFMSeverity.WARNING
            title = f"Tight connector access: {connector_label} on {port.label}"
            description = (
                f"Axial clearance {measured:.1f}mm < {threshold:.1f}mm "
                f"(need {threshold:.1f}mm for plug insertion). "
                f"Plug insertion may be difficult."
            )
    elif lateral < LATERAL_FINGER_CLEARANCE:
        severity = DFMSeverity.WARNING
        measured = lateral * 1000
        threshold = LATERAL_FINGER_CLEARANCE * 1000
        title = f"Tight finger access: {connector_label} on {port.label}"
        description = (
            f"Lateral clearance {measured:.1f}mm < {threshold:.1f}mm. "
            f"Fingers may not fit around {connector_label} to grip the plug."
        )

    if severity is None:
        return None

    return DFMFinding(
        check_name="connector_mating_access",
        severity=severity,
        body=body.name,
        target=target,
        assembly_step=assembly_step,
        title=title,
        description=description,
        pos=pos,
        direction=mating_dir,
        measured=measured,
        threshold=threshold,
        has_overlay=False,
    )


def _check_mount_port(
    *,
    body: Body,
    mount: Mount,
    port: WirePort,
    half_extents: tuple[float, float, float],
    step_index: dict[tuple[BodyId, str], int],
) -> DFMFinding | None:
    """Check a single wire port on a mounted component."""
    if port.permanent or not port.connector_type:
        return None

    try:
        spec = connector_spec(port.connector_type)
    except KeyError:
        return None

    pos = _port_pos_in_body_frame(mount, port)
    mating_dir = _mating_dir_in_body_frame(mount, spec.mating_direction)
    axial, lateral = _check_clearance_at(
        connector_pos=pos,
        mating_dir=mating_dir,
        half_extents=half_extents,
    )

    assembly_step = step_index.get((body.name, mount.label), 0)
    target = ComponentRef(body=body.name, mount_label=mount.label)

    return _make_finding(
        body=body,
        target=target,
        assembly_step=assembly_step,
        port=port,
        connector_label=spec.label,
        pos=pos,
        mating_dir=mating_dir,
        axial=axial,
        lateral=lateral,
    )


def _check_servo_port(
    *,
    body: Body,
    joint: Joint,
    port: WirePort,
    half_extents: tuple[float, float, float],
    step_index: dict[tuple[BodyId, str], int],
) -> DFMFinding | None:
    """Check a single wire port on a servo at a joint."""
    if port.permanent or not port.connector_type:
        return None

    try:
        spec = connector_spec(port.connector_type)
    except KeyError:
        return None

    pos = _servo_port_pos_in_body_frame(joint, port)
    mating_dir = _servo_mating_dir_in_body_frame(joint, spec.mating_direction)
    axial, lateral = _check_clearance_at(
        connector_pos=pos,
        mating_dir=mating_dir,
        half_extents=half_extents,
    )

    mount_label = f"servo_{joint.name}"
    assembly_step = step_index.get((body.name, mount_label), 0)
    target = ComponentRef(body=body.name, mount_label=mount_label)

    return _make_finding(
        body=body,
        target=target,
        assembly_step=assembly_step,
        port=port,
        connector_label=spec.label,
        pos=pos,
        mating_dir=mating_dir,
        axial=axial,
        lateral=lateral,
    )
