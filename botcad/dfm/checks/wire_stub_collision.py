"""Wire stub collision DFM check.

Detects overlapping connector bodies and wire stubs within the same body.
Wire stubs are the initial cable segments exiting each connector port:
cylinders extending along the wire exit direction.

Uses AABB overlap analysis to flag collisions (ERROR) and tight gaps (WARNING).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING

from botcad.assembly.refs import ComponentRef
from botcad.component import Vec3
from botcad.connectors import (
    WIRE_STUB_LENGTH,
    WIRE_STUB_RADIUS,
    ConnectorSpec,
    connector_spec,
)
from botcad.dfm.check import DFMCheck, DFMFinding, DFMSeverity
from botcad.dfm.utils import build_body_map
from botcad.geometry import rotate_vec
from botcad.units import Meters, Position

if TYPE_CHECKING:
    from botcad.assembly.sequence import AssemblySequence
    from botcad.ids import BodyId
    from botcad.skeleton import Body, Bot, Joint, Mount

# Gap threshold for warnings (meters)
GAP_WARNING_THRESHOLD = Meters(0.002)  # 2mm


@dataclass(frozen=True)
class _StubRecord:
    """A wire stub collected from a body."""

    label: str
    pos: Position  # connector port position in body frame
    stub_base: Position  # pos + rotated wire_exit_offset (where the stub starts)
    exit_dir: Vec3
    spec: ConnectorSpec
    source_ref: ComponentRef
    assembly_step: int


class WireStubCollision(DFMCheck):
    """Check that wire stubs and connector bodies don't collide within a body."""

    @property
    def name(self) -> str:
        return "wire_stub_collision"

    def run(
        self,
        bot: Bot,
        sequence: AssemblySequence,
    ) -> list[DFMFinding]:
        findings: list[DFMFinding] = []
        body_map = build_body_map(bot)
        step_index = _build_step_index(sequence)

        for body in body_map.values():
            stubs = _collect_stubs(body, step_index)
            if len(stubs) < 2:
                continue

            for a, b in combinations(stubs, 2):
                finding = _check_stub_pair(body, a, b)
                if finding is not None:
                    findings.append(finding)

        return findings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_step_index(
    sequence: AssemblySequence,
) -> dict[tuple[BodyId, str], int]:
    """Map (body_id, mount_label) -> assembly step for INSERT ops."""
    from botcad.assembly.sequence import AssemblyAction

    index: dict[tuple[BodyId, str], int] = {}
    for op in sequence.ops:
        if op.action == AssemblyAction.INSERT and isinstance(op.target, ComponentRef):
            index[(op.target.body, op.target.mount_label)] = op.step
    return index


def _port_pos_in_body_frame(
    mount: Mount,
    port_pos: Vec3,
) -> Vec3:
    """Transform a point from component-local to body frame via mount."""
    rotated = mount.rotate_point(port_pos)
    rp = mount.resolved_pos
    return (rp[0] + rotated[0], rp[1] + rotated[1], rp[2] + rotated[2])


def _servo_port_pos_in_body_frame(
    joint: Joint,
    port_pos: Vec3,
) -> Vec3:
    """Transform a point from servo-local to parent body frame."""
    center = joint.solved_servo_center
    quat = joint.solved_servo_quat
    rotated = rotate_vec(quat, port_pos)
    return (center[0] + rotated[0], center[1] + rotated[1], center[2] + rotated[2])


def _collect_stubs(
    body: Body,
    step_index: dict[tuple[BodyId, str], int],
) -> list[_StubRecord]:
    """Collect all wire stubs in body frame for a given body."""
    stubs: list[_StubRecord] = []

    # From mounted components
    for mount in body.mounts:
        for port in mount.component.wire_ports:
            if not port.connector_type:
                continue
            try:
                spec = connector_spec(port.connector_type)
            except KeyError:
                continue

            pos = _port_pos_in_body_frame(mount, port.pos)
            wire_exit_local = spec.wire_exit_direction
            exit_dir = mount.rotate_point(wire_exit_local)
            rotated_offset = mount.rotate_point(spec.wire_exit_offset)
            stub_base = (
                pos[0] + rotated_offset[0],
                pos[1] + rotated_offset[1],
                pos[2] + rotated_offset[2],
            )
            ref = ComponentRef(body=body.name, mount_label=mount.label)
            step = step_index.get((body.name, mount.label), 0)

            stubs.append(
                _StubRecord(
                    label=f"{mount.label}:{port.label}",
                    pos=pos,
                    stub_base=stub_base,
                    exit_dir=exit_dir,
                    spec=spec,
                    source_ref=ref,
                    assembly_step=step,
                )
            )

    # From servos at joints
    for joint in body.joints:
        for port in joint.servo.wire_ports:
            if not port.connector_type:
                continue
            try:
                spec = connector_spec(port.connector_type)
            except KeyError:
                continue

            pos = _servo_port_pos_in_body_frame(joint, port.pos)
            exit_dir = rotate_vec(joint.solved_servo_quat, spec.wire_exit_direction)
            rotated_offset = rotate_vec(joint.solved_servo_quat, spec.wire_exit_offset)
            stub_base = (
                pos[0] + rotated_offset[0],
                pos[1] + rotated_offset[1],
                pos[2] + rotated_offset[2],
            )
            mount_label = f"servo_{joint.name}"
            ref = ComponentRef(body=body.name, mount_label=mount_label)
            step = step_index.get((body.name, mount_label), 0)

            stubs.append(
                _StubRecord(
                    label=f"{mount_label}:{port.label}",
                    pos=pos,
                    stub_base=stub_base,
                    exit_dir=exit_dir,
                    spec=spec,
                    source_ref=ref,
                    assembly_step=step,
                )
            )

    return stubs


def _aabb(
    center: tuple[float, float, float],
    half: tuple[float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return (min_corner, max_corner) for an AABB."""
    return (
        (center[0] - half[0], center[1] - half[1], center[2] - half[2]),
        (center[0] + half[0], center[1] + half[1], center[2] + half[2]),
    )


def _connector_aabb(
    stub: _StubRecord,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """AABB for the connector body at the port position."""
    dims = stub.spec.body_dimensions
    half = (dims[0] / 2, dims[1] / 2, dims[2] / 2)
    return _aabb(stub.pos, half)


def _stub_aabb(
    stub: _StubRecord,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """AABB for the wire stub cylinder extending from wire-exit base along exit dir."""
    dx, dy, dz = stub.exit_dir
    px, py, pz = stub.stub_base
    length = float(WIRE_STUB_LENGTH)
    radius = float(WIRE_STUB_RADIUS)

    # Endpoint of the stub
    ex = px + dx * length
    ey = py + dy * length
    ez = pz + dz * length

    # AABB of cylinder: min/max of start/end, expanded by radius
    lo = (min(px, ex) - radius, min(py, ey) - radius, min(pz, ez) - radius)
    hi = (max(px, ex) + radius, max(py, ey) + radius, max(pz, ez) + radius)
    return (lo, hi)


def _aabb_overlap(
    a: tuple[tuple[float, float, float], tuple[float, float, float]],
    b: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> float:
    """Compute minimum overlap between two AABBs.

    Returns positive if overlapping, negative if separated (= gap distance).
    """
    overlaps: list[float] = []
    for i in range(3):
        overlap_i = min(a[1][i], b[1][i]) - max(a[0][i], b[0][i])
        overlaps.append(overlap_i)

    # If any axis has negative overlap, boxes are separated.
    # The gap is the most-negative overlap (closest axis).
    return min(overlaps)


def _check_stub_pair(
    body: Body,
    a: _StubRecord,
    b: _StubRecord,
) -> DFMFinding | None:
    """Check a pair of stubs for connector body or wire stub collision."""
    # Check connector body overlap
    conn_overlap = _aabb_overlap(_connector_aabb(a), _connector_aabb(b))

    # Check wire stub overlap
    stub_overlap = _aabb_overlap(_stub_aabb(a), _stub_aabb(b))

    # Use the worst (most positive) overlap
    worst_overlap = max(conn_overlap, stub_overlap)
    is_connector = conn_overlap >= stub_overlap

    if worst_overlap > 0:
        # Actual overlap
        measured_mm = worst_overlap * 1000
        kind = "Connector body" if is_connector else "Wire stub"
        return DFMFinding(
            check_name="wire_stub_collision",
            severity=DFMSeverity.ERROR,
            body=body.name,
            target=a.source_ref,
            assembly_step=a.assembly_step,
            title=f"{kind} collision: {a.label} vs {b.label}",
            description=(
                f"{kind} overlap of {measured_mm:.1f}mm between "
                f"{a.label} and {b.label}."
            ),
            pos=a.pos,
            direction=a.exit_dir,
            measured=measured_mm,
            threshold=0.0,
            has_overlay=False,
        )

    gap = -worst_overlap
    if gap < GAP_WARNING_THRESHOLD:
        gap_mm = gap * 1000
        kind = "Connector body" if is_connector else "Wire stub"
        return DFMFinding(
            check_name="wire_stub_collision",
            severity=DFMSeverity.WARNING,
            body=body.name,
            target=a.source_ref,
            assembly_step=a.assembly_step,
            title=f"Tight {kind.lower()} gap: {a.label} vs {b.label}",
            description=(
                f"{kind} gap of {gap_mm:.1f}mm between "
                f"{a.label} and {b.label} (< {float(GAP_WARNING_THRESHOLD) * 1000:.0f}mm)."
            ),
            pos=a.pos,
            direction=a.exit_dir,
            measured=gap_mm,
            threshold=float(GAP_WARNING_THRESHOLD) * 1000,
            has_overlay=False,
        )

    return None
