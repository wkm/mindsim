"""Component retention DFM check.

Detects mounted components that have no fasteners securing them in place.
Components without mounting points (e.g. LiPo batteries) that rely purely
on friction fit or gravity are flagged — they will fall out when the robot
tilts or experiences acceleration.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from botcad.assembly.refs import ComponentRef, FastenerRef
from botcad.assembly.sequence import AssemblyAction
from botcad.dfm.check import DFMCheck, DFMFinding, DFMSeverity
from botcad.ids import BodyId

if TYPE_CHECKING:
    from botcad.assembly.sequence import AssemblySequence
    from botcad.skeleton import Body, Bot


class ComponentRetention(DFMCheck):
    """Check that every mounted component is retained by at least one fastener."""

    @property
    def name(self) -> str:
        return "component_retention"

    def run(
        self,
        bot: Bot,
        sequence: AssemblySequence,
        body_solids: dict[BodyId, object],
    ) -> list[DFMFinding]:
        findings: list[DFMFinding] = []
        body_map = _build_body_map(bot)

        # Index: which components have FASTEN ops targeting their mounting points.
        # FastenerRef uses (body, index) where index matches the order in
        # build_assembly_sequence: first component mounting points across all
        # mounts, then servo ears across all joints.
        fastened_mounts = _find_fastened_mounts(sequence, body_map)

        # Walk every body and check each mount
        for body_id, body in body_map.items():
            for mount in body.mounts:
                has_mounting_points = len(mount.component.mounting_points) > 0
                is_fastened = (body_id, mount.label) in fastened_mounts

                if has_mounting_points and is_fastened:
                    # Component has screws and they are in the assembly sequence
                    continue

                if has_mounting_points and not is_fastened:
                    # Has mounting holes but no FASTEN ops — something is wrong
                    # in the assembly sequence, but that is a different check.
                    # Still flag it.
                    severity = DFMSeverity.WARNING
                    title = f"Unfastened component: {mount.label}"
                    description = (
                        f"{mount.component.name} in {body_id} has "
                        f"{len(mount.component.mounting_points)} mounting point(s) "
                        f"but no fasteners in the assembly sequence."
                    )
                else:
                    # No mounting points at all — friction/gravity only
                    severity = DFMSeverity.WARNING
                    title = f"No retention: {mount.label}"
                    description = (
                        f"{mount.component.name} in {body_id} has no mounting "
                        f"points and no fasteners — retained by friction fit only. "
                        f"It may fall out under tilt or acceleration "
                        f"(mass: {mount.component.mass * 1000:.0f}g)."
                    )

                # Find the INSERT step for this component (for assembly_step)
                insert_step = _find_insert_step(sequence, body_id, mount.label)

                findings.append(
                    DFMFinding(
                        check_name=self.name,
                        severity=severity,
                        body=body_id,
                        target=ComponentRef(body=body_id, mount_label=mount.label),
                        assembly_step=insert_step,
                        title=title,
                        description=description,
                        pos=mount.resolved_pos,
                        direction=None,
                        measured=None,
                        threshold=None,
                        has_overlay=False,
                    )
                )

        return findings


def _build_body_map(bot: Bot) -> dict[BodyId, Body]:
    """Walk the kinematic tree and collect all bodies by name."""
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


def _find_fastened_mounts(
    sequence: AssemblySequence,
    body_map: dict[BodyId, Body],
) -> set[tuple[BodyId, str]]:
    """Return the set of (body_id, mount_label) pairs that have FASTEN ops.

    A mount is considered fastened if any FastenerRef index falls within
    the range of mounting point indices for that mount's component.
    The index scheme from build_assembly_sequence:
      - Component mounting points first, ordered by mount order in body.mounts
      - Then servo mounting ears, ordered by joint order
    """
    # Build mapping: (body_id, fastener_index) -> mount_label
    fastener_to_mount: dict[tuple[BodyId, int], str] = {}
    for body_id, body in body_map.items():
        idx = 0
        for mount in body.mounts:
            for _mp in mount.component.mounting_points:
                fastener_to_mount[(body_id, idx)] = mount.label
                idx += 1
            # Skip mounts with no mounting points — they produce no fastener indices
        # Servo ears are not component mounts, skip them here

    # Collect all fastened mount labels
    fastened: set[tuple[BodyId, str]] = set()
    for op in sequence.ops:
        if op.action != AssemblyAction.FASTEN:
            continue
        if not isinstance(op.target, FastenerRef):
            continue
        key = (op.target.body, op.target.index)
        if key in fastener_to_mount:
            fastened.add((op.target.body, fastener_to_mount[key]))

    return fastened


def _find_insert_step(
    sequence: AssemblySequence,
    body_id: BodyId,
    mount_label: str,
) -> int:
    """Find the assembly step index for the INSERT op of a component."""
    for op in sequence.ops:
        if op.action != AssemblyAction.INSERT:
            continue
        if not isinstance(op.target, ComponentRef):
            continue
        if op.target.body == body_id and op.target.mount_label == mount_label:
            return op.step
    return -1
