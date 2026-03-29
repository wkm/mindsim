"""Shared DFM utility functions."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from botcad.assembly.refs import WireRef
from botcad.assembly.sequence import AssemblyAction

if TYPE_CHECKING:
    from botcad.assembly.sequence import AssemblySequence
    from botcad.ids import BodyId
    from botcad.skeleton import Body, Bot


def build_body_map(bot: Bot) -> dict[BodyId, Body]:
    """Return a {name: body} dict for all bodies in the bot.

    Uses bot.all_bodies when populated (post-solve), otherwise
    walks the kinematic tree from root.
    """
    if bot.all_bodies:
        return {b.name: b for b in bot.all_bodies}

    # Fallback: walk from root (pre-solve / test contexts)
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


def build_wire_steps(sequence: AssemblySequence) -> dict[str, int]:
    """Map wire route labels to their ROUTE_WIRE assembly step index."""
    return {
        op.target.label: op.step
        for op in sequence.ops
        if op.action == AssemblyAction.ROUTE_WIRE and isinstance(op.target, WireRef)
    }
