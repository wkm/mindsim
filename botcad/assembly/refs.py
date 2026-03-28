"""Typed references for assembly entities.

These are frozen, hashable identifiers used in AssemblyOp.target
to reference specific components, fasteners, or wires within a bot.
"""

from __future__ import annotations

from dataclasses import dataclass

from botcad.ids import BodyId


@dataclass(frozen=True)
class ComponentRef:
    """Reference to a mounted component within a body."""

    body: BodyId
    mount_label: str  # matches Mount.label within the body


@dataclass(frozen=True)
class FastenerRef:
    """Reference to a specific fastener within a body."""

    body: BodyId
    index: int  # positional index within the body's fastener list


@dataclass(frozen=True)
class WireRef:
    """Reference to a wire route."""

    label: str  # matches WireRoute.label
