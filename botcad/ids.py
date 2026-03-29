"""Typed identifiers for skeleton entities.

The StringId pattern: str subclasses that prevent accidental use of
one entity type where another is expected, while remaining fully
compatible with all str operations (JSON, dict keys, formatting, etc.)
during the migration period.
"""

from __future__ import annotations


class BodyId(str):
    """Unique identifier for a Body in the skeleton.

    Subclasses str so it works transparently in JSON serialization,
    f-strings, dict keys, set membership, and string comparisons.
    The .name property provides explicit access to the raw string.
    """

    @property
    def name(self) -> str:
        return str(self)

    def __repr__(self) -> str:
        return f"BodyId({str.__repr__(self)})"


class JointId(str):
    """Unique identifier for a Joint in the skeleton.

    Subclasses str so it works transparently in JSON serialization,
    f-strings, dict keys, set membership, and string comparisons.
    The .name property provides explicit access to the raw string.
    """

    @property
    def name(self) -> str:
        return str(self)

    def __repr__(self) -> str:
        return f"JointId({str.__repr__(self)})"


class ComponentId(str):
    """Unique identifier for a component instance in the wiring netlist.

    For mounted components, this is the mount label (e.g. "controller",
    "pi", "battery"). For servos, it is the joint name (e.g. "shoulder",
    "elbow") since servos live on joints, not mounts.
    """

    @property
    def name(self) -> str:
        return str(self)

    def __repr__(self) -> str:
        return f"ComponentId({str.__repr__(self)})"
