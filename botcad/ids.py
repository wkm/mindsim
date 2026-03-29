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
