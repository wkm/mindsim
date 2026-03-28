"""Typed identifiers for skeleton entities.

The StringId pattern: frozen, hashable wrappers that prevent
accidental use of one entity type where another is expected.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BodyId:
    """Unique identifier for a Body in the skeleton."""

    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class JointId:
    """Unique identifier for a Joint in the skeleton."""

    name: str

    def __str__(self) -> str:
        return self.name
