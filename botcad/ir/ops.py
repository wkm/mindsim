"""Typed IR operations for the CAD pipeline.

Every operation is a frozen dataclass. Shape-producing ops carry a ShapeRef
that downstream ops use to reference the result. Tags are optional string
labels that propagate through boolean operations so fillets/chamfers can
reference edges by name rather than by topological index.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Union

Vec3 = tuple[float, float, float]


class Align3(Enum):
    """3-axis alignment for primitives."""

    CENTER = "center"
    MIN_Z = "min_z"  # centered XY, min Z (bottom-aligned)


@dataclass(frozen=True)
class ShapeRef:
    """Opaque handle to a shape produced by an IR op."""

    id: str

    def __str__(self) -> str:
        return self.id


# ── Primitives ──


@dataclass(frozen=True)
class BoxOp:
    ref: ShapeRef
    width: float
    length: float
    height: float
    align: Align3 = Align3.CENTER
    tag: str | None = None


@dataclass(frozen=True)
class CylinderOp:
    ref: ShapeRef
    radius: float
    height: float
    align: Align3 = Align3.CENTER
    tag: str | None = None


@dataclass(frozen=True)
class SphereOp:
    ref: ShapeRef
    radius: float
    tag: str | None = None


@dataclass(frozen=True)
class PrebuiltOp:
    """Injects a pre-built solid into the IR (e.g. from bracket.py factories).

    The solid_hash is a content hash of the prebuilt geometry, used for
    cache invalidation. The actual solid is passed out-of-band to the backend.
    """

    ref: ShapeRef
    solid_hash: str
    tag: str | None = None


# ── Booleans ──


@dataclass(frozen=True)
class FuseOp:
    ref: ShapeRef
    target: ShapeRef
    tool: ShapeRef


@dataclass(frozen=True)
class CutOp:
    ref: ShapeRef
    target: ShapeRef
    tool: ShapeRef


# ── Transforms ──


@dataclass(frozen=True)
class LocateOp:
    ref: ShapeRef
    target: ShapeRef
    pos: Vec3 = (0.0, 0.0, 0.0)
    euler_deg: Vec3 = (0.0, 0.0, 0.0)


# ── Modifications ──


@dataclass(frozen=True)
class FilletOp:
    ref: ShapeRef
    target: ShapeRef
    tags: tuple[str, ...]
    radius: float


@dataclass(frozen=True)
class ChamferOp:
    ref: ShapeRef
    target: ShapeRef
    tags: tuple[str, ...]
    size: float


# ── Queries (no ShapeRef — these return values, not shapes) ──


@dataclass(frozen=True)
class QueryVolumeOp:
    target: ShapeRef


@dataclass(frozen=True)
class QueryCentroidOp:
    target: ShapeRef


@dataclass(frozen=True)
class QueryInertiaOp:
    target: ShapeRef


@dataclass(frozen=True)
class QueryBBoxOp:
    target: ShapeRef


@dataclass(frozen=True)
class QueryAreaOp:
    target: ShapeRef


# ── Export ──


@dataclass(frozen=True)
class ExportSTLOp:
    target: ShapeRef
    path: str


@dataclass(frozen=True)
class ExportSTEPOp:
    targets: tuple[ShapeRef, ...]
    path: str


# Union of all ops for type narrowing
PrimitiveOp = Union[BoxOp, CylinderOp, SphereOp, PrebuiltOp]
BooleanOp = Union[FuseOp, CutOp]
TransformOp = LocateOp
ModificationOp = Union[FilletOp, ChamferOp]
QueryOp = Union[
    QueryVolumeOp, QueryCentroidOp, QueryInertiaOp, QueryBBoxOp, QueryAreaOp
]
ExportOp = Union[ExportSTLOp, ExportSTEPOp]
CadOp = Union[
    PrimitiveOp, BooleanOp, TransformOp, ModificationOp, QueryOp, ExportOp
]
