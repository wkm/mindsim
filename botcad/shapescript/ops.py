"""Typed ShapeScript operations for the CAD pipeline.

Every operation is a frozen dataclass. Shape-producing ops carry a ShapeRef
that downstream ops use to reference the result. Tags are optional string
labels that propagate through boolean operations so fillets/chamfers can
reference edges by name rather than by topological index.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class Align3:
    """Per-axis alignment for primitives. Maps to build123d Align enum."""

    x: str = "center"  # "center" | "min" | "max"
    y: str = "center"
    z: str = "center"


# Convenience constants
ALIGN_CENTER = Align3()
ALIGN_MIN_Z = Align3(z="min")
ALIGN_MAX_Z = Align3(z="max")
ALIGN_MIN_X = Align3(x="min")


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
    align: Align3 = ALIGN_CENTER
    tag: str | None = None


@dataclass(frozen=True)
class CylinderOp:
    ref: ShapeRef
    radius: float
    height: float
    align: Align3 = ALIGN_CENTER
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


@dataclass(frozen=True)
class CallOp:
    """Invoke a sub-program and produce its output shape."""

    ref: ShapeRef
    sub_program_key: str
    tag: str | None = None


# ── Copy ──


@dataclass(frozen=True)
class CopyOp:
    """Create an independent copy of an existing shape.

    The copy can be located/modified without affecting the original.
    Create a prototype once, Copy it N times, Locate each differently.
    """

    ref: ShapeRef
    source: ShapeRef
    tag: str | None = None


# ── Patterns ──


@dataclass(frozen=True)
class RadialArrayOp:
    """Clone a shape N times in a radial pattern around an axis.

    The source shape is assumed to be already positioned at the desired
    radius. Each clone is rotated by 360/count degrees. All clones
    (including the original at angle=0) are fused into one shape.
    """

    ref: ShapeRef
    source: ShapeRef
    count: int
    axis: str = "z"  # "x", "y", or "z"
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
class FilletAllEdgesOp:
    """Fillet all edges on a shape with the given radius."""

    ref: ShapeRef
    target: ShapeRef
    radius: float


@dataclass(frozen=True)
class FilletByAxisOp:
    """Fillet edges aligned with a specific axis."""

    ref: ShapeRef
    target: ShapeRef
    axis: str  # "x", "y", or "z"
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
PrimitiveOp = Union[BoxOp, CylinderOp, SphereOp, PrebuiltOp, CallOp, CopyOp, RadialArrayOp]
BooleanOp = Union[FuseOp, CutOp]
TransformOp = LocateOp
ModificationOp = Union[FilletOp, FilletAllEdgesOp, FilletByAxisOp, ChamferOp]
QueryOp = Union[
    QueryVolumeOp, QueryCentroidOp, QueryInertiaOp, QueryBBoxOp, QueryAreaOp
]
ExportOp = Union[ExportSTLOp, ExportSTEPOp]
ShapeOp = Union[
    PrimitiveOp, BooleanOp, TransformOp, ModificationOp, QueryOp, ExportOp
]
