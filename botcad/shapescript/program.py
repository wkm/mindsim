"""ShapeScript — ordered op sequence with builder API and content hashing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

from botcad.shapescript.ops import (
    ALIGN_CENTER,
    Align3,
    BoxOp,
    CallOp,
    ChamferOp,
    CutOp,
    CylinderOp,
    ExportSTEPOp,
    ExportSTLOp,
    FilletOp,
    FuseOp,
    LocateOp,
    PrebuiltOp,
    QueryAreaOp,
    QueryBBoxOp,
    QueryCentroidOp,
    QueryInertiaOp,
    QueryVolumeOp,
    ShapeOp,
    ShapeRef,
    SphereOp,
    Vec3,
)


@dataclass
class ShapeScript:
    """Ordered sequence of IR ops with a builder API.

    Usage:
        prog = ShapeScript()
        box = prog.box(0.06, 0.04, 0.02)
        hole = prog.cylinder(0.01, 0.1, tag="pocket")
        hole = prog.locate(hole, pos=(0.01, 0, 0))
        result = prog.cut(box, hole)
        prog.query_volume(result)
        prog.output_ref = result
    """

    ops: list[ShapeOp] = field(default_factory=list)
    output_ref: ShapeRef | None = None  # explicit final shape
    prebuilt_solids: dict[str, object] = field(default_factory=dict)  # ref.id -> solid
    sub_programs: dict[str, ShapeScript] = field(default_factory=dict)
    _counter: int = field(default=0, repr=False)

    def _next_ref(self, prefix: str) -> ShapeRef:
        ref = ShapeRef(f"{prefix}_{self._counter}")
        self._counter += 1
        return ref

    # ── Primitives ──

    def box(
        self,
        width: float,
        length: float,
        height: float,
        align: Align3 = ALIGN_CENTER,
        tag: str | None = None,
    ) -> ShapeRef:
        ref = self._next_ref("box")
        self.ops.append(
            BoxOp(
                ref=ref,
                width=width,
                length=length,
                height=height,
                align=align,
                tag=tag,
            )
        )
        return ref

    def cylinder(
        self,
        radius: float,
        height: float,
        align: Align3 = ALIGN_CENTER,
        tag: str | None = None,
    ) -> ShapeRef:
        ref = self._next_ref("cyl")
        self.ops.append(
            CylinderOp(ref=ref, radius=radius, height=height, align=align, tag=tag)
        )
        return ref

    def sphere(self, radius: float, tag: str | None = None) -> ShapeRef:
        ref = self._next_ref("sph")
        self.ops.append(SphereOp(ref=ref, radius=radius, tag=tag))
        return ref

    def prebuilt(self, solid: object, tag: str | None = None) -> ShapeRef:
        """Inject a pre-built solid (e.g. from bracket.py factories).

        The solid is stored out-of-band in prebuilt_solids and referenced
        by its ShapeRef id. A content hash of the solid's volume is used
        for cache invalidation.
        """
        ref = self._next_ref("pre")
        solid_hash = str(hash(id(solid)))  # identity-based; real hash is volume-based
        try:
            solid_hash = f"vol:{abs(solid.volume):.10e}"
        except Exception:
            pass
        self.ops.append(PrebuiltOp(ref=ref, solid_hash=solid_hash, tag=tag))
        self.prebuilt_solids[ref.id] = solid
        return ref

    def call(self, key: str, tag: str | None = None) -> ShapeRef:
        """Invoke a sub-program stored in sub_programs[key]."""
        ref = self._next_ref("call")
        self.ops.append(CallOp(ref=ref, sub_program_key=key, tag=tag))
        return ref

    # ── Booleans ──

    def fuse(self, target: ShapeRef, tool: ShapeRef) -> ShapeRef:
        ref = self._next_ref("fuse")
        self.ops.append(FuseOp(ref=ref, target=target, tool=tool))
        return ref

    def cut(self, target: ShapeRef, tool: ShapeRef) -> ShapeRef:
        ref = self._next_ref("cut")
        self.ops.append(CutOp(ref=ref, target=target, tool=tool))
        return ref

    # ── Transforms ──

    def locate(
        self,
        target: ShapeRef,
        pos: Vec3 = (0.0, 0.0, 0.0),
        euler_deg: Vec3 = (0.0, 0.0, 0.0),
    ) -> ShapeRef:
        ref = self._next_ref("loc")
        self.ops.append(LocateOp(ref=ref, target=target, pos=pos, euler_deg=euler_deg))
        return ref

    # ── Modifications ──

    def fillet(
        self, target: ShapeRef, tags: tuple[str, ...], radius: float
    ) -> ShapeRef:
        ref = self._next_ref("fillet")
        self.ops.append(FilletOp(ref=ref, target=target, tags=tags, radius=radius))
        return ref

    def fillet_all(self, target: ShapeRef, radius: float) -> ShapeRef:
        from botcad.shapescript.ops import FilletAllEdgesOp

        ref = self._next_ref("fillet")
        self.ops.append(FilletAllEdgesOp(ref=ref, target=target, radius=radius))
        return ref

    def fillet_by_axis(self, target: ShapeRef, axis: str, radius: float) -> ShapeRef:
        from botcad.shapescript.ops import FilletByAxisOp

        ref = self._next_ref("fillet")
        self.ops.append(
            FilletByAxisOp(ref=ref, target=target, axis=axis, radius=radius)
        )
        return ref

    def chamfer(self, target: ShapeRef, tags: tuple[str, ...], size: float) -> ShapeRef:
        ref = self._next_ref("cham")
        self.ops.append(ChamferOp(ref=ref, target=target, tags=tags, size=size))
        return ref

    # ── Queries ──

    def query_volume(self, target: ShapeRef) -> None:
        self.ops.append(QueryVolumeOp(target=target))

    def query_centroid(self, target: ShapeRef) -> None:
        self.ops.append(QueryCentroidOp(target=target))

    def query_inertia(self, target: ShapeRef) -> None:
        self.ops.append(QueryInertiaOp(target=target))

    def query_bbox(self, target: ShapeRef) -> None:
        self.ops.append(QueryBBoxOp(target=target))

    def query_area(self, target: ShapeRef) -> None:
        self.ops.append(QueryAreaOp(target=target))

    # ── Export ──

    def export_stl(self, target: ShapeRef, path: str) -> None:
        self.ops.append(ExportSTLOp(target=target, path=path))

    def export_step(self, targets: tuple[ShapeRef, ...], path: str) -> None:
        self.ops.append(ExportSTEPOp(targets=targets, path=path))

    # ── Hashing & Serialization ──

    def content_hash(self) -> str:
        """SHA-256 of the canonical JSON representation.

        Includes prebuilt solid hashes so cache invalidates when
        bracket/component geometry changes. Also includes sub-program
        hashes so cache invalidates when sub-programs change.
        """
        parts = [self.to_json()]
        for key in sorted(self.sub_programs):
            parts.append(f"{key}:{self.sub_programs[key].content_hash()}")
        return hashlib.sha256("|".join(parts).encode()).hexdigest()

    def to_json(self) -> str:
        """Canonical JSON serialization (deterministic key order)."""
        ops_data = []
        for op in self.ops:
            d = asdict(op)
            d["_type"] = type(op).__name__
            ops_data.append(d)
        return json.dumps(ops_data, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, json_str: str) -> ShapeScript:
        """Deserialize from canonical JSON."""
        import botcad.shapescript.ops as ops_mod

        ops_data = json.loads(json_str)
        prog = cls()
        max_counter = 0
        for d in ops_data:
            type_name = d.pop("_type")
            op_cls = getattr(ops_mod, type_name)
            # Reconstruct ShapeRef fields
            for key, val in list(d.items()):
                if isinstance(val, dict) and "id" in val:
                    d[key] = ShapeRef(val["id"])
                elif isinstance(val, list) and val and isinstance(val[0], dict):
                    d[key] = tuple(ShapeRef(v["id"]) for v in val)
                elif isinstance(val, list):
                    d[key] = tuple(val)
            # Reconstruct Align3 dataclass from dict
            if "align" in d and isinstance(d["align"], dict):
                d["align"] = Align3(**d["align"])
            op = op_cls(**d)
            prog.ops.append(op)
            if hasattr(op, "ref"):
                # Track counter from ref IDs
                parts = op.ref.id.rsplit("_", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    max_counter = max(max_counter, int(parts[1]) + 1)
        prog._counter = max_counter
        return prog
