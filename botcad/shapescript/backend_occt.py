"""OCCT backend -- executes ShapeScript programs against build123d.

Walks the op list sequentially, maintaining a shape table (ShapeRef -> Solid)
and a tag registry. Returns an ExecutionResult with query answers and
the final shape table.

CRITICAL: Uses .moved() NOT .locate() for LocateOp. .locate() mutates in
place and corrupts cached shapes. See memory/feedback_build123d_locate.md.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

from botcad.shapescript.ops import (  # noqa: F401
    Align3,
    BoxOp,
    CallOp,
    ChamferOp,
    CopyOp,
    CutOp,
    CylinderOp,
    ExportSTEPOp,
    ExportSTLOp,
    FilletAllEdgesOp,
    FilletByAxisOp,
    FilletOp,
    FuseOp,
    LocateOp,
    PrebuiltOp,
    QueryAreaOp,
    QueryBBoxOp,
    QueryCentroidOp,
    QueryInertiaOp,
    QueryVolumeOp,
    RadialArrayOp,
    SphereOp,
)
from botcad.shapescript.program import ShapeScript
from botcad.shapescript.tags import TagRegistry

log = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of executing a ShapeScript."""

    shapes: dict[str, object] = field(default_factory=dict)  # ref.id -> build123d Solid
    queries: list[Any] = field(default_factory=list)  # ordered query results
    tags: TagRegistry = field(default_factory=TagRegistry)


class OcctBackend:
    """Executes ShapeScript ops against build123d/OCCT."""

    def execute(self, program: ShapeScript) -> ExecutionResult:
        import copy

        from build123d import (
            Align,
            Box,
            Compound,
            Cylinder,
            Location,
            Solid,
            Sphere,
            Unit,
            export_step,
            export_stl,
        )

        # Reuse battle-tested helpers -- do NOT reimplement
        from botcad.cad_utils import as_solid as _as_solid
        from botcad.emit.cad import _bboxes_overlap, _bool_cut, _ensure_solid

        result = ExecutionResult()
        shapes = result.shapes
        tags = result.tags

        def _align(a: Align3):
            _MAP = {"center": Align.CENTER, "min": Align.MIN, "max": Align.MAX}
            return (_MAP[a.x], _MAP[a.y], _MAP[a.z])

        def _to_solid(part):
            """Convert Part subclass (Box, Cylinder, etc.) to plain Solid.

            build123d Part subclasses override __init__ with required args,
            which breaks .fillet()/.chamfer() that internally call
            self.__class__(new_shape). Converting to Solid avoids this.
            """
            if type(part) is Solid:
                return part
            return Solid(part.wrapped)

        for op in program.ops:
            match op:
                # -- Primitives --
                case BoxOp(ref=ref, width=w, length=l, height=h, align=align, tag=tag):
                    s = _to_solid(Box(w, l, h, align=_align(align)))
                    shapes[ref.id] = s
                    if tag:
                        tags.declare(tag, ref)

                case CylinderOp(ref=ref, radius=r, height=h, align=align, tag=tag):
                    s = _to_solid(Cylinder(r, h, align=_align(align)))
                    shapes[ref.id] = s
                    if tag:
                        tags.declare(tag, ref)

                case SphereOp(ref=ref, radius=r, tag=tag):
                    s = _to_solid(Sphere(r))
                    shapes[ref.id] = s
                    if tag:
                        tags.declare(tag, ref)

                case PrebuiltOp(ref=ref, solid_hash=_, tag=tag):
                    # Prebuilt solids stored in program.prebuilt_solids
                    if ref.id in program.prebuilt_solids:
                        s = program.prebuilt_solids[ref.id]
                        shapes[ref.id] = s
                    else:
                        raise ValueError(f"PrebuiltOp {ref.id} has no associated solid")
                    if tag:
                        tags.declare(tag, ref)

                case CallOp(ref=ref, sub_program_key=key, tag=tag):
                    sub_prog = program.sub_programs.get(key)
                    if sub_prog is None:
                        raise ValueError(f"CallOp: sub-program '{key}' not found")
                    sub_result = self.execute(sub_prog)
                    if sub_prog.output_ref is None:
                        raise ValueError(
                            f"CallOp: sub-program '{key}' has no output_ref"
                        )
                    shapes[ref.id] = sub_result.shapes[sub_prog.output_ref.id]
                    if tag:
                        tags.declare(tag, ref)

                case CopyOp(ref=ref, source=src, tag=tag):
                    # copy.copy() creates an independent clone of the OCCT solid
                    s = shapes[src.id]
                    shapes[ref.id] = copy.copy(s)
                    if tag:
                        tags.declare(tag, ref)
                    tags.propagate_transform(ref, src)

                case RadialArrayOp(ref=ref, source=src, count=n, axis=axis, tag=tag):
                    s = shapes[src.id]
                    step = 360.0 / n
                    clones = [s]  # include original at angle 0
                    for i in range(1, n):
                        angle = i * step
                        if axis == "z":
                            euler = (0, 0, angle)
                        elif axis == "y":
                            euler = (0, angle, 0)
                        else:
                            euler = (angle, 0, 0)
                        clone = copy.copy(s)
                        clones.append(clone.moved(Location((0, 0, 0), euler)))
                    # Batch fuse all clones in one OCCT operation
                    if len(clones) == 1:
                        shapes[ref.id] = clones[0]
                    else:
                        shapes[ref.id] = _as_solid(clones[0].fuse(*clones[1:]))
                    if tag:
                        tags.declare(tag, ref)

                # -- Booleans --
                case FuseOp(ref=ref, target=t, tool=tl):
                    a = shapes[t.id]
                    b = shapes[tl.id]
                    # Use _as_solid (not _ensure_solid) to preserve small
                    # pieces like cradle solids that _ensure_solid's 1% sliver
                    # filter would discard.
                    shapes[ref.id] = _as_solid(a.fuse(b))
                    tags.propagate_boolean(ref, t, tl)

                case CutOp(ref=ref, target=t, tool=tl):
                    a = shapes[t.id]
                    b = shapes[tl.id]
                    # Guard: skip cut if bboxes don't overlap (prevents OCCT hangs)
                    if _bboxes_overlap(a, b):
                        # Use _bool_cut to handle Compound/ShapeList targets,
                        # then _ensure_solid to strip sliver fragments.
                        shapes[ref.id] = _ensure_solid(_bool_cut(a, b))
                    else:
                        shapes[ref.id] = a
                    tags.propagate_boolean(ref, t, tl)

                # -- Transforms --
                case LocateOp(ref=ref, target=t, pos=pos, euler_deg=euler):
                    s = shapes[t.id]
                    if any(e != 0 for e in euler):
                        loc = Location(pos, euler)
                    else:
                        loc = Location(pos)
                    # CRITICAL: use .moved() not .locate()
                    # .locate() mutates in place and corrupts cached shapes.
                    shapes[ref.id] = s.moved(loc)
                    tags.propagate_transform(ref, t)

                # -- Modifications --
                case FilletOp(ref=ref, target=t, tags=ftags, radius=radius):
                    s = shapes[t.id]
                    edges = _resolve_tagged_edges(s, ftags, tags, shapes)
                    if edges:
                        shapes[ref.id] = s.fillet(radius, edges)
                    else:
                        log.warning(
                            "Fillet: no edges resolved for tags %s, passing through",
                            ftags,
                        )
                        shapes[ref.id] = s
                    tags.propagate_transform(ref, t)

                case FilletAllEdgesOp(ref=ref, target=t, radius=radius):
                    s = shapes[t.id]
                    try:
                        shapes[ref.id] = s.fillet(radius, s.edges())
                    except Exception:
                        shapes[ref.id] = s  # fillet failed, pass through
                    tags.propagate_transform(ref, t)

                case FilletByAxisOp(ref=ref, target=t, axis=axis, radius=radius):
                    from build123d import Vector

                    s = shapes[t.id]
                    axis_vec = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
                    edges = [
                        e
                        for e in s.edges()
                        if abs(e.tangent_at(0).dot(Vector(*axis_vec))) > 0.9
                    ]
                    try:
                        shapes[ref.id] = s.fillet(radius, edges) if edges else s
                    except Exception:
                        shapes[ref.id] = s
                    tags.propagate_transform(ref, t)

                case ChamferOp(ref=ref, target=t, tags=ctags, size=size):
                    s = shapes[t.id]
                    edges = _resolve_tagged_edges(s, ctags, tags, shapes)
                    if edges:
                        shapes[ref.id] = s.chamfer(size, None, edges)
                    else:
                        log.warning(
                            "Chamfer: no edges resolved for tags %s, passing through",
                            ctags,
                        )
                        shapes[ref.id] = s
                    tags.propagate_transform(ref, t)

                # -- Queries --
                case QueryVolumeOp(target=t):
                    s = shapes[t.id]
                    result.queries.append(abs(s.volume))

                case QueryCentroidOp(target=t):
                    s = shapes[t.id]
                    c = s.center()
                    result.queries.append((c.X, c.Y, c.Z))

                case QueryInertiaOp(target=t):
                    s = shapes[t.id]
                    # build123d matrix_of_inertia is a gp_Mat
                    m = s.matrix_of_inertia
                    result.queries.append(
                        [[m[i][j] for j in range(3)] for i in range(3)]
                    )

                case QueryBBoxOp(target=t):
                    s = shapes[t.id]
                    bb = s.bounding_box()
                    result.queries.append(
                        (
                            (bb.min.X, bb.min.Y, bb.min.Z),
                            (bb.max.X, bb.max.Y, bb.max.Z),
                        )
                    )

                case QueryAreaOp(target=t):
                    s = shapes[t.id]
                    result.queries.append(abs(s.area))

                # -- Export --
                case ExportSTLOp(target=t, path=path):
                    export_stl(shapes[t.id], path, unit=Unit.METER)

                case ExportSTEPOp(targets=ts, path=path):
                    solids = [shapes[t.id] for t in ts]
                    compound = Compound(children=solids)
                    export_step(compound, path)

                case _:
                    raise ValueError(f"Unknown op: {op}")

        return result


def _resolve_tagged_edges(solid, tag_names, tag_registry, shapes):
    """Resolve tag names to actual OCCT edges on the solid.

    Strategy: For each tag, get the source shape (the primitive that declared
    the tag). Use its bounding box to filter edges on the target solid that
    are geometrically close to the tagged shape.

    This is a heuristic -- OCCT doesn't natively track "which edges came from
    which boolean operand." We use proximity to the tool shape's geometry
    as the selection criterion.
    """
    all_edges = []
    solid_edges = solid.edges()

    for tag_name in tag_names:
        source_ref = tag_registry.source_ref(tag_name)
        source_shape = shapes.get(source_ref.id)

        if source_shape is None:
            continue

        # For primitives used as boolean tools: edges near the tool surface
        # are the ones created by the boolean op. Use the tool's bounding box
        # as a spatial filter.
        try:
            tool_bb = source_shape.bounding_box()
        except Exception:
            # Fallback: return all edges (will over-fillet but won't crash)
            all_edges.extend(solid_edges)
            continue

        # Expand tool bbox slightly and filter solid edges by containment
        margin = 0.001  # 1mm tolerance
        for edge in solid_edges:
            try:
                ec = edge.center()
                if (
                    tool_bb.min.X - margin <= ec.X <= tool_bb.max.X + margin
                    and tool_bb.min.Y - margin <= ec.Y <= tool_bb.max.Y + margin
                    and tool_bb.min.Z - margin <= ec.Z <= tool_bb.max.Z + margin
                ):
                    all_edges.append(edge)
            except Exception:
                continue

    return all_edges if all_edges else None
