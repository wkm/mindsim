"""Convert ShapeScript execution results to CadStep objects.

Bridges the ShapeScript IR path with the web viewer's cad-steps API,
which expects CadStep(label, solid, op, tool) objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from botcad.emit.cad import CadStep
    from botcad.shapescript.backend_occt import ExecutionResult
    from botcad.shapescript.program import ShapeScript


def shapescript_to_cad_steps(
    prog: ShapeScript, result: ExecutionResult
) -> list[CadStep]:
    """Convert a ShapeScript program + execution result into CadStep objects.

    For each shape-producing op in the program, creates a CadStep with:
    - label: human-readable description from the op
    - solid: the shape at that step
    - op: "create" for primitives/prebuilt, "cut" for CutOp, "union" for FuseOp
    - tool: for CutOp/FuseOp, the tool solid
    """
    from botcad.emit.cad import CadStep
    from botcad.shapescript.ops import (
        BoxOp,
        CallOp,
        CutOp,
        CylinderOp,
        FuseOp,
        LocateOp,
        PrebuiltOp,
        SphereOp,
    )

    steps: list[CadStep] = []
    shapes = result.shapes

    for op in prog.ops:
        # Query/export ops have no ref — skip them early
        if not hasattr(op, "ref"):
            continue

        solid = shapes.get(op.ref.id)
        if solid is None:
            continue

        script_line = format_op(op)

        match op:
            # Primitives — create steps
            case BoxOp(ref=ref, tag=tag):
                label = "Create box" + (f" ({tag})" if tag else "")
                steps.append(CadStep(label=label, solid=solid, op="create", script=script_line))

            case CylinderOp(ref=ref, tag=tag):
                label = "Create cylinder" + (f" ({tag})" if tag else "")
                steps.append(CadStep(label=label, solid=solid, op="create", script=script_line))

            case SphereOp(ref=ref, tag=tag):
                label = "Create sphere" + (f" ({tag})" if tag else "")
                steps.append(CadStep(label=label, solid=solid, op="create", script=script_line))

            case PrebuiltOp(ref=ref, tag=tag):
                label = "Prebuilt" + (f" ({tag})" if tag else "")
                steps.append(CadStep(label=label, solid=solid, op="create", script=script_line))

            case CallOp(ref=ref, sub_program_key=key, tag=tag):
                label = f"Call: {key}" + (f" ({tag})" if tag else "")
                steps.append(
                    CadStep(label=label, solid=solid, op="create", group=key, script=script_line)
                )

            case CopyOp(ref=ref, source=src, tag=tag):
                label = "Copy" + (f" ({tag})" if tag else f" of {src.id}")
                steps.append(CadStep(label=label, solid=solid, op="create", script=script_line))

            # Booleans — cut/union steps with tool
            case CutOp(ref=ref, target=t, tool=tl):
                tool_solid = shapes.get(tl.id)
                tool_tag = _find_tag_for_ref(prog, tl)
                label = "Cut" + (f" {tool_tag}" if tool_tag else "")
                steps.append(
                    CadStep(label=label, solid=solid, op="cut", tool=tool_solid, script=script_line)
                )

            case FuseOp(ref=ref, target=t, tool=tl):
                tool_solid = shapes.get(tl.id)
                tool_tag = _find_tag_for_ref(prog, tl)
                label = "Union" + (f" {tool_tag}" if tool_tag else "")
                steps.append(
                    CadStep(label=label, solid=solid, op="union", tool=tool_solid, script=script_line)
                )

            # Transforms — show as their own step (purple in the viewer)
            case LocateOp(ref=ref, target=t):
                tool_solid = shapes.get(t.id)  # the pre-move shape
                locate_tag = _find_tag_for_ref(prog, t)
                label = "Locate" + (f" {locate_tag}" if locate_tag else "")
                steps.append(
                    CadStep(label=label, solid=solid, op="locate", tool=tool_solid, script=script_line)
                )

            case _:
                pass

    return steps


def _find_tag_for_ref(prog: ShapeScript, ref) -> str | None:
    """Find the tag associated with a ShapeRef by scanning the program ops."""
    for op in prog.ops:
        if hasattr(op, "ref") and op.ref == ref:
            return getattr(op, "tag", None)
    return None


def format_op(op) -> str:
    """Format a ShapeScript op as a code-like string for the viewer.

    Produces output like:
        box_0 = Box(w=0.0600, l=0.0400, h=0.0200)
        cut_3 = Cut(box_0, loc_2)
        loc_5 = Locate(cyl_4, pos=(0.01, 0, 0))
    """
    from botcad.shapescript.ops import (
        BoxOp,
        CallOp,
        ChamferOp,
        CutOp,
        CylinderOp,
        FilletAllEdgesOp,
        FilletByAxisOp,
        FilletOp,
        FuseOp,
        LocateOp,
        PrebuiltOp,
        SphereOp,
    )

    ref = getattr(op, "ref", None)
    prefix = f"{ref.id} = " if ref else ""

    match op:
        case BoxOp(width=w, length=l, height=h, tag=tag):
            s = f"Box(w={w:.4f}, l={l:.4f}, h={h:.4f})"
            if tag:
                s += f"  # {tag}"
            return prefix + s

        case CylinderOp(radius=r, height=h, tag=tag):
            s = f"Cylinder(r={r:.4f}, h={h:.4f})"
            if tag:
                s += f"  # {tag}"
            return prefix + s

        case SphereOp(radius=r, tag=tag):
            s = f"Sphere(r={r:.4f})"
            if tag:
                s += f"  # {tag}"
            return prefix + s

        case PrebuiltOp(solid_hash=sh, tag=tag):
            s = f"Prebuilt({sh[:8]})"
            if tag:
                s += f"  # {tag}"
            return prefix + s

        case CallOp(sub_program_key=key, tag=tag):
            s = f"Call({key!r})"
            if tag:
                s += f"  # {tag}"
            return prefix + s

        case CopyOp(source=src, tag=tag):
            s = f"Copy({src.id})"
            if tag:
                s += f"  # {tag}"
            return prefix + s

        case FuseOp(target=t, tool=tl):
            return prefix + f"Fuse({t.id}, {tl.id})"

        case CutOp(target=t, tool=tl):
            return prefix + f"Cut({t.id}, {tl.id})"

        case LocateOp(target=t, pos=pos, euler_deg=euler):
            parts = [t.id]
            if any(p != 0 for p in pos):
                parts.append(f"pos=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
            if any(e != 0 for e in euler):
                parts.append(f"rot=({euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f})")
            return prefix + f"Locate({', '.join(parts)})"

        case FilletOp(target=t, tags=tags, radius=r):
            return prefix + f"Fillet({t.id}, tags={tags}, r={r:.4f})"

        case FilletAllEdgesOp(target=t, radius=r):
            return prefix + f"FilletAll({t.id}, r={r:.4f})"

        case FilletByAxisOp(target=t, axis=ax, radius=r):
            return prefix + f"FilletAxis({t.id}, axis={ax!r}, r={r:.4f})"

        case ChamferOp(target=t, tags=tags, size=sz):
            return prefix + f"Chamfer({t.id}, tags={tags}, size={sz:.4f})"

        case _:
            return prefix + type(op).__name__
