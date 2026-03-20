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
        solid = shapes.get(op.ref.id)
        if solid is None:
            continue

        match op:
            # Primitives — create steps
            case BoxOp(ref=ref, tag=tag):
                label = "Create box" + (f" ({tag})" if tag else "")
                steps.append(CadStep(label=label, solid=solid, op="create"))

            case CylinderOp(ref=ref, tag=tag):
                label = "Create cylinder" + (f" ({tag})" if tag else "")
                steps.append(CadStep(label=label, solid=solid, op="create"))

            case SphereOp(ref=ref, tag=tag):
                label = "Create sphere" + (f" ({tag})" if tag else "")
                steps.append(CadStep(label=label, solid=solid, op="create"))

            case PrebuiltOp(ref=ref, tag=tag):
                label = "Prebuilt" + (f" ({tag})" if tag else "")
                steps.append(CadStep(label=label, solid=solid, op="create"))

            # Booleans — cut/union steps with tool
            case CutOp(ref=ref, target=t, tool=tl):
                tool_solid = shapes.get(tl.id)
                # Build a label from the tool's tag if available
                tool_tag = _find_tag_for_ref(prog, tl)
                label = "Cut" + (f" {tool_tag}" if tool_tag else "")
                steps.append(
                    CadStep(label=label, solid=solid, op="cut", tool=tool_solid)
                )

            case FuseOp(ref=ref, target=t, tool=tl):
                tool_solid = shapes.get(tl.id)
                tool_tag = _find_tag_for_ref(prog, tl)
                label = "Union" + (f" {tool_tag}" if tool_tag else "")
                steps.append(
                    CadStep(label=label, solid=solid, op="union", tool=tool_solid)
                )

            # Transforms — update the previous step's solid (don't create new step)
            case LocateOp(ref=ref, target=t):
                # A LocateOp repositions a shape. Replace the previous step's
                # solid with the repositioned version so the viewer shows the
                # shape in its final position.
                if steps and steps[-1].solid is shapes.get(t.id):
                    steps[-1] = CadStep(
                        label=steps[-1].label,
                        solid=solid,
                        op=steps[-1].op,
                        tool=steps[-1].tool,
                    )
                # Otherwise ignore — transforms between booleans are intermediate

            case _:
                # Query/export ops don't produce visible steps
                pass

    return steps


def _find_tag_for_ref(prog: ShapeScript, ref) -> str | None:
    """Find the tag associated with a ShapeRef by scanning the program ops."""
    for op in prog.ops:
        if op.ref == ref:
            return getattr(op, "tag", None)
    return None
