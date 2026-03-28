"""Rerun-based step-by-step debugger for ShapeScript programs.

Each IR op maps to a Rerun timestep. Shape-producing ops export their
mesh and log it as a 3D entity. Boolean ops log both operands (tool in
red overlay) plus the result.
"""

from __future__ import annotations

import logging
from pathlib import Path

from botcad.shapescript.backend_occt import OcctBackend
from botcad.shapescript.ops import CutOp, FuseOp
from botcad.shapescript.program import ShapeScriptBuilder

log = logging.getLogger(__name__)


def debug_program(
    prog: ShapeScriptBuilder,
    spawn_viewer: bool = True,
    mesh_dir: Path | None = None,
) -> None:
    """Execute program step-by-step, logging each op to Rerun.

    Args:
        prog: The ShapeScript to debug.
        spawn_viewer: If True, launch the Rerun viewer.
        mesh_dir: If set, export per-step STL files here.
    """
    import rerun as rr
    from build123d import export_stl

    rr.init("botcad_shapescript_debug", spawn=spawn_viewer)

    # Execute the full program to get all shapes
    backend = OcctBackend()
    result = backend.execute(prog)

    if mesh_dir is not None:
        mesh_dir = Path(mesh_dir)
        mesh_dir.mkdir(parents=True, exist_ok=True)

    for step, op in enumerate(prog.ops):
        rr.set_time("step", sequence=step)

        # Log op metadata
        op_desc = type(op).__name__
        if hasattr(op, "ref"):
            op_desc += f" -> {op.ref.id}"
        rr.log("ops/description", rr.TextLog(op_desc))

        # Log meshes for shape-producing ops
        if not hasattr(op, "ref") or op.ref.id not in result.shapes:
            continue

        solid = result.shapes[op.ref.id]
        try:
            if mesh_dir is not None:
                stl_path = mesh_dir / f"step_{step:03d}_{op.ref.id}.stl"
                export_stl(solid, str(stl_path))

            _log_solid(rr, solid, f"shapes/{op.ref.id}")

            # For boolean ops, also log the tool operand
            if isinstance(op, (CutOp, FuseOp)) and op.tool.id in result.shapes:
                _log_solid(rr, result.shapes[op.tool.id], f"shapes/{op.tool.id}_tool")
        except Exception as e:
            log.warning("Failed to log step %d: %s", step, e)


def _log_solid(rr, solid: object, entity_path: str) -> None:
    """Convert a build123d Solid to Rerun Mesh3D and log it."""
    try:
        tess = solid.tessellate(0.001)
        if tess and len(tess) == 2:
            raw_verts, faces = tess
            vertices = [(v.X, v.Y, v.Z) for v in raw_verts]
            rr.log(
                entity_path,
                rr.Mesh3D(
                    vertex_positions=vertices,
                    triangle_indices=faces,
                ),
            )
    except Exception as e:
        log.debug("Tessellation failed for %s: %s", entity_path, e)
