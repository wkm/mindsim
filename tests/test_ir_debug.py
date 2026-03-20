"""Smoke test for the Rerun IR debugger."""

from __future__ import annotations

import pytest

b3d = pytest.importorskip("build123d")
rr = pytest.importorskip("rerun")

from botcad.ir.program import CadProgram


class TestIRDebugRerun:
    def test_debug_simple_program(self):
        """Debug a simple program without crashing."""
        from botcad.ir.debug_rerun import debug_program

        prog = CadProgram()
        box = prog.box(1, 1, 1, tag="shell")
        hole = prog.cylinder(0.2, 2, tag="hole")
        result = prog.cut(box, hole)

        # Should not raise — produces Rerun data but does not spawn viewer
        debug_program(prog, spawn_viewer=False)

    def test_debug_with_export(self, tmp_path):
        """Debug produces per-step mesh files."""
        from botcad.ir.debug_rerun import debug_program

        prog = CadProgram()
        box = prog.box(1, 1, 1)
        hole = prog.cylinder(0.1, 2)
        prog.cut(box, hole)

        debug_program(prog, spawn_viewer=False, mesh_dir=tmp_path)
        # Should have mesh files for shape-producing ops
        stl_files = list(tmp_path.glob("*.stl"))
        assert len(stl_files) >= 2  # at least box and cut result

    def test_debug_fuse_op(self):
        """FuseOp should log without error."""
        from botcad.ir.debug_rerun import debug_program

        prog = CadProgram()
        a = prog.box(1, 1, 1)
        b = prog.sphere(0.3)
        prog.fuse(a, b)

        debug_program(prog, spawn_viewer=False)
