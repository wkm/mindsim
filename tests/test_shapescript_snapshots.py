"""ShapeScript snapshot tests — golden-file diffs for every emitter.

Each emitter's output is rendered to a human-readable text format and
compared against a checked-in baseline file. If the output changes,
the test fails with a diff showing exactly what changed.

Usage:
    uv run pytest tests/test_shapescript_snapshots.py -v          # check
    uv run pytest tests/test_shapescript_snapshots.py --update-shapescript-baselines  # accept
"""

from __future__ import annotations

import difflib
from pathlib import Path

import pytest

b3d = pytest.importorskip("build123d")

BASELINES_DIR = Path(__file__).parent / "shapescript_baselines"


def _render_program(prog) -> str:
    """Render a ShapeScript program as human-readable text."""
    from botcad.shapescript.cad_steps import format_op

    lines = []
    for op in prog.ops:
        lines.append(format_op(op))

    # Add sub-programs if any
    if prog.sub_programs:
        lines.append("")
        for key in sorted(prog.sub_programs):
            sub = prog.sub_programs[key]
            lines.append(f"--- sub-program: {key} ---")
            for op in sub.ops:
                lines.append(f"  {format_op(op)}")

    lines.append("")  # trailing newline
    return "\n".join(lines)


def _check_snapshot(name: str, actual: str, request):
    """Compare actual output against baseline, fail with diff if different."""
    baseline_path = BASELINES_DIR / f"{name}.shapescript"
    actual_path = BASELINES_DIR / f"{name}.shapescript.actual"

    update = request.config.getoption("--update-shapescript-baselines", default=False)

    if update:
        baseline_path.write_text(actual)
        actual_path.unlink(missing_ok=True)
        return

    if not baseline_path.exists():
        # First run — write baseline and skip
        baseline_path.write_text(actual)
        pytest.skip(f"Baseline created: {baseline_path.name} (run again to verify)")

    expected = baseline_path.read_text()
    if actual == expected:
        actual_path.unlink(missing_ok=True)
        return

    # Mismatch — write .actual file and fail with diff
    actual_path.write_text(actual)
    diff = difflib.unified_diff(
        expected.splitlines(keepends=True),
        actual.splitlines(keepends=True),
        fromfile=f"{name}.shapescript (baseline)",
        tofile=f"{name}.shapescript.actual (current)",
    )
    diff_str = "".join(diff)
    pytest.fail(
        f"ShapeScript snapshot mismatch for {name}.\n"
        f"Diff:\n{diff_str}\n\n"
        f"To accept: uv run pytest tests/test_shapescript_snapshots.py "
        f"--update-shapescript-baselines"
    )


# ── Component Snapshots ──


class TestComponentSnapshots:
    def test_camera_ov5647(self, request):
        from botcad.components.camera import OV5647
        from botcad.shapescript.emit_components import camera_script

        prog = camera_script(OV5647())
        _check_snapshot("camera_ov5647", _render_program(prog), request)

    def test_battery_lipo2s(self, request):
        from botcad.components.battery import LiPo2S
        from botcad.shapescript.emit_components import battery_script

        prog = battery_script(LiPo2S(1000))
        _check_snapshot("battery_lipo2s_1000", _render_program(prog), request)

    def test_bearing_6x3x3(self, request):
        from botcad.components.bearing import Bearing6x3x3
        from botcad.shapescript.emit_components import bearing_script

        prog = bearing_script(Bearing6x3x3())
        _check_snapshot("bearing_6x3x3", _render_program(prog), request)

    def test_horn_sts3215(self, request):
        from botcad.components.servo import STS3215
        from botcad.shapescript.emit_components import horn_script

        prog = horn_script(STS3215())
        _check_snapshot("horn_sts3215", _render_program(prog), request)


# ── Servo Snapshots ──


class TestServoSnapshots:
    def test_servo_sts3215(self, request):
        from botcad.components.servo import STS3215
        from botcad.shapescript.emit_servo import servo_script

        prog = servo_script(STS3215())
        _check_snapshot("servo_sts3215", _render_program(prog), request)

    def test_servo_scs0009(self, request):
        from botcad.components.servo import SCS0009
        from botcad.shapescript.emit_servo import servo_script

        prog = servo_script(SCS0009())
        _check_snapshot("servo_scs0009", _render_program(prog), request)


# ── Bracket Snapshots ──


class TestBracketSnapshots:
    def test_bracket_solid_sts3215(self, request):
        from botcad.bracket import BracketSpec
        from botcad.bracket import bracket_solid as bracket_solid_script
        from botcad.components.servo import STS3215

        prog = bracket_solid_script(STS3215(), BracketSpec())
        _check_snapshot("bracket_solid_sts3215", _render_program(prog), request)

    def test_bracket_envelope_sts3215(self, request):
        from botcad.bracket import BracketSpec
        from botcad.bracket import bracket_envelope as bracket_envelope_script
        from botcad.components.servo import STS3215

        prog = bracket_envelope_script(STS3215(), BracketSpec())
        _check_snapshot("bracket_envelope_sts3215", _render_program(prog), request)

    def test_cradle_solid_sts3215(self, request):
        from botcad.bracket import BracketSpec
        from botcad.bracket import cradle_solid as cradle_solid_script
        from botcad.components.servo import STS3215

        prog = cradle_solid_script(STS3215(), BracketSpec())
        _check_snapshot("cradle_solid_sts3215", _render_program(prog), request)


# ── Body Snapshots (the main event) ──


class TestBodySnapshots:
    @pytest.fixture(scope="class")
    def wheeler_base(self):
        import importlib

        mod = importlib.import_module("bots.wheeler_base.design")
        bot = mod.build()
        bot.solve()
        return bot

    def test_wheeler_base_base(self, wheeler_base, request):
        from botcad.shapescript.emit_body import emit_body_ir

        bot = wheeler_base
        pj_map = {}
        for b in bot.all_bodies:
            for j in b.joints:
                if j.child is not None:
                    pj_map[j.child.name] = j

        body = bot.root
        prog = emit_body_ir(body, pj_map.get(body.name))
        _check_snapshot("body_wheeler_base_base", _render_program(prog), request)
