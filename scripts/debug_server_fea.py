#!/usr/bin/env python3
"""Diagnostic script to run the exact FEA path the server takes."""

import sys
import traceback
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))


def run_diagnostic():
    bot_name = "wheeler_base"
    print(f"--- FEA Diagnostic for {bot_name} ---")

    try:
        from botcad.fea.component import analyze_component
        from botcad.shapescript.backend_occt import OcctBackend
        from botcad.shapescript.emit_body import emit_body_ir
        from botcad.skeleton import BodyKind
        from mindsim.server import _load_bot

        # 1. Load Bot
        print("Loading bot...")
        bot_obj, cad = _load_bot(bot_name)

        # 2. Find target body
        target_body = None
        for body in bot_obj.all_bodies:
            if body.kind == BodyKind.FABRICATED:
                target_body = body
                break

        if not target_body:
            print("ERROR: No fabricated body found")
            return

        print(f"Target body: {target_body.name}")

        # 3. Generate Geometry
        print("Generating ShapeScript and executing...")
        parent_joint = cad.parent_joint_map.get(target_body.name)
        wire_segs = cad.body_wire_segments.get(target_body.name)
        wire_segs_tuple = tuple(wire_segs) if wire_segs else None

        prog = emit_body_ir(target_body, parent_joint, wire_segs_tuple)
        backend = OcctBackend()
        result = backend.execute(prog)
        solid = result.shapes[prog.output_ref.id]

        # 4. Run FEA
        print("Running FEA solve...")
        torque = 2.94
        for joint in target_body.joints:
            if joint.servo:
                torque = joint.servo.stall_torque
                break

        # Using same res as server
        solve_res = analyze_component(
            solid, result, torque_nm=torque, res=(15, 15, 15), body=target_body
        )

        if solve_res:
            _u_field, stress_array, _vd = solve_res
            print(f"SUCCESS: Max Stress {stress_array.numpy().max() / 1e6:.2f} MPa")
        else:
            print("FAILED: FEA solve returned None")

    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    run_diagnostic()
