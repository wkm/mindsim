"""ShapeScript emitters for servo_solid (STS-series and SCS0009).

All geometry uses native ShapeScript ops including FilletByAxisOp for
Z-aligned edge fillets. Connector receptacles use CallOp sub-programs.
No PrebuiltOps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from botcad.shapescript.ops import Align3
from botcad.shapescript.program import ShapeScript

if TYPE_CHECKING:
    from botcad.component import ServoSpec


def servo_script(servo: ServoSpec) -> ShapeScript:
    """Dispatch to form-factor-specific emitter based on servo name."""
    if servo.name == "SCS0009":
        return scs0009_script(servo)
    return sts_series_script(servo)


def sts_series_script(servo: ServoSpec) -> ShapeScript:
    """STS-series servo body (STS3215, STS3250, etc.) as ShapeScript.

    All geometry uses native ShapeScript ops — no PrebuiltOps.
    Shaft boss, flanges, rear boss, mounting holes, and connector are native ops.
    """
    prog = ShapeScript()

    # ── Exact geometry constants (from reference CAD) ──
    body_x = 0.0454
    body_y = 0.0248
    body_z = 0.0326
    r = 0.0040

    z_top_surface = body_z / 2
    z_mid_top = z_top_surface - 0.0026
    z_mid_bot = z_mid_top - 0.0288
    z_cap_bot = z_mid_bot - 0.0012

    # ── 1. Middle section (box + fillet Z edges) ──
    mid_h = z_mid_top - z_mid_bot
    middle = prog.box(body_x, body_y, mid_h, tag="sts_middle")
    middle = prog.locate(middle, pos=(0, 0, (z_mid_top + z_mid_bot) / 2))
    middle = prog.fillet_by_axis(middle, "z", r)

    # ── 2. Top cap (box + fillet Z edges + filleted step) ──
    top_h = z_top_surface - z_mid_top
    top_cap = prog.box(body_x, body_y, top_h, align=Align3(z="min"), tag="sts_top_cap")
    top_cap = prog.locate(top_cap, pos=(0, 0, z_mid_top))
    top_cap = prog.fillet_by_axis(top_cap, "z", r)

    step_h = 0.0017
    step = prog.box(0.0200, 0.0200, step_h, align=Align3(z="min"))
    step = prog.locate(step, pos=(0.0125, 0, z_top_surface))
    step = prog.fillet_by_axis(step, "z", 0.002)
    top_cap = prog.fuse(top_cap, step)

    # Output shaft boss (native)
    sx, sy, _sz = servo.shaft_offset
    boss_h = 0.0015
    boss = prog.cylinder(0.0045, boss_h, align=Align3(z="min"))
    boss = prog.locate(boss, pos=(sx, sy, z_top_surface + step_h))
    top_cap = prog.fuse(top_cap, boss)

    # ── 3. Bottom cap (box + fillet Z edges) ──
    bot_h = z_mid_bot - z_cap_bot
    bottom_cap = prog.box(
        body_x, body_y, bot_h, align=Align3(z="max"), tag="sts_bot_cap"
    )
    bottom_cap = prog.locate(bottom_cap, pos=(0, 0, z_mid_bot))
    bottom_cap = prog.fillet_by_axis(bottom_cap, "z", r)

    # Mounting flanges (native)
    f_lx = 0.0404
    f_cx = -0.0005
    f_z_top = z_cap_bot
    f_z_bot = f_z_top - 0.0021
    f_h = f_z_top - f_z_bot

    if servo.mounting_ears:
        for ears in _group_ears_by_y_side(servo.mounting_ears).values():
            f_w = 0.004
            f_cy = (
                (body_y / 2 - f_w / 2)
                if ears[0].pos[1] > 0
                else -(body_y / 2 - f_w / 2)
            )
            flange = prog.box(f_lx, f_w, f_h, align=Align3(z="max"))
            flange = prog.locate(flange, pos=(f_cx, f_cy, f_z_top))
            bottom_cap = prog.fuse(bottom_cap, flange)

    # Rear support bearing boss (native)
    rear_boss = prog.cylinder(0.003, 0.0013, align=Align3(z="max"))
    rear_boss = prog.locate(rear_boss, pos=(sx, sy, z_cap_bot))
    bottom_cap = prog.fuse(bottom_cap, rear_boss)

    # ── 4. Final union ──
    body = prog.fuse(middle, top_cap)
    body = prog.fuse(body, bottom_cap)

    # ── 5. Mounting holes (native cuts) ──
    if servo.mounting_ears:
        for ear in servo.mounting_ears:
            hole = prog.cylinder(ear.diameter / 2, 0.020)
            hole = prog.locate(hole, pos=(ear.pos[0], ear.pos[1], f_z_bot))
            body = prog.cut(body, hole)

    # ── 6. Connector receptacle (prebuilt if present) ──
    if servo.connector_pos is not None:
        body = _emit_servo_connector(prog, body, servo)

    prog.output_ref = body
    return prog


def scs0009_script(servo: ServoSpec) -> ShapeScript:
    """SCS0009 micro servo body (SG90-style) as ShapeScript.

    All geometry uses native ShapeScript ops — no PrebuiltOps.
    Shaft boss, ears, mounting holes, and connector are native ops.
    """
    prog = ShapeScript()

    body_x, body_y, body_z = servo.effective_body_dims
    r = 0.0010
    z_top = body_z / 2

    # ── 1. Main body (box + fillet Z edges) ──
    body = prog.box(body_x, body_y, body_z, tag="scs_body")
    body = prog.fillet_by_axis(body, "z", r)

    # ── 2. Output shaft boss (native) ──
    sx, sy, _sz = servo.shaft_offset
    boss_r = servo.shaft_boss_radius
    boss_h = servo.shaft_boss_height
    boss = prog.cylinder(boss_r, boss_h, align=Align3(z="min"))
    boss = prog.locate(boss, pos=(sx, sy, z_top))
    body = prog.fuse(body, boss)

    # ── 3. Mounting ears (native box) ──
    ear_ext = 0.00465
    ear_thick = 0.0025
    ear_total_x = body_x + 2 * ear_ext

    ear_top_z = z_top - 0.00775
    ear_bot_z = ear_top_z - ear_thick
    ear_cz = (ear_top_z + ear_bot_z) / 2

    ear = prog.box(ear_total_x, body_y, ear_thick)
    ear = prog.locate(ear, pos=(0, 0, ear_cz))
    body = prog.fuse(body, ear)

    # ── 4. Mounting holes (native cuts) ──
    if servo.mounting_ears:
        for mp in servo.mounting_ears:
            hole = prog.cylinder(mp.diameter / 2, 0.010)
            hole = prog.locate(hole, pos=(mp.pos[0], mp.pos[1], ear_cz))
            body = prog.cut(body, hole)

    # ── 5. Connector receptacle (prebuilt if present) ──
    if servo.connector_pos is not None:
        body = _emit_servo_connector(prog, body, servo)

    prog.output_ref = body
    return prog


# ── Helpers ──


def _group_ears_by_y_side(ears) -> dict[str, list]:
    """Group mounting ears into +Y and -Y sides."""
    sides: dict[str, list] = {}
    for ear in ears:
        side = "pos" if ear.pos[1] > 0 else "neg"
        sides.setdefault(side, []).append(ear)
    return sides


def _emit_servo_connector(prog: ShapeScript, body_ref, servo: ServoSpec):
    """Fuse connector receptacles onto the servo body via CallOp sub-programs.

    Mirrors _fuse_servo_connector logic: each non-permanent wire port gets
    a receptacle sub-program called and located at the wire port position.
    """
    from botcad.connectors import connector_spec
    from botcad.shapescript.emit_components import receptacle_script

    for wp in servo.wire_ports:
        if not wp.connector_type or wp.permanent:
            continue
        try:
            cspec = connector_spec(wp.connector_type)
        except KeyError:
            continue

        # Register sub-program once per connector type
        key = f"receptacle_{wp.connector_type}"
        if key not in prog.sub_programs:
            prog.sub_programs[key] = receptacle_script(cspec)

        cx, cy, cz = wp.pos

        mx, _my, mz = cspec.mating_direction
        if abs(mz) > 0.5:
            flip = 180 if cz < 0 else 0
            euler = (float(flip), 0.0, 90.0)
        elif abs(mx) > 0.5:
            euler = (0.0, -90.0 if cx < 0 else 90.0, 0.0)
        else:
            euler = (90.0 if cy < 0 else -90.0, 0.0, 0.0)

        connector = prog.call(key, tag="servo_connector")
        connector = prog.locate(connector, pos=(cx, cy, cz), euler_deg=euler)
        body_ref = prog.fuse(body_ref, connector)

    return body_ref
