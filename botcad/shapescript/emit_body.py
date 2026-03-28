"""Emit ShapeScript programs that build body solids.

This is the single source of truth for body geometry. Each body's parametric
definition (shell shape, component mounts, bracket cuts, wire channels) is
translated into ShapeScript IR ops, then executed by OcctBackend to produce
the final build123d Solid.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from botcad.shapescript.ops import ALIGN_CENTER, ALIGN_MIN_Z
from botcad.shapescript.program import ShapeScript

if TYPE_CHECKING:
    from botcad.skeleton import Body, Bot, Joint


def emit_body_ir(
    body: Body,
    parent_joint: Joint | None = None,
    wire_segments: tuple | None = None,
    bot: Bot | None = None,
) -> ShapeScript:
    """Emit a ShapeScript that builds the solid for a single body.

    This is a 1:1 translation of cad.py:_make_body_solid(). The result
    can be executed via OcctBackend to produce an identical solid.

    Args:
        body: The Body to build geometry for.
        parent_joint: The joint connecting this body to its parent (None for root).
        wire_segments: Tuple of (segment, bus_type) pairs for wire channel cutting.

    Returns:
        A ShapeScript whose output_ref is the final body solid.
    """
    from botcad.bracket import (
        BracketSpec,
    )
    from botcad.component import ComponentKind
    from botcad.emit.cad import _ensure_solid
    from botcad.geometry import quat_to_euler
    from botcad.skeleton import BodyShape, BracketStyle

    placements = bot.packing_result.placements if bot and bot.packing_result else {}

    prog = ShapeScript()
    dims = body.dimensions

    # ── 1. Base shell (cad.py:748-801) ──

    if body.custom_solid is not None:
        shell = prog.prebuilt(_ensure_solid(body.custom_solid), tag="custom_solid")

    elif body.shape is BodyShape.CYLINDER:
        r = body.radius or dims[0] / 2
        h = body.width or dims[2]

        if body.is_wheel_body:
            # Wheels: sub-program (define once per size, call via CallOp)
            from botcad.shapescript.emit_components import wheel_script

            wheel_key = f"wheel_{r:.6f}_{h:.6f}"
            if wheel_key not in prog.sub_programs:
                prog.sub_programs[wheel_key] = wheel_script(r, h)
            shell = prog.call(wheel_key, tag="wheel")

        elif parent_joint is not None:
            # Child cylinder: bottom face at origin, offset so z=0 is joint end
            shell = prog.cylinder(r, h, align=ALIGN_CENTER)
            shell = prog.locate(shell, pos=(0, 0, h / 2))
        else:
            shell = prog.cylinder(r, h, align=ALIGN_CENTER)

        # CRITICAL: orient Z to axis UNCONDITIONALLY after all sub-branches
        # if parent_joint is not None (cad.py:770-771)
        if parent_joint is not None:
            shell = _emit_orient_z_to_axis(
                prog, shell, parent_joint.axis, quat=body.frame_quat
            )

    elif body.shape is BodyShape.TUBE:
        r = body.outer_r or dims[0] / 2
        length = body.length or dims[2]
        shell = prog.cylinder(r, length, align=ALIGN_CENTER)
        shell = prog.locate(shell, pos=(0, 0, length / 2))

    elif body.shape is BodyShape.SPHERE:
        r = body.radius or dims[0] / 2
        shell = prog.sphere(r)

    elif body.shape is BodyShape.JAW:
        jw = body.jaw_width or dims[0]
        jt = body.jaw_thickness or dims[1]
        jl = body.jaw_length or dims[2]
        knuckle_h = 0.008  # thicker base for horn attachment
        # Knuckle at base (z=0 to z=knuckle_h)
        knuckle = prog.box(jw, jt * 2, knuckle_h, align=ALIGN_MIN_Z)
        # Jaw plate extending from knuckle to full length
        plate = prog.box(jw, jt, jl - knuckle_h, align=ALIGN_MIN_Z)
        plate = prog.locate(plate, pos=(0, 0, knuckle_h))
        shell = prog.fuse(knuckle, plate)

    else:
        # BOX (default)
        shell = prog.box(dims[0], dims[1], dims[2], align=ALIGN_CENTER)

    # ── 2. Cut component pockets (cad.py:802-825) ──

    if not body.is_wheel_body:
        for mount in body.mounts:
            cd = mount.placed_dimensions
            if mount.component.kind == ComponentKind.BEARING:
                # Bearings: cylindrical pocket with tolerance
                tol = 0.0005  # 0.5mm total diameter clearance
                pocket_r = mount.component.od / 2 + tol / 2
                pocket_h = mount.component.width + 0.001  # through-cut margin
                pocket = prog.cylinder(pocket_r, pocket_h, align=ALIGN_CENTER)
            else:
                # Default: box pocket
                pocket = prog.box(
                    cd[0] + 0.0005,
                    cd[1] + 0.0005,
                    cd[2] + 0.0005,
                    align=ALIGN_CENTER,
                )
            mount_pos = (
                placements[mount].pose.pos
                if mount in placements
                else mount.resolved_pos
            )
            pocket = prog.locate(pocket, pos=mount_pos)
            shell = prog.cut(shell, pocket)

            # Camera-specific cuts: lens aperture + ribbon cable exit
            if mount.component.kind == ComponentKind.CAMERA:
                shell = _emit_camera_cuts(
                    prog, shell, mount, body.dimensions, dims, placements
                )

    # ── 3. Union coupler (cad.py:827-843) ──

    bracket_spec = BracketSpec()
    if parent_joint is not None and parent_joint.bracket_style is BracketStyle.COUPLER:
        servo = parent_joint.servo
        # Coupler is built in shaft-centered frame; child body origin = shaft
        # position (MuJoCo places child at parent_joint.pos).  Place at origin
        # with servo orientation only.
        pj_pose = placements.get(parent_joint)
        quat = pj_pose.pose.quat if pj_pose else parent_joint.solved_servo_quat
        center = (0.0, 0.0, 0.0)
        euler = quat_to_euler(quat)

        # Sub-program produces coupler in servo-local frame; locate moves it.
        from botcad.bracket import coupler_solid as coupler_solid_script

        coupler_key = f"coupler_{servo.name}"
        if coupler_key not in prog.sub_programs:
            prog.sub_programs[coupler_key] = coupler_solid_script(servo, bracket_spec)
        coupler_ref = prog.call(coupler_key, tag="coupler")
        coupler_ref = prog.locate(coupler_ref, pos=center, euler_deg=euler)
        shell = prog.fuse(shell, coupler_ref)

    # ── 4. Bracket footprint cut + bracket union (cad.py:845-865) ──
    # Brackets are sub-programs: defined once per servo type, called at each joint.

    from botcad.bracket import (
        bracket_insertion_channel as bracket_insertion_channel_script,
    )
    from botcad.bracket import (
        bracket_solid as bracket_solid_script,
    )
    from botcad.bracket import (
        cradle_insertion_channel as cradle_insertion_channel_script,
    )
    from botcad.bracket import (
        cradle_solid as cradle_solid_script,
    )

    for joint in body.joints:
        servo = joint.servo
        j_pose = placements.get(joint)
        center = j_pose.pose.pos if j_pose else joint.solved_servo_center
        j_quat = j_pose.pose.quat if j_pose else joint.solved_servo_quat
        euler = quat_to_euler(j_quat)

        # Register sub-programs once per servo type (define once, use many)
        if joint.bracket_style is BracketStyle.COUPLER:
            env_key = f"cradle_env_{servo.name}"
            brk_key = f"cradle_{servo.name}"
            if env_key not in prog.sub_programs:
                prog.sub_programs[env_key] = cradle_insertion_channel_script(
                    servo, bracket_spec
                )
                prog.sub_programs[brk_key] = cradle_solid_script(servo, bracket_spec)
        else:
            env_key = f"bracket_env_{servo.name}"
            brk_key = f"bracket_{servo.name}"
            if env_key not in prog.sub_programs:
                prog.sub_programs[env_key] = bracket_insertion_channel_script(
                    servo, bracket_spec
                )
                prog.sub_programs[brk_key] = bracket_solid_script(servo, bracket_spec)

        # Call sub-programs and locate at joint position
        env_ref = prog.call(env_key, tag=f"bracket_env_{joint.name}")
        env_ref = prog.locate(env_ref, pos=center, euler_deg=euler)
        brk_ref = prog.call(brk_key, tag=f"bracket_{joint.name}")
        brk_ref = prog.locate(brk_ref, pos=center, euler_deg=euler)
        shell = prog.cut(shell, env_ref)
        shell = prog.fuse(shell, brk_ref)

    # ── 5. Child clearance volumes (cad.py:867-873) ──

    from botcad.shapescript.emit_components import emit_child_clearance

    for joint in body.joints:
        if joint.child is not None:
            clr_ref = emit_child_clearance(prog, joint.child, joint)
            if clr_ref is not None:
                shell = prog.cut(shell, clr_ref)

    # ── 6. Wire channels (cad.py:875-881) ──

    from botcad.shapescript.emit_components import emit_wire_channel

    if wire_segments:
        for seg, bus_type in wire_segments:
            ch_ref = emit_wire_channel(prog, seg, bus_type)
            if ch_ref is not None:
                shell = prog.cut(shell, ch_ref)

    prog.output_ref = shell
    return prog


def _emit_orient_z_to_axis(
    prog: ShapeScript,
    shape_ref,
    axis: tuple[float, float, float],
    quat: tuple[float, float, float, float] | None = None,
):
    """Emit a LocateOp that rotates from Z-up to the given axis.

    Mirrors cad.py:_orient_z_to_axis — if quat is identity, returns
    shape_ref unchanged (no op emitted).
    """
    from botcad.geometry import quat_to_euler

    if quat is not None:
        w, x, y, z = quat
        if abs(w - 1.0) < 1e-9 and abs(x) + abs(y) + abs(z) < 1e-9:
            return shape_ref  # identity — no rotation needed
        euler = quat_to_euler(quat)
        return prog.locate(shape_ref, pos=(0, 0, 0), euler_deg=euler)

    from botcad.emit.cad import _axis_to_quat

    q = _axis_to_quat(axis)
    w, x, y, z = q
    if abs(w - 1.0) < 1e-9 and abs(x) + abs(y) + abs(z) < 1e-9:
        return shape_ref  # identity
    euler = quat_to_euler(q)
    return prog.locate(shape_ref, pos=(0, 0, 0), euler_deg=euler)


def _emit_camera_cuts(prog, shell, mount, body_dims, solved_dims, placements=None):
    """Emit IR ops for camera lens aperture + ribbon cable slot.

    Mirrors cad.py:_cut_camera_features().
    """
    from botcad.component import BusType
    from botcad.geometry import pose_transform_dir

    cd = mount.component.dimensions
    if placements and mount in placements:
        pos = placements[mount].pose.pos
        ins = pose_transform_dir(placements[mount].pose, (0.0, 0.0, 1.0))
    else:
        pos = mount.resolved_pos
        ins = mount.resolved_insertion_axis

    # Lens aperture — 8mm diameter cylinder punched through body wall
    aperture_r = 0.004
    wall_depth = max(solved_dims) + 0.002

    ax, ay, _az = ins
    if abs(ax) > 0.5:
        _euler = (0, 90, 0)
    elif abs(ay) > 0.5:
        _euler = (90, 0, 0)
    else:
        _euler = (0, 0, 0)

    aperture = prog.cylinder(aperture_r, wall_depth, align=ALIGN_CENTER)
    # NOTE: The direct path does two .locate() calls (rotation then position),
    # but .locate() replaces the location, so the rotation is lost.
    # We match that behavior by only emitting the position LocateOp.
    aperture = prog.locate(aperture, pos=pos)
    shell = prog.cut(shell, aperture)

    # Ribbon cable exit slot
    ribbon_w = 0.017
    ribbon_h = 0.002

    csi_ports = [wp for wp in mount.component.wire_ports if wp.bus_type == BusType.CSI]
    if csi_ports:
        wp = csi_ports[0]
        slot_pos = (pos[0] + wp.pos[0], pos[1] + wp.pos[1], pos[2] + wp.pos[2])
    else:
        slot_pos = (pos[0], pos[1], pos[2] - cd[1] / 2)

    if abs(ax) > 0.5:
        ribbon_slot = prog.box(wall_depth, ribbon_w, ribbon_h, align=ALIGN_CENTER)
    elif abs(ay) > 0.5:
        ribbon_slot = prog.box(ribbon_w, wall_depth, ribbon_h, align=ALIGN_CENTER)
    else:
        ribbon_slot = prog.box(ribbon_w, ribbon_h, wall_depth, align=ALIGN_CENTER)

    ribbon_slot = prog.locate(ribbon_slot, pos=slot_pos)
    shell = prog.cut(shell, ribbon_slot)

    return shell
