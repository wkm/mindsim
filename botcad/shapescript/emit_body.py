"""Translate _make_body_solid logic into ShapeScript programs.

This module mirrors botcad/emit/cad.py:_make_body_solid() line-by-line,
but emits IR ops instead of calling build123d directly. The resulting
ShapeScript can be executed by the OCCT backend to produce the same solid.

The key invariant: for every bot, the ShapeScript path must produce body solids
whose volumes and bounding boxes match the direct build123d path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from botcad.shapescript.ops import ALIGN_CENTER, ALIGN_MIN_Z
from botcad.shapescript.program import ShapeScript

if TYPE_CHECKING:
    from botcad.skeleton import Body, Joint


def emit_body_ir(
    body: Body,
    parent_joint: Joint | None = None,
    wire_segments: tuple | None = None,
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
    from build123d import Location

    from botcad.bracket import (
        BracketSpec,
        coupler_solid,
        cradle_solid,
    )
    from botcad.component import BearingSpec, CameraSpec
    from botcad.emit.cad import _child_clearance_volume, _ensure_solid, _wire_channel
    from botcad.geometry import quat_to_euler, rotate_vec
    from botcad.skeleton import BodyShape, BracketStyle

    prog = ShapeScript()
    dims = body.dimensions

    # ── 1. Base shell (cad.py:748-801) ──

    if body.custom_solid is not None:
        shell = prog.prebuilt(_ensure_solid(body.custom_solid), tag="custom_solid")

    elif body.shape is BodyShape.CYLINDER:
        r = body.radius or dims[0] / 2
        h = body.width or dims[2]

        if body.is_wheel_body:
            # Wheels: centered on origin (hub at center)
            from botcad.emit.cad import _make_wheel_solid

            wheel_solid = _make_wheel_solid(r, h)
            shell = prog.prebuilt(wheel_solid, tag="wheel")

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
            if isinstance(mount.component, BearingSpec):
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
            pocket = prog.locate(pocket, pos=mount.resolved_pos)
            shell = prog.cut(shell, pocket)

            # Camera-specific cuts: lens aperture + ribbon cable exit
            if isinstance(mount.component, CameraSpec):
                shell = _emit_camera_cuts(prog, shell, mount, body.dimensions, dims)

    # ── 3. Union coupler (cad.py:827-843) ──

    bracket_spec = BracketSpec()
    if parent_joint is not None and parent_joint.bracket_style is BracketStyle.COUPLER:
        servo = parent_joint.servo
        quat = parent_joint.solved_servo_quat
        rotated_offset = rotate_vec(quat, servo.shaft_offset)
        center = (-rotated_offset[0], -rotated_offset[1], -rotated_offset[2])
        euler = quat_to_euler(quat)

        # Use .moved() to match direct path (cad.py:986) — .locate() mutates
        # @lru_cache'd shapes in-place. See memory/feedback_build123d_locate.md.
        coupler = coupler_solid(servo, bracket_spec)
        coupler = coupler.moved(Location(center, euler))
        coupler_ref = prog.prebuilt(coupler, tag="coupler")
        shell = prog.fuse(shell, coupler_ref)

    # ── 4. Bracket footprint cut + bracket union (cad.py:845-865) ──
    # Brackets are sub-programs: defined once per servo type, called at each joint.

    from botcad.shapescript.emit_bracket import (
        bracket_envelope_script,
        bracket_solid_script,
        cradle_envelope_script,
    )

    for joint in body.joints:
        servo = joint.servo
        center = joint.solved_servo_center
        euler = quat_to_euler(joint.solved_servo_quat)

        # Register sub-programs once per servo type (define once, use many)
        if joint.bracket_style is BracketStyle.COUPLER:
            env_key = f"cradle_env_{servo.name}"
            brk_key = f"cradle_{servo.name}"
            if env_key not in prog.sub_programs:
                prog.sub_programs[env_key] = cradle_envelope_script(servo, bracket_spec)
                # cradle_solid_script not yet ported — wrap as PrebuiltOp sub-program
                sub = ShapeScript()
                sub.output_ref = sub.prebuilt(
                    cradle_solid(servo, bracket_spec), tag="cradle"
                )
                prog.sub_programs[brk_key] = sub
        else:
            env_key = f"bracket_env_{servo.name}"
            brk_key = f"bracket_{servo.name}"
            if env_key not in prog.sub_programs:
                prog.sub_programs[env_key] = bracket_envelope_script(
                    servo, bracket_spec
                )
                prog.sub_programs[brk_key] = bracket_solid_script(
                    servo, bracket_spec
                )

        # Call sub-programs and locate at joint position
        env_ref = prog.call(env_key, tag=f"bracket_env_{joint.name}")
        env_ref = prog.locate(env_ref, pos=center, euler_deg=euler)
        brk_ref = prog.call(brk_key, tag=f"bracket_{joint.name}")
        brk_ref = prog.locate(brk_ref, pos=center, euler_deg=euler)
        shell = prog.cut(shell, env_ref)
        shell = prog.fuse(shell, brk_ref)

    # ── 5. Child clearance volumes (cad.py:867-873) ──


    for joint in body.joints:
        if joint.child is not None:
            clearance = _child_clearance_volume(joint.child, joint)
            if clearance is not None:
                clr_ref = prog.prebuilt(clearance, tag=f"clearance_{joint.name}")
                shell = prog.cut(shell, clr_ref)

    # ── 6. Wire channels (cad.py:875-881) ──


    if wire_segments:
        for seg, bus_type in wire_segments:
            channel = _wire_channel(seg, bus_type)
            if channel is not None:
                ch_ref = prog.prebuilt(channel, tag=f"wire_{bus_type}")
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


def _emit_camera_cuts(prog, shell, mount, body_dims, solved_dims):
    """Emit IR ops for camera lens aperture + ribbon cable slot.

    Mirrors cad.py:_cut_camera_features().
    """
    from botcad.component import BusType

    cd = mount.component.dimensions
    pos = mount.resolved_pos
    ins = mount.resolved_insertion_axis

    # Lens aperture — 8mm diameter cylinder punched through body wall
    aperture_r = 0.004
    wall_depth = max(solved_dims) + 0.002

    ax, ay, az = ins
    if abs(ax) > 0.5:
        euler = (0, 90, 0)
    elif abs(ay) > 0.5:
        euler = (90, 0, 0)
    else:
        euler = (0, 0, 0)

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
