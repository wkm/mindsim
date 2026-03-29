"""MuJoCo XML emitter — generates bot.xml + scene.xml from a Bot skeleton.

Follows existing conventions:
- Root body named "base" with freejoint
- Joints named "{name}" with hinge type
- Actuators: position actuators with kp/damping from servo specs
- Sensors: jointpos + jointvel per joint, gyro + accelerometer on IMU
- Camera: "camera_1_cam" at camera mount location
- Contact excludes for all parent-child body pairs
"""

from __future__ import annotations

from pathlib import Path
from xml.dom import minidom
from xml.etree.ElementTree import Element, SubElement, tostring

from botcad.colors import (
    COLOR_STRUCTURE_DARK,
    COLOR_STRUCTURE_HORN_DISC,
    COLOR_WIRE_CSI,
    COLOR_WIRE_DEFAULT,
    COLOR_WIRE_POWER,
    COLOR_WIRE_UART,
)
from botcad.component import BusType
from botcad.component import Vec3 as Vec3Type
from botcad.fasteners import fastener_key as _hw_key
from botcad.fasteners import fastener_stl_stem as _hw_name
from botcad.geometry import Pose, Quat, fastener_pose, rotate_vec
from botcad.skeleton import BaseType, Body, BodyKind, Bot, Joint


def _relative_pos(pb_world: Vec3Type, parent_world: Vec3Type) -> Vec3Type:
    """Convert a purchased body's world position to parent-body-relative position."""
    return (
        pb_world[0] - parent_world[0],
        pb_world[1] - parent_world[1],
        pb_world[2] - parent_world[2],
    )


def generate_mujoco_xml(bot: Bot) -> str:
    """Return the bot.xml MuJoCo XML string without writing to disk.

    This is the pure-function entry point used by the API server.
    """
    return _build_bot_xml(bot)


def generate_scene_xml(bot: Bot) -> str:
    """Return the scene.xml MuJoCo XML string without writing to disk."""
    return _build_scene_xml(bot)


def emit_mujoco(bot: Bot, output_dir: Path) -> None:
    """Generate bot.xml and scene.xml in output_dir."""
    bot_xml = generate_mujoco_xml(bot)
    scene_xml = generate_scene_xml(bot)

    (output_dir / "bot.xml").write_text(bot_xml)
    (output_dir / "scene.xml").write_text(scene_xml)

    print(f"MuJoCo: wrote bot.xml + scene.xml to {output_dir}")


def _build_bot_xml(bot: Bot) -> str:
    root = Element("mujoco", model=bot.name)

    # Compiler
    SubElement(root, "compiler", angle="radian", meshdir="meshes")

    # Defaults
    default = SubElement(root, "default")
    SubElement(default, "motor", ctrlrange="-1 1", gear="1.0")

    # Assets (mesh references) — structural bodies only
    asset = SubElement(root, "asset")
    structural_bodies = [b for b in bot.all_bodies if b.kind != BodyKind.PURCHASED]
    for body in structural_bodies:
        mesh_name = f"{body.name}_mesh"
        SubElement(
            asset,
            "mesh",
            name=mesh_name,
            file=f"{body.name}.stl",
            scale="1 1 1",  # STL already in meters
        )

    # Component mesh assets (Pi, battery, camera, etc.)
    for body in structural_bodies:
        if body.is_wheel_body:
            continue
        for mount in body.mounts:
            mesh_name = f"comp_{body.name}_{mount.label}_mesh"
            SubElement(
                asset,
                "mesh",
                name=mesh_name,
                file=f"comp_{body.name}_{mount.label}.stl",
                scale="1 1 1",
            )

    # Servo mesh assets (one per unique servo model)
    seen_servos: set[str] = set()
    for joint in bot.all_joints:
        if joint.servo.name not in seen_servos:
            seen_servos.add(joint.servo.name)
            SubElement(
                asset,
                "mesh",
                name=f"servo_{joint.servo.name}_mesh",
                file=f"servo_{joint.servo.name}.stl",
                scale="1 1 1",
            )

    # Hardware mesh assets (one per unique designation+head_type)
    seen_hw: set[tuple[str, str]] = set()
    for body in structural_bodies:
        for joint in body.joints:
            for ear in joint.servo.mounting_ears:
                k = _hw_key(ear)
                if k not in seen_hw:
                    seen_hw.add(k)
                    SubElement(
                        asset,
                        "mesh",
                        name=f"{_hw_name(ear)}_mesh",
                        file=f"{_hw_name(ear)}.stl",
                        scale="1 1 1",
                    )
            for mp in joint.servo.horn_mounting_points:
                k = _hw_key(mp)
                if k not in seen_hw:
                    seen_hw.add(k)
                    SubElement(
                        asset,
                        "mesh",
                        name=f"{_hw_name(mp)}_mesh",
                        file=f"{_hw_name(mp)}.stl",
                        scale="1 1 1",
                    )
        for mount in body.mounts:
            for mp in mount.component.mounting_points:
                k = _hw_key(mp)
                if k not in seen_hw:
                    seen_hw.add(k)
                    SubElement(
                        asset,
                        "mesh",
                        name=f"{_hw_name(mp)}_mesh",
                        file=f"{_hw_name(mp)}.stl",
                        scale="1 1 1",
                    )

    # Horn mesh assets
    for body in structural_bodies:
        for joint in body.joints:
            from botcad.bracket import horn_disc_params

            if horn_disc_params(joint.servo):
                SubElement(
                    asset,
                    "mesh",
                    name=f"horn_{joint.name}_mesh",
                    file=f"horn_{joint.name}.stl",
                    scale="1 1 1",
                )

    # Wire mesh assets
    for route in bot.wire_routes:
        for i, seg in enumerate(route.segments):
            if seg.straight_length >= 0.001:
                mesh_name = f"wire_{route.label}_{seg.body_name}_{i}_mesh"
                SubElement(
                    asset,
                    "mesh",
                    name=mesh_name,
                    file=f"wire_{route.label}_{seg.body_name}_{i}.stl",
                    scale="1 1 1",
                )

    # Worldbody
    worldbody = SubElement(root, "worldbody")
    if bot.root is not None:
        _emit_body_tree(worldbody, bot.root, bot, is_root=True)

    # Contact excludes — bodies within 3 joints of each other collide due to
    # bracket/coupler geometry, not real contact. Exclude up to 3 joints apart
    # to cover compact mechanisms like wrist-roll + gripper chains.
    contact = SubElement(root, "contact")
    exclude_pairs: set[tuple[str, str]] = set()
    for body in structural_bodies:
        for j1 in body.joints:
            if j1.child is None:
                continue
            # 1 joint apart
            exclude_pairs.add((body.name, j1.child.name))
            for j2 in j1.child.joints:
                if j2.child is None:
                    continue
                # 2 joints apart
                exclude_pairs.add((body.name, j2.child.name))
                for j3 in j2.child.joints:
                    if j3.child is None:
                        continue
                    # 3 joints apart
                    exclude_pairs.add((body.name, j3.child.name))
    for b1, b2 in sorted(exclude_pairs):
        SubElement(contact, "exclude", body1=b1, body2=b2)

    # Actuators
    actuator_el = SubElement(root, "actuator")
    for joint in bot.all_joints:
        if joint.servo.continuous:
            # Continuous rotation motor — gear maps ctrl=1 to stall torque.
            # Joint damping is kept low (bearing friction); the servo's
            # torque-speed curve is modeled by gear + controller, not damping.
            SubElement(
                actuator_el,
                "motor",
                name=f"{joint.name}_motor",
                joint=joint.name,
                gear=f"{joint.servo.stall_torque:.2f}",
                ctrlrange="-1 1",
            )
        elif joint.grip:
            # Gripper: position actuator with reduced gain and force limit
            # for compliant grasping (jaw yields on contact).
            lo, hi = joint.effective_range
            grip_force = joint.servo.stall_torque * 0.2
            SubElement(
                actuator_el,
                "position",
                name=f"{joint.name}_motor",
                joint=joint.name,
                kp=f"{joint.servo.kp * 0.3:.1f}",
                ctrlrange=f"{lo:.4f} {hi:.4f}",
                forcerange=f"{-grip_force:.2f} {grip_force:.2f}",
            )
        else:
            # Position servo: position actuator with kp from servo specs.
            # forcerange limits output to actual servo stall torque so
            # reaction torques on the body are physically realistic.
            lo, hi = joint.effective_range
            st = joint.servo.stall_torque
            SubElement(
                actuator_el,
                "position",
                name=f"{joint.name}_motor",
                joint=joint.name,
                kp=f"{joint.servo.kp:.1f}",
                ctrlrange=f"{lo:.4f} {hi:.4f}",
                forcerange=f"{-st:.2f} {st:.2f}",
            )

    # Sensors
    sensor_el = SubElement(root, "sensor")
    for joint in bot.all_joints:
        SubElement(
            sensor_el,
            "jointpos",
            name=f"{joint.name}_pos_sensor",
            joint=joint.name,
        )
        SubElement(
            sensor_el,
            "jointvel",
            name=f"{joint.name}_vel_sensor",
            joint=joint.name,
        )

    # IMU sensors on base
    _add_imu_sensors(sensor_el, bot)

    return _prettify(root)


def _emit_body_tree(
    parent_el: Element,
    body: Body,
    bot: Bot,
    parent_joint: Joint | None = None,
    is_root: bool = False,
) -> None:
    """Recursively emit body elements.

    Each body in MuJoCo contains the joint that connects it to its parent.
    The recursive structure is:

        <body name="parent">
          <body name="child" pos="(from parent_joint.pos)">
            <joint name="..." />   <!-- connects parent → child -->
            <geom ... />
            <body name="grandchild" pos="(from child_joint.pos)">
              <joint name="..." /> <!-- connects child → grandchild -->
              ...
    """
    attribs: dict[str, str] = {"name": body.name}

    if is_root:
        if bot.base_type is BaseType.FIXED:
            # Fixed base: sit on ground plane (half body height + small gap)
            base_z = body.dimensions[2] / 2 + 0.001
            attribs["pos"] = f"0 0 {base_z:.4f}"
        else:
            ground_clearance = _estimate_ground_clearance(body, bot)
            attribs["pos"] = f"0 0 {ground_clearance:.4f}"
    elif parent_joint is not None:
        if body.is_wheel_body:
            # Wheel child body: offset outward from servo along joint axis
            # so the wheel clears the servo body + shaft boss.
            wheel_offset = parent_joint.wheel_outboard_offset()
            ax, ay, az = parent_joint.axis
            child_pos = (
                parent_joint.pos[0] + ax * wheel_offset,
                parent_joint.pos[1] + ay * wheel_offset,
                parent_joint.pos[2] + az * wheel_offset,
            )
            attribs["pos"] = _fmt_vec3(child_pos)
        else:
            attribs["pos"] = _fmt_vec3(parent_joint.pos)

    body_el = SubElement(parent_el, "body", **attribs)

    # Root gets a freejoint (unless fixed base)
    if is_root and bot.base_type is not BaseType.FIXED:
        SubElement(body_el, "freejoint", name="root")

    # Inertial
    SubElement(
        body_el,
        "inertial",
        pos=_fmt_vec3(body.solved_com),
        mass=f"{body.solved_mass:.6f}",
        fullinertia=_fmt_inertia(body.solved_inertia),
    )

    # Joint connecting this body to its parent (if not root)
    if parent_joint is not None:
        lo, hi = parent_joint.effective_range
        # Wheel joints use low damping (bearing friction only); the servo's
        # torque-speed characteristic is handled by the motor actuator + gear.
        # Position servo joints use the full servo damping.
        if parent_joint.servo.continuous:
            damping = 0.01
        else:
            damping = parent_joint.servo.damping
        # For wheel bodies offset outward, the joint anchor must point back
        # to the shaft position (compensate the body offset).
        if body.is_wheel_body:
            wheel_offset = parent_joint.wheel_outboard_offset()
            ax, ay, az = parent_joint.axis
            joint_pos = _fmt_vec3(
                (-ax * wheel_offset, -ay * wheel_offset, -az * wheel_offset)
            )
        else:
            joint_pos = "0 0 0"

        joint_attribs: dict[str, str] = {
            "name": parent_joint.name,
            "type": "hinge",
            "pos": joint_pos,
            "axis": _fmt_vec3(parent_joint.axis),
            "damping": f"{damping:.4f}",
        }
        if not parent_joint.servo.continuous:
            joint_attribs["range"] = f"{lo:.4f} {hi:.4f}"
        SubElement(body_el, "joint", **joint_attribs)

    # Horn disc geom (purchased part visualization on child bodies)
    # The horn belongs to the parent structural body in the assembly tree,
    # but is placed as a geom on the *child* MuJoCo body (this body).
    # Read position from the purchased horn Body, converting world→relative
    # to this child body's world position.
    if parent_joint is not None:
        # Find the horn body for this joint (parented to the parent structural body)
        horn_key = f"horn_{parent_joint.name}"
        horn_pb = None
        for pb in bot.all_bodies:
            if pb.kind == BodyKind.PURCHASED and pb.name == horn_key:
                horn_pb = pb
                break

        if horn_pb is not None:
            disc_pos = _relative_pos(horn_pb.world_pos, body.world_pos)

            disc_attribs: dict[str, str] = {
                "name": f"{parent_joint.name}_horn_disc",
                "type": "mesh",
                "mesh": f"horn_{parent_joint.name}_mesh",
                "pos": _fmt_vec3(disc_pos),
                "rgba": COLOR_STRUCTURE_HORN_DISC.with_alpha(0.9).rgba_str,
                "contype": "0",
                "conaffinity": "0",
                "group": "1",
            }

            # Rotation is already baked into the horn STL (it was oriented in cad.py)
            # So we don't need quat here — the STL is already in the correct
            # orientation relative to the disc position.
            SubElement(body_el, "geom", **disc_attribs)

    # Visual geom (mesh) — for wheel bodies, mesh is visual-only;
    # a primitive cylinder handles collision instead.
    mesh_name = f"{body.name}_mesh"
    mesh_attribs: dict[str, str] = {
        "type": "mesh",
        "mesh": mesh_name,
        "rgba": _body_color(body),
    }
    if body.is_wheel_body:
        # Mesh is visual only; cylinder primitive handles collision
        mesh_attribs["contype"] = "0"
        mesh_attribs["conaffinity"] = "0"
    else:
        mesh_attribs["contype"] = "1"
        mesh_attribs["conaffinity"] = "1"
    SubElement(body_el, "geom", **mesh_attribs)

    # Wheel bodies get a primitive cylinder geom for reliable rolling contact
    if body.is_wheel_body:
        r = body.radius or body.dimensions[0] / 2
        half_w = (body.width or body.dimensions[2]) / 2
        # Cylinder default axis is Z; rotate to match body frame orientation
        cyl_attribs: dict[str, str] = {
            "type": "cylinder",
            "size": f"{r} {half_w}",
            "rgba": COLOR_STRUCTURE_DARK.with_alpha(0.0).rgba_str,
            "contype": "1",
            "conaffinity": "1",
            "friction": "1.5 0.005 0.0001",
        }
        w, qx, qy, qz = body.frame_quat
        if abs(w - 1.0) > 1e-9 or abs(qx) + abs(qy) + abs(qz) > 1e-9:
            cyl_attribs["quat"] = _fmt_quat(body.frame_quat)
        SubElement(body_el, "geom", **cyl_attribs)

    # Camera mount
    _emit_camera(body_el, body, bot)

    # IMU site on root body
    if is_root:
        SubElement(
            body_el,
            "site",
            name="imu",
            pos="0 0 0",
            size="0.005",
        )

    # Mounted component visualization — meshes at positions from purchased bodies
    _emit_mounted_components(body_el, body, bot, is_wheel=body.is_wheel_body)

    # Servo visualization geoms — read positions from purchased Body instances
    # Build lookup of purchased bodies parented to this structural body
    purchased = [
        pb
        for pb in bot.all_bodies
        if pb.kind == BodyKind.PURCHASED and pb.parent_body_name == body.name
    ]
    servo_bodies = {pb.name: pb for pb in purchased if pb.name.startswith("servo_")}

    joint_placements: dict[str, tuple] = {}
    for joint in body.joints:
        servo_key = f"servo_{joint.name}"
        sb = servo_bodies.get(servo_key)
        if sb is not None:
            center = _relative_pos(sb.world_pos, body.world_pos)
            quat = sb.world_quat
        else:
            # Fallback (should not happen after solve)
            center = joint.solved_servo_center
            quat = joint.solved_servo_quat
        joint_placements[joint.name] = (center, quat)
        SubElement(
            body_el,
            "geom",
            name=f"{joint.name}_servo",
            type="mesh",
            mesh=f"servo_{joint.servo.name}_mesh",
            pos=_fmt_vec3(center),
            quat=_fmt_quat(quat),
            rgba=COLOR_STRUCTURE_DARK.rgba_str,
            contype="0",
            conaffinity="0",
            group="1",
        )

    # Mounting hardware visualization — screws at ear/mount positions
    _emit_mounting_hardware(body_el, body, joint_placements, bot)

    # Wire route visualization — segments on their respective bodies
    _emit_body_wires(body_el, body, bot)

    # Recurse: each child joint creates a child body
    for joint in body.joints:
        if joint.child is not None:
            _emit_body_tree(
                body_el, joint.child, bot, parent_joint=joint, is_root=False
            )


def _emit_mounted_components(
    parent_el: Element, body: Body, bot: Bot, is_wheel: bool = False
) -> None:
    """Emit visualization geoms for all mounted components (Pi, battery, camera, etc.).

    Positions are read from purchased Body instances (world_pos converted to
    parent-body-relative), not from mount.resolved_pos directly.
    """
    if is_wheel:
        return  # wheel mesh already represents the component

    # Build lookup of component purchased bodies for this structural body
    comp_bodies: dict[str, Body] = {}
    for pb in bot.all_bodies:
        if (
            pb.kind == BodyKind.PURCHASED
            and pb.parent_body_name == body.name
            and pb.name.startswith("comp_")
        ):
            comp_bodies[pb.name] = pb

    for mount in body.mounts:
        comp = mount.component
        r, g, b, a = (
            comp.default_material.color
            if comp.default_material
            else (0.541, 0.608, 0.659, 1.0)
        )
        comp_key = f"comp_{body.name}_{mount.label}"
        mesh_name = f"{comp_key}_mesh"

        # Position from purchased Body, fall back to mount.resolved_pos
        cb = comp_bodies.get(comp_key)
        if cb is not None:
            pos = _relative_pos(cb.world_pos, body.world_pos)
        else:
            pos = mount.resolved_pos

        # Component meshes are in component-local frame; apply placement
        # rotation so the mesh is oriented correctly in the body.
        placements = bot.packing_result.placements if bot.packing_result else {}
        geom_attribs: dict[str, str] = {
            "name": comp_key,
            "type": "mesh",
            "mesh": mesh_name,
            "pos": _fmt_vec3(pos),
            "rgba": f"{r:.4f} {g:.4f} {b:.4f} {a:.4f}",
            "contype": "0",
            "conaffinity": "0",
            "group": "1",
        }
        if mount in placements:
            q = placements[mount].pose.quat
            if q != (1.0, 0.0, 0.0, 0.0):
                geom_attribs["quat"] = _fmt_quat(q)
        SubElement(parent_el, "geom", **geom_attribs)


def _emit_mounting_hardware(
    body_el: Element,
    body: Body,
    joint_placements: dict[str, tuple],
    bot: Bot,
) -> None:
    """Emit mesh geoms at screw/mounting positions.

    Servo screw positions are derived from joint_placements (which come from
    purchased Body world_pos). Component screw positions use the purchased
    component Body's world_pos.
    """
    from botcad.colors import COLOR_METAL_FASTENER

    _SCREW_RGBA = COLOR_METAL_FASTENER.with_alpha(0.9).rgba_str

    def _screw_attribs(name: str, mesh: str, pos: str, quat: Quat):
        """Common geom attributes for a fastener, including orientation."""
        return {
            "name": name,
            "type": "mesh",
            "mesh": mesh,
            "pos": pos,
            "quat": _fmt_quat(quat),
            "rgba": _SCREW_RGBA,
            "contype": "0",
            "conaffinity": "0",
            "group": "1",
        }

    # Servo mounting ears (bracket screw holes on this body's joints)
    for joint in body.joints:
        center, quat = joint_placements[joint.name]
        servo_pose = Pose(center, quat)
        for ear in joint.servo.mounting_ears:
            fp = fastener_pose(servo_pose, ear)
            world_pos = _add_vec3(center, rotate_vec(quat, ear.pos))
            SubElement(
                body_el,
                "geom",
                **_screw_attribs(
                    f"screw_{joint.name}_{ear.label}",
                    f"{_hw_name(ear)}_mesh",
                    _fmt_vec3(world_pos),
                    fp.quat,
                ),
            )

        # Horn mounting points (output face screw holes)
        for mp in joint.servo.horn_mounting_points:
            fp = fastener_pose(servo_pose, mp)
            world_pos = _add_vec3(center, rotate_vec(quat, mp.pos))
            SubElement(
                body_el,
                "geom",
                **_screw_attribs(
                    f"horn_{joint.name}_{mp.label}",
                    f"{_hw_name(mp)}_mesh",
                    _fmt_vec3(world_pos),
                    fp.quat,
                ),
            )

        # Rear horn mounting points (blind side screw holes)
        for mp in joint.servo.rear_horn_mounting_points:
            fp = fastener_pose(servo_pose, mp)
            world_pos = _add_vec3(center, rotate_vec(quat, mp.pos))
            SubElement(
                body_el,
                "geom",
                **_screw_attribs(
                    f"rear_{joint.name}_{mp.label}",
                    f"{_hw_name(mp)}_mesh",
                    _fmt_vec3(world_pos),
                    fp.quat,
                ),
            )

    # Component mount points (screw holes on mounted components)
    # Mount points are defined in component-local space. body.to_body_frame()
    # applies the body's frame_quat (e.g. Z→joint_axis for cylinders) so
    # positions and axes land in the correct orientation automatically.
    # Component base position comes from purchased Body (world→relative).
    comp_bodies: dict[str, Body] = {}
    for pb in bot.all_bodies:
        if (
            pb.kind == BodyKind.PURCHASED
            and pb.parent_body_name == body.name
            and pb.name.startswith("comp_")
        ):
            comp_bodies[pb.name] = pb

    for mount in body.mounts:
        comp_key = f"comp_{body.name}_{mount.label}"
        cb = comp_bodies.get(comp_key)
        if cb is not None:
            base_pos = _relative_pos(cb.world_pos, body.world_pos)
        else:
            base_pos = mount.resolved_pos

        mount_pose = bot.packing_result.placements[mount].pose
        for mp in mount.component.mounting_points:
            fp = fastener_pose(mount_pose, mp)
            mp_pos = body.to_body_frame(mount.rotate_point(mp.pos))
            pos = _add_vec3(base_pos, mp_pos)
            SubElement(
                body_el,
                "geom",
                **_screw_attribs(
                    f"mount_{body.name}_{mount.label}_{mp.label}",
                    f"{_hw_name(mp)}_mesh",
                    _fmt_vec3(pos),
                    fp.quat,
                ),
            )


from botcad.geometry import add_vec3 as _add_vec3  # noqa: E402

# Camera xyaxes by mount position.
# "x1 x2 x3 y1 y2 y3" — camera X and Y axes in world frame; looks along -(X×Y).
# Positions correspond to mount face normals defined by DIR_* in geometry.py.
_CAMERA_XYAXES: dict[str, str] = {
    "front": "1 0 0 0 0 1",  # look +Y, up +Z
    "back": "-1 0 0 0 0 1",  # look -Y, up +Z
    "left": "0 1 0 0 0 1",  # look -X, up +Z
    "right": "0 -1 0 0 0 1",  # look +X, up +Z
    "top": "1 0 0 0 1 0",  # look +Z, up +Y
    "bottom": "1 0 0 0 -1 0",  # look -Z, up -Y
}


def _camera_xyaxes(position: str) -> str:
    """MuJoCo camera xyaxes for a given mount position."""
    return _CAMERA_XYAXES.get(position, "1 0 0 0 0 1")


def _emit_camera(parent_el: Element, body: Body, bot: Bot) -> None:
    """Emit camera element if this body has a camera mounted.

    Camera position is read from the purchased Body instance.
    """
    from botcad.component import ComponentKind

    # Build lookup of component purchased bodies for this structural body
    comp_bodies: dict[str, Body] = {}
    for pb in bot.all_bodies:
        if (
            pb.kind == BodyKind.PURCHASED
            and pb.parent_body_name == body.name
            and pb.name.startswith("comp_")
        ):
            comp_bodies[pb.name] = pb

    for mount in body.mounts:
        if mount.component.kind == ComponentKind.CAMERA:
            cam = mount.component
            comp_key = f"comp_{body.name}_{mount.label}"
            cb = comp_bodies.get(comp_key)
            if cb is not None:
                pos = _relative_pos(cb.world_pos, body.world_pos)
            else:
                pos = mount.resolved_pos
            # MuJoCo cameras look along local -Z with +Y up.
            # Use xyaxes to orient the camera to look along the mount
            # face normal with +Z as world-up.
            xyaxes = _camera_xyaxes(mount.position)
            SubElement(
                parent_el,
                "camera",
                name=f"{mount.label}_cam",
                fovy=f"{cam.fov:.1f}",
                pos=_fmt_vec3(pos),
                xyaxes=xyaxes,
            )
            return


_WIRE_COLORS = {
    BusType.UART_HALF_DUPLEX: COLOR_WIRE_UART.rgba_str,
    BusType.CSI: COLOR_WIRE_CSI.rgba_str,
    BusType.POWER: COLOR_WIRE_POWER.rgba_str,
}
_WIRE_RADIUS = {
    BusType.UART_HALF_DUPLEX: "0.0009",  # 0.9mm — servo bus (channel is 1.5mm)
    BusType.CSI: "0.0018",  # 1.8mm — CSI ribbon (channel is 3mm)
    BusType.POWER: "0.0012",  # 1.2mm — power cable (channel is 2mm)
}


def _emit_body_wires(body_el: Element, body: Body, bot: Bot) -> None:
    """Emit wire segments that belong to this body.

    Each segment's body_name is checked against the current body — only
    segments belonging to this body become geoms here. Coordinates are
    already in body-local frame from the routing solver.
    """
    for route in bot.wire_routes:
        color = _WIRE_COLORS.get(route.bus_type, COLOR_WIRE_DEFAULT.rgba_str)
        for i, seg in enumerate(route.segments):
            if seg.body_name != body.name:
                continue
            # Skip degenerate segments (endpoints too close)
            if seg.straight_length < 0.001:
                continue

            # Wires were exported at their coordinates in cad.py
            # So MuJoCo geom pos should be 0,0,0
            SubElement(
                body_el,
                "geom",
                name=f"wire_{route.label}_{body.name}_{i}",
                type="mesh",
                mesh=f"wire_{route.label}_{body.name}_{i}_mesh",
                pos="0 0 0",
                rgba=color,
                contype="0",
                conaffinity="0",
                group="2",
            )


def _add_imu_sensors(sensor_el: Element, bot: Bot) -> None:
    """Add IMU sensors (gyro + accelerometer) on the base body."""
    SubElement(sensor_el, "gyro", name="imu_gyro", site="imu")
    SubElement(sensor_el, "accelerometer", name="imu_accel", site="imu")


def _estimate_ground_clearance(root_body: Body, bot: Bot) -> float:
    """Estimate how high the base should be above the ground.

    For a wheeled bot, this is the wheel radius. For others, half the body height.
    """
    max_wheel_radius = 0.0
    for joint in root_body.joints:
        if joint.child is not None and joint.servo.continuous:
            # This is a wheel — use the child body's radius
            r = joint.child.radius or joint.child.dimensions[0] / 2
            max_wheel_radius = max(max_wheel_radius, r)

    if max_wheel_radius > 0:
        # Base origin at wheel center height. Also ensure the base mesh
        # (centered at origin) doesn't poke below the ground.
        base_half_height = root_body.dimensions[2] / 2
        return max(max_wheel_radius, base_half_height + 0.002)

    # No wheels — use half body height + small clearance
    return root_body.dimensions[2] / 2 + 0.01


def _build_scene_xml(bot: Bot) -> str:
    root = Element("mujoco", model=f"{bot.name}_scene")
    # Self-balancing wheeled bots need a small timestep (500 Hz) for stable
    # PID control. Fixed-base arms use moderate timestep. Others use large.
    has_wheels = any(j.servo.continuous for j in bot.all_joints)
    if has_wheels:
        timestep = "0.002"
    elif bot.base_type is BaseType.FIXED:
        timestep = "0.005"  # 200 Hz — good for position-controlled arms
    else:
        timestep = "0.02"
    SubElement(root, "option", timestep=timestep)
    SubElement(root, "include", file="bot.xml")
    # Fixed-base arms use tabletop scene if available, else room
    if bot.base_type is BaseType.FIXED:
        SubElement(root, "include", file="../../worlds/tabletop.xml")
    else:
        SubElement(root, "include", file="../../worlds/room.xml")
    return _prettify(root)


def _body_color(body: Body) -> str:
    """Pick a color for a body based on its shape/role."""
    r, g, b = _body_color_rgb(body)
    return f"{r:.4f} {g:.4f} {b:.4f} 1.0000"


def _body_color_rgb(body: Body) -> tuple[float, float, float]:
    """Read body color from material. Fallback to shape-based default."""
    r, g, b, _a = body.material.color
    return (r, g, b)


def _fmt_quat(q: tuple[float, float, float, float]) -> str:
    """Format quaternion (w, x, y, z) for MuJoCo XML."""
    return f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}"


def _half_dims(dims: Vec3Type) -> str:
    """Convert full dimensions to MuJoCo box half-extents string."""
    return f"{dims[0] / 2:.6f} {dims[1] / 2:.6f} {dims[2] / 2:.6f}"


def _fmt_vec3(v: tuple[float, float, float]) -> str:
    return f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}"


def _fmt_inertia(i: tuple[float, float, float, float, float, float]) -> str:
    return " ".join(f"{x:.6e}" for x in i)


def _prettify(elem: Element) -> str:
    """Return pretty-printed XML string."""
    raw = tostring(elem, encoding="unicode")
    parsed = minidom.parseString(raw)
    lines = parsed.toprettyxml(indent="  ", encoding=None).split("\n")
    # Remove blank lines that minidom loves to add
    cleaned = [line for line in lines if line.strip()]
    # Replace the XML declaration with a simpler one
    if cleaned and cleaned[0].startswith("<?xml"):
        cleaned[0] = "<?xml version='1.0' encoding='utf-8'?>"
    return "\n".join(cleaned) + "\n"
