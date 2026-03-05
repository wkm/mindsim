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

import math
from pathlib import Path
from typing import TYPE_CHECKING
from xml.dom import minidom
from xml.etree.ElementTree import Element, SubElement, tostring

from botcad.component import Vec3 as Vec3Type
from botcad.geometry import rotate_vec, servo_placement

if TYPE_CHECKING:
    from botcad.skeleton import Body, Bot, Joint


def emit_mujoco(bot: Bot, output_dir: Path) -> None:
    """Generate bot.xml and scene.xml in output_dir."""
    bot_xml = _build_bot_xml(bot)
    scene_xml = _build_scene_xml(bot)

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

    # Assets (mesh references)
    asset = SubElement(root, "asset")
    for body in bot.all_bodies:
        mesh_name = f"{body.name}_mesh"
        SubElement(
            asset,
            "mesh",
            name=mesh_name,
            file=f"{body.name}.stl",
            scale="1 1 1",  # STL already in meters
        )

    # Worldbody
    worldbody = SubElement(root, "worldbody")
    if bot.root is not None:
        _emit_body_tree(worldbody, bot.root, bot, is_root=True)

    # Contact excludes
    contact = SubElement(root, "contact")
    for body in bot.all_bodies:
        for joint in body.joints:
            if joint.child is not None:
                SubElement(
                    contact,
                    "exclude",
                    body1=body.name,
                    body2=joint.child.name,
                )

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
        ground_clearance = _estimate_ground_clearance(body, bot)
        attribs["pos"] = f"0 0 {ground_clearance:.4f}"
    elif parent_joint is not None:
        attribs["pos"] = _fmt_vec3(parent_joint.pos)

    body_el = SubElement(parent_el, "body", **attribs)

    # Root gets a freejoint
    if is_root:
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
        joint_attribs: dict[str, str] = {
            "name": parent_joint.name,
            "type": "hinge",
            "pos": "0 0 0",
            "axis": _fmt_vec3(parent_joint.axis),
            "damping": f"{damping:.4f}",
        }
        if not parent_joint.servo.continuous:
            joint_attribs["range"] = f"{lo:.4f} {hi:.4f}"
        SubElement(body_el, "joint", **joint_attribs)

    # Horn disc geom (purchased part visualization on child bodies)
    is_wheel = parent_joint is not None and parent_joint.servo.continuous
    if parent_joint is not None and not is_wheel:
        from botcad.bracket import horn_disc_params

        params = horn_disc_params(parent_joint.servo)
        if params is not None:
            ax, ay, az = parent_joint.axis
            ht = params.thickness / 2
            disc_pos = (ax * ht, ay * ht, az * ht)

            disc_attribs: dict[str, str] = {
                "name": f"{parent_joint.name}_horn_disc",
                "type": "cylinder",
                "size": f"{params.radius:.6f} {ht:.6f}",
                "pos": _fmt_vec3(disc_pos),
                "rgba": "0.85 0.85 0.88 0.9",
                "contype": "0",
                "conaffinity": "0",
                "group": "1",
            }

            quat = _z_to_axis_quat(parent_joint.axis)
            if quat is not None:
                disc_attribs["quat"] = _fmt_quat(quat)

            SubElement(body_el, "geom", **disc_attribs)

    # Visual geom (mesh) — for wheel bodies, mesh is visual-only;
    # a primitive cylinder handles collision instead.
    mesh_name = f"{body.name}_mesh"
    mesh_attribs: dict[str, str] = {
        "type": "mesh",
        "mesh": mesh_name,
        "rgba": _body_color(body),
    }
    if is_wheel:
        # Mesh is visual only; cylinder primitive handles collision
        mesh_attribs["contype"] = "0"
        mesh_attribs["conaffinity"] = "0"
    else:
        mesh_attribs["contype"] = "1"
        mesh_attribs["conaffinity"] = "1"
    SubElement(body_el, "geom", **mesh_attribs)

    # Wheel bodies get a primitive cylinder geom for reliable rolling contact
    if is_wheel:
        r = body.radius or body.dimensions[0] / 2
        half_w = (body.width or body.dimensions[2]) / 2
        # Cylinder default axis is Z; rotate 90° around Y to align with X
        SubElement(
            body_el,
            "geom",
            type="cylinder",
            size=f"{r} {half_w}",
            quat="0.7071068 0 0.7071068 0",
            rgba="0.15 0.15 0.15 0.0",
            contype="1",
            conaffinity="1",
            friction="1.5 0.005 0.0001",
        )

    # Camera mount
    _emit_camera(body_el, body)

    # IMU site on root body
    if is_root:
        SubElement(
            body_el,
            "site",
            name="imu",
            pos="0 0 0",
            size="0.005",
        )

    # Mounted component visualization — boxes at resolved positions
    _emit_mounted_components(body_el, body, is_wheel=is_wheel)

    # Mounting hardware visualization — screws at ear/mount positions
    _emit_mounting_hardware(body_el, body)

    # Wire route visualization — segments on their respective bodies
    _emit_body_wires(body_el, body, bot)

    # Servo visualization geoms — green boxes at each joint location
    for joint in body.joints:
        servo = joint.servo
        center, quat = servo_placement(
            servo.shaft_offset, servo.shaft_axis, joint.axis, joint.pos
        )
        SubElement(
            body_el,
            "geom",
            name=f"{joint.name}_servo",
            type="box",
            size=_half_dims(
                servo.body_dimensions
                if any(servo.body_dimensions)
                else servo.dimensions
            ),
            pos=_fmt_vec3(center),
            quat=_fmt_quat(quat),
            rgba="0.2 0.8 0.2 0.8",
            contype="0",
            conaffinity="0",
            group="1",
        )

    # Shaft boss visualization geoms — cylinders on top of servo boxes
    for joint in body.joints:
        servo = joint.servo
        if servo.shaft_boss_radius > 0 and servo.shaft_boss_height > 0:
            center, quat = servo_placement(
                servo.shaft_offset, servo.shaft_axis, joint.axis, joint.pos
            )
            # Boss sits at the shaft offset position, on top of the body
            boss_local = (
                servo.shaft_offset[0],
                servo.shaft_offset[1],
                servo.body_dimensions[2] / 2 + servo.shaft_boss_height / 2,
            )
            boss_world = _add_vec3(center, rotate_vec(quat, boss_local))
            boss_quat = _z_to_axis_quat(joint.axis)

            boss_attribs: dict[str, str] = {
                "name": f"{joint.name}_servo_boss",
                "type": "cylinder",
                "size": f"{servo.shaft_boss_radius:.6f} {servo.shaft_boss_height / 2:.6f}",
                "pos": _fmt_vec3(boss_world),
                "rgba": "0.2 0.8 0.2 0.8",
                "contype": "0",
                "conaffinity": "0",
                "group": "1",
            }
            if boss_quat is not None:
                boss_attribs["quat"] = _fmt_quat(boss_quat)
            SubElement(body_el, "geom", **boss_attribs)

    # Recurse: each child joint creates a child body
    for joint in body.joints:
        if joint.child is not None:
            _emit_body_tree(
                body_el, joint.child, bot, parent_joint=joint, is_root=False
            )


def _emit_mounted_components(
    parent_el: Element, body: Body, is_wheel: bool = False
) -> None:
    """Emit visualization geoms for all mounted components (Pi, battery, camera, etc.)."""
    if is_wheel:
        return  # wheel mesh already represents the component
    for mount in body.mounts:
        comp = mount.component
        r, g, b, a = comp.color
        SubElement(
            parent_el,
            "geom",
            name=f"comp_{body.name}_{mount.label}",
            type="box",
            size=_half_dims(comp.dimensions),
            pos=_fmt_vec3(mount.resolved_pos),
            rgba=f"{r} {g} {b} {a}",
            contype="0",
            conaffinity="0",
            group="1",
        )


def _emit_mounting_hardware(body_el: Element, body: Body) -> None:
    """Emit small cylinder geoms at screw/mounting positions."""
    _SCREW_RGBA = "0.7 0.7 0.7 0.9"
    _SCREW_HEIGHT = "0.001"  # 1mm half-height

    # Servo mounting ears (bracket screw holes on this body's joints)
    for joint in body.joints:
        center, quat = servo_placement(
            joint.servo.shaft_offset,
            joint.servo.shaft_axis,
            joint.axis,
            joint.pos,
        )
        for ear in joint.servo.mounting_ears:
            world_pos = _add_vec3(center, rotate_vec(quat, ear.pos))
            SubElement(
                body_el,
                "geom",
                name=f"screw_{joint.name}_{ear.label}",
                type="cylinder",
                size=f"{ear.diameter / 2:.4f} {_SCREW_HEIGHT}",
                pos=_fmt_vec3(world_pos),
                rgba=_SCREW_RGBA,
                contype="0",
                conaffinity="0",
                group="1",
            )

        # Horn mounting points (output face screw holes)
        for mp in joint.servo.horn_mounting_points:
            world_pos = _add_vec3(center, rotate_vec(quat, mp.pos))
            SubElement(
                body_el,
                "geom",
                name=f"horn_{joint.name}_{mp.label}",
                type="cylinder",
                size=f"{mp.diameter / 2:.4f} {_SCREW_HEIGHT}",
                pos=_fmt_vec3(world_pos),
                rgba=_SCREW_RGBA,
                contype="0",
                conaffinity="0",
                group="1",
            )

        # Rear horn mounting points (blind side screw holes)
        for mp in joint.servo.rear_horn_mounting_points:
            world_pos = _add_vec3(center, rotate_vec(quat, mp.pos))
            SubElement(
                body_el,
                "geom",
                name=f"rear_{joint.name}_{mp.label}",
                type="cylinder",
                size=f"{mp.diameter / 2:.4f} {_SCREW_HEIGHT}",
                pos=_fmt_vec3(world_pos),
                rgba=_SCREW_RGBA,
                contype="0",
                conaffinity="0",
                group="1",
            )

    # Component mount points (screw holes on mounted components)
    for mount in body.mounts:
        for mp in mount.component.mounting_points:
            pos = _add_vec3(mount.resolved_pos, mp.pos)
            SubElement(
                body_el,
                "geom",
                name=f"mount_{body.name}_{mount.label}_{mp.label}",
                type="cylinder",
                size=f"{mp.diameter / 2:.4f} {_SCREW_HEIGHT}",
                pos=_fmt_vec3(pos),
                rgba=_SCREW_RGBA,
                contype="0",
                conaffinity="0",
                group="1",
            )


def _add_vec3(a: Vec3Type, b: Vec3Type) -> Vec3Type:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _emit_camera(parent_el: Element, body: Body) -> None:
    """Emit camera element if this body has a camera mounted."""
    for mount in body.mounts:
        if mount.component.name == "OV5647":
            pos = mount.resolved_pos
            # Camera looks forward (+Y in MuJoCo convention)
            # Rotate 180° around Y to flip the image right-side up
            SubElement(
                parent_el,
                "camera",
                name="camera_1_cam",
                fovy="72.0",
                pos=_fmt_vec3(pos),
                euler=f"0 {math.pi} 0",
            )
            return


_WIRE_COLORS = {
    "uart_half_duplex": "0.2 0.4 1.0 0.9",  # blue — servo daisy-chain
    "csi": "1.0 0.8 0.0 0.9",  # yellow — camera ribbon
    "power": "0.9 0.2 0.2 0.9",  # red — power
}
_WIRE_RADIUS = {
    "uart_half_duplex": "0.0009",  # 0.9mm — servo bus (channel is 1.5mm)
    "csi": "0.0018",  # 1.8mm — CSI ribbon (channel is 3mm)
    "power": "0.0012",  # 1.2mm — power cable (channel is 2mm)
}


def _emit_body_wires(body_el: Element, body: Body, bot: Bot) -> None:
    """Emit wire segments that belong to this body.

    Each segment's body_name is checked against the current body — only
    segments belonging to this body become geoms here. Coordinates are
    already in body-local frame from the routing solver.
    """
    for route in bot.wire_routes:
        color = _WIRE_COLORS.get(route.bus_type, "0.5 0.5 0.5 0.9")
        radius = _WIRE_RADIUS.get(route.bus_type, "0.0015")
        for i, seg in enumerate(route.segments):
            if seg.body_name != body.name:
                continue
            # Skip degenerate segments (endpoints too close)
            dx = seg.end[0] - seg.start[0]
            dy = seg.end[1] - seg.start[1]
            dz = seg.end[2] - seg.start[2]
            if dx * dx + dy * dy + dz * dz < 1e-6:
                continue
            SubElement(
                body_el,
                "geom",
                name=f"wire_{route.label}_{body.name}_{i}",
                type="capsule",
                fromto=(
                    f"{seg.start[0]:.6f} {seg.start[1]:.6f} {seg.start[2]:.6f} "
                    f"{seg.end[0]:.6f} {seg.end[1]:.6f} {seg.end[2]:.6f}"
                ),
                size=radius,
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
    # PID control. Non-wheeled bots can use a larger step.
    has_wheels = any(j.servo.continuous for j in bot.all_joints)
    timestep = "0.002" if has_wheels else "0.02"
    SubElement(root, "option", timestep=timestep)
    SubElement(root, "include", file="bot.xml")
    SubElement(root, "include", file="../../worlds/room.xml")
    return _prettify(root)


def _body_color(body: Body) -> str:
    """Pick a color for a body based on its shape/role."""
    if body.shape == "cylinder" and body.radius and body.radius > 0.03:
        return "0.15 0.15 0.15 1.0"  # dark for wheels
    if body.shape == "tube":
        return "0.7 0.7 0.7 1.0"  # light gray for structural tubes
    return "0.9 0.9 0.9 1.0"  # off-white default


def _z_to_axis_quat(
    axis: tuple[float, float, float],
) -> tuple[float, float, float, float] | None:
    """Quaternion rotating Z to the given axis. Returns None if already Z-aligned."""
    ax, ay, az = axis
    if abs(az) > 0.999:
        if az < 0:
            return (0.0, 1.0, 0.0, 0.0)  # 180° around X
        return None  # identity, no rotation needed

    angle = math.acos(max(-1.0, min(1.0, az)))
    half = angle / 2
    # Rotation axis = Z × target = (-ay, ax, 0)
    rot_mag = math.sqrt(ax * ax + ay * ay)
    s = math.sin(half)
    return (math.cos(half), -ay / rot_mag * s, ax / rot_mag * s, 0.0)


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
