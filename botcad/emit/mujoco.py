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

from botcad.component import BusType
from botcad.component import Vec3 as Vec3Type
from botcad.geometry import rotate_vec, rotation_between, servo_placement
from botcad.skeleton import BaseType, BodyShape

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

    # Component mesh assets (Pi, battery, camera, etc.)
    for body in bot.all_bodies:
        is_wheel_body = any("Wheel" in m.component.name for m in body.mounts)
        if is_wheel_body:
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

    # Worldbody
    worldbody = SubElement(root, "worldbody")
    if bot.root is not None:
        _emit_body_tree(worldbody, bot.root, bot, is_root=True)

    # Contact excludes — bodies within 3 joints of each other collide due to
    # bracket/coupler geometry, not real contact. Exclude up to 3 joints apart
    # to cover compact mechanisms like wrist-roll + gripper chains.
    contact = SubElement(root, "contact")
    exclude_pairs: set[tuple[str, str]] = set()
    for body in bot.all_bodies:
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
        if parent_joint.servo.continuous and body.shape is BodyShape.CYLINDER:
            # Wheel child body: offset outward from servo along joint axis
            # so the wheel clears the servo body + shaft boss.
            wheel_offset = _wheel_outboard_offset(parent_joint, body)
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
        if parent_joint.servo.continuous and body.shape is BodyShape.CYLINDER:
            wheel_offset = _wheel_outboard_offset(parent_joint, body)
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

    # Mounted component visualization — meshes at resolved positions
    _emit_mounted_components(body_el, body, is_wheel=is_wheel)

    # Servo visualization geoms — detailed mesh at each joint
    # Cache servo_placement per joint (also used by _emit_mounting_hardware).
    joint_placements: dict[str, tuple] = {}
    for joint in body.joints:
        servo = joint.servo
        center, quat = servo_placement(
            servo.shaft_offset, servo.shaft_axis, joint.axis, joint.pos
        )
        joint_placements[joint.name] = (center, quat)
        SubElement(
            body_el,
            "geom",
            name=f"{joint.name}_servo",
            type="mesh",
            mesh=f"servo_{servo.name}_mesh",
            pos=_fmt_vec3(center),
            quat=_fmt_quat(quat),
            rgba="0.15 0.15 0.15 1.0",
            contype="0",
            conaffinity="0",
            group="1",
        )

    # Mounting hardware visualization — screws at ear/mount positions
    _emit_mounting_hardware(body_el, body, joint_placements)

    # Wire route visualization — segments on their respective bodies
    _emit_body_wires(body_el, body, bot)

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
        mesh_name = f"comp_{body.name}_{mount.label}_mesh"
        SubElement(
            parent_el,
            "geom",
            name=f"comp_{body.name}_{mount.label}",
            type="mesh",
            mesh=mesh_name,
            pos=_fmt_vec3(mount.resolved_pos),
            rgba=f"{r} {g} {b} {a}",
            contype="0",
            conaffinity="0",
            group="1",
        )


def _emit_mounting_hardware(
    body_el: Element, body: Body, joint_placements: dict[str, tuple]
) -> None:
    """Emit small cylinder geoms at screw/mounting positions."""
    _SCREW_RGBA = "0.7 0.7 0.7 0.9"
    _SCREW_HEIGHT = "0.001"  # 1mm half-height

    # Servo mounting ears (bracket screw holes on this body's joints)
    for joint in body.joints:
        center, quat = joint_placements[joint.name]
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
            pos = _add_vec3(mount.resolved_pos, mount.rotate_point(mp.pos))
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


from botcad.geometry import add_vec3 as _add_vec3  # noqa: E402


def _emit_camera(parent_el: Element, body: Body) -> None:
    """Emit camera element if this body has a camera mounted."""
    from botcad.component import CameraSpec

    for mount in body.mounts:
        if isinstance(mount.component, CameraSpec):
            cam = mount.component
            pos = mount.resolved_pos
            # Camera looks forward (+Y in MuJoCo convention)
            # Rotate 180° around Y to flip the image right-side up
            SubElement(
                parent_el,
                "camera",
                name=f"{mount.label}_cam",
                fovy=f"{cam.fov_deg:.1f}",
                pos=_fmt_vec3(pos),
                euler=f"0 {math.pi} 0",
            )
            return


_WIRE_COLORS = {
    BusType.UART_HALF_DUPLEX: "0.2 0.4 1.0 0.9",  # blue — servo daisy-chain
    BusType.CSI: "1.0 0.8 0.0 0.9",  # yellow — camera ribbon
    BusType.POWER: "0.9 0.2 0.2 0.9",  # red — power
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
        color = _WIRE_COLORS.get(route.bus_type, "0.5 0.5 0.5 0.9")
        radius = _WIRE_RADIUS.get(route.bus_type, "0.0015")
        for i, seg in enumerate(route.segments):
            if seg.body_name != body.name:
                continue
            # Skip degenerate segments (endpoints too close)
            if seg.straight_length < 0.001:
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


def _wheel_outboard_offset(joint: Joint, child: Body) -> float:
    """Compute how far a wheel child body sits outboard from the joint (shaft).

    The wheel hub mates with the servo shaft boss, so the wheel's inner face
    is approximately at the boss tip. Offset = boss_height + half_wheel_width.
    """
    boss_h = joint.servo.shaft_boss_height or 0.0
    half_w = (child.width or child.dimensions[2]) / 2
    return boss_h + half_w


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
    return f"{r} {g} {b} 1.0"


def _body_color_rgb(body: Body) -> tuple[float, float, float]:
    """Shape-based body color. Shared between CAD and MuJoCo emitters."""
    if body.shape is BodyShape.CYLINDER and body.radius and body.radius > 0.03:
        return (0.15, 0.15, 0.15)  # dark gray: wheels
    if body.shape is BodyShape.TUBE:
        return (0.7, 0.7, 0.7)  # light gray: structural tubes
    if body.shape is BodyShape.JAW:
        return (0.85, 0.85, 0.85)  # light gray: gripper jaw
    return (0.9, 0.9, 0.9)  # off-white: default


def _z_to_axis_quat(
    axis: tuple[float, float, float],
) -> tuple[float, float, float, float] | None:
    """Quaternion rotating Z to the given axis. Returns None if already Z-aligned."""
    ax, ay, az = axis
    if abs(az) > 0.999:
        if az < 0:
            return (0.0, 1.0, 0.0, 0.0)  # 180° around X
        return None  # identity, no rotation needed
    q = rotation_between((0.0, 0.0, 1.0), axis)
    return q


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
