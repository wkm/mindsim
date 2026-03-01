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
from botcad.geometry import servo_placement

if TYPE_CHECKING:
    from botcad.skeleton import Body, Bot, Joint


def emit_mujoco(bot: Bot, output_dir: Path) -> None:
    """Generate bot.xml and scene.xml in output_dir."""
    bot_xml = _build_bot_xml(bot)
    scene_xml = _build_scene_xml(bot)

    (output_dir / "bot.xml").write_text(bot_xml)
    (output_dir / "scene.xml").write_text(scene_xml)

    # Generate per-body STL meshes (simple box/cylinder primitives)
    _generate_meshes(bot, output_dir / "meshes")

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

    # Visual geom (mesh) — for wheel bodies, mesh is visual-only;
    # a primitive cylinder handles collision instead.
    is_wheel = parent_joint is not None and parent_joint.servo.continuous
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
            size=_half_dims(servo.dimensions),
            pos=_fmt_vec3(center),
            quat=_fmt_quat(quat),
            rgba="0.2 0.8 0.2 0.8",
            contype="0",
            conaffinity="0",
            group="1",
        )

    # Recurse: each child joint creates a child body
    for joint in body.joints:
        if joint.child is not None:
            _emit_body_tree(
                body_el, joint.child, bot, parent_joint=joint, is_root=False
            )


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


def _generate_meshes(bot: Bot, meshes_dir: Path) -> None:
    """Generate simple STL meshes for each body.

    Creates box/cylinder primitives as STL files. These are placeholder
    visuals — the CAD emitter generates proper geometry with cutouts.
    """
    meshes_dir.mkdir(parents=True, exist_ok=True)

    # Build body → parent joint mapping so we can orient cylinders
    # along their rotation axis
    parent_joint: dict[str, Joint] = {}
    for joint in bot.all_joints:
        if joint.child is not None:
            parent_joint[joint.child.name] = joint

    for body in bot.all_bodies:
        dims = body.dimensions
        if body.shape == "cylinder":
            r = body.radius or dims[0] / 2
            h = body.width or dims[2]
            # Orient cylinder along parent joint's rotation axis
            pj = parent_joint.get(body.name)
            axis = pj.axis if pj else (0.0, 0.0, 1.0)
            stl_data = _cylinder_stl(r, h, orient_axis=axis)
        elif body.shape == "tube":
            tube_len = body.length or dims[2]
            stl_data = _cylinder_stl(
                body.outer_r or dims[0] / 2, tube_len, z_offset=tube_len / 2
            )
        elif body.shape == "sphere":
            stl_data = _sphere_stl(body.radius or dims[0] / 2)
        else:
            stl_data = _box_stl(dims[0], dims[1], dims[2])

        (meshes_dir / f"{body.name}.stl").write_bytes(stl_data)


def _box_stl(sx: float, sy: float, sz: float) -> bytes:
    """Generate a binary STL for a box centered at origin."""

    hx, hy, hz = sx / 2, sy / 2, sz / 2

    # 8 vertices of a box
    v = [
        (-hx, -hy, -hz),
        (hx, -hy, -hz),
        (hx, hy, -hz),
        (-hx, hy, -hz),
        (-hx, -hy, hz),
        (hx, -hy, hz),
        (hx, hy, hz),
        (-hx, hy, hz),
    ]

    # 12 triangles (2 per face)
    faces = [
        # Bottom (-Z)
        (0, 2, 1),
        (0, 3, 2),
        # Top (+Z)
        (4, 5, 6),
        (4, 6, 7),
        # Front (+Y)
        (3, 7, 6),
        (3, 6, 2),
        # Back (-Y)
        (0, 1, 5),
        (0, 5, 4),
        # Right (+X)
        (1, 2, 6),
        (1, 6, 5),
        # Left (-X)
        (0, 4, 7),
        (0, 7, 3),
    ]

    return _pack_stl(v, faces)


def _cylinder_stl(
    radius: float,
    height: float,
    segments: int = 16,
    z_offset: float = 0.0,
    orient_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> bytes:
    """Generate a binary STL for a cylinder.

    By default the cylinder axis is along Z, centered at origin.
    - z_offset: shift center along the cylinder axis (e.g. height/2 for tubes)
    - orient_axis: final axis direction — vertices are rotated from Z to this axis
    """
    hz = height / 2
    z_lo = -hz + z_offset
    z_hi = hz + z_offset
    vertices = []
    faces = []

    # Top and bottom center vertices
    bot_center = len(vertices)
    vertices.append((0.0, 0.0, z_lo))
    top_center = len(vertices)
    vertices.append((0.0, 0.0, z_hi))

    # Ring vertices
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        vertices.append((x, y, z_lo))  # bottom ring
        vertices.append((x, y, z_hi))  # top ring

    for i in range(segments):
        bi = 2 + i * 2  # bottom ring vertex
        ti = 3 + i * 2  # top ring vertex
        ni = (i + 1) % segments
        bni = 2 + ni * 2
        tni = 3 + ni * 2

        # Bottom cap
        faces.append((bot_center, bni, bi))
        # Top cap
        faces.append((top_center, ti, tni))
        # Side quads (2 triangles)
        faces.append((bi, bni, tni))
        faces.append((bi, tni, ti))

    # Rotate from Z-axis to the desired orient_axis
    ax, ay, az = orient_axis
    mag = math.sqrt(ax * ax + ay * ay + az * az)
    if mag > 0:
        ax, ay, az = ax / mag, ay / mag, az / mag
    # Only rotate if orient_axis differs significantly from Z
    if abs(az) < 0.999:
        vertices = _rotate_vertices_z_to(vertices, (ax, ay, az))

    return _pack_stl(vertices, faces)


def _rotate_vertices_z_to(
    vertices: list[tuple[float, float, float]],
    target: tuple[float, float, float],
) -> list[tuple[float, float, float]]:
    """Rotate all vertices so that Z-axis maps to target axis.

    Uses Rodrigues' rotation formula around the cross product of Z and target.
    """
    # Z cross target
    zx, zy, zz = 0.0, 0.0, 1.0
    tx, ty, tz = target
    # cross product
    kx = zy * tz - zz * ty
    ky = zz * tx - zx * tz
    kz = zx * ty - zy * tx
    sin_angle = math.sqrt(kx * kx + ky * ky + kz * kz)
    cos_angle = zx * tx + zy * ty + zz * tz  # dot product

    if sin_angle < 1e-9:
        # Parallel or anti-parallel
        if cos_angle < 0:
            # 180° rotation around X
            return [(x, -y, -z) for x, y, z in vertices]
        return vertices

    # Normalize rotation axis
    kx, ky, kz = kx / sin_angle, ky / sin_angle, kz / sin_angle

    result = []
    for x, y, z in vertices:
        # k cross v
        kcx = ky * z - kz * y
        kcy = kz * x - kx * z
        kcz = kx * y - ky * x
        # k dot v
        kdv = kx * x + ky * y + kz * z
        # Rodrigues: v*cos + (k x v)*sin + k*(k.v)*(1-cos)
        rx = x * cos_angle + kcx * sin_angle + kx * kdv * (1 - cos_angle)
        ry = y * cos_angle + kcy * sin_angle + ky * kdv * (1 - cos_angle)
        rz = z * cos_angle + kcz * sin_angle + kz * kdv * (1 - cos_angle)
        result.append((rx, ry, rz))
    return result


def _sphere_stl(radius: float, rings: int = 8, segments: int = 16) -> bytes:
    """Generate a binary STL for a UV sphere centered at origin."""
    vertices = []
    faces = []

    # Top pole
    vertices.append((0.0, 0.0, radius))
    # Bottom pole
    vertices.append((0.0, 0.0, -radius))

    for i in range(1, rings):
        phi = math.pi * i / rings
        for j in range(segments):
            theta = 2 * math.pi * j / segments
            x = radius * math.sin(phi) * math.cos(theta)
            y = radius * math.sin(phi) * math.sin(theta)
            z = radius * math.cos(phi)
            vertices.append((x, y, z))

    # Top cap triangles
    for j in range(segments):
        nj = (j + 1) % segments
        faces.append((0, 2 + j, 2 + nj))

    # Middle quads
    for i in range(rings - 2):
        for j in range(segments):
            nj = (j + 1) % segments
            a = 2 + i * segments + j
            b = 2 + i * segments + nj
            c = 2 + (i + 1) * segments + nj
            d = 2 + (i + 1) * segments + j
            faces.append((a, d, c))
            faces.append((a, c, b))

    # Bottom cap triangles
    base = 2 + (rings - 2) * segments
    for j in range(segments):
        nj = (j + 1) % segments
        faces.append((1, base + nj, base + j))

    return _pack_stl(vertices, faces)


def _pack_stl(
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int]],
) -> bytes:
    """Pack vertices and faces into binary STL format."""
    import struct

    header = b"\x00" * 80
    num_triangles = len(faces)
    data = bytearray(header)
    data.extend(struct.pack("<I", num_triangles))

    for i0, i1, i2 in faces:
        v0, v1, v2 = vertices[i0], vertices[i1], vertices[i2]
        # Compute face normal
        e1 = (v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2])
        e2 = (v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2])
        nx = e1[1] * e2[2] - e1[2] * e2[1]
        ny = e1[2] * e2[0] - e1[0] * e2[2]
        nz = e1[0] * e2[1] - e1[1] * e2[0]
        mag = math.sqrt(nx * nx + ny * ny + nz * nz)
        if mag > 0:
            nx, ny, nz = nx / mag, ny / mag, nz / mag

        data.extend(struct.pack("<fff", nx, ny, nz))
        data.extend(struct.pack("<fff", *v0))
        data.extend(struct.pack("<fff", *v1))
        data.extend(struct.pack("<fff", *v2))
        data.extend(struct.pack("<H", 0))  # attribute byte count

    return bytes(data)


def _body_color(body: Body) -> str:
    """Pick a color for a body based on its shape/role."""
    if body.shape == "cylinder" and body.radius and body.radius > 0.03:
        return "0.15 0.15 0.15 1.0"  # dark for wheels
    if body.shape == "tube":
        return "0.7 0.7 0.7 1.0"  # light gray for structural tubes
    return "0.9 0.9 0.9 1.0"  # off-white default


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
