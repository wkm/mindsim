"""ShapeScript emitters for component factories.

Translates camera_solid(), battery_solid(), _make_bearing_solid(),
_horn_solid(), connector_solid(), receptacle_solid(), fastener_solid(),
_make_wheel_solid(), _wire_channel(), and _child_clearance_volume()
into ShapeScript IR programs or inline ops.

Battery cells use FilletAllEdgesOp (native ShapeScript). Fastener heads
still use PrebuiltOp due to edge-selection chamfers and RegularPolygon
extrude that ShapeScript doesn't yet support.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from botcad.shapescript.ops import ALIGN_MIN_Z, Align3
from botcad.shapescript.program import ShapeScript

if TYPE_CHECKING:
    from botcad.component import (
        BatterySpec,
        BearingSpec,
        CameraSpec,
        ServoSpec,
    )


def camera_script(spec: CameraSpec) -> ShapeScript:
    """Translate camera_solid() to ShapeScript ops.

    Models PCB, mounting holes, lens base, lens barrel, and CSI connector.
    """
    prog = ShapeScript()

    pcb_w, pcb_h, _pcb_t = spec.dimensions
    pcb_thick = 0.0016  # standard 1.6mm FR4

    # 1. PCB
    pcb = prog.box(pcb_w, pcb_h, pcb_thick)

    # 2. Mounting holes — subtract from PCB
    for mp in spec.mounting_points:
        hole = prog.cylinder(mp.diameter / 2, pcb_thick + 0.001)
        hole = prog.locate(hole, pos=mp.pos)
        pcb = prog.cut(pcb, hole)

    # 3. Lens base (8.5 x 8.5mm, z-min aligned, offset +2.5mm Y)
    base_size = 0.0085
    base_height = 0.0050
    lens_y_offset = 0.0025
    lens_base = prog.box(base_size, base_size, base_height, align=Align3(z="min"))
    lens_base = prog.locate(lens_base, pos=(0, lens_y_offset, pcb_thick / 2))

    # 4. Lens barrel
    barrel_r = 0.00325
    barrel_h = 0.0025
    lens_barrel = prog.cylinder(barrel_r, barrel_h, align=Align3(z="min"))
    lens_barrel = prog.locate(
        lens_barrel, pos=(0, lens_y_offset, pcb_thick / 2 + base_height)
    )

    # 5. CSI connector (y="max" aligns top edge to origin, then locate to bottom)
    conn_w = 0.016
    conn_h_dim = 0.005
    conn_t = 0.002
    connector = prog.box(conn_w, conn_t, conn_h_dim, align=Align3(y="max", z="min"))
    connector = prog.locate(connector, pos=(0, -pcb_h / 2, pcb_thick / 2))

    # 6. Fuse everything
    result = prog.fuse(pcb, lens_base)
    result = prog.fuse(result, lens_barrel)
    result = prog.fuse(result, connector)

    prog.output_ref = result
    return prog


def battery_script(spec: BatterySpec) -> ShapeScript:
    """Translate battery_solid() to ShapeScript ops.

    All geometry uses native ShapeScript ops — no PrebuiltOps.
    Label and cable exit are simple box primitives.
    """
    prog = ShapeScript()
    w, length, h = spec.dimensions

    # 1. Main battery body (filleted cells — native ops)
    if spec.cells_s == 2:
        cell_w = length / 2 - 0.001
        cell = prog.box(w, cell_w, h)
        cell = prog.fillet_all(cell, 0.003)
        cell_neg = prog.locate(cell, pos=(0, -length / 4, 0))
        cell_pos = prog.locate(cell, pos=(0, length / 4, 0))
        body = prog.fuse(cell_neg, cell_pos)
    else:
        body = prog.box(w, length, h, tag="battery_body")
        body = prog.fillet_all(body, 0.003)

    # 2. Label (sits on top face)
    label_w = w * 0.7
    label_l = length * 0.6
    label_t = 0.0005
    label = prog.box(label_w, label_l, label_t, align=Align3(z="min"))
    label = prog.locate(label, pos=(0, 0, h / 2 - 0.0001))

    # 3. Cable exit block
    exit_w = 0.012
    exit_l = 0.006
    exit_h = h * 0.8
    exit_block = prog.box(exit_l, exit_w, exit_h, align=Align3(x="min"))
    exit_block = prog.locate(exit_block, pos=(w / 2 - 0.001, 0, 0))

    # 4. Fuse
    result = prog.fuse(body, label)
    result = prog.fuse(result, exit_block)

    prog.output_ref = result
    return prog


def bearing_script(spec: BearingSpec) -> ShapeScript:
    """Translate _make_bearing_solid() to ShapeScript: outer - inner cylinder."""
    prog = ShapeScript()

    outer = prog.cylinder(spec.od / 2, spec.width)
    inner = prog.cylinder(spec.id / 2, spec.width + 0.001)
    result = prog.cut(outer, inner)

    prog.output_ref = result
    return prog


def horn_script(servo: ServoSpec) -> ShapeScript | None:
    """Translate _horn_solid() to ShapeScript: single cylinder from horn_disc_params().

    Returns None if the servo has no horn mounting points.
    """
    from botcad.bracket import horn_disc_params

    params = horn_disc_params(servo)
    if params is None:
        return None

    prog = ShapeScript()
    horn = prog.cylinder(params.radius, params.thickness)
    prog.output_ref = horn
    return prog


# ── Connector / Receptacle / Fastener emitters ──────────────────────


def connector_script(spec: ConnectorSpec) -> ShapeScript:
    """Translate connector_solid() to ShapeScript ops.

    All 5 connector types are expressed as native Box/Cylinder + fuse/cut ops:
    polarization keys, pin blades, locking tabs, wire entry channels, etc.
    """
    from botcad.connectors import ConnectorType
    from botcad.shapescript.ops import Align3

    prog = ShapeScript()
    bx, by, bz = spec.body_dimensions

    # Main housing body
    body = prog.box(bx, by, bz, tag="connector_body")

    if spec.connector_type == ConnectorType.MOLEX_5264_3PIN:
        # Polarization rib on one side
        rib = prog.box(bx * 0.15, by * 0.12, bz * 0.7)
        rib = prog.locate(rib, pos=(-bx * 0.35, by / 2, -bz * 0.05))
        body = prog.fuse(body, rib)

        # Locking tab
        tab = prog.box(bx * 0.4, by * 0.08, bz * 0.15)
        tab = prog.locate(tab, pos=(0, -by / 2, bz * 0.25))
        body = prog.fuse(body, tab)

        # 3 pin blades on mating face (+Z)
        pin_w, pin_d, pin_h = 0.0006, 0.0002, 0.002
        pitch = 0.00254
        for i in range(3):
            px = -pitch + i * pitch
            pin = prog.box(pin_w, pin_d, pin_h, align=ALIGN_MIN_Z)
            pin = prog.locate(pin, pos=(px, 0, bz / 2))
            body = prog.fuse(body, pin)

        # Wire entry channels on back face (-Z)
        for i in range(3):
            px = -pitch + i * pitch
            chan = prog.cylinder(0.0005, by * 0.3)
            chan = prog.locate(chan, pos=(px, 0, -bz / 2))
            body = prog.cut(body, chan)

    elif spec.connector_type == ConnectorType.CSI_15PIN:
        # ZIF lever on top
        lever_h = bz * 0.2
        lever = prog.box(bx * 0.92, by * 0.6, lever_h)
        lever = prog.locate(lever, pos=(0, 0, bz / 2 + lever_h / 2 - 0.0002))
        body = prog.fuse(body, lever)

        # Ribbon cable slot
        slot = prog.box(bx * 0.85, by * 0.4, bz * 0.3)
        slot = prog.locate(slot, pos=(0, -by * 0.1, -bz * 0.35))
        body = prog.cut(body, slot)

        # Contact ridges inside
        for i in range(7):
            rx = -bx * 0.38 + i * bx * 0.76 / 6
            ridge = prog.box(0.0003, by * 0.2, bz * 0.15)
            ridge = prog.locate(ridge, pos=(rx, -by * 0.1, -bz * 0.15))
            body = prog.fuse(body, ridge)

    elif spec.connector_type == ConnectorType.XT30:
        # Keying: chamfered corner
        chamfer_block = prog.box(bx * 0.15, by * 0.15, bz + 0.001)
        chamfer_block = prog.locate(chamfer_block, pos=(bx / 2, by / 2, 0))
        body = prog.cut(body, chamfer_block)

        # Round pin on mating face (+X)
        round_pin = prog.cylinder(
            0.001, 0.003, align=ALIGN_MIN_Z
        )
        round_pin = prog.locate(round_pin, pos=(bx / 2, -by * 0.2, 0))
        body = prog.fuse(body, round_pin)

        # Flat pin
        flat_pin = prog.box(0.002, 0.001, 0.003, align=ALIGN_MIN_Z)
        flat_pin = prog.locate(flat_pin, pos=(bx / 2, by * 0.2, 0))
        body = prog.fuse(body, flat_pin)

        # Grip ridges on sides
        for i in range(3):
            gz = -bz * 0.3 + i * bz * 0.3
            ridge = prog.box(bx * 0.8, by * 0.06, 0.0005)
            ridge = prog.locate(ridge, pos=(0, by / 2, gz))
            body = prog.fuse(body, ridge)
            ridge2 = prog.box(bx * 0.8, by * 0.06, 0.0005)
            ridge2 = prog.locate(ridge2, pos=(0, -by / 2, gz))
            body = prog.fuse(body, ridge2)

    elif spec.connector_type == ConnectorType.JST_XH_3PIN:
        # Locking tab
        tab_w = bx * 0.35
        tab_h = bz * 0.12
        tab_d = by * 0.08
        tab = prog.box(tab_w, tab_d, tab_h)
        tab = prog.locate(tab, pos=(0, by / 2 + tab_d / 2, bz * 0.2))
        body = prog.fuse(body, tab)

        # Polarization groove
        groove = prog.box(bx * 0.12, by + 0.001, bz * 0.15)
        groove = prog.locate(groove, pos=(-bx * 0.35, 0, bz * 0.3))
        body = prog.cut(body, groove)

        # 3 pin blades on mating face (+Z)
        pitch = 0.0025
        pin_w, pin_d, pin_h = 0.0006, 0.0002, 0.002
        for i in range(3):
            px = -pitch + i * pitch
            pin = prog.box(pin_w, pin_d, pin_h, align=ALIGN_MIN_Z)
            pin = prog.locate(pin, pos=(px, 0, bz / 2))
            body = prog.fuse(body, pin)

        # Strain relief bumps on back face (-Z)
        for i in range(3):
            px = -pitch + i * pitch
            bump = prog.cylinder(
                0.0006, 0.001, align=Align3(z="max")
            )
            bump = prog.locate(bump, pos=(px, 0, -bz / 2))
            body = prog.fuse(body, bump)

    elif spec.connector_type == ConnectorType.GPIO_2X20:
        # Keying notch
        notch = prog.box(0.003, by + 0.001, bz * 0.4)
        notch = prog.locate(notch, pos=(bx / 2 - 0.001, 0, bz * 0.15))
        body = prog.cut(body, notch)

        # Two rows of socket holes — batch fuse all holes then cut once
        pitch = 0.00254
        holes = None
        for row_y in [-0.00127, 0.00127]:
            for i in range(20):
                px = -pitch * 9.5 + i * pitch
                hole = prog.cylinder(
                    0.0004, bz * 0.5, align=Align3(z="max")
                )
                hole = prog.locate(hole, pos=(px, row_y, -bz / 2))
                if holes is None:
                    holes = hole
                else:
                    holes = prog.fuse(holes, hole)
        body = prog.cut(body, holes)

        # Cable exit on top
        cable_exit = prog.box(bx * 0.8, by * 0.6, 0.002)
        cable_exit = prog.locate(cable_exit, pos=(0, 0, bz / 2 + 0.001))
        body = prog.fuse(body, cable_exit)

    prog.output_ref = body
    return prog


def receptacle_script(spec: ConnectorSpec) -> ShapeScript:
    """Translate receptacle_solid() to ShapeScript ops.

    Slightly larger housing with a cavity on the mating face. Connector-type-
    specific features (GPIO shroud, XT30 flange) and solder pins.
    """
    from botcad.connectors import ConnectorType
    from botcad.shapescript.ops import Align3

    prog = ShapeScript()
    bx, by, bz = spec.body_dimensions

    wall = 0.0008
    rx, ry, rz = bx + 2 * wall, by + 2 * wall, bz * 0.7

    body = prog.box(rx, ry, rz, tag="receptacle_body")

    # Cavity facing the mating direction
    mx, my, mz = spec.mating_direction
    cavity_depth = rz * 0.6

    if abs(mz) > 0.5:
        cavity = prog.box(bx + 0.0002, by + 0.0002, cavity_depth)
        cz = (rz / 2 - cavity_depth / 2) * (1 if mz > 0 else -1)
        cavity = prog.locate(cavity, pos=(0, 0, cz))
    elif abs(mx) > 0.5:
        cavity = prog.box(cavity_depth, by + 0.0002, bz * 0.5)
        cx = (rx / 2 - cavity_depth / 2) * (1 if mx > 0 else -1)
        cavity = prog.locate(cavity, pos=(cx, 0, 0))
    elif abs(my) > 0.5:
        cavity = prog.box(bx + 0.0002, cavity_depth, bz * 0.5)
        cy = (ry / 2 - cavity_depth / 2) * (1 if my > 0 else -1)
        cavity = prog.locate(cavity, pos=(0, cy, 0))

    body = prog.cut(body, cavity)

    if spec.connector_type == ConnectorType.GPIO_2X20:
        # Shroud walls around pin header
        shroud_h = rz * 0.4
        shroud = prog.box(rx, ry, shroud_h, align=Align3(z="min"))
        shroud = prog.locate(shroud, pos=(0, 0, rz / 2))
        inner = prog.box(bx - 0.001, by - 0.001, shroud_h + 0.001, align=Align3(z="min"))
        inner = prog.locate(inner, pos=(0, 0, rz / 2))
        shroud = prog.cut(shroud, inner)
        body = prog.fuse(body, shroud)

    elif spec.connector_type == ConnectorType.XT30:
        # Panel mount flange
        flange_w = rx + 0.004
        flange_h = ry + 0.004
        flange_t = 0.001
        flange = prog.box(flange_w, flange_h, flange_t)
        fx = (
            (-rx / 2 - flange_t / 2)
            if mx > 0
            else (rx / 2 + flange_t / 2)
            if mx < 0
            else 0
        )
        flange = prog.locate(flange, pos=(fx, 0, 0))
        body = prog.fuse(body, flange)

    # Solder pins on bottom
    pin_h = 0.003
    n_pins = max(2, int(bx / 0.00254))
    if spec.connector_type != ConnectorType.GPIO_2X20:
        for i in range(min(n_pins, 6)):
            px = -bx / 2 + bx * (i + 0.5) / min(n_pins, 6)
            pin = prog.cylinder(0.0003, pin_h, align=Align3(z="max"))
            pin = prog.locate(pin, pos=(px, 0, -rz / 2))
            body = prog.fuse(body, pin)

    prog.output_ref = body
    return prog


def fastener_script(spec: FastenerSpec, length: float) -> ShapeScript:
    """Translate fastener_solid() to ShapeScript ops.

    The head (with chamfer and hex/Phillips recess) uses edge selection and
    RegularPolygon extrude — wrapped as PrebuiltOp. The shank is a native
    cylinder op.
    """
    from botcad.shapescript.ops import Align3

    prog = ShapeScript()

    head_h = spec.head_height
    shank_r = spec.thread_diameter / 2

    # Head with chamfer + socket recess — requires edge selection, use PrebuiltOp
    head_solid = _build_fastener_head(spec)
    head = prog.prebuilt(head_solid, tag="fastener_head")

    # Shank (extends from Z=-head_h downward)
    shank = prog.cylinder(shank_r, length, align=Align3(z="max"))
    shank = prog.locate(shank, pos=(0, 0, -head_h))

    result = prog.fuse(head, shank)
    prog.output_ref = result
    return prog


def _build_fastener_head(spec: FastenerSpec):
    """Build the fastener head solid via direct build123d (chamfer + recess).

    Extracted so fastener_script can wrap it as PrebuiltOp.
    """
    from build123d import (
        Align,
        Axis,
        Box,
        Cylinder,
        Location,
        RegularPolygon,
        chamfer,
        extrude,
    )

    from botcad.fasteners import HeadType

    head_r = spec.head_diameter / 2
    head_h = spec.head_height

    head = Cylinder(head_r, head_h, align=(Align.CENTER, Align.CENTER, Align.MAX))

    # Chamfer the top edge
    top_face = head.faces().sort_by(Axis.Z)[-1]
    chamfer_size = min(0.2 * head_h, 0.0003)
    head = chamfer(top_face.edges(), chamfer_size)

    if spec.head_type == HeadType.SOCKET_HEAD_CAP and spec.socket_size > 0:
        import math

        hex_r = spec.socket_size / 2 / math.cos(math.radians(30))
        recess_depth = spec.head_height * 0.6
        hex_profile = RegularPolygon(hex_r, 6)
        hex_solid = extrude(hex_profile, recess_depth)
        hex_solid = hex_solid.moved(Location((0, 0, -recess_depth)))
        head = head - hex_solid

    elif spec.head_type == HeadType.PAN_HEAD_PHILLIPS:
        slot_w = spec.head_diameter * 0.12
        slot_l = spec.head_diameter * 0.7
        slot_d = spec.head_height * 0.4
        slot1 = Box(
            slot_l, slot_w, slot_d, align=(Align.CENTER, Align.CENTER, Align.MAX)
        )
        slot2 = Box(
            slot_w, slot_l, slot_d, align=(Align.CENTER, Align.CENTER, Align.MAX)
        )
        head = head - slot1 - slot2

    return head


# ── Wheel / Wire Channel / Clearance Volume emitters ─────────────


def wheel_script(radius: float, width: float) -> ShapeScript:
    """Translate _make_wheel_solid() to ShapeScript ops.

    Pololu-style wheel: tire ring with 60 tread grooves, rim, hub disc
    with 6 M3 holes, 6 spokes with M2 accessory slots, center bore.
    All rotation loops (treads, holes, spokes) use native LocateOp with
    euler angles — no PrebuiltOp needed.
    """
    import math

    prog = ShapeScript()

    # Radial dimensions (proportional to wheel radius)
    tire_r = radius
    rim_outer_r = radius * 0.889
    rim_inner_r = radius * 0.756
    spoke_inner_r = radius * 0.267
    hub_r = spoke_inner_r
    bore_r = 0.0029  # 5.8mm diameter (25T spline)

    # Axial widths
    tire_w = width
    rim_w = width
    spoke_w = width * 0.5
    hub_w = width * 0.68

    # --- Tire ring (silicone rubber) ---
    tire_outer = prog.cylinder(tire_r, tire_w)
    tire_inner = prog.cylinder(rim_outer_r, tire_w + 0.001)
    tire = prog.cut(tire_outer, tire_inner)

    # --- Treads (60 horizontal grooves) ---
    # Fuse all tread boxes first, then cut once from tire (avoids 60 sequential
    # boolean cuts that cause cascading OCCT numerical issues).
    n_treads = 60
    tread_w = 0.0015
    tread_depth = 0.0008
    all_treads = None
    for i in range(n_treads):
        angle = i * (360 / n_treads)
        tread = prog.box(tread_depth * 2, tire_w + 0.001, tread_w)
        tread = prog.locate(tread, pos=(radius, 0, 0), euler_deg=(0, angle, 0))
        if all_treads is None:
            all_treads = tread
        else:
            all_treads = prog.fuse(all_treads, tread)
    tire = prog.cut(tire, all_treads)

    # --- Rim ring ---
    rim_outer_cyl = prog.cylinder(rim_outer_r, rim_w)
    rim_inner_cyl = prog.cylinder(rim_inner_r, rim_w + 0.001)
    rim = prog.cut(rim_outer_cyl, rim_inner_cyl)

    # --- Hub disc ---
    hub = prog.cylinder(hub_r, hub_w)

    # --- M3 mounting holes (6x) ---
    m3_r = 0.0016
    for i in range(6):
        angle_rad = i * math.pi / 3
        r_bcd = 0.00635 if (i % 3 == 0) else 0.00955
        mx = r_bcd * math.cos(angle_rad)
        my = r_bcd * math.sin(angle_rad)
        hole = prog.cylinder(m3_r, hub_w + 0.002)
        hole = prog.locate(hole, pos=(mx, my, 0))
        hub = prog.cut(hub, hole)

    # --- 6 spokes connecting hub to rim ---
    spoke_tangential = radius * 0.16
    spoke_length = rim_inner_r - spoke_inner_r + 0.004
    mid_r = (spoke_inner_r + rim_inner_r) / 2

    spokes_ref = None
    for i in range(6):
        angle_rad = i * math.pi / 3
        angle_deg = math.degrees(angle_rad)
        cx = mid_r * math.cos(angle_rad)
        cy = mid_r * math.sin(angle_rad)

        spoke = prog.box(spoke_length, spoke_tangential, spoke_w)
        spoke = prog.locate(spoke, pos=(cx, cy, 0), euler_deg=(0, 0, angle_deg))

        # --- M2 accessory slot in spoke ---
        slot_r_start = spoke_inner_r + 0.006
        slot_r_end = rim_inner_r - 0.004
        slot_dia = 0.0022
        slot_len = slot_r_end - slot_r_start
        slot_thick = spoke_w + 0.002
        slot_mid = (slot_r_start + slot_r_end) / 2

        # Slot = box + 2 end cylinders (stadium shape)
        s_box = prog.box(slot_len - slot_dia, slot_dia, slot_thick)
        c1 = prog.cylinder(slot_dia / 2, slot_thick)
        c1 = prog.locate(c1, pos=((slot_len - slot_dia) / 2, 0, 0))
        c2 = prog.cylinder(slot_dia / 2, slot_thick)
        c2 = prog.locate(c2, pos=(-(slot_len - slot_dia) / 2, 0, 0))
        slot = prog.fuse(prog.fuse(s_box, c1), c2)

        # Place slot at radial midpoint, then rotate to spoke angle
        slot = prog.locate(slot, pos=(slot_mid, 0, 0))
        slot = prog.locate(slot, euler_deg=(0, 0, angle_deg))

        spoke = prog.cut(spoke, slot)

        if spokes_ref is None:
            spokes_ref = spoke
        else:
            spokes_ref = prog.fuse(spokes_ref, spoke)

    # --- Center bore ---
    bore = prog.cylinder(bore_r, width + 0.002)

    # --- Assembly: hub + spokes + rim + tire - bore ---
    wheel = prog.fuse(hub, spokes_ref)
    wheel = prog.fuse(wheel, rim)
    wheel = prog.fuse(wheel, tire)
    wheel = prog.cut(wheel, bore)

    prog.output_ref = wheel
    return prog


# ── Wire channel radius table (mirrors cad.py:_CHANNEL_RADIUS) ──

_CHANNEL_RADIUS = {
    "uart_half_duplex": 0.0015,
    "csi": 0.003,
    "power": 0.002,
}


def emit_wire_channel(prog: ShapeScript, seg, bus_type: BusType) -> ShapeRef | None:
    """Emit ShapeScript ops for a cylindrical wire channel along a segment.

    Translates cad.py:_wire_channel() to native Cylinder + LocateOp.
    Returns None for degenerate (too-short) segments.
    """
    import math


    dx = seg.end[0] - seg.start[0]
    dy = seg.end[1] - seg.start[1]
    dz = seg.end[2] - seg.start[2]
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if length < 0.001:
        return None

    radius = _CHANNEL_RADIUS.get(str(bus_type), 0.0015)

    mid = (
        (seg.start[0] + seg.end[0]) / 2,
        (seg.start[1] + seg.end[1]) / 2,
        (seg.start[2] + seg.end[2]) / 2,
    )

    # Compute euler angles to rotate Z-aligned cylinder to segment direction
    ax, ay, az = dx / length, dy / length, dz / length

    if abs(az) > 0.999:
        if az < 0:
            euler = (180.0, 0.0, 0.0)
        else:
            euler = (0.0, 0.0, 0.0)
    else:
        from botcad.emit.cad import _axis_angle_to_quat
        from botcad.geometry import quat_to_euler

        rot_angle = math.acos(max(-1.0, min(1.0, az)))
        rot_axis_x = -ay
        rot_axis_y = ax
        rot_mag = math.sqrt(rot_axis_x**2 + rot_axis_y**2)
        if rot_mag < 1e-9:
            euler = (0.0, 0.0, 0.0)
        else:
            quat = _axis_angle_to_quat(
                (rot_axis_x / rot_mag, rot_axis_y / rot_mag, 0), rot_angle
            )
            euler = quat_to_euler(quat)

    channel = prog.cylinder(radius, length, tag=f"wire_{bus_type}")
    channel = prog.locate(channel, pos=mid, euler_deg=euler)
    return channel


def _emit_envelope_local(prog: ShapeScript, child: Body) -> ShapeRef | None:
    """Emit ShapeScript ops for a child body's outer envelope in Z-up local frame.

    Translates cad.py:_child_outer_envelope_local() to native ops.
    """
    from botcad.skeleton import BodyShape

    dims = child.dimensions
    tol = 0.005 if child.is_wheel_body else 0.0005

    if child.shape is BodyShape.CYLINDER:
        r = (child.radius or dims[0] / 2) + tol
        nominal_h = child.width or child.height or dims[2]
        if child.is_wheel_body:
            h = nominal_h + tol
            outer = prog.cylinder(r, h)
            outer = prog.locate(outer, pos=(0, 0, tol / 2))
        else:
            h = nominal_h + 2 * tol
            outer = prog.cylinder(r, h)
            outer = prog.locate(outer, pos=(0, 0, h / 2))
        return outer

    elif child.shape is BodyShape.TUBE:
        r = (child.outer_r or dims[0] / 2) + tol
        length = (child.length or dims[2]) + tol
        outer = prog.cylinder(r, length)
        outer = prog.locate(outer, pos=(0, 0, length / 2))
        return outer

    elif child.shape is BodyShape.SPHERE:
        r = (child.radius or dims[0] / 2) + tol
        return prog.sphere(r)

    elif child.shape is BodyShape.JAW:
        jw = (child.jaw_width or dims[0]) + 2 * tol
        jt = (child.jaw_thickness or dims[1]) * 2 + 2 * tol
        jl = (child.jaw_length or dims[2]) + tol
        from botcad.shapescript.ops import ALIGN_MIN_Z

        return prog.box(jw, jt, jl, align=ALIGN_MIN_Z)

    else:  # box
        return prog.box(
            dims[0] + 2 * tol, dims[1] + 2 * tol, dims[2] + 2 * tol
        )


def emit_child_clearance(
    prog: ShapeScript, child: Body, joint: Joint
) -> ShapeRef | None:
    """Emit ShapeScript ops for a child body's swept clearance volume.

    Translates cad.py:_child_clearance_volume() to native ops:
    1. Build envelope as ShapeScript primitives
    2. Sweep via LocateOp rotations + FuseOp union chain
    3. Place with orientation + position LocateOp

    Uses native ops throughout — no PrebuiltOp.
    """
    import math

    from botcad.emit.cad import _axis_to_quat
    from botcad.geometry import quat_to_euler
    from botcad.skeleton import BodyShape

    envelope = _emit_envelope_local(prog, child)
    if envelope is None:
        return None

    # Sweep around local Z axis for non-symmetric shapes
    is_symmetric = child.shape in (BodyShape.SPHERE, BodyShape.CYLINDER)
    if is_symmetric:
        clearance = envelope
    else:
        lo, hi = joint.effective_range
        if lo == 0.0 and hi == 0.0:
            lo, hi = 0.0, 2 * math.pi

        range_deg = abs(math.degrees(hi - lo))
        n_samples = max(8, int(range_deg / 15))

        clearance = envelope  # angle=0 (rest pose)
        for i in range(1, n_samples + 1):
            angle = lo + (hi - lo) * i / n_samples
            angle_deg = math.degrees(angle)
            rotated = prog.locate(envelope, euler_deg=(0, 0, angle_deg))
            clearance = prog.fuse(clearance, rotated)

    # Place: orientation from axis + position from joint
    orient_quat = _axis_to_quat(joint.axis)
    euler = quat_to_euler(orient_quat)

    if child.is_wheel_body:
        offset = joint.wheel_outboard_offset()
        ax, ay, az = joint.axis
        pos = (
            joint.pos[0] + ax * offset,
            joint.pos[1] + ay * offset,
            joint.pos[2] + az * offset,
        )
    else:
        pos = joint.pos

    clearance = prog.locate(clearance, pos=pos, euler_deg=euler)
    return clearance
