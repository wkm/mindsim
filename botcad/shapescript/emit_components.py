"""ShapeScript emitters for component factories.

Translates camera_solid(), battery_solid(), _make_bearing_solid(),
_horn_solid(), connector_solid(), receptacle_solid(), fastener_solid(),
_make_wheel_solid(), _wire_channel(), and _child_clearance_volume()
into ShapeScript IR programs or inline ops.

For components that use fillets (battery cells) or edge-selection chamfers
(fastener heads), those parts are injected as PrebuiltOp since ShapeScript
doesn't yet support edge-selection-based modifications on primitives.
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

    Battery cells use fillets which require edge selection — the filleted
    cell body is injected as a PrebuiltOp. Label and cable exit are simple
    box primitives.
    """
    from build123d import Align, Box, Location

    from botcad.cad_utils import as_solid as _as_solid

    prog = ShapeScript()
    w, length, h = spec.dimensions

    # 1. Main battery body (filleted cells — prebuilt)
    C = (Align.CENTER, Align.CENTER, Align.CENTER)
    if spec.cells_s == 2:
        cell_w = length / 2 - 0.001
        cell = Box(w, cell_w, h, align=C)
        cell = _as_solid(cell).fillet(0.003, cell.edges())
        body_solid = cell.locate(Location((0, -length / 4, 0))).fuse(
            cell.locate(Location((0, length / 4, 0)))
        )
    else:
        body_solid = Box(w, length, h, align=C)
        body_solid = _as_solid(body_solid).fillet(0.003, body_solid.edges())

    body = prog.prebuilt(body_solid, tag="battery_body")

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
