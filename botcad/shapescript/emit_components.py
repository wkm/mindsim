"""ShapeScript emitters for simple component factories.

Translates camera_solid(), battery_solid(), _make_bearing_solid(), and
_horn_solid() into ShapeScript IR programs. Each function returns a
ShapeScript whose output_ref is the final composite solid.

For components that use fillets (battery cells), the filleted body is
injected as a PrebuiltOp since ShapeScript doesn't yet support
edge-selection-based fillets on primitives without tags.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from botcad.shapescript.ops import Align3
from botcad.shapescript.program import ShapeScript

if TYPE_CHECKING:
    from botcad.component import BatterySpec, BearingSpec, CameraSpec, ServoSpec


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
