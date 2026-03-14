"""Battery components."""

from __future__ import annotations

from botcad.component import BatterySpec, BusType, WirePort


def LiPo2S(capacity_mah: int = 1000) -> BatterySpec:
    """LiPo 2S battery pack.

    Args:
        capacity_mah: Capacity in mAh. Affects dimensions and mass.
    """
    # Scale dimensions roughly with capacity (1000mAh baseline)
    scale = capacity_mah / 1000.0
    length = 0.073 * scale**0.33  # grows slowly with capacity
    width = 0.035
    height = 0.018 * scale**0.33

    mass = 0.055 * scale  # ~55g per 1000mAh

    return BatterySpec(
        name=f"LiPo2S-{capacity_mah}",
        dimensions=(length, width, height),
        mass=mass,
        wire_ports=(
            # XT30 power connector on one end
            WirePort("power", pos=(length / 2, 0.0, 0.0), bus_type=BusType.POWER),
            # Balance lead
            WirePort("balance", pos=(length / 2, 0.01, 0.0), bus_type=BusType.BALANCE),
        ),
        color=(0.1, 0.1, 0.8, 1.0),  # Deep battery blue
        chemistry="LiPo",
        cells_s=2,
    )


def battery_solid(spec: BatterySpec):
    """Build a detailed parametric solid for a battery pack.

    Models the heat-shrink wrapped cells, cable exits, and a label.
    """
    from build123d import Align, Box, Location

    from botcad.emit.cad import _as_solid

    C = (Align.CENTER, Align.CENTER, Align.CENTER)
    w, length, h = spec.dimensions

    # 1. Main battery block (heat-shrink wrapped cells)
    # We'll model it as two cells side-by-side if it's 2S
    if spec.cells_s == 2:
        cell_w = length / 2 - 0.001
        cell = Box(w, cell_w, h, align=C)
        cell = _as_solid(cell).fillet(0.003, cell.edges())
        body = cell.locate(Location((0, -length / 4, 0))).fuse(
            cell.locate(Location((0, length / 4, 0)))
        )
    else:
        body = Box(w, length, h, align=C)
        body = _as_solid(body).fillet(0.003, body.edges())

    # 2. Label block (white rectangle on top)
    label_w = w * 0.7
    label_l = length * 0.6
    label_t = 0.0005  # 0.5mm thick for visibility
    label = Box(
        label_w, label_l, label_t, align=(Align.CENTER, Align.CENTER, Align.MIN)
    )
    label = label.locate(Location((0, 0, h / 2 - 0.0001)))  # slightly embedded

    # 3. Cable exit protrusions (small blocks where wires come out)
    exit_w = 0.012
    exit_l = 0.006
    exit_h = h * 0.8
    # Using Align.MIN so it protrudes from the face
    exit_block = Box(
        exit_l, exit_w, exit_h, align=(Align.MIN, Align.CENTER, Align.CENTER)
    )
    exit_block = exit_block.locate(Location((w / 2 - 0.001, 0, 0)))  # slightly embedded

    # Colors
    body.color = (0.1, 0.1, 0.7)  # Blue shrink wrap
    label.color = (0.9, 0.9, 0.9)  # White label
    exit_block.color = (0.2, 0.2, 0.2)  # Dark gray / black rubber exit

    # Combine
    res = body.fuse(label).fuse(exit_block)

    return _as_solid(res)
