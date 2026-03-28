"""Battery components."""

from __future__ import annotations

from botcad.component import BatterySpec, BusType, WirePort
from botcad.materials import MAT_LIPO_WRAP
from botcad.units import Kg, Meters


def LiPo2S(capacity_mah: int = 1000) -> BatterySpec:
    """LiPo 2S battery pack.

    Args:
        capacity_mah: Capacity in mAh. Affects dimensions and mass.
    """
    # Scale dimensions roughly with capacity (1000mAh baseline)
    scale = capacity_mah / 1000.0
    length = Meters(0.073 * scale**0.33)  # grows slowly with capacity
    width = Meters(0.035)
    height = Meters(0.018 * scale**0.33)

    mass = Kg(0.055 * scale)  # ~55g per 1000mAh

    return BatterySpec(
        name=f"LiPo2S-{capacity_mah}",
        dimensions=(length, width, height),
        mass=mass,
        wire_ports=(
            # XT30 power connector on one end
            WirePort(
                "power",
                pos=(Meters(length / 2), Meters(0.0), Meters(0.0)),
                bus_type=BusType.POWER,
                connector_type="xt30",
            ),
            # Balance lead
            WirePort(
                "balance",
                pos=(Meters(length / 2), Meters(0.01), Meters(0.0)),
                bus_type=BusType.BALANCE,
                connector_type="jst_xh_3pin",
            ),
        ),
        default_material=MAT_LIPO_WRAP,
        chemistry="LiPo",
        cells_s=2,
    )
