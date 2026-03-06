"""Motor controller board components."""

from __future__ import annotations

from botcad.component import Component, MountPoint, WirePort


def WaveshareSerialBus() -> Component:
    """Waveshare Serial Bus Servo Driver Board.

    USB-C host interface, drives up to 253 STS/SCS servos via UART
    half-duplex bus. Powers servos from external 6-12V supply.

    Product: https://www.waveshare.com/bus-servo-adapter-a.htm
    Approx dims from product photos: 55 x 25 x 12mm, ~15g.
    """
    return Component(
        name="WaveshareSerialBus",
        dimensions=(0.055, 0.025, 0.012),
        mass=0.015,
        wire_ports=(
            WirePort("usb_c", pos=(-0.0275, 0.0, 0.0), bus_type="usb"),
            WirePort(
                "servo_bus",
                pos=(0.0275, 0.0, 0.0),
                bus_type="uart_half_duplex",
            ),
        ),
        mounting_points=(
            MountPoint("m1", pos=(-0.022, -0.009, 0.0), diameter=0.003),
            MountPoint("m2", pos=(0.022, -0.009, 0.0), diameter=0.003),
            MountPoint("m3", pos=(-0.022, 0.009, 0.0), diameter=0.003),
            MountPoint("m4", pos=(0.022, 0.009, 0.0), diameter=0.003),
        ),
        color=(0.0, 0.3, 0.6, 1.0),  # PCB blue
        voltage=5.0,
        typical_current=0.1,  # board logic only, servo power is separate
    )
