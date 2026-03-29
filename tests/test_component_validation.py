"""Tests for component-level wire collision validation."""

from __future__ import annotations

from botcad.component import BusType, Component, WirePort
from botcad.component_validation import (
    CollisionKind,
    validate_wire_collisions,
)
from botcad.units import Meters, grams, mm, mm3


def test_same_position_collides() -> None:
    """Two ports at the same position with the same connector must collide."""
    comp = Component(
        name="test_overlap",
        dimensions=mm3(50, 30, 10),
        mass=grams(20),
        wire_ports=(
            WirePort(
                "a",
                pos=(mm(5), Meters(0.0), Meters(0.0)),
                bus_type=BusType.POWER,
                connector_type="xt30",
            ),
            WirePort(
                "b",
                pos=(mm(5), Meters(0.0), Meters(0.0)),
                bus_type=BusType.POWER,
                connector_type="xt30",
            ),
        ),
    )
    findings = validate_wire_collisions(comp)
    assert len(findings) >= 1
    connector_findings = [f for f in findings if f.kind == CollisionKind.CONNECTOR]
    assert len(connector_findings) == 1
    assert connector_findings[0].overlap_mm > 0


def test_far_apart_no_collision() -> None:
    """Two ports far apart should produce no findings."""
    comp = Component(
        name="test_far",
        dimensions=mm3(200, 50, 10),
        mass=grams(30),
        wire_ports=(
            WirePort(
                "left",
                pos=(mm(-80), Meters(0.0), Meters(0.0)),
                bus_type=BusType.POWER,
                connector_type="xt30",
            ),
            WirePort(
                "right",
                pos=(mm(80), Meters(0.0), Meters(0.0)),
                bus_type=BusType.POWER,
                connector_type="xt30",
            ),
        ),
    )
    findings = validate_wire_collisions(comp)
    assert len(findings) == 0


def test_all_components() -> None:
    """Run validation on every component spec — print findings."""
    from botcad.components.battery import LiPo2S
    from botcad.components.bec import BEC5V
    from botcad.components.camera import OV5647, PiCamera2, PiCamera3, PiCamera3Wide
    from botcad.components.compute import RaspberryPiZero2W
    from botcad.components.controller import WaveshareSerialBus
    from botcad.components.servo import SCS0009, STS3215, STS3250

    factories = [
        STS3215,
        STS3250,
        SCS0009,
        LiPo2S,
        RaspberryPiZero2W,
        WaveshareSerialBus,
        BEC5V,
        OV5647,
        PiCamera2,
        PiCamera3,
        PiCamera3Wide,
    ]

    for factory in factories:
        comp = factory()
        findings = validate_wire_collisions(comp)
        if findings:
            print(f"\n{comp.name}: {len(findings)} finding(s)")
            for f in findings:
                print(
                    f"  {f.port_a} <-> {f.port_b}: {f.kind} overlap={f.overlap_mm:.2f}mm"
                )
        else:
            print(f"\n{comp.name}: no wire collisions")


def test_stub_collision_converging_directions() -> None:
    """Stubs collide (converging directions) but connectors don't."""
    # Port A: 5264_3pin at origin, exit direction (0, 0, -1)
    # Port B: usb_c at (0, 0, -12.5mm), exit direction (-1, 0, 0)
    # Connector housings don't overlap (12.5mm Z separation > sum of half-extents)
    # But stubs cross paths in 3D space
    comp = Component(
        name="test_stub_only",
        dimensions=mm3(60, 30, 40),
        mass=grams(15),
        wire_ports=(
            WirePort(
                "a",
                pos=(Meters(0.0), Meters(0.0), Meters(0.0)),
                bus_type=BusType.UART_HALF_DUPLEX,
                connector_type="5264_3pin",
            ),
            WirePort(
                "b",
                pos=(Meters(0.0), Meters(0.0), mm(-12.5)),
                bus_type=BusType.USB,
                connector_type="usb_c",
            ),
        ),
    )
    findings = validate_wire_collisions(comp)

    connector_findings = [f for f in findings if f.kind == CollisionKind.CONNECTOR]
    stub_findings = [f for f in findings if f.kind == CollisionKind.STUB]

    assert len(connector_findings) == 0, (
        f"Expected no connector collision, got {connector_findings}"
    )
    assert len(stub_findings) == 1, f"Expected 1 stub collision, got {stub_findings}"
    assert stub_findings[0].overlap_mm > 0
