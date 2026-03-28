"""Tests for wire channel sizing DFM check."""

import math

from botcad.assembly.build import build_assembly_sequence
from botcad.connectors import MOLEX_5264_3PIN
from botcad.dfm.check import DFMSeverity
from botcad.dfm.checks.wire_channel_sizing import (
    WireChannelSizing,
    _ConnectorCrossSection,
    _connectors_for_route,
    _max_connector_cross_section,
)


def _wheeler_base():
    from bots.wheeler_base.design import build

    bot = build()
    bot.solve()
    return bot


def test_check_name():
    check = WireChannelSizing()
    assert check.name == "wire_channel_sizing"


def test_connector_cross_section_molex():
    """Molex 5264 3-pin: 7.5 x 4.5 x 6.0mm body dimensions."""
    cs = _ConnectorCrossSection(MOLEX_5264_3PIN)
    assert cs.label == "Molex 5264 3-pin"
    # Sorted dims: 4.5, 6.0, 7.5 (in mm)
    # min_cross = sqrt(4.5^2 + 6.0^2) = sqrt(20.25 + 36) = sqrt(56.25) = 7.5mm
    expected_min = math.sqrt(0.0045**2 + 0.006**2)
    assert abs(cs.min_cross - expected_min) < 1e-6


def test_molex_does_not_fit_uart_channel():
    """UART wire channel is 3mm diameter; Molex 5264 needs ~7.5mm diagonal."""
    cs = _ConnectorCrossSection(MOLEX_5264_3PIN)
    uart_channel_diameter = 0.0015 * 2  # 3mm
    assert cs.min_cross > uart_channel_diameter, (
        f"Molex min cross-section {cs.min_cross * 1000:.1f}mm "
        f"should exceed channel diameter {uart_channel_diameter * 1000:.1f}mm"
    )


def test_max_connector_cross_section():
    """max picks the connector with the largest min_cross."""
    specs = [MOLEX_5264_3PIN]
    result = _max_connector_cross_section(specs)
    assert result.label == "Molex 5264 3-pin"


def test_wheeler_base_wire_channel_findings():
    """Wheeler_base servo UART channels are too small for Molex connectors."""
    bot = _wheeler_base()
    seq = build_assembly_sequence(bot)
    check = WireChannelSizing()
    findings = check.run(bot, seq, {})
    # Should have findings about connector vs channel sizing
    assert len(findings) > 0, (
        "Expected wire channel sizing findings for wheeler_base "
        "(Molex 5264 connectors vs 3mm wire channels)"
    )
    # At least one should be an error
    errors = [f for f in findings if f.severity == DFMSeverity.ERROR]
    assert len(errors) > 0, (
        f"Expected ERROR-level findings, got: {[f.title for f in findings]}"
    )


def test_findings_reference_servo_bus():
    """Findings should reference the servo_bus wire route."""
    bot = _wheeler_base()
    seq = build_assembly_sequence(bot)
    check = WireChannelSizing()
    findings = check.run(bot, seq, {})
    servo_bus_findings = [f for f in findings if "servo_bus" in f.id]
    assert len(servo_bus_findings) > 0, (
        f"Expected findings for servo_bus route, got: {[f.id for f in findings]}"
    )


def test_findings_have_valid_structure():
    """Each finding has required fields populated correctly."""
    bot = _wheeler_base()
    seq = build_assembly_sequence(bot)
    check = WireChannelSizing()
    findings = check.run(bot, seq, {})
    for f in findings:
        assert f.check_name == "wire_channel_sizing"
        assert f.pos is not None
        assert len(f.pos) == 3
        assert f.assembly_step >= 0
        assert f.measured is not None
        assert f.threshold is not None
        assert f.measured < f.threshold, (
            f"measured ({f.measured}) should be less than threshold ({f.threshold}) "
            f"for a finding"
        )


def test_connectors_for_servo_bus_route():
    """solve_routing produces a servo_bus route; we should find its connectors."""
    from botcad.routing import solve_routing

    bot = _wheeler_base()
    routes = solve_routing(bot)
    servo_bus = next((r for r in routes if r.label == "servo_bus"), None)
    assert servo_bus is not None, "Wheeler_base should have a servo_bus route"

    specs = _connectors_for_route(servo_bus, bot)
    assert len(specs) > 0, "servo_bus route should have at least one connector spec"
    # Should find the Molex 5264
    labels = [s.label for s in specs]
    assert any("5264" in lbl or "Molex" in lbl for lbl in labels), (
        f"Expected Molex 5264 connector, got: {labels}"
    )


def test_finding_id_is_deterministic():
    """Finding IDs should be stable across runs."""
    bot = _wheeler_base()
    seq = build_assembly_sequence(bot)
    check = WireChannelSizing()
    ids_1 = sorted(f.id for f in check.run(bot, seq, {}))
    ids_2 = sorted(f.id for f in check.run(bot, seq, {}))
    assert ids_1 == ids_2
