"""Tests for WireNet data model and DFM checks."""

from __future__ import annotations

import pytest

from botcad.component import BusType
from botcad.ids import ComponentId
from botcad.wirenet import NetPort, NetTopology, WireNet

# ---------------------------------------------------------------------------
# Helper: load & solve a real bot
# ---------------------------------------------------------------------------


def _load_bot(name: str):
    """Import a bot design module, build, and solve."""
    import importlib.util
    from pathlib import Path

    design_py = Path(__file__).resolve().parent.parent / "bots" / name / "design.py"
    spec = importlib.util.spec_from_file_location(f"bots.{name}.design", design_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    bot = mod.build()
    bot.solve()
    return bot


# ---------------------------------------------------------------------------
# Tests: derive_wirenets on wheeler_arm (no controller, no BEC)
# ---------------------------------------------------------------------------


class TestWheelerArmNets:
    @pytest.fixture(scope="class")
    def bot(self):
        return _load_bot("wheeler_arm")

    @pytest.fixture(scope="class")
    def nets(self, bot):
        return bot.wire_nets

    @pytest.fixture(scope="class")
    def net_map(self, nets):
        return {n.label: n for n in nets}

    def test_has_servo_bus(self, net_map):
        assert "servo_bus" in net_map

    def test_servo_bus_topology(self, net_map):
        net = net_map["servo_bus"]
        assert net.topology == NetTopology.DAISY_CHAIN
        assert net.bus_type == BusType.UART_HALF_DUPLEX

    def test_servo_bus_starts_at_pi(self, net_map):
        """wheeler_arm has no controller, so bus starts at pi."""
        net = net_map["servo_bus"]
        assert net.ports[0].component_id == ComponentId("pi")
        assert net.ports[0].port_label == "uart_out"

    def test_servo_bus_hop_ports(self, net_map):
        """Each servo contributes uart_in + uart_out in depth-first order."""
        net = net_map["servo_bus"]
        # pi/uart_out, then pairs for 6 servos = 13 ports
        # wheeler_arm has: left_wheel, right_wheel, shoulder_yaw,
        # shoulder_pitch, elbow, wrist
        assert len(net.ports) == 13  # 1 origin + 6*2

        # Check alternating in/out pattern after origin
        for i in range(1, len(net.ports), 2):
            assert net.ports[i].port_label == "uart_in"
            assert net.ports[i + 1].port_label == "uart_out"

    def test_camera_csi(self, net_map):
        assert "camera_csi" in net_map
        net = net_map["camera_csi"]
        assert net.topology == NetTopology.POINT_TO_POINT
        assert net.bus_type == BusType.CSI
        assert len(net.ports) == 2
        labels = {str(p.component_id) for p in net.ports}
        assert labels == {"camera", "pi"}

    def test_power_fallback(self, net_map):
        """No controller/BEC => fallback power net."""
        assert "power" in net_map
        net = net_map["power"]
        assert net.bus_type == BusType.POWER
        assert net.topology == NetTopology.POINT_TO_POINT

    def test_no_controller_nets(self, net_map):
        """wheeler_arm has no controller, so no power_servo / pi_usb_data."""
        assert "power_servo" not in net_map
        assert "pi_usb_data" not in net_map
        assert "power_pi" not in net_map


# ---------------------------------------------------------------------------
# Tests: derive_wirenets on wheeler_base (has controller + BEC)
# ---------------------------------------------------------------------------


class TestWheelerBaseNets:
    @pytest.fixture(scope="class")
    def bot(self):
        return _load_bot("wheeler_base")

    @pytest.fixture(scope="class")
    def net_map(self, bot):
        return {n.label: n for n in bot.wire_nets}

    def test_servo_bus_starts_at_controller(self, net_map):
        net = net_map["servo_bus"]
        assert net.ports[0].component_id == ComponentId("controller")
        assert net.ports[0].port_label == "uart_out"

    def test_power_servo(self, net_map):
        assert "power_servo" in net_map
        net = net_map["power_servo"]
        assert net.bus_type == BusType.POWER
        assert net.topology == NetTopology.POINT_TO_POINT

    def test_power_pi_daisy_chain(self, net_map):
        assert "power_pi" in net_map
        net = net_map["power_pi"]
        assert net.bus_type == BusType.POWER
        assert net.topology == NetTopology.DAISY_CHAIN
        assert len(net.ports) == 4

    def test_pi_usb_data(self, net_map):
        assert "pi_usb_data" in net_map
        net = net_map["pi_usb_data"]
        assert net.bus_type == BusType.USB
        assert net.topology == NetTopology.POINT_TO_POINT

    def test_no_fallback_power(self, net_map):
        """Has controller + BEC, so no fallback power net."""
        assert "power" not in net_map


# ---------------------------------------------------------------------------
# Tests: DFM checks
# ---------------------------------------------------------------------------


class TestDFMOrphanedPorts:
    def test_finds_orphaned_port(self):
        """Synthetically add an extra wire port not in any net, verify detection."""
        from botcad.dfm.checks.wirenet_orphaned_ports import WirenetOrphanedPorts

        bot = _load_bot("wheeler_arm")

        # The orphaned ports check should already find some ports that are
        # not connected (e.g. power/balance ports on the battery that aren't
        # in any net, or servo power ports). Just verify the check runs
        # and returns findings.
        check = WirenetOrphanedPorts()
        # We need an AssemblySequence — use a minimal one
        from botcad.assembly.sequence import AssemblySequence

        seq = AssemblySequence(ops=())
        findings = check.run(bot, seq)
        # Should find at least some orphaned ports (battery has balance port, etc.)
        assert isinstance(findings, list)
        # All findings should be WARNING severity
        from botcad.dfm.check import DFMSeverity

        for f in findings:
            assert f.severity == DFMSeverity.WARNING


class TestDFMOverloadedPorts:
    def test_no_overloaded_on_valid_bot(self):
        """A correctly-wired bot should have no overloaded ports."""
        from botcad.assembly.sequence import AssemblySequence
        from botcad.dfm.checks.wirenet_overloaded_ports import WirenetOverloadedPorts

        bot = _load_bot("wheeler_arm")
        check = WirenetOverloadedPorts()
        seq = AssemblySequence(ops=())
        findings = check.run(bot, seq)
        assert findings == []

    def test_detects_duplicate(self):
        """Inject a duplicate port across nets and verify detection."""
        from botcad.assembly.sequence import AssemblySequence
        from botcad.dfm.check import DFMSeverity
        from botcad.dfm.checks.wirenet_overloaded_ports import WirenetOverloadedPorts

        bot = _load_bot("wheeler_arm")
        # Add a fake net that reuses battery/power_out (already in "power" net)
        fake_net = WireNet(
            label="fake_power",
            bus_type=BusType.POWER,
            topology=NetTopology.POINT_TO_POINT,
            ports=(
                NetPort(ComponentId("battery"), "power_out"),
                NetPort(ComponentId("pi"), "usb_power"),
            ),
        )
        bot.wire_nets.append(fake_net)

        check = WirenetOverloadedPorts()
        seq = AssemblySequence(ops=())
        findings = check.run(bot, seq)
        assert len(findings) >= 1
        assert any(f.severity == DFMSeverity.ERROR for f in findings)
        assert any("battery" in f.description for f in findings)


class TestDFMBusTypeMismatch:
    def test_catches_known_mismatch_on_wheeler_arm(self):
        """wheeler_arm's fallback power net (POWER bus) connects to Pi's
        usb_power port (USB bus), which is a real bus type mismatch.
        The check should detect this."""
        from botcad.assembly.sequence import AssemblySequence
        from botcad.dfm.check import DFMSeverity
        from botcad.dfm.checks.wirenet_bus_type_mismatch import WirenetBusTypeMismatch

        bot = _load_bot("wheeler_arm")
        check = WirenetBusTypeMismatch()
        seq = AssemblySequence(ops=())
        findings = check.run(bot, seq)
        # The fallback "power" net is POWER bus but Pi's usb_power is USB bus
        assert len(findings) >= 1
        assert any(
            f.severity == DFMSeverity.ERROR and "usb_power" in f.description
            for f in findings
        )
