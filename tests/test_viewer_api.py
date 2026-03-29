"""Tests for the viewer REST API data pipeline.

Validates the Python functions that serialize component and bot data
for the web viewer, without requiring build123d/OCCT.
"""

import json
from pathlib import Path
from typing import ClassVar
from xml.etree import ElementTree as ET

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BOTS_DIR = Path(__file__).resolve().parent.parent / "bots"
BOT_DIRS = sorted(p.parent for p in BOTS_DIR.glob("*/viewer_manifest.json"))


@pytest.fixture(scope="session")
def component_registry():
    """Build the component registry once per test session."""
    from mindsim.server import _build_component_registry

    return _build_component_registry()


# ---------------------------------------------------------------------------
# _build_component_registry
# ---------------------------------------------------------------------------


class TestBuildComponentRegistry:
    def test_non_empty(self, component_registry):
        assert len(component_registry) > 0, (
            "registry should contain at least one component"
        )

    def test_values_are_tuples(self, component_registry):
        from botcad.component import Component

        for name, entry in component_registry.items():
            assert isinstance(entry, tuple) and len(entry) == 3, (
                f"{name}: expected (callable, Component, str)"
            )
            factory, instance, module_name = entry
            assert callable(factory), f"{name}: factory not callable"
            assert isinstance(instance, Component), f"{name}: instance not a Component"
            assert isinstance(module_name, str), f"{name}: module name not str"

    def test_names_are_unique(self, component_registry):
        names = list(component_registry.keys())
        assert len(names) == len(set(names)), "duplicate component names"


# ---------------------------------------------------------------------------
# _component_to_json
# ---------------------------------------------------------------------------


class TestComponentToJson:
    REQUIRED_KEYS: ClassVar[set[str]] = {
        "name",
        "category",
        "dimensions_mm",
        "mass_g",
        "is_servo",
        "color",
        "layers",
        "mounting_points",
        "wire_ports",
        "drawings",
    }

    def test_required_keys_present(self, component_registry):
        from mindsim.server import _component_to_json

        for name, (_factory, comp, category) in component_registry.items():
            info = _component_to_json(comp, category)
            missing = self.REQUIRED_KEYS - info.keys()
            assert not missing, f"{name} missing keys: {missing}"

    def test_servo_has_servo_subdict(self, component_registry):
        from botcad.component import ServoSpec
        from mindsim.server import _component_to_json

        for name, (_factory, comp, category) in component_registry.items():
            info = _component_to_json(comp, category)
            if isinstance(comp, ServoSpec):
                assert "servo" in info, f"{name}: servo component missing 'servo' key"
                servo = info["servo"]
                for key in (
                    "stall_torque_nm",
                    "no_load_speed_rpm",
                    "voltage",
                    "range_deg",
                    "gear_ratio",
                    "continuous",
                ):
                    assert key in servo, f"{name}: servo sub-dict missing '{key}'"

    def test_lists_are_lists(self, component_registry):
        from mindsim.server import _component_to_json

        for name, (_factory, comp, category) in component_registry.items():
            info = _component_to_json(comp, category)
            assert isinstance(info["mounting_points"], list), (
                f"{name}: mounting_points not list"
            )
            assert isinstance(info["wire_ports"], list), f"{name}: wire_ports not list"
            assert isinstance(info["dimensions_mm"], list), (
                f"{name}: dimensions_mm not list"
            )
            assert isinstance(info["color"], list), f"{name}: color not list"

    def test_json_serializable(self, component_registry):
        from mindsim.server import _component_to_json

        for name, (_factory, comp, category) in component_registry.items():
            info = _component_to_json(comp, category)
            try:
                json.dumps(info)
            except (TypeError, ValueError) as exc:
                pytest.fail(f"{name}: not JSON-serializable: {exc}")


# ---------------------------------------------------------------------------
# _component_layers
# ---------------------------------------------------------------------------


class TestComponentLayers:
    def test_servo_layers(self, component_registry):
        from botcad.component import ServoSpec
        from mindsim.server import _component_layers

        expected_base = {
            "servo",
            "bracket",
            "cradle",
            "coupler",
            "bracket_insertion_channel",
            "cradle_insertion_channel",
        }
        for name, (_factory, comp, _cat) in component_registry.items():
            if isinstance(comp, ServoSpec):
                layers = _component_layers(comp)
                assert isinstance(layers, list)
                assert expected_base.issubset(set(layers)), (
                    f"{name}: servo missing layers, got {layers}"
                )

    def test_plain_component_has_body(self, component_registry):
        from botcad.component import ServoSpec
        from mindsim.server import _component_layers

        for name, (_factory, comp, _cat) in component_registry.items():
            if not isinstance(comp, ServoSpec):
                layers = _component_layers(comp)
                assert "body" in layers, f"{name}: plain component missing 'body'"


# ---------------------------------------------------------------------------
# Manifest schema validation (per bot)
# ---------------------------------------------------------------------------


class TestManifestSchema:
    @pytest.fixture(params=[p.name for p in BOT_DIRS], ids=lambda n: n)
    def manifest(self, request):
        path = BOTS_DIR / request.param / "viewer_manifest.json"
        return json.loads(path.read_text())

    def test_bodies(self, manifest):
        assert "bodies" in manifest
        bodies = manifest["bodies"]
        assert isinstance(bodies, list) and len(bodies) > 0
        for body in bodies:
            for key in ("name", "parent"):
                assert key in body, f"body missing '{key}': {body.get('name', '?')}"
            # mounts is optional (not all bodies have components)
            if "mounts" in body:
                assert isinstance(body["mounts"], list)

    def test_joints(self, manifest):
        assert "joints" in manifest
        joints = manifest["joints"]
        assert isinstance(joints, list) and len(joints) > 0
        for joint in joints:
            for key in ("name", "parent_body", "child_body"):
                assert key in joint, f"joint missing '{key}': {joint.get('name', '?')}"


# ---------------------------------------------------------------------------
# Manifest ↔ XML body count consistency
# ---------------------------------------------------------------------------


class TestManifestXmlConsistency:
    @pytest.fixture(params=[p.name for p in BOT_DIRS], ids=lambda n: n)
    def bot_name(self, request):
        return request.param

    def test_body_count_matches_xml(self, bot_name):
        manifest_path = BOTS_DIR / bot_name / "viewer_manifest.json"
        xml_path = BOTS_DIR / bot_name / "bot.xml"
        if not xml_path.exists():
            pytest.skip(f"{bot_name} has no bot.xml")

        manifest = json.loads(manifest_path.read_text())
        manifest_bodies = len(manifest.get("bodies", []))

        tree = ET.parse(xml_path)
        xml_bodies = len(tree.findall(".//{http://www.mujoco.org/}body[@name]"))
        if xml_bodies == 0:
            # Try without namespace (some bots use plain XML)
            xml_bodies = len(tree.findall(".//body[@name]"))

        # Manifest includes component/servo visualization bodies beyond
        # the structural bodies in MuJoCo XML, so manifest >= xml.
        assert manifest_bodies >= xml_bodies, (
            f"{bot_name}: manifest has {manifest_bodies} bodies, XML has {xml_bodies}"
        )
