"""Targeted botcad pipeline tests."""

from xml.etree import ElementTree as ET

from botcad.component import Component, WirePort
from botcad.emit.render3d import SceneBuilder
from botcad.routing import solve_routing
from botcad.skeleton import Bot, ServoSpec
from botcad.units import Kg, mm3


def test_scene_builder_scopes_hinges_per_body() -> None:
    """Each add_hinge call should attach to the last added body only."""
    scene = SceneBuilder(width=100, height=100)

    scene.add_body("fixed", geoms=['<geom name="fixed_geom" type="box" size="1 1 1"/>'])
    scene.add_hinge("fixed_joint", axis=(1, 0, 0), range_rad=(0.0, 1.0))

    scene.add_body(
        "moving",
        pos=(0.1, 0.0, 0.0),
        geoms=['<geom name="moving_geom" type="box" size="1 1 1"/>'],
    )
    scene.add_hinge("moving_joint", axis=(0, 1, 0), range_rad=(-1.0, 1.0))

    root = ET.fromstring(scene.to_xml())
    root_body = root.find("./worldbody/body[@name='root']")
    assert root_body is not None

    body_xml = {b.attrib["name"]: b for b in root_body.findall("body")}
    fixed_body = body_xml["fixed"]
    moving_body = body_xml["moving"]

    fixed_joints = [j.attrib["name"] for j in fixed_body.findall("joint")]
    moving_joints = [j.attrib["name"] for j in moving_body.findall("joint")]

    assert fixed_joints == ["fixed_joint"]
    assert moving_joints == ["moving_joint"]


def test_route_servo_bus_enters_child_body_at_joint_origin() -> None:
    """Child-body servo connections should start at child-local origin."""
    servo = ServoSpec(name="servo", dimensions=mm3(10, 20, 30), mass=Kg(0.1))
    controller = Component(
        name="controller",
        dimensions=mm3(30, 30, 10),
        mass=Kg(0.1),
        wire_ports=(WirePort("uart", (0.0, 0.0, 0.0), "uart_half_duplex"),),
    )

    bot = Bot("test")
    base = bot.body("base")
    base.mount(controller, label="controller")

    j1 = base.joint("j1", servo, pos=(0.1, 0.0, 0.0), axis="z")
    mid = j1.body("mid")
    _ = mid.joint("j2", servo, pos=(0.1, 0.0, 0.0), axis="z").body("tip")

    bot._collect_tree()
    route = solve_routing(bot)[0]

    from botcad.ids import BodyId

    body_names = {segment.body_name for segment in route.segments}
    assert body_names == {BodyId("base"), BodyId("mid")}
    assert all(segment.body_name in body_names for segment in route.segments)
    assert route.segments[1].start == (0.0, 0.0, 0.0)
