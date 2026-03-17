"""Validation renders for component inspection.

Generates per-component PNGs with orthographic views showing body envelope,
mounting points, wire ports, axis indicators, and component-specific features.

Uses a unified visual language where the same color always means the same thing:
    Blue   = mounting points (screws, snap-fits)
    Orange = wire ports (connectors, cables)
    RGB    = axis indicators (X=red, Y=green, Z=blue)

ServoSpec components get additional detail: shaft, horn disc, horn holes,
rear holes, and mounting ears.

Called as part of Bot.emit() pipeline, or standalone:
    uv run python -m botcad.emit.component_renders [bot_dir]
"""

from __future__ import annotations

import math
import shutil
import tempfile
import time
from pathlib import Path

from PIL import Image

from botcad.component import Component, ServoSpec
from botcad.emit.composite import grid, save_png
from botcad.emit.render3d import (
    COLOR_BRACKET,
    COLOR_COUPLER,
    COLOR_CRADLE,
    COLOR_ENVELOPE,
    COLOR_HORN,
    COLOR_HORN_HOLE,
    COLOR_MOUNTING,
    COLOR_REAR_HOLE,
    COLOR_SERVO_BODY,
    COLOR_SHAFT,
    COLOR_WIRE_PORT,
    VIEWS_6,
    Color,
    Renderer3D,
    SceneBuilder,
)

# ── Rendering config ──

WIDTH, HEIGHT = 800, 800  # per-view resolution


# ── Scene building: CAD geometry + annotation overlays ──
#
# Every physical part is rendered from real CAD geometry (build123d → STL → mesh).
# Annotation markers (spheres, cylinders) are non-physical overlays that highlight
# feature locations — mounting holes, wire ports, shaft positions, etc.


def _body_render_color(component: Component) -> Color:
    """Compute a render-safe body color, lightening very dark components."""
    c = component.color
    min_lum = 0.35
    r, g, b = c[0], c[1], c[2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    if lum < min_lum and lum > 0:
        scale = min_lum / lum
        r, g, b = min(r * scale, 1.0), min(g * scale, 1.0), min(b * scale, 1.0)
    return Color(r, g, b, 0.85, "body")


def _add_axis_annotations(scene: SceneBuilder, dimensions: tuple) -> None:
    """Annotate XYZ axis indicators sized to the component dimensions."""
    axis_len = max(max(dimensions) * 1.0, 0.015)
    axis_r = 0.0005
    scene.annotate_cylinder(
        "axis_x",
        pos=(axis_len / 2, 0, 0),
        radius=axis_r,
        half_height=axis_len / 2,
        color=Color(1, 0, 0, 0.5),
        quat="0.7071068 0 0.7071068 0",
    )
    scene.annotate_cylinder(
        "axis_y",
        pos=(0, axis_len / 2, 0),
        radius=axis_r,
        half_height=axis_len / 2,
        color=Color(0, 1, 0, 0.5),
        quat="0.7071068 0.7071068 0 0",
    )
    scene.annotate_cylinder(
        "axis_z",
        pos=(0, 0, axis_len / 2),
        radius=axis_r,
        half_height=axis_len / 2,
        color=Color(0, 0, 1, 0.5),
    )


def _add_common_geoms(scene: SceneBuilder, component: Component) -> None:
    """Body mesh (CAD geometry) + axis annotation indicators."""
    scene.add_mesh("comp", "component.stl", _body_render_color(component))

    d = component.dimensions
    if isinstance(component, ServoSpec):
        d = component.effective_body_dims
    _add_axis_annotations(scene, d)


def _axis_quat(axis: tuple[float, float, float]) -> str | None:
    """MuJoCo quaternion (w x y z) rotating Z-up to *axis*. None if already Z-up."""
    ax, ay, az = axis
    length = math.sqrt(ax * ax + ay * ay + az * az)
    if length < 1e-9:
        return None
    ax, ay, az = ax / length, ay / length, az / length

    dot = az  # dot( (0,0,1), (ax,ay,az) )
    if dot > 1.0 - 1e-8:
        return None  # already aligned with +Z
    if dot < -1.0 + 1e-8:
        return "0 1 0 0"  # 180° around X

    # cross( (0,0,1), (ax,ay,az) ) = (-ay, ax, 0)
    cx, cy, cz = -ay, ax, 0.0
    half_angle = math.acos(max(-1.0, min(1.0, dot))) / 2.0
    s = math.sin(half_angle)
    cl = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx, cy, cz = cx / cl * s, cy / cl * s, cz / cl * s
    w = math.cos(half_angle)
    return f"{w:.7f} {cx:.7f} {cy:.7f} {cz:.7f}"


def _add_mounting_annotations(scene: SceneBuilder, component: Component) -> None:
    """Annotation cylinders showing mounting point locations and axes."""
    if not component.mounting_points:
        return
    for i, mp in enumerate(component.mounting_points):
        radius = max(mp.diameter / 2, 0.001)  # at least 1mm for visibility
        half_height = radius * 2.5  # screw-like proportions
        # Offset along axis so the fastener protrudes from the surface
        ax, ay, az = mp.axis
        pos = (
            mp.pos[0] + ax * half_height,
            mp.pos[1] + ay * half_height,
            mp.pos[2] + az * half_height,
        )
        scene.annotate_cylinder(
            f"mount_{i}",
            pos=pos,
            radius=radius,
            half_height=half_height,
            color=COLOR_MOUNTING,
            quat=_axis_quat(mp.axis),
        )


def _add_wire_port_annotations(scene: SceneBuilder, component: Component) -> None:
    """Annotation spheres showing wire port locations."""
    if not component.wire_ports:
        return
    for i, wp in enumerate(component.wire_ports):
        scene.annotate_sphere(
            f"wire_{i}", pos=wp.pos, size=0.003, color=COLOR_WIRE_PORT
        )


def _annotate_ears_and_shaft(scene: SceneBuilder, servo: ServoSpec) -> None:
    """Mounting ear and shaft annotations shared by servo and bracket scenes."""
    if servo.mounting_ears:
        for i, ear in enumerate(servo.mounting_ears):
            scene.annotate_sphere(
                f"ear_{i}", pos=ear.pos, size=0.002, color=COLOR_MOUNTING
            )

    so = servo.shaft_offset
    shaft_h = 0.008
    shaft_z = so[2] + shaft_h / 2
    scene.annotate_cylinder(
        "shaft",
        pos=(so[0], so[1], shaft_z),
        radius=0.003,
        half_height=shaft_h / 2,
        color=COLOR_SHAFT,
    )


def _add_servo_annotations(scene: SceneBuilder, servo: ServoSpec) -> None:
    """Servo-specific annotation markers: ears, shaft, horn, horn holes, rear holes."""
    _annotate_ears_and_shaft(scene, servo)

    # Horn disc (thin yellow cylinder at shaft)
    so = servo.shaft_offset
    horn_z = so[2] + 0.003
    scene.annotate_cylinder(
        "horn",
        pos=(so[0], so[1], horn_z),
        radius=0.01,
        half_height=0.0005,
        color=COLOR_HORN,
    )

    # Horn mounting holes (green spheres)
    if servo.horn_mounting_points:
        for i, mp in enumerate(servo.horn_mounting_points):
            scene.annotate_sphere(
                f"horn_mount_{i}", pos=mp.pos, size=0.0015, color=COLOR_HORN_HOLE
            )

    # Rear horn mounting holes (cyan spheres)
    if servo.rear_horn_mounting_points:
        for i, mp in enumerate(servo.rear_horn_mounting_points):
            scene.annotate_sphere(
                f"rear_mount_{i}", pos=mp.pos, size=0.0015, color=COLOR_REAR_HOLE
            )


def _add_bracket_annotations(scene: SceneBuilder, servo: ServoSpec) -> None:
    """Common annotations for bracket/cradle/coupler-assembly scenes."""
    _annotate_ears_and_shaft(scene, servo)

    # Wire port (orange)
    if servo.connector_pos:
        cp = servo.connector_pos
        scene.annotate_sphere("wire_0", pos=cp, size=0.003, color=COLOR_WIRE_PORT)


def _add_horn_annotations(scene: SceneBuilder, servo: ServoSpec) -> None:
    """Horn + rear hole annotations in servo body frame."""
    if servo.horn_mounting_points:
        for i, mp in enumerate(servo.horn_mounting_points):
            scene.annotate_sphere(
                f"horn_{i}", pos=mp.pos, size=0.0015, color=COLOR_HORN_HOLE
            )
    if servo.rear_horn_mounting_points:
        for i, mp in enumerate(servo.rear_horn_mounting_points):
            scene.annotate_sphere(
                f"rear_{i}", pos=mp.pos, size=0.0015, color=COLOR_REAR_HOLE
            )


def _render_scene(scene: SceneBuilder, temp_dir: Path, title: str) -> Image.Image:
    """Render all views, composite into a grid, and clean up temp_dir."""
    try:
        scene.set_mesh_dir(temp_dir)
        xml = scene.to_xml()
        with Renderer3D(xml, WIDTH, HEIGHT) as r:
            views = r.render_views(VIEWS_6)
        return grid(title, scene.legends, views)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ── Scene builders ──


def build_component_scene(component: Component) -> tuple[SceneBuilder, Path]:
    """Build scene showing any component with all its features.

    Returns (scene, temp_dir). Caller must clean up temp_dir.
    """
    from build123d import export_stl

    from botcad.emit.cad import make_component_solid

    temp_dir = Path(tempfile.mkdtemp(prefix="botcad_comp_"))
    solid = make_component_solid(component)
    export_stl(solid, str(temp_dir / "component.stl"))

    scene = SceneBuilder(model_name="component_debug", width=WIDTH, height=HEIGHT)
    _add_common_geoms(scene, component)
    _add_mounting_annotations(scene, component)
    _add_wire_port_annotations(scene, component)
    if isinstance(component, ServoSpec):
        _add_servo_annotations(scene, component)

    return scene, temp_dir


def build_bracket_scene(servo: ServoSpec) -> tuple[SceneBuilder, Path]:
    """Build scene showing bracket + servo + envelope + annotations."""
    from build123d import export_stl

    from botcad.bracket import BracketSpec, bracket_envelope, bracket_solid, servo_solid

    temp_dir = Path(tempfile.mkdtemp(prefix="botcad_bracket_"))
    export_stl(bracket_solid(servo, BracketSpec()), str(temp_dir / "bracket.stl"))
    export_stl(servo_solid(servo), str(temp_dir / "servo.stl"))
    export_stl(bracket_envelope(servo, BracketSpec()), str(temp_dir / "envelope.stl"))

    scene = SceneBuilder(model_name="bracket_debug", width=WIDTH, height=HEIGHT)
    scene.add_mesh("bracket", "bracket.stl", COLOR_BRACKET)
    scene.add_mesh("servo", "servo.stl", COLOR_SERVO_BODY)
    scene.add_mesh("envelope", "envelope.stl", COLOR_ENVELOPE)
    _add_bracket_annotations(scene, servo)
    _add_horn_annotations(scene, servo)

    return scene, temp_dir


def build_cradle_scene(servo: ServoSpec) -> tuple[SceneBuilder, Path]:
    """Build scene showing cradle + servo + envelope + annotations."""
    from build123d import export_stl

    from botcad.bracket import BracketSpec, cradle_envelope, cradle_solid, servo_solid

    temp_dir = Path(tempfile.mkdtemp(prefix="botcad_cradle_"))
    export_stl(cradle_solid(servo, BracketSpec()), str(temp_dir / "cradle.stl"))
    export_stl(servo_solid(servo), str(temp_dir / "servo.stl"))
    export_stl(cradle_envelope(servo, BracketSpec()), str(temp_dir / "envelope.stl"))

    scene = SceneBuilder(model_name="cradle_debug", width=WIDTH, height=HEIGHT)
    scene.add_mesh("cradle", "cradle.stl", COLOR_CRADLE)
    scene.add_mesh("servo", "servo.stl", COLOR_SERVO_BODY)
    scene.add_mesh("envelope", "envelope.stl", COLOR_ENVELOPE)
    _add_bracket_annotations(scene, servo)

    return scene, temp_dir


def build_coupler_scene(servo: ServoSpec) -> tuple[SceneBuilder, Path]:
    """Build scene showing coupler in shaft-centered frame."""
    from build123d import export_stl

    from botcad.bracket import BracketSpec, coupler_solid

    temp_dir = Path(tempfile.mkdtemp(prefix="botcad_coupler_"))
    export_stl(coupler_solid(servo, BracketSpec()), str(temp_dir / "coupler.stl"))

    scene = SceneBuilder(model_name="coupler_debug", width=WIDTH, height=HEIGHT)
    scene.add_mesh("coupler", "coupler.stl", COLOR_COUPLER)

    sx, sy, sz = servo.shaft_offset

    # Horn holes in shaft-centered frame
    if servo.horn_mounting_points:
        for i, mp in enumerate(servo.horn_mounting_points):
            scene.annotate_sphere(
                f"horn_{i}",
                pos=(mp.pos[0] - sx, mp.pos[1] - sy, mp.pos[2] - sz),
                size=0.0015,
                color=COLOR_HORN_HOLE,
            )
    if servo.rear_horn_mounting_points:
        for i, mp in enumerate(servo.rear_horn_mounting_points):
            scene.annotate_sphere(
                f"rear_{i}",
                pos=(mp.pos[0] - sx, mp.pos[1] - sy, mp.pos[2] - sz),
                size=0.0015,
                color=COLOR_REAR_HOLE,
            )

    # Shaft origin indicator
    scene.annotate_cylinder(
        "shaft", pos=(0, 0, 0.004), radius=0.003, half_height=0.004, color=COLOR_SHAFT
    )

    return scene, temp_dir


def build_fastener_showcase_scene(component: Component) -> tuple[SceneBuilder, Path]:
    """Build scene showing a component with actual CAD screw solids at each mount point.

    Unlike the standard component scene (which uses blue cylinder primitives),
    this renders real screw geometry from screw_solid() — chamfered heads,
    proper diameters, oriented to each mount point's insertion axis.
    """
    from build123d import export_stl

    from botcad.emit.cad import make_component_solid, screw_solid

    temp_dir = Path(tempfile.mkdtemp(prefix="botcad_fastener_"))
    solid = make_component_solid(component)
    export_stl(solid, str(temp_dir / "component.stl"))

    scene = SceneBuilder(model_name="fastener_showcase", width=WIDTH, height=HEIGHT)
    scene.add_mesh("comp", "component.stl", _body_render_color(component))

    # Export and place a screw mesh at each mount point
    from botcad.colors import COLOR_METAL_BRASS

    screw_color = Color(*COLOR_METAL_BRASS.rgb, 1.0, "screw (brass)")
    exported_diameters: dict[float, str] = {}

    for i, mp in enumerate(component.mounting_points):
        if mp.diameter not in exported_diameters:
            stl_name = f"screw_{mp.diameter:.4f}.stl"
            export_stl(screw_solid(mp.diameter), str(temp_dir / stl_name))
            exported_diameters[mp.diameter] = stl_name

        scene.add_mesh(
            f"screw_{i}",
            exported_diameters[mp.diameter],
            screw_color,
            pos=mp.pos,
            quat=_axis_quat(mp.axis),
        )

    _add_axis_annotations(scene, component.dimensions)

    return scene, temp_dir


def build_coupler_assembly_scene(servo: ServoSpec) -> tuple[SceneBuilder, Path]:
    """Build scene showing cradle + servo + coupler together in servo frame."""
    from build123d import Location, export_stl

    from botcad.bracket import BracketSpec, coupler_solid, cradle_solid, servo_solid

    temp_dir = Path(tempfile.mkdtemp(prefix="botcad_coupler_asm_"))
    export_stl(cradle_solid(servo, BracketSpec()), str(temp_dir / "cradle.stl"))
    export_stl(servo_solid(servo), str(temp_dir / "servo.stl"))

    # Coupler in shaft-centered frame → move to servo frame
    sx, sy, sz = servo.shaft_offset
    coupler = coupler_solid(servo, BracketSpec()).moved(Location((sx, sy, sz)))
    export_stl(coupler, str(temp_dir / "coupler.stl"))

    scene = SceneBuilder(
        model_name="coupler_assembly_debug", width=WIDTH, height=HEIGHT
    )
    scene.add_mesh("cradle", "cradle.stl", COLOR_CRADLE)
    scene.add_mesh("servo", "servo.stl", COLOR_SERVO_BODY)
    scene.add_mesh("coupler", "coupler.stl", COLOR_COUPLER)
    _add_bracket_annotations(scene, servo)
    _add_horn_annotations(scene, servo)

    return scene, temp_dir


# ── Public render functions ──


def _dims_str(d: tuple) -> str:
    return f"{d[0] * 1000:.1f} x {d[1] * 1000:.1f} x {d[2] * 1000:.1f} mm"


def render_component_views(component: Component, name: str) -> Image.Image:
    """Render all views of a component and composite into a single image."""
    scene, temp_dir = build_component_scene(component)
    return _render_scene(scene, temp_dir, f"{name} — {_dims_str(component.dimensions)}")


def render_bracket_views(servo: ServoSpec, name: str) -> Image.Image:
    """Render bracket + servo assembly from all views."""
    scene, temp_dir = build_bracket_scene(servo)
    return _render_scene(scene, temp_dir, f"Pocket Bracket — {name}")


def render_cradle_views(servo: ServoSpec, name: str) -> Image.Image:
    """Render cradle + servo from all views."""
    scene, temp_dir = build_cradle_scene(servo)
    return _render_scene(scene, temp_dir, f"Cradle (static side) — {name}")


def render_coupler_views(servo: ServoSpec, name: str) -> Image.Image:
    """Render coupler alone from all views."""
    scene, temp_dir = build_coupler_scene(servo)
    return _render_scene(scene, temp_dir, f"Coupler (moving side) — {name}")


def render_coupler_assembly_views(servo: ServoSpec, name: str) -> Image.Image:
    """Render complete coupler assembly (cradle + servo + coupler)."""
    scene, temp_dir = build_coupler_assembly_scene(servo)
    return _render_scene(scene, temp_dir, f"Coupler Assembly — {name}")


def render_fastener_showcase(component: Component, name: str) -> Image.Image:
    """Render a component with actual CAD screw solids at each mount point.

    Shows real screw geometry (chamfered heads, proper diameters) oriented
    to each mounting point's insertion axis. Visual regression baseline for
    fastener rendering quality.
    """
    scene, temp_dir = build_fastener_showcase_scene(component)
    title = f"Fastener Showcase — {name} — {_dims_str(component.dimensions)}"
    return _render_scene(scene, temp_dir, title)


# ── Pipeline entry point ──


_COMPONENT_CATEGORY_REGISTRY: dict[str, str] | None = None


def _component_category(comp: Component) -> str:
    """Derive category from the component's module (battery, camera, servo, etc.).

    Builds a name→module registry on first call, avoiding repeated brute-force
    module scanning and factory instantiation.
    """
    global _COMPONENT_CATEGORY_REGISTRY
    if _COMPONENT_CATEGORY_REGISTRY is None:
        import importlib
        import pkgutil

        import botcad.components as pkg

        registry: dict[str, str] = {}
        for info in pkgutil.iter_modules(pkg.__path__):
            mod = importlib.import_module(f"botcad.components.{info.name}")
            for attr in dir(mod):
                obj = getattr(mod, attr)
                if callable(obj) and not isinstance(obj, type):
                    try:
                        instance = obj()
                        if isinstance(instance, Component):
                            registry[instance.name] = info.name
                    except TypeError:
                        pass
        _COMPONENT_CATEGORY_REGISTRY = registry

    return _COMPONENT_CATEGORY_REGISTRY.get(comp.name, "component")


def emit_component_renders(bot, output_dir: Path) -> None:
    """Render every unique component used in the bot.

    Collects components from body mounts and joint servos, deduplicates
    by name, and renders each one. Output goes to botcad/components/ —
    tear sheets live next to the component definitions, not the bot.
    """
    import dataclasses

    t0 = time.perf_counter()

    # Output next to the component source files
    components_dir = Path(__file__).resolve().parent.parent / "components"

    # Collect unique components by name
    seen: dict[str, Component] = {}

    for body in bot.all_bodies:
        for mount in body.mounts:
            comp = mount.component
            if comp.name not in seen:
                seen[comp.name] = comp

    # Collect servos separately (they get position + continuous variants)
    servos: dict[str, ServoSpec] = {}
    for joint in bot.all_joints:
        servo = joint.servo
        if servo.name not in servos:
            servos[servo.name] = servo

    if not seen and not servos:
        print("  component renders: no components found, skipping")
        return

    print("Component renders:")

    # Render non-servo components
    for name, comp in seen.items():
        if isinstance(comp, ServoSpec):
            continue  # servos handled below
        category = _component_category(comp)
        safe_name = name.lower().replace(" ", "_")
        filename = f"test_{category}_{safe_name}.png"

        img = render_component_views(comp, name)
        out_path = components_dir / filename
        save_png(img, out_path)
        print(f"  {out_path}")

    # Render servo variants (position + continuous)
    for name, servo in servos.items():
        safe_name = name.lower().replace(" ", "_")
        for mode in (False, True):
            mode_name = "continuous" if mode else "position"
            variant = dataclasses.replace(servo, continuous=mode)
            display_name = f"{name} ({mode_name})"
            filename = f"test_servo_{safe_name}_{mode_name}.png"

            img = render_component_views(variant, display_name)
            out_path = components_dir / filename
            save_png(img, out_path)
            print(f"  {out_path}")

    # Render bracket assemblies for each servo
    for name, servo in servos.items():
        safe_name = name.lower().replace(" ", "_")
        filename = f"test_bracket_{safe_name}.png"

        img = render_bracket_views(servo, name)
        out_path = components_dir / filename
        save_png(img, out_path)
        print(f"  {out_path}")

    print(f"  component renders done ({time.perf_counter() - t0:.1f}s)")


# ── Standalone ──


if __name__ == "__main__":
    from botcad.components import (
        OV5647,
        STS3215,
        LiPo2S,
        PiCamera2,
        PololuWheel90mm,
        RaspberryPiZero2W,
    )

    out_dir = Path(__file__).resolve().parent.parent / "components"

    # (category, display_name, component)
    components: list[tuple[str, str, Component]] = [
        ("servo", "STS3215 (position)", STS3215(continuous=False)),
        ("servo", "STS3215 (continuous)", STS3215(continuous=True)),
        ("wheel", "Pololu 90x10mm Wheel", PololuWheel90mm()),
        ("camera", "OV5647", OV5647()),
        ("camera", "PiCamera2", PiCamera2()),
        ("battery", "LiPo2S-1000", LiPo2S(1000)),
        ("compute", "RaspberryPiZero2W", RaspberryPiZero2W()),
    ]

    for category, name, comp in components:
        img = render_component_views(comp, name)
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        out_path = out_dir / f"test_{category}_{safe_name}.png"
        save_png(img, out_path)
        print(f"Saved: {out_path}")

    # Bracket tear sheets
    servo = STS3215()
    for render_fn, label in [
        (render_bracket_views, "bracket"),
        (render_cradle_views, "cradle"),
        (render_coupler_views, "coupler"),
        (render_coupler_assembly_views, "coupler_assembly"),
    ]:
        img = render_fn(servo, "STS3215")
        out_path = out_dir / f"test_{label}_sts3215.png"
        save_png(img, out_path)
        print(f"Saved: {out_path}")
