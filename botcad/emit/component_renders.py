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

from botcad.component import Component, ComponentKind, ServoSpec, get_component_meta
from botcad.emit.composite import grid, save_png
from botcad.emit.render3d import (
    COLOR_BRACKET,
    COLOR_COUPLER,
    COLOR_CRADLE,
    COLOR_HORN,
    COLOR_HORN_HOLE,
    COLOR_INSERTION_CHANNEL,
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
    c = component.appearance.color
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
    if component.kind == ComponentKind.SERVO:
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


# ── Shared helpers for scene builders ──


def _add_screws_to_scene(
    scene: SceneBuilder,
    component: Component,
    temp_dir: Path,
    exported_screws: dict[tuple[str, str], str],
) -> None:
    """Export and place screw meshes at each mount point."""
    from build123d import export_stl

    from botcad.colors import COLOR_METAL_BRASS
    from botcad.emit.cad import screw_solid
    from botcad.fasteners import fastener_key

    screw_color = Color(*COLOR_METAL_BRASS.rgb, 1.0, "screw (brass)")

    for i, mp in enumerate(component.mounting_points):
        key = fastener_key(mp)
        if key not in exported_screws:
            stl_name = f"screw_{key[0]}_{key[1] or 'shc'}.stl"
            export_stl(screw_solid(key[0], key[1]), str(temp_dir / stl_name))
            exported_screws[key] = stl_name

        scene.add_mesh(
            f"screw_{i}",
            exported_screws[key],
            screw_color,
            pos=mp.pos,
            quat=_axis_quat(mp.axis),
        )


def _add_connectors_to_scene(
    scene: SceneBuilder,
    component: Component,
    temp_dir: Path,
    exported_connectors: dict[str, str],
    pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    cable_len: float = 0.015,
) -> None:
    """Export and place connector meshes + cable stubs at each wire port."""
    from build123d import export_stl

    from botcad.connectors import connector_solid, connector_spec

    connector_color = Color(*COLOR_WIRE_PORT.rgb, 1.0, "connector (orange)")
    cable_color = Color(0.15, 0.15, 0.15, 1.0, "cable (black)")

    for i, wp in enumerate(component.wire_ports):
        if not wp.connector_type:
            continue
        try:
            cspec = connector_spec(wp.connector_type)
        except KeyError:
            continue

        # Export connector housing STL (once per type)
        if wp.connector_type not in exported_connectors:
            c_solid = connector_solid(cspec)
            c_stl = f"connector_{wp.connector_type}.stl"
            export_stl(c_solid, str(temp_dir / c_stl))
            exported_connectors[wp.connector_type] = c_stl

        conn_pos = (
            wp.pos[0] + pos_offset[0],
            wp.pos[1] + pos_offset[1],
            wp.pos[2] + pos_offset[2],
        )
        scene.add_mesh(
            f"connector_{i}",
            exported_connectors[wp.connector_type],
            connector_color,
            pos=conn_pos,
        )

        # Cable stub — cylinder along wire_exit_direction
        cable_r = max(cspec.body_dimensions) * 0.2
        ex, ey, ez = cspec.wire_exit_direction
        ox, oy, oz = cspec.wire_exit_offset
        cable_mid = (
            conn_pos[0] + ox + ex * cable_len / 2,
            conn_pos[1] + oy + ey * cable_len / 2,
            conn_pos[2] + oz + ez * cable_len / 2,
        )
        scene.annotate_cylinder(
            f"cable_{i}",
            pos=cable_mid,
            radius=cable_r,
            half_height=cable_len / 2,
            color=cable_color,
            quat=_axis_quat((ex, ey, ez)),
        )


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
    if component.kind == ComponentKind.SERVO:
        _add_servo_annotations(scene, component)

    return scene, temp_dir


def build_bracket_scene(servo: ServoSpec) -> tuple[SceneBuilder, Path]:
    """Build scene showing bracket + servo + insertion channel + annotations."""
    from build123d import export_stl

    from botcad.bracket import (
        BracketSpec,
        bracket_insertion_channel_solid,
        bracket_solid_solid,
        servo_solid,
    )

    temp_dir = Path(tempfile.mkdtemp(prefix="botcad_bracket_"))
    export_stl(bracket_solid_solid(servo, BracketSpec()), str(temp_dir / "bracket.stl"))
    export_stl(servo_solid(servo), str(temp_dir / "servo.stl"))
    export_stl(
        bracket_insertion_channel_solid(servo, BracketSpec()),
        str(temp_dir / "insertion_channel.stl"),
    )

    scene = SceneBuilder(model_name="bracket_debug", width=WIDTH, height=HEIGHT)
    scene.add_mesh("bracket", "bracket.stl", COLOR_BRACKET)
    scene.add_mesh("servo", "servo.stl", COLOR_SERVO_BODY)
    scene.add_mesh(
        "insertion_channel", "insertion_channel.stl", COLOR_INSERTION_CHANNEL
    )
    _add_bracket_annotations(scene, servo)
    _add_horn_annotations(scene, servo)

    return scene, temp_dir


def build_cradle_scene(servo: ServoSpec) -> tuple[SceneBuilder, Path]:
    """Build scene showing cradle + servo + insertion channel + annotations."""
    from build123d import export_stl

    from botcad.bracket import (
        BracketSpec,
        cradle_insertion_channel_solid,
        cradle_solid_solid,
        servo_solid,
    )

    temp_dir = Path(tempfile.mkdtemp(prefix="botcad_cradle_"))
    export_stl(cradle_solid_solid(servo, BracketSpec()), str(temp_dir / "cradle.stl"))
    export_stl(servo_solid(servo), str(temp_dir / "servo.stl"))
    export_stl(
        cradle_insertion_channel_solid(servo, BracketSpec()),
        str(temp_dir / "insertion_channel.stl"),
    )

    scene = SceneBuilder(model_name="cradle_debug", width=WIDTH, height=HEIGHT)
    scene.add_mesh("cradle", "cradle.stl", COLOR_CRADLE)
    scene.add_mesh("servo", "servo.stl", COLOR_SERVO_BODY)
    scene.add_mesh(
        "insertion_channel", "insertion_channel.stl", COLOR_INSERTION_CHANNEL
    )
    _add_bracket_annotations(scene, servo)

    return scene, temp_dir


def build_coupler_scene(servo: ServoSpec) -> tuple[SceneBuilder, Path]:
    """Build scene showing coupler in shaft-centered frame."""
    from build123d import export_stl

    from botcad.bracket import BracketSpec, coupler_solid_solid

    temp_dir = Path(tempfile.mkdtemp(prefix="botcad_coupler_"))
    export_stl(coupler_solid_solid(servo, BracketSpec()), str(temp_dir / "coupler.stl"))

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

    from botcad.emit.cad import make_component_solid

    temp_dir = Path(tempfile.mkdtemp(prefix="botcad_fastener_"))
    solid = make_component_solid(component)
    export_stl(solid, str(temp_dir / "component.stl"))

    scene = SceneBuilder(model_name="fastener_showcase", width=WIDTH, height=HEIGHT)
    scene.add_mesh("comp", "component.stl", _body_render_color(component))

    exported_screws: dict[tuple[str, str], str] = {}
    _add_screws_to_scene(scene, component, temp_dir, exported_screws)

    exported_connectors: dict[str, str] = {}
    _add_connectors_to_scene(scene, component, temp_dir, exported_connectors)

    _add_axis_annotations(scene, component.dimensions)

    return scene, temp_dir


def build_coupler_assembly_scene(servo: ServoSpec) -> tuple[SceneBuilder, Path]:
    """Build scene showing cradle + servo + coupler together in servo frame."""
    from build123d import Location, export_stl

    from botcad.bracket import (
        BracketSpec,
        coupler_solid_solid,
        cradle_solid_solid,
        servo_solid,
    )

    temp_dir = Path(tempfile.mkdtemp(prefix="botcad_coupler_asm_"))
    export_stl(cradle_solid_solid(servo, BracketSpec()), str(temp_dir / "cradle.stl"))
    export_stl(servo_solid(servo), str(temp_dir / "servo.stl"))

    # Coupler in shaft-centered frame → move to servo frame
    sx, sy, sz = servo.shaft_offset
    coupler = coupler_solid_solid(servo, BracketSpec()).moved(Location((sx, sy, sz)))
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


def _render_scene_4view(scene: SceneBuilder, temp_dir: Path, title: str) -> Image.Image:
    """Render 4 views (front, right, top, three-quarter), composite into a 2x2 grid."""
    from botcad.emit.render3d import VIEWS_4

    try:
        scene.set_mesh_dir(temp_dir)
        xml = scene.to_xml()
        with Renderer3D(xml, WIDTH, HEIGHT) as r:
            views = r.render_views(VIEWS_4)
        return grid(title, scene.legends, views, cols=2)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def build_fastener_detail_scene(component: Component) -> tuple[SceneBuilder, Path]:
    """Build scene showing only the fasteners against a ghost body.

    The component body is rendered translucent so fastener head detail
    (hex recesses, Phillips crosses) is clearly visible.
    """
    from build123d import export_stl

    from botcad.emit.cad import make_component_solid

    temp_dir = Path(tempfile.mkdtemp(prefix="botcad_fastener_detail_"))
    solid = make_component_solid(component)
    export_stl(solid, str(temp_dir / "component.stl"))

    scene = SceneBuilder(model_name="fastener_detail", width=WIDTH, height=HEIGHT)

    # Ghost body — translucent so screws are visible
    ghost_color = Color(0.6, 0.65, 0.7, 0.15, "body (ghost)")
    scene.add_mesh("comp", "component.stl", ghost_color)

    exported_screws: dict[tuple[str, str], str] = {}
    _add_screws_to_scene(scene, component, temp_dir, exported_screws)

    return scene, temp_dir


def render_fastener_detail(component: Component, name: str) -> Image.Image:
    """Render fastener detail: ghost body + brass screws from 4 views."""
    scene, temp_dir = build_fastener_detail_scene(component)
    n = len(component.mounting_points)
    title = f"Fastener Detail — {name} — {n} fastener{'s' if n != 1 else ''}"
    return _render_scene_4view(scene, temp_dir, title)


def build_connector_detail_scene(component: Component) -> tuple[SceneBuilder, Path]:
    """Build scene showing only connectors + cable stubs against a ghost body."""
    from build123d import export_stl

    from botcad.emit.cad import make_component_solid

    temp_dir = Path(tempfile.mkdtemp(prefix="botcad_connector_detail_"))
    solid = make_component_solid(component)
    export_stl(solid, str(temp_dir / "component.stl"))

    scene = SceneBuilder(model_name="connector_detail", width=WIDTH, height=HEIGHT)
    ghost_color = Color(0.6, 0.65, 0.7, 0.15, "body (ghost)")
    scene.add_mesh("comp", "component.stl", ghost_color)

    exported_connectors: dict[str, str] = {}
    _add_connectors_to_scene(scene, component, temp_dir, exported_connectors)

    return scene, temp_dir


def render_connector_detail(component: Component, name: str) -> Image.Image:
    """Render connector detail: ghost body + orange connectors + black cables from 4 views."""
    scene, temp_dir = build_connector_detail_scene(component)
    n = sum(1 for wp in component.wire_ports if wp.connector_type)
    title = f"Connector Detail — {name} — {n} connector{'s' if n != 1 else ''}"
    return _render_scene_4view(scene, temp_dir, title)


def render_fastener_catalog() -> Image.Image:
    """Render every fastener in the catalog as individual close-ups in a grid.

    Each cell shows one fastener (designation × head type) from a
    three-quarter view. No component body — just the screw itself
    so head geometry detail is clearly visible.
    """
    from build123d import export_stl

    from botcad.colors import COLOR_METAL_BRASS
    from botcad.fasteners import HeadType, fastener_solid, fastener_spec

    temp_dir = Path(tempfile.mkdtemp(prefix="botcad_fastener_catalog_"))
    screw_color = Color(*COLOR_METAL_BRASS.rgb, 1.0, "screw (brass)")

    designations = ["M2", "M2.5", "M3"]
    head_types = [
        (HeadType.SOCKET_HEAD_CAP, "Socket Head Cap"),
        (HeadType.PAN_HEAD_PHILLIPS, "Pan Head Phillips"),
    ]

    cells: list[tuple[Image.Image, str]] = []

    for ht_enum, ht_label in head_types:
        for des in designations:
            spec = fastener_spec(des, ht_enum)
            solid = fastener_solid(spec, 0.006)  # 6mm shank for visibility
            stl_name = f"fastener_{des}_{ht_enum.value}.stl"
            export_stl(solid, str(temp_dir / stl_name))

            scene = SceneBuilder(
                model_name=f"fastener_{des}_{ht_enum.value}",
                width=WIDTH,
                height=HEIGHT,
            )
            scene.add_mesh("screw", stl_name, screw_color)
            scene.set_mesh_dir(temp_dir)
            xml = scene.to_xml()

            # Render three views: three-quarter, top-down, front
            views_single = {
                "three-quarter": {"azimuth": 135, "elevation": -30},
                "top": {"azimuth": 90, "elevation": -90},
                "front": {"azimuth": 90, "elevation": 0},
            }
            with Renderer3D(xml, WIDTH, HEIGHT) as r:
                rendered = r.render_views(views_single)

            # Take the three-quarter view as the main cell
            img_3q = rendered[0][0]
            cells.append((img_3q, f"{des} {ht_label}"))

    try:
        result = grid(
            "Fastener Catalog — ISO metric screws",
            [(screw_color.label, screw_color.rgb_int)],
            cells,
            cols=3,
        )
        return result
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def render_connector_catalog() -> Image.Image:
    """Render every connector as a mating pair — plug approaching receptacle.

    Each cell shows one connector type with plug and receptacle in a single
    scene, separated along the mating axis so the insertion mechanism is
    visible. Both parts at the same scale.
    """
    from build123d import Location, export_stl

    from botcad.connectors import (
        ConnectorType,
        connector_solid,
        connector_spec,
        receptacle_solid,
    )

    temp_dir = Path(tempfile.mkdtemp(prefix="botcad_connector_catalog_"))

    plug_color = Color(*COLOR_WIRE_PORT.rgb, 1.0, "plug (orange)")
    receptacle_color = Color(0.75, 0.75, 0.78, 1.0, "receptacle (silver)")
    cable_color = Color(0.15, 0.15, 0.15, 1.0, "cable (black)")

    cells: list[tuple[Image.Image, str]] = []

    for ct in ConnectorType:
        cspec = connector_spec(ct.value)
        mx, my, mz = cspec.mating_direction
        bx, by, bz = cspec.body_dimensions

        # Separation gap along mating axis — enough to see both parts
        gap = max(bx, by, bz) * 1.5

        # Plug offset: pulled back along mating direction
        plug_offset = (-mx * gap, -my * gap, -mz * gap)

        # Export plug (moved away from receptacle along mating axis)
        plug_s = connector_solid(cspec).moved(Location(plug_offset))
        plug_stl = f"pair_plug_{ct.value}.stl"
        export_stl(plug_s, str(temp_dir / plug_stl))

        # Export receptacle at origin
        rcpt_s = receptacle_solid(cspec)
        rcpt_stl = f"pair_rcpt_{ct.value}.stl"
        export_stl(rcpt_s, str(temp_dir / rcpt_stl))

        scene = SceneBuilder(model_name=f"pair_{ct.value}", width=WIDTH, height=HEIGHT)
        scene.add_mesh("plug", plug_stl, plug_color)
        scene.add_mesh("receptacle", rcpt_stl, receptacle_color)

        # Cable stub on the plug
        cable_r = max(bx, by, bz) * 0.2
        cable_len = 0.012
        ex, ey, ez = cspec.wire_exit_direction
        ox, oy, oz = cspec.wire_exit_offset
        cable_mid = (
            plug_offset[0] + ox + ex * cable_len / 2,
            plug_offset[1] + oy + ey * cable_len / 2,
            plug_offset[2] + oz + ez * cable_len / 2,
        )
        scene.annotate_cylinder(
            "cable",
            pos=cable_mid,
            radius=cable_r,
            half_height=cable_len / 2,
            color=cable_color,
            quat=_axis_quat((ex, ey, ez)),
        )

        scene.set_mesh_dir(temp_dir)
        xml = scene.to_xml()
        with Renderer3D(xml, WIDTH, HEIGHT) as r:
            rendered = r.render_views(
                {"three-quarter": {"azimuth": 135, "elevation": -30}}
            )
        cells.append((rendered[0][0], cspec.label))

    try:
        # Single column so each pair gets a wide cell
        result = grid(
            "Connector Catalog — plug + receptacle mating pairs",
            [
                (plug_color.label, plug_color.rgb_int),
                (receptacle_color.label, receptacle_color.rgb_int),
                (cable_color.label, cable_color.rgb_int),
            ],
            cells,
            cols=1,
        )
        return result
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ── Pipeline entry point ──


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
        if comp.kind == ComponentKind.SERVO:
            continue  # servos handled below
        category = get_component_meta(comp.kind).category
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
