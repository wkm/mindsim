"""FastAPI server for the MindSim web viewer.

Replaces the stdlib ThreadingHTTPServer/ViewerHTTPHandler with typed routes,
automatic validation, and uvicorn's --reload for development.

Usage (dev, with hot-reload):
    uvicorn mindsim.server:app --host 0.0.0.0 --port 8081 --reload --reload-dir botcad --reload-dir mindsim

Usage (programmatic, from main.py):
    from mindsim.server import app, init_registry
    init_registry()
    uvicorn.run(app, host="0.0.0.0", port=port)
"""

from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import structlog
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.staticfiles import StaticFiles

log = structlog.get_logger("mindsim.server")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Ensure the component registry is built when uvicorn starts."""
    init_registry()
    yield


app = FastAPI(title="MindSim Viewer API", lifespan=_lifespan)

# ---------------------------------------------------------------------------
# Request timing middleware — logs duration for /api/ requests
# ---------------------------------------------------------------------------

_request_log = structlog.get_logger("mindsim.server.requests")


@app.middleware("http")
async def _log_api_requests(request: Request, call_next):
    if not request.url.path.startswith("/api/"):
        return await call_next(request)

    t0 = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - t0) * 1000, 1)
    _request_log.info(
        "request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_ms=duration_ms,
    )
    return response


# ---------------------------------------------------------------------------
# Project root — needed for static file serving and relative path resolution
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Component registry — populated at startup
# ---------------------------------------------------------------------------

_component_registry: dict = {}

# ---------------------------------------------------------------------------
# Caches (same semantics as the old stdlib server)
# ---------------------------------------------------------------------------

# Solid object cache: (component_name, part_name) -> Solid
_solid_cache: dict[tuple[str, str], object] = {}
_solid_cache_lock = threading.Lock()

# STL generation cache: (component_name, part_name) -> bytes
_stl_cache: dict[tuple[str, str], bytes] = {}
_stl_cache_lock = threading.Lock()

# Cached JSON responses (built once, served on every request)
_catalog_json: bytes | None = None
_bots_json: bytes | None = None

# CAD steps cache: (bot_name, body_name) -> list[CadStep]
_cad_steps_cache: dict[tuple[str, str], list] = {}
_cad_steps_lock = threading.Lock()

# FEA cache: (bot_name, body_name, file_name) -> bytes
_fea_cache: dict[tuple[str, str, str], bytes] = {}
_fea_cache_lock = threading.Lock()

# Bot object cache: bot_name -> (bot, cad_model)
_bot_cache: dict[str, tuple] = {}
_bot_cache_lock = threading.Lock()

# Layer colors — mirrors client-side LAYER_META.colorHex
_LAYER_COLORS: dict[str, tuple[int, int, int]] = {
    "servo": (24, 32, 38),
    "horn": (232, 232, 232),
    "bracket": (206, 217, 224),
    "cradle": (206, 217, 224),
    "coupler": (245, 86, 86),
    "bracket_insertion_channel": (245, 86, 86),
    "cradle_insertion_channel": (245, 86, 86),
    "fasteners": (212, 168, 67),
}

# Design layer metadata for component manifest — color as linear [r,g,b], opacity
_DESIGN_LAYER_META: dict[str, dict] = {
    "servo": {"label": "Servo", "color": [0.094, 0.125, 0.149], "opacity": 1.0},
    "horn": {"label": "Horn", "color": [0.910, 0.910, 0.910], "opacity": 1.0},
    "bracket": {"label": "Bracket", "color": [0.808, 0.851, 0.878], "opacity": 1.0},
    "cradle": {"label": "Cradle", "color": [0.808, 0.851, 0.878], "opacity": 1.0},
    "coupler": {"label": "Coupler", "color": [0.961, 0.337, 0.337], "opacity": 1.0},
    "bracket_insertion_channel": {
        "label": "Bracket Insertion Channel",
        "color": [0.961, 0.337, 0.337],
        "opacity": 0.25,
    },
    "cradle_insertion_channel": {
        "label": "Cradle Insertion Channel",
        "color": [0.961, 0.337, 0.337],
        "opacity": 0.25,
    },
    # bracket_envelope / cradle_envelope omitted — no STL generation exists
    # for these in _generate_solid(). They're listed in ComponentMeta.layers
    # but were never implemented.
}

# Bus-type → RGBA color for wire stubs / connectors
_BUS_COLORS: dict[str, list[float]] = {
    "uart_half_duplex": [0.20, 0.60, 0.86, 1.0],
    "csi": [0.40, 0.73, 0.42, 1.0],
    "power": [0.90, 0.30, 0.25, 1.0],
    "usb": [0.55, 0.35, 0.75, 1.0],
}


# ---------------------------------------------------------------------------
# Registry initialisation
# ---------------------------------------------------------------------------


def init_registry() -> None:
    """Build the component registry. Call once before serving requests."""
    global _component_registry
    if _component_registry:
        return  # already initialised
    log.info("building_component_registry")
    _component_registry = _build_component_registry()
    log.info("component_registry_ready", count=len(_component_registry))


# ---------------------------------------------------------------------------
# Helper functions (moved from main.py)
# ---------------------------------------------------------------------------


def _discover_bots() -> list[dict]:
    """Scan bots/*/scene.xml and return info about each bot."""
    bots_dir = PROJECT_ROOT / "bots"
    results = []
    if bots_dir.is_dir():
        for scene in sorted(bots_dir.glob("*/scene.xml")):
            name = scene.parent.name
            results.append({"name": name, "scene_path": str(scene)})
    return results


def _build_component_registry() -> dict:
    """Build a registry of all components from botcad.components factories.

    Returns dict mapping component name -> (factory_func, Component instance, category).
    """
    import importlib
    import pkgutil

    import botcad.components as pkg
    from botcad.component import Component

    registry: dict[str, tuple] = {}
    for info in pkgutil.iter_modules(pkg.__path__):
        mod = importlib.import_module(f"botcad.components.{info.name}")
        for attr in dir(mod):
            obj = getattr(mod, attr)
            if callable(obj) and not isinstance(obj, type):
                try:
                    instance = obj()
                    if isinstance(instance, Component):
                        registry[instance.name] = (obj, instance, info.name)
                except TypeError:
                    pass
    return registry


def _component_layers(comp) -> list[str]:
    """Return the list of available STL layer IDs for a component."""
    from botcad.component import get_component_meta

    meta = get_component_meta(comp.kind)
    layers = list(meta.layers)
    # Add wires layer if component has wire ports with connectors
    if any(wp.connector_type for wp in comp.wire_ports):
        layers.append("wires")
    return layers


def _component_to_json(comp, category: str) -> dict:
    """Serialize a Component instance to JSON-safe dict."""
    from botcad.component import ComponentKind

    info = {
        "name": comp.name,
        "category": category,
        "dimensions_mm": [round(d * 1000, 1) for d in comp.dimensions],
        "mass_g": round(comp.mass * 1000, 1),
        "is_servo": comp.kind == ComponentKind.SERVO,
        "color": list(comp.default_material.color)
        if comp.default_material
        else [0.541, 0.608, 0.659, 1.0],
        "layers": _component_layers(comp),
    }
    if comp.kind == ComponentKind.SERVO:
        import math

        info["servo"] = {
            "stall_torque_nm": round(comp.stall_torque, 3),
            "no_load_speed_rpm": round(comp.no_load_speed * 60 / (2 * math.pi), 1),
            "voltage": comp.voltage,
            "range_deg": [round(math.degrees(r), 1) for r in comp.range_rad],
            "gear_ratio": comp.gear_ratio,
            "continuous": comp.continuous,
        }
    # Discover available 2D technical drawings
    if comp.kind == ComponentKind.SERVO:
        safe = comp.name.lower()
        drawing_types = ["pocket", "coupler", "cradle", "coupler_assembly"]
        drawings = []
        for dt in drawing_types:
            svg_path = (
                PROJECT_ROOT / "botcad" / "components" / f"drawing_{dt}_{safe}.svg"
            )
            if svg_path.exists():
                drawings.append(
                    {
                        "type": dt,
                        "label": dt.replace("_", " ").title(),
                        "url": f"/botcad/components/drawing_{dt}_{safe}.svg",
                    }
                )
        info["drawings"] = drawings
    else:
        info["drawings"] = []

    info["mounting_points"] = [
        {
            "label": mp.label,
            "diameter_mm": round(mp.diameter * 1000, 2),
            "pos": list(mp.pos),
            "axis": list(mp.axis),
        }
        for mp in comp.mounting_points
    ]
    info["wire_ports"] = [
        {
            "label": wp.label,
            "bus_type": str(wp.bus_type),
            "pos": list(wp.pos),
        }
        for wp in comp.wire_ports
    ]
    return info


def _layer_color(comp, layer_id: str) -> tuple[int, int, int]:
    """Return RGB color for a component layer."""
    if layer_id in _LAYER_COLORS:
        return _LAYER_COLORS[layer_id]
    color = (
        comp.default_material.color
        if comp.default_material
        else (0.541, 0.608, 0.659, 1.0)
    )
    r, g, b = color[:3]
    return (int(r * 255), int(g * 255), int(b * 255))


def _solid_to_stl_bytes(solid) -> bytes:
    """Export a build123d Solid to STL bytes via a temp file."""
    import tempfile

    from build123d import export_stl

    with tempfile.NamedTemporaryFile(suffix=".stl", delete=True) as tmp:
        export_stl(solid, tmp.name)
        return Path(tmp.name).read_bytes()


def _generate_solid(comp, part: str):
    """Generate a build123d Solid for a component part. Returns None if N/A.

    Thread-safe with caching so solids are built once and reused for both
    STL export and SVG rendering.
    """
    from botcad.component import ComponentKind
    from botcad.emit.cad import make_component_solid

    cache_key = (comp.name, part)
    with _solid_cache_lock:
        if cache_key in _solid_cache:
            return _solid_cache[cache_key]

    solid = None
    if part == "body":
        solid = make_component_solid(comp)
    elif comp.kind == ComponentKind.SERVO:
        from botcad.bracket import (
            BracketSpec,
            bracket_solid_solid,
            servo_solid,
        )

        spec = BracketSpec()
        if part == "bracket":
            solid = bracket_solid_solid(comp, spec)
        elif part == "servo":
            solid = servo_solid(comp)
        elif part == "horn":
            from botcad.bracket import horn_disc_params
            from botcad.emit.cad import _horn_solid

            params = horn_disc_params(comp)
            if params is not None:
                from build123d import Location

                horn = _horn_solid(comp)
                if horn is not None:
                    # .moved() not .locate() — _horn_solid is @lru_cache'd
                    solid = horn.moved(
                        Location(
                            (params.center_xy[0], params.center_xy[1], params.center_z)
                        )
                    )
        elif part == "cradle":
            from botcad.bracket import cradle_solid_solid

            solid = cradle_solid_solid(comp, spec)
        elif part == "coupler":
            from build123d import Location

            from botcad.bracket import coupler_solid_solid

            raw = coupler_solid_solid(comp, spec)
            if raw is not None and comp.rear_horn_mounting_points:
                # Coupler is built in shaft-centered frame; shift to servo local
                # frame.  Include the Z component — the shaft sits at the body
                # top face, not at the body center.
                sx, sy, sz = comp.shaft_offset
                solid = raw.moved(Location((sx, sy, sz)))
        elif part == "bracket_insertion_channel":
            from botcad.bracket import bracket_insertion_channel_solid

            solid = bracket_insertion_channel_solid(comp, spec)
        elif part == "cradle_insertion_channel":
            from botcad.bracket import cradle_insertion_channel_solid

            solid = cradle_insertion_channel_solid(comp, spec)

    if solid is not None:
        with _solid_cache_lock:
            _solid_cache[cache_key] = solid

    return solid


def _generate_stl_bytes(comp, part: str) -> bytes | None:
    """Generate STL bytes for a component part. Returns None if not applicable."""
    cache_key = (comp.name, part)
    with _stl_cache_lock:
        if cache_key in _stl_cache:
            return _stl_cache[cache_key]

    solid = _generate_solid(comp, part)
    if solid is None:
        return None

    stl_bytes = _solid_to_stl_bytes(solid)
    with _stl_cache_lock:
        _stl_cache[cache_key] = stl_bytes
    return stl_bytes


def _generate_design_layer_mesh(joint, layer_kind: str, bot=None) -> bytes | None:
    """Generate STL bytes for a design layer solid in body-local frame.

    Builds the bracket/coupler/clearance/insertion solid in servo-local frame,
    then applies the servo placement transform to get body-local coordinates.
    """
    from build123d import Location

    from botcad.bracket import BracketSpec
    from botcad.geometry import quat_to_euler
    from botcad.skeleton import BracketStyle

    servo = joint.servo
    spec = BracketSpec()

    # Build solid in servo-local frame
    solid = None
    if layer_kind == "bracket":
        if joint.bracket_style is BracketStyle.COUPLER:
            from botcad.bracket import cradle_solid_solid

            solid = cradle_solid_solid(servo, spec)
        else:
            from botcad.bracket import bracket_solid_solid

            solid = bracket_solid_solid(servo, spec)
    elif layer_kind == "coupler":
        from botcad.bracket import coupler_solid_solid

        raw = coupler_solid_solid(servo, spec)
        if raw is not None:
            # Coupler is built in shaft-centered frame; shift to servo-local
            # frame by applying shaft_offset.
            sx, sy, sz = servo.shaft_offset
            solid = raw.moved(Location((sx, sy, sz)))
    elif layer_kind == "clearance" or layer_kind == "insertion":
        if joint.bracket_style is BracketStyle.COUPLER:
            from botcad.bracket import cradle_insertion_channel_solid

            solid = cradle_insertion_channel_solid(servo, spec)
        else:
            from botcad.bracket import bracket_insertion_channel_solid

            solid = bracket_insertion_channel_solid(servo, spec)

    if solid is None:
        return None

    # Transform from servo-local frame to body-local frame using .moved()
    if bot and bot.packing_result:
        placement = bot.packing_result.placements.get(joint)
        if placement:
            center = placement.pose.pos
            euler = quat_to_euler(placement.pose.quat)
        else:
            center = joint.solved_servo_center
            euler = quat_to_euler(joint.solved_servo_quat)
    else:
        center = joint.solved_servo_center
        euler = quat_to_euler(joint.solved_servo_quat)
    solid = solid.moved(Location(center, euler))

    return _solid_to_stl_bytes(solid)


def _generate_bot_mesh(bot, cad, stem: str) -> bytes | None:
    """Generate STL bytes for a named mesh in a bot's mesh directory.

    Dispatches based on the mesh stem prefix to the appropriate solid factory.
    """

    body_solids = cad.body_solids

    # Body mesh (structural or component): look up pre-built solid from build_cad().
    # Component solids are in component-local frame; placement rotation is
    # applied at render time by the viewer/MuJoCo via PackingResult poses.
    if stem in body_solids:
        return _solid_to_stl_bytes(body_solids[stem])

    # Multi-material component mesh: e.g. comp_base_pi__fr4_green
    # Look up in cad.multi_material_solids by body_name + material_name.
    if "__" in stem:
        body_name, mat_name = stem.rsplit("__", 1)
        mat_solids = cad.multi_material_solids.get(body_name, [])
        for ms in mat_solids:
            if ms.material_name == mat_name:
                return _solid_to_stl_bytes(ms.solid)
        return None

    # Servo mesh: servo_{servo_name}
    if stem.startswith("servo_"):
        from botcad.emit.cad import servo_solid

        servo_name = stem.removeprefix("servo_")
        for joint in bot.all_joints:
            if joint.servo.name == servo_name:
                return _solid_to_stl_bytes(servo_solid(joint.servo))
        return None

    # Hardware/fastener mesh: hardware_{designation}_{head_type}
    if stem.startswith("hardware_"):
        from botcad.emit.cad import screw_solid
        from botcad.fasteners import fastener_key, fastener_stl_stem

        for body in bot.all_bodies:
            for joint in body.joints:
                for mp in (
                    *joint.servo.mounting_ears,
                    *joint.servo.horn_mounting_points,
                    *joint.servo.rear_horn_mounting_points,
                ):
                    if fastener_stl_stem(mp) == stem:
                        k = fastener_key(mp)
                        return _solid_to_stl_bytes(screw_solid(k[0], k[1]))
            for mount in body.mounts:
                for mp in mount.component.mounting_points:
                    if fastener_stl_stem(mp) == stem:
                        k = fastener_key(mp)
                        return _solid_to_stl_bytes(screw_solid(k[0], k[1]))
        return None

    # Horn mesh: horn_{joint_name}
    if stem.startswith("horn_"):
        from botcad.emit.cad import _horn_solid, _orient_z_to_axis

        joint_name = stem.removeprefix("horn_")
        for joint in bot.all_joints:
            if joint.name == joint_name:
                horn = _horn_solid(joint.servo)
                if horn is None:
                    return None
                horn = _orient_z_to_axis(horn, joint.axis)
                return _solid_to_stl_bytes(horn)
        return None

    # Design layer meshes: bracket, coupler, clearance, insertion
    # These are built in servo-local frame and transformed to body-local frame
    # using the servo placement (solved_servo_center + solved_servo_quat).
    for prefix, layer_kind in (
        ("bracket_", "bracket"),
        ("coupler_", "coupler"),
        ("clearance_", "clearance"),
        ("insertion_", "insertion"),
    ):
        if stem.startswith(prefix):
            joint_name = stem.removeprefix(prefix)
            for body in bot.all_bodies:
                for joint in body.joints:
                    if joint.name != joint_name:
                        continue
                    return _generate_design_layer_mesh(joint, layer_kind, bot=bot)
            return None

    # Connector housing mesh: connector_{type}.stl
    if stem.startswith("connector_"):
        connector_type = stem[len("connector_") :]
        from botcad.connectors import connector_solid, connector_spec

        cspec = connector_spec(connector_type)
        return _solid_to_stl_bytes(connector_solid(cspec))

    # Wire stub mesh: shared cylinder for all wire stubs
    if stem == "wire_stub":
        from botcad.shapescript.backend_occt import OcctBackend
        from botcad.shapescript.program import ShapeScriptBuilder

        prog = ShapeScriptBuilder()
        stub = prog.cylinder(0.0015, 0.025, tag="wire_stub")
        prog.output_ref = stub
        result = OcctBackend().execute(prog)
        return _solid_to_stl_bytes(result.shapes[stub.id])

    # Wire mesh: wire_{label}_{body}_{idx}
    if stem.startswith("wire_"):
        from botcad.component import BusType
        from botcad.emit.cad import _wire_segment_solid

        bus_radii = {
            BusType.UART_HALF_DUPLEX: 0.0009,
            BusType.CSI: 0.0018,
            BusType.POWER: 0.0012,
        }
        for route in bot.wire_routes:
            radius = bus_radii.get(route.bus_type, 0.0015)
            for i, seg in enumerate(route.segments):
                if seg.straight_length < 0.001:
                    continue
                expected = f"wire_{route.label}_{seg.body_name}_{i}"
                if expected == stem:
                    wire_solid = _wire_segment_solid(seg.start, seg.end, radius)
                    if wire_solid is None:
                        return None
                    return _solid_to_stl_bytes(wire_solid)
        return None

    return None


def _axis_to_quat(axis: tuple) -> list[float]:
    """Quaternion rotating Z-up to the given axis. Returns [w, x, y, z]."""
    from botcad.geometry import rotation_between

    return list(rotation_between((0.0, 0.0, 1.0), axis))


def _load_bot(bot_name: str):
    """Lazily load and solve a bot, returning (bot, cad_model). Thread-safe."""

    with _bot_cache_lock:
        if bot_name in _bot_cache:
            return _bot_cache[bot_name]

    import importlib.util

    design_py = PROJECT_ROOT / "bots" / bot_name / "design.py"
    if not design_py.exists():
        raise FileNotFoundError(f"No design.py for bot {bot_name}")

    t0 = time.monotonic()
    log.info("loading_bot", bot=bot_name)

    spec = importlib.util.spec_from_file_location("design", design_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    bot = mod.build()
    bot.solve()
    t1 = time.monotonic()
    log.info("bot_solved", bot=bot_name, duration_s=round(t1 - t0, 1))

    from botcad.emit.cad import build_cad

    cad = build_cad(bot)
    t2 = time.monotonic()
    log.info(
        "bot_cad_built",
        bot=bot_name,
        cad_duration_s=round(t2 - t1, 1),
        total_duration_s=round(t2 - t0, 1),
        body_count=len(bot.all_bodies),
    )

    with _bot_cache_lock:
        _bot_cache[bot_name] = (bot, cad)
    return bot, cad


def _get_cad_steps(bot_name: str, body_name: str) -> list:
    """Get cached CAD steps for a body, computing on first access."""

    cache_key = (bot_name, body_name)
    with _cad_steps_lock:
        if cache_key in _cad_steps_cache:
            return _cad_steps_cache[cache_key]

    bot, cad = _load_bot(bot_name)

    # Find the body
    target = None
    for body in bot.all_bodies:
        if body.name == body_name:
            target = body
            break
    if target is None:
        raise KeyError(f"Body '{body_name}' not found in bot '{bot_name}'")

    parent_joint = cad.parent_joint_map.get(body_name)
    wire_segs = cad.body_wire_segments.get(body_name)
    wire_segs_tuple = tuple(wire_segs) if wire_segs else None

    from botcad.emit.cad import make_body_solid_with_steps

    t0 = time.monotonic()
    log.info("building_cad_steps", bot=bot_name, body=body_name)
    steps = make_body_solid_with_steps(target, parent_joint, wire_segs_tuple)
    t1 = time.monotonic()
    log.info(
        "cad_steps_built",
        bot=bot_name,
        body=body_name,
        step_count=len(steps),
        duration_s=round(t1 - t0, 1),
    )

    with _cad_steps_lock:
        _cad_steps_cache[cache_key] = steps
    return steps


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------


@app.get("/api/dev-info")
def get_dev_info():
    """Return git branch and worktree info for the dev UI banner."""
    import subprocess

    def _run(cmd: list[str]) -> str:
        try:
            return subprocess.check_output(
                cmd, cwd=str(PROJECT_ROOT), text=True
            ).strip()
        except Exception:
            return ""

    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    worktree = _run(["git", "rev-parse", "--show-toplevel"])
    # The "main" worktree is the one without a .git file (it has a .git directory)
    git_path = Path(worktree) / ".git" if worktree else None
    is_main_worktree = git_path.is_dir() if git_path and git_path.exists() else True
    is_default = branch in ("master", "main") and is_main_worktree

    return {
        "branch": branch,
        "worktree": worktree,
        "is_default": is_default,
    }


@app.get("/api/bots")
def get_bots():
    global _bots_json
    if _bots_json is None:
        bots = [{"name": b["name"], "category": "bot"} for b in _discover_bots()]
        _bots_json = json.dumps(bots).encode()
    return Response(content=_bots_json, media_type="application/json")


@app.get("/api/components")
def get_components():
    global _catalog_json
    if _catalog_json is None:
        catalog = [
            _component_to_json(comp, category)
            for _factory, comp, category in _component_registry.values()
        ]
        _catalog_json = json.dumps(catalog).encode()
    return Response(content=_catalog_json, media_type="application/json")


@app.get("/api/components/{name:path}/stl/{part}")
def get_component_stl(name: str, part: str):
    # Handle screw STL requests (generated by fasteners endpoint)
    # Shared STL cache lookups (screws, wire stubs, connectors)
    if (
        name.startswith(("_screw_", "_connector_")) or name == "_wire_stub"
    ) and part == "body":
        cache_key = (name, "body")
        with _stl_cache_lock:
            stl_bytes = _stl_cache.get(cache_key)
        if stl_bytes:
            return Response(
                content=stl_bytes,
                media_type="application/octet-stream",
                headers={
                    "Content-Disposition": f'attachment; filename="{name}_body.stl"'
                },
            )
        raise HTTPException(404, f"Shared STL not found: {name}")

    if name not in _component_registry:
        raise HTTPException(404, f"Unknown component: {name}")

    _factory, comp, _category = _component_registry[name]
    try:
        stl_bytes = _generate_stl_bytes(comp, part)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"STL generation failed: {e}") from e

    if stl_bytes is None:
        raise HTTPException(404, f"Part '{part}' not available for {name}")

    return Response(
        content=stl_bytes,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{name}_{part}.stl"'},
    )


# ---------------------------------------------------------------------------
# Shared helpers — used by individual endpoints AND the manifest endpoint
# ---------------------------------------------------------------------------


def _collect_multi_material(comp, meta) -> tuple[list[dict], dict[str, dict]]:
    """Return (meshes, materials) for a component's multi-material rendering.

    *meshes* is a list of ``{"file": mat_key, "material": mat_name}`` entries.
    *materials* maps material name → {color, metallic, roughness, opacity}.
    Both are empty when no multi-material emitter exists.
    """
    if meta.multi_material_emitter is None:
        return [], {}

    try:
        mm_result = meta.multi_material_emitter(comp)
    except Exception:
        return [], {}

    if mm_result is None:
        return [], {}

    meshes: list[dict] = []
    materials: dict[str, dict] = {}
    for mp in mm_result.material_programs:
        mat = mp.material
        mat_key = f"mat__{mat.name}"
        meshes.append({"file": mat_key, "material": mat.name})
        materials[mat.name] = {
            "color": list(mat.color[:3]),
            "metallic": mat.metallic,
            "roughness": mat.roughness,
            "opacity": mat.opacity,
        }

        # Pre-generate and cache the per-material STL
        cache_key = (comp.name, mat_key)
        with _stl_cache_lock:
            if cache_key not in _stl_cache:
                from botcad.shapescript.backend_occt import OcctBackend

                result = OcctBackend().execute(mp.program)
                solid = result.shapes[mp.program.output_ref.id]
                _stl_cache[cache_key] = _solid_to_stl_bytes(solid)

    return meshes, materials


def _collect_fastener_parts(comp, name: str) -> list[dict]:
    """Return fastener part entries for a component's mounting points.

    Each entry has id, name, category, parent_body, mesh, pos, quat.
    Screw STLs are cached as a side effect.
    """
    for mp in comp.mounting_points:
        cache_key = (f"_screw_{mp.diameter:.4f}", "body")
        with _stl_cache_lock:
            if cache_key in _stl_cache:
                continue
        try:
            from botcad.emit.cad import screw_solid

            stl_bytes = _solid_to_stl_bytes(screw_solid(mp.diameter))
            with _stl_cache_lock:
                _stl_cache[cache_key] = stl_bytes
        except Exception:
            pass

    parts: list[dict] = []
    for mp in comp.mounting_points:
        d_key = f"{mp.diameter:.4f}"
        parts.append(
            {
                "id": f"fastener_{name}_{mp.label}",
                "name": f"M{mp.diameter * 1000:.0f} screw",
                "category": "fastener",
                "parent_body": name,
                "mesh": f"_screw_{d_key}",
                "pos": list(mp.pos),
                "quat": _axis_to_quat(mp.axis),
            }
        )
    return parts


def _collect_wire_parts(comp, name: str) -> tuple[list[dict], list[dict]]:
    """Return (stubs, connectors) for a component's wire ports.

    Wire stub and connector STLs are cached as a side effect.
    """
    # Ensure shared wire stub STL is generated
    stub_cache_key = ("_wire_stub", "body")
    with _stl_cache_lock:
        if stub_cache_key not in _stl_cache:
            from botcad.shapescript.backend_occt import OcctBackend
            from botcad.shapescript.program import ShapeScriptBuilder

            prog = ShapeScriptBuilder()
            stub = prog.cylinder(0.0015, 0.025, tag="wire_stub")
            prog.output_ref = stub
            result = OcctBackend().execute(prog)
            _stl_cache[stub_cache_key] = _solid_to_stl_bytes(result.shapes[stub.id])

    from botcad.connectors import connector_spec
    from botcad.geometry import rotation_between

    stubs: list[dict] = []
    connectors: list[dict] = []
    for wp in comp.wire_ports:
        if not wp.connector_type:
            continue

        try:
            cspec = connector_spec(wp.connector_type)
        except KeyError:
            continue

        exit_dir = cspec.wire_exit_direction
        quat = rotation_between((0.0, 0.0, 1.0), exit_dir)
        half_len = 0.0125
        center = (
            wp.pos[0] + exit_dir[0] * half_len,
            wp.pos[1] + exit_dir[1] * half_len,
            wp.pos[2] + exit_dir[2] * half_len,
        )
        color = _BUS_COLORS.get(str(wp.bus_type), [0.53, 0.53, 0.53, 1.0])

        stubs.append(
            {
                "label": wp.label,
                "bus_type": str(wp.bus_type),
                "connector_type": wp.connector_type,
                "pos": list(center),
                "quat": [quat[0], quat[1], quat[2], quat[3]],
                "color": color,
                "stl_url": "/api/components/_wire_stub/stl/body",
            }
        )

        # Connector housing at the wire port position
        conn_key = f"_connector_{wp.connector_type}"
        cache_key = (conn_key, "body")
        with _stl_cache_lock:
            if cache_key not in _stl_cache:
                try:
                    from botcad.connectors import connector_solid

                    solid = connector_solid(cspec)
                    _stl_cache[cache_key] = _solid_to_stl_bytes(solid)
                except Exception:
                    pass

        conn_quat = rotation_between((0.0, 0.0, 1.0), cspec.mating_direction)
        connectors.append(
            {
                "label": f"{wp.label} ({cspec.label})",
                "raw_label": wp.label,
                "bus_type": str(wp.bus_type),
                "connector_type": wp.connector_type,
                "connector_mesh": conn_key,
                "pos": list(wp.pos),
                "quat": [conn_quat[0], conn_quat[1], conn_quat[2], conn_quat[3]],
                "color": color,
                "stl_url": f"/api/components/{conn_key}/stl/body",
            }
        )

    return stubs, connectors


@app.get("/api/components/{name:path}/materials")
def get_component_materials(name: str):
    """Return per-material mesh info for a component's multi-material rendering.

    Response: {"materials": [{"name": "fr4_green", "color": [r,g,b], "metallic": 0.0,
    "roughness": 0.85, "stl_url": "/api/components/BEC5V/stl/mat__fr4_green"}, ...]}
    Returns empty materials list if no multi-material emitter exists.
    """
    if name not in _component_registry:
        raise HTTPException(404, f"Unknown component: {name}")

    _factory, comp, _category = _component_registry[name]

    from botcad.component import get_component_meta

    meta = get_component_meta(comp.kind)
    meshes, materials = _collect_multi_material(comp, meta)
    if not meshes:
        return {"materials": []}

    result = []
    for m in meshes:
        mat = materials[m["material"]]
        result.append(
            {
                "name": m["material"],
                "color": mat["color"],
                "metallic": mat["metallic"],
                "roughness": mat["roughness"],
                "stl_url": f"/api/components/{name}/stl/{m['file']}",
            }
        )

    return {"materials": result}


@app.get("/api/components/{name:path}/fasteners")
def get_component_fasteners(name: str):
    if name not in _component_registry:
        raise HTTPException(404, f"Unknown component: {name}")

    _factory, comp, _category = _component_registry[name]
    parts = _collect_fastener_parts(comp, name)

    # Re-shape to the endpoint's legacy format (label, diameter_mm, stl_url)
    fasteners = []
    for part, mp in zip(parts, comp.mounting_points, strict=True):
        d_key = f"{mp.diameter:.4f}"
        fasteners.append(
            {
                "label": mp.label,
                "pos": part["pos"],
                "quat": part["quat"],
                "diameter_mm": round(mp.diameter * 1000, 2),
                "stl_url": f"/api/components/_screw_{d_key}/stl/body",
            }
        )

    return {"fasteners": fasteners}


@app.get("/api/components/{name:path}/wires")
def get_component_wires(name: str):
    """Return wire stub positions for a component's wire ports."""
    if name not in _component_registry:
        raise HTTPException(404, f"Unknown component: {name}")

    _factory, comp, _category = _component_registry[name]
    stubs, connectors = _collect_wire_parts(comp, name)

    # Return only the original wire-endpoint fields
    wires_out = [
        {k: v for k, v in s.items() if k not in ("connector_type",)} for s in stubs
    ]
    connectors_out = [
        {
            k: v
            for k, v in c.items()
            if k not in ("raw_label", "bus_type", "connector_type", "connector_mesh")
        }
        for c in connectors
    ]
    return {"wires": wires_out, "connectors": connectors_out}


@app.get("/api/components/{name:path}/manifest")
def get_component_manifest(name: str):
    """Return a ViewerManifest-compatible JSON for a single component.

    Shape matches the bot viewer manifest so ComponentTree can consume it
    directly. One body (identity pose), one self-referential mount,
    fasteners/wires as parts, and a materials dict.
    """
    if name not in _component_registry:
        raise HTTPException(404, f"Unknown component: {name}")

    _factory, comp, _category = _component_registry[name]

    from botcad.component import get_component_meta

    meta = get_component_meta(comp.kind)

    # ── Body: single structural body at origin ──
    comp_color = (
        list(comp.default_material.color[:3])
        if comp.default_material
        else [0.541, 0.608, 0.659]
    )
    body = {
        "name": name,
        "parent": None,
        "role": "structure",
        "mesh": "body",
        "pos": [0, 0, 0],
        "quat": [1, 0, 0, 0],
        "color": comp_color,
    }

    # ── Mount: self-referential ──
    mount: dict = {
        "body": name,
        "label": name,
        "component": name,
        "category": "component",
        "mesh": "body",
        "pos": [0, 0, 0],
        "quat": [1, 0, 0, 0],
    }

    # Multi-material meshes
    mm_meshes, materials = _collect_multi_material(comp, meta)
    if mm_meshes:
        mount["meshes"] = mm_meshes

    # Add component default material if not already in materials dict
    if comp.default_material and comp.default_material.name not in materials:
        mat = comp.default_material
        materials[mat.name] = {
            "color": list(mat.color[:3]),
            "metallic": mat.metallic,
            "roughness": mat.roughness,
            "opacity": mat.opacity,
        }

    # ── Parts: fasteners from mounting points ──
    parts: list[dict] = _collect_fastener_parts(comp, name)

    # ── Parts: wires from wire ports ──
    wire_stubs, wire_connectors = _collect_wire_parts(comp, name)
    parts.extend(
        {
            "id": f"wire_{name}_{stub['label']}",
            "name": stub["label"],
            "category": "wire",
            "parent_body": name,
            "mesh": "_wire_stub",
            "pos": stub["pos"],
            "quat": stub["quat"],
            "bus_type": stub["bus_type"],
            "connector_type": stub["connector_type"],
            "color": stub["color"],
        }
        for stub in wire_stubs
    )
    parts.extend(
        {
            "id": f"connector_{name}_{conn['raw_label']}",
            "name": conn["label"],
            "category": "wire",
            "parent_body": name,
            "mesh": conn["connector_mesh"],
            "pos": conn["pos"],
            "quat": conn["quat"],
            "bus_type": conn["bus_type"],
            "connector_type": conn["connector_type"],
            "color": conn["color"],
        }
        for conn in wire_connectors
    )

    # ── Design layer mounts (servo bracket, coupler, etc.) ──
    # "servo" is the housing mesh — same as body, skip to avoid doubling.
    skip_layers = {"body", "servo", "fasteners", "wires"}
    mounts: list[dict] = [mount]
    for layer_id in meta.layers:
        if layer_id in skip_layers:
            continue
        layer_meta = _DESIGN_LAYER_META.get(layer_id)
        if not layer_meta:
            continue

        # Skip layers that can't generate geometry for this component
        if _generate_solid(comp, layer_id) is None:
            continue

        is_clearance = "insertion" in layer_id or "envelope" in layer_id
        layer_color = layer_meta["color"] + [layer_meta["opacity"]]

        mounts.append(
            {
                "body": name,
                "label": layer_id,
                "component": layer_meta["label"],
                "category": "clearance" if is_clearance else "design_layer",
                "mesh": layer_id,
                "pos": [0, 0, 0],
                "quat": [1, 0, 0, 0],
                "color": layer_color,
            }
        )

        materials[layer_id] = {
            "color": layer_meta["color"],
            "metallic": 0.0,
            "roughness": 0.8,
            "opacity": layer_meta["opacity"],
        }

    return {
        "bot_name": name,
        "bodies": [body],
        "joints": [],
        "mounts": mounts,
        "parts": parts,
        "materials": materials,
        "assemblies": [],
    }


@app.post("/api/components/{name:path}/render-svg")
async def render_svg(name: str, request: Request):
    """Render a high-quality 2D SVG projection/section of a component."""
    if name not in _component_registry:
        raise HTTPException(404, f"Unknown component: {name}")

    body = await request.json()

    view_dir = body["view_dir"]
    view_up = body.get("view_up", [0, 0, 1])
    layers = body.get("layers", [])
    section_params = body.get("section")

    _factory, comp, _category = _component_registry[name]

    # Build solids for requested layers
    solids = []
    for layer_id in layers:
        solid = _generate_solid(comp, layer_id)
        if solid is None:
            continue
        rgb = _layer_color(comp, layer_id)
        solids.append((layer_id, solid, rgb))

    if not solids:
        raise HTTPException(400, "No valid layers to render")

    # Camera origin: far along view direction, looking toward world origin
    origin = tuple(d * 10 for d in view_dir)

    # Section plane
    section_plane = None
    if section_params:
        from build123d import Plane, Vector

        axis_idx = {"x": 0, "y": 1, "z": 2}[section_params["axis"]]
        normal = [0.0, 0.0, 0.0]
        normal[axis_idx] = 1.0
        plane_origin = [0.0, 0.0, 0.0]
        plane_origin[axis_idx] = section_params["position"]
        section_plane = Plane(
            origin=Vector(*plane_origin),
            z_dir=Vector(*normal),
        )

    try:
        from botcad.render_svg import render_component_svg

        svg = render_component_svg(
            solids,
            origin,
            tuple(view_up),
            section_plane,
            annotate=body.get("annotate"),
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"SVG render failed: {e}") from e

    return Response(content=svg.encode("utf-8"), media_type="image/svg+xml")


@app.get("/api/bots/{bot}/viewer_manifest")
def get_viewer_manifest(bot: str):
    """Serve viewer manifest on demand."""
    try:
        bot_obj, _cad = _load_bot(bot)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Viewer manifest generation failed: {e}") from e

    from botcad.emit.viewer import build_viewer_manifest

    return build_viewer_manifest(bot_obj, packing=bot_obj.packing_result)


@app.get("/api/bots/{bot}/assembly-sequence")
def get_assembly_sequence(bot: str):
    """Return the assembly sequence for a bot, enriched with mesh references.

    Each op carries a ``meshes`` array so the viewer can load STLs directly
    from the assembly sequence without a separate manifest lookup.
    """
    try:
        bot_obj, _cad = _load_bot(bot)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Assembly sequence generation failed: {e}") from e

    from botcad.assembly.build import build_assembly_sequence
    from botcad.assembly.refs import ComponentRef, FastenerRef, WireRef
    from botcad.assembly.tools import TOOL_LIBRARY
    from botcad.emit.viewer import build_viewer_manifest
    from botcad.formatting import pretty_repr

    seq = build_assembly_sequence(bot_obj)
    manifest = build_viewer_manifest(bot_obj, packing=bot_obj.packing_result)

    # Build lookup indices from the manifest for fast mesh resolution
    _body_by_name = {b["name"]: b for b in manifest["bodies"]}
    _mounts_by_key = {
        (m["body"], m["label"]): m for m in (manifest.get("mounts") or [])
    }
    # Fastener parts indexed by (body, per-body-index)
    _fastener_parts: dict[tuple[str, int], dict] = {}
    _fastener_count: dict[str, int] = {}
    for p in manifest.get("parts") or []:
        if p.get("category") == "fastener":
            body = p["parent_body"]
            idx = _fastener_count.get(body, 0)
            _fastener_count[body] = idx + 1
            _fastener_parts[(body, idx)] = p
    # Wire parts indexed by route label
    _wire_parts: dict[str, list[dict]] = {}
    for p in manifest.get("parts") or []:
        if p.get("category") == "wire" and p.get("wire_kind") != "stub":
            label = p["name"]
            _wire_parts.setdefault(label, []).append(p)

    def _round_vec(v):
        return [round(x, 6) for x in v]

    def _meshes_for_op(op) -> list[dict]:
        """Resolve mesh file + pos/quat for an assembly op."""
        meshes: list[dict] = []
        target = op.target

        if op.action.value == "insert" and isinstance(target, ComponentRef):
            if target.mount_label == "__body__":
                # Body shell placement
                body = _body_by_name.get(str(target.body))
                if body:
                    meshes.append(
                        {
                            "file": body["mesh"],
                            "pos": body["pos"],
                            "quat": body["quat"],
                        }
                    )
                    if body.get("color"):
                        meshes[-1]["color"] = body["color"]
            elif target.mount_label.startswith("servo_"):
                # Servo insertion — look for servo body in manifest
                joint_name = target.mount_label[len("servo_") :]
                servo_body = _body_by_name.get(f"servo_{joint_name}")
                if servo_body:
                    meshes.append(
                        {
                            "file": servo_body["mesh"],
                            "pos": servo_body["pos"],
                            "quat": servo_body["quat"],
                        }
                    )
                    if servo_body.get("color"):
                        meshes[-1]["color"] = servo_body["color"]
                # Also include the horn if it exists
                horn_body = _body_by_name.get(f"horn_{joint_name}")
                if horn_body:
                    meshes.append(
                        {
                            "file": horn_body["mesh"],
                            "pos": horn_body["pos"],
                            "quat": horn_body["quat"],
                        }
                    )
                    if horn_body.get("color"):
                        meshes[-1]["color"] = horn_body["color"]
            else:
                # Mounted component (battery, camera, etc.)
                mount = _mounts_by_key.get((str(target.body), target.mount_label))
                if mount:
                    if mount.get("meshes"):
                        # Multi-material component
                        meshes.extend(
                            {
                                "file": sub["file"],
                                "pos": mount["pos"],
                                "quat": mount["quat"],
                                "material": sub.get("material"),
                            }
                            for sub in mount["meshes"]
                        )
                    else:
                        entry: dict = {
                            "file": mount["mesh"],
                            "pos": mount["pos"],
                            "quat": mount["quat"],
                        }
                        if mount.get("color"):
                            entry["color"] = mount["color"]
                        meshes.append(entry)

        elif op.action.value == "fasten" and isinstance(target, FastenerRef):
            part = _fastener_parts.get((str(target.body), target.index))
            if part and part.get("pos") and part.get("quat"):
                meshes.append(
                    {
                        "file": part["mesh"],
                        "pos": part["pos"],
                        "quat": part["quat"],
                    }
                )

        elif op.action.value == "route_wire" and isinstance(target, WireRef):
            wire_parts = _wire_parts.get(target.label, [])
            for wp in wire_parts:
                entry = {
                    "file": wp["mesh"],
                    "pos": wp.get("pos", [0.0, 0.0, 0.0]),
                    "quat": wp.get("quat", [1.0, 0.0, 0.0, 0.0]),
                }
                if wp.get("color"):
                    entry["color"] = wp["color"]
                meshes.append(entry)

        # CONNECT and ARTICULATE ops have no mesh
        return meshes

    def _serialize_target(target):
        if isinstance(target, ComponentRef):
            return {
                "type": "component",
                "body": str(target.body),
                "mount_label": target.mount_label,
            }
        if isinstance(target, FastenerRef):
            return {"type": "fastener", "body": str(target.body), "index": target.index}
        if isinstance(target, WireRef):
            return {"type": "wire", "label": target.label}
        # JointId (str subclass)
        return {"type": "joint", "id": str(target)}

    ops = []
    for op in seq.ops:
        op_dict = {
            "step": op.step,
            "action": op.action.value,
            "target": _serialize_target(op.target),
            "body": str(op.body),
            "tool": op.tool.value if op.tool else None,
            "approach_axis": op.approach_axis,
            "angle": op.angle,
            "prerequisites": list(op.prerequisites),
            "description": op.description,
            "repr": pretty_repr(repr(op)),
            "repr_oneline": repr(op),
            "meshes": _meshes_for_op(op),
        }
        ops.append(op_dict)

    tool_library = {}
    for kind, spec in TOOL_LIBRARY.items():
        tool_library[kind.value] = {
            "shaft_diameter": spec.shaft_diameter,
            "shaft_length": spec.shaft_length,
            "head_diameter": spec.head_diameter,
            "grip_clearance": spec.grip_clearance,
        }

    return {"ops": ops, "tool_library": tool_library}


# ---------------------------------------------------------------------------
# Async DFM run/status/findings
# ---------------------------------------------------------------------------


@dataclass
class DFMRun:
    run_id: str
    bot_name: str
    created_at: float = field(default_factory=time.monotonic)
    state: str = "running"  # "running" | "complete" | "failed"
    checks: list[dict] = field(default_factory=list)  # per-check status
    findings: list[dict] = field(default_factory=list)  # serialized findings
    error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


_dfm_runs: dict[str, DFMRun] = {}

_DFM_RUN_TTL = 600.0  # 10 minutes


def _evict_stale_dfm_runs() -> None:
    """Remove completed/failed DFM runs older than _DFM_RUN_TTL."""
    now = time.monotonic()
    stale = [
        rid
        for rid, run in _dfm_runs.items()
        if run.state in ("complete", "failed") and now - run.created_at > _DFM_RUN_TTL
    ]
    for rid in stale:
        del _dfm_runs[rid]


def _serialize_finding(f) -> dict:
    """Serialize a DFMFinding to a JSON-safe dict."""
    from botcad.assembly.refs import ComponentRef, FastenerRef, WireRef

    if isinstance(f.target, ComponentRef):
        target = {
            "type": "component",
            "body": str(f.target.body),
            "mount_label": f.target.mount_label,
        }
    elif isinstance(f.target, FastenerRef):
        target = {
            "type": "fastener",
            "body": str(f.target.body),
            "index": f.target.index,
        }
    elif isinstance(f.target, WireRef):
        target = {"type": "wire", "label": f.target.label}
    else:
        target = {"type": "joint", "id": str(f.target)}

    return {
        "id": f.id,
        "check_name": f.check_name,
        "severity": f.severity.value,
        "body": str(f.body),
        "target": target,
        "assembly_step": f.assembly_step,
        "title": f.title,
        "description": f.description,
        "pos": list(f.pos),
        "direction": list(f.direction) if f.direction else None,
        "measured": f.measured,
        "threshold": f.threshold,
        "has_overlay": f.has_overlay,
    }


def _run_dfm_background(run: DFMRun) -> None:
    """Execute DFM checks in a background thread, updating the run incrementally."""
    try:
        bot_obj, _cad = _load_bot(run.bot_name)

        from botcad.assembly.build import build_assembly_sequence
        from botcad.dfm.runner import discover_checks

        seq = build_assembly_sequence(bot_obj)
        checks = discover_checks()

        with run.lock:
            run.checks = [{"name": c.name, "state": "pending"} for c in checks]

        for i, check in enumerate(checks):
            with run.lock:
                run.checks[i]["state"] = "running"

            try:
                findings = check.run(bot_obj, seq)
                serialized = [_serialize_finding(f) for f in findings]
                with run.lock:
                    run.findings.extend(serialized)
                    run.checks[i]["state"] = "complete"
                    run.checks[i]["findings_count"] = len(findings)
            except Exception as e:
                with run.lock:
                    run.checks[i]["state"] = "failed"
                    run.checks[i]["error"] = str(e)

        with run.lock:
            run.state = "complete"

    except Exception as e:
        traceback.print_exc()
        with run.lock:
            run.state = "failed"
            run.error = str(e)


@app.post("/api/bots/{bot}/dfm/run")
def start_dfm_run(bot: str):
    """Start an async DFM run. Returns a run_id for polling."""
    _evict_stale_dfm_runs()

    # Validate bot exists
    try:
        _load_bot(bot)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Failed to load bot: {e}") from e

    run_id = uuid.uuid4().hex[:12]
    run = DFMRun(run_id=run_id, bot_name=bot)
    _dfm_runs[run_id] = run

    t = threading.Thread(target=_run_dfm_background, args=(run,), daemon=True)
    t.start()

    return {"run_id": run_id}


@app.get("/api/bots/{bot}/dfm/{run_id}/status")
def get_dfm_status(bot: str, run_id: str):
    """Poll status of a DFM run."""
    run = _dfm_runs.get(run_id)
    if run is None:
        raise HTTPException(404, f"DFM run {run_id} not found")
    if run.bot_name != bot:
        raise HTTPException(404, f"DFM run {run_id} not found for bot {bot}")

    with run.lock:
        checks_total = len(run.checks)
        checks_complete = sum(
            1 for c in run.checks if c["state"] in ("complete", "failed")
        )
        return {
            "state": run.state,
            "checks_total": checks_total,
            "checks_complete": checks_complete,
            "checks": list(run.checks),
            "error": run.error,
        }


@app.get("/api/bots/{bot}/dfm/{run_id}/findings")
def get_dfm_findings(bot: str, run_id: str):
    """Get findings from a DFM run (may be partial if still running)."""
    run = _dfm_runs.get(run_id)
    if run is None:
        raise HTTPException(404, f"DFM run {run_id} not found")
    if run.bot_name != bot:
        raise HTTPException(404, f"DFM run {run_id} not found for bot {bot}")

    with run.lock:
        return {"findings": list(run.findings)}


@app.get("/api/bots/{bot}/bot.xml")
def get_bot_xml(bot: str):
    """Serve on-demand MuJoCo bot.xml generated from the bot skeleton."""
    try:
        bot_obj, _cad = _load_bot(bot)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"bot.xml generation failed: {e}") from e

    from botcad.emit.mujoco import generate_mujoco_xml

    xml_str = generate_mujoco_xml(bot_obj)
    return Response(content=xml_str.encode("utf-8"), media_type="application/xml")


@app.get("/api/bots/{bot}/meshes/{mesh_name:path}")
def get_bot_mesh(bot: str, mesh_name: str):
    """Serve on-demand mesh STL generated from cached CAD solids."""
    try:
        bot_obj, cad = _load_bot(bot)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Mesh generation failed: {e}") from e

    stem = mesh_name.removesuffix(".stl")

    try:
        stl_bytes = _generate_bot_mesh(bot_obj, cad, stem)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Mesh generation failed for {mesh_name}: {e}") from e

    if stl_bytes is None:
        raise HTTPException(404, f"Mesh not found: {mesh_name}")

    return Response(
        content=stl_bytes,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{mesh_name}"'},
    )


def _pretty_repr_cad(s: str) -> str:
    """Lazy-import pretty_repr for cad-steps endpoint."""
    from botcad.formatting import pretty_repr

    return pretty_repr(s)


@app.get("/api/bots/{bot}/body/{body}/cad-steps")
def get_cad_steps_meta(bot: str, body: str):
    """Return JSON metadata for all CAD construction steps of a body."""
    try:
        steps = _get_cad_steps(bot, body)
    except (FileNotFoundError, KeyError) as e:
        raise HTTPException(404, str(e)) from e
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"CAD step generation failed: {e}") from e

    return {
        "bot": bot,
        "body": body,
        "steps": [
            {
                "index": i,
                "label": s.label,
                "op": s.op,
                "has_tool": s.tool is not None,
                "script": s.script,
                "repr": _pretty_repr_cad(s.ir_repr),
            }
            for i, s in enumerate(steps)
        ],
    }


@app.get("/api/bots/{bot}/body/{body}/cad-steps/{idx}/stl")
def get_cad_step_stl(bot: str, body: str, idx: int):
    return _serve_cad_step_solid(bot, body, idx, attr="solid")


@app.get("/api/bots/{bot}/body/{body}/cad-steps/{idx}/tool-stl")
def get_cad_step_tool_stl(bot: str, body: str, idx: int):
    return _serve_cad_step_solid(bot, body, idx, attr="tool")


def _serve_cad_step_solid(bot_name: str, body_name: str, step_idx: int, attr: str):
    """Serve STL bytes for a CadStep's solid or tool attribute."""
    cache_key = (f"cadstep_{attr}_{bot_name}_{body_name}", str(step_idx))
    with _stl_cache_lock:
        cached = _stl_cache.get(cache_key)
    if cached:
        return Response(
            content=cached,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="{body_name}_step{step_idx}_{attr}.stl"'
            },
        )

    try:
        steps = _get_cad_steps(bot_name, body_name)
    except (FileNotFoundError, KeyError) as e:
        raise HTTPException(404, str(e)) from e
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"CAD step generation failed: {e}") from e

    if step_idx < 0 or step_idx >= len(steps):
        raise HTTPException(404, f"Step {step_idx} out of range (0-{len(steps) - 1})")

    solid = getattr(steps[step_idx], attr, None)
    if solid is None:
        raise HTTPException(404, f"Step {step_idx} has no {attr}")

    try:
        stl_bytes = _solid_to_stl_bytes(solid)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            500, f"STL export failed for step {step_idx} {attr}: {e}"
        ) from e

    with _stl_cache_lock:
        _stl_cache[cache_key] = stl_bytes

    return Response(
        content=stl_bytes,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{body_name}_step{step_idx}_{attr}.stl"'
        },
    )


@app.post("/api/bots/{bot}/fea/run")
async def run_fea(bot: str):
    """Trigger FEA analysis for all fabricated bodies."""
    try:
        t_total_start = time.monotonic()
        bot_obj, cad = _load_bot(bot)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Bot load failed: {e}") from e

    from botcad.skeleton import BodyKind

    fab_bodies = [b for b in bot_obj.all_bodies if b.kind == BodyKind.FABRICATED]
    if not fab_bodies:
        raise HTTPException(400, "No fabricated bodies found for FEA.")

    import tempfile

    from botcad.fea import analyze_component, export_stress_mesh, export_voxel_mesh
    from botcad.shapescript.backend_occt import OcctBackend
    from botcad.shapescript.emit_body import emit_body_ir

    def _get_ply_bytes(func, *args, **kwargs):
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=True) as tmp:
            func(*args, tmp.name, **kwargs)
            return Path(tmp.name).read_bytes()

    body_results: list[dict] = []
    worst_sf = float("inf")
    worst_body: str | None = None

    for target_body in fab_bodies:
        t_body_start = time.monotonic()
        try:
            parent_joint = cad.parent_joint_map.get(target_body.name)
            wire_segs = cad.body_wire_segments.get(target_body.name)
            wire_segs_tuple = tuple(wire_segs) if wire_segs else None

            prog = emit_body_ir(target_body, parent_joint, wire_segs_tuple, bot=bot_obj)
            backend = OcctBackend()
            result = backend.execute(prog)
            solid = result.shapes[prog.output_ref.id]

            torque = 2.94
            for joint in target_body.joints:
                if joint.servo:
                    torque = max(torque, joint.servo.stall_torque)

            log.info("fea_running", bot=bot, body=target_body.name)
            solve_res_tuple = analyze_component(
                solid, result, torque_nm=torque, res=(20, 20, 20), body=target_body
            )
            if not solve_res_tuple:
                body_results.append(
                    {
                        "body": target_body.name,
                        "status": "skipped",
                        "reason": "no BCs",
                    }
                )
                continue

            u_field, stress_array, vd, _fea_timings = solve_res_tuple
            max_stress = float(stress_array.numpy().max())
            eff_yield = target_body.material.effective_yield_strength
            sf = float(eff_yield / max_stress) if max_stress > 0 else 99.0

            log.info("fea_generating_meshes", body=target_body.name)
            # Find the fixed BC mask (coupler, fastener_hole, or largest bracket)
            import numpy as np

            fixed_tag_mask = None
            for tag in ["fastener_hole", "coupler"]:
                if tag in vd.tag_masks:
                    fixed_tag_mask = vd.tag_masks[tag]
                    break
            if fixed_tag_mask is None:
                # Fallback: largest non-envelope bracket
                for tag, mask in sorted(
                    vd.tag_masks.items(),
                    key=lambda kv: int(np.sum(kv[1].numpy())),
                    reverse=True,
                ):
                    if "bracket" in tag and "env" not in tag:
                        fixed_tag_mask = mask
                        break

            files = {
                f"{target_body.name}_stress_heatmap.ply": _get_ply_bytes(
                    lambda *a, _ey=eff_yield: export_stress_mesh(
                        *a, yield_strength=_ey
                    ),
                    solid,
                    u_field.space,
                    u_field,
                    stress_array,
                ),
                f"{target_body.name}_structure_voxels.ply": _get_ply_bytes(
                    export_voxel_mesh, vd, vd.inside_mask, [200, 200, 200, 100]
                ),
            }
            if fixed_tag_mask is not None:
                files[f"{target_body.name}_fixed_voxels.ply"] = _get_ply_bytes(
                    export_voxel_mesh, vd, fixed_tag_mask, [0, 255, 0, 255]
                )

            with _fea_cache_lock:
                for fname, bdata in files.items():
                    _fea_cache[(bot, target_body.name, fname)] = bdata

            t_body_end = time.monotonic()
            body_results.append(
                {
                    "body": target_body.name,
                    "status": "ok",
                    "max_stress_mpa": round(max_stress / 1e6, 2),
                    "safety_factor": round(sf, 2),
                    "yield_mpa": round(eff_yield / 1e6, 1),
                    "duration": round(t_body_end - t_body_start, 2),
                }
            )

            if sf < worst_sf:
                worst_sf = sf
                worst_body = target_body.name

        except Exception as e:
            traceback.print_exc()
            t_body_end = time.monotonic()
            body_results.append(
                {
                    "body": target_body.name,
                    "status": "error",
                    "reason": str(e),
                    "duration": round(t_body_end - t_body_start, 2),
                }
            )

    t_total_end = time.monotonic()

    if worst_sf == float("inf"):
        worst_sf = 0.0
        buildable = False
    else:
        buildable = worst_sf >= 1.0

    return {
        "status": "success",
        "buildable": buildable,
        "worst_body": worst_body,
        "worst_sf": round(worst_sf, 2),
        "bodies": body_results,
        "total_duration": round(t_total_end - t_total_start, 2),
    }


@app.get("/api/bots/{bot}/fea/{file_name}")
def get_fea_file(bot: str, file_name: str):
    """Serve cached FEA PLY files."""
    # Find any body that has this file in cache
    with _fea_cache_lock:
        for (bname, _body_name, fname), bdata in _fea_cache.items():
            if bname == bot and fname == file_name:
                return Response(content=bdata, media_type="application/octet-stream")

    raise HTTPException(404, f"FEA file {file_name} not found. Run analysis first.")


# ---------------------------------------------------------------------------
# Client log ingestion — browser logs written to logs/client.log
# ---------------------------------------------------------------------------

# Dedicated file for client logs, preserving client-side timestamps and order.
# Not routed through structlog — entries are written as JSON lines directly.
_client_log_lock = threading.Lock()


def _get_client_log_path() -> Path:
    path = PROJECT_ROOT / "logs"
    path.mkdir(exist_ok=True)
    return path / "client.log"


@app.post("/api/client-log")
async def post_client_log(request: Request):
    """Append browser log entries to logs/client.log as JSON lines.

    Accepts JSON: {"entries": [{"level": "info", "tag": "...", "msg": "...",
    "data": {...}, "ts": "..."}]}. Capped at 50 entries per request.
    Client timestamps are preserved as-is to maintain ordering.
    """
    body = await request.json()
    entries = body.get("entries", [])
    if not isinstance(entries, list):
        raise HTTPException(400, "entries must be a list")

    lines: list[str] = []
    for entry in entries[:50]:
        line = json.dumps(entry, separators=(",", ":"))
        lines.append(line)

    if lines:
        with _client_log_lock, open(_get_client_log_path(), "a") as f:
            f.write("\n".join(lines) + "\n")

    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Static file serving (for standalone / non-Vite usage)
# ---------------------------------------------------------------------------

# Mount project root as static fallback — serves viewer/, botcad/components/*.svg, etc.
# This must be last so API routes take priority.
app.mount("/", StaticFiles(directory=str(PROJECT_ROOT), html=True), name="static")
