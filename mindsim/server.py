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
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.staticfiles import StaticFiles

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


# ---------------------------------------------------------------------------
# Registry initialisation
# ---------------------------------------------------------------------------


def init_registry() -> None:
    """Build the component registry. Call once before serving requests."""
    global _component_registry
    if _component_registry:
        return  # already initialised
    print("Building component registry...")
    _component_registry = _build_component_registry()
    print(f"  {len(_component_registry)} components registered")


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
    from botcad.component import ServoSpec

    if isinstance(comp, ServoSpec):
        from botcad.bracket import horn_disc_params

        layers = [
            "servo",
            "bracket",
            "cradle",
            "coupler",
            "bracket_insertion_channel",
            "cradle_insertion_channel",
        ]
        if horn_disc_params(comp) is not None:
            layers.insert(1, "horn")
    else:
        layers = ["body"]
    if comp.mounting_points:
        layers.append("fasteners")
    return layers


def _component_to_json(comp, category: str) -> dict:
    """Serialize a Component instance to JSON-safe dict."""
    from botcad.component import ServoSpec

    info = {
        "name": comp.name,
        "category": category,
        "dimensions_mm": [round(d * 1000, 1) for d in comp.dimensions],
        "mass_g": round(comp.mass * 1000, 1),
        "is_servo": isinstance(comp, ServoSpec),
        "color": list(comp.color),
        "layers": _component_layers(comp),
    }
    if isinstance(comp, ServoSpec):
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
    if isinstance(comp, ServoSpec):
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
    r, g, b = comp.color
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
    from botcad.component import ServoSpec
    from botcad.emit.cad import make_component_solid

    cache_key = (comp.name, part)
    with _solid_cache_lock:
        if cache_key in _solid_cache:
            return _solid_cache[cache_key]

    solid = None
    if part == "body":
        solid = make_component_solid(comp)
    elif isinstance(comp, ServoSpec):
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
    print(f"[cad-steps] Loading bot {bot_name}...")

    spec = importlib.util.spec_from_file_location("design", design_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    bot = mod.build()
    bot.solve()
    t1 = time.monotonic()
    print(f"[cad-steps]   build + solve: {t1 - t0:.1f}s")

    from botcad.emit.cad import build_cad

    cad = build_cad(bot)
    t2 = time.monotonic()
    print(f"[cad-steps]   build_cad: {t2 - t1:.1f}s")
    print(f"[cad-steps]   total: {t2 - t0:.1f}s ({len(bot.all_bodies)} bodies)")

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
    print(f"[cad-steps] Building steps for {bot_name}:{body_name}...")
    steps = make_body_solid_with_steps(target, parent_joint, wire_segs_tuple)
    t1 = time.monotonic()
    print(f"[cad-steps]   {len(steps)} steps in {t1 - t0:.1f}s")

    with _cad_steps_lock:
        _cad_steps_cache[cache_key] = steps
    return steps


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------


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
    if name.startswith("_screw_") and part == "body":
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
        raise HTTPException(404, f"Screw STL not found: {name}")

    if name not in _component_registry:
        raise HTTPException(404, f"Unknown component: {name}")

    _factory, comp, _category = _component_registry[name]
    try:
        stl_bytes = _generate_stl_bytes(comp, part)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"STL generation failed: {e}")

    if stl_bytes is None:
        raise HTTPException(404, f"Part '{part}' not available for {name}")

    return Response(
        content=stl_bytes,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{name}_{part}.stl"'},
    )


@app.get("/api/components/{name:path}/fasteners")
def get_component_fasteners(name: str):
    if name not in _component_registry:
        raise HTTPException(404, f"Unknown component: {name}")

    _factory, comp, _category = _component_registry[name]

    # Ensure screw STLs are generated (one per unique diameter)
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

    fasteners = []
    for mp in comp.mounting_points:
        d_key = f"{mp.diameter:.4f}"
        fasteners.append(
            {
                "label": mp.label,
                "pos": list(mp.pos),
                "quat": _axis_to_quat(mp.axis),
                "diameter_mm": round(mp.diameter * 1000, 2),
                "stl_url": f"/api/components/_screw_{d_key}/stl/body",
            }
        )

    return {"fasteners": fasteners}


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
        raise HTTPException(500, f"SVG render failed: {e}")

    return Response(content=svg.encode("utf-8"), media_type="image/svg+xml")


@app.get("/api/bots/{bot}/body/{body}/cad-steps")
def get_cad_steps_meta(bot: str, body: str):
    """Return JSON metadata for all CAD construction steps of a body."""
    try:
        steps = _get_cad_steps(bot, body)
    except (FileNotFoundError, KeyError) as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"CAD step generation failed: {e}")

    return {
        "bot": bot,
        "body": body,
        "steps": [
            {
                "index": i,
                "label": s.label,
                "op": s.op,
                "has_tool": s.tool is not None,
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
        raise HTTPException(404, str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"CAD step generation failed: {e}")

    if step_idx < 0 or step_idx >= len(steps):
        raise HTTPException(404, f"Step {step_idx} out of range (0-{len(steps) - 1})")

    solid = getattr(steps[step_idx], attr, None)
    if solid is None:
        raise HTTPException(404, f"Step {step_idx} has no {attr}")

    try:
        stl_bytes = _solid_to_stl_bytes(solid)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"STL export failed for step {step_idx} {attr}: {e}")

    with _stl_cache_lock:
        _stl_cache[cache_key] = stl_bytes

    return Response(
        content=stl_bytes,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{body_name}_step{step_idx}_{attr}.stl"'
        },
    )


# ---------------------------------------------------------------------------
# Static file serving (for standalone / non-Vite usage)
# ---------------------------------------------------------------------------

# Mount project root as static fallback — serves viewer/, botcad/components/*.svg, etc.
# This must be last so API routes take priority.
app.mount("/", StaticFiles(directory=str(PROJECT_ROOT), html=True), name="static")
