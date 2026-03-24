"""
Textual TUI for MindSim training.

Single entry point for all MindSim modes: interactive TUI, view, play, train,
smoketest, quicksim, and visualize.

Usage:
    uv run mjpython main.py                    # Interactive TUI (default)
    uv run mjpython main.py view [--bot NAME]  # MuJoCo viewer
    uv run mjpython main.py play [CHECKPOINT] [--bot NAME]  # Play trained policy
    uv run mjpython main.py train [--smoketest] [--bot NAME] [--resume REF] [--num-workers N]
    uv run mjpython main.py smoketest          # Alias for train --smoketest
    uv run mjpython main.py quicksim           # Rerun debug vis
    uv run mjpython main.py visualize [--bot NAME] [--steps N]

Requires mjpython (not plain python) for MuJoCo viewer/play features.
"""

from __future__ import annotations

import argparse
import http.server
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

from textual import work
from textual.app import App
from textual.binding import Binding
from textual.theme import Theme

from botcad.colors import (
    TUI_BACKGROUND,
    TUI_ERROR,
    TUI_PRIMARY,
    TUI_PRIMARY_BG,
    TUI_SECONDARY,
    TUI_SUCCESS,
    TUI_SURFACE,
    TUI_WARNING,
)
from training.train import CommandChannel
from tui.screens.dirty_tree import DirtyTreeScreen
from tui.screens.main_menu import MainMenuScreen
from tui.screens.training_dashboard import TrainingDashboard

# Blueprint.js dark theme for Textual TUI
_BLUEPRINT_THEME = Theme(
    name="blueprint",
    primary=TUI_PRIMARY,
    secondary=TUI_SECONDARY,
    warning=TUI_WARNING,
    error=TUI_ERROR,
    success=TUI_SUCCESS,
    background=TUI_BACKGROUND,
    surface=TUI_SURFACE,
    panel=TUI_PRIMARY_BG,
    dark=True,
)

log = logging.getLogger(__name__)


class TuiLogHandler(logging.Handler):
    """Routes Python log records into the TUI log panel.

    This is the *only* path for messages to reach the log panel from
    worker threads.  ``dashboard.message()`` just calls ``log.info()``
    and this handler forwards the record to the RichLog widget.

    UI actions on the event-loop thread write to the log panel directly
    (calling ``call_from_thread`` from the event loop would deadlock),
    so we skip records originating on that thread.

    Installed when training starts, removed when training finishes
    to avoid stale references to a dead app.
    """

    def __init__(self, app: MindSimApp):
        super().__init__(level=logging.INFO)
        self._app = app
        self._event_loop_thread = threading.current_thread()
        fmt = logging.Formatter("%(message)s")
        self.setFormatter(fmt)

    def emit(self, record: logging.LogRecord):
        # UI actions on the event-loop thread write to the panel directly;
        # routing them through call_from_thread would deadlock.
        if threading.current_thread() is self._event_loop_thread:
            return
        try:
            msg = self.format(record)
            if record.levelno >= logging.ERROR:
                msg = f"[bold red]{msg}[/bold red]"
            elif record.levelno >= logging.WARNING:
                msg = f"[bold yellow]{msg}[/bold yellow]"
            self._app.call_from_thread(self._app.log_message, msg)
        except Exception:
            pass  # Don't let logging errors crash the app


def _discover_bots() -> list[dict]:
    """Scan bots/*/scene.xml and return info about each bot."""
    bots_dir = Path("bots")
    results = []
    if bots_dir.is_dir():
        for scene in sorted(bots_dir.glob("*/scene.xml")):
            name = scene.parent.name
            results.append({"name": name, "scene_path": str(scene)})
    return results


def _resolve_scene_path(bot_name: str | None) -> str:
    """Resolve a bot name to its scene.xml path.

    Args:
        bot_name: Bot directory name (e.g. "simplebiped", "simple2wheeler"),
                  or None for the default bot.

    Returns:
        Path to the bot's scene.xml file.

    Raises:
        SystemExit: If the bot name is not found.
    """
    bots = _discover_bots()
    if not bots:
        print("Error: No bots found in bots/*/scene.xml", file=sys.stderr)
        sys.exit(1)

    if bot_name is None:
        # Default to first bot (simple2wheeler comes before simplebiped alphabetically)
        return bots[0]["scene_path"]

    for bot in bots:
        if bot["name"] == bot_name:
            return bot["scene_path"]

    available = ", ".join(b["name"] for b in bots)
    print(f"Error: Unknown bot '{bot_name}'. Available: {available}", file=sys.stderr)
    sys.exit(1)


def _get_experiment_info(branch: str) -> str | None:
    """Look up the hypothesis for a branch from EXPERIMENTS.md.

    Parses the markdown table and returns the hypothesis text for the
    matching branch, or None if not found.
    """
    experiments_path = Path("EXPERIMENTS.md")
    if not experiments_path.exists():
        return None
    try:
        text = experiments_path.read_text()
    except OSError:
        return None

    for line in text.splitlines():
        # Match table rows: | `branch` | hypothesis | ... |
        m = re.match(r"\|\s*`([^`]+)`\s*\|([^|]+)\|", line)
        if m and m.group(1).strip() == branch:
            return m.group(2).strip()
    return None


def _git_is_clean() -> tuple[bool, str]:
    """Check whether the git working tree is clean.

    Returns (is_clean, status_lines) where status_lines is the raw
    ``git status --porcelain`` output.  If this isn't a git repo or
    git isn't available, returns (True, "") so training proceeds.
    """

    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = result.stdout.strip()
        return (output == "", output)
    except Exception:
        return (True, "")


def _run_claude_commit() -> bool:
    """Run Claude in print mode to commit all changes.

    Returns True if the worktree is clean afterward.
    """

    print("Running Claude to commit changes...")
    try:
        subprocess.run(
            [
                "claude",
                "-p",
                "Run git status and git diff to see current changes, then commit all changes with a descriptive commit message.",
                "--allowedTools",
                "Bash Read Grep Glob",
            ],
            timeout=120,
        )
    except Exception as e:
        print(f"Claude commit failed: {e}")
        return False

    clean, _ = _git_is_clean()
    return clean


# ---------------------------------------------------------------------------
# Main TUI App
# ---------------------------------------------------------------------------


class MindSimApp(App):
    """MindSim Textual TUI application."""

    TITLE = "MindSim"
    CSS = """
    Screen {
        background: $surface;
    }
    """
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=False),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_theme(_BLUEPRINT_THEME)
        self.theme = "blueprint"
        self.commands = CommandChannel()
        self._dashboard: TrainingDashboard | None = None
        # Set by screens to dispatch after app.run() returns
        self.next_action: str | None = None
        self.next_scene: str | None = None
        self.next_stage: int | None = None
        self.next_run_name: str | None = None
        self.next_checkpoint_path: str | None = None

    def on_mount(self) -> None:
        self.push_screen(MainMenuScreen())

    def start_viewing(self, scene_path: str | None = None, stage: int | None = None):
        """Exit TUI, then main() will launch the MuJoCo viewer."""
        if not scene_path:
            return
        self.next_action = "view"
        self.next_scene = scene_path
        self.next_stage = stage
        self.exit()

    def start_scene_preview(self):
        """Exit TUI, then main() will launch the scene preview."""
        self.next_action = "scene"
        self.exit()

    def start_playing(self, scene_path: str | None = None):
        """Exit TUI, then main() will launch play mode."""
        self.next_action = "play"
        self.next_scene = scene_path
        self.exit()

    def start_playing_run(
        self,
        run_name: str | None = None,
        scene_path: str | None = None,
        checkpoint_path: str | None = None,
    ):
        """Exit TUI, then main() will launch play mode for a specific run."""
        self.next_action = "play"
        self.next_scene = scene_path
        self.next_run_name = run_name
        self.next_checkpoint_path = checkpoint_path
        self.exit()

    def start_training(
        self,
        smoketest: bool = False,
        scene_path: str | None = None,
        resume: str | None = None,
    ):
        """Called by screens to start training (with dirty-tree gate)."""
        # If no scene_path provided (e.g. smoketest from main menu), use default
        if scene_path is None:
            bots = _discover_bots()
            scene_path = bots[0]["scene_path"] if bots else None

        # Smoketests skip the dirty-tree check
        if smoketest:
            self._do_start_training(
                smoketest=True, scene_path=scene_path, resume=resume
            )
            return

        clean, status = _git_is_clean()
        if clean:
            self._do_start_training(
                smoketest=False, scene_path=scene_path, resume=resume
            )
        else:
            self.push_screen(
                DirtyTreeScreen(
                    status_lines=status,
                    smoketest=False,
                    scene_path=scene_path,
                    resume=resume,
                )
            )

    def _do_start_training(
        self,
        smoketest: bool = False,
        scene_path: str | None = None,
        resume: str | None = None,
    ):
        """Actually start training (push dashboard and kick off worker)."""
        dashboard = TrainingDashboard()
        self._dashboard = dashboard
        self._smoketest = smoketest
        self._scene_path = scene_path
        self._resume = resume
        # Route Python log records into the TUI log panel
        self._tui_log_handler = TuiLogHandler(self)
        logging.getLogger().addHandler(self._tui_log_handler)
        self.push_screen(dashboard)
        self._run_training()

    @work(thread=True, exclusive=True)
    def _run_training(self) -> None:
        from training.train import run_training

        # Force serial collection in TUI to avoid multiprocessing FD issues
        # (Textual's event loop holds file descriptors that become invalid
        # when multiprocessing.spawn tries to inherit them)
        num_workers = 1
        try:
            run_training(
                self,
                self.commands,
                smoketest=self._smoketest,
                num_workers=num_workers,
                scene_path=self._scene_path,
                resume=self._resume,
            )
        except KeyboardInterrupt:
            log.warning("Training interrupted by user")
            try:
                self.call_from_thread(
                    self.log_message,
                    "[bold yellow]Training interrupted by user[/bold yellow]",
                )
            except Exception:
                pass
        except Exception:
            log.exception("Training crashed in TUI worker thread")
            try:
                import traceback

                err = traceback.format_exc().splitlines()[-1]
                self.call_from_thread(
                    self.log_message,
                    f"[bold red]Training crashed![/bold red] {err}",
                )
            except Exception:
                pass
        finally:
            # Remove TUI log handler to avoid stale references
            if hasattr(self, "_tui_log_handler"):
                logging.getLogger().removeHandler(self._tui_log_handler)

    def send_command(self, cmd: str):
        self.commands.send(cmd)

    def update_metrics(self, batch: int, metrics: dict):
        """Called from training thread via call_from_thread."""
        if self._dashboard:
            self._dashboard.update_metrics(batch, metrics)

    def log_message(self, text: str):
        """Called from training thread via call_from_thread."""
        if self._dashboard:
            self._dashboard.log_message(text)

    def mark_finished(self):
        """Called from training thread via call_from_thread."""
        if self._dashboard:
            self._dashboard.mark_finished()

    def ai_commentary(self, text: str):
        """Called from training thread via call_from_thread."""
        if self._dashboard:
            self._dashboard.log_ai_commentary(text)

    def set_header(
        self,
        run_name: str,
        branch: str,
        algorithm: str,
        wandb_url: str | None,
        bot_name: str | None = None,
        experiment_hypothesis: str | None = None,
    ):
        """Called from training thread via call_from_thread."""
        if self._dashboard:
            self._dashboard.set_header(
                run_name,
                branch,
                algorithm,
                wandb_url,
                bot_name,
                experiment_hypothesis,
            )

    def set_total_batches(self, total: int | None):
        """Called from training thread via call_from_thread."""
        if self._dashboard:
            self._dashboard._total_batches = total


def _build_component_registry() -> dict:
    """Build a registry of all components from botcad.components factories.

    Returns dict mapping component name → (factory_func, Component instance).
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
            "bracket_envelope",
            "cradle_envelope",
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
            svg_path = Path("botcad/components") / f"drawing_{dt}_{safe}.svg"
            if svg_path.exists():
                drawings.append(
                    {
                        "type": dt,
                        "label": dt.replace("_", " ").title(),
                        "url": f"/{svg_path}",
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


# Layer colors — mirrors client-side LAYER_META.colorHex
_LAYER_COLORS: dict[str, tuple[int, int, int]] = {
    "servo": (24, 32, 38),
    "horn": (232, 232, 232),
    "bracket": (206, 217, 224),
    "cradle": (206, 217, 224),
    "coupler": (245, 86, 86),
    "bracket_envelope": (245, 86, 86),
    "cradle_envelope": (245, 86, 86),
    "fasteners": (212, 168, 67),
}


def _layer_color(comp, layer_id: str) -> tuple[int, int, int]:
    """Return RGB color for a component layer."""
    if layer_id in _LAYER_COLORS:
        return _LAYER_COLORS[layer_id]
    r, g, b = comp.color
    return (int(r * 255), int(g * 255), int(b * 255))


# Cached JSON responses (built once, served on every request)
_catalog_json: bytes | None = None
_bots_json: bytes | None = None

# CAD steps cache: (bot_name, body_name) → list[CadStep]
_cad_steps_cache: dict[tuple[str, str], list] = {}
_cad_steps_lock = threading.Lock()

# Bot object cache: bot_name → (bot, cad_model)
_bot_cache: dict[str, tuple] = {}
_bot_cache_lock = threading.Lock()


def _load_bot(bot_name: str):
    """Lazily load and solve a bot, returning (bot, cad_model). Thread-safe."""

    with _bot_cache_lock:
        if bot_name in _bot_cache:
            return _bot_cache[bot_name]

    import importlib.util

    design_py = Path("bots") / bot_name / "design.py"
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


def _solid_to_stl_bytes(solid) -> bytes:
    """Export a build123d Solid to STL bytes via a temp file."""
    import tempfile

    from build123d import export_stl

    with tempfile.NamedTemporaryFile(suffix=".stl", delete=True) as tmp:
        export_stl(solid, tmp.name)
        return Path(tmp.name).read_bytes()


def _generate_solid(comp, part: str):
    """Generate a build123d Solid for a component part. Returns None if not applicable.

    All underlying factory functions are @lru_cache'd and route through
    ShapeScript (which uses DiskCache), so no additional caching is needed here.
    """
    from botcad.component import ServoSpec
    from botcad.emit.cad import make_component_solid

    if part == "body":
        return make_component_solid(comp)

    if isinstance(comp, ServoSpec):
        from botcad.bracket import (
            BracketSpec,
            bracket_envelope,
            bracket_solid,
            coupler_solid,
            cradle_envelope,
            cradle_solid,
            servo_solid,
        )

        spec = BracketSpec()
        if part == "bracket":
            return bracket_solid(comp, spec)
        if part == "servo":
            return servo_solid(comp)
        if part == "horn":
            from botcad.bracket import horn_disc_params
            from botcad.emit.cad import _horn_solid

            params = horn_disc_params(comp)
            if params is not None:
                from build123d import Location

                horn = _horn_solid(comp)
                if horn is not None:
                    return horn.locate(
                        Location(
                            (params.center_xy[0], params.center_xy[1], params.center_z)
                        )
                    )
        elif part == "cradle":
            return cradle_solid(comp, spec)
        elif part == "coupler":
            return coupler_solid(comp, spec)
        elif part == "bracket_envelope":
            return bracket_envelope(comp, spec)
        elif part == "cradle_envelope":
            return cradle_envelope(comp, spec)

    return None


def _generate_stl_bytes(comp, part: str) -> bytes | None:
    """Generate STL bytes for a component part. Returns None if not applicable.

    The underlying solid factories are @lru_cache'd, so repeated calls
    for the same component/part reuse the cached solid.
    """
    solid = _generate_solid(comp, part)
    if solid is None:
        return None
    return _solid_to_stl_bytes(solid)


def _axis_to_quat(axis: tuple) -> list[float]:
    """Quaternion rotating Z-up to the given axis. Returns [w, x, y, z]."""
    from botcad.geometry import rotation_between

    return list(rotation_between((0.0, 0.0, 1.0), axis))


class ViewerHTTPHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler that serves static files and component API endpoints."""

    # Set by server setup
    component_registry: dict = {}
    project_root: str = "."

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=self.project_root, **kwargs)

    def end_headers(self):
        # Disable all caching — localhost dev server, always serve fresh
        self.send_header(
            "Cache-Control", "no-store, no-cache, must-revalidate, max-age=0"
        )
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def _send_json(self, payload: bytes):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_binary(self, data: bytes, filename: str | None = None):
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        if filename:
            self.send_header(
                "Content-Disposition", f'attachment; filename="{filename}"'
            )
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        # Decode percent-encoded path segments (e.g. spaces in component names)
        from urllib.parse import unquote

        path = unquote(self.path)

        # API routes
        if path == "/api/bots":
            self._handle_bots_list()
            return

        if path == "/api/components":
            self._handle_component_catalog()
            return

        # /api/components/<name>/fasteners
        m = re.match(r"^/api/components/([^/]+)/fasteners$", path)
        if m:
            self._handle_component_fasteners(m.group(1))
            return

        # /api/bots/<bot>/body/<body>/cad-steps
        m = re.match(r"^/api/bots/([^/]+)/body/([^/]+)/cad-steps$", path)
        if m:
            self._handle_cad_steps_meta(m.group(1), m.group(2))
            return

        # /api/bots/<bot>/body/<body>/cad-steps/<idx>/stl
        m = re.match(r"^/api/bots/([^/]+)/body/([^/]+)/cad-steps/(\d+)/stl$", path)
        if m:
            self._handle_cad_step_stl(m.group(1), m.group(2), int(m.group(3)))
            return

        # /api/bots/<bot>/body/<body>/cad-steps/<idx>/tool-stl
        m = re.match(r"^/api/bots/([^/]+)/body/([^/]+)/cad-steps/(\d+)/tool-stl$", path)
        if m:
            self._handle_cad_step_tool_stl(m.group(1), m.group(2), int(m.group(3)))
            return

        # /api/components/<name>/stl/<part>
        m = re.match(r"^/api/components/([^/]+)/stl/(\w+)$", path)
        if m:
            self._handle_component_stl(m.group(1), m.group(2))
            return

        # Fall through to static file serving
        super().do_GET()

    def do_POST(self):
        from urllib.parse import unquote

        path = unquote(self.path)

        m = re.match(r"^/api/components/([^/]+)/render-svg$", path)
        if m:
            self._handle_render_svg(m.group(1))
            return

        self.send_error(404)

    def _handle_render_svg(self, name: str):
        """Render a high-quality 2D SVG projection/section of a component."""
        if name not in self.component_registry:
            self.send_error(404, f"Unknown component: {name}")
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length))

        view_dir = body["view_dir"]
        view_up = body.get("view_up", [0, 0, 1])
        layers = body.get("layers", [])
        section_params = body.get("section")

        _factory, comp, _category = self.component_registry[name]

        # Build solids for requested layers
        solids = []
        for layer_id in layers:
            solid = _generate_solid(comp, layer_id)
            if solid is None:
                continue
            rgb = _layer_color(comp, layer_id)
            solids.append((layer_id, solid, rgb))

        if not solids:
            self.send_error(400, "No valid layers to render")
            return

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
            self.send_error(500, f"SVG render failed: {e}")
            return

        svg_bytes = svg.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "image/svg+xml")
        self.send_header("Content-Length", str(len(svg_bytes)))
        self.end_headers()
        self.wfile.write(svg_bytes)

    def _handle_bots_list(self):
        global _bots_json
        if _bots_json is None:
            bots = [{"name": b["name"], "category": "bot"} for b in _discover_bots()]
            _bots_json = json.dumps(bots).encode()
        self._send_json(_bots_json)

    def _handle_component_catalog(self):
        global _catalog_json
        if _catalog_json is None:
            catalog = [
                _component_to_json(comp, category)
                for _factory, comp, category in self.component_registry.values()
            ]
            _catalog_json = json.dumps(catalog).encode()
        self._send_json(_catalog_json)

    def _handle_component_stl(self, name: str, part: str):
        # Handle screw STL requests (generated on-the-fly from @lru_cache'd solids)
        if name.startswith("_screw_") and part == "body":
            try:
                from botcad.emit.cad import screw_solid

                diameter = float(name.removeprefix("_screw_"))
                stl_bytes = _solid_to_stl_bytes(screw_solid(diameter))
                self._send_binary(stl_bytes)
            except Exception:
                self.send_error(404, f"Screw STL not found: {name}")
            return

        if name not in self.component_registry:
            self.send_error(404, f"Unknown component: {name}")
            return

        _factory, comp, _category = self.component_registry[name]
        try:
            stl_bytes = _generate_stl_bytes(comp, part)
        except Exception as e:
            self.send_error(500, f"STL generation failed: {e}")
            return

        if stl_bytes is None:
            self.send_error(404, f"Part '{part}' not available for {name}")
            return

        self._send_binary(stl_bytes, f"{name}_{part}.stl")

    def _handle_component_fasteners(self, name: str):
        """Return JSON with screw positions + STL URLs for each mounting point."""

        if name not in self.component_registry:
            self.send_error(404, f"Unknown component: {name}")
            return

        _factory, comp, _category = self.component_registry[name]

        # Screw STLs are generated on-the-fly by _handle_component_stl;
        # screw_solid() is @lru_cache'd so the underlying geometry is built once.

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

        self._send_json(json.dumps({"fasteners": fasteners}).encode())

    def _handle_cad_steps_meta(self, bot_name: str, body_name: str):
        """Return JSON metadata for all CAD construction steps of a body."""
        try:
            steps = _get_cad_steps(bot_name, body_name)
        except FileNotFoundError as e:
            self.send_error(404, str(e))
            return
        except KeyError as e:
            self.send_error(404, str(e))
            return
        except Exception as e:
            self.send_error(500, f"CAD step generation failed: {e}")
            return

        payload = json.dumps(
            {
                "bot": bot_name,
                "body": body_name,
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
        ).encode()
        self._send_json(payload)

    def _handle_cad_step_stl(self, bot_name: str, body_name: str, step_idx: int):
        self._serve_cad_step_solid(bot_name, body_name, step_idx, attr="solid")

    def _handle_cad_step_tool_stl(self, bot_name: str, body_name: str, step_idx: int):
        self._serve_cad_step_solid(bot_name, body_name, step_idx, attr="tool")

    def _serve_cad_step_solid(
        self, bot_name: str, body_name: str, step_idx: int, attr: str
    ):
        """Serve STL bytes for a CadStep's solid or tool attribute.

        Steps are cached in _cad_steps_cache; STL export runs each time
        (these are infrequent debug/visualization requests).
        """
        try:
            steps = _get_cad_steps(bot_name, body_name)
        except (FileNotFoundError, KeyError) as e:
            self.send_error(404, str(e))
            return
        except Exception as e:
            self.send_error(500, f"CAD step generation failed: {e}")
            return

        if step_idx < 0 or step_idx >= len(steps):
            self.send_error(404, f"Step {step_idx} out of range (0-{len(steps) - 1})")
            return

        solid = getattr(steps[step_idx], attr, None)
        if solid is None:
            self.send_error(404, f"Step {step_idx} has no {attr}")
            return

        try:
            stl_bytes = _solid_to_stl_bytes(solid)
        except Exception as e:
            self.send_error(500, f"STL export failed for step {step_idx} {attr}: {e}")
            return

        self._send_binary(stl_bytes, f"{body_name}_step{step_idx}_{attr}.stl")

    def log_message(self, format, *args):
        # Quieter logging — skip successful static file requests
        if len(args) >= 2 and str(args[1]) == "200":
            return
        super().log_message(format, *args)


def _run_web_viewer(bot_name: str | None, port: int = 8080, no_open: bool = False):
    """Launch a local HTTP server and open the 3D viewer in the browser."""
    project_root = Path(__file__).parent

    # Generate viewer manifest for bot if specified
    if bot_name:
        bot_dir = project_root / "bots" / bot_name
        if not bot_dir.exists():
            print(f"Error: Bot directory not found for {bot_name} at {bot_dir}")
            sys.exit(1)

        design_py = bot_dir / "design.py"
        manifest_json = bot_dir / "viewer_manifest.json"
        needs_regen = design_py.exists() and (
            not manifest_json.exists()
            or design_py.stat().st_mtime > manifest_json.stat().st_mtime
        )
        if needs_regen:
            try:
                import importlib.util

                spec = importlib.util.spec_from_file_location("design", design_py)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                bot = mod.build()
                bot.solve()
                from botcad.emit.viewer import emit_viewer_manifest

                emit_viewer_manifest(bot, bot_dir)
                print(f"Generated viewer_manifest.json for {bot_name}")
            except Exception as e:
                print(f"Warning: could not generate viewer manifest: {e}")

    # Build component registry for API
    print("Building component registry...")
    ViewerHTTPHandler.component_registry = _build_component_registry()
    ViewerHTTPHandler.project_root = str(project_root)
    print(f"  {len(ViewerHTTPHandler.component_registry)} components registered")

    # Determine URL
    if bot_name:
        url = f"http://localhost:{port}/viewer/?bot={bot_name}"
    else:
        url = f"http://localhost:{port}/viewer/"

    print(f"Viewer at {url}")
    print("Press Ctrl+C to stop.")
    if not no_open:
        webbrowser.open(url)

    # ThreadingHTTPServer so STL generation doesn't block other requests
    server = http.server.ThreadingHTTPServer(("", port), ViewerHTTPHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


def _setup_logging() -> None:
    """Configure root logger level and install an excepthook.

    Per-run file logging is set up in train.py when the run directory is
    created.  This function only sets the root level and installs an
    excepthook so unhandled exceptions are captured by whatever handlers
    are active at the time.
    """
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Capture unhandled exceptions to the log
    _original_excepthook = sys.excepthook

    def _logging_excepthook(exc_type, exc_value, exc_tb):
        if not issubclass(exc_type, KeyboardInterrupt):
            logging.critical(
                "Unhandled exception", exc_info=(exc_type, exc_value, exc_tb)
            )
        _original_excepthook(exc_type, exc_value, exc_tb)

    sys.excepthook = _logging_excepthook


def _is_mjpython() -> bool:
    """Check if we're running under mjpython."""
    return "MJPYTHON_BIN" in os.environ


def _check_mjpython():
    """Warn if not launched via mjpython (needed for MuJoCo viewer on macOS)."""
    if shutil.which("mjpython") is None:
        return  # mjpython not installed, nothing to check
    if not _is_mjpython():
        print(
            "Warning: main.py should be launched with mjpython for viewer/play support.\n"
            "  Use: uv run mjpython main.py   (or: make)\n"
        )


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="MindSim - single entry point for all modes",
    )
    sub = parser.add_subparsers(dest="command")

    # web
    p_web = sub.add_parser("web", help="Launch 3D bot viewer in browser")
    p_web.add_argument(
        "--bot", type=str, default=None, help="Bot name (default: wheeler_arm)"
    )
    p_web.add_argument(
        "--port", type=int, default=8080, help="HTTP server port (default: 8080)"
    )
    p_web.add_argument(
        "--no-open", action="store_true", help="Don't open browser (API-only mode)"
    )

    # scene
    sub.add_parser("scene", help="Scene gen preview (procedural furniture)")

    # view
    p_view = sub.add_parser("view", help="Launch MuJoCo viewer")
    p_view.add_argument(
        "--bot", type=str, default=None, help="Bot name (default: simple2wheeler)"
    )
    p_view.add_argument(
        "--stage", type=int, default=None, help="Curriculum stage 1-4 (default: none)"
    )

    # play
    p_play = sub.add_parser("play", help="Play trained policy in viewer")
    p_play.add_argument(
        "checkpoint",
        nargs="?",
        default="latest",
        help="Checkpoint ref (default: latest)",
    )
    p_play.add_argument(
        "--bot", type=str, default=None, help="Bot name (default: simple2wheeler)"
    )
    p_play.add_argument(
        "--run",
        type=str,
        default=None,
        help="Run name to play (resolves checkpoint from run dir)",
    )

    # train
    p_train = sub.add_parser("train", help="Train (headless CLI, no TUI)")
    p_train.add_argument(
        "--smoketest", action="store_true", help="Fast end-to-end smoketest"
    )
    p_train.add_argument(
        "--bot", type=str, default=None, help="Bot name (default: simple2wheeler)"
    )
    p_train.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint"
    )
    p_train.add_argument(
        "--num-workers", type=int, default=None, help="Number of parallel workers"
    )
    p_train.add_argument(
        "--no-dirty-check",
        action="store_true",
        help="Skip uncommitted-changes check",
    )

    # smoketest (alias for train --smoketest)
    sub.add_parser("smoketest", help="Alias for train --smoketest")

    # quicksim
    sub.add_parser("quicksim", help="Quick simulation with Rerun debug vis")

    # visualize
    p_viz = sub.add_parser("visualize", help="One-shot Rerun visualization")
    p_viz.add_argument(
        "--bot", type=str, default=None, help="Bot name (default: simple2wheeler)"
    )
    p_viz.add_argument("--steps", type=int, default=1000, help="Number of sim steps")

    # validate-rewards
    p_vr = sub.add_parser(
        "validate-rewards", help="Validate reward hierarchy for a bot"
    )
    p_vr.add_argument(
        "--bot", type=str, default=None, help="Bot name (default: all bots)"
    )

    # describe
    p_desc = sub.add_parser("describe", help="Print human-readable pipeline summary")
    p_desc.add_argument(
        "--bot", type=str, default=None, help="Bot name (default: simple2wheeler)"
    )

    # replay
    p_replay = sub.add_parser(
        "replay",
        help="Download or regenerate Rerun recordings for a run",
    )
    p_replay.add_argument("run", help="Run name (local or W&B)")
    p_replay.add_argument(
        "--batches",
        default=None,
        help="Comma-separated batch numbers (e.g. 500,1000,2000)",
    )
    p_replay.add_argument(
        "--last",
        type=int,
        default=None,
        help="Download only the N most recent recordings",
    )
    p_replay.add_argument(
        "--regenerate",
        action="store_true",
        help="Re-run episodes from checkpoints instead of downloading recordings",
    )
    p_replay.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for target placement (only with --regenerate)",
    )

    return parser


def _run_tui():
    """Launch the interactive Textual TUI."""
    app = MindSimApp()
    app.run()

    # Dispatch to selected mode after TUI exits.
    # Use os.execvp to replace the process entirely, giving GLFW a clean
    # AppKit state (Textual's event loop corrupts it on macOS).
    # Must go through "uv run mjpython" — sys.executable is the venv python,
    # not mjpython, because mjpython exec's the real interpreter after setup.
    if app.next_action == "view":
        cmd = ["uv", "run", "mjpython", "main.py", "view"]
        if app.next_scene:
            bot_name = Path(app.next_scene).parent.name
            cmd.extend(["--bot", bot_name])
        if app.next_stage:
            cmd.extend(["--stage", str(app.next_stage)])
        os.execvp("uv", cmd)

    elif app.next_action == "scene":
        cmd = ["uv", "run", "mjpython", "main.py", "scene"]
        os.execvp("uv", cmd)

    elif app.next_action == "play":
        cmd = ["uv", "run", "mjpython", "main.py", "play"]
        # If we have a specific checkpoint path (legacy), pass it directly
        if app.next_checkpoint_path:
            cmd.append(app.next_checkpoint_path)
        # If we have a run name, use --run
        elif app.next_run_name:
            cmd.extend(["--run", app.next_run_name])
        if app.next_scene:
            bot_name = Path(app.next_scene).parent.name
            cmd.extend(["--bot", bot_name])
        os.execvp("uv", cmd)


def main():
    _setup_logging()
    _check_mjpython()

    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        # No subcommand → interactive TUI
        _run_tui()

    elif args.command == "web":
        _run_web_viewer(args.bot, args.port, getattr(args, "no_open", False))

    elif args.command == "scene":
        from sim.scene_preview import run_scene_preview

        run_scene_preview()

    elif args.command == "view":
        scene_path = _resolve_scene_path(args.bot)
        from tools.view import run_view

        run_view(scene_path, stage=args.stage)

    elif args.command == "play":
        scene_path = _resolve_scene_path(args.bot)
        from tools.play import run_play

        # --run takes priority: resolve checkpoint from run directory
        checkpoint_ref = args.run if args.run else args.checkpoint
        run_play(checkpoint_ref=checkpoint_ref, scene_path=scene_path)

    elif args.command == "train":
        scene_path = _resolve_scene_path(args.bot)
        # Dirty-tree check (skip for smoketests and --no-dirty-check)
        if not args.smoketest and not args.no_dirty_check:
            clean, status = _git_is_clean()
            if not clean:
                print("Uncommitted changes:")
                print(status)
                print()
                while True:
                    choice = (
                        input("[c] Commit with Claude  [s] Start anyway  [q] Quit: ")
                        .strip()
                        .lower()
                    )
                    if choice == "q":
                        sys.exit(0)
                    elif choice == "s":
                        break
                    elif choice == "c":
                        if _run_claude_commit():
                            print("Worktree is clean. Starting training.")
                        else:
                            _, new_status = _git_is_clean()
                            print("Still dirty after commit attempt:")
                            print(new_status)
                            print()
                            continue
                        break
        from training.train import main as train_main

        train_main(
            smoketest=args.smoketest,
            bot=args.bot,
            resume=args.resume,
            num_workers=args.num_workers,
            scene_path=scene_path,
        )

    elif args.command == "smoketest":
        from training.train import main as train_main

        bots = _discover_bots()
        if not bots:
            print("Error: No bots found in bots/*/scene.xml", file=sys.stderr)
            sys.exit(1)
        for i, bot_info in enumerate(bots):
            name = bot_info["name"]
            print(f"\n{'=' * 60}")
            print(f"Smoketest [{i + 1}/{len(bots)}]: {name}")
            print(f"{'=' * 60}")
            train_main(smoketest=True, bot=name, scene_path=bot_info["scene_path"])
        print(f"\nAll {len(bots)} bots passed smoketest.")

    elif args.command == "quicksim":
        from tools.quick_sim import run_quick_sim

        run_quick_sim()

    elif args.command == "visualize":
        scene_path = _resolve_scene_path(args.bot)
        from viz.visualize import run_visualization

        run_visualization(scene_path=scene_path, num_steps=args.steps)

    elif args.command == "validate-rewards":
        _validate_rewards(args.bot)

    elif args.command == "describe":
        from training.pipeline import pipeline_for_bot

        bot_name = args.bot or "simple2wheeler"
        pipeline = pipeline_for_bot(bot_name)
        print(pipeline.describe())

    elif args.command == "replay":
        from viz.replay import run_replay

        batches = None
        if args.batches:
            batches = [int(b.strip()) for b in args.batches.split(",")]
        run_replay(
            args.run,
            batches=batches,
            seed=args.seed,
            regenerate=args.regenerate,
            last_n=args.last,
        )


def _validate_rewards(bot_name: str | None):
    """Print reward hierarchy summary and run dominance checks."""
    from training.pipeline import pipeline_for_bot
    from training.rewards import build_reward_hierarchy

    # If no bot specified, validate all bots
    if bot_name is None:
        bots = _discover_bots()
        bot_names = [b["name"] for b in bots]
    else:
        bot_names = [bot_name]

    for name in bot_names:
        cfg = pipeline_for_bot(name)
        hierarchy = build_reward_hierarchy(name, cfg.env)

        print(f"\nReward Hierarchy for {name}")
        print("=" * 50)
        print(hierarchy.summary_table())
        print()

        dom = hierarchy.dominance_check()
        if dom:
            print("Dominance check:")
            print(dom)
        else:
            print("Dominance check: (single priority level, no check needed)")
        print()

        # Scenario tests
        _run_scenario_tests(name, hierarchy, cfg)


def _run_scenario_tests(bot_name: str, hierarchy, cfg):
    """Run scenario tests showing per-step reward in different situations."""

    active = hierarchy.active_components()
    if not active:
        print("Scenario tests: no active components")
        return

    print("Scenario tests:")

    # Scenario 1: "Perfect stand" -- healthy, no movement
    stand_reward = 0.0
    for c in active:
        if c.name == "alive":
            stand_reward += c.scale * 1.0  # healthy
        elif c.name == "upright":
            stand_reward += c.scale * 1.0  # perfectly upright
        elif c.name == "time":
            stand_reward += c.scale * (-1.0)  # time penalty always applies
        # Everything else is 0 (no movement, no contact, etc.)
    print(f"  Perfect stand  (healthy, no movement):    {stand_reward:+.3f}/step")

    # Scenario 2: "Diving forward" -- unhealthy, fast forward
    dive_reward = 0.0
    for c in active:
        if c.name == "alive":
            dive_reward += 0.0  # unhealthy -- no alive bonus
        elif c.name == "forward_velocity":
            dive_reward += c.scale * 1.0  # max forward vel, but gated by up_z
            # If gated by is_healthy, diving doesn't earn this either
            if c.gated_by and "is_healthy" in c.gated_by:
                dive_reward -= c.scale * 1.0  # undo: not healthy
        elif c.name == "distance":
            dive_reward += c.scale * 0.5  # some distance progress
        elif c.name == "time":
            dive_reward += c.scale * (-1.0)
        elif c.name == "contact":
            dive_reward += c.scale * (-1.0)  # body on floor
    print(f"  Diving forward (unhealthy, fast):         {dive_reward:+.3f}/step")

    # Scenario 3: "Walking well" -- healthy, moderate forward, upright
    walk_reward = 0.0
    for c in active:
        if c.name == "alive":
            walk_reward += c.scale * 1.0
        elif c.name == "forward_velocity":
            walk_reward += c.scale * 0.5  # moderate speed
        elif c.name == "distance":
            walk_reward += c.scale * 0.3  # some progress
        elif c.name == "upright":
            walk_reward += c.scale * 0.9  # mostly upright
        elif c.name == "energy":
            walk_reward += c.scale * (-3.0)  # moderate energy
        elif c.name == "smoothness":
            walk_reward += c.scale * (-1.0)  # some jerk
        elif c.name == "time":
            walk_reward += c.scale * (-1.0)
    print(f"  Walking well   (healthy, forward):        {walk_reward:+.3f}/step")

    # Check key invariant: standing must beat diving
    if stand_reward > dive_reward:
        print(f"\n  Standing ({stand_reward:+.3f}) > Diving ({dive_reward:+.3f}) [ok]")
    else:
        print(
            f"\n  Standing ({stand_reward:+.3f}) <= Diving ({dive_reward:+.3f}) "
            "[!!] -- diving is more rewarding than standing!"
        )

    if walk_reward > stand_reward:
        print(
            f"  Walking ({walk_reward:+.3f}) > Standing ({stand_reward:+.3f}) "
            "[ok] -- walking is the best outcome"
        )
    else:
        print(
            f"  Walking ({walk_reward:+.3f}) <= Standing ({stand_reward:+.3f}) "
            "[!!] -- standing is better than walking"
        )


if __name__ == "__main__":
    main()
