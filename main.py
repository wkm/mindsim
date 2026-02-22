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
import logging
import os
import re
import shutil
import threading
import sys
import time
from datetime import datetime
from pathlib import Path
from train import CommandChannel

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import (
    Footer,
    OptionList,
    RadioButton,
    RadioSet,
    RichLog,
    Static,
    TabbedContent,
    TabPane,
)

from rich.markdown import Markdown as RichMarkdown

from dashboard import _fmt_int, _fmt_pct, _fmt_time
from run_manager import (
    bot_display_name,
    discover_local_runs,
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

    def __init__(self, app: "MindSimApp"):
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


def _fmt_elapsed(seconds: float) -> str:
    """Format elapsed seconds as h:mm:ss or m:ss."""
    s = int(seconds)
    if s >= 3600:
        h, rem = divmod(s, 3600)
        m, sec = divmod(rem, 60)
        return f"{h}:{m:02d}:{sec:02d}"
    m, sec = divmod(s, 60)
    return f"{m}:{sec:02d}"


def _fmt(value, precision=3, width=8):
    """Format a float right-aligned."""
    if value is None:
        return " " * width
    return f"{value:+.{precision}f}".rjust(width)


def _color_val(formatted: str, value: float | None, ranges: dict) -> str:
    """Wrap a formatted string in Rich color markup based on health ranges.

    Args:
        formatted: Already-formatted display string.
        value: Raw numeric value (None → no coloring).
        ranges: Dict with 'green', 'yellow', 'red' keys.
            Each is a callable(value) -> bool, checked in order: green, yellow, red.
            First match wins; no match → no coloring.
    """
    if value is None:
        return formatted
    if ranges.get("green") and ranges["green"](value):
        return f"[green]{formatted}[/green]"
    if ranges.get("yellow") and ranges["yellow"](value):
        return f"[yellow]{formatted}[/yellow]"
    if ranges.get("red") and ranges["red"](value):
        return f"[red]{formatted}[/red]"
    return formatted


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
    import subprocess

    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
        )
        output = result.stdout.strip()
        return (output == "", output)
    except Exception:
        return (True, "")


def _run_claude_commit() -> bool:
    """Run Claude in print mode to commit all changes.

    Returns True if the worktree is clean afterward.
    """
    import subprocess

    print("Running Claude to commit changes...")
    try:
        subprocess.run(
            [
                "claude", "-p",
                "Run git status and git diff to see current changes, then commit all changes with a descriptive commit message.",
                "--allowedTools", "Bash Read Grep Glob",
            ],
            timeout=120,
        )
    except Exception as e:
        print(f"Claude commit failed: {e}")
        return False

    clean, _ = _git_is_clean()
    return clean


# ---------------------------------------------------------------------------
# Main Menu Screen
# ---------------------------------------------------------------------------


class MainMenuScreen(Screen):
    """Top-level menu: smoketest, new run, browse runs, quit."""

    BINDINGS = [
        Binding("s", "select('smoketest')", "Smoketest", priority=True),
        Binding("n", "select('new')", "New Run", priority=True),
        Binding("v", "select('view')", "View Bot", priority=True),
        Binding("b", "select('browse')", "Browse Runs", priority=True),
        Binding("q", "select('quit')", "Quit", priority=True),
        Binding("escape", "select('quit')", "Quit", show=False, priority=True),
        Binding("backspace", "select('quit')", "Quit", show=False, priority=True),
    ]

    CSS = """
    MainMenuScreen {
        align: center middle;
    }

    #menu-box {
        width: 50;
        height: auto;
        border: ascii $accent;
        padding: 1 2;
    }

    #menu-title {
        text-style: bold;
        text-align: center;
        margin-bottom: 1;
    }

    #menu-list {
        height: auto;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="menu-box"):
            yield Static("MindSim", id="menu-title")
            yield OptionList(
                "[s] Smoketest",
                "[n] New training run",
                "[v] View bot",
                "[b] Browse runs",
                "[q] Quit",
                id="menu-list",
            )
        yield Footer()

    def action_select(self, choice: str) -> None:
        if choice == "smoketest":
            self.app.start_training(smoketest=True)
        elif choice == "new":
            self.app.push_screen(BotSelectorScreen(mode="train"))
        elif choice == "view":
            self.app.push_screen(BotSelectorScreen(mode="view"))
        elif choice == "browse":
            self.app.push_screen(RunBrowserScreen())
        elif choice == "quit":
            self.app.exit()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        idx = event.option_index
        choices = ["smoketest", "new", "view", "browse", "quit"]
        if 0 <= idx < len(choices):
            self.action_select(choices[idx])


# ---------------------------------------------------------------------------
# Bot Selector Screen
# ---------------------------------------------------------------------------


class BotSelectorScreen(Screen):
    """Select a bot for training or viewing."""

    BINDINGS = [
        Binding("escape", "go_back", "Back", priority=True),
        Binding("backspace", "go_back", "Back", show=False, priority=True),
        Binding("enter", "confirm", "Select", priority=True),
    ]

    CSS = """
    BotSelectorScreen {
        align: center middle;
    }

    #bot-box {
        width: 50;
        height: auto;
        border: ascii $accent;
        padding: 1 2;
    }

    #bot-title {
        text-style: bold;
        margin-bottom: 1;
    }

    #bot-selector {
        height: auto;
    }
    """

    def __init__(self, mode: str = "train", **kwargs):
        super().__init__(**kwargs)
        self._bots = _discover_bots()
        self._mode = mode

    def compose(self) -> ComposeResult:
        title = "Select Bot to View" if self._mode == "view" else "Select Bot"
        with Vertical(id="bot-box"):
            yield Static(title, id="bot-title")
            if self._bots:
                with RadioSet(id="bot-selector"):
                    for i, bot in enumerate(self._bots):
                        display = bot_display_name(bot["name"])
                        yield RadioButton(f"{display} ({bot['name']})", value=(i == 0))
            else:
                yield Static("  No bots found in bots/*/scene.xml")
        yield Footer()

    def action_go_back(self) -> None:
        self.app.pop_screen()

    def _get_selected_scene(self) -> str | None:
        if not self._bots:
            return None
        try:
            radio_set = self.query_one("#bot-selector", RadioSet)
            idx = radio_set.pressed_index
            if idx >= 0:
                return self._bots[idx]["scene_path"]
        except (IndexError, ValueError):
            pass
        return self._bots[0]["scene_path"]

    def action_confirm(self) -> None:
        scene_path = self._get_selected_scene()
        if not scene_path:
            return
        if self._mode == "view":
            self.app.start_viewing(scene_path=scene_path)
        else:
            self.app.start_training(smoketest=False, scene_path=scene_path)


# ---------------------------------------------------------------------------
# Dirty Tree Screen
# ---------------------------------------------------------------------------


class DirtyTreeScreen(Screen):
    """Shown before training when the git worktree has uncommitted changes."""

    BINDINGS = [
        Binding("c", "commit", "Commit with Claude", priority=True),
        Binding("s", "start_anyway", "Start Anyway", priority=True),
        Binding("escape", "go_back", "Back", priority=True),
        Binding("backspace", "go_back", "Back", show=False, priority=True),
    ]

    CSS = """
    DirtyTreeScreen {
        align: center middle;
    }

    #dirty-box {
        width: 60;
        height: auto;
        border: ascii $accent;
        padding: 1 2;
    }

    #dirty-title {
        text-style: bold;
        margin-bottom: 1;
    }

    #dirty-status {
        margin-bottom: 1;
    }

    #dirty-actions {
        height: auto;
    }
    """

    def __init__(
        self,
        status_lines: str,
        smoketest: bool,
        scene_path: str | None,
        resume: str | None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._status_lines = status_lines
        self._smoketest = smoketest
        self._scene_path = scene_path
        self._resume = resume

    def compose(self) -> ComposeResult:
        with Vertical(id="dirty-box"):
            yield Static("Uncommitted Changes", id="dirty-title")
            yield Static(self._status_lines, id="dirty-status")
            yield OptionList(
                "[c] Commit with Claude",
                "[s] Start anyway",
                "[Esc] Back",
                id="dirty-actions",
            )
        yield Footer()

    def action_go_back(self) -> None:
        self.app.pop_screen()

    def action_start_anyway(self) -> None:
        self.app.pop_screen()
        self.app._do_start_training(
            smoketest=self._smoketest,
            scene_path=self._scene_path,
            resume=self._resume,
        )

    def action_commit(self) -> None:
        self.query_one("#dirty-title", Static).update("Committing...")
        self._do_commit()

    @work(thread=True)
    def _do_commit(self) -> None:
        clean = _run_claude_commit()
        if clean:
            self.app.call_from_thread(self._start_after_commit)
        else:
            _, new_status = _git_is_clean()
            self.app.call_from_thread(self._update_after_commit, new_status)

    def _start_after_commit(self) -> None:
        self.app.pop_screen()
        self.app._do_start_training(
            smoketest=self._smoketest,
            scene_path=self._scene_path,
            resume=self._resume,
        )

    def _update_after_commit(self, new_status: str) -> None:
        self.query_one("#dirty-title", Static).update("Still Uncommitted Changes")
        self.query_one("#dirty-status", Static).update(new_status)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        idx = event.option_index
        actions = ["commit", "start_anyway", "go_back"]
        if 0 <= idx < len(actions):
            getattr(self, f"action_{actions[idx]}")()


# ---------------------------------------------------------------------------
# Run Browser Screen
# ---------------------------------------------------------------------------


class RunBrowserScreen(Screen):
    """Browse local runs."""

    BINDINGS = [
        Binding("escape", "go_back", "Back", priority=True),
        Binding("backspace", "go_back", "Back", show=False, priority=True),
        Binding("enter", "select_run", "Select", priority=True),
        Binding("w", "open_wandb", "W&B", priority=True),
    ]

    CSS = """
    RunBrowserScreen {
        align: center middle;
    }

    #browser-box {
        width: 80;
        height: auto;
        max-height: 30;
        border: ascii $accent;
        padding: 1 2;
    }

    #browser-title {
        text-style: bold;
        margin-bottom: 1;
    }

    #run-list {
        height: auto;
        max-height: 24;
    }

    #no-runs {
        color: $text-muted;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._runs = discover_local_runs()
        self._items: list[dict] = []  # maps list index -> run info

    def compose(self) -> ComposeResult:
        with Vertical(id="browser-box"):
            yield Static("Browse Runs", id="browser-title")
            items = []
            for run_dir, info in self._runs:
                date_str = info.created_at[:10] if info.created_at else "?"
                display_name = bot_display_name(info.bot_name)
                status_icon = {"running": "*", "completed": "+", "failed": "!"}.get(
                    info.status, "?"
                )
                batch_str = f"b{info.batch_idx}" if info.batch_idx else ""
                label = f"[{status_icon}] {info.name}  {display_name}  {date_str}  {batch_str}"
                items.append(label)
                self._items.append({"type": "run", "dir": run_dir, "info": info})
            if items:
                yield OptionList(*items, id="run-list")
            else:
                yield Static("  No runs found.", id="no-runs")
        yield Footer()

    def action_go_back(self) -> None:
        self.app.pop_screen()

    def action_select_run(self) -> None:
        try:
            run_list = self.query_one("#run-list", OptionList)
            idx = run_list.highlighted
            if idx is not None and 0 <= idx < len(self._items):
                item = self._items[idx]
                if item["type"] == "run":
                    self.app.push_screen(
                        RunActionScreen(run_dir=item["dir"], run_info=item["info"])
                    )
        except (IndexError, ValueError):
            pass

    def action_open_wandb(self) -> None:
        try:
            run_list = self.query_one("#run-list", OptionList)
            idx = run_list.highlighted
            if idx is not None and 0 <= idx < len(self._items):
                url = self._items[idx]["info"].wandb_url
                if url:
                    import webbrowser
                    webbrowser.open(url)
        except (IndexError, ValueError):
            pass

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.action_select_run()


# ---------------------------------------------------------------------------
# Run Action Screen
# ---------------------------------------------------------------------------


class RunActionScreen(Screen):
    """Actions for a selected run: play, resume, view, W&B, back."""

    BINDINGS = [
        Binding("p", "play_run", "Play", priority=True),
        Binding("r", "resume_run", "Resume", priority=True),
        Binding("v", "view_run", "View", priority=True),
        Binding("w", "open_wandb", "W&B", priority=True),
        Binding("escape", "go_back", "Back", priority=True),
        Binding("backspace", "go_back", "Back", show=False, priority=True),
    ]

    CSS = """
    RunActionScreen {
        align: center middle;
    }

    #action-box {
        width: 60;
        height: auto;
        border: ascii $accent;
        padding: 1 2;
    }

    #action-title {
        text-style: bold;
        margin-bottom: 1;
    }

    #run-metadata {
        color: $text-muted;
        margin-bottom: 1;
    }

    #action-list {
        height: auto;
    }
    """

    def __init__(self, run_dir: Path, run_info, **kwargs):
        super().__init__(**kwargs)
        self._run_dir = run_dir
        self._info = run_info

    def compose(self) -> ComposeResult:
        info = self._info
        display_name = bot_display_name(info.bot_name)
        with Vertical(id="action-box"):
            yield Static(f"Run: {info.name}", id="action-title")
            meta_lines = [
                f"  Bot: {display_name} ({info.bot_name})",
                f"  Algorithm: {info.algorithm}  |  Policy: {info.policy_type}",
                f"  Status: {info.status}  |  Batches: {info.batch_idx}  |  Episodes: {info.episode_count}",
                f"  Stage: {info.curriculum_stage}  |  Created: {info.created_at or '?'}",
            ]
            if info.wandb_url:
                meta_lines.append(f"  W&B: {info.wandb_url}")
            yield Static("\n".join(meta_lines), id="run-metadata")
            options = [
                "[p] Play checkpoint",
                "[r] Resume training",
                "[v] View in MuJoCo",
            ]
            self._action_map = ["play_run", "resume_run", "view_run"]
            if info.wandb_url:
                options.append("[w] Open W&B")
                self._action_map.append("open_wandb")
            options.append("[Esc] Back")
            self._action_map.append("go_back")
            yield OptionList(*options, id="action-list")
        yield Footer()

    def action_go_back(self) -> None:
        self.app.pop_screen()

    def action_play_run(self) -> None:
        self.app.push_screen(CheckpointPickerScreen(
            run_dir=self._run_dir, run_info=self._info, mode="play",
        ))

    def action_resume_run(self) -> None:
        self.app.push_screen(CheckpointPickerScreen(
            run_dir=self._run_dir, run_info=self._info, mode="resume",
        ))

    def action_view_run(self) -> None:
        self.app.start_viewing(scene_path=self._info.scene_path)

    def action_open_wandb(self) -> None:
        url = self._info.wandb_url
        if url:
            import webbrowser
            webbrowser.open(url)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        idx = event.option_index
        if 0 <= idx < len(self._action_map):
            getattr(self, f"action_{self._action_map[idx]}")()


# ---------------------------------------------------------------------------
# Checkpoint Picker Screen
# ---------------------------------------------------------------------------


class CheckpointPickerScreen(Screen):
    """Pick a checkpoint from a run before playing or resuming."""

    BINDINGS = [
        Binding("escape", "go_back", "Back", priority=True),
        Binding("backspace", "go_back", "Back", show=False, priority=True),
        Binding("enter", "confirm", "Select", priority=True),
    ]

    CSS = """
    CheckpointPickerScreen {
        align: center middle;
    }

    #picker-box {
        width: 70;
        height: auto;
        max-height: 30;
        border: ascii $accent;
        padding: 1 2;
    }

    #picker-title {
        text-style: bold;
        margin-bottom: 1;
    }

    #checkpoint-list {
        height: auto;
        max-height: 22;
    }

    #no-checkpoints {
        color: $text-muted;
    }
    """

    def __init__(self, run_dir: Path, run_info, mode: str, **kwargs):
        """
        Args:
            run_dir: Path to the run directory.
            run_info: RunInfo for the selected run.
            mode: "play" or "resume".
        """
        super().__init__(**kwargs)
        self._run_dir = run_dir
        self._info = run_info
        self._mode = mode
        from checkpoint import list_checkpoints
        self._checkpoints = list_checkpoints(run_dir)

    def compose(self) -> ComposeResult:
        action_label = "Play" if self._mode == "play" else "Resume"
        with Vertical(id="picker-box"):
            yield Static(f"{action_label}: {self._info.name}", id="picker-title")
            if self._checkpoints:
                items = []
                for i, ckpt in enumerate(self._checkpoints):
                    stage = f"Stage {ckpt['stage']}" if ckpt["stage"] is not None else "?"
                    batch = f"Batch {ckpt['batch']}" if ckpt["batch"] is not None else "?"
                    tag = "  (latest)" if i == 0 else ""
                    items.append(f"{stage}  {batch}{tag}")
                yield OptionList(*items, id="checkpoint-list")
            else:
                yield Static("  No checkpoints found.", id="no-checkpoints")
        yield Footer()

    def action_go_back(self) -> None:
        self.app.pop_screen()

    def action_confirm(self) -> None:
        if not self._checkpoints:
            self.app.pop_screen()
            return

        try:
            ol = self.query_one("#checkpoint-list", OptionList)
            idx = ol.highlighted
            if idx is None:
                idx = 0
        except Exception:
            idx = 0

        ckpt_path = self._checkpoints[idx]["path"]

        if self._mode == "play":
            self.app.start_playing_run(
                run_name=self._info.name,
                scene_path=self._info.scene_path,
                checkpoint_path=ckpt_path,
            )
        else:
            self.app.start_training(
                smoketest=False,
                scene_path=self._info.scene_path,
                resume=ckpt_path,
            )

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.action_confirm()


# ---------------------------------------------------------------------------
# Training Dashboard Screen
# ---------------------------------------------------------------------------


class TrainingDashboard(Screen):
    """Fullscreen training dashboard with metrics and controls."""

    BINDINGS = [
        Binding("space", "toggle_pause", "Pause/Resume"),
        Binding("n", "step_batch", "Step 1 Batch"),
        Binding("c", "checkpoint", "Checkpoint"),
        Binding("r", "send_rerun", "Send to Rerun"),
        Binding("w", "open_wandb", "W&B"),
        Binding("up", "advance_curriculum", "Advance"),
        Binding("down", "regress_curriculum", "Regress"),
        Binding("a", "ai_commentary", "AI"),
        Binding("left_square_bracket", "rerun_freq_down", "Rec \u2193"),
        Binding("right_square_bracket", "rerun_freq_up", "Rec \u2191"),
        Binding("q", "quit_app", "Quit"),
        Binding("escape", "quit_app", "Quit", show=False),
        Binding("backspace", "quit_app", "Quit", show=False),
    ]

    CSS = """
    TrainingDashboard {
        layout: vertical;
        overflow: hidden;
    }

    #header-bar {
        height: 1;
        background: $primary-background;
        color: $text;
        padding: 0 1;
    }

    #experiment-bar {
        height: auto;
        max-height: 2;
        padding: 0 1;
        color: $text-muted;
    }

    #progress-row {
        height: 1;
        padding: 0 1;
        margin-bottom: 1;
    }

    #progress-label {
        width: 100%;
        height: 1;
    }

    #body-content {
        height: 1fr;
    }

    #metrics-grid {
        height: auto;
        max-width: 90;
        padding: 0 1;
    }

    .metrics-col {
        width: 1fr;
        padding: 0 1;
    }

    .section-title {
        text-style: bold;
        color: $accent;
        margin-top: 1;
    }

    .metric-line {
        height: 1;
    }

    #log-panel {
        width: 1fr;
        height: 100%;
        border-left: solid $surface;
        padding: 0 1;
    }

    #log-tabs {
        height: 1fr;
    }

    #log-area {
        height: 1fr;
    }

    #ai-area {
        height: 1fr;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._total_batches = None
        self._algorithm = "PPO"
        self._start_time = time.monotonic()
        self._header_parts: list[str] = ["MindSim"]
        self._paused = False
        self._wandb_url: str | None = None

    def compose(self) -> ComposeResult:
        yield Static("MindSim", id="header-bar")
        yield Static("", id="experiment-bar")
        with Horizontal(id="progress-row"):
            yield Static("  batch 0", id="progress-label")
        with Horizontal(id="body-content"):
          with Horizontal(id="metrics-grid"):
            with Vertical(classes="metrics-col"):
                yield Static("EPISODE PERFORMANCE", classes="section-title")
                yield Static(
                    "  avg reward          ---",
                    id="m-avg-reward",
                    classes="metric-line",
                )
                yield Static(
                    "  best reward         ---",
                    id="m-best-reward",
                    classes="metric-line",
                )
                yield Static(
                    "  worst reward        ---",
                    id="m-worst-reward",
                    classes="metric-line",
                )
                yield Static(
                    "  avg distance        ---",
                    id="m-avg-distance",
                    classes="metric-line",
                )
                yield Static(
                    "  avg steps           ---", id="m-avg-steps", classes="metric-line"
                )
                yield Static("REWARD INPUTS", classes="section-title", id="ri-title")
                yield Static("  distance            ---", id="ri-distance", classes="metric-line")
                yield Static("  height              ---", id="ri-height", classes="metric-line")
                yield Static("  survival            ---", id="ri-survival", classes="metric-line")
                yield Static("  fwd distance        ---", id="ri-fwd-dist", classes="metric-line")
                yield Static("  speed               ---", id="ri-speed", classes="metric-line")
                yield Static("  lateral drift       ---", id="ri-lateral", classes="metric-line")
                yield Static("  uprightness         ---", id="ri-uprightness", classes="metric-line")
                yield Static("  fwd velocity        ---", id="ri-fwd-vel", classes="metric-line")
                yield Static("  energy              ---", id="ri-energy", classes="metric-line")
                yield Static("  contact             ---", id="ri-contact", classes="metric-line")
                yield Static("  action jerk         ---", id="ri-jerk", classes="metric-line")
                yield Static("  fell                ---", id="ri-fell", classes="metric-line")
                yield Static("  joint activity      ---", id="ri-joint", classes="metric-line")
                yield Static("SUCCESS RATES", classes="section-title")
                yield Static(
                    "  eval (rolling)      ---",
                    id="m-eval-rolling",
                    classes="metric-line",
                )
                yield Static(
                    "  eval (batch)        ---",
                    id="m-eval-batch",
                    classes="metric-line",
                )
                yield Static(
                    "  train (batch)       ---",
                    id="m-train-batch",
                    classes="metric-line",
                )
                yield Static(
                    "  policy std          ---",
                    id="m-policy-std",
                    classes="metric-line",
                )
            with Vertical(classes="metrics-col"):
                yield Static("OPTIMIZATION", classes="section-title")
                yield Static(
                    "  policy loss         ---",
                    id="m-policy-loss",
                    classes="metric-line",
                )
                yield Static(
                    "  value loss          ---",
                    id="m-value-loss",
                    classes="metric-line",
                )
                yield Static(
                    "  grad norm           ---", id="m-grad-norm", classes="metric-line"
                )
                yield Static(
                    "  entropy             ---", id="m-entropy", classes="metric-line"
                )
                yield Static(
                    "  clip fraction       ---",
                    id="m-clip-fraction",
                    classes="metric-line",
                )
                yield Static(
                    "  approx KL           ---", id="m-approx-kl", classes="metric-line"
                )
                yield Static(
                    "  explained var       ---", id="m-explained-var", classes="metric-line"
                )
                yield Static("CURRICULUM", classes="section-title")
                yield Static(
                    "  stage               ---", id="m-stage", classes="metric-line"
                )
                yield Static(
                    "  progress            ---", id="m-progress", classes="metric-line"
                )
                yield Static(
                    "  mastery             ---", id="m-mastery", classes="metric-line"
                )
                yield Static(
                    "  max steps           ---", id="m-max-steps", classes="metric-line"
                )
                yield Static(
                    "  rec interval        ---", id="m-rec-interval", classes="metric-line"
                )
                yield Static("TIMING", classes="section-title")
                yield Static("  batch               ---", id="m-timing-batch", classes="metric-line")
                yield Static("  \u251c collect           ---", id="m-timing-collect", classes="metric-line")
                yield Static("  \u251c train             ---", id="m-timing-train", classes="metric-line")
                yield Static("  \u251c eval              ---", id="m-timing-eval", classes="metric-line")
                yield Static("  \u2514 throughput         ---", id="m-timing-throughput", classes="metric-line")
          with Vertical(id="log-panel"):
              with TabbedContent(id="log-tabs"):
                  with TabPane("Log", id="tab-log"):
                      yield RichLog(id="log-area", wrap=True, max_lines=1000, markup=True)
                  with TabPane("AI", id="tab-ai"):
                      yield RichLog(id="ai-area", wrap=True, max_lines=200, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        self.log_message("Dashboard ready. Waiting for training to start...")
        self.set_interval(1.0, self._tick_elapsed)

    def action_toggle_pause(self) -> None:
        self._paused = not self._paused
        if self._paused:
            self.app.send_command("pause")
            self.log_message("[bold yellow]Paused[/bold yellow]")
            log.info("Paused")
        else:
            self.app.send_command("unpause")
            self.log_message("[bold green]Resumed[/bold green]")
            log.info("Resumed")

    def action_step_batch(self) -> None:
        self.app.send_command("step")
        self.log_message("Stepping one batch...")
        log.info("Stepping one batch")

    def action_checkpoint(self) -> None:
        self.app.send_command("checkpoint")
        self.log_message("[bold cyan]Checkpoint queued[/bold cyan] (saves after current batch)")

    def action_send_rerun(self) -> None:
        self.app.send_command("log_rerun")
        self.log_message("[bold cyan]Rerun recording queued[/bold cyan] (records next eval episode)")

    def action_advance_curriculum(self) -> None:
        self.app.send_command("advance_curriculum")
        self.log_message("Advancing curriculum...")

    def action_regress_curriculum(self) -> None:
        self.app.send_command("regress_curriculum")
        self.log_message("Regressing curriculum...")

    def action_rerun_freq_down(self) -> None:
        self.app.send_command("rerun_freq_down")
        self.log_message("Decreasing Rerun recording interval (more frequent)...")

    def action_rerun_freq_up(self) -> None:
        self.app.send_command("rerun_freq_up")
        self.log_message("Increasing Rerun recording interval (less frequent)...")

    def action_ai_commentary(self) -> None:
        self.app.send_command("ai_commentary")
        self.log_message("Generating AI commentary...")

    def action_open_wandb(self) -> None:
        if self._wandb_url:
            import webbrowser
            webbrowser.open(self._wandb_url)

    def action_quit_app(self) -> None:
        self.app.send_command("stop")
        self.app.exit()

    def _tick_elapsed(self) -> None:
        """Update elapsed time in the header bar every second."""
        elapsed = _fmt_elapsed(time.monotonic() - self._start_time)
        pause_tag = "  [bold yellow]PAUSED[/bold yellow]" if self._paused else ""
        header = " | ".join(self._header_parts) + f"  [{elapsed}]{pause_tag}"
        self.query_one("#header-bar", Static).update(header)

    def set_header(
        self, run_name: str, branch: str, algorithm: str, wandb_url: str | None,
        bot_name: str | None = None, experiment_hypothesis: str | None = None,
    ):
        self._algorithm = algorithm
        self._wandb_url = wandb_url
        self._start_time = time.monotonic()
        title = f"MindSim {bot_name}" if bot_name else "MindSim"
        parts = [f"{title} | {run_name}", branch, algorithm]
        if wandb_url:
            parts.append(wandb_url)
        self._header_parts = parts
        # Render immediately (timer will keep updating)
        self._tick_elapsed()
        # Show experiment hypothesis if available
        bar = self.query_one("#experiment-bar", Static)
        if experiment_hypothesis:
            bar.update(f"Experiment: {experiment_hypothesis}")
        else:
            bar.update("")

    def update_metrics(self, batch: int, metrics: dict):
        m = metrics
        is_ppo = self._algorithm == "PPO"
        total = self._total_batches

        # Progress bar (text-based)
        ep_count = m.get("episode_count")
        ep_str = f" | {ep_count:,} episodes" if ep_count else ""
        if total and total > 0:
            frac = min(batch / total, 1.0)
            pct = 100 * frac
            bar_w = 30
            filled = int(bar_w * frac)
            bar = "\u2588" * filled + "\u2591" * (bar_w - filled)
            self.query_one("#progress-label", Static).update(
                f"  {bar}  {pct:5.1f}%   batch {batch:,} / {total:,}{ep_str}"
            )
        else:
            self.query_one("#progress-label", Static).update(
                f"  batch {batch:,}{ep_str}"
            )

        # Episode performance
        self.query_one("#m-avg-reward").update(
            f"  avg reward       {_fmt(m.get('avg_reward'), precision=2)}"
        )
        self.query_one("#m-best-reward").update(
            f"  best reward      {_fmt(m.get('best_reward'), precision=2)}"
        )
        self.query_one("#m-worst-reward").update(
            f"  worst reward     {_fmt(m.get('worst_reward'), precision=2)}"
        )
        dist = m.get("avg_distance")
        dist_str = f"{_fmt(dist, precision=2)} m" if dist is not None else "---"
        self.query_one("#m-avg-distance").update(f"  avg distance     {dist_str}")
        self.query_one("#m-avg-steps").update(
            f"  avg steps        {_fmt_int(m.get('avg_steps'))}"
        )

        # Reward inputs (raw physical measures)
        ri = m.get("raw_inputs", {})
        scales = m.get("reward_scales", {})

        def _ri_row(widget_id, label, value, fmt_str, scale_key=None):
            """Update a reward-input row. Hidden when scale_key is set and its scale is 0."""
            show = scale_key is None or scales.get(scale_key, 0) > 0
            widget = self.query_one(widget_id)
            if show and value is not None:
                widget.update(f"  {label:<15s}{fmt_str.rjust(8)}")
                widget.display = True
            elif show:
                widget.update(f"  {label:<15s}{'---':>8s}")
                widget.display = True
            else:
                widget.display = False

        # Always shown
        ri_dist = ri.get("distance_to_target")
        _ri_row("#ri-distance", "distance", ri_dist, f"{ri_dist:.2f} m" if ri_dist is not None else None)
        ri_h = ri.get("torso_height")
        _ri_row("#ri-height", "height", ri_h, f"{ri_h:.2f} m" if ri_h is not None else None)
        ri_surv = ri.get("survival_time")
        _ri_row("#ri-survival", "survival", ri_surv, f"{ri_surv:.1f} s" if ri_surv is not None else None)

        # Walking stage measures (biped only)
        ri_fd = ri.get("forward_distance")
        _ri_row("#ri-fwd-dist", "fwd distance", ri_fd, f"{ri_fd:+.2f} m" if ri_fd is not None else None, "has_walking_stage")
        ri_spd = ri.get("avg_speed")
        _ri_row("#ri-speed", "speed", ri_spd, f"{ri_spd:.2f} m/s" if ri_spd is not None else None, "has_walking_stage")
        ri_lat = ri.get("lateral_drift")
        _ri_row("#ri-lateral", "lateral drift", ri_lat, f"{ri_lat:.2f} m" if ri_lat is not None else None, "has_walking_stage")

        # Reward-component-gated
        ri_up = ri.get("up_z")
        _ri_row("#ri-uprightness", "uprightness", ri_up, f"{ri_up:.2f}" if ri_up is not None else None, "upright")
        ri_fv = ri.get("forward_vel")
        _ri_row("#ri-fwd-vel", "fwd velocity", ri_fv, f"{ri_fv:+.2f} m/s" if ri_fv is not None else None, "forward_vel")
        ri_en = ri.get("energy")
        _ri_row("#ri-energy", "energy", ri_en, f"{ri_en:.3f}" if ri_en is not None else None, "energy")
        ri_ct = ri.get("contact_frac")
        _ri_row("#ri-contact", "contact", ri_ct, f"{ri_ct:.1%}" if ri_ct is not None else None, "contact")
        ri_jk = ri.get("action_jerk")
        _ri_row("#ri-jerk", "action jerk", ri_jk, f"{ri_jk:.3f}" if ri_jk is not None else None, "smoothness")

        # Fall detection gated
        ri_fell = ri.get("fell_frac")
        _ri_row("#ri-fell", "fell", ri_fell, f"{ri_fell:.0%}" if ri_fell is not None else None, "fall_detection")

        # Joint activity (shown when sensors exist, i.e. value > 0 or biped)
        ri_ja = ri.get("joint_activity")
        has_sensors = ri_ja is not None and ri_ja > 0
        widget = self.query_one("#ri-joint")
        if has_sensors:
            widget.update(f"  {'joint activity':<15s}{f'{ri_ja:.1f} rad/s'.rjust(8)}")
            widget.display = True
        elif ri_ja is not None and any(scales.get(k, 0) > 0 for k in ("upright", "energy", "contact")):
            # Biped with no joint movement yet
            widget.update(f"  {'joint activity':<15s}{f'{ri_ja:.1f} rad/s'.rjust(8)}")
            widget.display = True
        else:
            widget.display = False

        # Success rates
        self.query_one("#m-eval-rolling").update(
            f"  eval (rolling)   {_fmt_pct(m.get('rolling_eval_success_rate'))}"
        )
        self.query_one("#m-eval-batch").update(
            f"  eval (batch)     {_fmt_pct(m.get('eval_success_rate'))}"
        )
        self.query_one("#m-train-batch").update(
            f"  train (batch)    {_fmt_pct(m.get('batch_success_rate'))}"
        )

        ps = m.get("policy_std")
        if ps is not None:
            try:
                std_str = f"{ps[0]:.2f} / {ps[1]:.2f}"
            except (IndexError, TypeError):
                std_str = f"{ps:.2f}"
        else:
            std_str = "---"
        self.query_one("#m-policy-std").update(f"  policy std       {std_str.rjust(8)}")

        # Optimization (with health colorization for PPO diagnostics)
        if is_ppo:
            self.query_one("#m-policy-loss").update(
                f"  policy loss      {_fmt(m.get('policy_loss'), precision=4)}"
            )
            self.query_one("#m-value-loss").update(
                f"  value loss       {_fmt(m.get('value_loss'), precision=4)}"
            )
            # Clip fraction: green 0.05-0.30, yellow <0.05 or 0.30-0.50, red >0.50
            cf_val = m.get('clip_fraction')
            cf_str = _fmt(cf_val, precision=2)
            cf_str = _color_val(cf_str, cf_val, {
                "green": lambda v: 0.05 <= v <= 0.30,
                "yellow": lambda v: v < 0.05 or 0.30 < v <= 0.50,
                "red": lambda v: v > 0.50,
            })
            self.query_one("#m-clip-fraction").update(f"  clip fraction    {cf_str}")
            # Approx KL: green 0.005-0.05, yellow <0.005 or 0.05-0.1, red >0.1
            kl_val = m.get('approx_kl')
            kl_str = _fmt(kl_val, precision=4)
            kl_str = _color_val(kl_str, kl_val, {
                "green": lambda v: 0.005 <= v <= 0.05,
                "yellow": lambda v: v < 0.005 or 0.05 < v <= 0.1,
                "red": lambda v: v > 0.1,
            })
            self.query_one("#m-approx-kl").update(f"  approx KL        {kl_str}")
            # Explained variance: green >0.5, yellow 0.0-0.5, red <0.0
            ev_val = m.get('explained_variance')
            ev_str = _fmt(ev_val, precision=3)
            ev_str = _color_val(ev_str, ev_val, {
                "green": lambda v: v > 0.5,
                "yellow": lambda v: 0.0 <= v <= 0.5,
                "red": lambda v: v < 0.0,
            })
            self.query_one("#m-explained-var").update(f"  explained var    {ev_str}")
        else:
            self.query_one("#m-policy-loss").update(
                f"  loss             {_fmt(m.get('loss'), precision=4)}"
            )
            self.query_one("#m-value-loss").update("  value loss       ---")
            self.query_one("#m-clip-fraction").update("  clip fraction    ---")
            self.query_one("#m-approx-kl").update("  approx KL        ---")
            self.query_one("#m-explained-var").update("  explained var    ---")
        # Grad norm: colored relative to max_grad_norm config
        gn_val = m.get('grad_norm')
        max_gn = m.get('max_grad_norm', 0.5)
        gn_str = _fmt(gn_val, precision=3)
        if gn_val is not None and max_gn > 0:
            ratio = gn_val / max_gn
            if ratio > 10:
                gn_str = f"[red]{gn_str}[/red] [dim]({ratio:.0f}x)[/dim]"
            elif ratio > 2:
                gn_str = f"[yellow]{gn_str}[/yellow] [dim]({ratio:.0f}x)[/dim]"
            else:
                gn_str = f"[green]{gn_str}[/green]"
        self.query_one("#m-grad-norm").update(f"  grad norm        {gn_str}")
        self.query_one("#m-entropy").update(
            f"  entropy          {_fmt(m.get('entropy'), precision=3)}"
        )

        # Curriculum
        stage = m.get("curriculum_stage")
        num_stages = m.get("num_stages", 3)
        stage_str = f"{int(stage)} / {int(num_stages)}" if stage is not None else "---"
        self.query_one("#m-stage").update(f"  stage            {stage_str.rjust(8)}")
        self.query_one("#m-progress").update(
            f"  progress         {_fmt(m.get('stage_progress'), precision=2)}"
        )
        mc = m.get("mastery_count", 0)
        mb = m.get("mastery_batches", 20)
        self.query_one("#m-mastery").update(
            f"  mastery          {f'{int(mc):>3d} / {int(mb)}'.rjust(8)}"
        )
        self.query_one("#m-max-steps").update(
            f"  max steps        {_fmt_int(m.get('max_episode_steps'))}"
        )
        rec = m.get("log_rerun_every")
        rec_str = f"{int(rec):,} ep" if rec is not None else "---"
        self.query_one("#m-rec-interval").update(
            f"  rec interval     {rec_str.rjust(8)}"
        )

        # Timing
        bt = m.get("batch_time")
        ct = m.get("collect_time")
        tt = m.get("train_time")
        et = m.get("eval_time")
        bs = m.get("batch_size")
        self.query_one("#m-timing-batch").update(
            f"  batch            {_fmt_time(bt)}" if bt is not None else "  batch               ---"
        )
        self.query_one("#m-timing-collect").update(
            f"  \u251c collect        {_fmt_time(ct)}" if ct is not None else "  \u251c collect           ---"
        )
        self.query_one("#m-timing-train").update(
            f"  \u251c train          {_fmt_time(tt)}" if tt is not None else "  \u251c train             ---"
        )
        self.query_one("#m-timing-eval").update(
            f"  \u251c eval           {_fmt_time(et)}" if et is not None else "  \u251c eval              ---"
        )
        throughput_str = "---"
        if bt and bt > 0 and bs:
            throughput_str = f"{bs / bt:.1f} ep/s"
        self.query_one("#m-timing-throughput").update(
            f"  \u2514 throughput      {throughput_str.rjust(8)}"
        )

    def log_message(self, text: str):
        ts = datetime.now().strftime("%H:%M:%S")
        log_widget = self.query_one("#log-area", RichLog)
        log_widget.write(f"[dim]{ts}[/dim]  {text}")

    def log_ai_commentary(self, text: str):
        ts = datetime.now().strftime("%H:%M:%S")
        header = f"[dim]{ts}[/dim]  [bold cyan]AI:[/bold cyan]"
        md = RichMarkdown(text)
        # Write header + rendered markdown to both Log and AI tabs
        for widget_id in ("#log-area", "#ai-area"):
            log_widget = self.query_one(widget_id, RichLog)
            log_widget.write(header)
            log_widget.write(md)
        # Switch to AI tab
        self.query_one("#log-tabs", TabbedContent).active = "tab-ai"

    def mark_finished(self):
        self.log_message("[bold green]Training complete![/bold green]")


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

    def start_training(self, smoketest: bool = False, scene_path: str | None = None,
                       resume: str | None = None):
        """Called by screens to start training (with dirty-tree gate)."""
        # If no scene_path provided (e.g. smoketest from main menu), use default
        if scene_path is None:
            bots = _discover_bots()
            scene_path = bots[0]["scene_path"] if bots else None

        # Smoketests skip the dirty-tree check
        if smoketest:
            self._do_start_training(smoketest=True, scene_path=scene_path, resume=resume)
            return

        clean, status = _git_is_clean()
        if clean:
            self._do_start_training(smoketest=False, scene_path=scene_path, resume=resume)
        else:
            self.push_screen(DirtyTreeScreen(
                status_lines=status,
                smoketest=False,
                scene_path=scene_path,
                resume=resume,
            ))

    def _do_start_training(self, smoketest: bool = False, scene_path: str | None = None,
                           resume: str | None = None):
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
        from train import run_training

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
        except Exception:
            log.exception("Training crashed in TUI worker thread")
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
        self, run_name: str, branch: str, algorithm: str, wandb_url: str | None,
        bot_name: str | None = None, experiment_hypothesis: str | None = None,
    ):
        """Called from training thread via call_from_thread."""
        if self._dashboard:
            self._dashboard.set_header(
                run_name, branch, algorithm, wandb_url, bot_name, experiment_hypothesis,
            )

    def set_total_batches(self, total: int | None):
        """Called from training thread via call_from_thread."""
        if self._dashboard:
            self._dashboard._total_batches = total


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

    # view
    p_view = sub.add_parser("view", help="Launch MuJoCo viewer")
    p_view.add_argument("--bot", type=str, default=None, help="Bot name (default: simple2wheeler)")
    p_view.add_argument("--stage", type=int, default=None, help="Curriculum stage 1-4 (default: none)")

    # play
    p_play = sub.add_parser("play", help="Play trained policy in viewer")
    p_play.add_argument("checkpoint", nargs="?", default="latest", help="Checkpoint ref (default: latest)")
    p_play.add_argument("--bot", type=str, default=None, help="Bot name (default: simple2wheeler)")
    p_play.add_argument("--run", type=str, default=None, help="Run name to play (resolves checkpoint from run dir)")

    # train
    p_train = sub.add_parser("train", help="Train (headless CLI, no TUI)")
    p_train.add_argument("--smoketest", action="store_true", help="Fast end-to-end smoketest")
    p_train.add_argument("--bot", type=str, default=None, help="Bot name (default: simple2wheeler)")
    p_train.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    p_train.add_argument("--num-workers", type=int, default=None, help="Number of parallel workers")
    p_train.add_argument("--no-dirty-check", action="store_true", help="Skip uncommitted-changes check")

    # smoketest (alias for train --smoketest)
    sub.add_parser("smoketest", help="Alias for train --smoketest")

    # quicksim
    sub.add_parser("quicksim", help="Quick simulation with Rerun debug vis")

    # visualize
    p_viz = sub.add_parser("visualize", help="One-shot Rerun visualization")
    p_viz.add_argument("--bot", type=str, default=None, help="Bot name (default: simple2wheeler)")
    p_viz.add_argument("--steps", type=int, default=1000, help="Number of sim steps")

    # validate-rewards
    p_vr = sub.add_parser("validate-rewards", help="Validate reward hierarchy for a bot")
    p_vr.add_argument("--bot", type=str, default=None, help="Bot name (default: all bots)")

    # describe
    p_desc = sub.add_parser("describe", help="Print human-readable pipeline summary")
    p_desc.add_argument("--bot", type=str, default=None, help="Bot name (default: simple2wheeler)")

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

    elif args.command == "view":
        scene_path = _resolve_scene_path(args.bot)
        from view import run_view
        run_view(scene_path, stage=args.stage)

    elif args.command == "play":
        scene_path = _resolve_scene_path(args.bot)
        from play import run_play
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
                    choice = input("[c] Commit with Claude  [s] Start anyway  [q] Quit: ").strip().lower()
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
        from train import main as train_main
        train_main(
            smoketest=args.smoketest,
            bot=args.bot,
            resume=args.resume,
            num_workers=args.num_workers,
            scene_path=scene_path,
        )

    elif args.command == "smoketest":
        from train import main as train_main
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
        from quick_sim import run_quick_sim
        run_quick_sim()

    elif args.command == "visualize":
        scene_path = _resolve_scene_path(args.bot)
        from visualize import run_visualization
        run_visualization(scene_path=scene_path, num_steps=args.steps)

    elif args.command == "validate-rewards":
        _validate_rewards(args.bot)

    elif args.command == "describe":
        from pipeline import pipeline_for_bot
        bot_name = args.bot or "simple2wheeler"
        pipeline = pipeline_for_bot(bot_name)
        print(pipeline.describe())


def _validate_rewards(bot_name: str | None):
    """Print reward hierarchy summary and run dominance checks."""
    from pipeline import pipeline_for_bot
    from reward_hierarchy import build_reward_hierarchy

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
    from reward_hierarchy import SURVIVE, TASK, STYLE, GUARD

    active = hierarchy.active_components()
    if not active:
        print("Scenario tests: no active components")
        return

    print("Scenario tests:")

    # Scenario 1: "Perfect stand" — healthy, no movement
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

    # Scenario 2: "Diving forward" — unhealthy, fast forward
    dive_reward = 0.0
    for c in active:
        if c.name == "alive":
            dive_reward += 0.0  # unhealthy — no alive bonus
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

    # Scenario 3: "Walking well" — healthy, moderate forward, upright
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
        print(f"\n  Standing ({stand_reward:+.3f}) <= Diving ({dive_reward:+.3f}) [!!] — diving is more rewarding than standing!")

    if walk_reward > stand_reward:
        print(f"  Walking ({walk_reward:+.3f}) > Standing ({stand_reward:+.3f}) [ok] — walking is the best outcome")
    else:
        print(f"  Walking ({walk_reward:+.3f}) <= Standing ({stand_reward:+.3f}) [!!] — standing is better than walking")


if __name__ == "__main__":
    main()
