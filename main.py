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
import logging.handlers
import os
import re
import shutil
import threading
import sys
import time
from datetime import datetime
from pathlib import Path
from queue import Queue

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
)

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
        self.app.start_playing_run(
            run_name=self._info.name,
            scene_path=self._info.scene_path,
        )

    def action_resume_run(self) -> None:
        self.app.start_training(
            smoketest=False,
            scene_path=self._info.scene_path,
            resume=self._info.name,
        )

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

    #log-panel-title {
        text-style: bold;
        color: $accent;
        height: 1;
    }

    #log-area {
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
                yield Static("TIMING", classes="section-title")
                yield Static("  batch               ---", id="m-timing-batch", classes="metric-line")
                yield Static("  \u251c collect           ---", id="m-timing-collect", classes="metric-line")
                yield Static("  \u251c train             ---", id="m-timing-train", classes="metric-line")
                yield Static("  \u251c eval              ---", id="m-timing-eval", classes="metric-line")
                yield Static("  \u2514 throughput         ---", id="m-timing-throughput", classes="metric-line")
          with Vertical(id="log-panel"):
              yield Static("LOG", id="log-panel-title")
              yield RichLog(id="log-area", wrap=True, max_lines=200, markup=True)
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

    def action_send_rerun(self) -> None:
        self.app.send_command("log_rerun")

    def action_advance_curriculum(self) -> None:
        self.app.send_command("advance_curriculum")

    def action_regress_curriculum(self) -> None:
        self.app.send_command("regress_curriculum")

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

        # Optimization
        if is_ppo:
            self.query_one("#m-policy-loss").update(
                f"  policy loss      {_fmt(m.get('policy_loss'), precision=4)}"
            )
            self.query_one("#m-value-loss").update(
                f"  value loss       {_fmt(m.get('value_loss'), precision=4)}"
            )
            self.query_one("#m-clip-fraction").update(
                f"  clip fraction    {_fmt(m.get('clip_fraction'), precision=2)}"
            )
            self.query_one("#m-approx-kl").update(
                f"  approx KL        {_fmt(m.get('approx_kl'), precision=4)}"
            )
        else:
            self.query_one("#m-policy-loss").update(
                f"  loss             {_fmt(m.get('loss'), precision=4)}"
            )
            self.query_one("#m-value-loss").update("  value loss       ---")
            self.query_one("#m-clip-fraction").update("  clip fraction    ---")
            self.query_one("#m-approx-kl").update("  approx KL        ---")
        self.query_one("#m-grad-norm").update(
            f"  grad norm        {_fmt(m.get('grad_norm'), precision=3)}"
        )
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
        self.command_queue: Queue[str] = Queue()
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
        """Called by screens to start training."""
        # If no scene_path provided (e.g. smoketest from main menu), use default
        if scene_path is None:
            bots = _discover_bots()
            scene_path = bots[0]["scene_path"] if bots else None
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
                self.command_queue,
                smoketest=self._smoketest,
                num_workers=num_workers,
                scene_path=self._scene_path,
                resume=self._resume,
            )
        finally:
            # Remove TUI log handler to avoid stale references
            if hasattr(self, "_tui_log_handler"):
                logging.getLogger().removeHandler(self._tui_log_handler)

    def send_command(self, cmd: str):
        self.command_queue.put(cmd)

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


def _setup_logging() -> Path:
    """Configure file logging for all MindSim operations.

    Returns the log file path. Logs are written to logs/mindsim.log with
    rotation (5 x 5MB). Also installs an excepthook so unhandled exceptions
    are captured in the log.
    """
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "mindsim.log"

    handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=5,
    )
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)

    # Capture unhandled exceptions to the log
    _original_excepthook = sys.excepthook

    def _logging_excepthook(exc_type, exc_value, exc_tb):
        if not issubclass(exc_type, KeyboardInterrupt):
            logging.critical(
                "Unhandled exception", exc_info=(exc_type, exc_value, exc_tb)
            )
        _original_excepthook(exc_type, exc_value, exc_tb)

    sys.excepthook = _logging_excepthook

    logging.info("MindSim started: %s", " ".join(sys.argv))
    return log_file


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

    # smoketest (alias for train --smoketest)
    sub.add_parser("smoketest", help="Alias for train --smoketest")

    # quicksim
    sub.add_parser("quicksim", help="Quick simulation with Rerun debug vis")

    # visualize
    p_viz = sub.add_parser("visualize", help="One-shot Rerun visualization")
    p_viz.add_argument("--bot", type=str, default=None, help="Bot name (default: simple2wheeler)")
    p_viz.add_argument("--steps", type=int, default=1000, help="Number of sim steps")

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
    log_file = _setup_logging()
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


if __name__ == "__main__":
    main()
