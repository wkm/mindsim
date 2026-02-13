"""
Textual TUI for MindSim training.

Single entry point for train, visualize, and smoketest modes.
Launches a fullscreen dashboard with interactive controls.

Usage:
    uv run python tui.py
"""

from __future__ import annotations

import subprocess
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
    Button,
    Footer,
    RadioButton,
    RadioSet,
    RichLog,
    Static,
)


def _discover_bots() -> list[dict]:
    """Scan bots/*/scene.xml and return info about each bot."""
    bots_dir = Path("bots")
    results = []
    if bots_dir.is_dir():
        for scene in sorted(bots_dir.glob("*/scene.xml")):
            name = scene.parent.name
            results.append({"name": name, "scene_path": str(scene)})
    return results


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


def _fmt_pct(value, width=8):
    if value is None:
        return " " * width
    return f"{value:.0%}".rjust(width)


def _fmt_int(value, width=8):
    if value is None:
        return " " * width
    return f"{int(value):,}".rjust(width)


def _fmt_time(seconds, width=8):
    if seconds is None:
        return " " * width
    if seconds < 100:
        return f"{seconds:.1f}s".rjust(width)
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s".rjust(width)


# ---------------------------------------------------------------------------
# Launcher Screen
# ---------------------------------------------------------------------------

class LauncherScreen(Screen):
    """Mode selection screen shown on startup."""

    BINDINGS = [
        Binding("v", "launch('btn-view')", "View", priority=True),
        Binding("s", "launch('btn-smoketest')", "Smoketest", priority=True),
        Binding("t", "launch('btn-train')", "Train", priority=True),
        Binding("p", "launch('btn-play')", "Play", priority=True),
        Binding("q", "launch('btn-quit')", "Quit", priority=True),
    ]

    CSS = """
    LauncherScreen {
        align: center middle;
    }

    #launcher-box {
        width: 60;
        height: auto;
        border: round $accent;
        padding: 1 2;
    }

    #launcher-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }

    #launcher-subtitle {
        text-align: center;
        color: $text-muted;
        margin-bottom: 1;
    }

    .launcher-section {
        text-style: bold;
        color: $accent;
        margin-top: 1;
    }

    #bot-selector {
        margin: 0 2;
        height: auto;
    }

    .launcher-btn {
        width: 100%;
        margin: 1 0 0 0;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._bots = _discover_bots()

    def compose(self) -> ComposeResult:
        with Vertical(id="launcher-box"):
            yield Static("MindSim", id="launcher-title")
            yield Static("Bot:", classes="launcher-section")
            if self._bots:
                with RadioSet(id="bot-selector"):
                    for i, bot in enumerate(self._bots):
                        yield RadioButton(bot["name"], value=(i == 0))
            else:
                yield Static("  No bots found in bots/*/scene.xml")
            yield Static("Mode:", classes="launcher-section")
            yield Button("[v] View", id="btn-view", classes="launcher-btn")
            yield Button("[s] Smoketest", id="btn-smoketest", classes="launcher-btn")
            yield Button("[t] Train", id="btn-train", classes="launcher-btn")
            yield Button("[p] Play", id="btn-play", classes="launcher-btn")
            yield Button("[q] Quit", id="btn-quit", classes="launcher-btn", variant="error")
        yield Footer()

    def _get_selected_scene(self) -> str | None:
        """Get scene_path for the selected bot."""
        if not self._bots:
            return None
        try:
            radio_set = self.query_one("#bot-selector", RadioSet)
            idx = radio_set.pressed_index
            if idx >= 0:
                return self._bots[idx]["scene_path"]
        except Exception:
            pass
        return self._bots[0]["scene_path"] if self._bots else None

    def action_launch(self, button_id: str) -> None:
        """Handle keyboard shortcut by simulating button press."""
        try:
            btn = self.query_one(f"#{button_id}", Button)
            btn.press()
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        scene_path = self._get_selected_scene()
        if event.button.id == "btn-train":
            self.app.start_training(smoketest=False, scene_path=scene_path)
        elif event.button.id == "btn-smoketest":
            self.app.start_training(smoketest=True, scene_path=scene_path)
        elif event.button.id == "btn-view":
            self.app.start_viewing(scene_path=scene_path)
        elif event.button.id == "btn-play":
            self.app.start_playing(scene_path=scene_path)
        elif event.button.id == "btn-quit":
            self.app.exit()


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
        Binding("up", "advance_curriculum", "Advance"),
        Binding("down", "regress_curriculum", "Regress"),
        Binding("q", "quit_app", "Quit"),
        Binding("escape", "quit_app", "Quit", show=False),
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

    #progress-row {
        height: 1;
        padding: 0 1;
        margin-bottom: 1;
    }

    #progress-label {
        width: 100%;
        height: 1;
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

    #log-area {
        height: 1fr;
        min-height: 6;
        border-top: solid $surface;
        padding: 0 1;
        margin-top: 1;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._total_batches = None
        self._algorithm = "PPO"
        self._start_time = time.monotonic()
        self._header_parts: list[str] = ["MindSim"]
        self._paused = False

    def compose(self) -> ComposeResult:
        yield Static("MindSim", id="header-bar")
        with Horizontal(id="progress-row"):
            yield Static("  batch 0", id="progress-label")
        with Horizontal(id="metrics-grid"):
            with Vertical(classes="metrics-col"):
                yield Static("EPISODE PERFORMANCE", classes="section-title")
                yield Static("  avg reward          ---", id="m-avg-reward", classes="metric-line")
                yield Static("  best reward         ---", id="m-best-reward", classes="metric-line")
                yield Static("  worst reward        ---", id="m-worst-reward", classes="metric-line")
                yield Static("  avg distance        ---", id="m-avg-distance", classes="metric-line")
                yield Static("  avg steps           ---", id="m-avg-steps", classes="metric-line")
                yield Static("SUCCESS RATES", classes="section-title")
                yield Static("  eval (rolling)      ---", id="m-eval-rolling", classes="metric-line")
                yield Static("  eval (batch)        ---", id="m-eval-batch", classes="metric-line")
                yield Static("  train (batch)       ---", id="m-train-batch", classes="metric-line")
                yield Static("  policy std          ---", id="m-policy-std", classes="metric-line")
            with Vertical(classes="metrics-col"):
                yield Static("OPTIMIZATION", classes="section-title")
                yield Static("  policy loss         ---", id="m-policy-loss", classes="metric-line")
                yield Static("  value loss          ---", id="m-value-loss", classes="metric-line")
                yield Static("  grad norm           ---", id="m-grad-norm", classes="metric-line")
                yield Static("  entropy             ---", id="m-entropy", classes="metric-line")
                yield Static("  clip fraction       ---", id="m-clip-fraction", classes="metric-line")
                yield Static("  approx KL           ---", id="m-approx-kl", classes="metric-line")
                yield Static("CURRICULUM", classes="section-title")
                yield Static("  stage               ---", id="m-stage", classes="metric-line")
                yield Static("  progress            ---", id="m-progress", classes="metric-line")
                yield Static("  mastery             ---", id="m-mastery", classes="metric-line")
                yield Static("  max steps           ---", id="m-max-steps", classes="metric-line")
                yield Static("TIMING", classes="section-title")
                yield Static("  ---", id="m-timing", classes="metric-line")
        yield RichLog(id="log-area", wrap=True, max_lines=100, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        self.log_message("Dashboard ready. Waiting for training to start...")
        self.set_interval(1.0, self._tick_elapsed)

    def action_toggle_pause(self) -> None:
        self._paused = not self._paused
        if self._paused:
            self.app.send_command("pause")
            self.log_message("[bold yellow]Paused[/bold yellow]")
        else:
            self.app.send_command("unpause")
            self.log_message("[bold green]Resumed[/bold green]")

    def action_step_batch(self) -> None:
        self.app.send_command("step")
        self.log_message("Stepping one batch...")

    def action_checkpoint(self) -> None:
        self.app.send_command("checkpoint")

    def action_send_rerun(self) -> None:
        self.app.send_command("log_rerun")

    def action_advance_curriculum(self) -> None:
        self.app.send_command("advance_curriculum")

    def action_regress_curriculum(self) -> None:
        self.app.send_command("regress_curriculum")

    def action_quit_app(self) -> None:
        self.app.send_command("stop")
        self.app.exit()

    def _tick_elapsed(self) -> None:
        """Update elapsed time in the header bar every second."""
        elapsed = _fmt_elapsed(time.monotonic() - self._start_time)
        pause_tag = "  [bold yellow]PAUSED[/bold yellow]" if self._paused else ""
        header = " | ".join(self._header_parts) + f"  [{elapsed}]{pause_tag}"
        self.query_one("#header-bar", Static).update(header)

    def set_header(self, run_name: str, branch: str, algorithm: str, wandb_url: str | None):
        self._algorithm = algorithm
        self._start_time = time.monotonic()
        parts = [f"MindSim | {run_name}", branch, algorithm]
        if wandb_url:
            parts.append(wandb_url)
        self._header_parts = parts
        # Render immediately (timer will keep updating)
        self._tick_elapsed()

    def update_metrics(self, batch: int, metrics: dict):
        m = metrics
        is_ppo = self._algorithm == "PPO"
        total = self._total_batches

        # Progress bar (text-based)
        if total and total > 0:
            frac = min(batch / total, 1.0)
            pct = 100 * frac
            bar_w = 30
            filled = int(bar_w * frac)
            bar = "\u2588" * filled + "\u2591" * (bar_w - filled)
            self.query_one("#progress-label", Static).update(
                f"  {bar}  {pct:5.1f}%   batch {batch:,} / {total:,}"
            )
        else:
            self.query_one("#progress-label", Static).update(f"  batch {batch:,}")

        # Episode performance
        self.query_one("#m-avg-reward").update(f"  avg reward       {_fmt(m.get('avg_reward'), precision=2)}")
        self.query_one("#m-best-reward").update(f"  best reward      {_fmt(m.get('best_reward'), precision=2)}")
        self.query_one("#m-worst-reward").update(f"  worst reward     {_fmt(m.get('worst_reward'), precision=2)}")
        dist = m.get('avg_distance')
        dist_str = f"{_fmt(dist, precision=2)} m" if dist is not None else "---"
        self.query_one("#m-avg-distance").update(f"  avg distance     {dist_str}")
        self.query_one("#m-avg-steps").update(f"  avg steps        {_fmt_int(m.get('avg_steps'))}")

        # Success rates
        self.query_one("#m-eval-rolling").update(f"  eval (rolling)   {_fmt_pct(m.get('rolling_eval_success_rate'))}")
        self.query_one("#m-eval-batch").update(f"  eval (batch)     {_fmt_pct(m.get('eval_success_rate'))}")
        self.query_one("#m-train-batch").update(f"  train (batch)    {_fmt_pct(m.get('batch_success_rate'))}")

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
            self.query_one("#m-policy-loss").update(f"  policy loss      {_fmt(m.get('policy_loss'), precision=4)}")
            self.query_one("#m-value-loss").update(f"  value loss       {_fmt(m.get('value_loss'), precision=4)}")
            self.query_one("#m-clip-fraction").update(f"  clip fraction    {_fmt(m.get('clip_fraction'), precision=2)}")
            self.query_one("#m-approx-kl").update(f"  approx KL        {_fmt(m.get('approx_kl'), precision=4)}")
        else:
            self.query_one("#m-policy-loss").update(f"  loss             {_fmt(m.get('loss'), precision=4)}")
            self.query_one("#m-value-loss").update(f"  value loss       ---")
            self.query_one("#m-clip-fraction").update(f"  clip fraction    ---")
            self.query_one("#m-approx-kl").update(f"  approx KL        ---")
        self.query_one("#m-grad-norm").update(f"  grad norm        {_fmt(m.get('grad_norm'), precision=3)}")
        self.query_one("#m-entropy").update(f"  entropy          {_fmt(m.get('entropy'), precision=3)}")

        # Curriculum
        stage = m.get("curriculum_stage")
        num_stages = m.get("num_stages", 3)
        stage_str = f"{int(stage)} / {int(num_stages)}" if stage is not None else "---"
        self.query_one("#m-stage").update(f"  stage            {stage_str.rjust(8)}")
        self.query_one("#m-progress").update(f"  progress         {_fmt(m.get('stage_progress'), precision=2)}")
        mc = m.get("mastery_count", 0)
        mb = m.get("mastery_batches", 20)
        self.query_one("#m-mastery").update(f"  mastery          {f'{int(mc):>3d} / {int(mb)}'.rjust(8)}")
        self.query_one("#m-max-steps").update(f"  max steps        {_fmt_int(m.get('max_episode_steps'))}")

        # Timing
        bt = m.get("batch_time")
        ct = m.get("collect_time")
        tt = m.get("train_time")
        et = m.get("eval_time")
        bs = m.get("batch_size")
        throughput = ""
        if bt and bt > 0 and bs:
            throughput = f" | {bs / bt:.1f} ep/s"
        timing_parts = []
        if bt is not None:
            timing_parts.append(f"batch {bt*1000:.0f}ms")
        if ct is not None:
            timing_parts.append(f"collect {ct*1000:.0f}ms")
        if tt is not None:
            timing_parts.append(f"train {tt*1000:.0f}ms")
        if et is not None:
            timing_parts.append(f"eval {et*1000:.0f}ms")
        self.query_one("#m-timing").update(f"  {' | '.join(timing_parts)}{throughput}")

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
        # Set by launcher to dispatch after app.run() returns
        self.next_action: str | None = None
        self.next_scene: str | None = None

    def on_mount(self) -> None:
        self.push_screen(LauncherScreen())

    def start_viewing(self, scene_path: str | None = None):
        """Exit TUI, then main() will launch the MuJoCo viewer."""
        if not scene_path:
            return
        self.next_action = "view"
        self.next_scene = scene_path
        self.exit()

    def start_playing(self, scene_path: str | None = None):
        """Exit TUI, then main() will launch play mode."""
        self.next_action = "play"
        self.next_scene = scene_path
        self.exit()

    def start_training(self, smoketest: bool = False, scene_path: str | None = None):
        """Called by launcher to start training."""
        dashboard = TrainingDashboard()
        self._dashboard = dashboard
        self._smoketest = smoketest
        self._scene_path = scene_path
        self.push_screen(dashboard)
        self._run_training()

    @work(thread=True, exclusive=True)
    def _run_training(self) -> None:
        from train import run_training
        # Force serial collection in smoketest to avoid multiprocessing issues
        num_workers = 1 if self._smoketest else None
        run_training(
            self, self.command_queue,
            smoketest=self._smoketest,
            num_workers=num_workers,
            scene_path=self._scene_path,
        )

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

    def set_header(self, run_name: str, branch: str, algorithm: str, wandb_url: str | None):
        """Called from training thread via call_from_thread."""
        if self._dashboard:
            self._dashboard.set_header(run_name, branch, algorithm, wandb_url)

    def set_total_batches(self, total: int | None):
        """Called from training thread via call_from_thread."""
        if self._dashboard:
            self._dashboard._total_batches = total


def main():
    app = MindSimApp()
    app.run()

    # Dispatch to selected mode after TUI exits
    if app.next_action == "view":
        import mujoco
        import mujoco.viewer
        model = mujoco.MjModel.from_xml_path(app.next_scene)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        mujoco.viewer.launch(model, data)

    elif app.next_action == "play":
        from play import main as play_main
        # Override sys.argv so play.py sees the right args
        argv = ["play.py", "latest"]
        if app.next_scene:
            argv.append(app.next_scene)
        sys.argv = argv
        play_main()


if __name__ == "__main__":
    main()
