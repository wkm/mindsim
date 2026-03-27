"""Training dashboard screen for MindSim TUI."""

from __future__ import annotations

import logging
import time
from datetime import datetime

from rich.markdown import Markdown as RichMarkdown
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import (
    Footer,
    RichLog,
    Static,
    TabbedContent,
    TabPane,
)

from botcad.colors import TUI_ERROR, TUI_SUCCESS, TUI_WARNING
from training.dashboard import _fmt_int, _fmt_pct, _fmt_time
from training.train import Cmd

log = logging.getLogger(__name__)


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
    """Wrap a formatted string in Rich color markup based on health ranges."""
    if value is None:
        return formatted
    if ranges.get("green") and ranges["green"](value):
        return f"[{TUI_SUCCESS}]{formatted}[/{TUI_SUCCESS}]"
    if ranges.get("yellow") and ranges["yellow"](value):
        return f"[{TUI_WARNING}]{formatted}[/{TUI_WARNING}]"
    if ranges.get("red") and ranges["red"](value):
        return f"[{TUI_ERROR}]{formatted}[/{TUI_ERROR}]"
    return formatted


class TrainingDashboard(Screen):
    """Fullscreen training dashboard with metrics and controls."""

    BINDINGS = [  # noqa: RUF012
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
        self._last_curriculum_cmd: float = 0.0

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
                        "  avg steps           ---",
                        id="m-avg-steps",
                        classes="metric-line",
                    )
                    yield Static(
                        "REWARD INPUTS",
                        classes="section-title",
                        id="ri-title",
                    )
                    yield Static(
                        "  distance            ---",
                        id="ri-distance",
                        classes="metric-line",
                    )
                    yield Static(
                        "  height              ---",
                        id="ri-height",
                        classes="metric-line",
                    )
                    yield Static(
                        "  survival            ---",
                        id="ri-survival",
                        classes="metric-line",
                    )
                    yield Static(
                        "  fwd distance        ---",
                        id="ri-fwd-dist",
                        classes="metric-line",
                    )
                    yield Static(
                        "  speed               ---",
                        id="ri-speed",
                        classes="metric-line",
                    )
                    yield Static(
                        "  lateral drift       ---",
                        id="ri-lateral",
                        classes="metric-line",
                    )
                    yield Static(
                        "  uprightness         ---",
                        id="ri-uprightness",
                        classes="metric-line",
                    )
                    yield Static(
                        "  fwd velocity        ---",
                        id="ri-fwd-vel",
                        classes="metric-line",
                    )
                    yield Static(
                        "  energy              ---",
                        id="ri-energy",
                        classes="metric-line",
                    )
                    yield Static(
                        "  contact             ---",
                        id="ri-contact",
                        classes="metric-line",
                    )
                    yield Static(
                        "  action jerk         ---",
                        id="ri-jerk",
                        classes="metric-line",
                    )
                    yield Static(
                        "  fell                ---",
                        id="ri-fell",
                        classes="metric-line",
                    )
                    yield Static(
                        "  joint activity      ---",
                        id="ri-joint",
                        classes="metric-line",
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
                        "  grad norm           ---",
                        id="m-grad-norm",
                        classes="metric-line",
                    )
                    yield Static(
                        "  entropy             ---",
                        id="m-entropy",
                        classes="metric-line",
                    )
                    yield Static(
                        "  clip fraction       ---",
                        id="m-clip-fraction",
                        classes="metric-line",
                    )
                    yield Static(
                        "  approx KL           ---",
                        id="m-approx-kl",
                        classes="metric-line",
                    )
                    yield Static(
                        "  explained var       ---",
                        id="m-explained-var",
                        classes="metric-line",
                    )
                    yield Static("CURRICULUM", classes="section-title")
                    yield Static(
                        "  stage               ---",
                        id="m-stage",
                        classes="metric-line",
                    )
                    yield Static(
                        "  progress            ---",
                        id="m-progress",
                        classes="metric-line",
                    )
                    yield Static(
                        "  mastery             ---",
                        id="m-mastery",
                        classes="metric-line",
                    )
                    yield Static(
                        "  max steps           ---",
                        id="m-max-steps",
                        classes="metric-line",
                    )
                    yield Static(
                        "  rec interval        ---",
                        id="m-rec-interval",
                        classes="metric-line",
                    )
                    yield Static("TIMING", classes="section-title")
                    yield Static(
                        "  batch               ---",
                        id="m-timing-batch",
                        classes="metric-line",
                    )
                    yield Static(
                        "  \u251c collect           ---",
                        id="m-timing-collect",
                        classes="metric-line",
                    )
                    yield Static(
                        "  \u251c train             ---",
                        id="m-timing-train",
                        classes="metric-line",
                    )
                    yield Static(
                        "  \u251c eval              ---",
                        id="m-timing-eval",
                        classes="metric-line",
                    )
                    yield Static(
                        "  \u2514 throughput         ---",
                        id="m-timing-throughput",
                        classes="metric-line",
                    )
            with Vertical(id="log-panel"), TabbedContent(id="log-tabs"):
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
            self.app.send_command(Cmd.PAUSE)
            self.log_message("[bold yellow]Paused[/bold yellow]")
            log.info("Paused")
        else:
            self.app.send_command(Cmd.UNPAUSE)
            self.log_message("[bold green]Resumed[/bold green]")
            log.info("Resumed")

    def action_step_batch(self) -> None:
        self.app.send_command(Cmd.STEP)
        self.log_message("Stepping one batch...")
        log.info("Stepping one batch")

    def action_checkpoint(self) -> None:
        self.app.send_command(Cmd.CHECKPOINT)
        self.log_message(
            "[bold #2B95D6]Checkpoint queued[/bold #2B95D6] (saves after current batch)"
        )

    def action_send_rerun(self) -> None:
        self.app.send_command(Cmd.LOG_RERUN)
        self.log_message(
            "[bold #2B95D6]Rerun recording queued[/bold #2B95D6] (records next eval episode)"
        )

    def action_advance_curriculum(self) -> None:
        now = time.monotonic()
        if now - self._last_curriculum_cmd < 0.3:
            return  # Debounce: ignore key repeats within 300ms
        self._last_curriculum_cmd = now
        self.app.send_command(Cmd.ADVANCE_CURRICULUM)
        self.log_message("Advancing curriculum...")

    def action_regress_curriculum(self) -> None:
        now = time.monotonic()
        if now - self._last_curriculum_cmd < 0.3:
            return  # Debounce: ignore key repeats within 300ms
        self._last_curriculum_cmd = now
        self.app.send_command(Cmd.REGRESS_CURRICULUM)
        self.log_message("Regressing curriculum...")

    def action_rerun_freq_down(self) -> None:
        self.app.send_command(Cmd.RERUN_FREQ_DOWN)
        self.log_message("Decreasing Rerun recording interval (more frequent)...")

    def action_rerun_freq_up(self) -> None:
        self.app.send_command(Cmd.RERUN_FREQ_UP)
        self.log_message("Increasing Rerun recording interval (less frequent)...")

    def action_ai_commentary(self) -> None:
        self.app.send_command(Cmd.AI_COMMENTARY)
        self.log_message("Generating AI commentary...")

    def action_open_wandb(self) -> None:
        if self._wandb_url:
            import webbrowser

            webbrowser.open(self._wandb_url)

    def action_quit_app(self) -> None:
        self.app.send_command(Cmd.STOP)
        self.app.exit()

    def _tick_elapsed(self) -> None:
        """Update elapsed time in the header bar every second."""
        elapsed = _fmt_elapsed(time.monotonic() - self._start_time)
        pause_tag = "  [bold yellow]PAUSED[/bold yellow]" if self._paused else ""
        header = " | ".join(self._header_parts) + f"  [{elapsed}]{pause_tag}"
        self.query_one("#header-bar", Static).update(header)

    def set_header(
        self,
        run_name: str,
        branch: str,
        algorithm: str,
        wandb_url: str | None,
        bot_name: str | None = None,
        experiment_hypothesis: str | None = None,
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
        _ri_row(
            "#ri-distance",
            "distance",
            ri_dist,
            f"{ri_dist:.2f} m" if ri_dist is not None else None,
        )
        ri_h = ri.get("torso_height")
        _ri_row(
            "#ri-height", "height", ri_h, f"{ri_h:.2f} m" if ri_h is not None else None
        )
        ri_surv = ri.get("survival_time")
        _ri_row(
            "#ri-survival",
            "survival",
            ri_surv,
            f"{ri_surv:.1f} s" if ri_surv is not None else None,
        )

        # Walking stage measures (biped only)
        ri_fd = ri.get("forward_distance")
        _ri_row(
            "#ri-fwd-dist",
            "fwd distance",
            ri_fd,
            f"{ri_fd:+.2f} m" if ri_fd is not None else None,
            "has_walking_stage",
        )
        ri_spd = ri.get("avg_speed")
        _ri_row(
            "#ri-speed",
            "speed",
            ri_spd,
            f"{ri_spd:.2f} m/s" if ri_spd is not None else None,
            "has_walking_stage",
        )
        ri_lat = ri.get("lateral_drift")
        _ri_row(
            "#ri-lateral",
            "lateral drift",
            ri_lat,
            f"{ri_lat:.2f} m" if ri_lat is not None else None,
            "has_walking_stage",
        )

        # Reward-component-gated
        ri_up = ri.get("up_z")
        _ri_row(
            "#ri-uprightness",
            "uprightness",
            ri_up,
            f"{ri_up:.2f}" if ri_up is not None else None,
            "upright",
        )
        ri_fv = ri.get("forward_vel")
        _ri_row(
            "#ri-fwd-vel",
            "fwd velocity",
            ri_fv,
            f"{ri_fv:+.2f} m/s" if ri_fv is not None else None,
            "forward_vel",
        )
        ri_en = ri.get("energy")
        _ri_row(
            "#ri-energy",
            "energy",
            ri_en,
            f"{ri_en:.3f}" if ri_en is not None else None,
            "energy",
        )
        ri_ct = ri.get("contact_frac")
        _ri_row(
            "#ri-contact",
            "contact",
            ri_ct,
            f"{ri_ct:.1%}" if ri_ct is not None else None,
            "contact",
        )
        ri_jk = ri.get("action_jerk")
        _ri_row(
            "#ri-jerk",
            "action jerk",
            ri_jk,
            f"{ri_jk:.3f}" if ri_jk is not None else None,
            "smoothness",
        )

        # Fall detection gated
        ri_fell = ri.get("fell_frac")
        _ri_row(
            "#ri-fell",
            "fell",
            ri_fell,
            f"{ri_fell:.0%}" if ri_fell is not None else None,
            "fall_detection",
        )

        # Joint activity (shown when sensors exist, i.e. value > 0 or biped)
        ri_ja = ri.get("joint_activity")
        has_sensors = ri_ja is not None and ri_ja > 0
        widget = self.query_one("#ri-joint")
        if has_sensors:
            widget.update(f"  {'joint activity':<15s}{f'{ri_ja:.1f} rad/s'.rjust(8)}")
            widget.display = True
        elif ri_ja is not None and any(
            scales.get(k, 0) > 0 for k in ("upright", "energy", "contact")
        ):
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
            cf_val = m.get("clip_fraction")
            cf_str = _fmt(cf_val, precision=2)
            cf_str = _color_val(
                cf_str,
                cf_val,
                {
                    "green": lambda v: 0.05 <= v <= 0.30,
                    "yellow": lambda v: v < 0.05 or 0.30 < v <= 0.50,
                    "red": lambda v: v > 0.50,
                },
            )
            self.query_one("#m-clip-fraction").update(f"  clip fraction    {cf_str}")
            # Approx KL: green 0.005-0.05, yellow <0.005 or 0.05-0.1, red >0.1
            kl_val = m.get("approx_kl")
            kl_str = _fmt(kl_val, precision=4)
            kl_str = _color_val(
                kl_str,
                kl_val,
                {
                    "green": lambda v: 0.005 <= v <= 0.05,
                    "yellow": lambda v: v < 0.005 or 0.05 < v <= 0.1,
                    "red": lambda v: v > 0.1,
                },
            )
            self.query_one("#m-approx-kl").update(f"  approx KL        {kl_str}")
            # Explained variance: green >0.5, yellow 0.0-0.5, red <0.0
            ev_val = m.get("explained_variance")
            ev_str = _fmt(ev_val, precision=3)
            ev_str = _color_val(
                ev_str,
                ev_val,
                {
                    "green": lambda v: v > 0.5,
                    "yellow": lambda v: 0.0 <= v <= 0.5,
                    "red": lambda v: v < 0.0,
                },
            )
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
        gn_val = m.get("grad_norm")
        max_gn = m.get("max_grad_norm", 0.5)
        gn_str = _fmt(gn_val, precision=3)
        if gn_val is not None and max_gn > 0:
            ratio = gn_val / max_gn
            if ratio > 10:
                gn_str = (
                    f"[{TUI_ERROR}]{gn_str}[/{TUI_ERROR}] [dim]({ratio:.0f}x)[/dim]"
                )
            elif ratio > 2:
                gn_str = (
                    f"[{TUI_WARNING}]{gn_str}[/{TUI_WARNING}] [dim]({ratio:.0f}x)[/dim]"
                )
            else:
                gn_str = f"[{TUI_SUCCESS}]{gn_str}[/{TUI_SUCCESS}]"
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
            f"  batch            {_fmt_time(bt)}"
            if bt is not None
            else "  batch               ---"
        )
        self.query_one("#m-timing-collect").update(
            f"  \u251c collect        {_fmt_time(ct)}"
            if ct is not None
            else "  \u251c collect           ---"
        )
        self.query_one("#m-timing-train").update(
            f"  \u251c train          {_fmt_time(tt)}"
            if tt is not None
            else "  \u251c train             ---"
        )
        self.query_one("#m-timing-eval").update(
            f"  \u251c eval           {_fmt_time(et)}"
            if et is not None
            else "  \u251c eval              ---"
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
        header = f"[dim]{ts}[/dim]  [bold #2B95D6]AI:[/bold #2B95D6]"
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
