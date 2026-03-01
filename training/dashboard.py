"""
Dashboard helpers for training progress.

TuiDashboard delegates to a Textual TUI app via call_from_thread.
Format helpers are shared with main.py's TrainingDashboard widget.
"""

import logging
import re

log = logging.getLogger(__name__)


def _fmt_float(value, width=8, precision=3):
    """Format a float right-aligned to width."""
    if value is None:
        return " " * width
    s = (
        f"{value:+.{precision}f}"
        if value < 0 or precision > 2
        else f"{value:.{precision}f}"
    )
    return s.rjust(width)


def _fmt_pct(value, width=8):
    """Format a fraction as percentage, right-aligned."""
    if value is None:
        return " " * width
    return f"{value:.0%}".rjust(width)


def _fmt_int(value, width=8):
    """Format an integer right-aligned."""
    if value is None:
        return " " * width
    return f"{int(value):,}".rjust(width)


def _fmt_time(seconds, width=8):
    """Format seconds as human-readable, right-aligned."""
    if seconds is None:
        return " " * width
    if seconds < 100:
        return f"{seconds:.1f}s".rjust(width)
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s".rjust(width)


class TuiDashboard:
    """
    Dashboard that delegates to a Textual TUI app.

    Training code calls update/message/finish; these are forwarded
    to the Textual app thread via call_from_thread.
    """

    def __init__(self, app, total_batches=None, algorithm="PPO"):
        self.app = app
        self.total_batches = total_batches
        self.algorithm = algorithm
        # Push total_batches to the TUI
        self.app.call_from_thread(self.app.set_total_batches, total_batches)

    def update(self, batch, metrics):
        self.app.call_from_thread(self.app.update_metrics, batch, metrics)

    def message(self, text):
        # Single path: Python logging is the source of truth.
        # TuiLogHandler routes the record to the RichLog panel;
        # the file handler writes it to logs/mindsim.log.
        plain = re.sub(r"\[/?[^\]]*\]", "", text)
        log.info(plain)

    def finish(self):
        self.app.call_from_thread(self.app.mark_finished)


class LogDashboard:
    """Minimal headless dashboard that logs a one-line summary per batch.

    Used by the CLI entry point (main.py train / main.py smoketest) where
    there is no TUI.
    """

    def update(self, batch, metrics):
        m = metrics
        reward = m.get("avg_reward")
        dist = m.get("avg_distance")
        stage = m.get("curriculum_stage")
        r_str = f"reward={reward:+.2f}" if reward is not None else ""
        d_str = f"dist={dist:.2f}m" if dist is not None else ""
        s_str = f"S{int(stage)}" if stage is not None else ""
        log.info("batch %d  %s  %s  %s", batch, r_str, d_str, s_str)

    def message(self, text):
        plain = re.sub(r"\[/?[^\]]*\]", "", text)
        log.info(plain)

    def finish(self):
        log.info("Training complete.")
