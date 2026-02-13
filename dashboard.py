"""
Terminal dashboard for training progress.

Two implementations:
- TuiDashboard: Delegates to a Textual TUI app via call_from_thread.
- AnsiDashboard: Hand-rolled ANSI cursor movement (headless fallback).

Factory function `Dashboard()` picks the right one based on context.
"""

import shutil
import sys
import time


def _fmt_float(value, width=8, precision=3):
    """Format a float right-aligned to width."""
    if value is None:
        return " " * width
    s = f"{value:+.{precision}f}" if value < 0 or precision > 2 else f"{value:.{precision}f}"
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

    Same update/message/finish API as AnsiDashboard.
    Training code doesn't need to know which backend is active.
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
        self.app.call_from_thread(self.app.log_message, text)

    def finish(self):
        self.app.call_from_thread(self.app.mark_finished)


class AnsiDashboard:
    """
    In-place terminal dashboard using ANSI cursor movement.

    Headless fallback when no TUI is active (e.g. `python train.py --smoketest`).
    """

    def __init__(self, total_batches=None, algorithm="PPO"):
        self.total_batches = total_batches
        self.algorithm = algorithm
        self._lines_printed = 0
        self._last_render = 0.0
        self._min_interval = 0.1
        self._finished = False

    def _get_width(self):
        try:
            w = shutil.get_terminal_size().columns
        except Exception:
            w = 80
        return max(60, min(w, 120))

    def _progress_bar(self, batch, total, width):
        if total is None or total <= 0:
            return f"  batch {batch:,}"
        frac = min(batch / total, 1.0)
        bar_width = max(10, width - 40)
        filled = int(bar_width * frac)
        bar = "\u2588" * filled + "\u2591" * (bar_width - filled)
        pct = f"{100 * frac:5.1f}%"
        return f"  {bar}  {pct}   batch {batch:,}"

    def update(self, batch, metrics):
        now = time.monotonic()
        if now - self._last_render < self._min_interval:
            return
        self._last_render = now

        m = metrics
        width = self._get_width()
        is_ppo = self.algorithm == "PPO"

        lines = []
        header = " MindSim Training "
        rule = "\u2500" * ((width - len(header)) // 2)
        lines.append(f"\u2500\u2500{header}{rule}")
        lines.append(self._progress_bar(batch, self.total_batches, width))
        lines.append("")

        col1_label = "  EPISODE PERFORMANCE"
        col2_label = "OPTIMIZATION"
        pad = max(2, 36 - len(col1_label))
        lines.append(f"{col1_label}{' ' * pad}{col2_label}")

        def row(l_name, l_val, r_name, r_val):
            left = f"  {l_name:<18s}{l_val}"
            right = f"{r_name:<17s}{r_val}" if r_name else ""
            pad_mid = max(2, 36 - len(left))
            return f"{left}{' ' * pad_mid}{right}"

        lines.append(row(
            "avg reward", _fmt_float(m.get("avg_reward"), precision=2),
            "policy loss", _fmt_float(m.get("policy_loss"), precision=4) if is_ppo else _fmt_float(m.get("loss"), precision=4),
        ))
        lines.append(row(
            "best reward", _fmt_float(m.get("best_reward"), precision=2),
            "value loss", _fmt_float(m.get("value_loss"), precision=4) if is_ppo else "",
        ))
        lines.append(row(
            "worst reward", _fmt_float(m.get("worst_reward"), precision=2),
            "grad norm", _fmt_float(m.get("grad_norm"), precision=3),
        ))
        lines.append(row(
            "avg distance", f"{_fmt_float(m.get('avg_distance'), precision=2)} m",
            "entropy", _fmt_float(m.get("entropy"), precision=3),
        ))

        if is_ppo:
            lines.append(row(
                "avg steps", _fmt_int(m.get("avg_steps")),
                "clip fraction", _fmt_float(m.get("clip_fraction"), precision=2),
            ))
            lines.append(row(
                "", "",
                "approx KL", _fmt_float(m.get("approx_kl"), precision=4),
            ))
        else:
            lines.append(row(
                "avg steps", _fmt_int(m.get("avg_steps")),
                "", "",
            ))

        lines.append("")

        col1_label2 = "  SUCCESS RATES"
        col2_label2 = ""
        pad2 = max(2, 36 - len(col1_label2))
        lines.append(f"{col1_label2}{' ' * pad2}{col2_label2}")

        ps = m.get("policy_std")
        if ps is not None:
            try:
                std_str = f"{ps[0]:.2f} / {ps[1]:.2f}"
            except (IndexError, TypeError):
                std_str = f"{ps:.2f}"
        else:
            std_str = ""

        lines.append(row(
            "eval (rolling)", _fmt_pct(m.get("rolling_eval_success_rate")),
            "expl. variance" if is_ppo else "policy std",
            _fmt_float(m.get("explained_variance"), precision=2) if is_ppo else std_str.rjust(8),
        ))
        lines.append(row(
            "eval (batch)", _fmt_pct(m.get("eval_success_rate")),
            "policy std" if is_ppo else "",
            std_str.rjust(8) if is_ppo else "",
        ))
        lines.append(row(
            "train (batch)", _fmt_pct(m.get("batch_success_rate")),
            "mean value" if is_ppo else "",
            _fmt_float(m.get("mean_value"), precision=2) if is_ppo else "",
        ))
        if is_ppo:
            lines.append(row(
                "", "",
                "mean return", _fmt_float(m.get("mean_return"), precision=2),
            ))

        lines.append("")

        col1_label3 = "  CURRICULUM"
        col2_label3 = "TIMING"
        pad3 = max(2, 36 - len(col1_label3))
        lines.append(f"{col1_label3}{' ' * pad3}{col2_label3}")

        stage = m.get("curriculum_stage")
        num_stages = m.get("num_stages", 3)
        stage_str = f"{int(stage)} / {int(num_stages)}" if stage is not None else ""

        batch_time = m.get("batch_time")
        collect_time = m.get("collect_time")
        train_time = m.get("train_time")
        eval_time = m.get("eval_time")

        lines.append(row("stage", stage_str.rjust(8), "last batch", _fmt_time(batch_time)))
        lines.append(row("progress", _fmt_float(m.get("stage_progress"), precision=2), "\u251c collection", _fmt_time(collect_time)))
        lines.append(row("mastery", f"{int(m.get('mastery_count', 0)):>3d} / {int(m.get('mastery_batches', 20))}".rjust(8), "\u251c train", _fmt_time(train_time)))

        throughput = None
        if batch_time and batch_time > 0:
            bs = m.get("batch_size")
            if bs:
                throughput = bs / batch_time

        lines.append(row("max steps", _fmt_int(m.get("max_episode_steps")), "\u251c eval", _fmt_time(eval_time)))
        lines.append(row("", "", "\u2514 throughput", f"{throughput:.1f} ep/s".rjust(8) if throughput else "".rjust(8)))
        lines.append("\u2500" * width)

        if self._lines_printed > 0:
            sys.stdout.write(f"\033[{self._lines_printed}A\033[J")

        output = "\n".join(lines) + "\n"
        sys.stdout.write(output)
        sys.stdout.flush()
        self._lines_printed = len(lines)

    def message(self, text):
        if self._lines_printed > 0:
            sys.stdout.write(f"\033[{self._lines_printed}A\033[J")
            self._lines_printed = 0
        sys.stdout.write(text + "\n")
        sys.stdout.flush()
        self._last_render = 0.0

    def finish(self):
        self._finished = True
