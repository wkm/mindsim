#!/usr/bin/env python3
"""W&B run analysis — CLI tool and importable library.

Pulls metrics from W&B runs for post-hoc analysis, diagnostics, and
comparison. Outputs markdown optimized for Claude to read and reason about.

Usage:
    uv run python wandb_analysis.py <run>              # Full report
    uv run python wandb_analysis.py <run> --overview    # Config + status
    uv run python wandb_analysis.py <run> --metrics     # Key metric trajectories
    uv run python wandb_analysis.py <run> --rewards     # Reward decomposition
    uv run python wandb_analysis.py <run> --diagnostics # Health checks
    uv run python wandb_analysis.py <run> --raw         # Dump raw latest values
    uv run python wandb_analysis.py <run> --csv         # Export timeseries as CSV

<run> accepts: W&B run ID, full URL, or run name.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
from dataclasses import dataclass, field
from typing import Any

import wandb

WANDB_PROJECT = "mindsim"

# Metrics to track in trajectory analysis, grouped by category
TRAJECTORY_METRICS = {
    "performance": [
        "batch/avg_reward",
        "batch/avg_steps",
        "batch/success_rate",
        "batch/fell_fraction",
    ],
    "training": [
        "training/entropy",
        "training/grad_norm",
        "training/policy_loss",
        "training/value_loss",
        "training/clip_fraction",
        "training/approx_kl",
        "training/explained_variance",
    ],
    "curriculum": [
        "curriculum/stage",
        "curriculum/eval_rolling_success_rate",
        "curriculum/train_rolling_success_rate",
        "curriculum/max_episode_steps",
    ],
    "raw": [
        "raw/up_z",
        "raw/torso_height",
        "raw/forward_vel",
        "raw/energy",
        "raw/avg_speed",
        "raw/survival_time",
    ],
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RunData:
    """All data fetched from a single W&B run."""

    run_id: str
    name: str
    state: str  # running, finished, crashed, failed
    tags: list[str]
    config: dict[str, Any]
    summary: dict[str, Any]
    history: list[dict[str, Any]]  # timeseries (sampled)
    url: str = ""
    created_at: str = ""
    duration_seconds: float = 0.0


@dataclass
class Diagnostic:
    """A detected issue or observation."""

    severity: str  # "critical", "warning", "info"
    category: str  # "entropy", "gradient", "curriculum", "reward", etc.
    message: str
    evidence: str  # supporting data


@dataclass
class MetricTrajectory:
    """A metric's values at early/mid/late points."""

    name: str
    early: float | None  # first 10% of history
    mid: float | None  # middle 10%
    late: float | None  # last 10%
    trend: str  # "rising", "falling", "stable", "insufficient_data"
    latest: float | None


@dataclass
class AnalysisReport:
    """Complete analysis of a run."""

    run_data: RunData
    trajectories: dict[str, list[MetricTrajectory]] = field(default_factory=dict)
    reward_components: dict[str, float] = field(default_factory=dict)
    diagnostics: list[Diagnostic] = field(default_factory=list)


# ---------------------------------------------------------------------------
# W&B API helpers
# ---------------------------------------------------------------------------

_api: wandb.Api | None = None


def _get_api() -> wandb.Api:
    global _api
    if _api is None:
        _api = wandb.Api(timeout=15)
    return _api


def resolve_run_ref(ref: str) -> str:
    """Resolve a run reference to a W&B run path.

    Accepts:
        - Run ID: "9ndvkyic"
        - Full URL: "https://wandb.ai/wkm/mindsim/runs/9ndvkyic"
        - Run name: "child-lstm-0222-2238"

    Returns the run path like "wkm/mindsim/9ndvkyic".
    """
    # URL format
    url_match = re.match(
        r"https?://wandb\.ai/([^/]+)/([^/]+)/runs/([^/?#]+)", ref
    )
    if url_match:
        entity, project, run_id = url_match.groups()
        return f"{entity}/{project}/{run_id}"

    # If it looks like a run ID (short alphanumeric), use it directly
    if re.match(r"^[a-z0-9]{6,12}$", ref):
        api = _get_api()
        # Need to discover entity
        runs = api.runs(WANDB_PROJECT, per_page=1)
        if runs:
            entity = runs[0].entity
            return f"{entity}/{WANDB_PROJECT}/{ref}"
        raise ValueError(f"Could not determine entity for project {WANDB_PROJECT}")

    # Otherwise assume it's a run name — search for it
    api = _get_api()
    runs = api.runs(WANDB_PROJECT, filters={"display_name": ref}, per_page=1)
    if runs:
        r = runs[0]
        return f"{r.entity}/{r.project}/{r.id}"

    raise ValueError(
        f"Could not resolve run reference: {ref!r}. "
        f"Try a run ID, URL, or exact run name."
    )


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------


def fetch_run(ref: str, samples: int = 500) -> RunData:
    """Fetch all data for a run from W&B."""
    api = _get_api()
    path = resolve_run_ref(ref)
    run = api.run(path)

    # Fetch history as list of dicts (no pandas dependency)
    history = []
    for row in run.scan_history(page_size=samples):
        history.append(dict(row))

    # Subsample if we got more than requested
    if len(history) > samples:
        step = len(history) / samples
        history = [history[int(i * step)] for i in range(samples)]

    # Duration from summary or metadata
    duration = 0.0
    if hasattr(run, "summary") and "_runtime" in run.summary:
        duration = float(run.summary["_runtime"])

    return RunData(
        run_id=run.id,
        name=run.name,
        state=run.state,
        tags=list(run.tags or []),
        config=dict(run.config or {}),
        summary=dict(run.summary or {}),
        history=history,
        url=run.url,
        created_at=str(getattr(run, "created_at", "")),
        duration_seconds=duration,
    )


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _get_history_values(history: list[dict], key: str) -> list[float]:
    """Extract non-None numeric values for a key from history."""
    values = []
    for row in history:
        v = row.get(key)
        if v is not None and isinstance(v, (int, float)):
            values.append(float(v))
    return values


def _compute_trajectory(
    history: list[dict], key: str
) -> MetricTrajectory:
    """Compute early/mid/late trajectory for a metric."""
    values = _get_history_values(history, key)
    if len(values) < 3:
        return MetricTrajectory(
            name=key,
            early=values[0] if values else None,
            mid=None,
            late=values[-1] if values else None,
            trend="insufficient_data",
            latest=values[-1] if values else None,
        )

    n = len(values)
    # Chunk into early (first 10%), mid (middle 10%), late (last 10%)
    chunk = max(1, n // 10)
    early = sum(values[:chunk]) / chunk
    mid_start = n // 2 - chunk // 2
    mid_vals = values[mid_start : mid_start + chunk]
    mid = sum(mid_vals) / len(mid_vals) if mid_vals else None
    late = sum(values[-chunk:]) / chunk

    # Determine trend
    if early == 0 and late == 0:
        trend = "stable"
    elif early == 0:
        trend = "rising" if late > 0 else "falling"
    else:
        ratio = (late - early) / abs(early) if early != 0 else 0
        if ratio > 0.1:
            trend = "rising"
        elif ratio < -0.1:
            trend = "falling"
        else:
            trend = "stable"

    return MetricTrajectory(
        name=key, early=early, mid=mid, late=late, trend=trend, latest=values[-1]
    )


def compute_trajectories(
    run_data: RunData,
) -> dict[str, list[MetricTrajectory]]:
    """Compute trajectories for all tracked metrics."""
    result: dict[str, list[MetricTrajectory]] = {}
    for category, keys in TRAJECTORY_METRICS.items():
        trajs = []
        for key in keys:
            t = _compute_trajectory(run_data.history, key)
            # Only include if we have any data
            if t.latest is not None:
                trajs.append(t)
        if trajs:
            result[category] = trajs
    return result


def get_reward_components(run_data: RunData) -> dict[str, float]:
    """Extract reward component breakdown from latest summary."""
    components: dict[str, float] = {}
    for key, value in run_data.summary.items():
        if key.startswith("rewards/") and isinstance(value, (int, float)):
            name = key.removeprefix("rewards/")
            components[name] = float(value)
    return dict(sorted(components.items(), key=lambda x: abs(x[1]), reverse=True))


def diagnose_run(run_data: RunData) -> list[Diagnostic]:
    """Run automated health checks on a run."""
    diagnostics: list[Diagnostic] = []
    history = run_data.history
    summary = run_data.summary

    # --- Entropy explosion ---
    entropy_vals = _get_history_values(history, "training/entropy")
    if len(entropy_vals) >= 10:
        early_entropy = sum(entropy_vals[: len(entropy_vals) // 5]) / (
            len(entropy_vals) // 5
        )
        late_entropy = sum(entropy_vals[-len(entropy_vals) // 5 :]) / (
            len(entropy_vals) // 5
        )
        if late_entropy > early_entropy * 1.5 and late_entropy > 1.0:
            diagnostics.append(
                Diagnostic(
                    severity="critical",
                    category="entropy",
                    message="Entropy explosion detected — entropy is increasing over training",
                    evidence=f"Early avg: {early_entropy:.3f}, Late avg: {late_entropy:.3f} "
                    f"({late_entropy / early_entropy:.1f}x increase)",
                )
            )
        elif late_entropy > early_entropy and late_entropy > 5.0:
            diagnostics.append(
                Diagnostic(
                    severity="critical",
                    category="entropy",
                    message="Entropy explosion — policy is nearly random",
                    evidence=f"Early avg: {early_entropy:.3f}, Late avg: {late_entropy:.3f} "
                    f"(still rising, well above convergence range)",
                )
            )
        elif late_entropy > 3.0:
            diagnostics.append(
                Diagnostic(
                    severity="warning",
                    category="entropy",
                    message="Entropy remains very high",
                    evidence=f"Late avg entropy: {late_entropy:.3f}",
                )
            )

    # --- Gradient instability ---
    grad_vals = _get_history_values(history, "training/grad_norm")
    if len(grad_vals) >= 10:
        late_grads = grad_vals[-len(grad_vals) // 5 :]
        avg_grad = sum(late_grads) / len(late_grads)
        max_grad = max(late_grads)
        max_grad_norm = run_data.config.get("training", {}).get("max_grad_norm", 10.0)
        if avg_grad > max_grad_norm * 0.9:
            diagnostics.append(
                Diagnostic(
                    severity="critical",
                    category="gradient",
                    message="Gradient norm consistently near clip threshold — training is unstable",
                    evidence=f"Late avg grad_norm: {avg_grad:.1f}, "
                    f"max: {max_grad:.1f}, clip threshold: {max_grad_norm}",
                )
            )
        elif max_grad > max_grad_norm * 5:
            diagnostics.append(
                Diagnostic(
                    severity="warning",
                    category="gradient",
                    message="Gradient norm spikes detected",
                    evidence=f"Max grad_norm: {max_grad:.1f}, "
                    f"avg: {avg_grad:.1f}, clip threshold: {max_grad_norm}",
                )
            )

    # --- Curriculum stagnation ---
    stage_vals = _get_history_values(history, "curriculum/stage")
    if len(stage_vals) >= 20:
        if stage_vals[-1] == stage_vals[0] and len(stage_vals) > 50:
            diagnostics.append(
                Diagnostic(
                    severity="warning",
                    category="curriculum",
                    message=f"No curriculum advancement — stuck at stage {int(stage_vals[-1])} "
                    f"for entire run ({len(history)} batches)",
                    evidence=f"Stage at start: {int(stage_vals[0])}, "
                    f"Stage at end: {int(stage_vals[-1])}",
                )
            )

    # --- Fall rate ---
    fell_vals = _get_history_values(history, "batch/fell_fraction")
    if len(fell_vals) >= 10:
        late_fell = fell_vals[-len(fell_vals) // 5 :]
        avg_fell = sum(late_fell) / len(late_fell)
        if avg_fell > 0.95:
            diagnostics.append(
                Diagnostic(
                    severity="critical",
                    category="stability",
                    message="Robot falls in nearly every episode",
                    evidence=f"Late fell_fraction avg: {avg_fell:.3f} "
                    f"({avg_fell * 100:.0f}% of episodes)",
                )
            )
        elif avg_fell > 0.7:
            diagnostics.append(
                Diagnostic(
                    severity="warning",
                    category="stability",
                    message="Robot falls frequently",
                    evidence=f"Late fell_fraction avg: {avg_fell:.3f} "
                    f"({avg_fell * 100:.0f}% of episodes)",
                )
            )

    # --- Reward dominance ---
    reward_components = get_reward_components(run_data)
    if reward_components:
        total = sum(abs(v) for v in reward_components.values())
        if total > 0:
            for name, value in reward_components.items():
                frac = abs(value) / total
                if frac > 0.8:
                    diagnostics.append(
                        Diagnostic(
                            severity="warning",
                            category="reward",
                            message=f"Reward dominated by '{name}' ({frac:.0%} of total magnitude)",
                            evidence=f"{name}: {value:.4f}, total magnitude: {total:.4f}",
                        )
                    )
                    break  # Only report top dominator

    # --- Policy collapse (action std) ---
    # Check for action std approaching 0 or exploding
    for key, value in summary.items():
        if key.startswith("policy/std_") and isinstance(value, (int, float)):
            if value < 0.01:
                diagnostics.append(
                    Diagnostic(
                        severity="warning",
                        category="policy",
                        message=f"Policy std near zero for {key.removeprefix('policy/')} — possible collapse",
                        evidence=f"{key}: {value:.6f}",
                    )
                )
            elif value > 5.0:
                diagnostics.append(
                    Diagnostic(
                        severity="warning",
                        category="policy",
                        message=f"Policy std very high for {key.removeprefix('policy/')} — not converging",
                        evidence=f"{key}: {value:.4f}",
                    )
                )

    # --- Value loss explosion ---
    vloss_vals = _get_history_values(history, "training/value_loss")
    if len(vloss_vals) >= 10:
        late_vloss = vloss_vals[-len(vloss_vals) // 5 :]
        early_vloss = vloss_vals[: len(vloss_vals) // 5]
        avg_late = sum(late_vloss) / len(late_vloss)
        avg_early = sum(early_vloss) / len(early_vloss)
        if avg_early > 0 and avg_late > avg_early * 10:
            diagnostics.append(
                Diagnostic(
                    severity="warning",
                    category="training",
                    message="Value loss increased significantly over training",
                    evidence=f"Early avg: {avg_early:.4f}, Late avg: {avg_late:.4f} "
                    f"({avg_late / avg_early:.1f}x)",
                )
            )

    # --- Short episodes (not learning to survive) ---
    step_vals = _get_history_values(history, "batch/avg_steps")
    if len(step_vals) >= 10:
        late_steps = step_vals[-len(step_vals) // 5 :]
        avg_steps = sum(late_steps) / len(late_steps)
        max_steps = run_data.config.get("env", {}).get("max_episode_steps", 500)
        if avg_steps < max_steps * 0.1:
            diagnostics.append(
                Diagnostic(
                    severity="warning",
                    category="stability",
                    message="Episodes are very short — robot terminates quickly",
                    evidence=f"Late avg_steps: {avg_steps:.0f}, "
                    f"max_episode_steps: {max_steps}",
                )
            )

    # Sort: critical first, then warning, then info
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    diagnostics.sort(key=lambda d: severity_order.get(d.severity, 3))

    if not diagnostics:
        diagnostics.append(
            Diagnostic(
                severity="info",
                category="general",
                message="No issues detected",
                evidence="All health checks passed",
            )
        )

    return diagnostics


def analyze_run(run_data: RunData) -> AnalysisReport:
    """Run full analysis on a fetched run."""
    return AnalysisReport(
        run_data=run_data,
        trajectories=compute_trajectories(run_data),
        reward_components=get_reward_components(run_data),
        diagnostics=diagnose_run(run_data),
    )


# ---------------------------------------------------------------------------
# Formatting (markdown output)
# ---------------------------------------------------------------------------


def _fmt(value: float | None, precision: int = 4) -> str:
    """Format a numeric value for display."""
    if value is None:
        return "—"
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 1:
        return f"{value:.2f}"
    return f"{value:.{precision}f}"


def _trend_arrow(trend: str) -> str:
    if trend == "rising":
        return "^"
    if trend == "falling":
        return "v"
    if trend == "stable":
        return "="
    return "?"


def _duration_str(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    hours = seconds / 3600
    mins = (seconds % 3600) / 60
    return f"{hours:.0f}h {mins:.0f}m"


def format_overview(run_data: RunData) -> str:
    """Format run overview section."""
    lines = ["## Overview", ""]
    lines.append(f"- **Run**: {run_data.name} (`{run_data.run_id}`)")
    lines.append(f"- **State**: {run_data.state}")
    if run_data.url:
        lines.append(f"- **URL**: {run_data.url}")
    if run_data.tags:
        lines.append(f"- **Tags**: {', '.join(run_data.tags)}")
    if run_data.created_at:
        lines.append(f"- **Created**: {run_data.created_at}")
    if run_data.duration_seconds > 0:
        lines.append(f"- **Duration**: {_duration_str(run_data.duration_seconds)}")

    # Extract key config values
    cfg = run_data.config
    # Bot name is in tags (first tag), not typically a top-level config key
    tags = run_data.tags
    bot = cfg.get("bot", "unknown")
    # Tags are [policy, algorithm, bot] or [bot, algorithm, policy] — find the non-policy/algo one
    known_policies = {"LSTMPolicy", "TinyPolicy", "MLPPolicy"}
    known_algos = {"PPO", "REINFORCE"}
    for tag in tags:
        if tag not in known_policies and tag not in known_algos:
            bot = tag
            break
    policy = cfg.get("policy", {}).get("policy_type", "unknown")
    algorithm = cfg.get("training", {}).get("algorithm", "unknown")
    lr = cfg.get("training", {}).get("learning_rate")
    entropy = cfg.get("training", {}).get("entropy_coeff")
    batch_size = cfg.get("training", {}).get("batch_size")
    num_workers = cfg.get("training", {}).get("num_workers")

    lines.append(f"- **Bot**: {bot}")
    lines.append(f"- **Policy**: {policy} | **Algorithm**: {algorithm}")
    if lr is not None:
        lines.append(f"- **LR**: {lr} | **Entropy coeff**: {entropy}")
    if batch_size is not None:
        lines.append(f"- **Batch size**: {batch_size} | **Workers**: {num_workers}")

    # Episode/batch counts from summary
    summary = run_data.summary
    batch_count = summary.get("batch")
    episode_count = summary.get("episode")
    stage = summary.get("curriculum/stage")
    if batch_count is not None:
        lines.append(f"- **Batches**: {batch_count}")
    if episode_count is not None:
        lines.append(f"- **Episodes**: {episode_count}")
    if stage is not None:
        progress = summary.get("curriculum/overall_progress", 0)
        lines.append(f"- **Curriculum stage**: {int(stage)} (overall progress: {progress:.1%})")

    return "\n".join(lines)


def format_trajectories(trajectories: dict[str, list[MetricTrajectory]]) -> str:
    """Format metric trajectories section."""
    lines = ["## Metric Trajectories", ""]

    for category, trajs in trajectories.items():
        lines.append(f"### {category.title()}")
        lines.append("")
        lines.append(f"{'Metric':<45} {'Early':>10} {'Mid':>10} {'Late':>10} {'Trend':>6}")
        lines.append(f"{'-'*45} {'-'*10} {'-'*10} {'-'*10} {'-'*6}")
        for t in trajs:
            short_name = t.name.split("/", 1)[-1] if "/" in t.name else t.name
            lines.append(
                f"{short_name:<45} {_fmt(t.early):>10} {_fmt(t.mid):>10} "
                f"{_fmt(t.late):>10} {_trend_arrow(t.trend):>6}"
            )
        lines.append("")

    return "\n".join(lines)


def format_rewards(reward_components: dict[str, float]) -> str:
    """Format reward decomposition section."""
    lines = ["## Reward Decomposition", ""]

    if not reward_components:
        lines.append("No reward components found in summary.")
        return "\n".join(lines)

    total = sum(abs(v) for v in reward_components.values())
    net = sum(reward_components.values())

    lines.append(f"{'Component':<30} {'Value':>12} {'|Fraction|':>12} {'Bar'}")
    lines.append(f"{'-'*30} {'-'*12} {'-'*12} {'-'*20}")

    for name, value in reward_components.items():
        frac = abs(value) / total if total > 0 else 0
        bar_len = int(frac * 20)
        bar_char = "+" if value >= 0 else "-"
        bar = bar_char * bar_len
        lines.append(f"{name:<30} {_fmt(value):>12} {frac:>11.0%} {bar}")

    lines.append(f"{'':>30} {'-'*12}")
    lines.append(f"{'NET REWARD':<30} {_fmt(net):>12}")

    return "\n".join(lines)


def format_diagnostics(diagnostics: list[Diagnostic]) -> str:
    """Format diagnostics section."""
    lines = ["## Diagnostics", ""]

    severity_icons = {"critical": "[CRITICAL]", "warning": "[WARNING]", "info": "[INFO]"}

    for d in diagnostics:
        icon = severity_icons.get(d.severity, "[???]")
        lines.append(f"**{icon} {d.category}**: {d.message}")
        lines.append(f"  Evidence: {d.evidence}")
        lines.append("")

    return "\n".join(lines)


def format_raw(summary: dict[str, Any]) -> str:
    """Format raw summary dump."""
    lines = ["## Raw Summary Values", ""]

    # Group by prefix
    groups: dict[str, list[tuple[str, Any]]] = {}
    for key, value in sorted(summary.items()):
        if key.startswith("_"):
            continue  # Skip W&B internal keys
        # Skip non-scalar values (histograms, images, etc.)
        if not isinstance(value, (int, float, str, bool)):
            continue
        prefix = key.split("/")[0] if "/" in key else "misc"
        groups.setdefault(prefix, []).append((key, value))

    for prefix, items in sorted(groups.items()):
        lines.append(f"### {prefix}")
        for key, value in items:
            if isinstance(value, float):
                lines.append(f"  {key}: {_fmt(value)}")
            else:
                lines.append(f"  {key}: {value}")
        lines.append("")

    return "\n".join(lines)


def format_csv(run_data: RunData) -> str:
    """Export history as CSV."""
    if not run_data.history:
        return "# No history data available"

    # Collect all keys that appear in history
    all_keys: set[str] = set()
    for row in run_data.history:
        for key in row:
            if key.startswith("_"):
                continue
            if isinstance(row[key], (int, float)):
                all_keys.add(key)

    sorted_keys = sorted(all_keys)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(sorted_keys)
    for row in run_data.history:
        writer.writerow([row.get(k, "") for k in sorted_keys])

    return output.getvalue()


def format_full_report(report: AnalysisReport) -> str:
    """Format the complete analysis report."""
    sections = [
        f"# W&B Run Analysis: {report.run_data.name}",
        "",
        format_overview(report.run_data),
        "",
        format_trajectories(report.trajectories),
        "",
        format_rewards(report.reward_components),
        "",
        format_diagnostics(report.diagnostics),
    ]
    return "\n".join(sections)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a W&B training run",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "run",
        help="W&B run reference: run ID, full URL, or run name",
    )
    parser.add_argument("--overview", action="store_true", help="Config + status only")
    parser.add_argument(
        "--metrics", action="store_true", help="Key metric trajectories"
    )
    parser.add_argument(
        "--rewards", action="store_true", help="Reward decomposition"
    )
    parser.add_argument(
        "--diagnostics", action="store_true", help="Automated health checks"
    )
    parser.add_argument("--raw", action="store_true", help="Dump raw latest values")
    parser.add_argument(
        "--csv", action="store_true", help="Export timeseries as CSV"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=500,
        help="Number of history samples to fetch (default: 500)",
    )

    args = parser.parse_args()

    # Determine which sections to show
    show_all = not any(
        [args.overview, args.metrics, args.rewards, args.diagnostics, args.raw, args.csv]
    )

    print(f"Fetching run: {args.run}...", file=sys.stderr)
    run_data = fetch_run(args.run, samples=args.samples)
    print(
        f"Fetched {len(run_data.history)} history rows for {run_data.name}",
        file=sys.stderr,
    )

    if args.csv:
        print(format_csv(run_data))
        return

    if show_all:
        report = analyze_run(run_data)
        print(format_full_report(report))
        return

    if args.overview:
        print(format_overview(run_data))
    if args.metrics:
        trajectories = compute_trajectories(run_data)
        print(format_trajectories(trajectories))
    if args.rewards:
        components = get_reward_components(run_data)
        print(format_rewards(components))
    if args.diagnostics:
        diagnostics = diagnose_run(run_data)
        print(format_diagnostics(diagnostics))
    if args.raw:
        print(format_raw(run_data.summary))


if __name__ == "__main__":
    main()
