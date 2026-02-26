"""
Declarative reward hierarchy system.

Defines reward components with explicit priority, kind, bounds, and scale.
Validates dominance ordering so higher-priority rewards always outweigh
lower-priority ones — catches reward imbalance bugs (like alive_bonus=1
with forward_velocity=8) at init time rather than after hours of training.

Priority levels:
    0 = SURVIVE  (alive/standing — must dominate everything)
    1 = TASK     (distance, forward_velocity — the actual objective)
    2 = STYLE    (upright, energy, smoothness — quality of movement)
    3 = GUARD    (contact, time — small penalties for bad habits)
"""

from __future__ import annotations

import math
import warnings
from collections import defaultdict
from dataclasses import dataclass
from enum import IntEnum


class Priority(IntEnum):
    """Reward priority levels. Higher-priority (lower value) should dominate."""
    SURVIVE = 0  # Standing > everything
    TASK = 1     # Moving toward goal
    STYLE = 2    # Quality of movement
    GUARD = 3    # Small penalties for bad habits

# Module-level aliases for convenience
SURVIVE = Priority.SURVIVE
TASK = Priority.TASK
STYLE = Priority.STYLE
GUARD = Priority.GUARD


@dataclass
class RewardComponent:
    """A single per-step reward component with metadata."""

    name: str                        # e.g. "alive", "forward_velocity"
    priority: int                    # 0=SURVIVE, 1=TASK, 2=STYLE, 3=GUARD
    kind: str                        # "bonus" or "penalty"
    scale: float                     # Multiplier applied to raw value
    raw_range: tuple[float, float]   # (min, max) of raw value per step
    label: str                       # Human-readable label for dashboard
    unit: str = ""                   # e.g. "m/s", "m", "" (dimensionless)
    gated_by: str | None = None      # Name of gating condition
    kernel: str = "linear"           # "linear", "gaussian", or "squared"
    kernel_sigma: float = 0.0        # Sigma for Gaussian kernel

    def compute(self, raw_value: float) -> float:
        """Apply kernel and scale to a raw value.

        - gaussian: scale * exp(-raw² / sigma²) — raw is tracking error
        - squared: scale * raw — raw is pre-computed negative squared quantity
        - linear: scale * raw — direct scaling
        """
        if self.kernel == "gaussian" and self.kernel_sigma > 0:
            return self.scale * math.exp(-(raw_value ** 2) / (self.kernel_sigma ** 2))
        return self.scale * raw_value

    @property
    def max_per_step(self) -> float:
        """Maximum absolute contribution per step."""
        lo, hi = self.raw_range
        return self.scale * max(abs(lo), abs(hi))

    @property
    def range_per_step(self) -> tuple[float, float]:
        """(min, max) scaled contribution per step."""
        lo, hi = self.raw_range
        if self.kind == "bonus":
            return (self.scale * lo, self.scale * hi)
        else:
            # Penalties: scale * raw, where raw is negative
            return (self.scale * hi, self.scale * lo)  # hi is closer to 0


@dataclass
class BonusEvent:
    """One-time reward event (success, failure, instability)."""

    name: str
    value: float      # Positive or negative
    label: str


class RewardHierarchy:
    """Declarative reward structure with dominance validation."""

    def __init__(self, components: list[RewardComponent], events: list[BonusEvent] | None = None):
        self.components = {c.name: c for c in components}
        self.events = {e.name: e for e in (events or [])}
        self._validation_warnings: list[str] = []
        self._validate()

    def _validate(self):
        """Check that higher-priority components dominate lower-priority ones.

        The rule: the minimum possible per-step contribution of all components
        at priority P must exceed the maximum possible per-step contribution
        of all components at priority P+1 and below combined.
        """
        by_priority: dict[int, list[RewardComponent]] = defaultdict(list)
        for c in self.components.values():
            if c.scale > 0:
                by_priority[c.priority].append(c)

        priorities = sorted(by_priority.keys())
        self._validation_warnings = []

        for i, p in enumerate(priorities[:-1]):
            # Minimum contribution from this priority level
            higher_min = 0.0
            for c in by_priority[p]:
                lo, hi = c.raw_range
                if c.kind == "bonus":
                    higher_min += c.scale * lo
                else:
                    higher_min += c.scale * hi  # hi is closer to 0

            # Maximum contribution from all lower priority levels
            lower_max = 0.0
            for q in priorities[i + 1:]:
                for c in by_priority[q]:
                    lower_max += c.max_per_step

            if higher_min <= lower_max:
                p_name = Priority(p).name
                msg = (
                    f"Priority {p} ({p_name}) min ({higher_min:.3f}) <= "
                    f"lower priority max ({lower_max:.3f}). "
                    f"Lower-priority rewards may dominate."
                )
                self._validation_warnings.append(msg)
                warnings.warn(msg, stacklevel=3)

    def get_scale(self, name: str) -> float:
        return self.components[name].scale

    def active_components(self) -> list[RewardComponent]:
        """Components with scale > 0."""
        return [c for c in self.components.values() if c.scale > 0]

    def component_names(self) -> list[str]:
        """All component names (matching reward_<name> keys in info dicts)."""
        return list(self.components.keys())

    def reward_component_keys(self) -> list[str]:
        """Prefixed keys for collection.py / train.py (e.g. 'reward_alive')."""
        return [f"reward_{name}" for name in self.component_names()]

    def summary_table(self) -> str:
        """Pretty-print for logging/CLI."""
        lines = []
        by_priority: dict[int, list[RewardComponent]] = defaultdict(list)
        for c in self.components.values():
            by_priority[c.priority].append(c)

        for p in sorted(by_priority.keys()):
            p_name = Priority(p).name
            for c in by_priority[p]:
                if c.scale == 0:
                    continue
                lo, hi = c.raw_range
                scaled_lo = c.scale * lo
                scaled_hi = c.scale * hi
                gate_str = f"  (gated: {c.gated_by})" if c.gated_by else ""
                unit_str = f" {c.unit}" if c.unit else ""

                # Kernel annotation
                if c.kernel == "gaussian":
                    kernel_str = f"  [gauss σ={c.kernel_sigma}]"
                elif c.kernel == "squared":
                    kernel_str = "  [sq]"
                else:
                    kernel_str = ""

                if lo == hi:
                    range_str = f"{scaled_lo:+.3f}/step{unit_str}"
                else:
                    range_str = f"[{scaled_lo:+.3f}, {scaled_hi:+.3f}]/step{unit_str}"
                lines.append(f"  P{p} {p_name:<8s} {c.label:<22s} {range_str}{gate_str}{kernel_str}")

        return "\n".join(lines)

    def dominance_check(self) -> str:
        """Human-readable dominance check results."""
        lines = []
        by_priority: dict[int, list[RewardComponent]] = defaultdict(list)
        for c in self.components.values():
            if c.scale > 0:
                by_priority[c.priority].append(c)

        priorities = sorted(by_priority.keys())

        for i, p in enumerate(priorities[:-1]):
            next_p = priorities[i + 1]
            p_name = Priority(p).name
            next_name = Priority(next_p).name

            higher_min = 0.0
            for c in by_priority[p]:
                lo, hi = c.raw_range
                if c.kind == "bonus":
                    higher_min += c.scale * lo
                else:
                    higher_min += c.scale * hi

            lower_max = 0.0
            for q in priorities[i + 1:]:
                for c in by_priority[q]:
                    lower_max += c.max_per_step

            ok = higher_min > lower_max
            symbol = "YES" if ok else "NO "
            marker = "ok" if ok else "!!"
            lines.append(
                f"  P{p} ({p_name}) min ({higher_min:.3f}) > "
                f"P{next_p}+ ({next_name}+) max ({lower_max:.3f})? "
                f"{symbol} [{marker}]"
            )

        return "\n".join(lines)

    def to_wandb_config(self) -> dict:
        """Flat dict for W&B config logging."""
        result = {}
        for c in self.components.values():
            prefix = f"reward/{c.name}"
            result[f"{prefix}/priority"] = c.priority
            result[f"{prefix}/kind"] = c.kind
            result[f"{prefix}/scale"] = c.scale
            result[f"{prefix}/raw_min"] = c.raw_range[0]
            result[f"{prefix}/raw_max"] = c.raw_range[1]
            result[f"{prefix}/kernel"] = c.kernel
            if c.kernel == "gaussian":
                result[f"{prefix}/kernel_sigma"] = c.kernel_sigma
            if c.gated_by:
                result[f"{prefix}/gated_by"] = c.gated_by
        for e in self.events.values():
            result[f"reward_event/{e.name}"] = e.value
        return result

    def reward_scales_for_dashboard(self) -> dict:
        """Dict the dashboard uses to show/hide inactive metric rows.

        Keys are component names (e.g. "alive", "orientation") plus
        convenience flags ("has_walking_stage", "fall_detection").
        """
        result = {c.name: c.scale for c in self.components.values()}
        # Convenience flags for dashboard gating
        result["has_walking_stage"] = float(
            result.get("forward_velocity", 0) > 0
            or result.get("vel_tracking", 0) > 0
        )
        result["fall_detection"] = float(any(
            c.gated_by and "is_healthy" in c.gated_by
            for c in self.components.values()
        ))
        return result


# ---------------------------------------------------------------------------
# Preset builders — one per bot
# ---------------------------------------------------------------------------


def childbiped_rewards(cfg) -> RewardHierarchy:
    """RSL-RL style reward hierarchy for the 12-DOF child biped.

    Gaussian kernel velocity tracking + squared regularization penalties.
    Used with only_positive_rewards clipping.

    Args:
        cfg: EnvConfig with the reward scale values.
    """
    components = [
        # --- TASK: velocity tracking (Gaussian kernel) ---
        RewardComponent("vel_tracking", TASK, "bonus", cfg.vel_tracking_scale,
                        (0.0, 1.0), "vel tracking", unit="m/s",
                        kernel="gaussian", kernel_sigma=cfg.vel_tracking_sigma),
        # --- SURVIVE: stay alive and upright ---
        RewardComponent("alive", SURVIVE, "bonus", cfg.alive_bonus,
                        (0.0, 1.0), "alive"),
        RewardComponent("orientation", SURVIVE, "penalty", cfg.orientation_scale,
                        (-2.0, 0.0), "orientation", kernel="squared"),
        RewardComponent("base_height", SURVIVE, "penalty", cfg.base_height_scale,
                        (-0.25, 0.0), "base height", unit="m", kernel="squared"),
        # --- STYLE: smooth, efficient movement ---
        RewardComponent("z_velocity", STYLE, "penalty", cfg.z_velocity_scale,
                        (-4.0, 0.0), "z velocity", unit="m/s", kernel="squared"),
        RewardComponent("ang_vel_xy", STYLE, "penalty", cfg.ang_vel_xy_scale,
                        (-50.0, 0.0), "ang vel xy", unit="rad/s", kernel="squared"),
        RewardComponent("action_rate", STYLE, "penalty", cfg.action_rate_scale,
                        (-48.0, 0.0), "action rate", kernel="squared"),
        # --- GUARD: small regularization ---
        RewardComponent("torques", GUARD, "penalty", cfg.torques_scale,
                        (-1e6, 0.0), "torques", kernel="squared"),
        RewardComponent("joint_acc", GUARD, "penalty", cfg.joint_acc_scale,
                        (-1e6, 0.0), "joint acc", kernel="squared"),
    ]
    return RewardHierarchy(components)


def biped_rewards(cfg) -> RewardHierarchy:
    """Reward hierarchy for the 8-joint duck biped."""
    n_actuators = 8
    components = [
        RewardComponent("alive", SURVIVE, "bonus", cfg.alive_bonus,
                        (1.0, 1.0), "alive", gated_by="is_healthy"),
        RewardComponent("distance", TASK, "bonus", cfg.distance_reward_scale,
                        (-1.0, 1.0), "distance", unit="m"),
        RewardComponent("forward_velocity", TASK, "bonus", cfg.forward_velocity_reward_scale,
                        (0.0, 1.0), "forward velocity", unit="m/s",
                        gated_by="in_walking_stage, is_healthy"),
        RewardComponent("upright", STYLE, "bonus", cfg.upright_reward_scale,
                        (0.0, 1.0), "upright"),
        RewardComponent("energy", STYLE, "penalty", cfg.energy_penalty_scale,
                        (-float(n_actuators), 0.0), "energy"),
        RewardComponent("smoothness", STYLE, "penalty", cfg.action_smoothness_scale,
                        (-4.0 * n_actuators, 0.0), "smoothness"),
        RewardComponent("exploration", TASK, "bonus", cfg.movement_bonus,
                        (0.0, 0.1), "exploration", unit="m"),
        RewardComponent("contact", GUARD, "penalty", cfg.ground_contact_penalty,
                        (-1.0, 0.0), "contact"),
        RewardComponent("time", GUARD, "penalty", cfg.time_penalty,
                        (-1.0, -1.0), "time"),
    ]
    return RewardHierarchy(components)


def wheeler_rewards(cfg) -> RewardHierarchy:
    """Reward hierarchy for the 2-wheel robot (simple2wheeler)."""
    components = [
        RewardComponent("distance", TASK, "bonus", cfg.distance_reward_scale,
                        (-1.0, 1.0), "distance", unit="m"),
        RewardComponent("exploration", TASK, "bonus", cfg.movement_bonus,
                        (0.0, 0.1), "exploration", unit="m"),
        RewardComponent("time", GUARD, "penalty", cfg.time_penalty,
                        (-1.0, -1.0), "time"),
    ]
    return RewardHierarchy(components)


def walker2d_rewards(cfg) -> RewardHierarchy:
    """Reward hierarchy for Walker2d Gymnasium-style walker."""
    n_actuators = 6
    components = [
        RewardComponent("alive", SURVIVE, "bonus", cfg.alive_bonus,
                        (1.0, 1.0), "alive", gated_by="is_healthy"),
        RewardComponent("distance", TASK, "bonus", cfg.distance_reward_scale,
                        (-1.0, 1.0), "distance", unit="m"),
        RewardComponent("forward_velocity", TASK, "bonus", cfg.forward_velocity_reward_scale,
                        (0.0, 1.0), "forward velocity", unit="m/s",
                        gated_by="in_walking_stage, is_healthy"),
        RewardComponent("upright", STYLE, "bonus", cfg.upright_reward_scale,
                        (0.0, 1.0), "upright"),
        RewardComponent("energy", STYLE, "penalty", cfg.energy_penalty_scale,
                        (-float(n_actuators), 0.0), "energy"),
        RewardComponent("exploration", TASK, "bonus", cfg.movement_bonus,
                        (0.0, 0.1), "exploration", unit="m"),
        RewardComponent("contact", GUARD, "penalty", cfg.ground_contact_penalty,
                        (-1.0, 0.0), "contact"),
        RewardComponent("time", GUARD, "penalty", cfg.time_penalty,
                        (-1.0, -1.0), "time"),
    ]
    return RewardHierarchy(components)


def build_reward_hierarchy(bot_name: str, env_config) -> RewardHierarchy:
    """Build the appropriate reward hierarchy for a bot.

    Args:
        bot_name: Bot directory name (e.g. "childbiped", "simplebiped").
        env_config: EnvConfig instance with reward scale values.
    """
    builders = {
        "childbiped": childbiped_rewards,
        "simplebiped": biped_rewards,
        "simple2wheeler": wheeler_rewards,
        "walker2d": walker2d_rewards,
    }
    builder = builders.get(bot_name, wheeler_rewards)
    return builder(env_config)


if __name__ == "__main__":
    """Print reward hierarchies for all bots.

    Two views per bot:
      1. Structure — priority tree with scaled ranges (like an NN architecture diagram)
      2. Scenarios — concrete per-step values for specific situations

    Usage:
        uv run python reward_hierarchy.py              # all bots
        uv run python reward_hierarchy.py childbiped   # one bot
    """
    import sys
    from pipeline import pipeline_for_bot

    all_bots = ["childbiped", "simplebiped", "simple2wheeler", "walker2d"]

    # Filter to requested bot(s)
    requested = sys.argv[1:] or all_bots

    for name in requested:
        if name not in all_bots:
            print(f"Unknown bot: {name}")
            continue

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = pipeline_for_bot(name)
            h = build_reward_hierarchy(name, cfg.env)

        # --- View 1: Structure ---
        print(f"\n{'=' * 60}")
        print(f"  {name} — Reward Structure")
        print(f"{'=' * 60}")
        print(h.summary_table())
        print()
        print(h.dominance_check())

        # --- View 2: Scenarios (concrete values) ---
        print(f"\n  Scenarios (per-step reward):")
        active = h.active_components()

        def scenario(label, values):
            """Sum reward given {component_name: raw_value} overrides.

            Uses component.compute() so Gaussian kernels are applied correctly.
            """
            total = 0.0
            for c in active:
                raw = values.get(c.name)
                if raw is not None:
                    total += c.compute(raw)
                # else: component contributes 0 (no movement, no contact, etc.)
            print(f"    {label:<40s} {total:+.4f}/step")
            return total

        # Detect RSL-RL mode (has vel_tracking component)
        is_rsl = "vel_tracking" in h.components

        if is_rsl:
            stand = scenario("Healthy stand (no movement)", {
                "vel_tracking": 0.5,  # error = 0 - 0.5 = -0.5
                "alive": 1.0, "orientation": 0.0, "base_height": 0.0,
                "z_velocity": 0.0, "ang_vel_xy": 0.0, "action_rate": 0.0,
                "torques": 0.0, "joint_acc": 0.0,
            })
            dive = scenario("Diving forward (unhealthy)", {
                "vel_tracking": 0.0,  # error = 0.5 - 0.5 = 0 (perfect but falling)
                "alive": 0.0, "orientation": -1.5, "base_height": -0.1,
                "z_velocity": -2.0, "ang_vel_xy": -10.0, "action_rate": -5.0,
                "torques": -1000.0, "joint_acc": -5000.0,
            })
            walk = scenario("Walking well (healthy, forward)", {
                "vel_tracking": 0.1,  # error = 0.4 - 0.5 = -0.1
                "alive": 1.0, "orientation": -0.05, "base_height": -0.001,
                "z_velocity": -0.1, "ang_vel_xy": -0.5, "action_rate": -2.0,
                "torques": -500.0, "joint_acc": -2000.0,
            })
        else:
            stand = scenario("Healthy stand (no movement)", {
                "alive": 1.0, "upright": 1.0, "time": -1.0,
            })
            dive = scenario("Diving forward (unhealthy)", {
                "distance": 0.5, "time": -1.0, "contact": -1.0,
            })
            walk = scenario("Walking well (healthy, forward)", {
                "alive": 1.0, "forward_velocity": 0.5, "distance": 0.3,
                "upright": 0.9, "energy": -3.0, "smoothness": -1.0, "time": -1.0,
            })

        print()
        if stand > dive:
            print(f"    Standing > Diving: {stand:+.4f} > {dive:+.4f} [ok]")
        else:
            print(f"    Standing <= Diving: {stand:+.4f} <= {dive:+.4f} [!!]")
        if walk > stand:
            print(f"    Walking > Standing: {walk:+.4f} > {stand:+.4f} [ok]")
        else:
            print(f"    Walking <= Standing: {walk:+.4f} <= {stand:+.4f} [!!]")
        print()