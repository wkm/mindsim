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

                if lo == hi:
                    range_str = f"{scaled_lo:+.3f}/step{unit_str}"
                else:
                    range_str = f"[{scaled_lo:+.3f}, {scaled_hi:+.3f}]/step{unit_str}"
                lines.append(f"  P{p} {p_name:<8s} {c.label:<22s} {range_str}{gate_str}")

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
            if c.gated_by:
                result[f"{prefix}/gated_by"] = c.gated_by
        for e in self.events.values():
            result[f"reward_event/{e.name}"] = e.value
        return result

    def reward_scales_for_dashboard(self) -> dict:
        """Replace the manual reward_scales dict in train.py.

        Returns the dict the dashboard uses to hide inactive metric rows.
        """
        return {
            "upright": self.components.get("upright", _zero_component).scale,
            "alive": self.components.get("alive", _zero_component).scale,
            "energy": self.components.get("energy", _zero_component).scale,
            "contact": self.components.get("contact", _zero_component).scale,
            "forward_vel": self.components.get("forward_velocity", _zero_component).scale,
            "smoothness": self.components.get("smoothness", _zero_component).scale,
            "has_walking_stage": 1.0 if self.components.get("forward_velocity", _zero_component).scale > 0 else 0.0,
            "fall_detection": 1.0 if self.components.get("alive", _zero_component).gated_by == "is_healthy" else 0.0,
        }


# Sentinel for missing components
_zero_component = RewardComponent(
    name="_zero", priority=GUARD, kind="bonus", scale=0.0,
    raw_range=(0.0, 0.0), label="",
)


# ---------------------------------------------------------------------------
# Preset builders — one per bot
# ---------------------------------------------------------------------------


def childbiped_rewards(cfg) -> RewardHierarchy:
    """Reward hierarchy for the 12-DOF child biped.

    Args:
        cfg: EnvConfig with the reward scale values.
    """
    n_actuators = 12
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
    from config import Config

    configs = {
        "childbiped": Config.for_childbiped,
        "simplebiped": Config.for_biped,
        "simple2wheeler": lambda: Config(),
        "walker2d": Config.for_walker2d,
    }

    # Filter to requested bot(s)
    requested = sys.argv[1:] or list(configs.keys())

    for name in requested:
        factory = configs.get(name)
        if not factory:
            print(f"Unknown bot: {name}")
            continue

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = factory()
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
            """Sum reward given {component_name: raw_value} overrides."""
            total = 0.0
            for c in active:
                raw = values.get(c.name)
                if raw is not None:
                    total += c.scale * raw
                # else: component contributes 0 (no movement, no contact, etc.)
            print(f"    {label:<40s} {total:+.3f}/step")
            return total

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
            print(f"    Standing > Diving: {stand:+.3f} > {dive:+.3f} [ok]")
        else:
            print(f"    Standing <= Diving: {stand:+.3f} <= {dive:+.3f} [!!]")
        if walk > stand:
            print(f"    Walking > Standing: {walk:+.3f} > {stand:+.3f} [ok]")
        else:
            print(f"    Walking <= Standing: {walk:+.3f} <= {stand:+.3f} [!!]")
        print()