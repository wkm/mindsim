"""
Live hyperparameter tweaks via a JSON file.

The training loop checks tweaks.json each batch. When the file changes,
values are applied to the live config and logged. This lets you adjust
hyperparameters mid-run without restarting.

Usage:
    uv run python tweaks.py learning_rate=0.0005 entropy_coeff=0.1
    uv run python tweaks.py          # show current tweaks
    uv run python tweaks.py --clear   # remove tweaks file
"""

from __future__ import annotations

import json
import os
import sys

TWEAKS_PATH = "tweaks.json"

# Whitelist: param_name -> (config_section, field_name, type)
TWEAKABLE = {
    # Training params
    "learning_rate": ("training", "learning_rate", float),
    "entropy_coeff": ("training", "entropy_coeff", float),
    "clip_epsilon": ("training", "clip_epsilon", float),
    "ppo_epochs": ("training", "ppo_epochs", int),
    "gamma": ("training", "gamma", float),
    "gae_lambda": ("training", "gae_lambda", float),
    "value_coeff": ("training", "value_coeff", float),
    "batch_size": ("training", "batch_size", int),
    "mastery_threshold": ("training", "mastery_threshold", float),
    "mastery_batches": ("training", "mastery_batches", int),
    # Curriculum params
    "advance_threshold": ("curriculum", "advance_threshold", float),
    "advance_rate": ("curriculum", "advance_rate", float),
    # Env params
    "max_episode_steps": ("env", "max_episode_steps", int),
    "max_episode_steps_final": ("env", "max_episode_steps_final", int),
    "time_penalty": ("env", "time_penalty", float),
    "distance_reward_scale": ("env", "distance_reward_scale", float),
    "patience_window": ("env", "patience_window", int),
    "patience_min_delta": ("env", "patience_min_delta", float),
}

# Module-level mtime cache for detecting file changes
_last_mtime = 0.0


def load_tweaks(path=TWEAKS_PATH):
    """
    Check tweaks file and return changes if it was modified.

    Returns:
        dict of {param_name: value} if file changed since last check,
        None if unchanged or file doesn't exist.
    """
    global _last_mtime

    try:
        mtime = os.path.getmtime(path)
    except OSError:
        _last_mtime = 0.0
        return None

    if mtime <= _last_mtime:
        return None

    _last_mtime = mtime

    with open(path) as f:
        raw = json.load(f)

    # Validate and cast types
    tweaks = {}
    for name, value in raw.items():
        if name not in TWEAKABLE:
            print(f"  tweak WARNING: unknown param '{name}', skipping")
            continue
        _, _, typ = TWEAKABLE[name]
        try:
            tweaks[name] = typ(value)
        except (ValueError, TypeError) as e:
            print(f"  tweak WARNING: bad value for '{name}': {e}")

    return tweaks if tweaks else None


def apply_tweaks(cfg, optimizer, env, tweaks):
    """
    Apply tweak values to live config, optimizer, and env.

    Args:
        cfg: Config dataclass
        optimizer: torch optimizer (for LR updates)
        env: TrainingEnv instance (for env param updates)
        tweaks: dict from load_tweaks()

    Returns:
        List of (param_name, old_value, new_value) for each applied change.
    """
    changes = []

    for name, new_value in tweaks.items():
        section_name, field_name, _ = TWEAKABLE[name]
        section = getattr(cfg, section_name)
        old_value = getattr(section, field_name)

        if old_value == new_value:
            continue

        # Apply to config
        setattr(section, field_name, new_value)
        changes.append((name, old_value, new_value))

        # Side effects for specific params
        if name == "learning_rate":
            for pg in optimizer.param_groups:
                pg["lr"] = new_value

        elif name == "max_episode_steps":
            env.update_episode_limits(base=new_value)

        elif name == "max_episode_steps_final":
            env.update_episode_limits(final=new_value)

        elif name == "time_penalty":
            env.time_penalty = new_value

        elif name == "distance_reward_scale":
            env.distance_reward_scale = new_value

        elif name == "patience_window":
            env.patience_window = new_value
            if new_value > 0:
                from collections import deque

                env._distance_deltas = deque(maxlen=new_value)
            else:
                env._distance_deltas = None

        elif name == "patience_min_delta":
            env.patience_min_delta = new_value

    return changes


def _cli():
    """CLI entrypoint: show, set, or clear tweaks."""
    args = sys.argv[1:]

    if not args:
        # Show current tweaks
        if not os.path.exists(TWEAKS_PATH):
            print("No tweaks file.")
            return
        with open(TWEAKS_PATH) as f:
            data = json.load(f)
        if not data:
            print("Tweaks file is empty.")
        else:
            print("Current tweaks:")
            for k, v in sorted(data.items()):
                marker = "" if k in TWEAKABLE else " (UNKNOWN)"
                print(f"  {k} = {v}{marker}")
        return

    if args == ["--clear"]:
        try:
            os.remove(TWEAKS_PATH)
            print("Tweaks file removed.")
        except FileNotFoundError:
            print("No tweaks file to remove.")
        return

    # Parse key=value pairs
    tweaks = {}
    for arg in args:
        if "=" not in arg:
            print(f"Error: expected key=value, got '{arg}'")
            sys.exit(1)
        key, val_str = arg.split("=", 1)
        if key not in TWEAKABLE:
            print(f"Error: unknown param '{key}'")
            print(f"Tweakable params: {', '.join(sorted(TWEAKABLE))}")
            sys.exit(1)
        _, _, typ = TWEAKABLE[key]
        try:
            tweaks[key] = typ(val_str)
        except ValueError:
            print(f"Error: cannot convert '{val_str}' to {typ.__name__} for '{key}'")
            sys.exit(1)

    # Merge with existing file
    existing = {}
    if os.path.exists(TWEAKS_PATH):
        with open(TWEAKS_PATH) as f:
            existing = json.load(f)

    existing.update(tweaks)

    with open(TWEAKS_PATH, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"Wrote {TWEAKS_PATH}:")
    for k, v in sorted(existing.items()):
        print(f"  {k} = {v}")


if __name__ == "__main__":
    _cli()
