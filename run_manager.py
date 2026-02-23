"""
Run directory management for MindSim training.

Each training run gets its own directory under runs/<run_name>/ with
checkpoints/ and recordings/ subdirs. Provides run discovery, metadata
persistence, and unified W&B initialization.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import wandb

log = logging.getLogger(__name__)

RUNS_DIR = Path("runs")

# Short abbreviations for bot names used in run naming
BOT_ABBREVIATIONS: dict[str, str] = {
    "simple2wheeler": "s2w",
    "simplebiped": "biped",
    "walker2d": "w2d",
    "childbiped": "child",
}

# Display names for bots
BOT_DISPLAY_NAMES: dict[str, str] = {
    "simple2wheeler": "2-Wheeler",
    "simplebiped": "Biped",
    "walker2d": "Walker2d",
    "childbiped": "Child Biped",
}


def bot_name_from_scene_path(scene_path: str) -> str:
    """Extract bot directory name from a scene path like 'bots/simple2wheeler/scene.xml'."""
    return Path(scene_path).parent.name


def bot_abbreviation(bot_name: str) -> str:
    """Get the short abbreviation for a bot name."""
    return BOT_ABBREVIATIONS.get(bot_name, bot_name[:4])


def bot_display_name(bot_name: str) -> str:
    """Get human-readable display name for a bot."""
    return BOT_DISPLAY_NAMES.get(bot_name, bot_name)


def generate_run_name(bot_name: str, policy_type: str) -> str:
    """Generate a unique run name like 's2w-lstm-0218-1045'.

    If the name already exists under runs/, appends -2, -3, etc.
    """
    abbr = bot_abbreviation(bot_name)
    policy_abbr = policy_type.lower().replace("policy", "")
    timestamp = datetime.now().strftime("%m%d-%H%M")
    base_name = f"{abbr}-{policy_abbr}-{timestamp}"

    # Check for collisions
    name = base_name
    suffix = 2
    while (RUNS_DIR / name).exists():
        name = f"{base_name}-{suffix}"
        suffix += 1

    return name


def create_run_dir(run_name: str) -> Path:
    """Create runs/<run_name>/ with checkpoints/ and recordings/ subdirs.

    Returns the run directory path.
    """
    run_dir = RUNS_DIR / run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "recordings").mkdir(parents=True, exist_ok=True)
    return run_dir


@dataclass
class RunInfo:
    """Metadata about a training run, persisted as run_info.json."""

    name: str
    bot_name: str
    policy_type: str
    algorithm: str
    scene_path: str
    status: str = "running"  # running, completed, failed
    wandb_id: str | None = None
    wandb_url: str | None = None
    git_branch: str | None = None
    git_sha: str | None = None
    created_at: str | None = None
    finished_at: str | None = None
    # Last known training state
    batch_idx: int = 0
    episode_count: int = 0
    curriculum_stage: int = 1
    best_eval_success_rate: float = 0.0
    error_message: str | None = None
    tags: list[str] = field(default_factory=list)


def save_run_info(run_dir: Path, info: RunInfo) -> None:
    """Write run_info.json to the run directory."""
    path = run_dir / "run_info.json"
    path.write_text(json.dumps(asdict(info), indent=2) + "\n")


def load_run_info(run_dir: Path) -> RunInfo | None:
    """Load run_info.json from a run directory, or None if missing/corrupt."""
    path = run_dir / "run_info.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        # Handle missing fields gracefully
        return RunInfo(
            **{k: v for k, v in data.items() if k in RunInfo.__dataclass_fields__}
        )
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def discover_local_runs() -> list[tuple[Path, RunInfo]]:
    """Scan runs/*/run_info.json and return (run_dir, info) sorted by date desc."""
    results = []
    if not RUNS_DIR.is_dir():
        return results
    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        info = load_run_info(run_dir)
        if info is not None:
            results.append((run_dir, info))
    # Sort by created_at descending (newest first)
    results.sort(key=lambda x: x[1].created_at or "", reverse=True)
    return results


def discover_wandb_runs(limit: int = 50) -> list[dict]:
    """Query W&B API for recent runs. Best-effort, returns empty on failure."""
    try:
        api = wandb.Api(timeout=5)
        runs = api.runs("mindsim", per_page=limit)
        results = []
        for run in runs:
            results.append(
                {
                    "id": run.id,
                    "name": run.name,
                    "state": run.state,
                    "created_at": run.created_at,
                    "tags": run.tags,
                    "url": run.url,
                }
            )
        return results
    except Exception:
        return []


def init_wandb_for_run(
    run_name: str,
    cfg,
    bot_name: str,
    smoketest: bool = False,
    run_notes: str | None = None,
) -> wandb.sdk.wandb_run.Run | None:
    """Initialize W&B with unified project 'mindsim' and standard tags.

    Returns the wandb run object (or None if disabled).
    """
    wandb_mode = "disabled" if smoketest else "online"
    tags = [bot_name, cfg.training.algorithm, cfg.policy.policy_type]

    wandb.init(
        project="mindsim",
        name=run_name,
        notes=run_notes,
        mode=wandb_mode,
        config=cfg.to_wandb_config(),
        tags=tags,
    )
    # Use "batch" as the x-axis instead of W&B's auto-incremented "step"
    wandb.define_metric("batch")
    wandb.define_metric("*", step_metric="batch")

    return wandb.run
