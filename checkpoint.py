"""
Model checkpointing with wandb artifact lineage.

Saves/loads training state at curriculum milestones so future runs
can fast-forward past mastered stages. Wandb artifacts track which
run produced each checkpoint and which run consumed it.
"""

import os
from datetime import UTC, datetime
from pathlib import Path

import torch

import wandb
from git_utils import get_git_sha


def build_policy(ckpt_config):
    """Reconstruct a policy network from a checkpoint's embedded config dict.

    Delegates to Pipeline.from_wandb_dict() + Pipeline.build_policy() so there
    is a single source of truth for policy construction.
    """
    from pipeline import Pipeline

    pipeline = Pipeline.from_wandb_dict(ckpt_config)
    return pipeline.build_policy(device="cpu")


def resolve_resume_ref(ref: str) -> str:
    """
    Resolve a resume reference to an actual path or artifact ref.

    For "latest": searches runs/*/checkpoints/*.pt (newest by mtime).

    Also accepts a run name (e.g. "s2w-lstm-0218-1045") and looks in that
    run's checkpoints dir for the latest .pt file.

    Args:
        ref: A local .pt path, wandb artifact ref, run name, or "latest"

    Returns:
        Resolved path or artifact ref
    """
    runs_dir = Path("runs")

    if ref == "latest":
        all_pts: list[Path] = []
        if runs_dir.is_dir():
            all_pts.extend(runs_dir.glob("*/checkpoints/*.pt"))

        if not all_pts:
            raise FileNotFoundError(
                "No checkpoint files found in runs/*/checkpoints/"
            )
        # Return newest by modification time
        return str(max(all_pts, key=os.path.getmtime))

    # Check if ref is a run name (directory under runs/)
    run_dir = runs_dir / ref
    if run_dir.is_dir():
        ckpt_dir = run_dir / "checkpoints"
        pt_files = sorted(ckpt_dir.glob("*.pt"), key=os.path.getmtime)
        if not pt_files:
            raise FileNotFoundError(f"No .pt files found in {ckpt_dir}")
        return str(pt_files[-1])

    return ref


def list_checkpoints(run_dir: str | Path) -> list[dict]:
    """
    List all checkpoints in a run directory, sorted newest-first by mtime.

    Each entry has: path, filename, stage, batch, mtime.
    Stage/batch are parsed from the `_stage{N}_batch{M}.pt` pattern;
    None if the filename doesn't match.

    Args:
        run_dir: Path to the run directory (e.g. runs/s2w-lstm-0218-1045)

    Returns:
        List of checkpoint dicts, newest first.
    """
    import re

    ckpt_dir = Path(run_dir) / "checkpoints"
    if not ckpt_dir.is_dir():
        return []

    pattern = re.compile(r"_stage(\d+)_batch(\d+)\.pt$")
    results = []
    for pt in ckpt_dir.glob("*.pt"):
        m = pattern.search(pt.name)
        results.append({
            "path": str(pt),
            "filename": pt.name,
            "stage": int(m.group(1)) if m else None,
            "batch": int(m.group(2)) if m else None,
            "mtime": os.path.getmtime(pt),
        })

    results.sort(key=lambda x: x["mtime"], reverse=True)
    return results


def save_checkpoint(
    policy,
    optimizer,
    cfg,
    curriculum_stage: int,
    stage_progress: float,
    mastery_count: int,
    batch_idx: int,
    episode_count: int,
    trigger: str = "periodic",
    aliases: list[str] | None = None,
    run_dir: Path | None = None,
) -> str:
    """
    Save a training checkpoint locally and upload as wandb artifact.

    Args:
        policy: The policy network
        optimizer: The optimizer
        cfg: Config object (snapshot via to_wandb_config)
        curriculum_stage: Current curriculum stage (1-indexed)
        stage_progress: Progress within current stage [0, 1]
        mastery_count: Consecutive batches at mastery level
        batch_idx: Current batch index (for wandb x-axis continuity)
        episode_count: Total episodes collected so far
        trigger: What triggered the save ("milestone", "periodic", "final")
        aliases: Extra wandb artifact aliases (e.g. ["stage1-mastered"])
        run_dir: Run directory (saves to run_dir/checkpoints/). Required.

    Returns:
        Local path to the saved checkpoint file
    """
    if run_dir is None:
        raise ValueError("run_dir is required for save_checkpoint")

    run = wandb.run
    run_name = run.name if run else "offline"
    run_id = run.id if run else "offline"

    # Build checkpoint dict
    ckpt = {
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "curriculum_stage": curriculum_stage,
        "stage_progress": stage_progress,
        "mastery_count": mastery_count,
        "batch_idx": batch_idx,
        "episode_count": episode_count,
        "config": cfg.to_wandb_config(),
        "run_id": run_id,
        "run_name": run_name,
        "timestamp": datetime.now(UTC).isoformat(),
        "git_sha": get_git_sha(),
    }

    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{run_name}_stage{curriculum_stage}_batch{batch_idx}.pt"
    local_path = ckpt_dir / filename
    torch.save(ckpt, local_path)
    print(f"  Checkpoint saved: {local_path} (trigger: {trigger})")

    # Upload as wandb artifact
    if run and not run.disabled:
        policy_type = cfg.policy.policy_type.lower()
        artifact = wandb.Artifact(
            name=f"checkpoint-{policy_type}",
            type="model-checkpoint",
            metadata={
                "curriculum_stage": curriculum_stage,
                "stage_progress": stage_progress,
                "batch_idx": batch_idx,
                "episode_count": episode_count,
                "policy_type": cfg.policy.policy_type,
                "algorithm": cfg.training.algorithm,
                "hidden_size": cfg.policy.hidden_size,
                "trigger": trigger,
            },
        )
        artifact.add_file(str(local_path))
        artifact_aliases = ["latest"]
        if aliases:
            artifact_aliases.extend(aliases)
        run.log_artifact(artifact, aliases=artifact_aliases)

    # Log checkpoint event to wandb timeline
    if run and not run.disabled:
        run.log(
            {
                "checkpoint/saved": 1,
                "checkpoint/stage": curriculum_stage,
                "checkpoint/batch_idx": batch_idx,
            }
        )

    return str(local_path)


def validate_checkpoint_config(ckpt_config: dict, current_config: dict) -> None:
    """
    Validate that a checkpoint is compatible with the current config.

    Two categories:
        Hard keys: Must match exactly. Mismatch raises ValueError because the
            checkpoint's state_dict is structurally incompatible (wrong bot,
            wrong network shape).
        Soft keys: May differ intentionally. Logged for visibility but the
            current config always wins — the checkpoint's values for these
            are informational only.

    Args:
        ckpt_config: Config dict from the checkpoint (nested format from to_wandb_config)
        current_config: Config dict from the current run (same format)
    """
    # Hard keys: mismatch = ValueError (incompatible checkpoint)
    hard_keys = [
        ("env", "scene_path"),
        ("policy", "policy_type"),
        ("policy", "hidden_size"),
        ("policy", "image_height"),
        ("policy", "image_width"),
        ("policy", "fc_output_size"),
    ]

    for section, key in hard_keys:
        ckpt_val = ckpt_config.get(section, {}).get(key)
        curr_val = current_config.get(section, {}).get(key)
        if ckpt_val is not None and curr_val is not None and ckpt_val != curr_val:
            raise ValueError(
                f"Architecture mismatch: checkpoint has {section}/{key}={ckpt_val}, "
                f"current config has {curr_val}. Cannot resume."
            )

    # Soft keys: current config wins, log deltas for visibility
    soft_keys = [
        ("training", "learning_rate"),
        ("training", "entropy_coeff"),
        ("training", "batch_size"),
        ("curriculum", "advance_threshold"),
        ("curriculum", "advance_rate"),
        ("curriculum", "num_stages"),
        ("env", "distance_reward_scale"),
    ]

    for section, key in soft_keys:
        ckpt_val = ckpt_config.get(section, {}).get(key)
        curr_val = current_config.get(section, {}).get(key)
        if ckpt_val is not None and curr_val is not None and ckpt_val != curr_val:
            print(
                f"  Resume: {section}/{key} changed {ckpt_val} -> {curr_val} (using current config)"
            )


def load_checkpoint(resume_ref: str, current_cfg, device: str = "cpu") -> dict:
    """
    Load a checkpoint from a local path or wandb artifact reference.

    Args:
        resume_ref: Either a local .pt file path or a wandb artifact ref
            (e.g. "checkpoint-lstmpolicy:stage1-mastered" or "checkpoint-lstmpolicy:v3")
        current_cfg: Current Config object for validation
        device: Device to load tensors to

    Returns:
        Checkpoint dict with keys: policy_state_dict, optimizer_state_dict,
        curriculum_stage, stage_progress, mastery_count, batch_idx, episode_count, etc.
    """
    if os.path.isfile(resume_ref):
        # Local file
        ckpt = torch.load(resume_ref, map_location=device, weights_only=False)
    else:
        # Wandb artifact reference
        run = wandb.run
        if run is None:
            raise RuntimeError("wandb must be initialized before loading artifacts")
        artifact = run.use_artifact(resume_ref)
        artifact_dir = artifact.download()
        # Find the .pt file in the artifact
        pt_files = list(Path(artifact_dir).glob("*.pt"))
        if not pt_files:
            raise FileNotFoundError(f"No .pt file found in artifact {resume_ref}")
        ckpt = torch.load(pt_files[0], map_location=device, weights_only=False)

    # Validate config compatibility
    if "config" in ckpt:
        validate_checkpoint_config(ckpt["config"], current_cfg.to_wandb_config())

    return ckpt
