"""
Model checkpointing with wandb artifact lineage.

Saves/loads training state at curriculum milestones so future runs
can fast-forward past mastered stages. Wandb artifacts track which
run produced each checkpoint and which run consumed it.
"""

import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import torch

import wandb


def _get_git_sha() -> str:
    """Get short git SHA, or 'unknown' if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def resolve_resume_ref(ref: str) -> str:
    """
    Resolve a resume reference to an actual path or artifact ref.

    Handles the special value "latest" by finding the most recent .pt file
    in the checkpoints/ directory.

    Args:
        ref: A local .pt path, wandb artifact ref, or "latest"

    Returns:
        Resolved path or artifact ref
    """
    if ref == "latest":
        ckpt_dir = Path("checkpoints")
        if not ckpt_dir.exists():
            raise FileNotFoundError("No checkpoints/ directory found")
        pt_files = sorted(ckpt_dir.glob("*.pt"), key=os.path.getmtime)
        if not pt_files:
            raise FileNotFoundError("No .pt files found in checkpoints/")
        return str(pt_files[-1])
    return ref


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

    Returns:
        Local path to the saved checkpoint file
    """
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
        "git_sha": _get_git_sha(),
    }

    # Save locally
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
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
