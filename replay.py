"""
Replay command: view or regenerate Rerun recordings for a training run.

Two modes:
  Download (default): fetch existing .rrd recordings from W&B — fast, shows
      what was actually captured during training.
  Regenerate (--regenerate): re-run episodes from checkpoints at specific
      batch numbers with consistent seeding — slower, but lets you pick
      exact batches and compare with identical target placement.

Usage:
    uv run mjpython main.py replay <run_name>                              # download all
    uv run mjpython main.py replay <run_name> --batches 500,1000,2000      # download nearest
    uv run mjpython main.py replay <run_name> --batches 500,1000 --regenerate  # re-run from checkpoints
"""

from __future__ import annotations

import re
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import rerun as rr
import torch

import rerun_logger
from collection import collect_episode, log_episode_value_trace
from config import EnvConfig
from play import build_policy
from training_blueprint import create_training_blueprint
from training_env import TrainingEnv

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class RecordingInfo:
    """Metadata about a discovered .rrd recording on W&B."""

    batch_idx: int | None  # None if metadata predates batch_idx tracking
    episode: int
    artifact: object  # wandb Artifact ref
    path: str | None = None  # local path after download


@dataclass
class CheckpointInfo:
    """Metadata about a discovered checkpoint."""

    batch_idx: int
    source: str  # "local" or "wandb"
    path: str | None = None  # local path (set after download for wandb)
    artifact: object | None = None  # wandb artifact ref (for lazy download)


# ---------------------------------------------------------------------------
# W&B run lookup (shared by both modes)
# ---------------------------------------------------------------------------


def _find_wandb_run(run_ref: str):
    """Find a W&B run by display name. Returns (api, run) or (None, None)."""
    try:
        import wandb

        api = wandb.Api(timeout=10)
        runs = api.runs("mindsim", filters={"display_name": run_ref})
        if runs:
            return api, runs[0]
    except Exception as e:
        print(f"  W&B lookup failed: {e}")
    return None, None


# ---------------------------------------------------------------------------
# Download mode: fetch existing .rrd artifacts from W&B
# ---------------------------------------------------------------------------


def discover_recordings(run_ref: str) -> list[RecordingInfo]:
    """Find all rerun recordings for a run.

    Search order:
    1. Local: runs/<run_ref>/recordings/*.rrd (already on disk)
    2. W&B run.logged_artifacts() (fast API, but can silently skip artifacts)
    3. W&B artifact collection scan (slower but robust fallback)

    Returns list sorted by batch_idx (or episode if batch_idx unavailable).
    """
    recordings: list[RecordingInfo] = []
    seen_episodes: set[int] = set()

    # 1. Local recordings
    local_dir = Path("runs") / run_ref / "recordings"
    if local_dir.is_dir():
        for rrd_file in sorted(local_dir.glob("*.rrd")):
            episode = _parse_episode_from_filename(rrd_file.name)
            if episode is not None:
                recordings.append(
                    RecordingInfo(
                        batch_idx=None,
                        episode=episode,
                        artifact=None,
                        path=str(rrd_file),
                    )
                )
                seen_episodes.add(episode)

    # 2. W&B: try run.logged_artifacts() first
    api, wandb_run = _find_wandb_run(run_ref)
    if wandb_run is not None:
        try:
            for art in wandb_run.logged_artifacts():
                if art.type == "rerun-recording":
                    episode = art.metadata.get("episode", 0)
                    if episode not in seen_episodes:
                        recordings.append(
                            RecordingInfo(
                                batch_idx=art.metadata.get("batch_idx"),
                                episode=episode,
                                artifact=art,
                            )
                        )
                        seen_episodes.add(episode)
        except Exception:
            pass

        # 3. Fallback: scan artifact collections by type, match by run_id
        #    run.logged_artifacts() can silently return empty due to a
        #    wandb SDK bug (silent None skip in _convert). This fallback
        #    is slower but reliable.
        if not any(r.artifact is not None for r in recordings):
            run_id = wandb_run.id
            entity = wandb_run.entity
            project = wandb_run.project
            try:
                art_type = api.artifact_type(
                    "rerun-recording", project=f"{entity}/{project}"
                )
                for coll in art_type.collections():
                    for version in coll.versions():
                        if version.metadata.get("run_id") == run_id:
                            episode = version.metadata.get("episode", 0)
                            if episode not in seen_episodes:
                                recordings.append(
                                    RecordingInfo(
                                        batch_idx=version.metadata.get("batch_idx"),
                                        episode=episode,
                                        artifact=version,
                                    )
                                )
                                seen_episodes.add(episode)
                            break  # Only need one version per collection
            except Exception as e:
                print(f"  W&B artifact scan failed: {e}")

    # Sort by batch_idx when available, fall back to episode number
    recordings.sort(
        key=lambda r: (r.batch_idx if r.batch_idx is not None else r.episode)
    )
    return recordings


def _parse_episode_from_filename(filename: str) -> int | None:
    """Extract episode number from recording filename like 'episode_00064.rrd'."""
    m = re.search(r"episode_(\d+)\.rrd$", filename)
    if m:
        return int(m.group(1))
    return None


def _find_closest_recordings(
    available: list[RecordingInfo], requested_batches: list[int]
) -> list[RecordingInfo]:
    """For each requested batch, find the recording with the closest batch_idx.

    Only considers recordings that have batch_idx metadata.
    """
    with_batch = [r for r in available if r.batch_idx is not None]
    if not with_batch:
        return []

    result: list[RecordingInfo] = []
    seen: set[int] = set()

    for req in requested_batches:
        best = min(with_batch, key=lambda r: abs(r.batch_idx - req))
        key = best.batch_idx if best.batch_idx is not None else best.episode
        if key not in seen:
            result.append(best)
            seen.add(key)

    result.sort(key=lambda r: r.batch_idx if r.batch_idx is not None else r.episode)
    return result


def _ensure_recording_local(rec: RecordingInfo, output_dir: Path) -> str:
    """Ensure a recording is available locally. Downloads from W&B if needed."""
    if rec.path and Path(rec.path).exists():
        return rec.path

    if rec.artifact is None:
        raise FileNotFoundError(f"No artifact or local path for episode {rec.episode}")

    artifact_dir = rec.artifact.download()
    rrd_files = list(Path(artifact_dir).glob("*.rrd"))
    if not rrd_files:
        raise FileNotFoundError(f"No .rrd in artifact for episode {rec.episode}")

    # Symlink into output dir with a meaningful name to avoid doubling disk usage
    src = rrd_files[0]
    if rec.batch_idx is not None:
        dest = output_dir / f"batch_{rec.batch_idx}.rrd"
    else:
        dest = output_dir / f"episode_{rec.episode}.rrd"

    if not dest.exists():
        dest.symlink_to(src.resolve())

    rec.path = str(dest)
    return rec.path


def _open_in_rerun(rrd_paths: list[str]) -> None:
    """Open .rrd files in the Rerun viewer (non-blocking)."""
    if not rrd_paths:
        return

    import subprocess

    subprocess.Popen(
        ["rerun"] + rrd_paths,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def run_download(
    run_ref: str,
    batches: list[int] | None = None,
    last_n: int | None = None,
) -> None:
    """Download existing recordings from W&B and open in Rerun.

    Args:
        run_ref: Run name
        batches: If set, find recordings closest to these batch numbers
        last_n: If set (and batches is None), download the N most recent
    """
    print(f"\nDiscovering recordings for {run_ref}...")
    available = discover_recordings(run_ref)

    if not available:
        print(f"  No recordings found on W&B for '{run_ref}'.")
        return

    # Summarize
    with_batch = [r for r in available if r.batch_idx is not None]
    if with_batch:
        batch_range = f"batches {with_batch[0].batch_idx}-{with_batch[-1].batch_idx}"
    else:
        batch_range = "no batch metadata (older run)"
    print(f"  Found {len(available)} recordings ({batch_range})")

    # Select which recordings to download
    if batches:
        selected = _find_closest_recordings(available, batches)
        if not selected:
            print("  No recordings have batch metadata — try --regenerate instead.")
            return

        print("\nMatching requested batches:")
        for req in batches:
            best = min(
                [r for r in available if r.batch_idx is not None],
                key=lambda r: abs(r.batch_idx - req),
                default=None,
            )
            if best:
                exact = (
                    "exact"
                    if best.batch_idx == req
                    else f"closest: batch {best.batch_idx}"
                )
                print(
                    f"  batch {req:>5d} -> recording at batch {best.batch_idx:>5d} ({exact})"
                )
    elif last_n:
        selected = available[-last_n:]
        print(f"\nDownloading last {len(selected)} recordings...")
    else:
        selected = available
        print(f"\nDownloading all {len(selected)} recordings...")

    # Create output dir and download
    output_dir = Path(f"replay_{run_ref}")
    output_dir.mkdir(exist_ok=True)

    rrd_paths = []
    for i, rec in enumerate(selected):
        label = (
            f"batch {rec.batch_idx}"
            if rec.batch_idx is not None
            else f"episode {rec.episode}"
        )
        print(f"  [{i + 1}/{len(selected)}] {label}...", end=" ", flush=True)
        path = _ensure_recording_local(rec, output_dir)
        rrd_paths.append(path)
        print(f"-> {Path(path).name}")

    print(f"\nRecordings saved to {output_dir}/")

    # Open in Rerun
    _open_in_rerun(rrd_paths)
    print(f"Rerun viewer launched with {len(rrd_paths)} recordings.")


# ---------------------------------------------------------------------------
# Regenerate mode: re-run episodes from checkpoints
# ---------------------------------------------------------------------------


def discover_checkpoints(run_ref: str) -> list[CheckpointInfo]:
    """Find all available checkpoints for a run.

    Search order:
    1. Local: runs/<run_ref>/checkpoints/*.pt — parse batch from filename
    2. W&B: query wandb.Api() for the run, list model-checkpoint artifacts
    """
    checkpoints: list[CheckpointInfo] = []
    seen_batches: set[int] = set()

    # 1. Local checkpoints
    local_dir = Path("runs") / run_ref / "checkpoints"
    if local_dir.is_dir():
        for pt_file in sorted(local_dir.glob("*.pt")):
            batch_idx = _parse_batch_from_filename(pt_file.name)
            if batch_idx is not None and batch_idx not in seen_batches:
                checkpoints.append(
                    CheckpointInfo(
                        batch_idx=batch_idx,
                        source="local",
                        path=str(pt_file),
                    )
                )
                seen_batches.add(batch_idx)

    # 2. W&B checkpoints
    _, wandb_run = _find_wandb_run(run_ref)
    if wandb_run is not None:
        for art in wandb_run.logged_artifacts():
            if art.type == "model-checkpoint":
                batch_idx = art.metadata.get("batch_idx")
                if batch_idx is not None and batch_idx not in seen_batches:
                    checkpoints.append(
                        CheckpointInfo(
                            batch_idx=batch_idx,
                            source="wandb",
                            artifact=art,
                        )
                    )
                    seen_batches.add(batch_idx)

    checkpoints.sort(key=lambda c: c.batch_idx)
    return checkpoints


def _parse_batch_from_filename(filename: str) -> int | None:
    """Extract batch number from checkpoint filename like '*_batch500.pt'."""
    m = re.search(r"_batch(\d+)\.pt$", filename)
    if m:
        return int(m.group(1))
    return None


def find_closest_checkpoints(
    available: list[CheckpointInfo], requested_batches: list[int]
) -> list[CheckpointInfo]:
    """For each requested batch, find the closest available checkpoint."""
    if not available:
        return []

    result: list[CheckpointInfo] = []
    seen: set[int] = set()

    for req in requested_batches:
        best = min(available, key=lambda c: abs(c.batch_idx - req))
        if best.batch_idx not in seen:
            result.append(best)
            seen.add(best.batch_idx)

    result.sort(key=lambda c: c.batch_idx)
    return result


def _ensure_downloaded(ckpt: CheckpointInfo) -> str:
    """Ensure checkpoint is available locally. Downloads from W&B if needed."""
    if ckpt.path and Path(ckpt.path).exists():
        return ckpt.path

    if ckpt.source == "wandb" and ckpt.artifact is not None:
        artifact_dir = ckpt.artifact.download()
        pt_files = list(Path(artifact_dir).glob("*.pt"))
        if not pt_files:
            raise FileNotFoundError(
                f"No .pt file in W&B artifact for batch {ckpt.batch_idx}"
            )
        ckpt.path = str(pt_files[0])
        return ckpt.path

    raise FileNotFoundError(f"Cannot locate checkpoint for batch {ckpt.batch_idx}")


def _build_env_config(ckpt_config: dict) -> EnvConfig:
    """Reconstruct EnvConfig from checkpoint config dict.

    Filters to known fields for forward-compatibility.
    """
    env_dict = ckpt_config.get("env", {})
    known_fields = {f.name for f in fields(EnvConfig)}
    filtered = {k: v for k, v in env_dict.items() if k in known_fields}
    return EnvConfig(**filtered)


def record_episode(
    ckpt_path: str,
    batch_idx: int,
    seed: int,
    output_dir: Path,
    show_camera: bool = True,
    is_first: bool = False,
) -> dict:
    """Record a single eval episode from a checkpoint.

    Returns dict with: steps, final_distance, total_reward, success, rrd_path
    """
    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Build policy
    policy = build_policy(ckpt["config"])
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()

    # Reconstruct env config and create environment
    env_config = _build_env_config(ckpt["config"])
    env_config.max_episode_steps = env_config.max_episode_steps_final

    env = TrainingEnv.from_config(env_config)

    # Set curriculum from checkpoint state
    curriculum_cfg = ckpt["config"].get("curriculum", {})
    num_stages = curriculum_cfg.get("num_stages", 4)
    ckpt_stage = ckpt.get("curriculum_stage", 1)
    ckpt_progress = ckpt.get("stage_progress", 1.0)
    env.set_curriculum_stage(ckpt_stage, ckpt_progress, num_stages=num_stages)

    # Seed for consistent target placement across checkpoints
    np.random.seed(seed)

    # Set up Rerun recording
    run_name = ckpt.get("run_name", "replay")
    rr.init(run_name, recording_id=f"batch-{batch_idx}")

    rrd_path = output_dir / f"batch_{batch_idx}.rrd"
    sinks = [rr.FileSink(str(rrd_path))]
    if is_first:
        rr.spawn(connect=False)
    try:
        sinks.append(rr.GrpcSink())
    except Exception:
        pass  # Viewer may not be running
    rr.set_sinks(*sinks)

    rr.send_recording_name(f"batch {batch_idx}")

    # Embed blueprint
    control_fps = round(1.0 / env.action_dt)
    rr.send_blueprint(
        create_training_blueprint(control_fps=control_fps, show_camera=show_camera)
    )

    # Log metadata
    rr.log(
        "meta/replay",
        rr.TextDocument(
            f"Run: {run_name}\n"
            f"Batch: {batch_idx}\n"
            f"Seed: {seed}\n"
            f"Checkpoint: {Path(ckpt_path).name}"
        ),
        static=True,
    )

    # Set up scene
    arena_boundary = getattr(env, "arena_boundary", None)
    rerun_logger.setup_scene(
        env,
        namespace="eval",
        arena_boundary=arena_boundary,
        show_camera=show_camera,
    )

    # Run episode
    episode_data = collect_episode(
        env, policy, device="cpu", log_rerun=True, deterministic=True
    )

    # Log value traces (PPO only)
    if hasattr(policy, "evaluate_actions"):
        training_cfg = ckpt["config"].get("training", {})
        gamma = training_cfg.get("gamma", 0.99)
        gae_lambda = training_cfg.get("gae_lambda", 0.95)
        log_episode_value_trace(
            policy, episode_data, gamma, gae_lambda, device="cpu", namespace="eval"
        )

    rr.disconnect()
    env.close()

    return {
        "steps": episode_data["steps"],
        "final_distance": episode_data["final_distance"],
        "total_reward": episode_data["total_reward"],
        "success": episode_data["success"],
        "rrd_path": str(rrd_path),
    }


def run_regenerate(run_ref: str, batches: list[int], seed: int = 42) -> None:
    """Re-run episodes from checkpoints at specific batch numbers.

    Args:
        run_ref: Run name (e.g. "s2w-lstm-0218-1045")
        batches: List of batch numbers to replay
        seed: Random seed for consistent target placement
    """
    print(f"\nDiscovering checkpoints for {run_ref}...")
    available = discover_checkpoints(run_ref)

    if not available:
        print(f"  No checkpoints found for '{run_ref}'.")
        print(
            "  Check that the run exists locally (runs/<name>/checkpoints/) or on W&B."
        )
        return

    # Summarize
    local_count = sum(1 for c in available if c.source == "local")
    wandb_count = sum(1 for c in available if c.source == "wandb")
    batch_range = f"{available[0].batch_idx}-{available[-1].batch_idx}"
    source_parts = []
    if local_count:
        source_parts.append(f"{local_count} local")
    if wandb_count:
        source_parts.append(f"{wandb_count} on W&B")
    print(
        f"  Found {len(available)} checkpoints ({', '.join(source_parts)}, batches {batch_range})"
    )

    # Match requested batches
    selected = find_closest_checkpoints(available, batches)
    if not selected:
        print("  No checkpoints matched the requested batches.")
        return

    print("\nMatching requested batches:")
    for req in batches:
        best = min(available, key=lambda c: abs(c.batch_idx - req))
        exact = "exact" if best.batch_idx == req else f"closest: batch {best.batch_idx}"
        print(
            f"  batch {req:>5d} -> checkpoint at batch {best.batch_idx:>5d} ({exact})"
        )

    # Download W&B checkpoints
    wandb_selected = [c for c in selected if c.source == "wandb"]
    if wandb_selected:
        print(f"\nDownloading {len(wandb_selected)} checkpoints from W&B...")
        for i, ckpt in enumerate(wandb_selected, 1):
            print(
                f"  [{i}/{len(wandb_selected)}] batch {ckpt.batch_idx}...",
                end=" ",
                flush=True,
            )
            _ensure_downloaded(ckpt)
            print(f"-> {Path(ckpt.path).name}")

    # Create output directory
    output_dir = Path(f"replay_{run_ref}")
    output_dir.mkdir(exist_ok=True)

    # Record episodes
    print(f"\nRecording episodes (seed={seed})...")
    results = []
    for i, ckpt in enumerate(selected):
        path = _ensure_downloaded(ckpt)
        print(
            f"  [{i + 1}/{len(selected)}] batch {ckpt.batch_idx}:", end=" ", flush=True
        )

        result = record_episode(
            ckpt_path=path,
            batch_idx=ckpt.batch_idx,
            seed=seed,
            output_dir=output_dir,
            is_first=(i == 0),
        )
        results.append(result)

        dist_str = f"{result['final_distance']:.2f}m"
        success_str = " (success)" if result["success"] else ""
        print(
            f"{result['steps']} steps, distance {dist_str}{success_str} -> {result['rrd_path']}"
        )

    print(f"\nRecordings saved to {output_dir}/")
    print(f"Rerun viewer launched with {len(results)} recordings.")


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def run_replay(
    run_ref: str,
    batches: list[int] | None = None,
    seed: int = 42,
    regenerate: bool = False,
    last_n: int | None = None,
) -> None:
    """Main entry point for the replay command.

    Default: download existing recordings from W&B.
    With --regenerate: re-run episodes from checkpoints.
    """
    if regenerate:
        if not batches:
            print("Error: --regenerate requires --batches")
            return
        run_regenerate(run_ref, batches, seed=seed)
    else:
        run_download(run_ref, batches=batches, last_n=last_n)
