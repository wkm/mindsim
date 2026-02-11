"""
Parallel episode collection using multiprocessing.

Each worker process gets its own MuJoCo environment and policy copy.
Workers persist across batches to amortize process startup cost.
Policy weights are synced from the main process at the start of each batch
as numpy arrays to avoid torch shared-memory serialization issues.
"""

import multiprocessing as mp
import os
from dataclasses import asdict

import torch

# Worker-local globals (initialized once per worker process)
_worker_env = None
_worker_policy = None


def _init_worker(env_config_dict, policy_class_name, policy_kwargs):
    """Initialize a persistent environment and policy in the worker process."""
    global _worker_env, _worker_policy

    from config import EnvConfig
    from train import LSTMPolicy, TinyPolicy
    from training_env import TrainingEnv

    env_config = EnvConfig(**env_config_dict)
    _worker_env = TrainingEnv.from_config(env_config)

    if policy_class_name == "LSTMPolicy":
        _worker_policy = LSTMPolicy(**policy_kwargs)
    else:
        _worker_policy = TinyPolicy(**policy_kwargs)
    _worker_policy.eval()


def _collect_one(args):
    """Collect a single episode in a worker process."""
    global _worker_env, _worker_policy

    state_dict_np, curriculum_stage, stage_progress, num_stages, deterministic = args

    from train import collect_episode

    # Convert numpy arrays back to torch tensors
    state_dict = {k: torch.from_numpy(v) for k, v in state_dict_np.items()}
    _worker_policy.load_state_dict(state_dict)
    _worker_env.set_curriculum_stage(curriculum_stage, stage_progress, num_stages)

    return collect_episode(
        _worker_env,
        _worker_policy,
        device="cpu",
        show_progress=False,
        log_rerun=False,
        deterministic=deterministic,
    )


def resolve_num_workers(requested):
    """Resolve num_workers config value.

    0 = auto (cpu_count - 1, minimum 2), 1 = serial (no multiprocessing).
    """
    if requested == 0:
        return max(2, (os.cpu_count() or 2) - 1)
    return requested


class ParallelCollector:
    """Pool of worker processes for parallel episode collection.

    Each worker holds its own MuJoCo environment and policy network.
    Policy weights are synced from the main process before each batch.
    """

    def __init__(self, num_workers, env_config, policy_config):
        self.num_workers = num_workers

        # Build policy constructor kwargs from config
        policy_kwargs = {
            "image_height": policy_config.image_height,
            "image_width": policy_config.image_width,
            "init_std": policy_config.init_std,
            "max_log_std": policy_config.max_log_std,
        }
        if policy_config.use_lstm:
            policy_kwargs["hidden_size"] = policy_config.hidden_size

        ctx = mp.get_context("spawn")
        self.pool = ctx.Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(
                asdict(env_config),
                policy_config.policy_type,
                policy_kwargs,
            ),
        )

    def collect_batch(
        self, policy, batch_size, curriculum_stage, stage_progress,
        num_stages=3, deterministic=False,
    ):
        """Collect batch_size episodes in parallel.

        Syncs current policy weights to all workers, then fans out
        episode collection across the pool.

        Returns:
            List of episode_data dicts (same format as collect_episode).
        """
        # Convert state_dict to numpy for pickle-friendly serialization
        # (avoids torch's shared-memory manager which can have permission issues)
        state_dict_np = {k: v.cpu().numpy() for k, v in policy.state_dict().items()}

        args_list = [
            (state_dict_np, curriculum_stage, stage_progress, num_stages, deterministic)
            for _ in range(batch_size)
        ]

        return self.pool.map(_collect_one, args_list)

    def close(self):
        """Shut down the worker pool."""
        self.pool.close()
        self.pool.join()
