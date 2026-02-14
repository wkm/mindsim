"""
Centralized configuration for training.

All hyperparameters and settings in one place.
Automatically converts to dict for W&B logging.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal


@dataclass
class EnvConfig:
    """Environment configuration."""

    scene_path: str = "bots/simple2wheeler/scene.xml"  # Selects which bot to load
    render_width: int = 64
    render_height: int = 64
    max_episode_steps: int = 200  # 20 seconds at 10 Hz (stage 1 baseline)
    max_episode_steps_final: int = 500  # 50 seconds at 10 Hz (at full curriculum)
    control_frequency_hz: int = 10
    mujoco_steps_per_action: int = 5

    # Termination thresholds
    success_distance: float = 0.3
    failure_distance: float = 10.0

    # Target spawn range
    min_target_distance: float = 0.8
    max_target_distance: float = 2.5
    # Stage 2: moving target + distance
    target_max_speed: float = 0.3  # Max target speed (m/s) at stage 2 progress=1
    arena_boundary: float = 4.0  # Target bounces off ±boundary
    max_target_distance_stage2: float = 4.0  # Max spawn distance at stage 2 progress=1

    # Stage 3: visual distractors
    max_distractors: int = 4  # Max distractor cubes at stage 3 progress=1
    distractor_min_distance: float = 0.5  # Min spawn distance from origin
    distractor_max_distance: float = 3.0  # Max spawn distance from origin

    # Distance-patience early truncation
    patience_window: int = 100  # Steps to look back (10 sec at 10Hz, 0=disabled)
    patience_min_delta: float = 0.0  # Min cumulative distance reduction to stay alive

    # Reward shaping
    distance_reward_scale: float = 20.0
    movement_bonus: float = 0.0  # Disabled: was rewarding spinning
    time_penalty: float = 0.005

    # Biped-specific reward shaping (all 0.0 = disabled for wheeler)
    upright_reward_scale: float = 0.0
    alive_bonus: float = 0.0
    energy_penalty_scale: float = 0.0
    fall_height_threshold: float = 0.0  # 0 = disabled; 0.3 for biped
    fall_tilt_threshold: float = 0.7    # cos(tilt) below this = fallen


@dataclass
class CurriculumConfig:
    """Curriculum learning configuration."""

    num_stages: int = 3  # Total curriculum stages
    window_size: int = 10  # Batches to average for success rate
    advance_threshold: float = 0.6  # Advance when success rate > 60%
    advance_rate: float = 0.02  # Per-batch advancement

    # Deterministic evaluation for curriculum decisions
    eval_episodes_per_batch: int = 8  # Deterministic eval episodes per batch
    use_eval_for_curriculum: bool = True  # Use eval success rate for curriculum


@dataclass
class PolicyConfig:
    """Neural network policy configuration."""

    policy_type: Literal["TinyPolicy", "LSTMPolicy"] = "LSTMPolicy"

    # Image input
    image_height: int = 64
    image_width: int = 64

    # FC / LSTM layers
    hidden_size: int = 256  # FC1 for TinyPolicy, LSTM hidden for LSTMPolicy
    fc_output_size: int = 2  # Motor commands

    # Stochastic policy
    init_std: float = 0.5
    max_log_std: float = 0.7  # max std ≈ 2.0

    @property
    def use_lstm(self) -> bool:
        return self.policy_type == "LSTMPolicy"


@dataclass
class TrainingConfig:
    """Training loop configuration."""

    learning_rate: float = 1e-3

    # Algorithm
    algorithm: str = "PPO"
    gamma: float = 0.99
    entropy_coeff: float = 0.05  # Entropy bonus to prevent policy collapse

    # PPO-specific
    ppo_epochs: int = 4
    clip_epsilon: float = 0.2
    gae_lambda: float = 0.95
    value_coeff: float = 0.5

    # Batching
    batch_size: int = 64  # Episodes per gradient update

    # Mastery criteria
    mastery_threshold: float = 0.7  # Success rate required at curriculum=1.0
    mastery_batches: int = 20  # Must maintain mastery for N batches

    # Logging
    log_rerun_every: int = 500  # Episodes between Rerun recordings

    # Parallelism
    num_workers: int = 0  # 0 = auto, 1 = serial (no multiprocessing)

    # Checkpointing
    checkpoint_every: int | None = (
        50  # Periodic checkpoint every N batches (None = disabled)
    )

    # Limits
    max_batches: int | None = None  # None = run until mastery


@dataclass
class Config:
    """Complete training configuration."""

    env: EnvConfig = field(default_factory=EnvConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_flat_dict(self) -> dict:
        """
        Convert to flat dict for W&B logging.

        Prefixes each section's keys with section name.
        Example: env.render_width -> "env/render_width"
        """
        result = {}
        for section_name, section in [
            ("env", self.env),
            ("curriculum", self.curriculum),
            ("policy", self.policy),
            ("training", self.training),
        ]:
            for key, value in asdict(section).items():
                result[f"{section_name}/{key}"] = value
        return result

    def to_wandb_config(self) -> dict:
        """
        Convert to dict suitable for wandb.init(config=...).

        Returns a nested dict that W&B will flatten with dots.
        """
        return {
            "env": asdict(self.env),
            "curriculum": asdict(self.curriculum),
            "policy": asdict(self.policy),
            "training": asdict(self.training),
        }

    @classmethod
    def for_smoketest(cls) -> Config:
        """Config for fast end-to-end validation. Runs in seconds."""
        return cls(
            env=EnvConfig(
                render_width=64,
                render_height=64,
                max_episode_steps=10,  # Very short episodes
            ),
            curriculum=CurriculumConfig(
                window_size=1,
                advance_threshold=0.0,  # Always advance
                advance_rate=1.0,  # Jump to full progress immediately
                eval_episodes_per_batch=1,
                num_stages=3,
            ),
            policy=PolicyConfig(
                policy_type="LSTMPolicy",
                image_height=64,
                image_width=64,
                hidden_size=32,  # Tiny network
            ),
            training=TrainingConfig(
                batch_size=2,
                mastery_batches=1,
                mastery_threshold=0.0,  # Always consider mastered
                max_batches=3,  # Just a few batches
                log_rerun_every=9999,  # Effectively disable
                ppo_epochs=2,
            ),
        )


    @classmethod
    def for_biped(cls) -> Config:
        """Config for the 6-joint biped walking experiment."""
        return cls(
            env=EnvConfig(
                scene_path="bots/simplebiped/scene.xml",
                render_width=64,
                render_height=64,
                max_episode_steps=200,
                mujoco_steps_per_action=20,  # 0.005s timestep * 20 = 10 Hz control
                success_distance=0.3,
                failure_distance=10.0,
                min_target_distance=0.8,
                max_target_distance=1.5,  # Closer targets initially
                # Biped rewards
                upright_reward_scale=1.0,
                alive_bonus=0.1,
                energy_penalty_scale=0.001,
                fall_height_threshold=0.3,
                fall_tilt_threshold=0.5,
                # Distance reward lower scale — upright is priority early
                distance_reward_scale=10.0,
                time_penalty=0.0,  # Disabled: alive_bonus replaces time_penalty
            ),
            curriculum=CurriculumConfig(
                num_stages=3,
                window_size=10,
                advance_threshold=0.4,  # Lower threshold — walking is harder
                advance_rate=0.01,
            ),
            policy=PolicyConfig(
                policy_type="LSTMPolicy",
                image_height=64,
                image_width=64,
                hidden_size=256,
                fc_output_size=6,  # 6 joint motors
                init_std=0.3,  # Lower initial exploration
            ),
            training=TrainingConfig(
                learning_rate=3e-4,
                batch_size=64,
                algorithm="PPO",
            ),
        )

    @classmethod
    def for_biped_smoketest(cls) -> Config:
        """Config for fast biped end-to-end validation."""
        return cls(
            env=EnvConfig(
                scene_path="bots/simplebiped/scene.xml",
                render_width=64,
                render_height=64,
                max_episode_steps=10,
                mujoco_steps_per_action=20,
                upright_reward_scale=1.0,
                alive_bonus=0.1,
                energy_penalty_scale=0.001,
                fall_height_threshold=0.3,
                fall_tilt_threshold=0.5,
                distance_reward_scale=10.0,
                time_penalty=0.0,
            ),
            curriculum=CurriculumConfig(
                window_size=1,
                advance_threshold=0.0,
                advance_rate=1.0,
                eval_episodes_per_batch=1,
                num_stages=3,
            ),
            policy=PolicyConfig(
                policy_type="LSTMPolicy",
                image_height=64,
                image_width=64,
                hidden_size=32,
                fc_output_size=6,
            ),
            training=TrainingConfig(
                batch_size=2,
                mastery_batches=1,
                mastery_threshold=0.0,
                max_batches=3,
                log_rerun_every=9999,
                ppo_epochs=2,
            ),
        )
