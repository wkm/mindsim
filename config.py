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

    # Stage 4: moving distractors
    distractor_max_speed: float = 0.2  # Max distractor speed (m/s) at stage 4 progress=1

    # Distance-patience early truncation
    patience_window: int = 100  # Steps to look back (10 sec at 10Hz, 0=disabled)
    patience_min_delta: float = 0.0  # Min cumulative distance reduction to stay alive

    # Joint-stagnation early truncation (0=disabled)
    joint_stagnation_window: int = 0  # Steps to look back
    joint_stagnation_threshold: float = 1.0  # Min total joint movement over window (sum of |delta| across all joints and steps)

    # Walking stage: learn to stand/walk before target navigation
    has_walking_stage: bool = False
    walking_target_pos: tuple[float, float, float] = (0.0, -10.0, 0.08)  # Where to place target in walking stage
    forward_velocity_axis: tuple[float, float, float] = (0.0, -1.0, 0.0)  # "Forward" for velocity reward
    walking_success_min_forward: float = 0.5  # Min forward distance (meters) for walking stage success

    # Reward shaping
    distance_reward_scale: float = 20.0
    movement_bonus: float = 0.0  # Disabled: was rewarding spinning
    time_penalty: float = 0.005

    # Biped-specific reward shaping (all 0.0 = disabled for wheeler)
    upright_reward_scale: float = 0.0
    alive_bonus: float = 0.0
    energy_penalty_scale: float = 0.0
    ground_contact_penalty: float = 0.0  # Penalty per step when non-foot geoms touch floor
    forward_velocity_reward_scale: float = 0.0  # Reward forward movement (walking stage)


@dataclass
class CurriculumConfig:
    """Curriculum learning configuration."""

    num_stages: int = 4  # Total curriculum stages
    window_size: int = 10  # Batches to average for success rate
    advance_threshold: float = 0.6  # Advance when success rate > 60%
    advance_rate: float = 0.02  # Per-batch advancement

    # Deterministic evaluation for curriculum decisions
    eval_episodes_per_batch: int = 8  # Deterministic eval episodes per batch
    use_eval_for_curriculum: bool = True  # Use eval success rate for curriculum


@dataclass
class PolicyConfig:
    """Neural network policy configuration."""

    policy_type: Literal["TinyPolicy", "LSTMPolicy", "MLPPolicy"] = "LSTMPolicy"

    # Image input
    image_height: int = 64
    image_width: int = 64

    # FC / LSTM layers
    hidden_size: int = 256  # FC1 for TinyPolicy, LSTM hidden for LSTMPolicy
    fc_output_size: int = 2  # Motor commands

    # Proprioceptive sensor input (0 = image-only, >0 = concat with CNN features)
    sensor_input_size: int = 0

    # Stochastic policy
    init_std: float = 0.5
    max_log_std: float = 0.7  # max std ≈ 2.0

    @property
    def use_lstm(self) -> bool:
        return self.policy_type == "LSTMPolicy"

    @property
    def use_mlp(self) -> bool:
        return self.policy_type == "MLPPolicy"


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

    @property
    def bot_name(self) -> str:
        """Extract bot directory name from env.scene_path (e.g. 'simple2wheeler')."""
        from pathlib import Path
        return Path(self.env.scene_path).parent.name

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
                num_stages=4,
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
        """Config for the 8-joint duck biped with MLPPolicy (hip_abd + hip + knee + ankle per leg)."""
        return cls(
            env=EnvConfig(
                scene_path="bots/simplebiped/scene.xml",
                render_width=64,
                render_height=64,
                max_episode_steps=1000,  # 8s at 125Hz (matching Walker2d)
                max_episode_steps_final=1000,
                control_frequency_hz=125,
                mujoco_steps_per_action=4,  # 0.002s * 4 = 125Hz control (matching Walker2d)
                success_distance=0.3,
                failure_distance=10.0,
                min_target_distance=0.8,
                max_target_distance=1.5,  # Closer targets initially
                # Biped rewards
                alive_bonus=0.1,
                energy_penalty_scale=0.001,
                distance_reward_scale=10.0,
                time_penalty=0.005,  # Small per-step cost for efficiency
                upright_reward_scale=0.5,  # Reward staying upright
                ground_contact_penalty=0.5,  # Penalize non-foot ground contact
                forward_velocity_reward_scale=8.0,  # Strong forward signal — must clearly beat standing-still rewards
                walking_success_min_forward=0.5,  # ~1 body length (biped is ~0.3m tall)
                joint_stagnation_window=375,  # 3 sec at 125Hz — abort frozen episodes
                has_walking_stage=True,
            ),
            curriculum=CurriculumConfig(
                num_stages=5,  # Walking + 4 standard stages
                window_size=10,
                advance_threshold=0.4,  # Lower threshold — walking is harder
                advance_rate=0.01,
            ),
            policy=PolicyConfig(
                policy_type="MLPPolicy",
                image_height=64,
                image_width=64,
                hidden_size=256,
                fc_output_size=8,  # 8 torque motors (hip_abd + hip + knee + ankle per leg)
                sensor_input_size=22,  # 8 pos + 8 vel + 3 gyro + 3 accel
                init_std=1.0,  # Wide exploration
            ),
            training=TrainingConfig(
                learning_rate=3e-4,  # SB3 default
                batch_size=64,
                algorithm="PPO",
                entropy_coeff=0.0,  # SB3 default
                ppo_epochs=10,  # SB3 default
            ),
        )

    @classmethod
    def for_walker2d(cls) -> Config:
        """Config for Walker2d PPO diagnostic baseline with MLPPolicy."""
        return cls(
            env=EnvConfig(
                scene_path="bots/walker2d/scene.xml",
                render_width=64,
                render_height=64,
                max_episode_steps=1000,  # 8s at 125Hz (canonical Walker2d)
                max_episode_steps_final=1000,
                control_frequency_hz=125,
                mujoco_steps_per_action=4,  # 0.002s * 4 = 125Hz control (Gymnasium frame_skip)
                success_distance=0.3,
                failure_distance=15.0,  # Walker2d can travel far
                min_target_distance=0.8,
                max_target_distance=1.5,
                # Walking stage: target in +X (Walker2d forward direction)
                walking_target_pos=(10.0, 0.0, 0.08),
                forward_velocity_axis=(1.0, 0.0, 0.0),
                walking_success_min_forward=1.4,  # ~1 body length (Walker2d is ~1.4m tall)
                has_walking_stage=True,
                # Same reward structure as biped
                alive_bonus=0.1,
                energy_penalty_scale=0.001,
                distance_reward_scale=10.0,
                time_penalty=0.005,
                upright_reward_scale=0.5,
                ground_contact_penalty=0.5,
                forward_velocity_reward_scale=8.0,
                joint_stagnation_window=30,
            ),
            curriculum=CurriculumConfig(
                num_stages=5,  # Walking + 4 standard stages
                window_size=10,
                advance_threshold=0.4,
                advance_rate=0.01,
            ),
            policy=PolicyConfig(
                policy_type="MLPPolicy",
                image_height=64,
                image_width=64,
                hidden_size=256,
                fc_output_size=6,  # 6 torque motors
                sensor_input_size=18,  # 6 pos + 6 vel + 3 gyro + 3 accel
                init_std=1.0,
            ),
            training=TrainingConfig(
                learning_rate=3e-4,  # SB3 default
                batch_size=64,
                algorithm="PPO",
                entropy_coeff=0.0,  # SB3 default for Walker2d
                ppo_epochs=10,  # SB3 default
            ),
        )

    @classmethod
    def for_walker2d_smoketest(cls) -> Config:
        """Config for fast Walker2d end-to-end validation."""
        return cls(
            env=EnvConfig(
                scene_path="bots/walker2d/scene.xml",
                render_width=64,
                render_height=64,
                max_episode_steps=10,
                control_frequency_hz=125,
                mujoco_steps_per_action=4,
                walking_target_pos=(10.0, 0.0, 0.08),
                forward_velocity_axis=(1.0, 0.0, 0.0),
                walking_success_min_forward=0.0,  # Smoketest: no forward requirement
                has_walking_stage=True,
                alive_bonus=0.1,
                energy_penalty_scale=0.001,
                distance_reward_scale=10.0,
                time_penalty=0.005,
                upright_reward_scale=0.5,
                ground_contact_penalty=0.5,
                forward_velocity_reward_scale=8.0,
            ),
            curriculum=CurriculumConfig(
                window_size=1,
                advance_threshold=0.0,
                advance_rate=1.0,
                eval_episodes_per_batch=1,
                num_stages=5,
            ),
            policy=PolicyConfig(
                policy_type="MLPPolicy",
                image_height=64,
                image_width=64,
                hidden_size=32,
                fc_output_size=6,
                sensor_input_size=18,
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

    @classmethod
    def for_biped_smoketest(cls) -> Config:
        """Config for fast biped end-to-end validation."""
        return cls(
            env=EnvConfig(
                scene_path="bots/simplebiped/scene.xml",
                render_width=64,
                render_height=64,
                max_episode_steps=10,
                control_frequency_hz=125,
                mujoco_steps_per_action=4,
                alive_bonus=0.1,
                energy_penalty_scale=0.001,
                distance_reward_scale=10.0,
                time_penalty=0.005,
                upright_reward_scale=0.5,
                ground_contact_penalty=0.5,
                forward_velocity_reward_scale=8.0,
                walking_success_min_forward=0.0,  # Smoketest: no forward requirement
                has_walking_stage=True,
            ),
            curriculum=CurriculumConfig(
                window_size=1,
                advance_threshold=0.0,
                advance_rate=1.0,
                eval_episodes_per_batch=1,
                num_stages=5,  # Walking + 4 standard stages
            ),
            policy=PolicyConfig(
                policy_type="MLPPolicy",
                image_height=64,
                image_width=64,
                hidden_size=32,
                fc_output_size=8,  # 8 torque motors
                sensor_input_size=22,  # 8 pos + 8 vel + 3 gyro + 3 accel
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
