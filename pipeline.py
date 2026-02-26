"""
Pipeline: unified construction and description of the training setup.

Replaces the former config.py. The Pipeline class (formerly Config) holds all
hyperparameters and provides methods to build objects (policy, optimizer) and
describe the setup in human-readable form.

All bot-specific defaults live in the _BOT_DEFAULTS registry, accessed via
pipeline_for_bot(). Smoketest is a uniform modifier applied on top.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from reward_hierarchy import RewardHierarchy


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
    arena_boundary: float = 4.0  # Target bounces off +/-boundary
    max_target_distance_stage2: float = 4.0  # Max spawn distance at stage 2 progress=1

    # Stage 3: visual distractors
    max_distractors: int = 4  # Max distractor cubes at stage 3 progress=1
    distractor_min_distance: float = 0.5  # Min spawn distance from origin
    distractor_max_distance: float = 3.0  # Max spawn distance from origin

    # Stage 4: moving distractors
    distractor_max_speed: float = (
        0.2  # Max distractor speed (m/s) at stage 4 progress=1
    )

    # Distance-patience early truncation
    patience_window: int = 100  # Steps to look back (10 sec at 10Hz, 0=disabled)
    patience_min_delta: float = 0.0  # Min cumulative distance reduction to stay alive

    # Joint-stagnation early truncation (0=disabled)
    joint_stagnation_window: int = 0  # Steps to look back
    joint_stagnation_threshold: float = 1.0  # Min total joint movement over window (sum of |delta| across all joints and steps)

    # Walking stage: learn to stand/walk before target navigation
    has_walking_stage: bool = False
    walking_target_pos: tuple[float, float, float] = (
        0.0,
        -10.0,
        0.08,
    )  # Where to place target in walking stage
    forward_velocity_axis: tuple[float, float, float] = (
        0.0,
        -1.0,
        0.0,
    )  # "Forward" for velocity reward
    walking_success_min_forward: float = (
        0.5  # Min forward distance (meters) for walking stage success
    )

    # Reward shaping
    distance_reward_scale: float = 20.0
    movement_bonus: float = 0.0  # Disabled: was rewarding spinning
    time_penalty: float = 0.005

    # Biped-specific reward shaping (all 0.0 = disabled for wheeler)
    upright_reward_scale: float = 0.0
    alive_bonus: float = 0.0
    energy_penalty_scale: float = 0.0
    ground_contact_penalty: float = (
        0.0  # Penalty per step when non-foot geoms touch floor
    )
    forward_velocity_reward_scale: float = (
        0.0  # Reward forward movement (walking stage)
    )

    # Fall detection (0.0 = disabled for wheeler)
    fall_height_fraction: float = (
        0.0  # Fraction of initial height below which = fallen (e.g. 0.5)
    )
    fall_up_z_threshold: float = (
        0.0  # Min torso up_z to be "healthy" (e.g. 0.54 = ~57deg)
    )
    fall_grace_steps: int = (
        0  # Consecutive unhealthy steps before termination (0 = immediate)
    )

    # Action smoothness penalty (0.0 = disabled)
    action_smoothness_scale: float = (
        0.0  # Penalty for action jerk: -scale * ||a_t - a_{t-1}||^2
    )

    # Gait phase encoding (0.0 = disabled)
    gait_phase_period: float = 0.0  # Period in seconds (e.g. 0.6s for ~1.67Hz stride)

    # RSL-RL style rewards (all 0.0 = disabled, uses old reward path)
    vel_tracking_scale: float = 0.0      # Gaussian kernel velocity tracking
    vel_tracking_sigma: float = 0.25     # Sigma for Gaussian kernel
    vel_tracking_cmd: float = 0.5        # Target forward velocity (m/s)
    orientation_scale: float = 0.0       # -sum(projected_gravity_xy^2)
    base_height_scale: float = 0.0       # -(height - target)^2
    base_height_target: float = 0.34     # Target standing height (m)
    z_velocity_scale: float = 0.0        # -z_vel^2
    ang_vel_xy_scale: float = 0.0        # -sum(ang_vel_xy^2)
    action_rate_scale: float = 0.0       # -sum((a_t - a_{t-1})^2)
    torques_scale: float = 0.0           # -sum(torques^2)
    joint_acc_scale: float = 0.0         # -sum(joint_acc^2)
    only_positive_rewards: bool = False  # Clip total reward to max(0, total)


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
    max_log_std: float = 0.7  # max std ~ 2.0

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
    max_grad_norm: float = 0.5  # Max gradient norm for clipping

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
class CommentaryConfig:
    """AI commentary configuration for training dashboard."""

    enabled: bool = True
    interval_seconds: float = 1800.0  # 30 minutes between commentary
    model: str = "opus"


@dataclass
class Pipeline:
    """Complete training pipeline configuration."""

    env: EnvConfig = field(default_factory=EnvConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    commentary: CommentaryConfig = field(default_factory=CommentaryConfig)

    @property
    def bot_name(self) -> str:
        """Extract bot directory name from env.scene_path (e.g. 'simple2wheeler')."""
        from pathlib import Path

        return Path(self.env.scene_path).parent.name

    @property
    def reward_hierarchy(self) -> RewardHierarchy:
        """Build (and cache) the reward hierarchy for this config's bot."""
        # Cache to avoid duplicate validation warnings
        cache_attr = "_reward_hierarchy_cache"
        if not hasattr(self, cache_attr):
            from reward_hierarchy import build_reward_hierarchy

            object.__setattr__(
                self, cache_attr, build_reward_hierarchy(self.bot_name, self.env)
            )
        return getattr(self, cache_attr)

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
            ("commentary", self.commentary),
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
            "commentary": asdict(self.commentary),
        }

    def build_policy(self, device="cpu"):
        """Construct the policy network from this pipeline's config.

        Single source of truth for policy construction — used by train.py,
        checkpoint.py, and play.py.

        Returns:
            nn.Module: The policy network, moved to the specified device.
        """
        import torch

        from policies import LSTMPolicy, MLPPolicy, TinyPolicy

        cfg = self.policy
        if isinstance(device, str):
            device = torch.device(device)

        if cfg.use_mlp:
            policy = MLPPolicy(
                num_actions=cfg.fc_output_size,
                hidden_size=cfg.hidden_size,
                init_std=cfg.init_std,
                max_log_std=cfg.max_log_std,
                sensor_input_size=cfg.sensor_input_size,
            )
        elif cfg.use_lstm:
            policy = LSTMPolicy(
                image_height=cfg.image_height,
                image_width=cfg.image_width,
                hidden_size=cfg.hidden_size,
                num_actions=cfg.fc_output_size,
                init_std=cfg.init_std,
                max_log_std=cfg.max_log_std,
                sensor_input_size=cfg.sensor_input_size,
            )
        else:
            policy = TinyPolicy(
                image_height=cfg.image_height,
                image_width=cfg.image_width,
                num_actions=cfg.fc_output_size,
                init_std=cfg.init_std,
                max_log_std=cfg.max_log_std,
                sensor_input_size=cfg.sensor_input_size,
            )
        return policy.to(device)

    def build_optimizer(self, policy):
        """Construct the Adam optimizer for this pipeline's config.

        Returns:
            optim.Adam: Optimizer for the policy's parameters.
        """
        import torch.optim as optim

        return optim.Adam(policy.parameters(), lr=self.training.learning_rate)

    def describe(self) -> str:
        """Produce a human-readable summary of this pipeline.

        Not a round-trip format. For serialization use to_wandb_config().
        """
        import warnings
        from pathlib import Path

        lines = []
        bot = self.bot_name
        lines.append(f"Pipeline: {bot}")
        lines.append("=" * 40)

        # Bot section
        lines.append("")
        lines.append("Bot")
        lines.append(f"  Scene:       {self.env.scene_path}")
        lines.append(
            f"  Control:     {self.env.control_frequency_hz} Hz ({self.env.mujoco_steps_per_action} MuJoCo steps/action)"
        )

        # Policy section
        lines.append("")
        lines.append("Policy")
        lines.append(f"  Type:        {self.policy.policy_type}")
        lines.append(f"  Hidden:      {self.policy.hidden_size}")
        lines.append(
            f"  Actions:     {self.policy.fc_output_size} (init_std={self.policy.init_std})"
        )
        if self.policy.sensor_input_size > 0:
            lines.append(f"  Sensors:     {self.policy.sensor_input_size}")

        # Rewards section
        lines.append("")
        lines.append("Rewards")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hierarchy = self.reward_hierarchy
        for line in hierarchy.summary_table().split("\n"):
            lines.append(line)

        # Training section
        lines.append("")
        lines.append("Training")
        lines.append(
            f"  Algorithm:   {self.training.algorithm} (clip={self.training.clip_epsilon}, epochs={self.training.ppo_epochs})"
        )
        lines.append(f"  LR:          {self.training.learning_rate}")
        lines.append(f"  Batch:       {self.training.batch_size} episodes")
        lines.append(f"  Curriculum:  {self.curriculum.num_stages} stages")

        # Environment section
        lines.append("")
        lines.append("Environment")
        ep_duration = (
            self.env.max_episode_steps * self.env.mujoco_steps_per_action * 0.002
        )
        lines.append(
            f"  Episode:     {self.env.max_episode_steps} steps ({ep_duration:.1f}s)"
        )
        if self.env.has_walking_stage:
            axis = self.env.forward_velocity_axis
            lines.append(f"  Walk stage:  yes (forward axis {axis})")
        if self.env.fall_height_fraction > 0:
            lines.append(
                f"  Fall detect: height < {self.env.fall_height_fraction:.0%}, tilt > {_tilt_deg(self.env.fall_up_z_threshold):.0f}deg"
            )
        if self.env.gait_phase_period > 0:
            lines.append(f"  Gait phase:  {self.env.gait_phase_period}s period")

        return "\n".join(lines)

    @classmethod
    def from_wandb_dict(cls, d: dict) -> Pipeline:
        """Reconstruct a Pipeline from a checkpoint's nested config dict.

        Unknown keys use defaults (backward compat). Missing sections use defaults.
        """

        def _safe_dataclass(dc_cls, data: dict):
            """Create a dataclass instance, ignoring unknown keys."""
            import dataclasses

            valid_keys = {f.name for f in dataclasses.fields(dc_cls)}
            filtered = {k: v for k, v in data.items() if k in valid_keys}
            return dc_cls(**filtered)

        env = _safe_dataclass(EnvConfig, d.get("env", {}))
        curriculum = _safe_dataclass(CurriculumConfig, d.get("curriculum", {}))
        policy = _safe_dataclass(PolicyConfig, d.get("policy", {}))
        training = _safe_dataclass(TrainingConfig, d.get("training", {}))
        commentary = _safe_dataclass(CommentaryConfig, d.get("commentary", {}))

        return cls(
            env=env,
            curriculum=curriculum,
            policy=policy,
            training=training,
            commentary=commentary,
        )


def _tilt_deg(up_z_threshold: float) -> float:
    """Convert up_z threshold to approximate tilt angle in degrees."""
    import math

    if up_z_threshold <= 0 or up_z_threshold >= 1:
        return 0.0
    return math.degrees(math.acos(up_z_threshold))


# ---------------------------------------------------------------------------
# Bot defaults registry
# ---------------------------------------------------------------------------

# Each entry maps bot_name -> dict of {section: {field: value}} overrides
# applied on top of the Pipeline defaults.
_BOT_DEFAULTS: dict[str, dict] = {
    "simple2wheeler": {
        # All defaults are fine for simple2wheeler
    },
    "simplebiped": {
        "env": dict(
            scene_path="bots/simplebiped/scene.xml",
            max_episode_steps=1000,
            max_episode_steps_final=1000,
            control_frequency_hz=125,
            mujoco_steps_per_action=4,
            success_distance=0.3,
            failure_distance=10.0,
            min_target_distance=0.8,
            max_target_distance=1.5,
            alive_bonus=1.0,
            energy_penalty_scale=0.001,
            distance_reward_scale=10.0,
            time_penalty=0.005,
            upright_reward_scale=0.3,
            ground_contact_penalty=0.5,
            forward_velocity_reward_scale=8.0,
            walking_success_min_forward=0.5,
            joint_stagnation_window=375,
            has_walking_stage=True,
            fall_height_fraction=0.5,
            fall_up_z_threshold=0.54,
            fall_grace_steps=50,
            action_smoothness_scale=0.1,
            gait_phase_period=0.6,
        ),
        "curriculum": dict(
            num_stages=5,
            window_size=10,
            advance_threshold=1.0,
            advance_rate=0.01,
        ),
        "policy": dict(
            policy_type="MLPPolicy",
            hidden_size=256,
            fc_output_size=8,
            sensor_input_size=26,
            init_std=1.0,
        ),
        "training": dict(
            learning_rate=3e-4,
            batch_size=64,
            algorithm="PPO",
            entropy_coeff=0.0,
            ppo_epochs=10,
        ),
    },
    "walker2d": {
        "env": dict(
            scene_path="bots/walker2d/scene.xml",
            max_episode_steps=1000,
            max_episode_steps_final=1000,
            control_frequency_hz=125,
            mujoco_steps_per_action=4,
            success_distance=0.3,
            failure_distance=15.0,
            min_target_distance=0.8,
            max_target_distance=1.5,
            walking_target_pos=(10.0, 0.0, 0.08),
            forward_velocity_axis=(1.0, 0.0, 0.0),
            walking_success_min_forward=1.4,
            has_walking_stage=True,
            alive_bonus=0.1,
            energy_penalty_scale=0.001,
            distance_reward_scale=10.0,
            time_penalty=0.005,
            upright_reward_scale=0.5,
            ground_contact_penalty=0.5,
            forward_velocity_reward_scale=8.0,
            joint_stagnation_window=30,
        ),
        "curriculum": dict(
            num_stages=5,
            window_size=10,
            advance_threshold=0.4,
            advance_rate=0.01,
        ),
        "policy": dict(
            policy_type="MLPPolicy",
            hidden_size=256,
            fc_output_size=6,
            sensor_input_size=18,
            init_std=1.0,
        ),
        "training": dict(
            learning_rate=3e-4,
            batch_size=64,
            algorithm="PPO",
            entropy_coeff=0.0,
            ppo_epochs=10,
        ),
    },
    "childbiped": {
        "env": dict(
            scene_path="bots/childbiped/scene.xml",
            max_episode_steps=1000,
            max_episode_steps_final=1000,
            control_frequency_hz=125,
            mujoco_steps_per_action=4,
            success_distance=0.3,
            failure_distance=10.0,
            min_target_distance=0.8,
            max_target_distance=1.5,
            walking_success_min_forward=0.5,
            joint_stagnation_window=375,
            has_walking_stage=True,
            walking_target_pos=(0.0, -10.0, 0.08),
            forward_velocity_axis=(0.0, -1.0, 0.0),
            fall_height_fraction=0.5,
            fall_up_z_threshold=0.50,
            fall_grace_steps=125,
            gait_phase_period=0.85,
            # Old reward scales zeroed out — replaced by RSL-RL terms below
            distance_reward_scale=0.0,
            movement_bonus=0.0,
            time_penalty=0.0,
            upright_reward_scale=0.0,
            alive_bonus=0.15,
            energy_penalty_scale=0.0,
            ground_contact_penalty=0.0,
            forward_velocity_reward_scale=0.0,
            action_smoothness_scale=0.0,
            # RSL-RL reward structure
            vel_tracking_scale=1.0,
            vel_tracking_sigma=0.25,
            vel_tracking_cmd=0.5,
            orientation_scale=1.0,
            base_height_scale=10.0,
            base_height_target=0.34,
            z_velocity_scale=2.0,
            ang_vel_xy_scale=0.05,
            action_rate_scale=0.01,
            torques_scale=1e-5,
            joint_acc_scale=2.5e-7,
            only_positive_rewards=True,
        ),
        "curriculum": dict(
            num_stages=5,
            window_size=10,
            advance_threshold=0.8,
            advance_rate=0.015,
        ),
        "policy": dict(
            policy_type="LSTMPolicy",
            hidden_size=256,
            fc_output_size=12,
            sensor_input_size=34,
            init_std=0.2,
        ),
        "training": dict(
            learning_rate=1e-4,
            batch_size=64,
            algorithm="PPO",
            entropy_coeff=0.0,
            ppo_epochs=4,
            max_grad_norm=1.0,
            log_rerun_every=9999,
        ),
    },
}

# Smoketest overrides applied uniformly on top of any bot's pipeline
_SMOKETEST_OVERRIDES = {
"env": dict(
        render_width=64,
        render_height=64,
        max_episode_steps=10,
        walking_success_min_forward=0.0,
    ),
    "curriculum": dict(
        window_size=1,
        advance_threshold=0.0,
        advance_rate=1.0,
        eval_episodes_per_batch=1,
    ),
    "policy": dict(
        image_height=64,
        image_width=64,
        hidden_size=32,
    ),
    "training": dict(
        batch_size=2,
        mastery_batches=1,
        mastery_threshold=0.0,
        max_batches=3,
        log_rerun_every=9999,
        ppo_epochs=2,
    ),
    "commentary": dict(
        enabled=False,
    ),
}


def pipeline_for_bot(bot_name: str, smoketest: bool = False) -> Pipeline:
    """Build a Pipeline for a given bot, optionally with smoketest overrides.

    This replaces the 8 factory methods (for_biped, for_biped_smoketest, etc.)
    with a data-driven registry. Unknown bot names fall back to simple2wheeler
    defaults.

    Args:
        bot_name: Bot directory name (e.g. "childbiped", "simplebiped").
        smoketest: If True, apply smoketest overrides for fast validation.

    Returns:
        Pipeline with all bot-specific defaults applied.
    """
    # Start from defaults
    pipeline = Pipeline()

    # Apply bot-specific overrides
    bot_overrides = _BOT_DEFAULTS.get(bot_name, {})
    pipeline = _apply_overrides(pipeline, bot_overrides)

    # Apply smoketest overrides on top
    if smoketest:
        pipeline = _apply_overrides(pipeline, _SMOKETEST_OVERRIDES)

    return pipeline


def _apply_overrides(pipeline: Pipeline, overrides: dict) -> Pipeline:
    """Apply nested overrides to a Pipeline using dataclasses.replace().

    Args:
        pipeline: The base Pipeline.
        overrides: Dict of {section_name: {field: value}} to apply.

    Returns:
        New Pipeline with overrides applied.
    """
    sections = {}
    for section_name, section_overrides in overrides.items():
        if not section_overrides:
            continue
        current = getattr(pipeline, section_name)
        sections[section_name] = replace(current, **section_overrides)

    if sections:
        return replace(pipeline, **sections)
    return pipeline
