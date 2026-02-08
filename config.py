"""
Centralized configuration for training.

All hyperparameters and settings in one place.
Automatically converts to dict for W&B logging.
"""
from dataclasses import dataclass, field, asdict
from typing import Literal, Optional


@dataclass
class EnvConfig:
    """Environment configuration."""
    render_width: int = 64
    render_height: int = 64
    max_episode_steps: int = 100  # 10 seconds at 10 Hz
    control_frequency_hz: int = 10
    mujoco_steps_per_action: int = 5

    # Termination thresholds
    success_distance: float = 0.3
    failure_distance: float = 5.0

    # Target spawn range
    min_target_distance: float = 0.8
    max_target_distance: float = 2.5
    randomize_target: bool = True

    # Reward shaping
    distance_reward_scale: float = 20.0
    distance_reward_type: str = "linear"  # potential-based shaping
    movement_bonus: float = 0.0  # Disabled: was rewarding spinning
    time_penalty: float = 0.005


@dataclass
class CurriculumConfig:
    """Curriculum learning configuration."""
    window_size: int = 10  # Batches to average for success rate
    advance_threshold: float = 0.6  # Advance when success rate > 60%
    retreat_threshold: float = 0.3  # Retreat when success rate < 30%
    advance_rate: float = 0.02  # Per-batch advancement
    retreat_rate: float = 0.01  # Per-batch retreat


@dataclass
class PolicyConfig:
    """Neural network policy configuration."""
    policy_type: Literal["TinyPolicy", "LSTMPolicy"] = "LSTMPolicy"

    # Image input
    image_height: int = 64
    image_width: int = 64

    # CNN architecture
    conv1_out_channels: int = 8
    conv1_kernel: int = 8
    conv1_stride: int = 4
    conv2_out_channels: int = 16
    conv2_kernel: int = 4
    conv2_stride: int = 2

    # FC / LSTM layers
    hidden_size: int = 64  # FC1 for TinyPolicy, LSTM hidden for LSTMPolicy
    fc_output_size: int = 2  # Motor commands

    # Activation
    activation: str = "relu"
    output_activation: str = "tanh"

    # Stochastic policy
    init_std: float = 0.5

    @property
    def use_lstm(self) -> bool:
        return self.policy_type == "LSTMPolicy"


@dataclass
class TrainingConfig:
    """Training loop configuration."""
    # Optimizer
    optimizer: str = "Adam"
    learning_rate: float = 3e-2

    # Algorithm
    algorithm: str = "REINFORCE"
    gamma: float = 0.99

    # Batching
    batch_size: int = 64  # Episodes per gradient update

    # Mastery criteria
    training_mode: str = "run_until_mastery"
    mastery_threshold: float = 0.7  # Success rate required at curriculum=1.0
    mastery_batches: int = 20  # Must maintain mastery for N batches

    # Logging
    log_rerun_every: int = 100  # Episodes between Rerun recordings


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


# Default configuration
def get_default_config() -> Config:
    """Get the default training configuration."""
    return Config()
