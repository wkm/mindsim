"""
Policy networks for continuous control.

Contains LSTMPolicy, TinyPolicy, and MLPPolicy — all with tanh-squashed
Gaussian action distributions and PPO-compatible evaluate_actions().
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _tanh_log_prob(gaussian_log_prob, pre_tanh_value):
    """Correct Gaussian log-prob for tanh squashing.

    For action = tanh(z) where z ~ Normal(mean, std):
        log π(action) = log Normal(z; mean, std) - Σ log(1 - tanh(z)²)

    Uses numerically stable formula: log(1 - tanh(z)²) = 2(log2 - z - softplus(-2z))
    """
    correction = 2 * (math.log(2) - pre_tanh_value - F.softplus(-2 * pre_tanh_value))
    return gaussian_log_prob - correction.sum(dim=-1)


class LSTMPolicy(nn.Module):
    """
    LSTM-based stochastic policy network with memory.

    Input: RGB image (HxWx3)
    Output: Mean of Gaussian distribution over motor commands [left, right]

    Uses CNN to extract features, then LSTM to maintain temporal context.
    This allows the policy to remember past observations and actions.
    """

    def __init__(
        self,
        image_height=128,
        image_width=128,
        hidden_size=64,
        num_actions=2,
        init_std=0.5,
        max_log_std=0.7,
        sensor_input_size=0,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.sensor_input_size = sensor_input_size

        # CNN feature extractor (same as TinyPolicy)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)

        # Compute flattened CNN output size from input dimensions
        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_height, image_width)
            dummy = self.conv2(self.conv1(dummy))
            conv_out_size = dummy.numel()

        # Compress CNN features (+ optional sensor data) to compact vector before LSTM
        self.fc_embed = nn.Linear(conv_out_size + sensor_input_size, hidden_size)

        # LSTM for temporal memory (operates on compact feature vector)
        self.lstm = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size, batch_first=True
        )

        # Output layers
        self.fc = nn.Linear(hidden_size, num_actions)
        self.value_fc = nn.Linear(hidden_size, 1)  # Value head for PPO

        # Learnable log_std (state-independent), clamped in forward()
        self.log_std = nn.Parameter(torch.ones(num_actions) * np.log(init_std))
        self.max_log_std = max_log_std

        # Hidden state (will be set during episode)
        self.hidden = None

    def reset_hidden(self, batch_size=1, device="cpu"):
        """Reset LSTM hidden state at the start of each episode."""
        self.hidden = (
            torch.zeros(1, batch_size, self.hidden_size, device=device),
            torch.zeros(1, batch_size, self.hidden_size, device=device),
        )

    def _backbone(self, x, hidden=None, sensors=None):
        """
        Shared CNN+LSTM backbone.

        Args:
            x: Input image (B, H, W, 3) or (B, T, H, W, 3) for sequences
            hidden: Optional LSTM hidden state tuple (h, c)
            sensors: Optional sensor data (B, sensor_dim) or (B, T, sensor_dim)

        Returns:
            features: (B, 2) or (B, T, hidden_size) LSTM output features
            hidden: Updated hidden state
            is_sequence: Whether the input was a sequence
        """
        is_sequence = x.dim() == 5
        if is_sequence:
            B, T, H, W, C = x.shape
            x = x.reshape(B * T, H, W, C)
        else:
            B = x.size(0)
            T = 1

        x = x.permute(0, 3, 1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)

        # Concatenate sensor data with CNN features before embedding
        if sensors is not None and self.sensor_input_size > 0:
            if sensors.dim() == 3:
                sensors = sensors.reshape(B * T, -1)
            x = torch.cat([x, sensors], dim=-1)

        x = torch.relu(self.fc_embed(x))
        x = x.reshape(B, T, -1)

        if hidden is None:
            hidden = self.hidden

        lstm_out, hidden = self.lstm(x, hidden)
        self.hidden = hidden

        if is_sequence:
            features = lstm_out  # (B, T, hidden_size)
        else:
            features = lstm_out.squeeze(1)  # (B, hidden_size)

        return features, hidden, is_sequence

    def forward(self, x, hidden=None, sensors=None):
        """
        Forward pass - returns action distribution parameters.

        Args:
            x: Input image (B, H, W, 3) or (B, T, H, W, 3) for sequences
            hidden: Optional LSTM hidden state tuple (h, c)
            sensors: Optional sensor data

        Returns:
            mean: (B, 2) or (B, T, 2) unbounded mean (tanh applied at sampling)
            std: (2,) standard deviation (shared across batch)
        """
        features, hidden, is_sequence = self._backbone(x, hidden, sensors=sensors)
        mean = self.fc(features)
        clamped_log_std = self.log_std.clamp(max=self.max_log_std)
        std = torch.exp(clamped_log_std)
        return mean, std

    def evaluate_actions(self, observations, actions, sensors=None):
        """
        Single forward pass returning everything PPO needs.

        Processes full episode sequence with fresh hidden state.

        Args:
            observations: (T, H, W, 3) episode observations
            actions: (T, 2) tanh-squashed actions
            sensors: Optional (T, sensor_dim) sensor data

        Returns:
            log_probs: (T,) log probability of actions
            values: (T,) state value estimates
            entropy: scalar mean entropy
        """
        device = observations.device
        self.reset_hidden(batch_size=1, device=device)

        # Process entire sequence: (1, T, H, W, 3)
        x = observations.unsqueeze(0)
        s = sensors.unsqueeze(0) if sensors is not None else None
        features, _, _ = self._backbone(x, sensors=s)  # (1, T, hidden_size)
        features = features.squeeze(0)  # (T, hidden_size)

        # Action head
        mean = self.fc(features)  # (T, 2)
        clamped_log_std = self.log_std.clamp(max=self.max_log_std)
        std = torch.exp(clamped_log_std)

        # Value head
        values = self.value_fc(features).squeeze(-1)  # (T,)

        # Log prob with tanh correction
        z = torch.atanh(actions.clamp(-1 + 1e-6, 1 - 1e-6))
        dist = torch.distributions.Normal(mean, std)
        gaussian_log_prob = dist.log_prob(z).sum(dim=-1)
        log_probs = _tanh_log_prob(gaussian_log_prob, z)

        # Entropy
        entropy = dist.entropy().sum(dim=-1).mean()

        return log_probs, values, entropy

    def sample_action(self, x, sensors=None):
        """
        Sample an action using tanh-squashed Gaussian.

        Args:
            x: Input image (B, H, W, 3)
            sensors: Optional sensor data (B, sensor_dim)

        Returns:
            action: (B, 2) sampled actions in (-1, 1)
            log_prob: (B,) log probability of the sampled actions
        """
        mean, std = self.forward(x, sensors=sensors)
        dist = torch.distributions.Normal(mean, std)
        z = dist.sample()
        gaussian_log_prob = dist.log_prob(z).sum(dim=-1)
        action = torch.tanh(z)
        log_prob = _tanh_log_prob(gaussian_log_prob, z)
        return action, log_prob

    def get_deterministic_action(self, x, sensors=None):
        """
        Get deterministic action (tanh of mean).

        Args:
            x: Input image (B, H, W, 3)
            sensors: Optional sensor data (B, sensor_dim)

        Returns:
            action: (B, 2) mean actions in (-1, 1)
        """
        mean, _ = self.forward(x, sensors=sensors)
        return torch.tanh(mean)

    def log_prob(self, x, action, sensors=None):
        """
        Compute log probability of given tanh-squashed actions for a sequence.

        Args:
            x: Input images (T, H, W, 3) - full episode
            action: (T, 2) tanh-squashed actions to evaluate
            sensors: Optional (T, sensor_dim) sensor data

        Returns:
            log_prob: (T,) log probability of actions
        """
        device = x.device

        # Reset hidden state for fresh forward pass
        self.reset_hidden(batch_size=1, device=device)

        # Process entire sequence at once
        x = x.unsqueeze(0)  # (1, T, H, W, 3)
        s = sensors.unsqueeze(0) if sensors is not None else None
        mean, std = self.forward(x, sensors=s)  # mean: (1, T, 2)
        mean = mean.squeeze(0)  # (T, 2)

        # Recover pre-tanh values
        z = torch.atanh(action.clamp(-1 + 1e-6, 1 - 1e-6))

        dist = torch.distributions.Normal(mean, std)
        gaussian_log_prob = dist.log_prob(z).sum(dim=-1)
        return _tanh_log_prob(gaussian_log_prob, z)


class TinyPolicy(nn.Module):
    """
    Stochastic policy network for REINFORCE.

    Input: RGB image (HxWx3)
    Output: Mean of Gaussian distribution over motor commands [left, right]

    The policy is stochastic: actions are sampled from N(mean, std).
    std is a learnable parameter (not state-dependent for simplicity).
    """

    def __init__(
        self, image_height=128, image_width=128, num_actions=2, init_std=0.5, max_log_std=0.7,
        sensor_input_size=0,
    ):
        super().__init__()

        self.num_actions = num_actions
        self.sensor_input_size = sensor_input_size

        # CNN: 2 conv layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)

        # Compute flattened CNN output size from input dimensions
        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_height, image_width)
            dummy = self.conv2(self.conv1(dummy))
            conv_out_size = dummy.numel()

        # FC layers (CNN features + optional sensor data)
        self.fc1 = nn.Linear(conv_out_size + sensor_input_size, 128)
        self.fc2 = nn.Linear(128, num_actions)
        self.value_fc = nn.Linear(128, 1)  # Value head for PPO

        # Learnable log_std (state-independent), clamped in forward()
        self.log_std = nn.Parameter(torch.ones(num_actions) * np.log(init_std))
        self.max_log_std = max_log_std

    def _backbone(self, x, sensors=None):
        """
        Shared CNN+FC backbone.

        Args:
            x: Input image (B, H, W, 3) in range [0, 1]
            sensors: Optional sensor data (B, sensor_dim)

        Returns:
            features: (B, 128) features after fc1+relu
        """
        x = x.permute(0, 3, 1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)
        if sensors is not None and self.sensor_input_size > 0:
            x = torch.cat([x, sensors], dim=-1)
        return torch.relu(self.fc1(x))

    def forward(self, x, sensors=None):
        """
        Forward pass - returns action distribution parameters.

        Args:
            x: Input image (B, H, W, 3) in range [0, 1]
            sensors: Optional sensor data (B, sensor_dim)

        Returns:
            mean: (B, 2) unbounded mean (tanh applied at sampling)
            std: (2,) standard deviation (shared across batch)
        """
        features = self._backbone(x, sensors=sensors)
        mean = self.fc2(features)
        clamped_log_std = self.log_std.clamp(max=self.max_log_std)
        std = torch.exp(clamped_log_std)
        return mean, std

    def evaluate_actions(self, observations, actions, sensors=None):
        """
        Single forward pass returning everything PPO needs.

        Args:
            observations: (B, H, W, 3) observations
            actions: (B, 2) tanh-squashed actions
            sensors: Optional (B, sensor_dim) sensor data

        Returns:
            log_probs: (B,) log probability of actions
            values: (B,) state value estimates
            entropy: scalar mean entropy
        """
        features = self._backbone(observations, sensors=sensors)

        # Action head
        mean = self.fc2(features)
        clamped_log_std = self.log_std.clamp(max=self.max_log_std)
        std = torch.exp(clamped_log_std)

        # Value head
        values = self.value_fc(features).squeeze(-1)

        # Log prob with tanh correction
        z = torch.atanh(actions.clamp(-1 + 1e-6, 1 - 1e-6))
        dist = torch.distributions.Normal(mean, std)
        gaussian_log_prob = dist.log_prob(z).sum(dim=-1)
        log_probs = _tanh_log_prob(gaussian_log_prob, z)

        # Entropy
        entropy = dist.entropy().sum(dim=-1).mean()

        return log_probs, values, entropy

    def sample_action(self, x, sensors=None):
        """
        Sample an action using tanh-squashed Gaussian.

        Args:
            x: Input image (B, H, W, 3)
            sensors: Optional sensor data (B, sensor_dim)

        Returns:
            action: (B, 2) sampled actions in (-1, 1)
            log_prob: (B,) log probability of the sampled actions
        """
        mean, std = self.forward(x, sensors=sensors)
        dist = torch.distributions.Normal(mean, std)
        z = dist.sample()
        gaussian_log_prob = dist.log_prob(z).sum(dim=-1)
        action = torch.tanh(z)
        log_prob = _tanh_log_prob(gaussian_log_prob, z)
        return action, log_prob

    def get_deterministic_action(self, x, sensors=None):
        """
        Get deterministic action (tanh of mean).

        Args:
            x: Input image (B, H, W, 3)
            sensors: Optional sensor data (B, sensor_dim)

        Returns:
            action: (B, 2) mean actions in (-1, 1)
        """
        mean, _ = self.forward(x, sensors=sensors)
        return torch.tanh(mean)

    def log_prob(self, x, action, sensors=None):
        """
        Compute log probability of given tanh-squashed actions.

        Args:
            x: Input image (B, H, W, 3)
            action: (B, 2) tanh-squashed actions to evaluate
            sensors: Optional (B, sensor_dim) sensor data

        Returns:
            log_prob: (B,) log probability of actions
        """
        mean, std = self.forward(x, sensors=sensors)
        z = torch.atanh(action.clamp(-1 + 1e-6, 1 - 1e-6))
        dist = torch.distributions.Normal(mean, std)
        gaussian_log_prob = dist.log_prob(z).sum(dim=-1)
        return _tanh_log_prob(gaussian_log_prob, z)


class MLPPolicy(nn.Module):
    """
    MLP policy for sensor-only continuous control (no CNN, no LSTM).

    Separate actor/critic networks with tanh activations.
    Includes running observation normalization as registered buffers.
    """

    def __init__(
        self,
        num_actions=6,
        hidden_size=256,
        init_std=1.0,
        max_log_std=0.7,
        sensor_input_size=18,
        # Unused — kept for interface compatibility with CNN policies
        image_height=64,
        image_width=64,
    ):
        super().__init__()

        self.num_actions = num_actions
        self.sensor_input_size = sensor_input_size

        # Actor network
        self.actor_fc1 = nn.Linear(sensor_input_size, hidden_size)
        self.actor_fc2 = nn.Linear(hidden_size, hidden_size)
        self.actor_out = nn.Linear(hidden_size, num_actions)

        # Critic network (separate weights)
        self.critic_fc1 = nn.Linear(sensor_input_size, hidden_size)
        self.critic_fc2 = nn.Linear(hidden_size, hidden_size)
        self.critic_out = nn.Linear(hidden_size, 1)

        # Learnable log_std (state-independent)
        self.log_std = nn.Parameter(torch.ones(num_actions) * np.log(init_std))
        self.max_log_std = max_log_std

        # Running observation normalizer (Welford's online algorithm)
        self.register_buffer("obs_mean", torch.zeros(sensor_input_size))
        self.register_buffer("obs_var", torch.ones(sensor_input_size))
        self.register_buffer("obs_count", torch.tensor(1e-4))

    def update_normalizer(self, sensor_batch):
        """Update running mean/var from a batch of sensor observations.

        Args:
            sensor_batch: (N, sensor_dim) tensor of sensor observations
        """
        batch_mean = sensor_batch.mean(dim=0)
        batch_var = sensor_batch.var(dim=0)
        batch_count = sensor_batch.shape[0]

        delta = batch_mean - self.obs_mean
        total_count = self.obs_count + batch_count
        new_mean = self.obs_mean + delta * batch_count / total_count
        m_a = self.obs_var * self.obs_count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.obs_count * batch_count / total_count
        new_var = m2 / total_count

        self.obs_mean.copy_(new_mean)
        self.obs_var.copy_(new_var)
        self.obs_count.copy_(total_count)

    def _normalize(self, sensors):
        """Normalize sensor observations using running stats."""
        return ((sensors - self.obs_mean) / torch.sqrt(self.obs_var + 1e-8)).clamp(-10, 10)

    def _actor(self, sensors):
        x = self._normalize(sensors)
        x = torch.tanh(self.actor_fc1(x))
        x = torch.tanh(self.actor_fc2(x))
        return self.actor_out(x)

    def _critic(self, sensors):
        x = self._normalize(sensors)
        x = torch.tanh(self.critic_fc1(x))
        x = torch.tanh(self.critic_fc2(x))
        return self.critic_out(x).squeeze(-1)

    def forward(self, x, sensors=None):
        """Forward pass — returns action mean and std. Image x is ignored."""
        mean = self._actor(sensors)
        clamped_log_std = self.log_std.clamp(max=self.max_log_std)
        std = torch.exp(clamped_log_std)
        return mean, std

    def evaluate_actions(self, observations, actions, sensors=None):
        """Single forward pass returning log_probs, values, entropy."""
        mean = self._actor(sensors)
        values = self._critic(sensors)
        clamped_log_std = self.log_std.clamp(max=self.max_log_std)
        std = torch.exp(clamped_log_std)

        z = torch.atanh(actions.clamp(-1 + 1e-6, 1 - 1e-6))
        dist = torch.distributions.Normal(mean, std)
        gaussian_log_prob = dist.log_prob(z).sum(dim=-1)
        log_probs = _tanh_log_prob(gaussian_log_prob, z)
        entropy = dist.entropy().sum(dim=-1).mean()

        return log_probs, values, entropy

    def sample_action(self, x, sensors=None):
        """Sample tanh-squashed Gaussian action. Image x is ignored."""
        mean, std = self.forward(x, sensors=sensors)
        dist = torch.distributions.Normal(mean, std)
        z = dist.sample()
        gaussian_log_prob = dist.log_prob(z).sum(dim=-1)
        action = torch.tanh(z)
        log_prob = _tanh_log_prob(gaussian_log_prob, z)
        return action, log_prob

    def get_deterministic_action(self, x, sensors=None):
        """Get deterministic action (tanh of mean). Image x is ignored."""
        mean, _ = self.forward(x, sensors=sensors)
        return torch.tanh(mean)
