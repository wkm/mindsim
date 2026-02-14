"""
Minimal training script for 2-wheeler robot.

Starts with a trivially small neural network to validate the training loop.
Can be extended with more sophisticated networks and RL algorithms.
"""

import argparse
import math
import subprocess
import sys
import time
from collections import deque
from datetime import datetime

import numpy as np
import rerun as rr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import rerun_logger
import wandb
from checkpoint import load_checkpoint, resolve_resume_ref, save_checkpoint
from config import Config
from dashboard import AnsiDashboard, TuiDashboard
from parallel import ParallelCollector, resolve_num_workers
from rerun_wandb import RerunWandbLogger
from training_env import TrainingEnv
from tweaks import apply_tweaks, load_tweaks


def set_terminal_title(title):
    """Set terminal tab/window title using ANSI escape sequence."""
    sys.stdout.write(f"\033]0;{title}\007")
    sys.stdout.flush()


def set_terminal_progress(percent):
    """
    Set terminal progress indicator using OSC 9;4 sequence.

    Supported by iTerm2, Windows Terminal, and others.
    Shows progress bar in terminal tab.

    Args:
        percent: 0-100 for progress, or -1 to clear
    """
    if percent < 0:
        # Clear progress indicator
        sys.stdout.write("\033]9;4;0\007")
    else:
        # Set progress (state=1 means normal progress)
        sys.stdout.write(f"\033]9;4;1;{int(percent)}\007")
    sys.stdout.flush()


def notify_completion(run_name, message=None):
    """Show macOS notification and play sound when training completes."""
    if message is None:
        message = f"Training run '{run_name}' has finished."

    # macOS notification
    subprocess.run(
        [
            "osascript",
            "-e",
            f'display notification "{message}" with title "MindSim Training Complete" sound name "Glass"',
        ],
        check=False,
    )

    # Fallback beep in case notification sound doesn't play
    print("\a", end="", flush=True)


def generate_run_notes():
    """
    Use Claude CLI to generate a summary of what changed since last run.

    Returns:
        str: Markdown-formatted notes for W&B, or None if generation fails
    """
    try:
        # Get git info
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        # Get diff from parent commit
        diff = subprocess.run(
            ["git", "diff", "HEAD~1", "--stat"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        # Get full diff for context (limited to avoid token limits)
        full_diff = subprocess.run(
            ["git", "diff", "HEAD~1"], capture_output=True, text=True, check=True
        ).stdout[:4000]  # Limit to ~4k chars

        # Get recent commit message
        commit_msg = subprocess.run(
            ["git", "log", "-1", "--pretty=%B"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        # Build prompt for Claude CLI
        prompt = f"""Summarize what changed in this training run as a short bullet-point list for experiment tracking.

Branch: {branch}
Recent commit: {commit_msg}

Changes:
{diff}

Diff excerpt:
{full_diff}

Rules:
- Output ONLY markdown bullet points (- ...), nothing else
- 2-5 bullets covering: what changed, what hypothesis is being tested, key parameter/architecture differences from baseline
- Be concise and technical, each bullet one line
- No preamble, no headings, no trailing text"""

        # Call Claude CLI (handles auth automatically)
        result = subprocess.run(
            ["claude", "-p", prompt, "--model", "haiku"],
            capture_output=True,
            text=True,
            check=True,
        )
        summary = result.stdout.strip()

        # Format as markdown notes
        notes = f"""{summary}

---
**Branch:** `{branch}` | **Commit:** {commit_msg.split(chr(10))[0][:60]}"""
        return notes

    except Exception as e:
        print(f"  Note: Could not generate run notes: {e}")
        return None


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
        init_std=0.5,
        max_log_std=0.7,
    ):
        super().__init__()

        self.hidden_size = hidden_size

        # CNN feature extractor (same as TinyPolicy)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)

        # Compute flattened CNN output size from input dimensions
        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_height, image_width)
            dummy = self.conv2(self.conv1(dummy))
            conv_out_size = dummy.numel()

        # Compress CNN spatial features to compact vector before LSTM
        self.fc_embed = nn.Linear(conv_out_size, hidden_size)

        # LSTM for temporal memory (operates on compact feature vector)
        self.lstm = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size, batch_first=True
        )

        # Output layers
        self.fc = nn.Linear(hidden_size, 2)  # Output: mean of [left_motor, right_motor]
        self.value_fc = nn.Linear(hidden_size, 1)  # Value head for PPO

        # Learnable log_std (state-independent), clamped in forward()
        self.log_std = nn.Parameter(torch.ones(2) * np.log(init_std))
        self.max_log_std = max_log_std

        # Hidden state (will be set during episode)
        self.hidden = None

    def reset_hidden(self, batch_size=1, device="cpu"):
        """Reset LSTM hidden state at the start of each episode."""
        self.hidden = (
            torch.zeros(1, batch_size, self.hidden_size, device=device),
            torch.zeros(1, batch_size, self.hidden_size, device=device),
        )

    def _backbone(self, x, hidden=None):
        """
        Shared CNN+LSTM backbone.

        Args:
            x: Input image (B, H, W, 3) or (B, T, H, W, 3) for sequences
            hidden: Optional LSTM hidden state tuple (h, c)

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

    def forward(self, x, hidden=None):
        """
        Forward pass - returns action distribution parameters.

        Args:
            x: Input image (B, H, W, 3) or (B, T, H, W, 3) for sequences
            hidden: Optional LSTM hidden state tuple (h, c)

        Returns:
            mean: (B, 2) or (B, T, 2) unbounded mean (tanh applied at sampling)
            std: (2,) standard deviation (shared across batch)
        """
        features, hidden, is_sequence = self._backbone(x, hidden)
        mean = self.fc(features)
        clamped_log_std = self.log_std.clamp(max=self.max_log_std)
        std = torch.exp(clamped_log_std)
        return mean, std

    def evaluate_actions(self, observations, actions):
        """
        Single forward pass returning everything PPO needs.

        Processes full episode sequence with fresh hidden state.

        Args:
            observations: (T, H, W, 3) episode observations
            actions: (T, 2) tanh-squashed actions

        Returns:
            log_probs: (T,) log probability of actions
            values: (T,) state value estimates
            entropy: scalar mean entropy
        """
        device = observations.device
        self.reset_hidden(batch_size=1, device=device)

        # Process entire sequence: (1, T, H, W, 3)
        x = observations.unsqueeze(0)
        features, _, _ = self._backbone(x)  # (1, T, hidden_size)
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

    def sample_action(self, x):
        """
        Sample an action using tanh-squashed Gaussian.

        Samples z ~ Normal(mean, std), then action = tanh(z).
        Log-prob includes the Jacobian correction for the tanh transform.

        Args:
            x: Input image (B, H, W, 3)

        Returns:
            action: (B, 2) sampled actions in (-1, 1)
            log_prob: (B,) log probability of the sampled actions
        """
        mean, std = self.forward(x)
        dist = torch.distributions.Normal(mean, std)
        z = dist.sample()
        gaussian_log_prob = dist.log_prob(z).sum(dim=-1)
        action = torch.tanh(z)
        log_prob = _tanh_log_prob(gaussian_log_prob, z)
        return action, log_prob

    def get_deterministic_action(self, x):
        """
        Get deterministic action (tanh of mean).

        Used for evaluation to measure true policy capability without
        exploration noise.

        Args:
            x: Input image (B, H, W, 3)

        Returns:
            action: (B, 2) mean actions in (-1, 1)
        """
        mean, _ = self.forward(x)
        return torch.tanh(mean)

    def log_prob(self, x, action):
        """
        Compute log probability of given tanh-squashed actions for a sequence.

        Recovers pre-tanh values via atanh, then computes Gaussian log-prob
        with Jacobian correction.

        Args:
            x: Input images (T, H, W, 3) - full episode
            action: (T, 2) tanh-squashed actions to evaluate

        Returns:
            log_prob: (T,) log probability of actions
        """
        device = x.device

        # Reset hidden state for fresh forward pass
        self.reset_hidden(batch_size=1, device=device)

        # Process entire sequence at once
        x = x.unsqueeze(0)  # (1, T, H, W, 3)
        mean, std = self.forward(x)  # mean: (1, T, 2)
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
        self, image_height=128, image_width=128, init_std=0.5, max_log_std=0.7
    ):
        super().__init__()

        # CNN: 2 conv layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)

        # Compute flattened CNN output size from input dimensions
        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_height, image_width)
            dummy = self.conv2(self.conv1(dummy))
            conv_out_size = dummy.numel()

        # FC layers
        self.fc1 = nn.Linear(conv_out_size, 128)
        self.fc2 = nn.Linear(128, 2)  # Output: mean of [left_motor, right_motor]
        self.value_fc = nn.Linear(128, 1)  # Value head for PPO

        # Learnable log_std (state-independent), clamped in forward()
        self.log_std = nn.Parameter(torch.ones(2) * np.log(init_std))
        self.max_log_std = max_log_std

    def _backbone(self, x):
        """
        Shared CNN+FC backbone.

        Args:
            x: Input image (B, H, W, 3) in range [0, 1]

        Returns:
            features: (B, 128) features after fc1+relu
        """
        x = x.permute(0, 3, 1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)
        return torch.relu(self.fc1(x))

    def forward(self, x):
        """
        Forward pass - returns action distribution parameters.

        Args:
            x: Input image (B, H, W, 3) in range [0, 1]

        Returns:
            mean: (B, 2) unbounded mean (tanh applied at sampling)
            std: (2,) standard deviation (shared across batch)
        """
        features = self._backbone(x)
        mean = self.fc2(features)
        clamped_log_std = self.log_std.clamp(max=self.max_log_std)
        std = torch.exp(clamped_log_std)
        return mean, std

    def evaluate_actions(self, observations, actions):
        """
        Single forward pass returning everything PPO needs.

        Args:
            observations: (B, H, W, 3) observations
            actions: (B, 2) tanh-squashed actions

        Returns:
            log_probs: (B,) log probability of actions
            values: (B,) state value estimates
            entropy: scalar mean entropy
        """
        features = self._backbone(observations)

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

    def sample_action(self, x):
        """
        Sample an action using tanh-squashed Gaussian.

        Samples z ~ Normal(mean, std), then action = tanh(z).
        Log-prob includes the Jacobian correction for the tanh transform.

        Args:
            x: Input image (B, H, W, 3)

        Returns:
            action: (B, 2) sampled actions in (-1, 1)
            log_prob: (B,) log probability of the sampled actions
        """
        mean, std = self.forward(x)
        dist = torch.distributions.Normal(mean, std)
        z = dist.sample()
        gaussian_log_prob = dist.log_prob(z).sum(dim=-1)
        action = torch.tanh(z)
        log_prob = _tanh_log_prob(gaussian_log_prob, z)
        return action, log_prob

    def get_deterministic_action(self, x):
        """
        Get deterministic action (tanh of mean).

        Used for evaluation to measure true policy capability without
        exploration noise.

        Args:
            x: Input image (B, H, W, 3)

        Returns:
            action: (B, 2) mean actions in (-1, 1)
        """
        mean, _ = self.forward(x)
        return torch.tanh(mean)

    def log_prob(self, x, action):
        """
        Compute log probability of given tanh-squashed actions.

        Recovers pre-tanh values via atanh, then computes Gaussian log-prob
        with Jacobian correction.

        Args:
            x: Input image (B, H, W, 3)
            action: (B, 2) tanh-squashed actions to evaluate

        Returns:
            log_prob: (B,) log probability of actions
        """
        mean, std = self.forward(x)
        z = torch.atanh(action.clamp(-1 + 1e-6, 1 - 1e-6))
        dist = torch.distributions.Normal(mean, std)
        gaussian_log_prob = dist.log_prob(z).sum(dim=-1)
        return _tanh_log_prob(gaussian_log_prob, z)


def collect_episode(env, policy, device="cpu", log_rerun=False, deterministic=False):
    """
    Run one episode and collect data.

    Args:
        env: TrainingEnv instance
        policy: Neural network policy
        device: torch device
        log_rerun: Log episode to Rerun for visualization
        deterministic: If True, use mean actions (no sampling) for evaluation.
                       If False, sample from policy distribution for training.

    Returns:
        episode_data: Dict with observations, actions, rewards, log_probs (if not deterministic), etc.
    """
    # Rerun namespace depends on mode
    ns = "eval" if deterministic else "training"

    observations = []
    actions = []
    log_probs = []  # Only populated when not deterministic
    rewards = []
    distances = []

    obs = env.reset()
    env_config = env.last_reset_config

    # Reset LSTM hidden state if policy has one
    if hasattr(policy, "reset_hidden"):
        policy.reset_hidden(batch_size=1, device=device)
    done = False
    truncated = False
    total_reward = 0
    steps = 0
    info = {}

    # Track trajectory for Rerun
    trajectory_points = []

    # Set up video encoder for Rerun (H.264 instead of per-frame JPEG)
    video_encoder = None
    if log_rerun:
        video_encoder = rerun_logger.VideoEncoder(
            f"{ns}/camera",
            width=env.observation_shape[1],
            height=env.observation_shape[0],
        )

    while not (done or truncated):
        # Convert observation to torch tensor
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)

        # Get action (deterministic or stochastic)
        with torch.no_grad():
            if deterministic:
                action = policy.get_deterministic_action(obs_tensor)
                action = action.cpu().numpy()[0]
                log_prob = None
            else:
                action, log_prob = policy.sample_action(obs_tensor)
                action = action.cpu().numpy()[0]
                log_prob = log_prob.cpu().numpy()[0]

        # Store data
        observations.append(obs)
        actions.append(action)
        if not deterministic:
            log_probs.append(log_prob)

        # Take step
        obs, reward, done, truncated, info = env.step(action)
        rewards.append(reward)
        distances.append(info["distance"])
        total_reward += reward

        # Log to Rerun in real-time
        if log_rerun:
            rr.set_time("step", sequence=steps)
            video_encoder.log_frame(observations[-1])
            rr.log(f"{ns}/action/left_motor", rr.Scalars([action[0]]))
            rr.log(f"{ns}/action/right_motor", rr.Scalars([action[1]]))
            rr.log(f"{ns}/reward/total", rr.Scalars([reward]))
            rr.log(f"{ns}/reward/cumulative", rr.Scalars([total_reward]))
            rr.log(f"{ns}/distance_to_target", rr.Scalars([info["distance"]]))

            # Log body transforms
            rerun_logger.log_body_transforms(env, namespace=ns)

            # Build and log trajectory
            trajectory_points.append(info["position"])
            if len(trajectory_points) > 1:
                rr.log(
                    f"{ns}/trajectory",
                    rr.LineStrips3D([trajectory_points], colors=[[100, 200, 100]]),
                )

        steps += 1

    # Flush video encoder and log episode summary
    if log_rerun:
        video_encoder.flush()
        rr.log(f"{ns}/episode/total_reward", rr.Scalars([total_reward]))
        rr.log(f"{ns}/episode/final_distance", rr.Scalars([info["distance"]]))
        rr.log(f"{ns}/episode/steps", rr.Scalars([steps]))

    # Compute action statistics
    actions_array = np.array(actions)
    left_actions = actions_array[:, 0]
    right_actions = actions_array[:, 1]

    # Determine if episode was a success (reached target)
    success = done and info["distance"] < env.success_distance

    result = {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "distances": distances,
        "total_reward": total_reward,
        "steps": steps,
        "final_distance": info["distance"],
        "success": success,
        "env_config": env_config,
        "done": done,
        "truncated": truncated,
        "patience_truncated": info.get("patience_truncated", False),
        # Action statistics for logging
        "left_motor_mean": float(np.mean(left_actions)),
        "left_motor_std": float(np.std(left_actions)),
        "left_motor_min": float(np.min(left_actions)),
        "left_motor_max": float(np.max(left_actions)),
        "right_motor_mean": float(np.mean(right_actions)),
        "right_motor_std": float(np.std(right_actions)),
        "right_motor_min": float(np.min(right_actions)),
        "right_motor_max": float(np.max(right_actions)),
    }

    # Store final observation for GAE bootstrapping on truncated episodes
    if truncated and not done:
        result["final_observation"] = obs

    # Only include log_probs for training episodes
    if not deterministic:
        result["log_probs"] = log_probs

    return result


def replay_episode(env, episode_data):
    """
    Replay a recorded episode with Rerun logging.

    Resets the environment to the saved configuration and replays the
    exact actions from the episode, logging each step to Rerun under
    the "worst" namespace.

    Args:
        env: TrainingEnv instance
        episode_data: Dict from collect_episode (must include env_config and actions)
    """
    ns = "worst"

    env.reset_to_config(episode_data["env_config"])

    video_encoder = rerun_logger.VideoEncoder(
        f"{ns}/camera",
        width=env.observation_shape[1],
        height=env.observation_shape[0],
    )

    trajectory_points = []
    total_reward = 0

    for step_idx, action in enumerate(episode_data["actions"]):
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        rr.set_time("step", sequence=step_idx)
        video_encoder.log_frame(episode_data["observations"][step_idx])
        rr.log(f"{ns}/action/left_motor", rr.Scalars([action[0]]))
        rr.log(f"{ns}/action/right_motor", rr.Scalars([action[1]]))
        rr.log(f"{ns}/reward/total", rr.Scalars([reward]))
        rr.log(f"{ns}/reward/cumulative", rr.Scalars([total_reward]))
        rr.log(f"{ns}/distance_to_target", rr.Scalars([info["distance"]]))

        rerun_logger.log_body_transforms(env, namespace=ns)

        trajectory_points.append(info["position"])
        if len(trajectory_points) > 1:
            rr.log(
                f"{ns}/trajectory",
                rr.LineStrips3D([trajectory_points], colors=[[255, 100, 100]]),
            )

        if done or truncated:
            break

    video_encoder.flush()

    # Log episode summary
    rr.log(f"{ns}/episode/total_reward", rr.Scalars([total_reward]))
    rr.log(f"{ns}/episode/final_distance", rr.Scalars([info["distance"]]))
    rr.log(f"{ns}/episode/steps", rr.Scalars([step_idx + 1]))


def compute_reward_to_go(rewards, gamma=0.99):
    """
    Compute discounted reward-to-go for a single episode.

    Args:
        rewards: Tensor of per-step rewards (T,)
        gamma: Discount factor

    Returns:
        reward_to_go: Tensor of discounted returns (T,)
    """
    reward_to_go = torch.zeros_like(rewards)
    running_sum = 0
    for t in reversed(range(len(rewards))):
        running_sum = rewards[t] + gamma * running_sum
        reward_to_go[t] = running_sum
    return reward_to_go


def compute_gae(rewards, values, gamma, gae_lambda, next_value=0.0):
    """
    Compute Generalized Advantage Estimation (GAE).

    Args:
        rewards: Tensor of per-step rewards (T,)
        values: Tensor of per-step value estimates (T,)
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        next_value: Bootstrap value for the state after the last step
                    (0 for terminated episodes, V(s_final) for truncated)

    Returns:
        advantages: Tensor of GAE advantages (T,)
        returns: Tensor of GAE returns / value targets (T,)
    """
    T = len(rewards)
    advantages = torch.zeros(T, dtype=rewards.dtype, device=rewards.device)
    gae = 0.0
    for t in reversed(range(T)):
        next_val = next_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_val - values[t]
        gae = delta + gamma * gae_lambda * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


def train_step_batched(
    policy, optimizer, episode_batch, gamma=0.99, entropy_coeff=0.01
):
    """
    REINFORCE policy gradient training step on a batch of episodes.

    Computes reward-to-go for all episodes, normalizes advantages across
    the entire batch (so good episodes get positive advantage, bad episodes
    get negative), then takes one optimizer step.

    Includes an entropy bonus to prevent policy collapse: when advantage
    signal is weak (all episodes similar), the entropy term provides a
    non-zero gradient that keeps exploration alive.

    Args:
        policy: Stochastic neural network policy
        optimizer: PyTorch optimizer
        episode_batch: List of episode_data dicts from collect_episode
        gamma: Discount factor for reward-to-go
        entropy_coeff: Weight for entropy bonus (0 = disabled)

    Returns:
        avg_loss: Average loss across batch
        grad_norm: Gradient norm after averaging
        policy_std: Current policy standard deviation
        entropy: Policy entropy value
    """
    optimizer.zero_grad()

    # First pass: compute reward-to-go for all episodes
    all_rtg = []
    for episode_data in episode_batch:
        rewards = torch.tensor(episode_data["rewards"], dtype=torch.float32)
        all_rtg.append(compute_reward_to_go(rewards, gamma))

    # Normalize advantages across the entire batch
    all_rtg_cat = torch.cat(all_rtg)
    batch_mean = all_rtg_cat.mean()
    batch_std = all_rtg_cat.std()
    if batch_std > 1e-8:
        all_rtg = [(rtg - batch_mean) / (batch_std + 1e-8) for rtg in all_rtg]

    # Second pass: compute losses with batch-normalized advantages
    # Weight each episode by its length so every timestep contributes equally,
    # preventing long episodes from dominating the gradient.
    total_steps = sum(len(ep["rewards"]) for ep in episode_batch)
    total_loss = 0.0
    for episode_data, advantage in zip(episode_batch, all_rtg):
        observations = torch.from_numpy(np.array(episode_data["observations"]))
        actions = torch.from_numpy(np.array(episode_data["actions"]))

        log_probs = policy.log_prob(observations, actions)
        # Sum (not mean) within episode, then divide by total batch timesteps
        loss = -torch.sum(advantage * log_probs) / total_steps

        loss.backward()
        total_loss += loss.item()

    avg_loss = total_loss / len(episode_batch)

    # Clip REINFORCE gradients before adding entropy bonus.
    # This prevents large REINFORCE gradients from drowning out the
    # entropy signal — without this, clip_grad_norm_ scales everything
    # together and the small entropy gradient on log_std gets zeroed out.
    total_grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
    total_grad_norm = total_grad_norm.item()

    # Entropy bonus: H(π) = 0.5 * (1 + log(2π)) + log_std per action dim
    # Applied AFTER clipping so the entropy gradient reaches log_std at
    # full strength every step, preventing std collapse.
    if entropy_coeff > 0:
        clamped_log_std = policy.log_std.clamp(max=policy.max_log_std)
        std = torch.exp(clamped_log_std)
        entropy = torch.distributions.Normal(torch.zeros_like(std), std).entropy().sum()
        entropy_loss = -entropy_coeff * entropy
        entropy_loss.backward()
        entropy_val = entropy.item()
    else:
        entropy_val = 0.0

    optimizer.step()

    # Get current policy std for logging
    clamped = policy.log_std.clamp(max=policy.max_log_std)
    policy_std = torch.exp(clamped).detach().cpu().numpy()

    return avg_loss, total_grad_norm, policy_std, entropy_val


def train_step_ppo(
    policy,
    optimizer,
    episode_batch,
    gamma=0.99,
    gae_lambda=0.95,
    clip_epsilon=0.2,
    ppo_epochs=4,
    entropy_coeff=0.01,
    value_coeff=0.5,
    max_grad_norm=0.5,
):
    """
    PPO training step on a batch of episodes.

    Phase 1: Compute GAE advantages (once, before epochs).
    Phase 2: K optimization epochs over the data with clipped surrogate objective.

    Args:
        policy: Stochastic neural network policy with evaluate_actions()
        optimizer: PyTorch optimizer
        episode_batch: List of episode_data dicts from collect_episode
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        clip_epsilon: PPO clip range
        ppo_epochs: Number of optimization passes over the data
        entropy_coeff: Entropy bonus coefficient
        value_coeff: Value loss coefficient
        max_grad_norm: Max gradient norm for clipping

    Returns:
        policy_loss: Average policy loss across epochs
        value_loss: Average value loss across epochs
        entropy: Average entropy across epochs
        grad_norm: Average gradient norm across epochs
        policy_std: Current policy standard deviation
        clip_fraction: Fraction of clipped ratios (last epoch)
        approx_kl: Approximate KL divergence (last epoch)
        explained_variance: How well V(s) predicts returns (1.0 = perfect)
        mean_value: Mean V(s) across batch
        mean_return: Mean GAE return across batch
    """
    device = next(policy.parameters()).device

    # Phase 1: Compute advantages with current policy (no grad)
    episode_data_tensors = []
    all_advantages = []
    all_returns = []

    with torch.no_grad():
        for ep in episode_batch:
            obs = torch.from_numpy(np.array(ep["observations"])).to(device)
            acts = torch.from_numpy(np.array(ep["actions"])).to(device)
            rewards = torch.tensor(ep["rewards"], dtype=torch.float32, device=device)
            old_log_probs = torch.tensor(
                ep["log_probs"], dtype=torch.float32, device=device
            )

            _, values, _ = policy.evaluate_actions(obs, acts)

            # Bootstrap value for truncated episodes
            next_value = 0.0
            if ep.get("truncated") and not ep.get("done") and "final_observation" in ep:
                final_obs = (
                    torch.from_numpy(ep["final_observation"]).unsqueeze(0).to(device)
                )  # (1, H, W, 3)
                dummy_act = torch.zeros(1, 2, device=device)  # (1, 2)
                _, final_val, _ = policy.evaluate_actions(final_obs, dummy_act)
                next_value = final_val[0].item()

            advantages, returns = compute_gae(
                rewards, values, gamma, gae_lambda, next_value
            )

            episode_data_tensors.append(
                {
                    "observations": obs,
                    "actions": acts,
                    "old_log_probs": old_log_probs,
                    "advantages": advantages,
                    "returns": returns,
                }
            )

            all_advantages.append(advantages)
            all_returns.append(returns)

    # Value function diagnostics (before normalization)
    all_adv_cat = torch.cat(all_advantages)
    all_ret_cat = torch.cat(all_returns)
    # Explained variance: 1 - Var(returns - values) / Var(returns)
    # values = returns - advantages (before normalization)
    all_val_cat = all_ret_cat - all_adv_cat
    ret_var = all_ret_cat.var()
    explained_variance = (
        1.0 - (all_ret_cat - all_val_cat).var() / (ret_var + 1e-8)
        if ret_var > 1e-8
        else 0.0
    )
    mean_value = all_val_cat.mean().item()
    mean_return = all_ret_cat.mean().item()

    # Normalize advantages across the entire batch
    adv_mean = all_adv_cat.mean()
    adv_std = all_adv_cat.std()
    if adv_std > 1e-8:
        for ep_tensors in episode_data_tensors:
            ep_tensors["advantages"] = (ep_tensors["advantages"] - adv_mean) / (
                adv_std + 1e-8
            )

    # Phase 2: PPO epochs
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_grad_norm = 0.0
    last_clip_fraction = 0.0
    last_approx_kl = 0.0

    for epoch in range(ppo_epochs):
        optimizer.zero_grad()

        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_entropy = 0.0
        epoch_clip_count = 0
        epoch_total_steps = 0
        epoch_kl_sum = 0.0

        total_steps = sum(len(ep["observations"]) for ep in episode_data_tensors)

        # Process episodes sequentially (LSTM hidden state requirement)
        for ep_tensors in episode_data_tensors:
            obs = ep_tensors["observations"]
            acts = ep_tensors["actions"]
            old_lp = ep_tensors["old_log_probs"]
            adv = ep_tensors["advantages"]
            ret = ep_tensors["returns"]
            T = len(obs)

            new_log_probs, values, entropy = policy.evaluate_actions(obs, acts)

            # PPO clipped surrogate
            ratio = torch.exp(new_log_probs - old_lp)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv
            policy_loss = -torch.min(surr1, surr2).sum() / total_steps

            # Value loss
            value_loss = F.mse_loss(values, ret, reduction="sum") / total_steps

            # Combined loss
            loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy
            loss.backward()

            # Track metrics
            epoch_policy_loss += policy_loss.item() * T
            epoch_value_loss += value_loss.item() * T
            epoch_entropy += entropy.item() * T

            with torch.no_grad():
                clipped = ((ratio - 1.0).abs() > clip_epsilon).float().sum().item()
                epoch_clip_count += clipped
                epoch_total_steps += T
                epoch_kl_sum += (old_lp - new_log_probs).mean().item() * T

        grad_norm = nn.utils.clip_grad_norm_(
            policy.parameters(), max_norm=max_grad_norm
        )
        optimizer.step()

        total_policy_loss += epoch_policy_loss / epoch_total_steps
        total_value_loss += epoch_value_loss / epoch_total_steps
        total_entropy += epoch_entropy / epoch_total_steps
        total_grad_norm += grad_norm.item()

        if epoch == ppo_epochs - 1:
            last_clip_fraction = epoch_clip_count / max(epoch_total_steps, 1)
            last_approx_kl = epoch_kl_sum / max(epoch_total_steps, 1)

    # Average across epochs
    avg_policy_loss = total_policy_loss / ppo_epochs
    avg_value_loss = total_value_loss / ppo_epochs
    avg_entropy = total_entropy / ppo_epochs
    avg_grad_norm = total_grad_norm / ppo_epochs

    # Get current policy std for logging
    clamped = policy.log_std.clamp(max=policy.max_log_std)
    policy_std = torch.exp(clamped).detach().cpu().numpy()

    ev = (
        explained_variance.item()
        if torch.is_tensor(explained_variance)
        else explained_variance
    )

    return (
        avg_policy_loss,
        avg_value_loss,
        avg_entropy,
        avg_grad_norm,
        policy_std,
        last_clip_fraction,
        last_approx_kl,
        ev,
        mean_value,
        mean_return,
    )


def log_episode_value_trace(
    policy, episode_data, gamma, gae_lambda, device="cpu", namespace="eval"
):
    """
    Run a forward pass on a completed episode to log V(s_t) and A(s_t) to Rerun.

    Gives per-step visibility into the value function's beliefs during the episode.
    """
    obs = torch.from_numpy(np.array(episode_data["observations"])).to(device)
    acts = torch.from_numpy(np.array(episode_data["actions"])).to(device)
    rewards = torch.tensor(episode_data["rewards"], dtype=torch.float32, device=device)

    with torch.no_grad():
        # Use evaluate_actions even for deterministic episodes — we just need values
        # Need dummy actions for the log_prob computation but we only use the values
        _, values, _ = policy.evaluate_actions(obs, acts)
        advantages, returns = compute_gae(
            rewards, values, gamma, gae_lambda, next_value=0.0
        )

    values_np = values.cpu().numpy()
    advantages_np = advantages.cpu().numpy()
    returns_np = returns.cpu().numpy()
    cumulative_reward = np.cumsum(episode_data["rewards"])

    for t in range(len(values_np)):
        rr.set_time("step", sequence=t)
        rr.log(f"{namespace}/value/V_s", rr.Scalars([values_np[t]]))
        rr.log(f"{namespace}/value/advantage", rr.Scalars([advantages_np[t]]))
        rr.log(
            f"{namespace}/value/cumulative_reward", rr.Scalars([cumulative_reward[t]])
        )
        rr.log(f"{namespace}/value/gae_return", rr.Scalars([returns_np[t]]))


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train 2-wheeler robot")
    parser.add_argument(
        "--smoketest",
        action="store_true",
        help="Run a fast end-to-end smoketest (tiny config, no wandb, no rerun)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers for episode collection (0=auto, 1=serial, default: from config)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint: local .pt path or wandb artifact ref (e.g. checkpoint-lstmpolicy:stage1-mastered)",
    )
    return parser.parse_args()


def _get_git_branch() -> str:
    try:
        return subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _drain_command_queue(queue, dashboard, stage_progress, curr, pending_save):
    """
    Drain commands from the TUI command queue.

    Returns:
        (stage_progress, pending_save, should_stop, force_rerun)
    """
    if queue is None:
        return stage_progress, pending_save, False, False

    should_stop = False
    force_rerun = False

    while not queue.empty():
        try:
            cmd = queue.get_nowait()
        except Exception:
            break

        if cmd == "checkpoint":
            pending_save = ("manual", [])
            dashboard.message("Checkpoint will be saved after this batch")
        elif cmd == "log_rerun":
            force_rerun = True
            dashboard.message("Rerun recording queued for next eval")
        elif cmd == "advance_curriculum":
            old = stage_progress
            stage_progress = min(1.0, stage_progress + 0.1)
            dashboard.message(f"Curriculum advanced: {old:.2f} -> {stage_progress:.2f}")
        elif cmd == "regress_curriculum":
            old = stage_progress
            stage_progress = max(0.0, stage_progress - 0.1)
            dashboard.message(
                f"Curriculum regressed: {old:.2f} -> {stage_progress:.2f}"
            )
        elif cmd == "stop":
            should_stop = True
            dashboard.message("Stopping after this batch...")
        # pause/unpause/step handled by _wait_if_paused

    return stage_progress, pending_save, should_stop, force_rerun


def _wait_if_paused(queue):
    """
    Block the training thread while paused.

    Checks the queue for unpause/step/stop commands.
    Returns True if training should stop, False otherwise.
    """
    if queue is None:
        return False

    paused = False

    # Check for pause command without blocking
    while not queue.empty():
        try:
            cmd = queue.get_nowait()
        except Exception:
            break
        if cmd == "pause":
            paused = True
        elif cmd == "unpause":
            paused = False
        elif cmd == "step":
            return False  # Run one batch then re-check
        elif cmd == "stop":
            return True
        else:
            # Put non-pause commands back for _drain_command_queue
            queue.put(cmd)

    # If paused, block until unpaused/stepped/stopped
    while paused:
        time.sleep(0.05)  # 50ms poll interval
        while not queue.empty():
            try:
                cmd = queue.get_nowait()
            except Exception:
                break
            if cmd == "unpause":
                paused = False
            elif cmd == "step":
                return False  # Run one batch
            elif cmd == "stop":
                return True
            elif cmd == "pause":
                pass  # Already paused
            else:
                queue.put(cmd)

    return False


def _train_loop(
    cfg,
    dashboard,
    smoketest=False,
    resume=None,
    num_workers_override=None,
    command_queue=None,
    app=None,
    log_fn=print,
):
    """
    Core training loop.

    Args:
        cfg: Config object
        dashboard: Dashboard instance (TuiDashboard or AnsiDashboard)
        smoketest: Whether this is a smoketest run
        resume: Resume ref string (local path or wandb artifact)
        num_workers_override: Override num_workers from config
        command_queue: Optional Queue for TUI commands
        app: Optional TUI app for pushing metadata
        log_fn: Function for print-style logging (print or dashboard.message).
                In TUI mode this is dashboard.message (shows in log area).
                In CLI mode this is print (shows in terminal).
    """
    # In TUI mode, verbose setup messages are noise in the log area.
    # Use _verbose for setup chatter, log_fn for important events only.
    is_tui = app is not None
    _verbose = (lambda msg: None) if is_tui else log_fn

    _verbose("=" * 60)
    _verbose(f"Training 2-Wheeler Robot ({cfg.training.algorithm})")
    _verbose("=" * 60)

    # Generate run notes using Claude (summarizes git changes)
    if smoketest:
        run_notes = None
    else:
        _verbose("Generating run notes...")
        run_notes = generate_run_notes()
        if run_notes:
            _verbose(run_notes)

    # Initialize wandb early so the run URL and summary are available
    run_name = (
        f"{cfg.policy.policy_type.lower()}-{datetime.now().strftime('%m%d-%H%M')}"
    )
    wandb_mode = "disabled" if smoketest else "online"
    wandb.init(
        project="mindsim-2wheeler",
        name=run_name,
        notes=run_notes,
        mode=wandb_mode,
        config=cfg.to_wandb_config(),
    )
    # Use "batch" as the x-axis instead of W&B's auto-incremented "step"
    wandb.define_metric("batch")
    wandb.define_metric("*", step_metric="batch")

    wandb_url = None
    if not smoketest and wandb.run:
        wandb_url = wandb.run.url
        _verbose(f"  W&B run: {wandb_url}")

    # Push run metadata to TUI header
    if app is not None:
        branch = _get_git_branch()
        app.call_from_thread(
            app.set_header, run_name, branch, cfg.training.algorithm, wandb_url
        )

    # Create environment from config
    _verbose("Creating environment...")
    env = TrainingEnv.from_config(cfg.env)
    _verbose(f"  Observation shape: {env.observation_shape}")
    _verbose(f"  Action shape: {env.action_shape}")
    _verbose(f"  Control frequency: {cfg.env.control_frequency_hz} Hz")

    # Log bot model info
    mj_model = env.env.model
    bot_scene = env.env.scene_path.name
    bot_info = (
        f"Bot: {bot_scene} | "
        f"{mj_model.nbody} bodies, {mj_model.njnt} joints, "
        f"{mj_model.nu} actuators, {mj_model.ncam} cameras"
    )
    _verbose(f"  {bot_info}")
    log_fn(bot_info)

    # Create policy from config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.policy.use_lstm:
        policy = LSTMPolicy(
            image_height=cfg.policy.image_height,
            image_width=cfg.policy.image_width,
            hidden_size=cfg.policy.hidden_size,
            init_std=cfg.policy.init_std,
            max_log_std=cfg.policy.max_log_std,
        ).to(device)
    else:
        policy = TinyPolicy(
            image_height=cfg.policy.image_height,
            image_width=cfg.policy.image_width,
            init_std=cfg.policy.init_std,
            max_log_std=cfg.policy.max_log_std,
        ).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=cfg.training.learning_rate)

    # Print architecture and parameter count
    num_params = sum(p.numel() for p in policy.parameters())
    _verbose(f"Policy ({device}): {num_params:,} parameters")
    log_fn(f"Policy: {cfg.policy.policy_type} ({num_params:,} params) on {device}")

    # Log model info to wandb
    wandb.config.update({"policy_params": num_params}, allow_val_change=True)

    # Watch model for gradient/parameter histograms
    # log_freq is in backward passes (episodes), not steps
    wandb.watch(policy, log="all", log_freq=10)

    # Resume from checkpoint if requested
    resumed_batch_idx = 0
    resumed_episode_count = 0
    resumed_curriculum_stage = None
    resumed_stage_progress = None
    resumed_mastery_count = None
    if resume:
        resume_ref = resolve_resume_ref(resume)
        log_fn(f"Loading checkpoint: {resume_ref}")
        ckpt = load_checkpoint(resume_ref, cfg, device=str(device))
        policy.load_state_dict(ckpt["policy_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        resumed_batch_idx = ckpt["batch_idx"]
        resumed_episode_count = ckpt["episode_count"]
        resumed_curriculum_stage = ckpt["curriculum_stage"]
        resumed_stage_progress = ckpt["stage_progress"]
        resumed_mastery_count = ckpt["mastery_count"]
        log_fn(
            f"  Resumed from batch {resumed_batch_idx}, episode {resumed_episode_count}"
        )
        _verbose(
            f"  Curriculum: stage {resumed_curriculum_stage}, progress {resumed_stage_progress:.2f}"
        )
        wandb.config.update({"resumed_from": resume_ref}, allow_val_change=True)
        # Add resume info to wandb notes
        resume_note = f"\n**Resumed from:** `{resume_ref}`"
        if run_notes:
            run_notes += resume_note
        else:
            run_notes = resume_note
        if wandb.run and not wandb.run.disabled:
            wandb.run.notes = run_notes

    # Initialize Rerun-WandB integration (skip in smoketest)
    rr_wandb = None
    if not smoketest:
        rr_wandb = RerunWandbLogger(recordings_dir="recordings")
        _verbose(f"  Rerun recordings: {rr_wandb.run_dir}/")

    # Set up parallel episode collection
    num_workers = (
        num_workers_override
        if num_workers_override is not None
        else cfg.training.num_workers
    )
    num_workers = resolve_num_workers(num_workers)
    collector = None
    if num_workers > 1:
        _verbose(f"Starting {num_workers} parallel workers...")
        collector = ParallelCollector(num_workers, cfg.env, cfg.policy)
        if is_tui:
            log_fn(f"Started {num_workers} parallel workers")
        else:
            _verbose("  Workers ready")
    else:
        _verbose("Using serial episode collection (num_workers=1)")

    # Training loop - use config values
    batch_size = cfg.training.batch_size
    log_rerun_every = cfg.training.log_rerun_every

    # Curriculum config (shorthand for readability)
    curr = cfg.curriculum

    # Rolling window for success rate tracking
    success_history = deque(maxlen=curr.window_size)
    curriculum_stage = (
        resumed_curriculum_stage if resumed_curriculum_stage is not None else 1
    )
    stage_progress = (
        resumed_stage_progress if resumed_stage_progress is not None else 0.0
    )
    mastery_count = resumed_mastery_count if resumed_mastery_count is not None else 0

    _verbose(
        f"Training {curr.num_stages}-stage curriculum until mastery (success>={cfg.training.mastery_threshold:.0%} for {cfg.training.mastery_batches} batches)..."
    )
    if is_tui:
        log_fn(
            f"Training started: {curr.num_stages}-stage curriculum, batch_size={batch_size}"
        )

    # Timing accumulators
    timing = {
        "collect": 0.0,
        "eval": 0.0,
        "train": 0.0,
        "log": 0.0,
        "rerun": 0.0,
    }

    # Rolling window for eval success rate (used for curriculum)
    eval_success_history = deque(maxlen=curr.window_size)

    episode_count = resumed_episode_count
    batch_idx = resumed_batch_idx
    mastered = False
    max_batches = cfg.training.max_batches
    batches_this_session = 0
    stop_requested = False
    while (
        not mastered
        and not stop_requested
        and (max_batches is None or batch_idx < max_batches)
    ):
        # Block while paused (checks for unpause/step/stop)
        if _wait_if_paused(command_queue):
            stop_requested = True
            dashboard.message("Stopping...")
            break

        batch_start_time = time.perf_counter()
        # Set curriculum stage for this batch
        env.set_curriculum_stage(curriculum_stage, stage_progress, curr.num_stages)

        # Check for live hyperparameter tweaks
        tweaks = load_tweaks()
        if tweaks:
            changes = apply_tweaks(cfg, optimizer, env, tweaks)
            batch_size = cfg.training.batch_size
            for name, old, new in changes:
                dashboard.message(f"  tweak: {name} {old} -> {new}")
            if changes:
                wandb.log({f"tweaks/{name}": new for name, _, new in changes})

        # Drain TUI command queue
        pending_save = None  # (trigger, aliases) or None
        force_rerun = False
        stage_progress, pending_save, stop_requested, force_rerun = (
            _drain_command_queue(
                command_queue, dashboard, stage_progress, curr, pending_save
            )
        )

        # Overall progress: (stage-1 + progress) / num_stages
        overall_progress = (curriculum_stage - 1 + stage_progress) / curr.num_stages
        progress_pct = 100 * overall_progress
        set_terminal_title(
            f"{progress_pct:.0f}% S{curriculum_stage} p={stage_progress:.2f} {run_name}"
        )
        set_terminal_progress(progress_pct)

        # Determine if we should log to Rerun this batch (from eval, not training)
        log_every_n_batches = max(1, log_rerun_every // batch_size)
        should_log_rerun_this_batch = rr_wandb is not None and (
            batch_idx % log_every_n_batches == 0 or force_rerun
        )

        # Collect a batch of episodes
        t_collect_start = time.perf_counter()
        if collector is not None:
            episode_batch = collector.collect_batch(
                policy,
                batch_size,
                curriculum_stage,
                stage_progress,
                num_stages=curr.num_stages,
            )
        else:
            episode_batch = []
            for _ in range(batch_size):
                episode_data = collect_episode(
                    env,
                    policy,
                    device,
                )
                episode_batch.append(episode_data)
        timing["collect_batch"] = time.perf_counter() - t_collect_start
        timing["collect"] += timing["collect_batch"]

        batch_rewards = [ep["total_reward"] for ep in episode_batch]
        batch_distances = [ep["final_distance"] for ep in episode_batch]
        batch_steps = [ep["steps"] for ep in episode_batch]
        batch_successes = [ep["success"] for ep in episode_batch]
        episode_count += batch_size

        # Train on batch of episodes
        t_train_start = time.perf_counter()
        if cfg.training.algorithm == "PPO":
            (
                policy_loss,
                value_loss,
                entropy,
                grad_norm,
                policy_std,
                clip_fraction,
                approx_kl,
                explained_variance,
                mean_value,
                mean_return,
            ) = train_step_ppo(
                policy,
                optimizer,
                episode_batch,
                gamma=cfg.training.gamma,
                gae_lambda=cfg.training.gae_lambda,
                clip_epsilon=cfg.training.clip_epsilon,
                ppo_epochs=cfg.training.ppo_epochs,
                entropy_coeff=cfg.training.entropy_coeff,
                value_coeff=cfg.training.value_coeff,
            )
            loss = policy_loss  # For backward-compatible logging
        else:
            loss, grad_norm, policy_std, entropy = train_step_batched(
                policy,
                optimizer,
                episode_batch,
                entropy_coeff=cfg.training.entropy_coeff,
            )
        timing["train_batch"] = time.perf_counter() - t_train_start
        timing["train"] += timing["train_batch"]

        # Aggregate batch statistics
        avg_reward = np.mean(batch_rewards)
        avg_distance = np.mean(batch_distances)
        avg_steps = np.mean(batch_steps)
        best_reward = np.max(batch_rewards)
        worst_reward = np.min(batch_rewards)

        # Track training success rate (for logging, not curriculum)
        batch_success_rate = np.mean(batch_successes)
        success_history.append(batch_success_rate)
        rolling_success_rate = np.mean(success_history)

        # Run deterministic evaluation episodes for curriculum decisions
        t_eval_start = time.perf_counter()
        eval_successes = []
        if curr.use_eval_for_curriculum:
            for eval_idx in range(curr.eval_episodes_per_batch):
                # Log first eval episode to Rerun if this is a logging batch
                log_this_eval = should_log_rerun_this_batch and (eval_idx == 0)

                if log_this_eval:
                    t_rerun_start = time.perf_counter()
                    rr_wandb.start_episode(episode_count, env, namespace="eval")
                    timing["rerun"] += time.perf_counter() - t_rerun_start

                eval_data = collect_episode(
                    env, policy, device, log_rerun=log_this_eval, deterministic=True
                )
                eval_successes.append(eval_data["success"])

                if log_this_eval:
                    t_rerun_start = time.perf_counter()
                    # Log per-step value function traces for PPO
                    if cfg.training.algorithm == "PPO":
                        log_episode_value_trace(
                            policy,
                            eval_data,
                            gamma=cfg.training.gamma,
                            gae_lambda=cfg.training.gae_lambda,
                            device=device,
                            namespace="eval",
                        )
                    rr_wandb.finish_episode(eval_data, upload_artifact=True)
                    timing["rerun"] += time.perf_counter() - t_rerun_start

            eval_success_rate = np.mean(eval_successes)
            eval_success_history.append(eval_success_rate)
            rolling_eval_success_rate = np.mean(eval_success_history)
        else:
            # Fall back to training success rate if eval disabled
            eval_success_rate = batch_success_rate
            rolling_eval_success_rate = rolling_success_rate
        timing["eval_batch"] = time.perf_counter() - t_eval_start
        timing["eval"] += timing["eval_batch"]

        # Replay worst training episode with Rerun logging
        if should_log_rerun_this_batch and rr_wandb is not None:
            t_rerun_start = time.perf_counter()
            worst_idx = int(np.argmin(batch_rewards))
            worst_ep = episode_batch[worst_idx]
            rr_wandb.start_episode(episode_count, env, namespace="worst")
            replay_episode(env, worst_ep)
            rr_wandb.finish_episode(worst_ep, upload_artifact=True)
            timing["rerun"] += time.perf_counter() - t_rerun_start

        # Update curriculum based on EVAL success rate (deterministic)
        if (
            len(eval_success_history) >= curr.window_size
            or not curr.use_eval_for_curriculum
        ):
            rate_for_curriculum = (
                rolling_eval_success_rate
                if curr.use_eval_for_curriculum
                else rolling_success_rate
            )
            if rate_for_curriculum > curr.advance_threshold:
                stage_progress = min(1.0, stage_progress + curr.advance_rate)

        # Check for stage mastery: progress=1.0 AND sustained high success rate
        # Checkpoint saves are deferred until after batch_idx is incremented
        # so the saved batch_idx represents "resume from here" correctly.
        # pending_save may already be set by command queue (manual checkpoint)
        mastery_rate = (
            rolling_eval_success_rate
            if curr.use_eval_for_curriculum
            else rolling_success_rate
        )
        if stage_progress >= 1.0 and mastery_rate >= cfg.training.mastery_threshold:
            mastery_count += 1
            if mastery_count >= cfg.training.mastery_batches:
                if curriculum_stage >= curr.num_stages:
                    # Final stage mastered → training complete
                    pending_save = ("final", [f"stage{curriculum_stage}-mastered"])
                    mastered = True
                else:
                    # Save before advancing to next stage
                    pending_save = ("milestone", [f"stage{curriculum_stage}-mastered"])
                    # Advance to next stage
                    curriculum_stage += 1
                    stage_progress = 0.0
                    mastery_count = 0
                    # Clear success histories so new stage starts fresh
                    success_history.clear()
                    eval_success_history.clear()
                    dashboard.message(
                        f"  >>> Advanced to stage {curriculum_stage}/{curr.num_stages} <<<"
                    )
        else:
            mastery_count = 0  # Reset if we drop below mastery level

        # Collect all actions from batch for histograms
        all_left_actions = np.concatenate(
            [np.array(ep["actions"])[:, 0] for ep in episode_batch]
        )
        all_right_actions = np.concatenate(
            [np.array(ep["actions"])[:, 1] for ep in episode_batch]
        )

        log_dict = {
            "episode": episode_count,
            "batch": batch_idx,
            # Curriculum
            "curriculum/stage": curriculum_stage,
            "curriculum/stage_progress": stage_progress,
            "curriculum/overall_progress": overall_progress,
            "curriculum/max_episode_steps": env.max_episode_steps,
            # Training success rate (stochastic, with exploration noise)
            "curriculum/train_batch_success_rate": batch_success_rate,
            "curriculum/train_rolling_success_rate": rolling_success_rate,
            # Eval success rate (deterministic, no exploration noise)
            "curriculum/eval_batch_success_rate": eval_success_rate,
            "curriculum/eval_rolling_success_rate": rolling_eval_success_rate,
            # Batch metrics
            "batch/avg_reward": avg_reward,
            "batch/best_reward": best_reward,
            "batch/worst_reward": worst_reward,
            "batch/avg_final_distance": avg_distance,
            "batch/avg_steps": avg_steps,
            "batch/success_rate": batch_success_rate,
            "batch/loss": loss,
            "training/grad_norm": grad_norm,
            "training/entropy": entropy,
            # Policy std (exploration level)
            "policy/std_left": policy_std[0],
            "policy/std_right": policy_std[1],
            # Action statistics (across entire batch)
            "actions/left_motor_mean": float(np.mean(all_left_actions)),
            "actions/left_motor_std": float(np.std(all_left_actions)),
            "actions/right_motor_mean": float(np.mean(all_right_actions)),
            "actions/right_motor_std": float(np.std(all_right_actions)),
            # Action histograms (distribution across batch)
            "actions/left_motor_hist": wandb.Histogram(
                all_left_actions.tolist(), num_bins=20
            ),
            "actions/right_motor_hist": wandb.Histogram(
                all_right_actions.tolist(), num_bins=20
            ),
            # Reward histogram across batch
            "batch/reward_hist": wandb.Histogram(batch_rewards, num_bins=20),
        }

        # Episode termination breakdown
        batch_truncated = [ep.get("truncated", False) for ep in episode_batch]
        batch_done = [ep.get("done", False) for ep in episode_batch]
        log_dict.update(
            {
                "batch/truncated_fraction": np.mean(batch_truncated),
                "batch/done_fraction": np.mean(batch_done),
                "batch/patience_truncated_fraction": np.mean(
                    [ep.get("patience_truncated", False) for ep in episode_batch]
                ),
            }
        )

        # PPO-specific metrics
        if cfg.training.algorithm == "PPO":
            log_dict.update(
                {
                    "training/policy_loss": policy_loss,
                    "training/value_loss": value_loss,
                    "training/clip_fraction": clip_fraction,
                    "training/approx_kl": approx_kl,
                    "training/explained_variance": explained_variance,
                    "training/mean_value": mean_value,
                    "training/mean_return": mean_return,
                }
            )

        t_log_start = time.perf_counter()
        wandb.log(log_dict)
        timing["log"] += time.perf_counter() - t_log_start

        # Compute batch timing
        batch_time = time.perf_counter() - batch_start_time

        # Update dashboard
        dash_metrics = {
            # Episode performance
            "avg_reward": avg_reward,
            "best_reward": best_reward,
            "worst_reward": worst_reward,
            "avg_distance": avg_distance,
            "avg_steps": avg_steps,
            # Success rates
            "rolling_eval_success_rate": rolling_eval_success_rate,
            "eval_success_rate": eval_success_rate,
            "batch_success_rate": batch_success_rate,
            # Optimization
            "grad_norm": grad_norm,
            "entropy": entropy,
            "policy_std": policy_std,
            # Curriculum
            "curriculum_stage": curriculum_stage,
            "num_stages": curr.num_stages,
            "stage_progress": stage_progress,
            "mastery_count": mastery_count,
            "mastery_batches": cfg.training.mastery_batches,
            "max_episode_steps": env.max_episode_steps,
            # Timing
            "batch_time": batch_time,
            "collect_time": timing["collect_batch"],
            "train_time": timing["train_batch"],
            "eval_time": timing["eval_batch"],
            "batch_size": batch_size,
        }
        if cfg.training.algorithm == "PPO":
            dash_metrics.update(
                {
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "clip_fraction": clip_fraction,
                    "approx_kl": approx_kl,
                    "explained_variance": explained_variance,
                    "mean_value": mean_value,
                    "mean_return": mean_return,
                }
            )
        else:
            dash_metrics["loss"] = loss

        dashboard.update(batch_idx, dash_metrics)
        batch_idx += 1
        batches_this_session += 1

        # Save checkpoints (milestone/final take priority over periodic)
        if pending_save:
            trigger, aliases = pending_save
            save_checkpoint(
                policy,
                optimizer,
                cfg,
                curriculum_stage,
                stage_progress,
                mastery_count,
                batch_idx,
                episode_count,
                trigger=trigger,
                aliases=aliases,
            )
        elif (
            cfg.training.checkpoint_every
            and batch_idx % cfg.training.checkpoint_every == 0
        ):
            save_checkpoint(
                policy,
                optimizer,
                cfg,
                curriculum_stage,
                stage_progress,
                mastery_count,
                batch_idx,
                episode_count,
                trigger="periodic",
            )

    dashboard.finish()

    # Print timing summary (verbose in CLI, compact in TUI)
    total_time = (
        timing["collect"]
        + timing["eval"]
        + timing["train"]
        + timing["log"]
        + timing["rerun"]
    )
    if total_time > 0 and batches_this_session > 0:
        _verbose("=" * 60)
        _verbose("Timing Summary")
        _verbose("=" * 60)
        _verbose(
            f"  Episode collection: {timing['collect']:>8.2f}s ({100 * timing['collect'] / total_time:>5.1f}%)"
        )
        _verbose(
            f"  Eval episodes:      {timing['eval']:>8.2f}s ({100 * timing['eval'] / total_time:>5.1f}%)"
        )
        _verbose(
            f"  Training step:      {timing['train']:>8.2f}s ({100 * timing['train'] / total_time:>5.1f}%)"
        )
        _verbose(
            f"  Wandb logging:      {timing['log']:>8.2f}s ({100 * timing['log'] / total_time:>5.1f}%)"
        )
        _verbose(
            f"  Rerun recording:    {timing['rerun']:>8.2f}s ({100 * timing['rerun'] / total_time:>5.1f}%)"
        )
        _verbose(f"  Total:              {total_time:>8.2f}s")
        _verbose(
            f"  Per-batch average ({batch_size} episodes/batch, {batches_this_session} batches):"
        )
        _verbose(
            f"    Collection: {1000 * timing['collect'] / batches_this_session:.1f}ms"
        )
        _verbose(
            f"    Training:   {1000 * timing['train'] / batches_this_session:.1f}ms"
        )

    # Clean up
    if collector is not None:
        collector.close()
    set_terminal_title(f"Done: {run_name}")
    set_terminal_progress(-1)  # Clear progress indicator
    if not smoketest:
        notify_completion(run_name)
    wandb.finish()
    env.close()
    if smoketest:
        log_fn(f"Smoketest passed! ({batch_idx} batches, {episode_count} episodes)")
    else:
        log_fn("Training complete!")


def run_training(
    app, command_queue, smoketest=False, resume=None, num_workers=None, scene_path=None
):
    """
    Entry point for TUI-driven training (called from worker thread).

    Args:
        app: MindSimApp instance
        command_queue: Queue for TUI commands
        smoketest: Whether to use smoketest config
        resume: Checkpoint resume reference
        num_workers: Worker count override
        scene_path: Override bot scene XML path
    """
    if smoketest:
        cfg = Config.for_smoketest()
    else:
        cfg = Config()

    if scene_path:
        cfg.env.scene_path = scene_path

    dashboard = TuiDashboard(
        app=app,
        total_batches=cfg.training.max_batches,
        algorithm=cfg.training.algorithm,
    )

    _train_loop(
        cfg=cfg,
        dashboard=dashboard,
        smoketest=smoketest,
        resume=resume,
        num_workers_override=num_workers,
        command_queue=command_queue,
        app=app,
        log_fn=dashboard.message,
    )


def main():
    """CLI entry point (headless, no TUI)."""
    args = parse_args()

    if args.smoketest:
        cfg = Config.for_smoketest()
        print("[SMOKETEST MODE] Running fast end-to-end validation...")
    else:
        cfg = Config()

    dashboard = AnsiDashboard(
        total_batches=cfg.training.max_batches,
        algorithm=cfg.training.algorithm,
    )

    _train_loop(
        cfg=cfg,
        dashboard=dashboard,
        smoketest=args.smoketest,
        resume=args.resume,
        num_workers_override=args.num_workers,
    )


if __name__ == "__main__":
    main()
