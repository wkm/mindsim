"""
Minimal training script for 2-wheeler robot.

Starts with a trivially small neural network to validate the training loop.
Can be extended with more sophisticated networks and RL algorithms.
"""

import argparse
import subprocess
import sys
import time
from collections import deque
from datetime import datetime

import math

import numpy as np
import rerun as rr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import rerun_logger
import wandb
from config import Config
from parallel import ParallelCollector, resolve_num_workers
from rerun_wandb import RerunWandbLogger
from training_env import TrainingEnv


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
        prompt = f"""Summarize this training run in 2-3 sentences for experiment tracking.

Branch: {branch}
Recent commit: {commit_msg}

Changes:
{diff}

Diff excerpt:
{full_diff}

Focus on: What hypothesis is being tested? What changed from the baseline?
Be concise and technical. Start directly with the summary, no preamble."""

        # Call Claude CLI (handles auth automatically)
        result = subprocess.run(
            ["claude", "-p", prompt, "--model", "haiku"],
            capture_output=True,
            text=True,
            check=True,
        )
        summary = result.stdout.strip()

        # Format as markdown notes
        notes = f"""## Run Summary (auto-generated)

{summary}

---
**Branch:** `{branch}`
**Commit:** {commit_msg.split(chr(10))[0][:60]}
"""
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

    Input: RGB image (64x64x3)
    Output: Mean of Gaussian distribution over motor commands [left, right]

    Uses CNN to extract features, then LSTM to maintain temporal context.
    This allows the policy to remember past observations and actions.
    """

    def __init__(self, image_height=64, image_width=64, hidden_size=64, init_std=0.5, min_log_std=-3.0, max_log_std=0.7):
        super().__init__()

        self.hidden_size = hidden_size

        # CNN feature extractor (same as TinyPolicy)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)  # 64x64 -> 15x15
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)  # 15x15 -> 6x6

        # Flattened CNN output size
        conv_out_size = 64 * 6 * 6  # 2304

        # LSTM for temporal memory
        self.lstm = nn.LSTM(
            input_size=conv_out_size, hidden_size=hidden_size, batch_first=True
        )

        # Output layers
        self.fc = nn.Linear(hidden_size, 2)  # Output: mean of [left_motor, right_motor]

        # Learnable log_std (state-independent), clamped in forward()
        self.log_std = nn.Parameter(torch.ones(2) * np.log(init_std))
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

        # Hidden state (will be set during episode)
        self.hidden = None

    def reset_hidden(self, batch_size=1, device="cpu"):
        """Reset LSTM hidden state at the start of each episode."""
        self.hidden = (
            torch.zeros(1, batch_size, self.hidden_size, device=device),
            torch.zeros(1, batch_size, self.hidden_size, device=device),
        )

    def forward(self, x, hidden=None):
        """
        Forward pass - returns action distribution parameters.

        Args:
            x: Input image (B, H, W, 3) or (B, T, H, W, 3) for sequences
            hidden: Optional LSTM hidden state tuple (h, c)

        Returns:
            mean: (B, 2) or (B, T, 2) unbounded mean (tanh applied at sampling)
            std: (2,) standard deviation (shared across batch)
            hidden: Updated hidden state
        """
        # Handle both single timestep and sequence inputs
        is_sequence = x.dim() == 5
        if is_sequence:
            B, T, H, W, C = x.shape
            # Reshape to process all frames through CNN
            x = x.reshape(B * T, H, W, C)
        else:
            B = x.size(0)
            T = 1

        # Permute to (B*T, 3, H, W) for PyTorch conv layers
        x = x.permute(0, 3, 1, 2)

        # CNN feature extraction
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))

        # Flatten CNN output
        x = x.reshape(x.size(0), -1)  # (B*T, 576)

        # Reshape for LSTM: (B, T, features)
        x = x.reshape(B, T, -1)

        # Use stored hidden if none provided
        if hidden is None:
            hidden = self.hidden

        # LSTM forward pass
        lstm_out, hidden = self.lstm(x, hidden)  # lstm_out: (B, T, hidden_size)

        # Store updated hidden state
        self.hidden = hidden

        # Output layer (unbounded mean; tanh squashing applied at action sampling)
        if is_sequence:
            mean = self.fc(lstm_out)  # (B, T, 2)
        else:
            mean = self.fc(lstm_out.squeeze(1))  # (B, 2)

        clamped_log_std = self.log_std.clamp(self.min_log_std, self.max_log_std)
        std = torch.exp(clamped_log_std)
        return mean, std

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

    Input: RGB image (64x64x3)
    Output: Mean of Gaussian distribution over motor commands [left, right]

    The policy is stochastic: actions are sampled from N(mean, std).
    std is a learnable parameter (not state-dependent for simplicity).
    """

    def __init__(self, image_height=64, image_width=64, init_std=0.5, min_log_std=-3.0, max_log_std=0.7):
        super().__init__()

        # CNN: 2 conv layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)  # 64x64 -> 15x15
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)  # 15x15 -> 6x6

        # Calculate flattened size
        conv_out_size = 64 * 6 * 6  # 2304

        # FC layers
        self.fc1 = nn.Linear(conv_out_size, 128)
        self.fc2 = nn.Linear(128, 2)  # Output: mean of [left_motor, right_motor]

        # Learnable log_std (state-independent), clamped in forward()
        self.log_std = nn.Parameter(torch.ones(2) * np.log(init_std))
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, x):
        """
        Forward pass - returns action distribution parameters.

        Args:
            x: Input image (B, H, W, 3) in range [0, 1]

        Returns:
            mean: (B, 2) unbounded mean (tanh applied at sampling)
            std: (2,) standard deviation (shared across batch)
        """
        # Permute to (B, 3, H, W) for PyTorch conv layers
        x = x.permute(0, 3, 1, 2)

        # Conv layers with ReLU
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))

        # Flatten
        x = x.reshape(x.size(0), -1)

        # FC layers (unbounded mean; tanh squashing applied at action sampling)
        x = torch.relu(self.fc1(x))
        mean = self.fc2(x)

        clamped_log_std = self.log_std.clamp(self.min_log_std, self.max_log_std)
        std = torch.exp(clamped_log_std)
        return mean, std

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


def collect_episode(
    env, policy, device="cpu", show_progress=False, log_rerun=False, deterministic=False
):
    """
    Run one episode and collect data.

    Args:
        env: TrainingEnv instance
        policy: Neural network policy
        device: torch device
        show_progress: Show progress bar for episode steps
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

    # Optional progress bar for episode steps
    pbar = (
        tqdm(
            total=env.max_episode_steps, desc="  Episode steps", leave=False, position=1
        )
        if show_progress
        else None
    )

    # Track trajectory for Rerun
    trajectory_points = []

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
            rr.log(f"{ns}/camera", rr.Image(observations[-1]))
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

        # Update progress bar
        if pbar:
            pbar.update(1)
            pbar.set_postfix(
                {"reward": f"{total_reward:.3f}", "dist": f"{info['distance']:.3f}m"}
            )

    if pbar:
        pbar.close()

    # Log episode summary to Rerun
    if log_rerun:
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

    # Only include log_probs for training episodes
    if not deterministic:
        result["log_probs"] = log_probs

    return result


def replay_episode(env, episode_data):
    """
    Replay a recorded episode with Rerun logging.

    Resets the environment to the saved configuration and replays the
    exact actions from the episode, logging each step to Rerun.

    Args:
        env: TrainingEnv instance
        episode_data: Dict from collect_episode (must include env_config and actions)
    """
    ns = "worst"

    env.reset_to_config(episode_data["env_config"])

    trajectory_points = []
    total_reward = 0

    for step_idx, action in enumerate(episode_data["actions"]):
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        rr.set_time("step", sequence=step_idx)
        # Log camera (use previous obs from episode data for consistency)
        rr.log(f"{ns}/camera", rr.Image(episode_data["observations"][step_idx]))
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
        clamped_log_std = policy.log_std.clamp(policy.min_log_std, policy.max_log_std)
        std = torch.exp(clamped_log_std)
        entropy = torch.distributions.Normal(torch.zeros_like(std), std).entropy().sum()
        entropy_loss = -entropy_coeff * entropy
        entropy_loss.backward()
        entropy_val = entropy.item()
    else:
        entropy_val = 0.0

    optimizer.step()

    # Get current policy std for logging
    clamped = policy.log_std.clamp(policy.min_log_std, policy.max_log_std)
    policy_std = torch.exp(clamped).detach().cpu().numpy()

    return avg_loss, total_grad_norm, policy_std, entropy_val


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
    return parser.parse_args()


def main():
    """Main training loop."""
    args = parse_args()

    # Load configuration
    if args.smoketest:
        cfg = Config.for_smoketest()
        print("[SMOKETEST MODE] Running fast end-to-end validation...")
    else:
        cfg = Config()

    print("=" * 60)
    print("Training 2-Wheeler Robot with Tiny Neural Network")
    print("=" * 60)
    print()

    # Create environment from config
    print("Creating environment...")
    env = TrainingEnv.from_config(cfg.env)
    print(f"  Observation shape: {env.observation_shape}")
    print(f"  Action shape: {env.action_shape}")
    print(f"  Control frequency: {cfg.env.control_frequency_hz} Hz")
    print()

    # Create policy from config
    print(
        f"Creating {'LSTM' if cfg.policy.use_lstm else 'feedforward'} neural network..."
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.policy.use_lstm:
        policy = LSTMPolicy(
            image_height=cfg.policy.image_height,
            image_width=cfg.policy.image_width,
            hidden_size=cfg.policy.hidden_size,
            init_std=cfg.policy.init_std,
            min_log_std=cfg.policy.min_log_std,
            max_log_std=cfg.policy.max_log_std,
        ).to(device)
    else:
        policy = TinyPolicy(
            image_height=cfg.policy.image_height,
            image_width=cfg.policy.image_width,
            init_std=cfg.policy.init_std,
            min_log_std=cfg.policy.min_log_std,
            max_log_std=cfg.policy.max_log_std,
        ).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=cfg.training.learning_rate)

    # Count parameters
    num_params = sum(p.numel() for p in policy.parameters())
    print(f"  Policy parameters: {num_params:,}")
    print(f"  Device: {device}")
    print()

    # Generate run notes using Claude (summarizes git changes)
    if args.smoketest:
        run_notes = None
    else:
        print("Generating run notes...")
        run_notes = generate_run_notes()
        if run_notes:
            print("  Run notes generated successfully")
        print()

    # Initialize wandb with config from centralized config
    run_name = (
        f"{cfg.policy.policy_type.lower()}-{datetime.now().strftime('%m%d-%H%M')}"
    )
    wandb_mode = "disabled" if args.smoketest else "online"
    wandb.init(
        project="mindsim-2wheeler",
        name=run_name,
        notes=run_notes,
        mode=wandb_mode,
        config={
            **cfg.to_wandb_config(),
            # Add computed values not in config
            "policy_params": num_params,
        },
    )
    # Use "batch" as the x-axis instead of W&B's auto-incremented "step"
    wandb.define_metric("batch")
    wandb.define_metric("*", step_metric="batch")
    if not args.smoketest:
        print(f"  Logging to W&B: {wandb.run.url}")
        print()

    # Watch model for gradient/parameter histograms
    # log_freq is in backward passes (episodes), not steps
    wandb.watch(policy, log="all", log_freq=10)
    if not args.smoketest:
        print("  Watching model gradients every 10 episodes")

    # Initialize Rerun-WandB integration (skip in smoketest)
    rr_wandb = None
    if not args.smoketest:
        rr_wandb = RerunWandbLogger(recordings_dir="recordings")
        print(f"  Rerun recordings: {rr_wandb.run_dir}/")
        print()

    # Set up parallel episode collection
    num_workers = (
        args.num_workers if args.num_workers is not None else cfg.training.num_workers
    )
    num_workers = resolve_num_workers(num_workers)
    collector = None
    if num_workers > 1:
        print(f"Starting {num_workers} parallel workers...")
        collector = ParallelCollector(num_workers, cfg.env, cfg.policy)
        print("  Workers ready")
        print()
    else:
        print("Using serial episode collection (num_workers=1)")
        print()

    # Training loop - use config values
    batch_size = cfg.training.batch_size
    log_rerun_every = cfg.training.log_rerun_every

    # Curriculum config (shorthand for readability)
    curr = cfg.curriculum

    # Rolling window for success rate tracking
    success_history = deque(maxlen=curr.window_size)
    curriculum_stage = 1  # Start at stage 1 (angle variance)
    stage_progress = 0.0  # Start with target in front
    mastery_count = 0  # Count of consecutive batches at mastery level

    print(
        f"Training {curr.num_stages}-stage curriculum until mastery (success>={cfg.training.mastery_threshold:.0%} for {cfg.training.mastery_batches} batches)..."
    )
    print(
        "  Stage 1: Angle variance | Stage 2: Moving target + distance | Stage 3: Distractors"
    )
    print(f"  Curriculum: monotonic ramp-up (advance@{curr.advance_threshold:.0%})")
    if curr.use_eval_for_curriculum:
        print(
            f"  Using deterministic eval ({curr.eval_episodes_per_batch} eps/batch) for curriculum decisions"
        )
    print(f"  Logging every {log_rerun_every} episodes to Rerun")
    print()

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

    episode_count = 0
    batch_idx = 0
    mastered = False
    max_batches = cfg.training.max_batches
    pbar = tqdm(desc="Training", position=0, unit="batch", total=max_batches)
    while not mastered and (max_batches is None or batch_idx < max_batches):
        # Set curriculum stage for this batch
        env.set_curriculum_stage(curriculum_stage, stage_progress)

        # Overall progress: (stage-1 + progress) / num_stages
        overall_progress = (curriculum_stage - 1 + stage_progress) / curr.num_stages
        progress_pct = 100 * overall_progress
        set_terminal_title(
            f"{progress_pct:.0f}% S{curriculum_stage} p={stage_progress:.2f} {run_name}"
        )
        set_terminal_progress(progress_pct)

        # Determine if we should log to Rerun this batch (from eval, not training)
        log_every_n_batches = max(1, log_rerun_every // batch_size)
        should_log_rerun_this_batch = (
            rr_wandb is not None and batch_idx % log_every_n_batches == 0
        )

        # Anneal episode length: ramp from short to full over N batches
        ep_anneal_t = min(
            1.0, batch_idx / max(1, cfg.training.episode_length_anneal_batches)
        )
        effective_max_steps = int(
            cfg.training.episode_length_start
            + (cfg.env.max_episode_steps - cfg.training.episode_length_start)
            * ep_anneal_t
        )
        env.max_episode_steps = effective_max_steps

        # Collect a batch of episodes
        t_collect_start = time.perf_counter()
        if collector is not None:
            episode_batch = collector.collect_batch(
                policy,
                batch_size,
                curriculum_stage,
                stage_progress,
                max_episode_steps=effective_max_steps,
            )
        else:
            episode_batch = []
            episode_pbar = tqdm(
                range(batch_size), desc="  Collecting", leave=False, position=1
            )
            for _ in episode_pbar:
                episode_data = collect_episode(
                    env,
                    policy,
                    device,
                    show_progress=False,
                    log_rerun=False,
                )
                episode_batch.append(episode_data)
                episode_pbar.set_postfix({"r": f"{episode_data['total_reward']:.2f}"})
            episode_pbar.close()
        timing["collect"] += time.perf_counter() - t_collect_start

        batch_rewards = [ep["total_reward"] for ep in episode_batch]
        batch_distances = [ep["final_distance"] for ep in episode_batch]
        batch_steps = [ep["steps"] for ep in episode_batch]
        batch_successes = [ep["success"] for ep in episode_batch]
        episode_count += batch_size

        # Train on batch of episodes
        t_train_start = time.perf_counter()
        # Anneal entropy coefficient: linear decay from start to end
        anneal_t = min(1.0, batch_idx / max(1, cfg.training.entropy_anneal_batches))
        entropy_coeff = (
            cfg.training.entropy_coeff_start
            + (cfg.training.entropy_coeff_end - cfg.training.entropy_coeff_start) * anneal_t
        )
        loss, grad_norm, policy_std, entropy = train_step_batched(
            policy,
            optimizer,
            episode_batch,
            entropy_coeff=entropy_coeff,
        )
        timing["train"] += time.perf_counter() - t_train_start

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
                    rr_wandb.finish_episode(eval_data, upload_artifact=True)
                    timing["rerun"] += time.perf_counter() - t_rerun_start

            eval_success_rate = np.mean(eval_successes)
            eval_success_history.append(eval_success_rate)
            rolling_eval_success_rate = np.mean(eval_success_history)
        else:
            # Fall back to training success rate if eval disabled
            eval_success_rate = batch_success_rate
            rolling_eval_success_rate = rolling_success_rate
        timing["eval"] += time.perf_counter() - t_eval_start

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
                    mastered = True
                else:
                    # Advance to next stage
                    curriculum_stage += 1
                    stage_progress = 0.0
                    mastery_count = 0
                    # Clear success histories so new stage starts fresh
                    success_history.clear()
                    eval_success_history.clear()
                    print(
                        f"\n  >>> Advanced to stage {curriculum_stage}/{curr.num_stages} <<<"
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
            "training/entropy_coeff": entropy_coeff,
            "training/max_episode_steps": effective_max_steps,
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

        t_log_start = time.perf_counter()
        wandb.log(log_dict)
        timing["log"] += time.perf_counter() - t_log_start

        # Update progress bar
        pbar.set_postfix(
            {
                "avg_r": f"{avg_reward:.2f}",
                "eval": f"{rolling_eval_success_rate:.0%}",
                "train": f"{rolling_success_rate:.0%}",
                "stage": f"{curriculum_stage}/{curr.num_stages}",
                "prog": f"{stage_progress:.2f}",
                "mstr": f"{mastery_count}/{cfg.training.mastery_batches}",
            }
        )
        pbar.update(1)
        batch_idx += 1

    pbar.close()
    total_batches = batch_idx  # Store final count for timing summary

    # Print timing summary
    total_time = sum(timing.values())
    print()
    print("=" * 60)
    print("Timing Summary")
    print("=" * 60)
    print(
        f"  Episode collection: {timing['collect']:>8.2f}s ({100 * timing['collect'] / total_time:>5.1f}%)"
    )
    print(
        f"  Eval episodes:      {timing['eval']:>8.2f}s ({100 * timing['eval'] / total_time:>5.1f}%)"
    )
    print(
        f"  Training step:      {timing['train']:>8.2f}s ({100 * timing['train'] / total_time:>5.1f}%)"
    )
    print(
        f"  Wandb logging:      {timing['log']:>8.2f}s ({100 * timing['log'] / total_time:>5.1f}%)"
    )
    print(
        f"  Rerun recording:    {timing['rerun']:>8.2f}s ({100 * timing['rerun'] / total_time:>5.1f}%)"
    )
    print("  " + "─" * 40)
    print(f"  Total:              {total_time:>8.2f}s")
    print()
    print(
        f"  Per-batch average ({batch_size} episodes/batch, {total_batches} batches):"
    )
    print(f"    Collection: {1000 * timing['collect'] / total_batches:.1f}ms")
    print(f"    Training:   {1000 * timing['train'] / total_batches:.1f}ms")
    print()

    # Clean up
    if collector is not None:
        collector.close()
    set_terminal_title(f"Done: {run_name}")
    set_terminal_progress(-1)  # Clear progress indicator
    if not args.smoketest:
        notify_completion(run_name)
    wandb.finish()
    env.close()
    if args.smoketest:
        print(f"Smoketest passed! ({batch_idx} batches, {episode_count} episodes)")
    else:
        print("Training complete!")


if __name__ == "__main__":
    main()
