"""
Minimal training script for 2-wheeler robot.

Starts with a trivially small neural network to validate the training loop.
Can be extended with more sophisticated networks and RL algorithms.
"""

import subprocess
import sys
import time
from collections import deque
from datetime import datetime

import numpy as np
import rerun as rr
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import rerun_logger
import wandb
from config import Config
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


class LSTMPolicy(nn.Module):
    """
    LSTM-based stochastic policy network with memory.

    Input: RGB image (64x64x3)
    Output: Mean of Gaussian distribution over motor commands [left, right]

    Uses CNN to extract features, then LSTM to maintain temporal context.
    This allows the policy to remember past observations and actions.
    """

    def __init__(self, image_height=64, image_width=64, hidden_size=64, init_std=0.5):
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

        # Learnable log_std (state-independent)
        self.log_std = nn.Parameter(torch.ones(2) * np.log(init_std))

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
            mean: (B, 2) or (B, T, 2) mean motor commands in range [-1, 1]
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

        # Output layer
        if is_sequence:
            mean = torch.tanh(self.fc(lstm_out))  # (B, T, 2)
        else:
            mean = torch.tanh(self.fc(lstm_out.squeeze(1)))  # (B, 2)

        std = torch.exp(self.log_std)
        return mean, std

    def sample_action(self, x):
        """
        Sample an action from the policy distribution.

        Args:
            x: Input image (B, H, W, 3)

        Returns:
            action: (B, 2) sampled actions, clamped to [-1, 1]
            log_prob: (B,) log probability of the sampled actions
        """
        mean, std = self.forward(x)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        # Clamp action to valid range
        action = torch.clamp(action, -1.0, 1.0)
        return action, log_prob

    def get_deterministic_action(self, x):
        """
        Get deterministic action (mean of distribution, no sampling).

        Used for evaluation to measure true policy capability without
        exploration noise.

        Args:
            x: Input image (B, H, W, 3)

        Returns:
            action: (B, 2) mean actions, clamped to [-1, 1]
        """
        mean, _ = self.forward(x)
        return torch.clamp(mean, -1.0, 1.0)

    def log_prob(self, x, action):
        """
        Compute log probability of given actions for a sequence.

        For REINFORCE training, we need to recompute log probs for the
        entire episode. This resets hidden state and processes sequentially.

        Args:
            x: Input images (T, H, W, 3) - full episode
            action: (T, 2) actions to evaluate

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

        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(action).sum(dim=-1)


class TinyPolicy(nn.Module):
    """
    Stochastic policy network for REINFORCE.

    Input: RGB image (64x64x3)
    Output: Mean of Gaussian distribution over motor commands [left, right]

    The policy is stochastic: actions are sampled from N(mean, std).
    std is a learnable parameter (not state-dependent for simplicity).
    """

    def __init__(self, image_height=64, image_width=64, init_std=0.5):
        super().__init__()

        # CNN: 2 conv layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)  # 64x64 -> 15x15
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)  # 15x15 -> 6x6

        # Calculate flattened size
        conv_out_size = 64 * 6 * 6  # 2304

        # FC layers
        self.fc1 = nn.Linear(conv_out_size, 128)
        self.fc2 = nn.Linear(128, 2)  # Output: mean of [left_motor, right_motor]

        # Learnable log_std (state-independent)
        # Using log_std ensures std is always positive when we exp() it
        self.log_std = nn.Parameter(torch.ones(2) * np.log(init_std))

    def forward(self, x):
        """
        Forward pass - returns action distribution parameters.

        Args:
            x: Input image (B, H, W, 3) in range [0, 1]

        Returns:
            mean: (B, 2) mean motor commands in range [-1, 1]
            std: (2,) standard deviation (shared across batch)
        """
        # Permute to (B, 3, H, W) for PyTorch conv layers
        x = x.permute(0, 3, 1, 2)

        # Conv layers with ReLU
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))

        # Flatten
        x = x.reshape(x.size(0), -1)

        # FC layers
        x = torch.relu(self.fc1(x))
        mean = torch.tanh(self.fc2(x))  # Tanh to get [-1, 1] range

        std = torch.exp(self.log_std)  # Ensure positive
        return mean, std

    def sample_action(self, x):
        """
        Sample an action from the policy distribution.

        Args:
            x: Input image (B, H, W, 3)

        Returns:
            action: (B, 2) sampled actions, clamped to [-1, 1]
            log_prob: (B,) log probability of the sampled actions
        """
        mean, std = self.forward(x)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)  # Sum over action dimensions

        # Clamp action to valid range
        action = torch.clamp(action, -1.0, 1.0)
        return action, log_prob

    def get_deterministic_action(self, x):
        """
        Get deterministic action (mean of distribution, no sampling).

        Used for evaluation to measure true policy capability without
        exploration noise.

        Args:
            x: Input image (B, H, W, 3)

        Returns:
            action: (B, 2) mean actions, clamped to [-1, 1]
        """
        mean, _ = self.forward(x)
        return torch.clamp(mean, -1.0, 1.0)

    def log_prob(self, x, action):
        """
        Compute log probability of given actions.

        Args:
            x: Input image (B, H, W, 3)
            action: (B, 2) actions to evaluate

        Returns:
            log_prob: (B,) log probability of actions
        """
        mean, std = self.forward(x)
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(action).sum(dim=-1)


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


def train_step_batched(policy, optimizer, episode_batch, gamma=0.99):
    """
    REINFORCE policy gradient training step on a batch of episodes.

    Computes reward-to-go for all episodes, normalizes advantages across
    the entire batch (so good episodes get positive advantage, bad episodes
    get negative), then takes one optimizer step.

    Args:
        policy: Stochastic neural network policy
        optimizer: PyTorch optimizer
        episode_batch: List of episode_data dicts from collect_episode
        gamma: Discount factor for reward-to-go

    Returns:
        avg_loss: Average loss across batch
        grad_norm: Gradient norm after averaging
        policy_std: Current policy standard deviation
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
    total_loss = 0.0
    for episode_data, advantage in zip(episode_batch, all_rtg):
        observations = torch.from_numpy(np.array(episode_data["observations"]))
        actions = torch.from_numpy(np.array(episode_data["actions"]))

        log_probs = policy.log_prob(observations, actions)
        loss = -torch.mean(advantage * log_probs)

        (loss / len(episode_batch)).backward()
        total_loss += loss.item()

    avg_loss = total_loss / len(episode_batch)

    # Clip gradients and record norm
    total_grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
    total_grad_norm = total_grad_norm.item()

    optimizer.step()

    # Get current policy std for logging
    policy_std = torch.exp(policy.log_std).detach().cpu().numpy()

    return avg_loss, total_grad_norm, policy_std


def main():
    """Main training loop."""
    # Load centralized configuration
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
        ).to(device)
    else:
        policy = TinyPolicy(
            image_height=cfg.policy.image_height,
            image_width=cfg.policy.image_width,
            init_std=cfg.policy.init_std,
        ).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=cfg.training.learning_rate)

    # Count parameters
    num_params = sum(p.numel() for p in policy.parameters())
    print(f"  Policy parameters: {num_params:,}")
    print(f"  Device: {device}")
    print()

    # Generate run notes using Claude (summarizes git changes)
    print("Generating run notes...")
    run_notes = generate_run_notes()
    if run_notes:
        print("  Run notes generated successfully")
    print()

    # Initialize wandb with config from centralized config
    run_name = (
        f"{cfg.policy.policy_type.lower()}-{datetime.now().strftime('%m%d-%H%M')}"
    )
    wandb.init(
        project="mindsim-2wheeler",
        name=run_name,
        notes=run_notes,
        config={
            **cfg.to_wandb_config(),
            # Add computed values not in config
            "policy_params": num_params,
        },
    )
    print(f"  Logging to W&B: {wandb.run.url}")
    print()

    # Watch model for gradient/parameter histograms
    # log_freq is in backward passes (episodes), not steps
    wandb.watch(policy, log="all", log_freq=10)
    print("  Watching model gradients every 10 episodes")

    # Initialize Rerun-WandB integration
    rr_wandb = RerunWandbLogger(recordings_dir="recordings")
    print(f"  Rerun recordings: {rr_wandb.run_dir}/")
    print()

    # Training loop - use config values
    batch_size = cfg.training.batch_size
    log_rerun_every = cfg.training.log_rerun_every

    # Curriculum config (shorthand for readability)
    curr = cfg.curriculum

    # Rolling window for success rate tracking
    success_history = deque(maxlen=curr.window_size)
    curriculum_progress = 0.0  # Start with target in front
    mastery_count = 0  # Count of consecutive batches at mastery level

    print(
        f"Training until curriculum mastery (curriculum=1.0, success>={cfg.training.mastery_threshold:.0%} for {cfg.training.mastery_batches} batches)..."
    )
    print(
        f"  Curriculum: performance-based (advance@{curr.advance_threshold:.0%}, retreat@{curr.retreat_threshold:.0%})"
    )
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
    pbar = tqdm(desc="Training", position=0, unit="batch")
    while not mastered:
        # Set curriculum progress for this batch
        env.set_curriculum_progress(curriculum_progress)

        # Update terminal title and progress indicator (use curriculum as progress)
        progress_pct = 100 * curriculum_progress
        set_terminal_title(
            f"{progress_pct:.0f}% curr={curriculum_progress:.2f} {run_name}"
        )
        set_terminal_progress(progress_pct)

        # Collect a batch of episodes
        episode_batch = []
        batch_rewards = []
        batch_distances = []
        batch_steps = []
        batch_successes = []

        # Determine if we should log to Rerun this batch (from eval, not training)
        log_every_n_batches = max(1, log_rerun_every // batch_size)
        should_log_rerun_this_batch = batch_idx % log_every_n_batches == 0

        episode_pbar = tqdm(
            range(batch_size), desc="  Collecting", leave=False, position=1
        )
        for _ in episode_pbar:
            t_collect_start = time.perf_counter()
            episode_data = collect_episode(
                env,
                policy,
                device,
                show_progress=False,
                log_rerun=False,  # Never log training episodes to Rerun
            )
            timing["collect"] += time.perf_counter() - t_collect_start

            episode_batch.append(episode_data)
            batch_rewards.append(episode_data["total_reward"])
            batch_distances.append(episode_data["final_distance"])
            batch_steps.append(episode_data["steps"])
            batch_successes.append(episode_data["success"])

            episode_pbar.set_postfix({"r": f"{episode_data['total_reward']:.2f}"})

        episode_pbar.close()
        episode_count += batch_size

        # Train on batch of episodes
        t_train_start = time.perf_counter()
        loss, grad_norm, policy_std = train_step_batched(
            policy, optimizer, episode_batch
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
                curriculum_progress = min(1.0, curriculum_progress + curr.advance_rate)
            elif rate_for_curriculum < curr.retreat_threshold:
                curriculum_progress = max(0.0, curriculum_progress - curr.retreat_rate)

        # Check for mastery: curriculum at max AND maintaining high eval success rate
        mastery_rate = (
            rolling_eval_success_rate
            if curr.use_eval_for_curriculum
            else rolling_success_rate
        )
        if (
            curriculum_progress >= 1.0
            and mastery_rate >= cfg.training.mastery_threshold
        ):
            mastery_count += 1
            if mastery_count >= cfg.training.mastery_batches:
                mastered = True
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
            "curriculum/progress": curriculum_progress,
            "curriculum/max_angle_deviation_deg": curriculum_progress
            * 180,  # 0° to 180°
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
                "curr": f"{curriculum_progress:.2f}",
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
    set_terminal_title(f"Done: {run_name}")
    set_terminal_progress(-1)  # Clear progress indicator
    notify_completion(run_name)
    wandb.finish()
    env.close()
    print("Training complete!")


if __name__ == "__main__":
    main()
