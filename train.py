"""
Minimal training script for 2-wheeler robot.

Starts with a trivially small neural network to validate the training loop.
Can be extended with more sophisticated networks and RL algorithms.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from tqdm import tqdm
import wandb
import rerun as rr
from training_env import TrainingEnv
import rerun_logger
from rerun_wandb import RerunWandbLogger


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

        # Tiny CNN: just 2 conv layers
        self.conv1 = nn.Conv2d(3, 8, kernel_size=8, stride=4)  # 64x64 -> 15x15
        self.conv2 = nn.Conv2d(8, 16, kernel_size=4, stride=2)  # 15x15 -> 6x6

        # Calculate flattened size
        conv_out_size = 16 * 6 * 6  # 576

        # Tiny FC layers
        self.fc1 = nn.Linear(conv_out_size, 32)
        self.fc2 = nn.Linear(32, 2)  # Output: mean of [left_motor, right_motor]

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


def collect_episode(env, policy, device='cpu', show_progress=False, log_rerun=False):
    """
    Run one episode and collect data.

    Args:
        env: TrainingEnv instance
        policy: Neural network policy (stochastic)
        device: torch device
        show_progress: Show progress bar for episode steps
        log_rerun: Log episode to Rerun for visualization

    Returns:
        episode_data: Dict with observations, actions, rewards, log_probs, etc.
    """
    observations = []
    actions = []
    log_probs = []
    rewards = []
    distances = []

    obs = env.reset()
    done = False
    truncated = False
    total_reward = 0
    steps = 0
    info = {}  # Initialize info dict

    # Optional progress bar for episode steps
    pbar = tqdm(total=env.max_episode_steps, desc="  Episode steps", leave=False, position=1) if show_progress else None

    # Track trajectory for Rerun
    trajectory_points = []

    while not (done or truncated):
        # Convert observation to torch tensor
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)  # (1, H, W, 3)

        # Sample action from stochastic policy
        with torch.no_grad():
            action, log_prob = policy.sample_action(obs_tensor)
            action = action.cpu().numpy()[0]  # (2,)
            log_prob = log_prob.cpu().numpy()[0]  # scalar

        # Store data
        observations.append(obs)
        actions.append(action)
        log_probs.append(log_prob)

        # Take step
        obs, reward, done, truncated, info = env.step(action)
        rewards.append(reward)
        distances.append(info['distance'])
        total_reward += reward

        # Log to Rerun in real-time (during the episode)
        if log_rerun:
            # Set step timeline
            rr.set_time("step", sequence=steps)

            # Log camera view
            rr.log("training/camera", rr.Image(observations[-1]))  # Log the obs BEFORE step

            # Log actions
            rr.log("training/action/left_motor", rr.Scalars([action[0]]))
            rr.log("training/action/right_motor", rr.Scalars([action[1]]))

            # Log reward and distance
            rr.log("training/reward", rr.Scalars([reward]))
            rr.log("training/distance_to_target", rr.Scalars([info['distance']]))

            # Log body transforms (uses current MuJoCo state)
            rerun_logger.log_body_transforms(env, namespace="training")

            # Build and log trajectory
            trajectory_points.append(info['position'])
            if len(trajectory_points) > 1:
                rr.log("training/trajectory", rr.LineStrips3D([trajectory_points], colors=[[100, 200, 100]]))

        steps += 1

        # Update progress bar
        if pbar:
            pbar.update(1)
            pbar.set_postfix({'reward': f"{total_reward:.3f}", 'dist': f"{info['distance']:.3f}m"})

    if pbar:
        pbar.close()

    # Log episode summary to Rerun
    if log_rerun:
        rr.log("training/episode_reward", rr.Scalars([total_reward]))
        rr.log("training/episode_distance", rr.Scalars([info['distance']]))

    # Compute action statistics
    actions_array = np.array(actions)
    left_actions = actions_array[:, 0]
    right_actions = actions_array[:, 1]

    return {
        'observations': observations,
        'actions': actions,
        'log_probs': log_probs,
        'rewards': rewards,
        'distances': distances,
        'total_reward': total_reward,
        'steps': steps,
        'final_distance': info['distance'],
        # Action statistics for logging
        'left_motor_mean': float(np.mean(left_actions)),
        'left_motor_std': float(np.std(left_actions)),
        'left_motor_min': float(np.min(left_actions)),
        'left_motor_max': float(np.max(left_actions)),
        'right_motor_mean': float(np.mean(right_actions)),
        'right_motor_std': float(np.std(right_actions)),
        'right_motor_min': float(np.min(right_actions)),
        'right_motor_max': float(np.max(right_actions)),
    }


def train_step(policy, optimizer, episode_data, gamma=0.99):
    """
    REINFORCE policy gradient training step.

    Maximizes expected reward by increasing log probability of actions
    that led to higher-than-average returns.

    Args:
        policy: Stochastic neural network policy
        optimizer: PyTorch optimizer
        episode_data: Data from collect_episode
        gamma: Discount factor for reward-to-go
    """
    observations = torch.from_numpy(np.array(episode_data['observations']))  # (T, H, W, 3)
    actions = torch.from_numpy(np.array(episode_data['actions']))  # (T, 2)
    rewards = torch.tensor(episode_data['rewards'], dtype=torch.float32)  # (T,)

    # Compute discounted reward-to-go
    reward_to_go = torch.zeros_like(rewards)
    running_sum = 0
    for t in reversed(range(len(rewards))):
        running_sum = rewards[t] + gamma * running_sum
        reward_to_go[t] = running_sum

    # Normalize advantages (reward-to-go centered and scaled)
    # This reduces variance and helps training stability
    advantage = reward_to_go
    if advantage.std() > 1e-8:
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

    # Compute log probabilities of the actions taken
    log_probs = policy.log_prob(observations, actions)  # (T,)

    # REINFORCE loss: -E[advantage * log_prob]
    # Negative because we want to maximize expected reward
    policy_loss = -torch.mean(advantage * log_probs)

    # Backward pass
    optimizer.zero_grad()
    policy_loss.backward()

    # Compute gradient norm before optimizer step
    total_grad_norm = 0.0
    for p in policy.parameters():
        if p.grad is not None:
            total_grad_norm += p.grad.data.norm(2).item() ** 2
    total_grad_norm = total_grad_norm ** 0.5

    optimizer.step()

    # Get current policy std for logging
    policy_std = torch.exp(policy.log_std).detach().cpu().numpy()

    return policy_loss.item(), total_grad_norm, policy_std


def main():
    """Main training loop."""
    print("=" * 60)
    print("Training 2-Wheeler Robot with Tiny Neural Network")
    print("=" * 60)
    print()

    # Single source of truth for all run configuration.
    # This dict is passed directly to wandb.init() and used throughout training.
    config = {
        # Environment
        "render_width": 64,
        "render_height": 64,
        "max_episode_steps": 100,
        "control_frequency_hz": 10,
        "mujoco_steps_per_action": 5,
        "success_distance": 0.3,
        "failure_distance": 5.0,
        # Model architecture
        "policy_type": "TinyPolicy",
        "conv1_out_channels": 8,
        "conv1_kernel": 8,
        "conv1_stride": 4,
        "conv2_out_channels": 16,
        "conv2_kernel": 4,
        "conv2_stride": 2,
        "fc1_size": 32,
        "fc2_size": 2,
        "activation": "relu",
        "output_activation": "tanh",
        "init_std": 0.5,
        # Training
        "optimizer": "Adam",
        "learning_rate": 1e-3,
        "algorithm": "REINFORCE",
        "gamma": 0.99,
        "num_episodes": 1000,
        "log_rerun_every": 100,
        # Logging
        "wandb_project": "mindsim-2wheeler",
        "wandb_watch_log": "all",
        "wandb_watch_log_freq": 100,
        "recordings_dir": "recordings",
    }

    # Create environment
    print("Creating environment...")
    env = TrainingEnv(
        render_width=config["render_width"],
        render_height=config["render_height"],
        max_episode_steps=config["max_episode_steps"],
    )
    print(f"  Observation shape: {env.observation_shape}")
    print(f"  Action shape: {env.action_shape}")
    print(f"  Control frequency: {config['control_frequency_hz']} Hz")
    print()

    # Create policy
    print("Creating tiny neural network...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy = TinyPolicy(
        image_height=config["render_height"],
        image_width=config["render_width"],
        init_std=config["init_std"],
    ).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=config["learning_rate"])

    # Count parameters and add to config
    num_params = sum(p.numel() for p in policy.parameters())
    config["policy_params"] = num_params
    print(f"  Policy parameters: {num_params:,}")
    print(f"  Device: {device}")
    print()

    # Initialize wandb — config dict is the single source of truth
    run_name = f"tinypolicy-{datetime.now().strftime('%m%d-%H%M')}"
    wandb.init(
        project=config["wandb_project"],
        name=run_name,
        config=config,
    )
    print(f"  Logging to W&B: {wandb.run.url}")
    print()

    # Watch model for gradient/parameter logging
    wandb.watch(policy, log=config["wandb_watch_log"], log_freq=config["wandb_watch_log_freq"])
    print(f"  Watching model gradients every {config['wandb_watch_log_freq']} steps")

    # Initialize Rerun-WandB integration
    rr_wandb = RerunWandbLogger(recordings_dir=config["recordings_dir"])
    print(f"  Rerun recordings: {rr_wandb.run_dir}/")
    print()

    # Training loop
    print(f"Training for {config['num_episodes']} episodes...")
    print(f"  Logging every {config['log_rerun_every']} episodes to Rerun")
    print()

    pbar = tqdm(range(config["num_episodes"]), desc="Training", position=0)
    for episode in pbar:
        # Collect episode (log to Rerun periodically for visualization)
        should_log_rerun = (episode % config["log_rerun_every"] == 0)

        # Start new Rerun recording for this episode
        if should_log_rerun:
            rr_wandb.start_episode(episode, env, namespace="training")

        episode_data = collect_episode(
            env, policy, device,
            show_progress=False,
            log_rerun=should_log_rerun
        )

        # Finish Rerun recording and upload to wandb
        if should_log_rerun:
            rr_wandb.finish_episode(episode_data, upload_artifact=True)

        # Train on episode
        loss, grad_norm, policy_std = train_step(policy, optimizer, episode_data, gamma=config["gamma"])

        # Log to wandb
        actions_array = np.array(episode_data['actions'])
        left_actions = actions_array[:, 0]
        right_actions = actions_array[:, 1]

        log_dict = {
            "episode": episode,
            # Episode metrics
            "episode/reward": episode_data['total_reward'],
            "episode/final_distance": episode_data['final_distance'],
            "episode/steps": episode_data['steps'],
            "episode/loss": loss,
            "training/grad_norm": grad_norm,
            # Policy std (exploration level)
            "policy/std_left": policy_std[0],
            "policy/std_right": policy_std[1],
            # Action statistics
            "actions/left_motor_mean": episode_data['left_motor_mean'],
            "actions/left_motor_std": episode_data['left_motor_std'],
            "actions/right_motor_mean": episode_data['right_motor_mean'],
            "actions/right_motor_std": episode_data['right_motor_std'],
            # Action histograms (distribution of actions in this episode)
            "actions/left_motor_hist": wandb.Histogram(left_actions.tolist(), num_bins=20),
            "actions/right_motor_hist": wandb.Histogram(right_actions.tolist(), num_bins=20),
            # Reward histogram
            "episode/reward_hist": wandb.Histogram(list(episode_data['rewards']), num_bins=20),
        }

        # Log action time series as a table (for line charts in wandb)
        # This creates a chart showing motor values (y) over time steps (x)
        if should_log_rerun:  # Only log detailed time series periodically
            action_table = wandb.Table(
                columns=["step", "left_motor", "right_motor", "reward", "distance"],
                data=[
                    [t, left_actions[t], right_actions[t], episode_data['rewards'][t], episode_data['distances'][t]]
                    for t in range(len(left_actions))
                ]
            )
            log_dict["episode/action_timeseries"] = action_table

            # Log a sample camera image from the episode
            sample_idx = len(episode_data['observations']) // 2  # Middle of episode
            sample_img = (episode_data['observations'][sample_idx] * 255).astype(np.uint8)
            log_dict["episode/sample_camera"] = wandb.Image(sample_img, caption=f"Episode {episode}, step {sample_idx}")

        wandb.log(log_dict)

        # Update progress bar
        pbar.set_postfix({
            'reward': f"{episode_data['total_reward']:.3f}",
            'dist': f"{episode_data['final_distance']:.3f}m",
            'steps': episode_data['steps'],
            'loss': f"{loss:.6f}",
        })

    # Clean up
    print()
    wandb.finish()
    env.close()
    print("Training complete!")


if __name__ == "__main__":
    main()
