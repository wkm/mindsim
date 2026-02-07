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
    Trivially small neural network for proof of concept.

    Input: RGB image (64x64x3)
    Output: Motor commands [left, right]
    """

    def __init__(self, image_height=64, image_width=64):
        super().__init__()

        # Tiny CNN: just 2 conv layers
        self.conv1 = nn.Conv2d(3, 8, kernel_size=8, stride=4)  # 64x64 -> 15x15
        self.conv2 = nn.Conv2d(8, 16, kernel_size=4, stride=2)  # 15x15 -> 6x6

        # Calculate flattened size
        conv_out_size = 16 * 6 * 6  # 576

        # Tiny FC layers
        self.fc1 = nn.Linear(conv_out_size, 32)
        self.fc2 = nn.Linear(32, 2)  # Output: [left_motor, right_motor]

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input image (B, H, W, 3) in range [0, 1]

        Returns:
            actions: (B, 2) motor commands in range [-1, 1]
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
        x = torch.tanh(self.fc2(x))  # Tanh to get [-1, 1] range

        return x


def collect_episode(env, policy, device='cpu', show_progress=False, log_rerun=False):
    """
    Run one episode and collect data.

    Args:
        env: TrainingEnv instance
        policy: Neural network policy
        device: torch device
        show_progress: Show progress bar for episode steps
        log_rerun: Log episode to Rerun for visualization

    Returns:
        episode_data: Dict with observations, actions, rewards, etc.
    """
    observations = []
    actions = []
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

        # Get action from policy
        with torch.no_grad():
            action = policy(obs_tensor).cpu().numpy()[0]  # (2,)

        # Store data
        observations.append(obs)
        actions.append(action)

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


def train_step(policy, optimizer, episode_data):
    """
    Simple training step using policy gradient approach.

    For now, just optimize to maximize total reward (very basic).

    Args:
        policy: Neural network
        optimizer: PyTorch optimizer
        episode_data: Data from collect_episode
    """
    observations = torch.from_numpy(np.array(episode_data['observations']))  # (T, H, W, 3)
    actions = torch.from_numpy(np.array(episode_data['actions']))  # (T, 2)
    rewards = torch.tensor(episode_data['rewards'])  # (T,)

    # Simple reward-to-go
    reward_to_go = torch.zeros_like(rewards)
    running_sum = 0
    for t in reversed(range(len(rewards))):
        running_sum += rewards[t]
        reward_to_go[t] = running_sum

    # Normalize rewards
    if reward_to_go.std() > 0:
        reward_to_go = (reward_to_go - reward_to_go.mean()) / (reward_to_go.std() + 1e-8)

    # Forward pass
    predicted_actions = policy(observations)  # (T, 2)

    # Loss: MSE between predicted and taken actions, weighted by reward
    # (This is a simplified policy gradient)
    loss = torch.mean(reward_to_go.unsqueeze(1) * (predicted_actions - actions) ** 2)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Compute gradient norm before optimizer step
    total_grad_norm = 0.0
    for p in policy.parameters():
        if p.grad is not None:
            total_grad_norm += p.grad.data.norm(2).item() ** 2
    total_grad_norm = total_grad_norm ** 0.5

    optimizer.step()

    return loss.item(), total_grad_norm


def main():
    """Main training loop."""
    print("=" * 60)
    print("Training 2-Wheeler Robot with Tiny Neural Network")
    print("=" * 60)
    print()

    # Create environment
    print("Creating environment...")
    env = TrainingEnv(
        render_width=64,
        render_height=64,
        max_episode_steps=100,  # 10 seconds at 10 Hz
    )
    print(f"  Observation shape: {env.observation_shape}")
    print(f"  Action shape: {env.action_shape}")
    print(f"  Control frequency: 10 Hz")
    print()

    # Create policy
    print("Creating tiny neural network...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy = TinyPolicy(image_height=64, image_width=64).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    # Count parameters
    num_params = sum(p.numel() for p in policy.parameters())
    print(f"  Policy parameters: {num_params:,}")
    print(f"  Device: {device}")
    print()

    # Initialize wandb with comprehensive config
    run_name = f"tinypolicy-{datetime.now().strftime('%m%d-%H%M')}"
    wandb.init(
        project="mindsim-2wheeler",
        name=run_name,
        config={
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
            "policy_params": num_params,
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
            # Training
            "optimizer": "Adam",
            "learning_rate": 1e-3,
            "algorithm": "simple_policy_gradient",
            "num_episodes": 1000,
            "log_rerun_every": 100,
        }
    )
    print(f"  Logging to W&B: {wandb.run.url}")
    print()

    # Watch model for gradient/parameter logging
    wandb.watch(policy, log="all", log_freq=100)
    print("  Watching model gradients every 100 steps")

    # Initialize Rerun-WandB integration
    rr_wandb = RerunWandbLogger(recordings_dir="recordings")
    print(f"  Rerun recordings: {rr_wandb.run_dir}/")
    print()

    # Training loop
    num_episodes = 1000  # Baseline run
    log_rerun_every = 100  # Log every 100th episode to Rerun
    print(f"Training for {num_episodes} episodes...")
    print(f"  Logging every {log_rerun_every} episodes to Rerun")
    print()

    pbar = tqdm(range(num_episodes), desc="Training", position=0)
    for episode in pbar:
        # Collect episode (log to Rerun periodically for visualization)
        should_log_rerun = (episode % log_rerun_every == 0)

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
        loss, grad_norm = train_step(policy, optimizer, episode_data)

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
