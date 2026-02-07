"""
Minimal training script for 2-wheeler robot.

Starts with a trivially small neural network to validate the training loop.
Can be extended with more sophisticated networks and RL algorithms.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb
from training_env import TrainingEnv


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


def collect_episode(env, policy, device='cpu', show_progress=False):
    """
    Run one episode and collect data.

    Args:
        env: TrainingEnv instance
        policy: Neural network policy
        device: torch device
        show_progress: Show progress bar for episode steps

    Returns:
        episode_data: Dict with observations, actions, rewards, etc.
    """
    observations = []
    actions = []
    rewards = []

    obs = env.reset()
    done = False
    truncated = False
    total_reward = 0
    steps = 0

    # Optional progress bar for episode steps
    pbar = tqdm(total=env.max_episode_steps, desc="  Episode steps", leave=False, position=1) if show_progress else None

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
        total_reward += reward
        steps += 1

        # Update progress bar
        if pbar:
            pbar.update(1)
            pbar.set_postfix({'reward': f"{total_reward:.3f}", 'dist': f"{info['distance']:.3f}m"})

    if pbar:
        pbar.close()

    return {
        'observations': observations,
        'actions': actions,
        'rewards': rewards,
        'total_reward': total_reward,
        'steps': steps,
        'final_distance': info['distance'],
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
    optimizer.step()

    return loss.item()


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

    # Initialize wandb
    wandb.init(
        project="mindsim-2wheeler",
        config={
            "policy_params": num_params,
            "render_size": (64, 64),
            "max_episode_steps": 100,
            "control_frequency_hz": 10,
            "learning_rate": 1e-3,
        }
    )
    print(f"  Logging to W&B: {wandb.run.url}")
    print()

    # Training loop
    num_episodes = 1000
    print(f"Training for {num_episodes} episodes...")
    print()

    pbar = tqdm(range(num_episodes), desc="Training", position=0)
    for episode in pbar:
        # Collect episode (disable inner progress bar for speed)
        episode_data = collect_episode(env, policy, device, show_progress=False)

        # Train on episode
        loss = train_step(policy, optimizer, episode_data)

        # Log to wandb
        wandb.log({
            "episode": episode,
            "reward": episode_data['total_reward'],
            "distance": episode_data['final_distance'],
            "steps": episode_data['steps'],
            "loss": loss,
        })

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
