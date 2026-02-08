"""
Training-ready Gymnasium wrapper for the 2-wheeler robot.

Wraps SimpleWheelerEnv with:
- 10 Hz control frequency (1 action per 0.1 seconds)
- Reward function (distance-based)
- Episode termination logic
- Standard Gymnasium API (reset, step returning obs/reward/done/truncated/info)
"""
import mujoco
import numpy as np
from simple_wheeler_env import SimpleWheelerEnv

# Control frequency: 10 Hz (1 action every 0.1 seconds)
# MuJoCo timestep: 0.02s (50 Hz simulation)
# Therefore: 5 MuJoCo steps per action
MUJOCO_STEPS_PER_ACTION = 5


class TrainingEnv:
    """
    Gymnasium-style training environment for the 2-wheeler robot.

    Control: 10 Hz (10 actions per second)
    Episodes: 10 seconds (100 steps)

    Observation: Camera RGB image (H, W, 3) normalized to [0, 1]
    Action: [left_motor, right_motor] in range [-1, 1]
    Reward: Negative distance change to target
    """

    def __init__(
        self,
        render_width=64,
        render_height=64,
        max_episode_steps=100,  # 10 seconds at 10 Hz
        success_distance=0.3,  # Success if within 0.3m of target
        failure_distance=5.0,  # Failure if beyond 5m from target
        min_target_distance=0.8,  # Minimum spawn distance from robot
        max_target_distance=2.5,  # Maximum spawn distance from robot
    ):
        """
        Initialize training environment.

        Args:
            render_width: Camera image width
            render_height: Camera image height
            max_episode_steps: Maximum steps before episode truncation (default 100 = 10 seconds)
            success_distance: Distance threshold for success
            failure_distance: Distance threshold for failure
        """
        self.env = SimpleWheelerEnv(
            render_width=render_width,
            render_height=render_height
        )
        self.max_episode_steps = max_episode_steps
        self.success_distance = success_distance
        self.failure_distance = failure_distance
        self.min_target_distance = min_target_distance
        self.max_target_distance = max_target_distance

        # Episode tracking
        self.episode_step = 0
        self.prev_distance = None

        # Observation and action spaces (for reference)
        self.observation_shape = (render_height, render_width, 3)
        self.action_shape = (2,)  # [left_motor, right_motor]

    def reset(self):
        """
        Reset environment to initial state with randomized target position.

        Target is placed at a random angle around the robot, at a distance
        between min_target_distance and max_target_distance.

        Returns:
            observation: Camera image normalized to [0, 1]
        """
        camera_img = self.env.reset()
        self.episode_step = 0

        # Randomize target position
        # Random angle (full circle around the robot)
        angle = np.random.uniform(0, 2 * np.pi)
        # Random distance within bounds
        distance = np.random.uniform(self.min_target_distance, self.max_target_distance)

        # Calculate new target position (robot starts at origin)
        target_x = distance * np.cos(angle)
        target_y = distance * np.sin(angle)
        target_z = 0.08  # Keep same height as original (on ground)

        # Update target body position in MuJoCo model
        target_body_id = self.env.target_body_id
        self.env.model.body_pos[target_body_id] = [target_x, target_y, target_z]

        # Re-run forward kinematics to update positions
        mujoco.mj_forward(self.env.model, self.env.data)

        # Get updated camera image after target move
        camera_img = self.env.get_camera_image()

        self.prev_distance = self.env.get_distance_to_target()

        # Normalize to [0, 1]
        obs = camera_img.astype(np.float32) / 255.0
        return obs

    def step(self, action):
        """
        Take action and advance simulation.

        Runs MUJOCO_STEPS_PER_ACTION (5) MuJoCo steps to achieve 10 Hz control.

        Args:
            action: [left_motor, right_motor] in range [-1, 1]

        Returns:
            observation: Camera image normalized to [0, 1]
            reward: Float reward value
            done: Boolean indicating episode termination
            truncated: Boolean indicating episode timeout
            info: Dict with additional information
        """
        left_motor, right_motor = action

        # Run multiple MuJoCo steps with same action (10 Hz control)
        for _ in range(MUJOCO_STEPS_PER_ACTION):
            camera_img = self.env.step(left_motor, right_motor)

        # Get observation
        obs = camera_img.astype(np.float32) / 255.0

        # Calculate reward (negative distance change = reward for getting closer)
        current_distance = self.env.get_distance_to_target()
        reward = self.prev_distance - current_distance  # Positive if getting closer
        self.prev_distance = current_distance

        # Check termination conditions
        done = False
        truncated = False

        # Success: reached target
        if current_distance < self.success_distance:
            done = True
            reward += 10.0  # Bonus for reaching target

        # Failure: too far away
        elif current_distance > self.failure_distance:
            done = True
            reward -= 5.0  # Penalty for going too far

        # Timeout: max steps reached
        self.episode_step += 1
        if self.episode_step >= self.max_episode_steps:
            truncated = True

        # Additional info
        info = {
            'distance': current_distance,
            'position': self.env.get_bot_position(),
            'step': self.episode_step,
        }

        return obs, reward, done, truncated, info

    def close(self):
        """Clean up resources."""
        self.env.close()
