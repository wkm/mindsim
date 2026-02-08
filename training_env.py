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
        # Reward shaping coefficients
        movement_bonus=0.05,  # Small reward per meter moved (encourages exploration)
        time_penalty=0.005,  # Tiny penalty per step (discourages dawdling)
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
        self.movement_bonus = movement_bonus
        self.time_penalty = time_penalty

        # Episode tracking
        self.episode_step = 0
        self.prev_distance = None
        self.prev_position = None

        # Curriculum learning: controls target spawn angle variance
        # 0.0 = target always directly in front of camera
        # 1.0 = target at random angle (full circle)
        self.curriculum_progress = 1.0  # Default: full randomization

        # Observation and action spaces (for reference)
        self.observation_shape = (render_height, render_width, 3)
        self.action_shape = (2,)  # [left_motor, right_motor]

    def set_curriculum_progress(self, progress):
        """
        Set curriculum progress for target spawn positioning.

        Args:
            progress: Float in [0, 1]
                0.0 = target always directly in front of camera (+Y direction)
                1.0 = target at random angle (full 360° randomization)
        """
        self.curriculum_progress = np.clip(progress, 0.0, 1.0)

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

        # Randomize target position with curriculum-based angle constraint
        #
        # Coordinate system (looking down from above):
        #   +X = right
        #   +Y = backward (behind robot)
        #   -Y = forward (camera looks this way)
        #
        # Angles: 0=+X, π/2=+Y (behind), π=-X, -π/2=-Y (front)
        #
        # At progress=0: target directly in front (angle=-π/2)
        # At progress=1: target at any angle (full circle)
        front_angle = -np.pi / 2  # -Y direction = in front of camera
        max_deviation = self.curriculum_progress * np.pi  # 0 to π
        angle = np.random.uniform(front_angle - max_deviation, front_angle + max_deviation)

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
        self.prev_position = self.env.get_bot_position()

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

        # Get current state
        current_distance = self.env.get_distance_to_target()
        current_position = self.env.get_bot_position()

        # Calculate reward components:
        # 1. Distance reward: positive if getting closer to target
        distance_reward = self.prev_distance - current_distance

        # 2. Movement bonus: small reward for exploring (moving at all)
        distance_moved = np.linalg.norm(current_position - self.prev_position)
        exploration_reward = self.movement_bonus * distance_moved

        # 3. Time penalty: tiny cost per step to encourage efficiency
        time_cost = -self.time_penalty

        reward = distance_reward + exploration_reward + time_cost

        # Update state for next step
        self.prev_distance = current_distance
        self.prev_position = current_position

        # Check termination conditions
        done = False
        truncated = False

        # Simulation instability check (MuJoCo warnings or invalid state)
        has_warnings = np.any(self.env.data.warning.number > 0)
        has_nan = np.isnan(current_distance) or np.any(np.isnan(current_position))
        bot_z = current_position[2]
        out_of_bounds = bot_z < -0.5 or bot_z > 1.0  # Fell through floor or launched

        if has_warnings or has_nan or out_of_bounds:
            done = True
            reward -= 2.0  # Small penalty for unstable behavior
            # Reset warning counters for next episode
            self.env.data.warning.number[:] = 0

        # Success: reached target
        elif current_distance < self.success_distance:
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

        # Additional info (includes reward breakdown for visualization)
        info = {
            'distance': current_distance,
            'position': current_position,
            'step': self.episode_step,
            'distance_moved': distance_moved,
            # Reward components for visualization
            'reward_distance': distance_reward,
            'reward_exploration': exploration_reward,
            'reward_time': time_cost,
            'reward_total': reward,
            # Stability info
            'unstable': has_warnings or has_nan or out_of_bounds,
        }

        return obs, reward, done, truncated, info

    def close(self):
        """Clean up resources."""
        self.env.close()
