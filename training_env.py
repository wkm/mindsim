"""
Training-ready Gymnasium wrapper for MuJoCo robot environments.

Wraps SimpleWheelerEnv with:
- 10 Hz control frequency (1 action per 0.1 seconds)
- Reward function (distance-based + optional biped rewards)
- Episode termination logic (+ optional fall detection)
- Standard Gymnasium API (reset, step returning obs/reward/done/truncated/info)
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from simple_wheeler_env import SimpleWheelerEnv

if TYPE_CHECKING:
    from config import EnvConfig


class TrainingEnv:
    """
    Gymnasium-style training environment for MuJoCo robots.

    Works with any bot — action dimensions and actuator names are
    discovered from the loaded MuJoCo model.

    Observation: Camera RGB image (H, W, 3) normalized to [0, 1]
    Action: Array of motor commands, length = num_actuators, each in [-1, 1]
    """

    @classmethod
    def from_config(cls, config: EnvConfig) -> TrainingEnv:
        """Create TrainingEnv from an EnvConfig object."""
        return cls(
            scene_path=config.scene_path,
            render_width=config.render_width,
            render_height=config.render_height,
            max_episode_steps=config.max_episode_steps,
            max_episode_steps_final=config.max_episode_steps_final,
            mujoco_steps_per_action=config.mujoco_steps_per_action,
            success_distance=config.success_distance,
            failure_distance=config.failure_distance,
            min_target_distance=config.min_target_distance,
            max_target_distance=config.max_target_distance,
            distance_reward_scale=config.distance_reward_scale,
            movement_bonus=config.movement_bonus,
            time_penalty=config.time_penalty,
            target_max_speed=config.target_max_speed,
            arena_boundary=config.arena_boundary,
            max_target_distance_stage2=config.max_target_distance_stage2,
            max_distractors=config.max_distractors,
            distractor_min_distance=config.distractor_min_distance,
            distractor_max_distance=config.distractor_max_distance,
            distractor_max_speed=config.distractor_max_speed,
            # Biped-specific rewards (all default to 0 = disabled)
            upright_reward_scale=config.upright_reward_scale,
            alive_bonus=config.alive_bonus,
            energy_penalty_scale=config.energy_penalty_scale,
            ground_contact_penalty=config.ground_contact_penalty,
            patience_window=config.patience_window,
            patience_min_delta=config.patience_min_delta,
            joint_stagnation_window=config.joint_stagnation_window,
            joint_stagnation_threshold=config.joint_stagnation_threshold,
            has_walking_stage=config.has_walking_stage,
        )

    def __init__(
        self,
        scene_path="bots/simple2wheeler/scene.xml",
        render_width=64,
        render_height=64,
        max_episode_steps=100,  # 10 seconds at 10 Hz
        max_episode_steps_final=None,  # Scheduled max at full curriculum (None = no scheduling)
        mujoco_steps_per_action=5,  # 10 Hz control (50 Hz sim / 5 = 10 Hz)
        success_distance=0.3,  # Success if within 0.3m of target
        failure_distance=5.0,  # Failure if beyond 5m from target
        min_target_distance=0.8,  # Minimum spawn distance from robot
        max_target_distance=2.5,  # Maximum spawn distance from robot
        # Reward shaping coefficients
        distance_reward_scale=20.0,  # Scale factor for distance-based reward
        movement_bonus=0.0,  # Disabled: was rewarding spinning in place
        time_penalty=0.005,  # Tiny penalty per step (discourages dawdling)
        # Stage 2: moving target + distance
        target_max_speed=0.3,
        arena_boundary=4.0,
        max_target_distance_stage2=4.0,
        # Stage 3: visual distractors
        max_distractors=4,
        distractor_min_distance=0.5,
        distractor_max_distance=3.0,
        # Stage 4: moving distractors
        distractor_max_speed=0.2,
        # Biped-specific reward params (all 0.0 = disabled for wheeler)
        upright_reward_scale=0.0,
        alive_bonus=0.0,
        energy_penalty_scale=0.0,
        ground_contact_penalty=0.0,
        # Distance-patience early truncation
        patience_window=30,
        patience_min_delta=0.0,
        # Joint-stagnation early truncation
        joint_stagnation_window=0,
        joint_stagnation_threshold=0.05,
        # Walking stage (biped only)
        has_walking_stage=False,
    ):
        self.env = SimpleWheelerEnv(
            scene_path=scene_path,
            render_width=render_width,
            render_height=render_height,
        )
        self.base_episode_steps = max_episode_steps
        self.max_episode_steps_final = max_episode_steps_final or max_episode_steps
        self.max_episode_steps = max_episode_steps
        self.mujoco_steps_per_action = mujoco_steps_per_action
        self.success_distance = success_distance
        self.failure_distance = failure_distance
        self.min_target_distance = min_target_distance
        self.max_target_distance = max_target_distance
        self.distance_reward_scale = distance_reward_scale
        self.movement_bonus = movement_bonus
        self.time_penalty = time_penalty

        # Stage 2/3/4 params
        self.target_max_speed = target_max_speed
        self.arena_boundary = arena_boundary
        self.max_target_distance_stage2 = max_target_distance_stage2
        self.max_distractors = max_distractors
        self.distractor_min_distance = distractor_min_distance
        self.distractor_max_distance = distractor_max_distance
        self.distractor_max_speed = distractor_max_speed

        # Biped-specific rewards
        self.upright_reward_scale = upright_reward_scale
        self.alive_bonus = alive_bonus
        self.energy_penalty_scale = energy_penalty_scale
        self.ground_contact_penalty = ground_contact_penalty

        # Distance-patience early truncation
        self.patience_window = patience_window
        self.patience_min_delta = patience_min_delta
        self._distance_deltas = (
            deque(maxlen=patience_window) if patience_window > 0 else None
        )

        # Joint-stagnation early truncation
        self.joint_stagnation_window = joint_stagnation_window
        self.joint_stagnation_threshold = joint_stagnation_threshold
        self._joint_deltas = (
            deque(maxlen=joint_stagnation_window) if joint_stagnation_window > 0 else None
        )
        self._prev_joint_pos = None

        # Walking stage (biped only)
        self.has_walking_stage = has_walking_stage

        # Episode tracking
        self.episode_step = 0
        self.prev_distance = None
        self.prev_position = None

        # Multi-stage curriculum
        # When has_walking_stage: Stage 1 = walking, then stages shift up by 1
        # Effective stages: angle variance, moving target, distractors, moving distractors
        self.curriculum_stage = 1
        self.curriculum_stage_progress = 1.0  # Default: fully progressed

        # Target movement state (stage 2+)
        self.target_velocity = np.array([0.0, 0.0])  # XY velocity

        # Distractor movement state (stage 4+)
        self.distractor_velocities = [np.array([0.0, 0.0]) for _ in range(max_distractors)]

        # Time step for target movement (action dt = mujoco_steps * sim_dt)
        self.action_dt = mujoco_steps_per_action * self.env.model.opt.timestep

        # Expose actuator info from inner env (drives policy output size)
        self.num_actuators = self.env.num_actuators
        self.actuator_names = self.env.actuator_names

        # Sensor data
        self.sensor_dim = self.env.sensor_dim
        self.sensor_info = self.env.sensor_info
        self.current_sensors = np.zeros(self.sensor_dim, dtype=np.float32)

        # Observation and action spaces (for reference)
        self.observation_shape = (render_height, render_width, 3)
        self.action_shape = (self.num_actuators,)

    def update_episode_limits(self, base=None, final=None):
        """Update episode step scheduling limits for live tweaks."""
        if base is not None:
            self.base_episode_steps = base
        if final is not None:
            self.max_episode_steps_final = final

    @property
    def in_walking_stage(self):
        """True when the biped is in the walking curriculum stage (stage 1)."""
        return self.has_walking_stage and self.curriculum_stage == 1

    def set_curriculum_stage(self, stage, progress, num_stages=3):
        """
        Set multi-stage curriculum state.

        When has_walking_stage is True, stage 1 is the walking stage and
        the standard stages (angle variance, moving target, etc.) are
        shifted up by 1.

        Args:
            stage: Int 1-N (1 = walking if has_walking_stage, else angle variance)
            progress: Float in [0, 1] for current stage
                Previous stages stay at max when advancing.
            num_stages: Total number of curriculum stages (for episode length scheduling)
        """
        self.curriculum_stage = stage
        self.curriculum_stage_progress = np.clip(progress, 0.0, 1.0)

        # Schedule episode length: lerp from base to final based on overall progress
        overall = (stage - 1 + self.curriculum_stage_progress) / num_stages
        self.max_episode_steps = int(
            self.base_episode_steps
            + overall * (self.max_episode_steps_final - self.base_episode_steps)
        )

    def reset(self):
        """
        Reset environment to initial state with randomized target position.

        Stage 1: Target at curriculum-controlled angle, fixed distance range.
        Stage 2: + moving target, increased max distance.
        Stage 3: + visual distractor cubes.
        Stage 4: + moving distractor cubes.

        Returns:
            observation: Camera image normalized to [0, 1]
        """
        camera_img = self.env.reset()
        self.episode_step = 0
        if self._distance_deltas is not None:
            self._distance_deltas.clear()
        if self._joint_deltas is not None:
            self._joint_deltas.clear()
            self._prev_joint_pos = None

        # Effective stage: offset by 1 when has_walking_stage so that
        # angle variance = effective 1, moving target = effective 2, etc.
        stage_offset = 1 if self.has_walking_stage else 0
        effective_stage = self.curriculum_stage - stage_offset

        # --- Walking stage: target 10m ahead, no distractors ---
        if self.in_walking_stage:
            # Place target far ahead (-Y = forward) — unreachable, just for
            # distance reward to incentivize forward movement
            target_x = 0.0
            target_y = -10.0
            target_z = 0.08
            self.target_velocity = np.array([0.0, 0.0])

            mocap_id = self.env.target_mocap_id
            self.env.data.mocap_pos[mocap_id] = [target_x, target_y, target_z]
            self._hide_distractors()
            for i in range(self.max_distractors):
                self.distractor_velocities[i] = np.array([0.0, 0.0])

        else:
            # --- Angle variance (effective stage 1) ---
            # At effective stage 1, progress controls angle. At 2+, full 360°.
            angle_progress = (
                1.0 if effective_stage >= 2 else self.curriculum_stage_progress
            )

            # Coordinate system (looking down from above):
            #   +X = right, +Y = backward, -Y = forward (camera looks this way)
            # At progress=0: target directly in front (angle=-π/2)
            # At progress=1: target at any angle (full circle)
            front_angle = -np.pi / 2
            max_deviation = angle_progress * np.pi
            angle = np.random.uniform(
                front_angle - max_deviation, front_angle + max_deviation
            )

            # --- Moving target + increased distance (effective stage 2) ---
            if effective_stage >= 2:
                stage2_progress = (
                    self.curriculum_stage_progress if effective_stage == 2 else 1.0
                )
                # Lerp max distance from base to stage2 max
                max_dist = self.max_target_distance + stage2_progress * (
                    self.max_target_distance_stage2 - self.max_target_distance
                )
                # Set random target velocity
                speed = (
                    stage2_progress * self.target_max_speed * np.random.uniform(0.5, 1.0)
                )
                vel_angle = np.random.uniform(0, 2 * np.pi)
                self.target_velocity = np.array(
                    [
                        speed * np.cos(vel_angle),
                        speed * np.sin(vel_angle),
                    ]
                )
            else:
                max_dist = self.max_target_distance
                self.target_velocity = np.array([0.0, 0.0])

            # Random distance within bounds
            distance = np.random.uniform(self.min_target_distance, max_dist)

            # Calculate new target position (robot starts at origin)
            target_x = distance * np.cos(angle)
            target_y = distance * np.sin(angle)
            target_z = 0.08

            # Update target mocap position
            mocap_id = self.env.target_mocap_id
            self.env.data.mocap_pos[mocap_id] = [target_x, target_y, target_z]

            # --- Visual distractors (effective stage 3) ---
            if effective_stage >= 3:
                stage3_progress = (
                    self.curriculum_stage_progress if effective_stage == 3 else 1.0
                )
                n_distractors = max(1, int(round(stage3_progress * self.max_distractors)))
                self._place_distractors(n_distractors, target_x, target_y)
            else:
                self._hide_distractors()

            # --- Moving distractors (effective stage 4) ---
            if effective_stage >= 4:
                stage4_progress = (
                    self.curriculum_stage_progress if effective_stage == 4 else 1.0
                )
                n_moving = max(1, int(round(stage4_progress * self.max_distractors)))
                for i in range(self.max_distractors):
                    if i < n_moving:
                        speed = stage4_progress * self.distractor_max_speed * np.random.uniform(0.5, 1.0)
                        vel_angle = np.random.uniform(0, 2 * np.pi)
                        self.distractor_velocities[i] = np.array([
                            speed * np.cos(vel_angle),
                            speed * np.sin(vel_angle),
                        ])
                    else:
                        self.distractor_velocities[i] = np.array([0.0, 0.0])
            else:
                for i in range(self.max_distractors):
                    self.distractor_velocities[i] = np.array([0.0, 0.0])

        # Re-run forward kinematics to update positions
        mujoco.mj_forward(self.env.model, self.env.data)

        # Save reset configuration for replay
        self.last_reset_config = {
            "target_pos": [target_x, target_y, target_z],
            "target_velocity": self.target_velocity.copy().tolist(),
            "distractor_positions": [
                self.env.data.mocap_pos[mid].copy().tolist()
                for mid in self.env.distractor_mocap_ids
            ],
            "distractor_velocities": [v.copy().tolist() for v in self.distractor_velocities],
            "curriculum_stage": self.curriculum_stage,
            "curriculum_stage_progress": self.curriculum_stage_progress,
        }

        # Get updated camera image after target move
        camera_img = self.env.get_camera_image()

        self.prev_distance = self.env.get_distance_to_target()
        self.prev_position = self.env.get_bot_position()
        self.current_sensors = self.env.get_sensor_data()

        # Normalize to [0, 1]
        obs = camera_img.astype(np.float32) / 255.0
        return obs

    def reset_to_config(self, config):
        """
        Reset environment to a specific saved configuration for replay.

        Args:
            config: Dict from last_reset_config saved during reset()

        Returns:
            observation: Camera image normalized to [0, 1]
        """
        self.env.reset()
        self.episode_step = 0
        if self._distance_deltas is not None:
            self._distance_deltas.clear()
        if self._joint_deltas is not None:
            self._joint_deltas.clear()
            self._prev_joint_pos = None

        # Restore target position
        target_pos = config["target_pos"]
        mocap_id = self.env.target_mocap_id
        self.env.data.mocap_pos[mocap_id] = target_pos

        # Restore target velocity
        self.target_velocity = np.array(config["target_velocity"])

        # Restore distractor positions
        for mid, pos in zip(
            self.env.distractor_mocap_ids, config["distractor_positions"]
        ):
            self.env.data.mocap_pos[mid] = pos

        # Restore distractor velocities (backward-compatible: default to zeros)
        saved_vels = config.get("distractor_velocities")
        if saved_vels:
            self.distractor_velocities = [np.array(v) for v in saved_vels]
        else:
            self.distractor_velocities = [np.array([0.0, 0.0]) for _ in range(self.max_distractors)]

        # Restore curriculum state
        self.curriculum_stage = config["curriculum_stage"]
        self.curriculum_stage_progress = config["curriculum_stage_progress"]

        # Re-run forward kinematics
        mujoco.mj_forward(self.env.model, self.env.data)

        camera_img = self.env.get_camera_image()
        self.prev_distance = self.env.get_distance_to_target()
        self.prev_position = self.env.get_bot_position()

        obs = camera_img.astype(np.float32) / 255.0
        return obs

    def _place_distractors(self, n, target_x, target_y):
        """Place n distractor cubes at random positions, hide the rest."""
        for i, mocap_id in enumerate(self.env.distractor_mocap_ids):
            if i < n:
                # Place at random angle/distance from origin
                for _ in range(20):  # Rejection sampling
                    d_angle = np.random.uniform(0, 2 * np.pi)
                    d_dist = np.random.uniform(
                        self.distractor_min_distance, self.distractor_max_distance
                    )
                    dx = d_dist * np.cos(d_angle)
                    dy = d_dist * np.sin(d_angle)
                    # Reject if too close to target
                    if np.hypot(dx - target_x, dy - target_y) > 0.2:
                        break
                self.env.data.mocap_pos[mocap_id] = [dx, dy, 0.08]
            else:
                self.env.data.mocap_pos[mocap_id] = [0.0, 100.0, 0.08]

    def _hide_distractors(self):
        """Move all distractors off-screen."""
        for mocap_id in self.env.distractor_mocap_ids:
            self.env.data.mocap_pos[mocap_id] = [0.0, 100.0, 0.08]

    def _update_target_position(self):
        """Move target by velocity, bounce off arena boundaries."""
        mocap_id = self.env.target_mocap_id
        pos = self.env.data.mocap_pos[mocap_id].copy()

        # Update XY position
        pos[0] += self.target_velocity[0] * self.action_dt
        pos[1] += self.target_velocity[1] * self.action_dt

        # Bounce off arena boundaries
        boundary = self.arena_boundary
        for axis in range(2):
            if pos[axis] > boundary:
                pos[axis] = boundary
                self.target_velocity[axis] *= -1
            elif pos[axis] < -boundary:
                pos[axis] = -boundary
                self.target_velocity[axis] *= -1

        self.env.data.mocap_pos[mocap_id] = pos

    def _update_distractor_positions(self):
        """Move distractors by velocity, bounce off arena boundaries."""
        boundary = self.arena_boundary
        for i, mocap_id in enumerate(self.env.distractor_mocap_ids):
            vel = self.distractor_velocities[i]
            if vel[0] == 0.0 and vel[1] == 0.0:
                continue
            pos = self.env.data.mocap_pos[mocap_id].copy()
            pos[0] += vel[0] * self.action_dt
            pos[1] += vel[1] * self.action_dt
            for axis in range(2):
                if pos[axis] > boundary:
                    pos[axis] = boundary
                    self.distractor_velocities[i][axis] *= -1
                elif pos[axis] < -boundary:
                    pos[axis] = -boundary
                    self.distractor_velocities[i][axis] *= -1
            self.env.data.mocap_pos[mocap_id] = pos

    def step(self, action):
        """
        Take action and advance simulation.

        Runs mujoco_steps_per_action MuJoCo steps to achieve desired control frequency.

        Args:
            action: Array of motor commands, length num_actuators, each in [-1, 1]

        Returns:
            observation: Camera image normalized to [0, 1]
            reward: Float reward value
            done: Boolean indicating episode termination
            truncated: Boolean indicating episode timeout
            info: Dict with additional information
        """
        action = np.asarray(action, dtype=np.float64)

        # Run multiple MuJoCo steps with same action (10 Hz control)
        # Only render on the last step to avoid wasted work
        for i in range(self.mujoco_steps_per_action):
            is_last_step = i == self.mujoco_steps_per_action - 1
            camera_img = self.env.step(action, render=is_last_step)

        # Get observation
        obs = camera_img.astype(np.float32) / 255.0
        self.current_sensors = self.env.get_sensor_data()

        # Get current state (before moving target, so reward reflects robot's action)
        current_distance = self.env.get_distance_to_target()
        current_position = self.env.get_bot_position()

        # Calculate reward components:
        # 1. Distance reward: linear potential-based shaping (standard in literature)
        distance_reward = self.distance_reward_scale * (
            self.prev_distance - current_distance
        )

        # 2. Movement bonus: small reward for exploring (moving at all)
        distance_moved = np.linalg.norm(current_position - self.prev_position)
        exploration_reward = self.movement_bonus * distance_moved

        # 3. Time penalty: tiny cost per step to encourage efficiency
        time_cost = -self.time_penalty

        # 4. Upright reward (biped only, 0 when disabled)
        upright_reward = 0.0
        if self.upright_reward_scale > 0:
            up_vec = self.env.get_torso_up_vector()
            upright_reward = self.upright_reward_scale * up_vec[2]

        # 5. Alive bonus (biped only, 0 when disabled)
        alive_reward = self.alive_bonus

        # 6. Energy penalty (biped only, 0 when disabled)
        energy_cost = 0.0
        if self.energy_penalty_scale > 0:
            energy_cost = -self.energy_penalty_scale * np.sum(action ** 2)

        # 7. Ground contact penalty (biped: penalize non-foot body parts touching floor)
        contact_penalty = 0.0
        if self.ground_contact_penalty > 0:
            bad_contacts = self.env.get_non_foot_ground_contacts()
            if bad_contacts > 0:
                contact_penalty = -self.ground_contact_penalty

        reward = (distance_reward + exploration_reward + time_cost
                  + upright_reward + alive_reward + energy_cost
                  + contact_penalty)

        # Track distance delta for patience (before prev_distance is updated)
        distance_delta = self.prev_distance - current_distance

        # Move target after reward computation (stage 2+).
        # Next step's observation and reward will see the new position.
        if np.any(self.target_velocity != 0):
            self._update_target_position()

        # Move distractors (stage 4+)
        if self.curriculum_stage >= 4:
            self._update_distractor_positions()

        # Update state for next step (use post-move distance so next step's
        # reward delta captures both robot movement and target movement)
        self.prev_distance = self.env.get_distance_to_target()
        self.prev_position = current_position

        # Check termination conditions
        done = False
        truncated = False

        # Simulation instability check (MuJoCo warnings or invalid state)
        has_warnings = np.any(self.env.data.warning.number > 0)
        has_nan = np.isnan(current_distance) or np.any(np.isnan(current_position))
        bot_z = current_position[2]
        out_of_bounds = bot_z < -0.5 or bot_z > 2.0  # Fell through floor or launched

        if has_warnings or has_nan or out_of_bounds:
            done = True
            reward -= 2.0  # Small penalty for unstable behavior
            # Reset warning counters for next episode
            self.env.data.warning.number[:] = 0

        # Success: reached target (not applicable in walking stage)
        elif not self.in_walking_stage and current_distance < self.success_distance:
            done = True
            reward += 10.0  # Bonus for reaching target

        # Failure: too far away (disabled in walking stage — target is intentionally far)
        elif not self.in_walking_stage and current_distance > self.failure_distance:
            done = True
            reward -= 5.0  # Penalty for going too far

        # Distance-patience truncation: no net progress over rolling window
        # Disabled during walking stage (bot is learning to stand, not navigate)
        patience_truncated = False
        if not done and self._distance_deltas is not None and not self.in_walking_stage:
            self._distance_deltas.append(distance_delta)
            if (
                len(self._distance_deltas) == self._distance_deltas.maxlen
                and sum(self._distance_deltas) <= self.patience_min_delta
            ):
                truncated = True
                patience_truncated = True

        # Joint-stagnation truncation: abort if joints aren't moving
        joint_stagnation_truncated = False
        if not done and not truncated and self._joint_deltas is not None:
            joint_pos = self.env.get_actuated_joint_positions()
            if self._prev_joint_pos is not None:
                delta = float(np.sum(np.abs(joint_pos - self._prev_joint_pos)))
                self._joint_deltas.append(delta)
                if (
                    len(self._joint_deltas) == self._joint_deltas.maxlen
                    and sum(self._joint_deltas) < self.joint_stagnation_threshold
                ):
                    truncated = True
                    joint_stagnation_truncated = True
            self._prev_joint_pos = joint_pos

        # Timeout: max steps reached
        self.episode_step += 1
        if self.episode_step >= self.max_episode_steps:
            truncated = True

        # Additional info (includes reward breakdown for visualization)
        info = {
            "distance": current_distance,
            "position": current_position,
            "step": self.episode_step,
            "distance_moved": distance_moved,
            # Reward components for visualization
            "reward_distance": distance_reward,
            "reward_exploration": exploration_reward,
            "reward_time": time_cost,
            "reward_upright": upright_reward,
            "reward_alive": alive_reward,
            "reward_energy": energy_cost,
            "reward_contact": contact_penalty,
            "reward_total": reward,
            # Stability info
            "unstable": has_warnings or has_nan or out_of_bounds,
            "torso_height": bot_z,
            # Patience truncation
            "patience_truncated": patience_truncated,
            # Joint stagnation truncation
            "joint_stagnation_truncated": joint_stagnation_truncated,
            # Walking stage flag
            "in_walking_stage": self.in_walking_stage,
        }

        return obs, reward, done, truncated, info

    def close(self):
        """Clean up resources."""
        self.env.close()
