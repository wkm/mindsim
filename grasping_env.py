"""
Training-ready environment for grasping tasks.

Wraps SimpleWheelerEnv with:
- Reach/grasp/lift reward structure
- Contact-based grasp detection
- Standard Gymnasium API matching TrainingEnv's interface

Designed to work with the existing train.py/collection.py/parallel.py pipeline
without modifications to those files.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np

from simple_wheeler_env import SimpleWheelerEnv

if TYPE_CHECKING:
    from config import EnvConfig


class GraspingTrainingEnv:
    """
    Gymnasium-style training environment for grasping tasks.

    API-compatible with TrainingEnv so train.py, collection.py, and parallel.py
    work without changes.

    Observation: Camera RGB image (H, W, 3) normalized to [0, 1] (not used by MLPPolicy)
    Action: Array of motor commands [x, y, z, left_finger, right_finger], each in [-1, 1]
    """

    @classmethod
    def from_config(cls, config: EnvConfig) -> GraspingTrainingEnv:
        """Create GraspingTrainingEnv from an EnvConfig object."""
        return cls(
            scene_path=config.scene_path,
            render_width=config.render_width,
            render_height=config.render_height,
            max_episode_steps=config.max_episode_steps,
            max_episode_steps_final=config.max_episode_steps_final,
            mujoco_steps_per_action=config.mujoco_steps_per_action,
            success_distance=config.success_distance,
            time_penalty=config.time_penalty,
            reach_reward_scale=getattr(config, "reach_reward_scale", 10.0),
            grasp_bonus=getattr(config, "grasp_bonus", 0.5),
            lift_reward_scale=getattr(config, "lift_reward_scale", 20.0),
        )

    def __init__(
        self,
        scene_path="bots/simplepicker/scene.xml",
        render_width=64,
        render_height=64,
        max_episode_steps=500,
        max_episode_steps_final=None,
        mujoco_steps_per_action=4,
        success_distance=0.05,
        time_penalty=0.005,
        reach_reward_scale=10.0,
        grasp_bonus=0.5,
        lift_reward_scale=20.0,
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
        self.time_penalty = time_penalty
        self.reach_reward_scale = reach_reward_scale
        self.grasp_bonus = grasp_bonus
        self.lift_reward_scale = lift_reward_scale

        # Expose actuator info from inner env
        self.num_actuators = self.env.num_actuators
        self.actuator_names = self.env.actuator_names

        # Sensor data
        self.sensor_dim = self.env.sensor_dim  # 12 (bot) + 3 (cup_pos) = 15
        self.sensor_info = self.env.sensor_info
        self.current_sensors = np.zeros(self.sensor_dim, dtype=np.float32)

        # Observation and action spaces
        self.observation_shape = (render_height, render_width, 3)
        self.action_shape = (self.num_actuators,)

        # Time step for compatibility (used by rerun_wandb for video fps)
        self.action_dt = mujoco_steps_per_action * self.env.model.opt.timestep

        # Lookup geom IDs for contact detection
        self.cup_geom_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_GEOM, "cup_geom"
        )
        self.left_finger_geom_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_GEOM, "left_finger_geom"
        )
        self.right_finger_geom_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_GEOM, "right_finger_geom"
        )

        # Cup body ID for position queries
        self.cup_body_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_BODY, "cup"
        )

        # Table top z for cup-dropped detection
        self.table_top_z = 0.2

        # Episode tracking
        self.episode_step = 0
        self.prev_hand_cup_dist = None
        self.prev_cup_target_dist = None
        self.last_reset_config = {}

        # Curriculum (no-op for v1, but API required)
        self.curriculum_stage = 1
        self.curriculum_stage_progress = 1.0

    def update_episode_limits(self, base=None, final=None):
        """Update episode step scheduling limits for live tweaks."""
        if base is not None:
            self.base_episode_steps = base
        if final is not None:
            self.max_episode_steps_final = final

    @property
    def has_walking_stage(self):
        """Always False for grasping tasks."""
        return False

    @property
    def in_walking_stage(self):
        """Always False for grasping tasks."""
        return False

    @property
    def walking_success_min_forward(self):
        """API compatibility — not applicable to grasping."""
        return 0.0

    def set_curriculum_stage(self, stage, progress, num_stages=3):
        """Set curriculum state. No-op for v1 grasping (fixed cup position)."""
        self.curriculum_stage = stage
        self.curriculum_stage_progress = np.clip(progress, 0.0, 1.0)
        # Schedule episode length
        overall = (stage - 1 + self.curriculum_stage_progress) / num_stages
        self.max_episode_steps = int(
            self.base_episode_steps
            + overall * (self.max_episode_steps_final - self.base_episode_steps)
        )

    def _get_hand_position(self):
        """Get palm (base body) position in world coordinates."""
        return self.env.get_bot_position()

    def _get_cup_position(self):
        """Get cup position in world coordinates."""
        return self.env.data.xpos[self.cup_body_id].copy()

    def _get_target_position(self):
        """Get target zone position in world coordinates."""
        return self.env.get_target_position()

    def _check_grasp_contacts(self):
        """Check if both fingers are in contact with the cup.

        Returns True when both left and right finger geoms contact the cup geom.
        """
        left_contact = False
        right_contact = False
        for i in range(self.env.data.ncon):
            contact = self.env.data.contact[i]
            g1, g2 = contact.geom1, contact.geom2
            # Check if one geom is the cup
            if g1 == self.cup_geom_id:
                other = g2
            elif g2 == self.cup_geom_id:
                other = g1
            else:
                continue
            if other == self.left_finger_geom_id:
                left_contact = True
            elif other == self.right_finger_geom_id:
                right_contact = True
        return left_contact and right_contact

    def reset(self):
        """Reset environment: place cup on table, hand at home, target above.

        Returns:
            observation: Camera image normalized to [0, 1]
        """
        self.env.reset()
        self.episode_step = 0

        # Place cup at fixed position on table (v1: no randomization)
        cup_pos = np.array([0.0, 0.0, 0.242])  # centered on table, 2mm above surface
        # Set cup freejoint qpos: [x, y, z, qw, qx, qy, qz]
        # Find the cup's freejoint
        cup_joint_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_JOINT, "cup_joint"
        )
        qpos_adr = self.env.model.jnt_qposadr[cup_joint_id]
        self.env.data.qpos[qpos_adr:qpos_adr + 3] = cup_pos
        self.env.data.qpos[qpos_adr + 3:qpos_adr + 7] = [1, 0, 0, 0]  # identity quaternion
        # Zero out cup velocity
        qvel_adr = self.env.model.jnt_dofadr[cup_joint_id]
        self.env.data.qvel[qvel_adr:qvel_adr + 6] = 0

        # Place target mocap above table
        target_pos = np.array([0.0, 0.0, 0.45])
        mocap_id = self.env.target_mocap_id
        self.env.data.mocap_pos[mocap_id] = target_pos

        # Forward kinematics to compute positions
        mujoco.mj_forward(self.env.model, self.env.data)

        # Initialize distance tracking for potential-based rewards
        hand_pos = self._get_hand_position()
        cup_pos_actual = self._get_cup_position()
        target_pos_actual = self._get_target_position()
        self.prev_hand_cup_dist = np.linalg.norm(hand_pos - cup_pos_actual)
        self.prev_cup_target_dist = np.linalg.norm(cup_pos_actual - target_pos_actual)

        # Save reset config for replay
        self.last_reset_config = {
            "cup_pos": cup_pos.tolist(),
            "target_pos": target_pos.tolist(),
            "curriculum_stage": self.curriculum_stage,
            "curriculum_stage_progress": self.curriculum_stage_progress,
        }

        # Update sensors
        self.current_sensors = self.env.get_sensor_data()

        # Return blank observation (MLPPolicy uses sensors, not camera)
        return np.zeros(self.observation_shape, dtype=np.float32)

    def reset_to_config(self, config):
        """Reset to a specific saved configuration for replay."""
        self.env.reset()
        self.episode_step = 0

        cup_pos = np.array(config["cup_pos"])
        cup_joint_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_JOINT, "cup_joint"
        )
        qpos_adr = self.env.model.jnt_qposadr[cup_joint_id]
        self.env.data.qpos[qpos_adr:qpos_adr + 3] = cup_pos
        self.env.data.qpos[qpos_adr + 3:qpos_adr + 7] = [1, 0, 0, 0]
        qvel_adr = self.env.model.jnt_dofadr[cup_joint_id]
        self.env.data.qvel[qvel_adr:qvel_adr + 6] = 0

        target_pos = np.array(config["target_pos"])
        mocap_id = self.env.target_mocap_id
        self.env.data.mocap_pos[mocap_id] = target_pos

        self.curriculum_stage = config["curriculum_stage"]
        self.curriculum_stage_progress = config["curriculum_stage_progress"]

        mujoco.mj_forward(self.env.model, self.env.data)

        hand_pos = self._get_hand_position()
        cup_pos_actual = self._get_cup_position()
        target_pos_actual = self._get_target_position()
        self.prev_hand_cup_dist = np.linalg.norm(hand_pos - cup_pos_actual)
        self.prev_cup_target_dist = np.linalg.norm(cup_pos_actual - target_pos_actual)

        self.current_sensors = self.env.get_sensor_data()
        return np.zeros(self.observation_shape, dtype=np.float32)

    def step(self, action):
        """Take action and advance simulation.

        Args:
            action: [x, y, z, left_finger, right_finger], each in [-1, 1]

        Returns:
            observation, reward, done, truncated, info
        """
        action = np.asarray(action, dtype=np.float64)

        # Run multiple MuJoCo steps with same action
        for i in range(self.mujoco_steps_per_action):
            self.env.step(action, render=False)

        # Update sensors
        self.current_sensors = self.env.get_sensor_data()

        # Get current state
        hand_pos = self._get_hand_position()
        cup_pos = self._get_cup_position()
        target_pos = self._get_target_position()

        hand_cup_dist = float(np.linalg.norm(hand_pos - cup_pos))
        cup_target_dist = float(np.linalg.norm(cup_pos - target_pos))
        is_grasping = self._check_grasp_contacts()

        # --- Reward components ---

        # 1. Reach (potential-based): attract hand toward cup
        reach_reward = self.reach_reward_scale * (self.prev_hand_cup_dist - hand_cup_dist)

        # 2. Grasp bonus: per-step reward when both fingers contact cup
        grasp_reward = self.grasp_bonus if is_grasping else 0.0

        # 3. Lift (potential-based, gated on grasp): cup moving toward target
        if is_grasping:
            lift_reward = self.lift_reward_scale * (self.prev_cup_target_dist - cup_target_dist)
        else:
            lift_reward = 0.0

        # 4. Time penalty
        time_cost = -self.time_penalty

        reward = reach_reward + grasp_reward + lift_reward + time_cost

        # Update previous distances
        self.prev_hand_cup_dist = hand_cup_dist
        self.prev_cup_target_dist = cup_target_dist

        # --- Termination ---
        done = False
        truncated = False

        # Simulation instability
        has_warnings = np.any(self.env.data.warning.number > 0)
        has_nan = np.isnan(cup_target_dist) or np.any(np.isnan(cup_pos))
        cup_z = cup_pos[2]
        out_of_bounds = cup_z > 1.0 or np.any(np.abs(cup_pos[:2]) > 0.5)

        if has_warnings or has_nan or out_of_bounds:
            done = True
            reward -= 2.0
            self.env.data.warning.number[:] = 0

        # Success: cup reached target zone
        elif cup_target_dist < self.success_distance:
            done = True
            reward += 10.0

        # Cup dropped off table
        elif cup_z < self.table_top_z - 0.05:
            done = True
            reward -= 1.0

        # Timeout
        self.episode_step += 1
        if not done and self.episode_step >= self.max_episode_steps:
            truncated = True

        # Observation (blank — MLPPolicy uses sensors only)
        obs = np.zeros(self.observation_shape, dtype=np.float32)

        # Info dict — matches keys expected by collection.py
        info = {
            "distance": cup_target_dist,  # collection.py uses this for success check
            "position": hand_pos,  # trajectory logging
            "step": self.episode_step,
            "distance_moved": 0.0,
            # Grasping-specific
            "is_grasping": is_grasping,
            "cup_position": cup_pos,
            "hand_cup_distance": hand_cup_dist,
            "cup_target_distance": cup_target_dist,
            "cup_z": cup_z,
            # Reward components
            "reward_distance": reach_reward,  # re-use key name for W&B compat
            "reward_exploration": grasp_reward,
            "reward_time": time_cost,
            "reward_upright": lift_reward,
            "reward_alive": 0.0,
            "reward_energy": 0.0,
            "reward_contact": 0.0,
            "reward_forward_velocity": 0.0,
            "reward_smoothness": 0.0,
            "reward_total": reward,
            # Standard flags (all False/0 for grasping)
            "unstable": has_warnings or has_nan or out_of_bounds,
            "torso_height": hand_pos[2],
            "is_healthy": True,
            "fell": False,
            "patience_truncated": False,
            "joint_stagnation_truncated": False,
            "in_walking_stage": False,
            "forward_distance": 0.0,
        }

        return obs, reward, done, truncated, info

    def close(self):
        """Clean up resources."""
        self.env.close()
