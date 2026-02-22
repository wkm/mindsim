from pathlib import Path

import mujoco
import numpy as np


def assemble_sensor_data(base_sensors, gait_step_count, control_dt, gait_phase_period):
    """Assemble sensor vector with optional gait phase encoding.

    Appends [sin(phase), cos(phase), sin(phase+pi), cos(phase+pi)] to the
    base sensor array if gait_phase_period > 0.  Returns base_sensors
    unchanged otherwise.
    """
    if gait_phase_period <= 0:
        return base_sensors
    t = gait_step_count * control_dt
    phase = 2 * np.pi * t / gait_phase_period
    gait = np.array([
        np.sin(phase), np.cos(phase),
        np.sin(phase + np.pi), np.cos(phase + np.pi),
    ], dtype=np.float32)
    return np.concatenate([base_sensors, gait])


class SimpleWheelerEnv:
    """
    Bot-agnostic MuJoCo simulation environment.

    Discovers actuators from the loaded model so it works with any robot
    (2-wheel, biped, etc.). The scene_path constructor arg selects which bot.
    """

    def __init__(
        self,
        scene_path="bots/simple2wheeler/scene.xml",
        render_width=16,
        render_height=16,
    ):
        """
        Initialize the environment.

        Args:
            scene_path: Path to the MuJoCo scene XML file
            render_width: Width of camera image (default 16x16)
            render_height: Height of camera image (default 16x16)
        """
        self.scene_path = Path(scene_path)
        self.render_width = render_width
        self.render_height = render_height

        # Load model and create data
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.model)

        # Create renderer for camera images
        self.renderer = mujoco.Renderer(
            self.model, height=render_height, width=render_width
        )

        # Discover actuators from model (works for any bot)
        self.num_actuators = self.model.nu
        self.actuator_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.num_actuators)
        ]

        # Cache ctrlrange for remapping NN outputs [-1,1] -> [lo, hi]
        self.ctrl_range_lo = self.model.actuator_ctrlrange[:self.num_actuators, 0].copy()
        self.ctrl_range_hi = self.model.actuator_ctrlrange[:self.num_actuators, 1].copy()

        # Standard body/camera IDs (convention: all bots use these names)
        self.bot_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base"
        )
        self.target_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target"
        )
        self.target_mocap_id = self.model.body_mocapid[self.target_body_id]
        self.camera_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "camera_1_cam"
        )
        self.distractor_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"distractor_{i}")
            for i in range(4)
        ]
        self.distractor_mocap_ids = [
            self.model.body_mocapid[bid] for bid in self.distractor_body_ids
        ]

        # Geom IDs for ground contact detection (biped fall penalty)
        # These use mj_name2id which returns -1 if not found — safe for wheeler
        self.floor_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        self.foot_geom_ids = set()
        for foot_name in ("left_foot", "right_foot"):
            gid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, foot_name
            )
            if gid >= 0:
                self.foot_geom_ids.add(gid)

        # Sensor data dimension (0 if no sensors defined)
        self.sensor_dim = self.model.nsensordata

        # Discover sensor names and their spans in sensordata array
        self.sensor_info = []
        for i in range(self.model.nsensor):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            adr = self.model.sensor_adr[i]
            dim = self.model.sensor_dim[i]
            self.sensor_info.append({"name": name, "adr": adr, "dim": dim})

        # Viewer for debugging (optional)
        self.viewer = None

        # Reset to initial state
        self.reset()

    def reset(self):
        """Reset the environment to initial state."""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)  # Compute forward kinematics
        return self.get_camera_image()

    def step(self, actions, render=True):
        """
        Step the simulation with given motor commands.

        Args:
            actions: Array of motor commands, length num_actuators, each in [-1, 1]
            render: Whether to render camera image (default True)

        Returns:
            camera_image: RGB image from bot camera as numpy array (H, W, 3), or None if render=False
        """
        actions = np.clip(actions, -1.0, 1.0)
        # Remap [-1, 1] -> [ctrlrange_lo, ctrlrange_hi] per actuator
        lo = self.ctrl_range_lo
        hi = self.ctrl_range_hi
        self.data.ctrl[:self.num_actuators] = lo + (actions + 1.0) * 0.5 * (hi - lo)

        # Step physics
        mujoco.mj_step(self.model, self.data)

        # Update viewer if running
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()

        # Return camera image only if requested
        if render:
            return self.get_camera_image()
        return None

    def get_camera_image(self):
        """
        Get RGB image from the bot's camera.

        Returns:
            numpy array of shape (height, width, 3) with RGB values [0, 255]
        """
        self.renderer.update_scene(self.data, camera=self.camera_id)
        return self.renderer.render()

    def get_bot_position(self):
        """
        Get the bot's position in world coordinates.

        Returns:
            numpy array [x, y, z]
        """
        return self.data.xpos[self.bot_body_id].copy()

    def get_target_position(self):
        """
        Get the target cube's position in world coordinates.

        Returns:
            numpy array [x, y, z]
        """
        return self.data.xpos[self.target_body_id].copy()

    def get_distance_to_target(self):
        """
        Get Euclidean distance from bot to target.

        Returns:
            float: distance in world units
        """
        bot_pos = self.get_bot_position()
        target_pos = self.get_target_position()
        return np.linalg.norm(target_pos - bot_pos)

    def get_sensor_data(self):
        """Get all sensor readings as a flat array."""
        return self.data.sensordata[:self.sensor_dim].copy().astype(np.float32)

    def get_actuated_joint_positions(self):
        """Get current positions of all actuated joints."""
        positions = np.zeros(self.num_actuators, dtype=np.float32)
        for i in range(self.num_actuators):
            joint_id = self.model.actuator_trnid[i, 0]
            qpos_adr = self.model.jnt_qposadr[joint_id]
            positions[i] = self.data.qpos[qpos_adr]
        return positions

    def get_bot_velocity(self):
        """Get the bot's linear velocity in world frame."""
        return self.data.cvel[self.bot_body_id][3:].copy()  # linear part of 6D spatial vel

    def get_torso_up_vector(self):
        """
        Get the torso's local Z-axis in world frame (uprightness indicator).

        Returns [0, 0, 1] when perfectly upright.
        """
        quat = self.data.xquat[self.bot_body_id]
        w, x, y, z = quat
        up_x = 2 * (x * z + w * y)
        up_y = 2 * (y * z - w * x)
        up_z = 1 - 2 * (x * x + y * y)
        return np.array([up_x, up_y, up_z])

    def get_non_foot_ground_contacts(self):
        """
        Count contacts between the floor and non-foot robot geoms.

        Returns 0 when only feet touch the ground (good), >0 when the
        torso/legs are on the floor (bad — robot has fallen).
        """
        if self.floor_geom_id < 0:
            return 0
        bad_contacts = 0
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1, g2 = contact.geom1, contact.geom2
            # One geom must be the floor
            if g1 == self.floor_geom_id:
                other = g2
            elif g2 == self.floor_geom_id:
                other = g1
            else:
                continue
            # The other geom touching floor must NOT be a foot
            if other not in self.foot_geom_ids:
                bad_contacts += 1
        return bad_contacts

    def launch_viewer(self):
        """Launch interactive 3D viewer for debugging."""
        if self.viewer is None:
            try:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            except AttributeError:
                print("Warning: mujoco.viewer not available. Viewer disabled.")
                return None
        return self.viewer

    def close_viewer(self):
        """Close the interactive viewer."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def close(self):
        """Clean up resources."""
        self.close_viewer()
        self.renderer.close()
