import time
from pathlib import Path

import mujoco
import numpy as np


class SimpleWheelerEnv:
    """
    Simple 2-wheeler robot environment with camera for manual control and testing.

    The robot has two motors (left and right wheels) and a forward-facing camera.
    The goal is to navigate towards an orange target cube.
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

        # Get motor and body IDs for faster access
        self.left_motor_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "Revolute_1_motor"
        )
        self.right_motor_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "Revolute_2_motor"
        )
        self.bot_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base"
        )
        self.target_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target"
        )
        self.camera_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "camera_1_cam"
        )
        self.distractor_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"distractor_{i}")
            for i in range(4)
        ]

        # Viewer for debugging (optional)
        self.viewer = None

        # Reset to initial state
        self.reset()

    def reset(self):
        """Reset the environment to initial state."""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)  # Compute forward kinematics
        return self.get_camera_image()

    def step(self, left_motor, right_motor, render=True):
        """
        Step the simulation with given motor commands.

        Args:
            left_motor: Left motor strength [-1, 1]
            right_motor: Right motor strength [-1, 1]
            render: Whether to render camera image (default True)

        Returns:
            camera_image: RGB image from bot camera as numpy array (H, W, 3), or None if render=False
        """
        # Clip motor values to valid range
        left_motor = np.clip(left_motor, -1.0, 1.0)
        right_motor = np.clip(right_motor, -1.0, 1.0)

        # Set motor controls
        self.data.ctrl[self.left_motor_id] = left_motor
        self.data.ctrl[self.right_motor_id] = right_motor

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


def demo_manual_control():
    """
    Demo script showing manual control of the robot.
    """
    print("=== Simple Wheeler Environment Demo ===")
    print("Manual control demonstration")
    print()

    # Create environment
    env = SimpleWheelerEnv(render_width=64, render_height=64)

    # Launch viewer for visualization
    viewer = env.launch_viewer()
    print("3D viewer launched. Keep window open to continue.")
    print()

    # Get initial state
    print("Initial state:")
    print(f"  Bot position: {env.get_bot_position()}")
    print(f"  Target position: {env.get_target_position()}")
    print(f"  Distance to target: {env.get_distance_to_target():.2f}")
    print()

    # Run a simple control sequence
    print("Running control sequence...")
    print("  Phase 1: Drive forward (both motors positive)")

    start_time = time.time()
    step_count = 0

    try:
        while (viewer is None or viewer.is_running()) and time.time() - start_time < 10:
            step_start = time.time()

            # Simple control: drive forward
            if step_count < 300:
                left_motor = 0.5
                right_motor = 0.5
            # Turn right
            elif step_count < 600:
                left_motor = 0.5
                right_motor = -0.5
            # Drive forward again
            else:
                left_motor = 0.5
                right_motor = 0.5

            # Step simulation
            camera_img = env.step(left_motor, right_motor)

            # Print status every 100 steps
            if step_count % 100 == 0:
                print(
                    f"  Step {step_count}: "
                    f"pos={env.get_bot_position()[:2]}, "
                    f"dist={env.get_distance_to_target():.2f}, "
                    f"camera_shape={camera_img.shape}"
                )

            step_count += 1

            # Proper frame rate control
            time_until_next_step = env.model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    print()
    print("Final state:")
    print(f"  Bot position: {env.get_bot_position()}")
    print(f"  Distance to target: {env.get_distance_to_target():.2f}")
    print(f"  Total steps: {step_count}")

    # Clean up
    env.close()
    print("\nDemo complete!")


if __name__ == "__main__":
    demo_manual_control()
