"""
Quick one-shot Rerun visualization for any bot.
"""

import time

import numpy as np
import rerun as rr

import rerun_logger
from simple_wheeler_env import SimpleWheelerEnv

CAMERA_WIDTH = 128
CAMERA_HEIGHT = 128
LOGGING_FREQUENCY_HZ = 30.0
PROGRESS_PRINT_INTERVAL = 100


def run_visualization(scene_path="bots/simple2wheeler/scene.xml", output_dir="recordings", num_steps=1000):
    """Run a visualization recording for the given scene.

    Args:
        scene_path: Path to the MuJoCo scene XML file.
        output_dir: Directory to save the .rrd recording.
        num_steps: Number of simulation steps to run.
    """
    import os
    from pathlib import Path

    os.makedirs(output_dir, exist_ok=True)

    # Derive a label from the scene path for naming
    scene_label = Path(scene_path).parent.name

    print(f"Loading {scene_path} ({CAMERA_WIDTH}x{CAMERA_HEIGHT})...")
    env = SimpleWheelerEnv(
        scene_path=scene_path,
        render_width=CAMERA_WIDTH,
        render_height=CAMERA_HEIGHT,
    )

    rr.init(f"mindsim/{scene_label}")
    output_file = f"{output_dir}/{scene_label}_viz.rrd"
    rr.save(output_file)

    env.reset()
    camera_entity_path = rerun_logger.setup_scene(env)

    print(f"  {env.num_actuators} actuators: {env.actuator_names}")
    print(f"  Running {num_steps} steps...")
    print()

    log_interval = max(
        1, int(1.0 / (LOGGING_FREQUENCY_HZ * env.model.opt.timestep))
    )
    trajectory_points = []
    start_time = time.time()

    for step in range(num_steps):
        # Zero actions — just let physics run
        actions = np.zeros(env.num_actuators)
        camera_img = env.step(actions)

        bot_pos = env.get_bot_position()
        distance = env.get_distance_to_target()

        if step % log_interval == 0:
            rr.set_time("step", sequence=step)
            rr.set_time("sim_time", timestamp=env.data.time)

            # Log body transforms
            rerun_logger.log_body_transforms(env)

            # Log camera image
            rr.log(camera_entity_path, rr.Image(camera_img))

            # Log scalars
            rr.log("outputs/distance_to_target", rr.Scalars([distance]))

            # Trajectory trail
            trajectory_points.append(bot_pos.copy())
            if len(trajectory_points) > 1:
                rr.log(
                    "world/trajectory",
                    rr.LineStrips3D(
                        [trajectory_points], colors=[[100, 200, 100]]
                    ),
                )

        if step % PROGRESS_PRINT_INTERVAL == 0:
            elapsed = time.time() - start_time
            print(
                f"  Step {step}: pos={bot_pos[:2]}, dist={distance:.3f}m, {elapsed:.1f}s"
            )

    elapsed = time.time() - start_time
    print()
    print(f"Done: {num_steps} steps in {elapsed:.1f}s")
    print(f"Recording: {output_file}")
    print(f"View: rerun {output_file}")

    env.close()
