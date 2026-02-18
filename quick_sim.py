"""
Quick simulation for visual debugging of the training environment.

Drives the robot in a circle at max curriculum (stage 3, all distractors,
moving target, full angle randomization) and saves a Rerun recording.
"""

import rerun as rr

import rerun_logger
from training_env import TrainingEnv

OUTPUT_FILE = "quick_sim.rrd"

# Circle: left wheel slightly faster than right
LEFT_MOTOR = 0.5
RIGHT_MOTOR = 0.3


def run_quick_sim(num_steps=200):
    """Run a quick simulation and save a Rerun recording.

    Args:
        num_steps: Number of simulation steps to run.
    """
    env = TrainingEnv()

    # Max curriculum: stage 3, full progress
    env.set_curriculum_stage(3, 1.0)
    obs = env.reset()

    # Set up Rerun recording
    rr.init("mindsim-quick-sim")
    rr.save(OUTPUT_FILE)

    camera_path = rerun_logger.setup_scene(
        env, namespace="world", arena_boundary=env.arena_boundary
    )

    trajectory_points = []
    action_dt = env.action_dt

    for step in range(num_steps):
        rr.set_time("step", sequence=step)
        rr.set_time("sim_time", timestamp=step * action_dt)

        obs, reward, done, truncated, info = env.step([LEFT_MOTOR, RIGHT_MOTOR])

        # Log camera image
        rr.log(camera_path, rr.Image(obs).compress(jpeg_quality=85))

        # Log body transforms
        rerun_logger.log_body_transforms(env, namespace="world")

        # Trajectory
        pos = info["position"]
        trajectory_points.append(pos.copy())
        if len(trajectory_points) > 1:
            rr.log(
                "world/trajectory",
                rr.LineStrips3D([trajectory_points], colors=[[100, 200, 100]]),
            )

        # Log scalars
        rr.log("metrics/distance", rr.Scalars([info["distance"]]))
        rr.log("metrics/reward", rr.Scalars([reward]))

        if done or truncated:
            break

    env.close()
    print(f"Saved {OUTPUT_FILE} ({step + 1} steps)")
    print(f"  rerun {OUTPUT_FILE}")
