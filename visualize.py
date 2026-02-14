"""
Visualize the 2-wheeler robot simulation using Rerun.io
Shows motor inputs, bot state, and camera view in an interactive viewer.
"""

import time

import rerun as rr

import rerun_logger
from simple_wheeler_env import SimpleWheelerEnv

# Visualization constants
CAMERA_WIDTH = 128
CAMERA_HEIGHT = 128
LOGGING_FREQUENCY_HZ = 30.0
PROGRESS_PRINT_INTERVAL = 100  # steps
FLOOR_SIZE = 10.0  # meters (10x10 floor)


def run_simulation(env, camera_entity_path, phases, episode_index=0):
    """
    Run simulation with specified control phases.

    Args:
        env: SimpleWheelerEnv instance
        camera_entity_path: Path to camera entity in Rerun
        phases: List of (name, left_motor, right_motor, num_steps) tuples
        episode_index: Episode number for timeline

    Returns:
        Total elapsed time in seconds
    """
    trajectory_points = []
    step_count = 0
    start_time = time.time()

    # Calculate logging frequency
    log_interval = max(1, int(1.0 / (LOGGING_FREQUENCY_HZ * env.model.opt.timestep)))
    sim_freq = 1.0 / env.model.opt.timestep
    log_freq = sim_freq / log_interval

    print(
        f"Simulation: {sim_freq:.1f}Hz | Logging: {log_freq:.1f}Hz ({log_interval} step interval)"
    )
    print()

    for phase_name, left_motor, right_motor, num_steps in phases:
        for _ in range(num_steps):
            # Step simulation
            camera_img = env.step(left_motor, right_motor)

            # Get current state
            bot_pos = env.get_bot_position()
            distance = env.get_distance_to_target()

            # Log at 30Hz
            if step_count % log_interval == 0:
                # Set time for this frame - episode allows filtering by episode
                rr.set_time("episode", sequence=episode_index)
                rr.set_time("step", sequence=step_count)
                rr.set_time("sim_time", timestamp=env.data.time)

                # Log motor inputs
                rr.log("inputs/left_motor", rr.Scalars([left_motor]))
                rr.log("inputs/right_motor", rr.Scalars([right_motor]))

                # Log bot state
                rr.log("outputs/distance_to_target", rr.Scalars([distance]))
                rr.log("outputs/bot_position_x", rr.Scalars([bot_pos[0]]))
                rr.log("outputs/bot_position_y", rr.Scalars([bot_pos[1]]))
                rr.log("outputs/bot_position_z", rr.Scalars([bot_pos[2]]))

                # Log transforms for all bodies (automatically updates meshes)
                rerun_logger.log_body_transforms(env)

                # Add to trajectory and log trail
                trajectory_points.append(bot_pos.copy())
                if len(trajectory_points) > 1:
                    rr.log(
                        "world/trajectory",
                        rr.LineStrips3D([trajectory_points], colors=[[100, 200, 100]]),
                    )

                # Log camera image at the camera entity
                rr.log(
                    camera_entity_path, rr.Image(camera_img).compress(jpeg_quality=85)
                )

                # Log current phase
                rr.log("info/phase", rr.TextLog(phase_name))

            # Print progress
            if step_count % PROGRESS_PRINT_INTERVAL == 0:
                elapsed = time.time() - start_time
                print(
                    f"  Step {step_count}: pos={bot_pos[:2]}, dist={distance:.3f}m, {elapsed:.1f}s"
                )

            step_count += 1

    return time.time() - start_time


def run_episode(env, episode_index, phases, output_dir):
    """
    Run a single episode with its own Rerun recording file.

    Args:
        env: SimpleWheelerEnv instance
        episode_index: Episode number
        phases: List of (name, left_motor, right_motor, num_steps) tuples
        output_dir: Directory to save .rrd files

    Returns:
        Dict with episode stats
    """
    # Create a new recording for this episode
    rr.init(f"simple_wheeler/episode_{episode_index}")
    episode_file = f"{output_dir}/episode_{episode_index}.rrd"
    rr.save(episode_file)

    # Reset environment for new episode
    env.reset()

    # Set up scene (each recording needs its own static data)
    camera_entity_path = rerun_logger.setup_scene(env, floor_size=FLOOR_SIZE)

    total_steps = sum(n for _, _, _, n in phases)
    print(f"  Episode {episode_index}: {len(phases)} phases, {total_steps} steps")

    # Run simulation
    elapsed = run_simulation(env, camera_entity_path, phases, episode_index)

    return {
        "episode": episode_index,
        "total_steps": total_steps,
        "final_position": env.get_bot_position().copy(),
        "final_distance": env.get_distance_to_target(),
        "elapsed": elapsed,
        "file": episode_file,
    }


def run_manual_control_demo(output_dir="recordings", num_episodes=3, interactive=True):
    """
    Run a manual control demo with Rerun visualization.
    Each episode is saved to a separate .rrd file.

    Args:
        output_dir: Directory to save .rrd files
        num_episodes: Number of episodes to run
        interactive: If True, wait for Enter key between episodes
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    # Create environment
    print(f"Creating environment ({CAMERA_WIDTH}x{CAMERA_HEIGHT})...")
    env = SimpleWheelerEnv(render_width=CAMERA_WIDTH, render_height=CAMERA_HEIGHT)

    # Define control phases (same for each episode for demo)
    phases = [
        ("Forward", 0.5, 0.5, 200),
        ("Turn Right", 0.5, -0.5, 150),
        ("Forward Again", 0.5, 0.5, 200),
        ("Turn Left", -0.5, 0.5, 150),
        ("Forward More", 0.5, 0.5, 300),
    ]

    print(f"Running {num_episodes} episodes...")
    if interactive:
        print("(Press Enter to start each episode, Ctrl+C to quit)")
    print()

    # Run multiple episodes, each with its own recording file
    episode_stats = []
    try:
        for episode_idx in range(num_episodes):
            if interactive:
                input(f"Press Enter to start episode {episode_idx}...")

            stats = run_episode(env, episode_idx, phases, output_dir)
            episode_stats.append(stats)
            print()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")

    # Summary
    if episode_stats:
        print("=" * 60)
        print("Episodes Complete!")
        print(f"  Episodes: {len(episode_stats)}")
        total_steps = sum(s["total_steps"] for s in episode_stats)
        total_time = sum(s["elapsed"] for s in episode_stats)
        print(f"  Total steps: {total_steps}")
        print(
            f"  Total time: {total_time:.1f}s ({total_steps / total_time:.0f} steps/sec)"
        )
        print("=" * 60)
        print()
        print(f"Recordings saved to: {output_dir}/")
        for s in episode_stats:
            print(f"  - {s['file']}")
        print()
        print(f"View all episodes: rerun {output_dir}/*.rrd")
        print("(Use the recording dropdown to switch between episodes)")

    env.close()


if __name__ == "__main__":
    run_manual_control_demo()
