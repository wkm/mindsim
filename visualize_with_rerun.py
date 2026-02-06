"""
Visualize the 2-wheeler robot simulation using Rerun.io
Shows motor inputs, bot state, and camera view in an interactive viewer.
"""
import numpy as np
import rerun as rr
import time
from simple_wheeler_env import SimpleWheelerEnv


def run_manual_control_demo(output_file="robot_sim.rrd"):
    """
    Run a manual control demo with Rerun visualization.
    Saves to .rrd file that can be viewed with: rerun robot_sim.rrd
    """
    # Initialize Rerun - save to file instead of spawning viewer
    rr.init("simple_wheeler_robot")
    rr.save(output_file)

    # Create environment
    print("Creating environment...")
    env = SimpleWheelerEnv(render_width=128, render_height=128)

    # Log static scene elements
    print("Logging scene setup...")

    # Log world origin
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # Log target cube position
    target_pos = env.get_target_position()
    rr.log(
        "world/target",
        rr.Boxes3D(
            half_sizes=[0.05, 0.05, 0.05],
            centers=[target_pos],
            colors=[[255, 127, 0]],
            labels=["Target Cube"]
        ),
        static=True
    )

    # Run simulation with different control phases
    print("Starting simulation...")
    print("  Phase 1: Forward")
    print("  Phase 2: Turn right")
    print("  Phase 3: Forward again")
    print()

    phases = [
        ("Forward", 0.5, 0.5, 200),
        ("Turn Right", 0.5, -0.5, 150),
        ("Forward Again", 0.5, 0.5, 200),
        ("Turn Left", -0.5, 0.5, 150),
        ("Forward More", 0.5, 0.5, 300),
    ]

    step_count = 0
    start_time = time.time()

    # Calculate logging frequency (30Hz for all data)
    log_interval = max(1, int(1.0 / (30.0 * env.model.opt.timestep)))
    sim_freq = 1.0 / env.model.opt.timestep
    log_freq = sim_freq / log_interval
    print(f"Simulation frequency: {sim_freq:.1f}Hz (timestep={env.model.opt.timestep}s)")
    print(f"Logging every {log_interval} steps (~{log_freq:.1f}Hz)")
    print()

    for phase_name, left_motor, right_motor, num_steps in phases:
        print(f"Phase: {phase_name} (L={left_motor:+.1f}, R={right_motor:+.1f}) for {num_steps} steps")

        for i in range(num_steps):
            # Step simulation
            camera_img = env.step(left_motor, right_motor)

            # Get current state
            bot_pos = env.get_bot_position()
            target_pos = env.get_target_position()
            distance = env.get_distance_to_target()

            # Log at 30Hz
            if step_count % log_interval == 0:
                # Set time for this frame
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

                # Log bot position in 3D world
                rr.log(
                    "world/bot",
                    rr.Boxes3D(
                        half_sizes=[0.05, 0.05, 0.05],
                        centers=[bot_pos],
                        colors=[[100, 150, 255]],
                        labels=["Bot"]
                    )
                )

                # Log camera image
                rr.log("camera/rgb", rr.Image(camera_img))

                # Log current phase
                rr.log("info/phase", rr.TextLog(phase_name))

            # Print progress
            if step_count % 100 == 0:
                elapsed = time.time() - start_time
                print(f"  Step {step_count}: pos={bot_pos[:2]}, dist={distance:.3f}m, time={elapsed:.1f}s")

            step_count += 1

            # Small delay to make it easier to follow in real-time
            time.sleep(0.01)

    # Final summary
    print()
    print("=" * 60)
    print("Simulation Complete!")
    print(f"  Total steps: {step_count}")
    print(f"  Final position: {env.get_bot_position()}")
    print(f"  Final distance to target: {env.get_distance_to_target():.3f}m")
    print(f"  Elapsed time: {time.time() - start_time:.1f}s")
    print("=" * 60)
    print()
    print(f"Recording saved to: {output_file}")
    print()
    print("To view the recording:")
    print(f"  1. Install rerun viewer: cargo binstall rerun-cli")
    print(f"  2. View the file: rerun {output_file}")
    print()
    print("The recording includes:")
    print("  - 3D scene with bot and target positions")
    print("  - Motor input time series (L/R)")
    print("  - Distance to target over time")
    print("  - Camera feed")

    env.close()



if __name__ == "__main__":
    run_manual_control_demo()
