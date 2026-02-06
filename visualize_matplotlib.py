"""
Visualize the 2-wheeler robot simulation using Matplotlib.
Shows motor inputs, bot state, and camera view in real-time plots.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time
from simple_wheeler_env import SimpleWheelerEnv


def run_visualization_demo():
    """
    Run simulation with live matplotlib visualization.
    """
    print("=" * 60)
    print("2-Wheeler Robot Visualization")
    print("=" * 60)
    print()

    # Create environment
    env = SimpleWheelerEnv(render_width=64, render_height=64)

    # Setup matplotlib figure
    plt.ion()  # Interactive mode
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Create subplots
    ax_camera = fig.add_subplot(gs[0, :])  # Top row: camera view
    ax_motor = fig.add_subplot(gs[1, 0])   # Middle left: motor inputs
    ax_distance = fig.add_subplot(gs[1, 1])  # Middle center: distance
    ax_position = fig.add_subplot(gs[1, 2])  # Middle right: position
    ax_trajectory = fig.add_subplot(gs[2, :])  # Bottom: 2D trajectory

    # Setup axes
    ax_camera.set_title("Camera View (64x64)", fontsize=12, fontweight='bold')
    ax_camera.axis('off')

    ax_motor.set_title("Motor Inputs", fontsize=10)
    ax_motor.set_xlabel("Step")
    ax_motor.set_ylabel("Motor Command")
    ax_motor.set_ylim([-1.1, 1.1])
    ax_motor.grid(True, alpha=0.3)
    ax_motor.legend()

    ax_distance.set_title("Distance to Target", fontsize=10)
    ax_distance.set_xlabel("Step")
    ax_distance.set_ylabel("Distance (m)")
    ax_distance.grid(True, alpha=0.3)

    ax_position.set_title("Bot Position (Z)", fontsize=10)
    ax_position.set_xlabel("Step")
    ax_position.set_ylabel("Z Position (m)")
    ax_position.grid(True, alpha=0.3)

    ax_trajectory.set_title("2D Trajectory (Top View)", fontsize=10)
    ax_trajectory.set_xlabel("X Position (m)")
    ax_trajectory.set_ylabel("Y Position (m)")
    ax_trajectory.set_aspect('equal')
    ax_trajectory.grid(True, alpha=0.3)

    # Plot target position
    target_pos = env.get_target_position()
    ax_trajectory.plot(target_pos[0], target_pos[1], 'o', color='orange',
                      markersize=15, label='Target', zorder=10)
    ax_trajectory.legend()

    # Data storage
    history = {
        'steps': [],
        'left_motor': [],
        'right_motor': [],
        'distance': [],
        'pos_x': [],
        'pos_y': [],
        'pos_z': [],
    }

    # Run simulation phases
    phases = [
        ("Forward", 0.5, 0.5, 150),
        ("Turn Right", 0.5, -0.3, 100),
        ("Forward Again", 0.5, 0.5, 150),
        ("Turn Left", -0.3, 0.5, 100),
        ("Forward More", 0.5, 0.5, 200),
    ]

    step_count = 0
    camera_display = None
    line_left = None
    line_right = None
    line_distance = None
    line_pos_z = None
    trajectory_line = None

    print("Running simulation phases:")
    for phase_name, left_motor, right_motor, num_steps in phases:
        print(f"  {phase_name}: L={left_motor:+.1f}, R={right_motor:+.1f} for {num_steps} steps")

        for i in range(num_steps):
            # Step simulation
            camera_img = env.step(left_motor, right_motor)

            # Get state
            bot_pos = env.get_bot_position()
            distance = env.get_distance_to_target()

            # Store history
            history['steps'].append(step_count)
            history['left_motor'].append(left_motor)
            history['right_motor'].append(right_motor)
            history['distance'].append(distance)
            history['pos_x'].append(bot_pos[0])
            history['pos_y'].append(bot_pos[1])
            history['pos_z'].append(bot_pos[2])

            # Update plots every 10 steps
            if step_count % 10 == 0:
                # Update camera view
                if camera_display is None:
                    camera_display = ax_camera.imshow(camera_img)
                else:
                    camera_display.set_data(camera_img)

                # Update motor inputs
                if line_left is None:
                    line_left, = ax_motor.plot(history['steps'], history['left_motor'],
                                              'b-', label='Left', linewidth=2)
                    line_right, = ax_motor.plot(history['steps'], history['right_motor'],
                                               'r-', label='Right', linewidth=2)
                    ax_motor.legend()
                else:
                    line_left.set_data(history['steps'], history['left_motor'])
                    line_right.set_data(history['steps'], history['right_motor'])
                ax_motor.relim()
                ax_motor.autoscale_view()

                # Update distance
                if line_distance is None:
                    line_distance, = ax_distance.plot(history['steps'], history['distance'],
                                                     'g-', linewidth=2)
                else:
                    line_distance.set_data(history['steps'], history['distance'])
                ax_distance.relim()
                ax_distance.autoscale_view()

                # Update position Z
                if line_pos_z is None:
                    line_pos_z, = ax_position.plot(history['steps'], history['pos_z'],
                                                   'm-', linewidth=2)
                else:
                    line_pos_z.set_data(history['steps'], history['pos_z'])
                ax_position.relim()
                ax_position.autoscale_view()

                # Update trajectory
                if trajectory_line is None:
                    trajectory_line, = ax_trajectory.plot(history['pos_x'], history['pos_y'],
                                                         'b-', linewidth=2, alpha=0.6, label='Path')
                    ax_trajectory.plot(bot_pos[0], bot_pos[1], 'bo', markersize=10, label='Bot')
                else:
                    trajectory_line.set_data(history['pos_x'], history['pos_y'])
                    # Update bot position marker
                    for line in ax_trajectory.lines:
                        if 'Bot' in str(line.get_label()):
                            line.set_data([bot_pos[0]], [bot_pos[1]])
                ax_trajectory.relim()
                ax_trajectory.autoscale_view()

                # Add phase annotation
                ax_motor.set_title(f"Motor Inputs - {phase_name}", fontsize=10)

                # Redraw
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.001)

            step_count += 1

    print()
    print("=" * 60)
    print("Simulation Complete!")
    print(f"  Total steps: {step_count}")
    print(f"  Final position: {env.get_bot_position()}")
    print(f"  Final distance to target: {env.get_distance_to_target():.3f}m")
    print(f"  Initial distance: {history['distance'][0]:.3f}m")
    print(f"  Distance traveled: {np.linalg.norm([history['pos_x'][-1] - history['pos_x'][0], history['pos_y'][-1] - history['pos_y'][0]]):.3f}m")
    print("=" * 60)
    print()
    print("Close the plot window to exit.")

    env.close()
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    run_visualization_demo()
