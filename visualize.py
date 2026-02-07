"""
Visualize the 2-wheeler robot simulation using Rerun.io
Shows motor inputs, bot state, and camera view in an interactive viewer.
"""
import numpy as np
import rerun as rr
import time
from simple_wheeler_env import SimpleWheelerEnv
import mujoco

# Visualization constants
CAMERA_WIDTH = 128
CAMERA_HEIGHT = 128
LOGGING_FREQUENCY_HZ = 30.0
PROGRESS_PRINT_INTERVAL = 100  # steps
FLOOR_SIZE = 10.0  # meters (10x10 floor)


def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions (q1 * q2).
    Both quaternions are in xyzw format.
    Returns the result in xyzw format.
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return np.array([x, y, z, w])


def log_mujoco_scene(env):
    """
    Automatically log all bodies and meshes from the MuJoCo model.
    Extracts mesh data directly from MuJoCo (with correct scaling already applied).
    """
    model = env.model
    mesh_count = 0
    geom_count = 0

    # Iterate through all bodies in the model
    for body_id in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)

        # Skip the world body
        if body_name == "world":
            continue

        # Check if this body has any geometries
        for geom_id in range(model.ngeom):
            if model.geom_bodyid[geom_id] == body_id:
                # Get geometry type
                geom_type = model.geom_type[geom_id]

                # If it's a mesh geometry
                if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
                    mesh_id = model.geom_dataid[geom_id]
                    geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)

                    # Extract mesh vertices and faces from MuJoCo model
                    # This includes the correct scaling from the XML
                    vert_start = model.mesh_vertadr[mesh_id]
                    vert_num = model.mesh_vertnum[mesh_id]
                    face_start = model.mesh_faceadr[mesh_id]
                    face_num = model.mesh_facenum[mesh_id]

                    # Get vertices (already scaled)
                    vertices = model.mesh_vert[vert_start:vert_start + vert_num].copy()

                    # Get faces (triangles)
                    faces = model.mesh_face[face_start:face_start + face_num].copy()

                    # Get color
                    rgba = model.geom_rgba[geom_id]

                    # Get geom position and rotation relative to body
                    geom_pos = model.geom_pos[geom_id]
                    geom_quat = model.geom_quat[geom_id]

                    # Log geom transform (relative to body)
                    entity_path = f"world/{body_name}/{geom_name}"
                    rr.log(
                        entity_path,
                        rr.Transform3D(
                            translation=geom_pos,
                            rotation=rr.Quaternion(xyzw=[geom_quat[1], geom_quat[2], geom_quat[3], geom_quat[0]])
                        ),
                        static=True
                    )

                    # Log mesh under the geom transform
                    rr.log(
                        f"{entity_path}/mesh",
                        rr.Mesh3D(
                            vertex_positions=vertices,
                            triangle_indices=faces,
                            vertex_colors=np.tile(rgba[:3], (len(vertices), 1))
                        ),
                        static=True
                    )
                    mesh_count += 1

                # Handle non-mesh geometries (boxes, spheres, etc.)
                elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                    geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                    size = model.geom_size[geom_id]
                    rgba = model.geom_rgba[geom_id]
                    geom_pos = model.geom_pos[geom_id]
                    geom_quat = model.geom_quat[geom_id]

                    entity_path = f"world/{body_name}/{geom_name or 'geom'}"

                    # Log geom transform
                    rr.log(
                        entity_path,
                        rr.Transform3D(
                            translation=geom_pos,
                            rotation=rr.Quaternion(xyzw=[geom_quat[1], geom_quat[2], geom_quat[3], geom_quat[0]])
                        ),
                        static=True
                    )

                    # Log box under transform
                    rr.log(
                        f"{entity_path}/box",
                        rr.Boxes3D(
                            half_sizes=[size],
                            colors=[[int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255), int(rgba[3]*255)]]
                        ),
                        static=True
                    )
                    geom_count += 1

    # Log cameras
    for cam_id in range(model.ncam):
        cam_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_id)
        cam_body_id = model.cam_bodyid[cam_id]
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, cam_body_id)

        # Get camera position and orientation relative to body
        cam_pos = model.cam_pos[cam_id]
        cam_quat = model.cam_quat[cam_id]  # [w, x, y, z] format

        # Convert MuJoCo quaternion to xyzw format
        mj_quat_xyzw = np.array([cam_quat[1], cam_quat[2], cam_quat[3], cam_quat[0]])

        # MuJoCo to Rerun coordinate frame correction
        # Rerun cameras point along +Z, MuJoCo cameras need adjustment
        # 180° rotation around X to correct orientation
        frame_correction = np.array([1.0, 0.0, 0.0, 0.0])  # 180° around X (xyzw)

        # Compose: apply MuJoCo rotation, then frame correction
        # We need to invert the MuJoCo Y-rotation (180°) to point forward
        # The MuJoCo camera has euler (0, π, 0), so inverting the Y gives us forward
        # Inverting a quaternion: negate x,y,z components (for unit quaternions)
        inverted_mj_quat = np.array([-mj_quat_xyzw[0], -mj_quat_xyzw[1], -mj_quat_xyzw[2], mj_quat_xyzw[3]])

        # Compose the rotations
        corrected_quat = quaternion_multiply(frame_correction, inverted_mj_quat)

        # Camera transform relative to body
        entity_path = f"world/{body_name}/{cam_name}"
        rr.log(
            entity_path,
            rr.Transform3D(
                translation=cam_pos,
                rotation=rr.Quaternion(xyzw=corrected_quat)
            ),
            static=True
        )

    print(f"  Loaded: {model.nbody - 1} bodies, {mesh_count} meshes, {geom_count} geoms, {model.ncam} cameras")


def setup_camera(env):
    """Set up Pinhole camera for the robot's camera and return the entity path."""
    cam_id = env.camera_id
    cam_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_id)
    cam_body_id = env.model.cam_bodyid[cam_id]
    body_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, cam_body_id)
    fovy_deg = env.model.cam_fovy[cam_id]

    # Calculate focal length from FOV
    focal_length = float(env.render_height / (2.0 * np.tan(np.radians(fovy_deg) / 2.0)))

    camera_entity_path = f"world/{body_name}/{cam_name}"
    rr.log(
        camera_entity_path,
        rr.Pinhole(
            resolution=[env.render_width, env.render_height],
            focal_length=focal_length,
        ),
        static=True
    )

    return camera_entity_path


def setup_scene(env):
    """Log static scene elements (world origin, bodies, meshes, floor)."""
    # Log world origin
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # Log all bodies and meshes from MuJoCo model
    log_mujoco_scene(env)

    # Set up camera
    camera_entity_path = setup_camera(env)

    # Log floor plane
    half_size = FLOOR_SIZE / 2.0
    rr.log(
        "world/floor",
        rr.Boxes3D(
            half_sizes=[[half_size, half_size, 0.005]],
            centers=[[0, 0, -0.005]],
            colors=[[128, 128, 128, 100]],
        ),
        static=True
    )

    return camera_entity_path


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

    print(f"Simulation: {sim_freq:.1f}Hz | Logging: {log_freq:.1f}Hz ({log_interval} step interval)")
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
                for body_id in range(env.model.nbody):
                    body_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, body_id)

                    # Skip world body
                    if body_name == "world":
                        continue

                    # Get position and rotation from MuJoCo data
                    pos = env.data.xpos[body_id]
                    quat = env.data.xquat[body_id]  # [w, x, y, z] format

                    # Log transform (moves all child meshes/geoms)
                    rr.log(
                        f"world/{body_name}",
                        rr.Transform3D(
                            translation=pos,
                            rotation=rr.Quaternion(xyzw=[quat[1], quat[2], quat[3], quat[0]])
                        )
                    )

                # Add to trajectory and log trail
                trajectory_points.append(bot_pos.copy())
                if len(trajectory_points) > 1:
                    rr.log(
                        "world/trajectory",
                        rr.LineStrips3D([trajectory_points], colors=[[100, 200, 100]])
                    )

                # Log camera image at the camera entity
                rr.log(camera_entity_path, rr.Image(camera_img))

                # Log current phase
                rr.log("info/phase", rr.TextLog(phase_name))

            # Print progress
            if step_count % PROGRESS_PRINT_INTERVAL == 0:
                elapsed = time.time() - start_time
                print(f"  Step {step_count}: pos={bot_pos[:2]}, dist={distance:.3f}m, {elapsed:.1f}s")

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
    camera_entity_path = setup_scene(env)

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
        print(f"  Total time: {total_time:.1f}s ({total_steps/total_time:.0f} steps/sec)")
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
