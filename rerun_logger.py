"""
Rerun logging utilities for MuJoCo simulations.

Shared logging code for both visualization and training scripts.
"""

from __future__ import annotations

import logging
import mujoco
import numpy as np
import rerun as rr

logger = logging.getLogger(__name__)


class VideoEncoder:
    """H.264 video stream encoder for Rerun.

    Encodes frames on-the-fly and logs them as a VideoStream,
    dramatically reducing .rrd file sizes compared to per-frame JPEG images.

    Usage:
        encoder = VideoEncoder("eval/camera", width=128, height=128)
        for step, obs in enumerate(observations):
            rr.set_time("step", sequence=step)
            encoder.log_frame(obs)
        encoder.flush()
    """

    def __init__(self, entity_path: str, width: int, height: int, fps: int = 10):
        import av

        self.entity_path = entity_path

        # Set up H.264 encoder via pyav
        self.container = av.open("/dev/null", "w", format="h264")
        self.stream = self.container.add_stream("libx264", rate=fps)
        assert isinstance(self.stream, av.video.stream.VideoStream)
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = "yuv420p"
        self.stream.max_b_frames = 0  # B-frames not yet supported by Rerun
        self.stream.options = {"preset": "medium", "crf": "18", "tune": "zerolatency"}

        # Log codec once as static metadata
        rr.log(entity_path, rr.VideoStream(codec=rr.VideoCodec.H264), static=True)

    def log_frame(self, image: np.ndarray) -> None:
        """Encode and log a single RGB frame.

        Args:
            image: (H, W, 3) RGB array — uint8 [0,255] or float32 [0,1].
        """
        import av

        if image.dtype != np.uint8:
            image = (image * 255).clip(0, 255).astype(np.uint8)

        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        for packet in self.stream.encode(frame):
            if packet.pts is not None:
                rr.log(
                    self.entity_path,
                    rr.VideoStream.from_fields(sample=bytes(packet)),
                )

    def flush(self) -> None:
        """Flush remaining encoded frames and close the encoder."""
        for packet in self.stream.encode():
            if packet.pts is not None:
                rr.log(
                    self.entity_path,
                    rr.VideoStream.from_fields(sample=bytes(packet)),
                )
        self.container.close()


def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions (q1 * q2).
    Both quaternions are in xyzw format.
    Returns the result in xyzw format.
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([x, y, z, w])


def _is_body_hidden(data, body_id):
    """Check if a body is hidden off-screen (y > 50)."""
    return data.xpos[body_id][1] > 50


def log_mujoco_scene(env, namespace="world"):
    """
    Automatically log all bodies and meshes from the MuJoCo model.
    Extracts mesh data directly from MuJoCo (with correct scaling already applied).
    Skips bodies hidden off-screen (e.g. distractors at y=100).

    Args:
        env: SimpleWheelerEnv or TrainingEnv instance
        namespace: Root namespace for logging (default "world")
    """
    model = env.env.model if hasattr(env, "env") else env.model
    data = env.env.data if hasattr(env, "env") else env.data
    mesh_count = 0
    geom_count = 0

    # Iterate through all bodies in the model
    for body_id in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)

        # Skip the world body and hidden bodies (e.g. distractors at y=100)
        if body_name == "world" or _is_body_hidden(data, body_id):
            continue

        # Check if this body has any geometries
        for geom_id in range(model.ngeom):
            if model.geom_bodyid[geom_id] == body_id:
                # Get geometry type
                geom_type = model.geom_type[geom_id]

                # If it's a mesh geometry
                if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
                    mesh_id = model.geom_dataid[geom_id]
                    geom_name = mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
                    )

                    # Extract mesh vertices and faces from MuJoCo model
                    vert_start = model.mesh_vertadr[mesh_id]
                    vert_num = model.mesh_vertnum[mesh_id]
                    face_start = model.mesh_faceadr[mesh_id]
                    face_num = model.mesh_facenum[mesh_id]

                    # Get vertices (already scaled)
                    vertices = model.mesh_vert[
                        vert_start : vert_start + vert_num
                    ].copy()

                    # Get faces (triangles)
                    faces = model.mesh_face[face_start : face_start + face_num].copy()

                    # Get color
                    rgba = model.geom_rgba[geom_id]

                    # Get geom position and rotation relative to body
                    geom_pos = model.geom_pos[geom_id]
                    geom_quat = model.geom_quat[geom_id]

                    # Log geom transform (relative to body)
                    entity_path = f"{namespace}/{body_name}/{geom_name}"
                    rr.log(
                        entity_path,
                        rr.Transform3D(
                            translation=geom_pos,
                            rotation=rr.Quaternion(
                                xyzw=[
                                    geom_quat[1],
                                    geom_quat[2],
                                    geom_quat[3],
                                    geom_quat[0],
                                ]
                            ),
                        ),
                        static=True,
                    )

                    # Log mesh under the geom transform
                    rr.log(
                        f"{entity_path}/mesh",
                        rr.Mesh3D(
                            vertex_positions=vertices,
                            triangle_indices=faces,
                            vertex_colors=np.tile(rgba[:3], (len(vertices), 1)),
                        ),
                        static=True,
                    )
                    mesh_count += 1

                # Handle capsule geometries (biped limbs) — draw as line
                elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
                    geom_name = mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
                    )
                    half_length = model.geom_size[geom_id][1]
                    rgba = model.geom_rgba[geom_id]
                    geom_pos = model.geom_pos[geom_id]
                    geom_quat = model.geom_quat[geom_id]

                    entity_path = f"{namespace}/{body_name}/{geom_name or 'geom'}"

                    rr.log(
                        entity_path,
                        rr.Transform3D(
                            translation=geom_pos,
                            rotation=rr.Quaternion(
                                xyzw=[
                                    geom_quat[1],
                                    geom_quat[2],
                                    geom_quat[3],
                                    geom_quat[0],
                                ]
                            ),
                        ),
                        static=True,
                    )

                    color = [int(c * 255) for c in rgba[:4]]
                    rr.log(
                        f"{entity_path}/line",
                        rr.LineStrips3D(
                            [[[0, 0, -half_length], [0, 0, half_length]]],
                            colors=[color],
                            radii=[0.008],
                        ),
                        static=True,
                    )
                    geom_count += 1

                # Handle box geometries
                elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                    geom_name = mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
                    )
                    size = model.geom_size[geom_id]
                    rgba = model.geom_rgba[geom_id]
                    geom_pos = model.geom_pos[geom_id]
                    geom_quat = model.geom_quat[geom_id]

                    entity_path = f"{namespace}/{body_name}/{geom_name or 'geom'}"

                    # Log geom transform
                    rr.log(
                        entity_path,
                        rr.Transform3D(
                            translation=geom_pos,
                            rotation=rr.Quaternion(
                                xyzw=[
                                    geom_quat[1],
                                    geom_quat[2],
                                    geom_quat[3],
                                    geom_quat[0],
                                ]
                            ),
                        ),
                        static=True,
                    )

                    # Log box under transform
                    rr.log(
                        f"{entity_path}/box",
                        rr.Boxes3D(
                            half_sizes=[size],
                            colors=[
                                [
                                    int(rgba[0] * 255),
                                    int(rgba[1] * 255),
                                    int(rgba[2] * 255),
                                    int(rgba[3] * 255),
                                ]
                            ],
                        ),
                        static=True,
                    )
                    geom_count += 1

                # Handle ellipsoid geometries (e.g. duck torso)
                elif geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
                    geom_name = mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
                    )
                    size = model.geom_size[geom_id]  # half-sizes (rx, ry, rz)
                    rgba = model.geom_rgba[geom_id]
                    geom_pos = model.geom_pos[geom_id]
                    geom_quat = model.geom_quat[geom_id]

                    entity_path = f"{namespace}/{body_name}/{geom_name or 'geom'}"

                    rr.log(
                        entity_path,
                        rr.Transform3D(
                            translation=geom_pos,
                            rotation=rr.Quaternion(
                                xyzw=[
                                    geom_quat[1],
                                    geom_quat[2],
                                    geom_quat[3],
                                    geom_quat[0],
                                ]
                            ),
                        ),
                        static=True,
                    )

                    rr.log(
                        f"{entity_path}/ellipsoid",
                        rr.Ellipsoids3D(
                            half_sizes=[size],
                            colors=[
                                [
                                    int(rgba[0] * 255),
                                    int(rgba[1] * 255),
                                    int(rgba[2] * 255),
                                    int(rgba[3] * 255),
                                ]
                            ],
                            fill_mode=rr.components.FillMode.Solid,
                        ),
                        static=True,
                    )
                    geom_count += 1

                # Handle sphere geometries
                elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                    geom_name = mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
                    )
                    radius = model.geom_size[geom_id][0]
                    rgba = model.geom_rgba[geom_id]
                    geom_pos = model.geom_pos[geom_id]
                    geom_quat = model.geom_quat[geom_id]

                    entity_path = f"{namespace}/{body_name}/{geom_name or 'geom'}"

                    rr.log(
                        entity_path,
                        rr.Transform3D(
                            translation=geom_pos,
                            rotation=rr.Quaternion(
                                xyzw=[
                                    geom_quat[1],
                                    geom_quat[2],
                                    geom_quat[3],
                                    geom_quat[0],
                                ]
                            ),
                        ),
                        static=True,
                    )

                    rr.log(
                        f"{entity_path}/sphere",
                        rr.Ellipsoids3D(
                            half_sizes=[[radius, radius, radius]],
                            colors=[
                                [
                                    int(rgba[0] * 255),
                                    int(rgba[1] * 255),
                                    int(rgba[2] * 255),
                                    int(rgba[3] * 255),
                                ]
                            ],
                            fill_mode=rr.components.FillMode.Solid,
                        ),
                        static=True,
                    )
                    geom_count += 1

                # Warn on unhandled geom types
                elif geom_type not in (mujoco.mjtGeom.mjGEOM_PLANE,):
                    geom_name = mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
                    )
                    type_name = mujoco.mjtGeom(geom_type).name
                    logger.warning(
                        "Skipping unsupported geom type %s for geom '%s' on body '%s'",
                        type_name,
                        geom_name or geom_id,
                        body_name,
                    )

    # Log cameras
    for cam_id in range(model.ncam):
        cam_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_id)
        cam_body_id = model.cam_bodyid[cam_id]
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, cam_body_id)

        # Get camera position and orientation relative to body
        cam_pos = model.cam_pos[cam_id]
        cam_quat = model.cam_quat[cam_id]  # [w, x, y, z] format

        # Convert MuJoCo quaternion (wxyz) to xyzw format
        mj_quat_xyzw = np.array([cam_quat[1], cam_quat[2], cam_quat[3], cam_quat[0]])

        # MuJoCo cameras: -Z forward, +Y up
        # Rerun Pinhole: +Z forward, +Y down (OpenCV)
        # Fix: post-multiply 180° around local X to flip both Y and Z
        frame_correction = np.array([1.0, 0.0, 0.0, 0.0])  # 180° around X (xyzw)
        corrected_quat = quaternion_multiply(mj_quat_xyzw, frame_correction)

        # Camera transform relative to body
        entity_path = f"{namespace}/{body_name}/{cam_name}"
        rr.log(
            entity_path,
            rr.Transform3D(
                translation=cam_pos, rotation=rr.Quaternion(xyzw=corrected_quat)
            ),
            static=True,
        )

    return mesh_count, geom_count


def setup_camera(env, namespace="world"):
    """
    Set up Pinhole camera for the robot's camera.

    Args:
        env: SimpleWheelerEnv or TrainingEnv instance
        namespace: Root namespace for logging (default "world")

    Returns:
        camera_entity_path: Path to the camera entity
    """
    model = env.env.model if hasattr(env, "env") else env.model

    cam_id = env.env.camera_id if hasattr(env, "env") else env.camera_id
    render_width = env.env.render_width if hasattr(env, "env") else env.render_width
    render_height = env.env.render_height if hasattr(env, "env") else env.render_height

    cam_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_id)
    cam_body_id = model.cam_bodyid[cam_id]
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, cam_body_id)
    fovy_deg = model.cam_fovy[cam_id]

    # Calculate focal length from FOV
    focal_length = float(render_height / (2.0 * np.tan(np.radians(fovy_deg) / 2.0)))

    camera_entity_path = f"{namespace}/{body_name}/{cam_name}"
    rr.log(
        camera_entity_path,
        rr.Pinhole(
            resolution=[render_width, render_height],
            focal_length=focal_length,
        ),
        static=True,
    )

    return camera_entity_path


def setup_scene(env, namespace="world", floor_size=10.0, arena_boundary=None):
    """
    Log static scene elements (world origin, bodies, meshes, floor).

    Args:
        env: SimpleWheelerEnv or TrainingEnv instance
        namespace: Root namespace for logging (default "world")
        floor_size: Size of floor plane in meters (default 10.0)
        arena_boundary: If set, log arena boundary lines at ±boundary (meters)

    Returns:
        camera_entity_path: Path to the camera entity
    """
    # Log world origin
    rr.log(namespace, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # Log all bodies and meshes from MuJoCo model
    mesh_count, geom_count = log_mujoco_scene(env, namespace)
    model = env.env.model if hasattr(env, "env") else env.model
    print(
        f"  Scene: {model.nbody - 1} bodies, {mesh_count} meshes, {geom_count} primitive geoms, {model.ncam} cameras"
    )

    # Set up camera
    camera_entity_path = setup_camera(env, namespace)

    # Log floor plane
    half_size = floor_size / 2.0
    rr.log(
        f"{namespace}/floor",
        rr.Boxes3D(
            half_sizes=[[half_size, half_size, 0.005]],
            centers=[[0, 0, -0.005]],
            colors=[[128, 128, 128, 100]],
        ),
        static=True,
    )

    # Log arena boundary if specified
    if arena_boundary is not None:
        b = arena_boundary
        z = 0.02  # Slightly above floor
        corners = [
            [b, b, z],
            [-b, b, z],
            [-b, -b, z],
            [b, -b, z],
            [b, b, z],  # Close the loop
        ]
        rr.log(
            f"{namespace}/arena_boundary",
            rr.LineStrips3D([corners], colors=[[255, 100, 100, 180]]),
            static=True,
        )

    return camera_entity_path


def log_body_transforms(env, namespace="world"):
    """
    Log transforms for all bodies in the scene (updates meshes).

    Args:
        env: SimpleWheelerEnv or TrainingEnv instance
        namespace: Root namespace for logging (default "world")
    """
    model = env.env.model if hasattr(env, "env") else env.model
    data = env.env.data if hasattr(env, "env") else env.data

    for body_id in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)

        # Skip world body and hidden bodies (e.g. distractors at y=100)
        if body_name == "world" or _is_body_hidden(data, body_id):
            continue

        # Get position and rotation from MuJoCo data
        pos = data.xpos[body_id]
        quat = data.xquat[body_id]  # [w, x, y, z] format

        # Log transform (moves all child meshes/geoms)
        rr.log(
            f"{namespace}/{body_name}",
            rr.Transform3D(
                translation=pos,
                rotation=rr.Quaternion(xyzw=[quat[1], quat[2], quat[3], quat[0]]),
            ),
        )
