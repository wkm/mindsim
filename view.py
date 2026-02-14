"""Launch interactive MuJoCo viewer for any bot.

Controls:
    Arrow keys: move the target cube (10cm per tap)
    Ctrl+Right-click drag: move the target cube (freeform)
    Double-click body + Ctrl+drag: apply perturbation forces
    Space: pause/unpause
"""

import time

import mujoco
import mujoco.viewer

# GLFW key codes (passed through unchanged by MuJoCo)
KEY_RIGHT = 262
KEY_LEFT = 263
KEY_DOWN = 264
KEY_UP = 265

ARROW_STEP = 0.1  # meters per keypress


def run_view(scene_path: str, stage: int | None = None):
    """Open the MuJoCo viewer for the given scene.

    Args:
        scene_path: Path to the MuJoCo scene XML file.
        stage: Curriculum stage 1-4 to visualize, or None for raw scene.
    """
    if stage is not None:
        _run_view_with_curriculum(scene_path, stage)
    else:
        _run_view_raw(scene_path)


def _run_view_raw(scene_path: str):
    """Original raw viewer — no curriculum setup."""
    print(f"Loading {scene_path}...")

    m = mujoco.MjModel.from_xml_path(scene_path)
    d = mujoco.MjData(m)

    # Find target mocap body
    target_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "target")
    target_mocap_id = m.body_mocapid[target_body_id]
    target_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "target_cube")
    floor_z = float(m.geom_size[target_geom_id][2])

    def on_key(keycode):
        if keycode == KEY_UP:
            d.mocap_pos[target_mocap_id][1] += ARROW_STEP
        elif keycode == KEY_DOWN:
            d.mocap_pos[target_mocap_id][1] -= ARROW_STEP
        elif keycode == KEY_RIGHT:
            d.mocap_pos[target_mocap_id][0] += ARROW_STEP
        elif keycode == KEY_LEFT:
            d.mocap_pos[target_mocap_id][0] -= ARROW_STEP
        # Keep cube on the floor
        d.mocap_pos[target_mocap_id][2] = floor_z

    with mujoco.viewer.launch_passive(m, d, key_callback=on_key) as viewer:
        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(m, d)

            # Clamp target to floor (for Ctrl+drag which bypasses key_callback)
            d.mocap_pos[target_mocap_id][2] = floor_z

            viewer.sync()
            elapsed = time.time() - step_start
            remaining = m.opt.timestep - elapsed
            if remaining > 0:
                time.sleep(remaining)


def _run_view_with_curriculum(scene_path: str, stage: int):
    """Viewer with curriculum stage setup — targets and distractors animate."""
    from training_env import TrainingEnv

    print(f"Loading {scene_path} with curriculum stage {stage}...")

    env = TrainingEnv(scene_path=scene_path)
    env.set_curriculum_stage(stage, progress=1.0)
    env.reset()

    m = env.env.model
    d = env.env.data

    # Find target mocap for arrow key control
    target_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "target")
    target_mocap_id = m.body_mocapid[target_body_id]
    target_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "target_cube")
    floor_z = float(m.geom_size[target_geom_id][2])

    def on_key(keycode):
        if keycode == KEY_UP:
            d.mocap_pos[target_mocap_id][1] += ARROW_STEP
        elif keycode == KEY_DOWN:
            d.mocap_pos[target_mocap_id][1] -= ARROW_STEP
        elif keycode == KEY_RIGHT:
            d.mocap_pos[target_mocap_id][0] += ARROW_STEP
        elif keycode == KEY_LEFT:
            d.mocap_pos[target_mocap_id][0] -= ARROW_STEP
        d.mocap_pos[target_mocap_id][2] = floor_z

    # Track sim steps to trigger movement at action frequency
    step_counter = 0
    steps_per_action = env.mujoco_steps_per_action

    with mujoco.viewer.launch_passive(m, d, key_callback=on_key) as viewer:
        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(m, d)
            step_counter += 1

            # Animate at action frequency (every mujoco_steps_per_action steps)
            if step_counter % steps_per_action == 0:
                if stage >= 2:
                    env._update_target_position()
                if stage >= 4:
                    env._update_distractor_positions()

            # Clamp target to floor
            d.mocap_pos[target_mocap_id][2] = floor_z

            viewer.sync()
            elapsed = time.time() - step_start
            remaining = m.opt.timestep - elapsed
            if remaining > 0:
                time.sleep(remaining)
