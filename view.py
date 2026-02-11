"""Launch interactive MuJoCo viewer for any bot.

Usage:
    uv run mjpython view.py              # wheeler (default)
    uv run mjpython view.py --biped      # biped

Controls:
    Arrow keys: move the target cube (10cm per tap)
    Ctrl+Right-click drag: move the target cube (freeform)
    Double-click body + Ctrl+drag: apply perturbation forces
    Space: pause/unpause
"""

import argparse
import time

import mujoco
import mujoco.viewer

SCENES = {
    "wheeler": "bots/simple2wheeler/scene.xml",
    "biped": "bots/simplebiped/scene.xml",
}

# GLFW key codes (passed through unchanged by MuJoCo)
KEY_RIGHT = 262
KEY_LEFT = 263
KEY_DOWN = 264
KEY_UP = 265

ARROW_STEP = 0.1  # meters per keypress


def main():
    parser = argparse.ArgumentParser(description="Launch MuJoCo viewer")
    parser.add_argument(
        "--biped", action="store_true", help="View biped instead of wheeler"
    )
    args = parser.parse_args()

    scene = SCENES["biped"] if args.biped else SCENES["wheeler"]
    print(f"Loading {scene}...")

    m = mujoco.MjModel.from_xml_path(scene)
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


if __name__ == "__main__":
    main()
