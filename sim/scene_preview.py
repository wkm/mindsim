"""Interactive scene preview — generate and browse procedural scenes.

Opens the MuJoCo viewer with procedurally generated furniture.
Each scene gets a deterministic seed so it can be recreated.

Controls:
    Space:      Generate next random scene
    Backspace:  Regenerate current scene (verify reproducibility)
    Arrow keys: Move the target cube
    Q/Escape:   Quit (handled by MuJoCo viewer)
"""

import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from scene_gen import SceneComposer
from scene_gen.composer import describe_scene

# GLFW key codes
KEY_SPACE = 32
KEY_BACKSPACE = 259
KEY_RIGHT = 262
KEY_LEFT = 263
KEY_DOWN = 264
KEY_UP = 265

ARROW_STEP = 0.1


ROOM_XML = str(Path(__file__).parent / "worlds" / "room.xml")


def run_scene_preview():
    """Open MuJoCo viewer with procedurally generated scenes.

    Loads the standalone arena (worlds/room.xml) directly — no bot needed.
    Generates a random scene on startup. Press Space to cycle through
    scenes. Each scene's seed and description are printed to the terminal.
    """
    print(f"Loading {ROOM_XML}...")
    print("Controls: Space=next scene, Backspace=regenerate, Arrows=move target")
    print()

    spec = mujoco.MjSpec.from_file(ROOM_XML)
    SceneComposer.prepare_spec(spec)
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    composer = SceneComposer(m, d)
    if composer.max_objects == 0:
        print("Warning: No obstacle slots found in scene XML.")
        print("Scene generation requires obstacle_N bodies in the XML.")
        return

    # Target mocap for arrow key control
    target_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "target")
    target_mocap_id = m.body_mocapid[target_body_id]
    target_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "target_cube")
    floor_z = float(m.geom_size[target_geom_id][2])

    # Scene state
    current_seed = None
    scene_count = 0

    def _generate(seed=None):
        nonlocal current_seed, scene_count
        if seed is None:
            seed = int(np.random.default_rng().integers(0, 2**32))
        current_seed = seed
        scene_count += 1

        scene = composer.random_scene(seed=seed)
        composer.apply(scene)
        mujoco.mj_forward(m, d)

        desc = describe_scene(scene, seed=seed)
        print(f"{'=' * 55}")
        print(f"  Scene {scene_count}")
        print(f"  {desc.replace(chr(10), chr(10) + '  ')}")
        print(f"{'=' * 55}")
        print()

    def on_key(keycode):
        if keycode == KEY_SPACE:
            _generate()
        elif keycode == KEY_BACKSPACE:
            _generate(seed=current_seed)
        elif keycode == KEY_UP:
            d.mocap_pos[target_mocap_id][1] += ARROW_STEP
        elif keycode == KEY_DOWN:
            d.mocap_pos[target_mocap_id][1] -= ARROW_STEP
        elif keycode == KEY_RIGHT:
            d.mocap_pos[target_mocap_id][0] += ARROW_STEP
        elif keycode == KEY_LEFT:
            d.mocap_pos[target_mocap_id][0] -= ARROW_STEP
        d.mocap_pos[target_mocap_id][2] = floor_z

    # Generate initial scene
    _generate()

    with mujoco.viewer.launch_passive(m, d, key_callback=on_key) as viewer:
        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(m, d)

            d.mocap_pos[target_mocap_id][2] = floor_z

            viewer.sync()
            elapsed = time.time() - step_start
            remaining = m.opt.timestep - elapsed
            if remaining > 0:
                time.sleep(remaining)
