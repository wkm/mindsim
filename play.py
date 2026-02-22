"""
Interactive play mode: watch a trained policy drive the robot in real-time.

Loads a checkpoint, opens a MuJoCo viewer, and runs inference at the trained control frequency.
Use arrow keys to move the target cube; the robot will chase it.
"""

import time

import mujoco
import mujoco.viewer
import numpy as np
import torch

from checkpoint import resolve_resume_ref
from sim_env import SimEnv

# GLFW key constants (avoid importing glfw directly)
KEY_UP = 265
KEY_DOWN = 264
KEY_LEFT = 263
KEY_RIGHT = 262
KEY_SLASH = 47  # '/'
KEY_MINUS = 45  # '-'
KEY_EQUAL = 61  # '='

SPEED_OPTIONS = [1, 2, 4, 8]


def build_policy(ckpt_config):
    """Reconstruct the policy network from a checkpoint's embedded config."""
    # Import policy classes (defined in train.py)
    from train import LSTMPolicy, MLPPolicy, TinyPolicy

    policy_cfg = ckpt_config["policy"]
    policy_type = policy_cfg["policy_type"]

    common_kwargs = dict(
        image_height=policy_cfg["image_height"],
        image_width=policy_cfg["image_width"],
        num_actions=policy_cfg.get("fc_output_size", 2),
        init_std=policy_cfg.get("init_std", 0.5),
        max_log_std=policy_cfg.get("max_log_std", 0.7),
        sensor_input_size=policy_cfg.get("sensor_input_size", 0),
    )

    if policy_type == "LSTMPolicy":
        return LSTMPolicy(
            hidden_size=policy_cfg["hidden_size"],
            **common_kwargs,
        )
    elif policy_type == "MLPPolicy":
        return MLPPolicy(
            hidden_size=policy_cfg["hidden_size"],
            **common_kwargs,
        )
    elif policy_type == "TinyPolicy":
        return TinyPolicy(**common_kwargs)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")


def run_play(checkpoint_ref="latest", scene_path="bots/simple2wheeler/scene.xml"):
    """Run interactive play mode with a trained policy.

    Args:
        checkpoint_ref: Checkpoint reference (path or "latest").
        scene_path: Path to the MuJoCo scene XML file.
    """
    # Resolve and load checkpoint
    ckpt_path = resolve_resume_ref(checkpoint_ref)
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Reconstruct policy
    policy = build_policy(ckpt["config"])
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()
    print(
        f"Policy: {ckpt['config']['policy']['policy_type']} "
        f"(hidden={ckpt['config']['policy'].get('hidden_size', 'n/a')})"
    )

    # Image dimensions from checkpoint config
    img_h = ckpt["config"]["policy"]["image_height"]
    img_w = ckpt["config"]["policy"]["image_width"]

    # Create environment (raw env, no training wrapper)
    env = SimEnv(
        scene_path=scene_path,
        render_width=img_w,
        render_height=img_h,
    )

    model = env.model
    data = env.data

    # Physics sub-steps per action (read from checkpoint config, fallback to 5 for legacy checkpoints)
    env_cfg = ckpt["config"].get("env", {})
    mujoco_steps_per_action = env_cfg.get("mujoco_steps_per_action", 5)

    # Gait phase encoding: if the policy was trained with gait phase inputs,
    # we must compute and append them in play mode too.
    gait_phase_period = env_cfg.get("gait_phase_period", 0.0)
    gait_phase_dim = 4 if gait_phase_period > 0 else 0
    gait_step_count = 0
    control_dt = mujoco_steps_per_action * model.opt.timestep

    # Target move step size and arena bounds
    target_step = 0.3  # meters per key press
    arena_bound = 4.0

    # Speed multiplier: run N inference+physics steps per rendered frame
    speed_idx = 1  # index into SPEED_OPTIONS, default 2x

    # Thread-safe intent queue: key_callback appends, main loop drains
    pending_intents = []

    def key_callback(keycode):
        if keycode == KEY_UP:
            pending_intents.append(("move", 1, target_step))  # +Y (forward)
        elif keycode == KEY_DOWN:
            pending_intents.append(("move", 1, -target_step))  # -Y (backward)
        elif keycode == KEY_RIGHT:
            pending_intents.append(("move", 0, target_step))  # +X (right)
        elif keycode == KEY_LEFT:
            pending_intents.append(("move", 0, -target_step))  # -X (left)
        elif keycode == KEY_SLASH:
            pending_intents.append(("randomize",))
        elif keycode == KEY_EQUAL:
            pending_intents.append(("speed_up",))
        elif keycode == KEY_MINUS:
            pending_intents.append(("speed_down",))

    # Launch passive viewer
    viewer = mujoco.viewer.launch_passive(model, data, key_callback=key_callback)

    # Start in tracking camera mode following the robot
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = env.bot_body_id
    viewer.cam.distance = 3.0
    viewer.cam.elevation = -30

    # Reset hidden state for LSTM
    if hasattr(policy, "reset_hidden"):
        policy.reset_hidden(batch_size=1, device="cpu")

    print("Play mode active.")
    print(
        "  Arrows: move target   /: randomize   -/=: speed   [ ]: cameras   Esc: quit"
    )
    print("  Tip: In viewer UI, set Font: 150% and Color: Orange for best experience")

    try:
        while viewer.is_running():
            frame_start = time.monotonic()

            # --- Process pending intents from key_callback ---
            reset_hidden = False
            if pending_intents:
                intents = list(pending_intents)
                pending_intents.clear()

                for intent in intents:
                    if intent[0] == "move":
                        axis, delta = intent[1], intent[2]
                        pos = model.body_pos[env.target_body_id].copy()
                        pos[axis] = np.clip(
                            pos[axis] + delta, -arena_bound, arena_bound
                        )
                        model.body_pos[env.target_body_id] = pos
                        reset_hidden = True
                    elif intent[0] == "randomize":
                        angle = np.random.uniform(0, 2 * np.pi)
                        dist = np.random.uniform(1.0, 3.5)
                        model.body_pos[env.target_body_id] = [
                            dist * np.cos(angle),
                            dist * np.sin(angle),
                            0.08,
                        ]
                        reset_hidden = True
                    elif intent[0] == "speed_up":
                        speed_idx = min(speed_idx + 1, len(SPEED_OPTIONS) - 1)
                    elif intent[0] == "speed_down":
                        speed_idx = max(speed_idx - 1, 0)

                if reset_hidden:
                    mujoco.mj_forward(model, data)

            if reset_hidden and hasattr(policy, "reset_hidden"):
                policy.reset_hidden(batch_size=1, device="cpu")

            # --- Run N inference+physics steps per frame ---
            steps_per_frame = SPEED_OPTIONS[speed_idx]
            for _ in range(steps_per_frame):
                obs = env.get_camera_image()  # uint8 (H, W, 3)
                obs_normalized = obs.astype(np.float32) / 255.0
                obs_tensor = torch.from_numpy(obs_normalized).unsqueeze(0)

                sensor_tensor = None
                if env.sensor_dim > 0 and getattr(policy, "sensor_input_size", 0) > 0:
                    sensors = env.get_sensor_data()
                    if gait_phase_dim > 0:
                        t = gait_step_count * control_dt
                        phase = 2 * np.pi * t / gait_phase_period
                        gait_phase = np.array(
                            [
                                np.sin(phase),
                                np.cos(phase),
                                np.sin(phase + np.pi),
                                np.cos(phase + np.pi),
                            ],
                            dtype=np.float32,
                        )
                        sensors = np.concatenate([sensors, gait_phase])
                    sensor_tensor = torch.from_numpy(sensors).unsqueeze(0)

                with torch.no_grad():
                    action = policy.get_deterministic_action(
                        obs_tensor, sensors=sensor_tensor
                    )
                    action = action.cpu().numpy()[0]

                data.ctrl[: env.num_actuators] = action
                for _ in range(mujoco_steps_per_action):
                    mujoco.mj_step(model, data)
                gait_step_count += 1

            # --- Update viewer overlays ---
            distance = env.get_distance_to_target()
            speed = SPEED_OPTIONS[speed_idx]

            viewer.set_texts(
                [
                    (
                        mujoco.mjtFontScale.mjFONTSCALE_150,
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        f"Distance: {distance:.2f}m",
                        f"Motors: L={action[0]:+.2f}  R={action[1]:+.2f}",
                    ),
                    (
                        mujoco.mjtFontScale.mjFONTSCALE_100,
                        mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                        f"Speed: {speed}x (-/=)  Arrows: move  /: randomize  [ ]: cameras",
                        "",
                    ),
                ]
            )

            # --- Sync viewer ---
            viewer.sync()

            # --- Frame rate: target ~30 fps rendering ---
            elapsed = time.monotonic() - frame_start
            sleep_time = (1.0 / 30.0) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass

    viewer.close()
    env.close()
    print("Play mode ended.")
