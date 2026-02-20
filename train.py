"""
Training loop and run orchestration.

Policy networks are in policies.py, episode collection in collection.py,
and training algorithms (REINFORCE, PPO) in algorithms.py.
"""

import logging
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from queue import Empty, Queue

log = logging.getLogger(__name__)

import numpy as np
import torch
import torch.optim as optim

import wandb
from algorithms import train_step_batched, train_step_ppo  # noqa: F401 — re-export
from checkpoint import load_checkpoint, resolve_resume_ref, save_checkpoint
from collection import (  # noqa: F401 — re-export
    collect_episode,
    compute_gae,
    compute_reward_to_go,
    log_episode_value_trace,
)
from config import Config
from dashboard import AnsiDashboard, TuiDashboard
from git_utils import get_git_branch, get_git_sha
from parallel import ParallelCollector, resolve_num_workers
from policies import LSTMPolicy, MLPPolicy, TinyPolicy  # noqa: F401 — re-export
from rerun_wandb import RerunWandbLogger
from run_manager import (
    RunInfo,
    bot_display_name,
    bot_name_from_scene_path,
    create_run_dir,
    generate_run_name,
    init_wandb_for_run,
    save_run_info,
)
from training_env import TrainingEnv
from tweaks import apply_tweaks, load_tweaks


def _quat_rotate(quat, vec):
    """Rotate a 3D vector by a quaternion [w, x, y, z]."""
    w, x, y, z = quat
    vx, vy, vz = vec
    # q * v * q_conj (Hamilton product)
    t = np.array([
        2.0 * (y * vz - z * vy),
        2.0 * (z * vx - x * vz),
        2.0 * (x * vy - y * vx),
    ])
    return np.array([
        vx + w * t[0] + (y * t[2] - z * t[1]),
        vy + w * t[1] + (z * t[0] - x * t[2]),
        vz + w * t[2] + (x * t[1] - y * t[0]),
    ])


def verify_forward_direction(env, log_fn=print):
    """
    Verify that the forward axis is correctly configured by running a physics check.

    Three checks:
    1. Geometry: forward_velocity_axis points toward walking_target_pos
    2. Physics: applying a world-frame force in the forward direction produces
       positive forward displacement and decreasing distance to target
    3. Velocity: position-based velocity correctly reports the direction of movement

    Raises AssertionError if the forward direction is wrong.
    """
    import mujoco

    if not env.has_walking_stage:
        return  # Only relevant for bots with a walking stage

    fwd_axis = env.forward_velocity_axis
    target = np.array(env.walking_target_pos)

    # Check 1: Forward axis should point toward the walking target from origin
    target_dir = target[:3] / (np.linalg.norm(target[:3]) + 1e-8)
    axis_dot_target = float(np.dot(fwd_axis, target_dir))
    log_fn(f"  Geometry check: dot(forward_axis, target_dir) = {axis_dot_target:.3f}")
    assert axis_dot_target > 0.5, (
        f"Forward axis {fwd_axis.tolist()} does not point toward walking target "
        f"{target.tolist()} (dot={axis_dot_target:.3f}, expected > 0.5)"
    )

    # Check 2: Teleport the bot 1m forward by editing qpos, verify position & distance
    env.reset()
    model = env.env.model
    data = env.env.data
    base_body_id = env.env.bot_body_id
    start_pos = env.env.get_bot_position().copy()
    start_dist = env.env.get_distance_to_target()

    # Move the root body 1m forward by adjusting the root joint positions
    nudge = 1.0  # meters
    for j in range(model.njnt):
        if model.jnt_bodyid[j] != base_body_id:
            continue
        jnt_type = model.jnt_type[j]
        qpos_adr = model.jnt_qposadr[j]

        if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
            # Freejoint qpos: [x, y, z, qw, qx, qy, qz]
            data.qpos[qpos_adr + 0] += nudge * fwd_axis[0]
            data.qpos[qpos_adr + 1] += nudge * fwd_axis[1]
            data.qpos[qpos_adr + 2] += nudge * fwd_axis[2]
            break
        elif jnt_type == mujoco.mjtJoint.mjJNT_SLIDE:
            jnt_axis = model.jnt_axis[j]
            proj = float(np.dot(fwd_axis, jnt_axis))
            if abs(proj) > 0.1:
                data.qpos[qpos_adr] += nudge * proj

    mujoco.mj_forward(model, data)

    end_pos = env.env.get_bot_position()
    end_dist = env.env.get_distance_to_target()
    displacement = end_pos - start_pos
    forward_displacement = float(np.dot(displacement, fwd_axis))

    log_fn(f"  Teleport check: displacement={forward_displacement:.4f}m, "
           f"dist_delta={start_dist - end_dist:.4f}m")

    assert forward_displacement > 0.5, (
        f"Forward displacement after teleport is too small ({forward_displacement:.4f})! "
        f"Expected ~1.0m. Root joint axes may not align with forward_velocity_axis."
    )
    assert end_dist < start_dist, (
        f"Distance to walking target increased after teleporting forward "
        f"({start_dist:.3f} -> {end_dist:.3f})! "
        f"Forward axis may be pointing away from the walking target."
    )

    # Check 3: Verify position-based velocity reports correct direction
    # Set qvel to move forward, step the physics, and check xpos displacement.
    env.reset()

    vel_nudge = 2.0  # m/s
    for j in range(model.njnt):
        if model.jnt_bodyid[j] != base_body_id:
            continue
        jnt_type = model.jnt_type[j]
        qvel_adr = model.jnt_dofadr[j]

        if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
            qpos_adr = model.jnt_qposadr[j]
            quat = data.qpos[qpos_adr + 3:qpos_adr + 7].copy()
            quat_conj = np.array([quat[0], -quat[1], -quat[2], -quat[3]])
            local_fwd = _quat_rotate(quat_conj, fwd_axis)
            data.qvel[qvel_adr + 0] = vel_nudge * local_fwd[0]
            data.qvel[qvel_adr + 1] = vel_nudge * local_fwd[1]
            data.qvel[qvel_adr + 2] = vel_nudge * local_fwd[2]
            break
        elif jnt_type == mujoco.mjtJoint.mjJNT_SLIDE:
            jnt_axis = model.jnt_axis[j]
            proj = float(np.dot(fwd_axis, jnt_axis))
            if abs(proj) > 0.1:
                data.qvel[qvel_adr] = vel_nudge * proj

    pos_before = env.env.get_bot_position().copy()
    n_steps = 10
    for _ in range(n_steps):
        mujoco.mj_step(model, data)
    pos_after = env.env.get_bot_position()
    dt = n_steps * model.opt.timestep
    vel = (pos_after - pos_before) / dt
    forward_vel = float(np.dot(vel, fwd_axis))

    log_fn(f"  Velocity check: forward_vel={forward_vel:.3f} m/s "
           f"(vel={[f'{v:.3f}' for v in vel]})")

    assert forward_vel > 0.5, (
        f"Position-based forward velocity is too small ({forward_vel:.3f})! "
        f"Expected ~{vel_nudge:.1f}. vel={vel.tolist()}, axis={fwd_axis.tolist()}. "
        f"Forward axis may not match the root joint axes."
    )

    log_fn("  Forward direction: OK")

    # Reset env to clean state for training
    env.reset()


def set_terminal_title(title):
    """Set terminal tab/window title using ANSI escape sequence."""
    sys.stdout.write(f"\033]0;{title}\007")
    sys.stdout.flush()


def set_terminal_progress(percent):
    """
    Set terminal progress indicator using OSC 9;4 sequence.

    Supported by iTerm2, Windows Terminal, and others.
    Shows progress bar in terminal tab.

    Args:
        percent: 0-100 for progress, or -1 to clear
    """
    if percent < 0:
        # Clear progress indicator
        sys.stdout.write("\033]9;4;0\007")
    else:
        # Set progress (state=1 means normal progress)
        sys.stdout.write(f"\033]9;4;1;{int(percent)}\007")
    sys.stdout.flush()


def notify_completion(run_name, message=None):
    """Show macOS notification and play sound when training completes."""
    if message is None:
        message = f"Training run '{run_name}' has finished."

    # macOS notification
    subprocess.run(
        [
            "osascript",
            "-e",
            f'display notification "{message}" with title "MindSim Training Complete" sound name "Glass"',
        ],
        check=False,
    )

    # Fallback beep in case notification sound doesn't play
    print("\a", end="", flush=True)


def generate_run_notes():
    """
    Use Claude CLI to generate a summary of what changed since last run.

    Returns:
        str: Markdown-formatted notes for W&B, or None if generation fails
    """
    try:
        # Get git info
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        # Get diff from parent commit
        diff = subprocess.run(
            ["git", "diff", "HEAD~1", "--stat"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        # Get full diff for context (limited to avoid token limits)
        full_diff = subprocess.run(
            ["git", "diff", "HEAD~1"], capture_output=True, text=True, check=True
        ).stdout[:4000]  # Limit to ~4k chars

        # Get recent commit message
        commit_msg = subprocess.run(
            ["git", "log", "-1", "--pretty=%B"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        # Build prompt for Claude CLI
        prompt = f"""Summarize what changed in this training run as a short bullet-point list for experiment tracking.

Branch: {branch}
Recent commit: {commit_msg}

Changes:
{diff}

Diff excerpt:
{full_diff}

Rules:
- Output ONLY markdown bullet points (- ...), nothing else
- 2-5 bullets covering: what changed, what hypothesis is being tested, key parameter/architecture differences from baseline
- Be concise and technical, each bullet one line
- No preamble, no headings, no trailing text"""

        # Call Claude CLI (handles auth automatically)
        result = subprocess.run(
            ["claude", "-p", prompt, "--model", "haiku"],
            capture_output=True,
            text=True,
            check=True,
        )
        summary = result.stdout.strip()

        # Format as markdown notes
        notes = f"""{summary}

---
**Branch:** `{branch}` | **Commit:** {commit_msg.split(chr(10))[0][:60]}"""
        return notes

    except Exception as e:
        print(f"  Note: Could not generate run notes: {e}")
        return None


### Policies, collection, and algorithms extracted to separate modules.
### Re-exported above for backward compatibility.

class CommandChannel:
    """Thread-safe command channel between TUI and training loop.

    Uses two internal queues so flow-control commands (pause/unpause/step/stop)
    and action commands (checkpoint/log_rerun/curriculum) never interfere.
    wait_if_paused() only reads the flow queue; drain_actions() only reads the
    action queue.  No put-back logic needed.
    """

    _FLOW = frozenset({"pause", "unpause", "step", "stop"})

    def __init__(self):
        self._flow: Queue[str] = Queue()
        self._actions: Queue[str] = Queue()

    def send(self, cmd: str):
        """Queue a command (called from TUI thread)."""
        if cmd in self._FLOW:
            self._flow.put(cmd)
        else:
            self._actions.put(cmd)

    def wait_if_paused(self) -> bool:
        """Block while paused.  Returns True if stop was requested."""
        paused = False

        # Drain pending flow commands
        while not self._flow.empty():
            try:
                cmd = self._flow.get_nowait()
            except Empty:
                break
            if cmd == "pause":
                paused = True
            elif cmd == "unpause":
                paused = False
            elif cmd == "step":
                return False
            elif cmd == "stop":
                return True

        # Block until un-paused, stepped, or stopped
        while paused:
            time.sleep(0.05)
            while not self._flow.empty():
                try:
                    cmd = self._flow.get_nowait()
                except Empty:
                    break
                if cmd == "unpause":
                    paused = False
                elif cmd == "step":
                    return False
                elif cmd == "stop":
                    return True

        return False

    def drain_actions(self) -> list[str]:
        """Return all pending action commands (non-blocking)."""
        cmds: list[str] = []
        while not self._actions.empty():
            try:
                cmds.append(self._actions.get_nowait())
            except Empty:
                break
        return cmds


def _train_loop(
    cfg,
    dashboard,
    smoketest=False,
    resume=None,
    num_workers_override=None,
    commands: CommandChannel | None = None,
    app=None,
    log_fn=print,
):
    """
    Core training loop.

    Args:
        cfg: Config object
        dashboard: Dashboard instance (TuiDashboard or AnsiDashboard)
        smoketest: Whether this is a smoketest run
        resume: Resume ref string (local path or wandb artifact)
        num_workers_override: Override num_workers from config
        commands: Optional CommandChannel for TUI commands
        app: Optional TUI app for pushing metadata
        log_fn: Function for print-style logging (print or dashboard.message).
                In TUI mode this is dashboard.message (shows in log area).
                In CLI mode this is print (shows in terminal).
    """
    # In TUI mode, verbose setup messages are noise in the log area.
    # Use _verbose for setup chatter, log_fn for important events only.
    is_tui = app is not None
    _verbose = (lambda msg: None) if is_tui else log_fn

    bot_name = bot_name_from_scene_path(cfg.env.scene_path)
    robot_name = bot_display_name(bot_name)
    log.info("Training %s (%s) smoketest=%s", robot_name, cfg.training.algorithm, smoketest)
    log.info("Config: %s", cfg.to_wandb_config())
    _verbose("=" * 60)
    _verbose(f"Training {robot_name} ({cfg.training.algorithm})")
    _verbose("=" * 60)
    log_fn(f"Setting up {robot_name} ({cfg.training.algorithm})...")

    # Generate run name and create run directory
    run_name = generate_run_name(bot_name, cfg.policy.policy_type)
    run_dir = create_run_dir(run_name)
    log_fn(f"Run: {run_name}")

    # Generate run notes using Claude (summarizes git changes)
    if smoketest:
        run_notes = None
    else:
        log_fn("Generating run notes...")
        run_notes = generate_run_notes()
        if run_notes:
            _verbose(run_notes)

    # Initialize W&B (single project 'mindsim', tags for filtering)
    log_fn("Connecting to W&B...")
    init_wandb_for_run(run_name, cfg, bot_name, smoketest=smoketest, run_notes=run_notes)

    wandb_url = None
    if not smoketest and wandb.run:
        wandb_url = wandb.run.url
        log_fn(f"W&B: {wandb_url}")
        _verbose(f"  W&B run: {wandb_url}")

    # Write initial run_info.json
    run_info = RunInfo(
        name=run_name,
        bot_name=bot_name,
        policy_type=cfg.policy.policy_type,
        algorithm=cfg.training.algorithm,
        scene_path=cfg.env.scene_path,
        status="running",
        wandb_id=wandb.run.id if wandb.run else None,
        wandb_url=wandb_url,
        git_branch=get_git_branch(),
        git_sha=get_git_sha(),
        created_at=datetime.now().isoformat(),
        tags=[bot_name, cfg.training.algorithm, cfg.policy.policy_type],
    )
    save_run_info(run_dir, run_info)
    _verbose(f"  Run directory: {run_dir}/")

    # Push run metadata to TUI header
    if app is not None:
        from main import _get_experiment_info

        branch = get_git_branch()
        hypothesis = _get_experiment_info(branch)
        app.call_from_thread(
            app.set_header, run_name, branch, cfg.training.algorithm, wandb_url,
            robot_name, hypothesis,
        )

    # Create environment from config
    log_fn("Creating environment...")
    env = TrainingEnv.from_config(cfg.env)
    _verbose(f"  Observation shape: {env.observation_shape}")
    _verbose(f"  Action shape: {env.action_shape}")
    _verbose(f"  Control frequency: {cfg.env.control_frequency_hz} Hz")

    # Verify forward direction is correct (smoketest: assert, training: warn)
    if env.has_walking_stage:
        _verbose("Verifying forward direction...")
        verify_forward_direction(env, log_fn=_verbose)

    # Log bot model info
    mj_model = env.env.model
    bot_scene = env.env.scene_path.name
    bot_info = (
        f"Bot: {bot_scene} | "
        f"{mj_model.nbody} bodies, {mj_model.njnt} joints, "
        f"{mj_model.nu} actuators, {mj_model.ncam} cameras"
    )
    _verbose(f"  {bot_info}")
    log_fn(bot_info)

    # Create policy from config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.policy.use_mlp:
        policy = MLPPolicy(
            num_actions=cfg.policy.fc_output_size,
            hidden_size=cfg.policy.hidden_size,
            init_std=cfg.policy.init_std,
            max_log_std=cfg.policy.max_log_std,
            sensor_input_size=cfg.policy.sensor_input_size,
        ).to(device)
    elif cfg.policy.use_lstm:
        policy = LSTMPolicy(
            image_height=cfg.policy.image_height,
            image_width=cfg.policy.image_width,
            hidden_size=cfg.policy.hidden_size,
            num_actions=cfg.policy.fc_output_size,
            init_std=cfg.policy.init_std,
            max_log_std=cfg.policy.max_log_std,
            sensor_input_size=cfg.policy.sensor_input_size,
        ).to(device)
    else:
        policy = TinyPolicy(
            image_height=cfg.policy.image_height,
            image_width=cfg.policy.image_width,
            num_actions=cfg.policy.fc_output_size,
            init_std=cfg.policy.init_std,
            max_log_std=cfg.policy.max_log_std,
            sensor_input_size=cfg.policy.sensor_input_size,
        ).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=cfg.training.learning_rate)

    # Print architecture and parameter count
    num_params = sum(p.numel() for p in policy.parameters())
    _verbose(f"Policy ({device}): {num_params:,} parameters")
    log_fn(f"Policy: {cfg.policy.policy_type} ({num_params:,} params) on {device}")

    # Log model info to wandb
    wandb.config.update({"policy_params": num_params}, allow_val_change=True)

    # Watch model for gradient/parameter histograms
    # log_freq is in backward passes (episodes), not steps
    wandb.watch(policy, log="all", log_freq=10)

    # Resume from checkpoint if requested
    resumed_batch_idx = 0
    resumed_episode_count = 0
    resumed_curriculum_stage = None
    resumed_stage_progress = None
    resumed_mastery_count = None
    if resume:
        resume_ref = resolve_resume_ref(resume)
        log_fn(f"Loading checkpoint: {resume_ref}")
        ckpt = load_checkpoint(resume_ref, cfg, device=str(device))
        policy.load_state_dict(ckpt["policy_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        # optimizer.load_state_dict restores LR from checkpoint — override
        # with current config so soft key changes are respected
        for pg in optimizer.param_groups:
            pg["lr"] = cfg.training.learning_rate
        resumed_batch_idx = ckpt["batch_idx"]
        resumed_episode_count = ckpt["episode_count"]
        resumed_curriculum_stage = ckpt["curriculum_stage"]
        resumed_stage_progress = ckpt["stage_progress"]
        resumed_mastery_count = ckpt["mastery_count"]
        log_fn(
            f"  Resumed from batch {resumed_batch_idx}, episode {resumed_episode_count}"
        )
        _verbose(
            f"  Curriculum: stage {resumed_curriculum_stage}, progress {resumed_stage_progress:.2f}"
        )
        wandb.config.update({"resumed_from": resume_ref}, allow_val_change=True)
        # Add resume info to wandb notes
        resume_note = f"\n**Resumed from:** `{resume_ref}`"
        if run_notes:
            run_notes += resume_note
        else:
            run_notes = resume_note
        if wandb.run and not wandb.run.disabled:
            wandb.run.notes = run_notes

    # Initialize Rerun-WandB integration (skip in smoketest)
    rr_wandb = None
    if not smoketest:
        rr_wandb = RerunWandbLogger(run_dir=str(run_dir))
        _verbose(f"  Rerun recordings: {rr_wandb.run_dir}/")

    # Set up parallel episode collection
    num_workers = (
        num_workers_override
        if num_workers_override is not None
        else cfg.training.num_workers
    )
    num_workers = resolve_num_workers(num_workers)
    collector = None
    if num_workers > 1:
        _verbose(f"Starting {num_workers} parallel workers...")
        collector = ParallelCollector(num_workers, cfg.env, cfg.policy)
        if is_tui:
            log_fn(f"Started {num_workers} parallel workers")
        else:
            _verbose("  Workers ready")
    else:
        _verbose("Using serial episode collection (num_workers=1)")

    # Training loop - use config values
    batch_size = cfg.training.batch_size
    log_rerun_every = cfg.training.log_rerun_every
    last_rerun_time = time.perf_counter()  # Wall-clock time of last Rerun recording

    # Curriculum config (shorthand for readability)
    curr = cfg.curriculum

    # Rolling window for success rate tracking
    success_history = deque(maxlen=curr.window_size)
    curriculum_stage = (
        resumed_curriculum_stage if resumed_curriculum_stage is not None else 1
    )
    stage_progress = (
        resumed_stage_progress if resumed_stage_progress is not None else 0.0
    )
    mastery_count = resumed_mastery_count if resumed_mastery_count is not None else 0

    # If resuming with proven mastery and more stages are now available, advance
    # immediately. This handles resuming a "final" checkpoint from a run with
    # fewer stages (e.g., 3-stage checkpoint resumed with num_stages=4).
    if (
        resume
        and stage_progress >= 1.0
        and mastery_count >= cfg.training.mastery_batches
        and curriculum_stage < curr.num_stages
    ):
        old_stage = curriculum_stage
        curriculum_stage += 1
        stage_progress = 0.0
        mastery_count = 0
        log_fn(
            f"  Checkpoint had mastered stage {old_stage} — advancing to stage {curriculum_stage}/{curr.num_stages}"
        )

    _verbose(
        f"Training {curr.num_stages}-stage curriculum until mastery (success>={cfg.training.mastery_threshold:.0%} for {cfg.training.mastery_batches} batches)..."
    )
    if is_tui:
        log_fn(
            f"Training started: {curr.num_stages}-stage curriculum, batch_size={batch_size}"
        )

    # Timing accumulators
    timing = {
        "collect": 0.0,
        "eval": 0.0,
        "train": 0.0,
        "log": 0.0,
        "rerun": 0.0,
    }

    # Rolling window for eval success rate (used for curriculum)
    eval_success_history = deque(maxlen=curr.window_size)

    episode_count = resumed_episode_count
    batch_idx = resumed_batch_idx
    mastered = False
    max_batches = cfg.training.max_batches
    batches_this_session = 0
    stop_requested = False
    while (
        not mastered
        and not stop_requested
        and (max_batches is None or batch_idx < max_batches)
    ):
        # Block while paused (checks for unpause/step/stop)
        if commands and commands.wait_if_paused():
            stop_requested = True
            dashboard.message("Stopping...")
            break

        batch_start_time = time.perf_counter()
        # Set curriculum stage for this batch
        env.set_curriculum_stage(curriculum_stage, stage_progress, curr.num_stages)

        # Check for live hyperparameter tweaks
        tweaks = load_tweaks()
        if tweaks:
            changes = apply_tweaks(cfg, optimizer, env, tweaks)
            batch_size = cfg.training.batch_size
            log_rerun_every = cfg.training.log_rerun_every
            for name, old, new in changes:
                dashboard.message(f"  tweak: {name} {old} -> {new}")
            if changes:
                wandb.log({f"tweaks/{name}": new for name, _, new in changes})

        # Drain TUI action commands
        pending_save = None  # (trigger, aliases) or None
        force_rerun = False
        for cmd in (commands.drain_actions() if commands else []):
            if cmd == "checkpoint":
                pending_save = ("manual", [])
                dashboard.message("Checkpoint will be saved after this batch")
            elif cmd == "log_rerun":
                force_rerun = True
                dashboard.message("Rerun recording queued for next eval")
            elif cmd == "advance_curriculum":
                old = stage_progress
                stage_progress = min(1.0, stage_progress + 0.1)
                dashboard.message(
                    f"Curriculum advanced: {old:.2f} -> {stage_progress:.2f}"
                )
            elif cmd == "regress_curriculum":
                old = stage_progress
                stage_progress = max(0.0, stage_progress - 0.1)
                dashboard.message(
                    f"Curriculum regressed: {old:.2f} -> {stage_progress:.2f}"
                )
            elif cmd == "rerun_freq_down":
                old = log_rerun_every
                log_rerun_every = max(batch_size, log_rerun_every // 2)
                cfg.training.log_rerun_every = log_rerun_every
                dashboard.message(
                    f"Rerun recording interval: {old} -> {log_rerun_every} episodes"
                )
            elif cmd == "rerun_freq_up":
                old = log_rerun_every
                log_rerun_every = log_rerun_every * 2
                cfg.training.log_rerun_every = log_rerun_every
                dashboard.message(
                    f"Rerun recording interval: {old} -> {log_rerun_every} episodes"
                )
            elif cmd == "stop":
                stop_requested = True
                dashboard.message("Stopping after this batch...")

        # Overall progress: (stage-1 + progress) / num_stages
        overall_progress = (curriculum_stage - 1 + stage_progress) / curr.num_stages
        progress_pct = 100 * overall_progress
        set_terminal_title(
            f"{progress_pct:.0f}% S{curriculum_stage} p={stage_progress:.2f} {run_name}"
        )
        set_terminal_progress(progress_pct)

        # Log to Rerun every N batches, or if >60s since last recording
        log_every_n_batches = max(1, log_rerun_every // batch_size)
        time_since_rerun = time.perf_counter() - last_rerun_time
        should_log_rerun_this_batch = rr_wandb is not None and (
            batch_idx % log_every_n_batches == 0
            or time_since_rerun >= 60.0
            or force_rerun
        )

        # Collect a batch of episodes
        t_collect_start = time.perf_counter()
        if collector is not None:
            episode_batch = collector.collect_batch(
                policy,
                batch_size,
                curriculum_stage,
                stage_progress,
                num_stages=curr.num_stages,
            )
        else:
            episode_batch = []
            for _ in range(batch_size):
                episode_data = collect_episode(
                    env,
                    policy,
                    device,
                )
                episode_batch.append(episode_data)
        timing["collect_batch"] = time.perf_counter() - t_collect_start
        timing["collect"] += timing["collect_batch"]

        # Update observation normalizer for MLPPolicy
        if cfg.policy.use_mlp:
            all_sensors = []
            for ep in episode_batch:
                if "sensor_data" in ep:
                    all_sensors.append(torch.from_numpy(np.array(ep["sensor_data"])))
            if all_sensors:
                policy.update_normalizer(torch.cat(all_sensors, dim=0).to(device))

        batch_rewards = [ep["total_reward"] for ep in episode_batch]
        batch_distances = [ep["final_distance"] for ep in episode_batch]
        batch_steps = [ep["steps"] for ep in episode_batch]
        batch_successes = [ep["success"] for ep in episode_batch]
        episode_count += batch_size

        # Train on batch of episodes
        t_train_start = time.perf_counter()
        if cfg.training.algorithm == "PPO":
            (
                policy_loss,
                value_loss,
                entropy,
                grad_norm,
                policy_std,
                clip_fraction,
                approx_kl,
                explained_variance,
                mean_value,
                mean_return,
            ) = train_step_ppo(
                policy,
                optimizer,
                episode_batch,
                gamma=cfg.training.gamma,
                gae_lambda=cfg.training.gae_lambda,
                clip_epsilon=cfg.training.clip_epsilon,
                ppo_epochs=cfg.training.ppo_epochs,
                entropy_coeff=cfg.training.entropy_coeff,
                value_coeff=cfg.training.value_coeff,
            )
            loss = policy_loss  # For backward-compatible logging
        else:
            loss, grad_norm, policy_std, entropy = train_step_batched(
                policy,
                optimizer,
                episode_batch,
                entropy_coeff=cfg.training.entropy_coeff,
            )
        timing["train_batch"] = time.perf_counter() - t_train_start
        timing["train"] += timing["train_batch"]

        # Aggregate batch statistics
        avg_reward = np.mean(batch_rewards)
        avg_distance = np.mean(batch_distances)
        avg_steps = np.mean(batch_steps)
        best_reward = np.max(batch_rewards)
        worst_reward = np.min(batch_rewards)

        # Track training success rate (for logging, not curriculum)
        batch_success_rate = np.mean(batch_successes)
        success_history.append(batch_success_rate)
        rolling_success_rate = np.mean(success_history)

        # Run deterministic evaluation episodes for curriculum decisions
        t_eval_start = time.perf_counter()
        eval_successes = []
        if curr.use_eval_for_curriculum:
            for eval_idx in range(curr.eval_episodes_per_batch):
                # Log first eval episode to Rerun if this is a logging batch
                log_this_eval = should_log_rerun_this_batch and (eval_idx == 0)

                if log_this_eval:
                    try:
                        t_rerun_start = time.perf_counter()
                        dashboard.message("Recording eval episode to Rerun...")
                        rr_wandb.start_episode(episode_count, env, namespace="eval")
                        timing["rerun"] += time.perf_counter() - t_rerun_start
                    except Exception:
                        log.exception("Failed to start Rerun recording")
                        dashboard.message("Rerun recording failed (see log)")
                        log_this_eval = False

                eval_data = collect_episode(
                    env, policy, device, log_rerun=log_this_eval, deterministic=True
                )
                eval_successes.append(eval_data["success"])

                if log_this_eval:
                    try:
                        t_rerun_start = time.perf_counter()
                        # Log per-step value function traces for PPO
                        if cfg.training.algorithm == "PPO":
                            log_episode_value_trace(
                                policy,
                                eval_data,
                                gamma=cfg.training.gamma,
                                gae_lambda=cfg.training.gae_lambda,
                                device=device,
                                namespace="eval",
                            )
                        rr_wandb.finish_episode(eval_data, upload_artifact=True)
                        timing["rerun"] += time.perf_counter() - t_rerun_start
                        last_rerun_time = time.perf_counter()
                    except Exception:
                        log.exception("Failed to finish Rerun recording")
                        dashboard.message("Rerun upload failed (see log)")

            eval_success_rate = np.mean(eval_successes)
            eval_success_history.append(eval_success_rate)
            rolling_eval_success_rate = np.mean(eval_success_history)
        else:
            # Fall back to training success rate if eval disabled
            eval_success_rate = batch_success_rate
            rolling_eval_success_rate = rolling_success_rate
        timing["eval_batch"] = time.perf_counter() - t_eval_start
        timing["eval"] += timing["eval_batch"]


        # Update curriculum based on EVAL success rate (deterministic)
        if (
            len(eval_success_history) >= curr.window_size
            or not curr.use_eval_for_curriculum
        ):
            rate_for_curriculum = (
                rolling_eval_success_rate
                if curr.use_eval_for_curriculum
                else rolling_success_rate
            )
            if rate_for_curriculum > curr.advance_threshold:
                stage_progress = min(1.0, stage_progress + curr.advance_rate)

        # Check for stage mastery: progress=1.0 AND sustained high success rate
        # Checkpoint saves are deferred until after batch_idx is incremented
        # so the saved batch_idx represents "resume from here" correctly.
        # pending_save may already be set by command queue (manual checkpoint)
        mastery_rate = (
            rolling_eval_success_rate
            if curr.use_eval_for_curriculum
            else rolling_success_rate
        )
        if stage_progress >= 1.0 and mastery_rate >= cfg.training.mastery_threshold:
            mastery_count += 1
            if mastery_count >= cfg.training.mastery_batches:
                if curriculum_stage >= curr.num_stages:
                    # Final stage mastered → training complete
                    pending_save = ("final", [f"stage{curriculum_stage}-mastered"])
                    mastered = True
                else:
                    # Save before advancing to next stage
                    pending_save = ("milestone", [f"stage{curriculum_stage}-mastered"])
                    # Advance to next stage
                    curriculum_stage += 1
                    stage_progress = 0.0
                    mastery_count = 0
                    # Clear success histories so new stage starts fresh
                    success_history.clear()
                    eval_success_history.clear()
                    dashboard.message(
                        f"  >>> Advanced to stage {curriculum_stage}/{curr.num_stages} <<<"
                    )
        else:
            mastery_count = 0  # Reset if we drop below mastery level

        log_dict = {
            "episode": episode_count,
            "batch": batch_idx,
            # Curriculum
            "curriculum/stage": curriculum_stage,
            "curriculum/stage_progress": stage_progress,
            "curriculum/overall_progress": overall_progress,
            "curriculum/max_episode_steps": env.max_episode_steps,
            # Training success rate (stochastic, with exploration noise)
            "curriculum/train_batch_success_rate": batch_success_rate,
            "curriculum/train_rolling_success_rate": rolling_success_rate,
            # Eval success rate (deterministic, no exploration noise)
            "curriculum/eval_batch_success_rate": eval_success_rate,
            "curriculum/eval_rolling_success_rate": rolling_eval_success_rate,
            # Batch metrics
            "batch/avg_reward": avg_reward,
            "batch/best_reward": best_reward,
            "batch/worst_reward": worst_reward,
            "batch/avg_final_distance": avg_distance,
            "batch/avg_steps": avg_steps,
            "batch/success_rate": batch_success_rate,
            "batch/loss": loss,
            "training/grad_norm": grad_norm,
            "training/entropy": entropy,
            # Reward histogram across batch
            "batch/reward_hist": wandb.Histogram(batch_rewards, num_bins=20),
        }

        # Per-actuator action stats and policy std (works for any bot)
        for i, name in enumerate(env.actuator_names):
            motor_actions = np.concatenate(
                [np.array(ep["actions"])[:, i] for ep in episode_batch]
            )
            log_dict[f"actions/{name}_mean"] = float(np.mean(motor_actions))
            log_dict[f"actions/{name}_std"] = float(np.std(motor_actions))
            log_dict[f"actions/{name}_hist"] = wandb.Histogram(
                motor_actions.tolist(), num_bins=20
            )
            if i < len(policy_std):
                log_dict[f"policy/std_{name}"] = policy_std[i]

        # Episode termination breakdown
        batch_truncated = [ep.get("truncated", False) for ep in episode_batch]
        batch_done = [ep.get("done", False) for ep in episode_batch]
        log_dict.update(
            {
                "batch/truncated_fraction": np.mean(batch_truncated),
                "batch/done_fraction": np.mean(batch_done),
                "batch/patience_truncated_fraction": np.mean(
                    [ep.get("patience_truncated", False) for ep in episode_batch]
                ),
                "batch/joint_stagnation_fraction": np.mean(
                    [ep.get("joint_stagnation_truncated", False) for ep in episode_batch]
                ),
                "batch/fell_fraction": np.mean(
                    [ep.get("fell", False) for ep in episode_batch]
                ),
            }
        )

        # Per-component reward breakdown (averaged across batch)
        component_keys = [
            "reward_distance", "reward_exploration", "reward_time",
            "reward_upright", "reward_alive", "reward_energy",
            "reward_contact", "reward_forward_velocity", "reward_smoothness",
        ]
        for key in component_keys:
            vals = [ep.get("reward_components", {}).get(key, 0.0) for ep in episode_batch]
            log_dict[f"rewards/{key}"] = np.mean(vals)

        # PPO-specific metrics
        if cfg.training.algorithm == "PPO":
            log_dict.update(
                {
                    "training/policy_loss": policy_loss,
                    "training/value_loss": value_loss,
                    "training/clip_fraction": clip_fraction,
                    "training/approx_kl": approx_kl,
                    "training/explained_variance": explained_variance,
                    "training/mean_value": mean_value,
                    "training/mean_return": mean_return,
                }
            )

        t_log_start = time.perf_counter()
        wandb.log(log_dict)
        timing["log"] += time.perf_counter() - t_log_start

        # Compute batch timing
        batch_time = time.perf_counter() - batch_start_time

        # Update dashboard
        dash_metrics = {
            # Episode performance
            "avg_reward": avg_reward,
            "best_reward": best_reward,
            "worst_reward": worst_reward,
            "avg_distance": avg_distance,
            "avg_steps": avg_steps,
            # Success rates
            "rolling_eval_success_rate": rolling_eval_success_rate,
            "eval_success_rate": eval_success_rate,
            "batch_success_rate": batch_success_rate,
            # Optimization
            "grad_norm": grad_norm,
            "entropy": entropy,
            "policy_std": policy_std,
            # Curriculum
            "curriculum_stage": curriculum_stage,
            "num_stages": curr.num_stages,
            "stage_progress": stage_progress,
            "mastery_count": mastery_count,
            "mastery_batches": cfg.training.mastery_batches,
            "max_episode_steps": env.max_episode_steps,
            "log_rerun_every": log_rerun_every,
            # Timing
            "batch_time": batch_time,
            "collect_time": timing["collect_batch"],
            "train_time": timing["train_batch"],
            "eval_time": timing["eval_batch"],
            "batch_size": batch_size,
            "episode_count": episode_count,
        }
        if cfg.training.algorithm == "PPO":
            dash_metrics.update(
                {
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "clip_fraction": clip_fraction,
                    "approx_kl": approx_kl,
                    "explained_variance": explained_variance,
                    "mean_value": mean_value,
                    "mean_return": mean_return,
                }
            )
        else:
            dash_metrics["loss"] = loss

        dashboard.update(batch_idx, dash_metrics)
        log.info(
            "batch %d  reward=%+.2f  dist=%.2fm  eval=%.0f%%  S%d",
            batch_idx, avg_reward, avg_distance,
            100 * rolling_eval_success_rate, curriculum_stage,
        )
        batch_idx += 1
        batches_this_session += 1

        # Save checkpoints (milestone/final take priority over periodic)
        try:
            if pending_save:
                trigger, aliases = pending_save
                save_checkpoint(
                    policy,
                    optimizer,
                    cfg,
                    curriculum_stage,
                    stage_progress,
                    mastery_count,
                    batch_idx,
                    episode_count,
                    trigger=trigger,
                    aliases=aliases,
                    run_dir=run_dir,
                )
            elif (
                cfg.training.checkpoint_every
                and batch_idx % cfg.training.checkpoint_every == 0
            ):
                save_checkpoint(
                    policy,
                    optimizer,
                    cfg,
                    curriculum_stage,
                    stage_progress,
                    mastery_count,
                    batch_idx,
                    episode_count,
                    trigger="periodic",
                    run_dir=run_dir,
                )
        except Exception:
            log.exception("Failed to save checkpoint")
            dashboard.message("Checkpoint save failed (see log)")

    dashboard.finish()

    # Print timing summary (verbose in CLI, compact in TUI)
    total_time = (
        timing["collect"]
        + timing["eval"]
        + timing["train"]
        + timing["log"]
        + timing["rerun"]
    )
    if total_time > 0 and batches_this_session > 0:
        _verbose("=" * 60)
        _verbose("Timing Summary")
        _verbose("=" * 60)
        _verbose(
            f"  Episode collection: {timing['collect']:>8.2f}s ({100 * timing['collect'] / total_time:>5.1f}%)"
        )
        _verbose(
            f"  Eval episodes:      {timing['eval']:>8.2f}s ({100 * timing['eval'] / total_time:>5.1f}%)"
        )
        _verbose(
            f"  Training step:      {timing['train']:>8.2f}s ({100 * timing['train'] / total_time:>5.1f}%)"
        )
        _verbose(
            f"  Wandb logging:      {timing['log']:>8.2f}s ({100 * timing['log'] / total_time:>5.1f}%)"
        )
        _verbose(
            f"  Rerun recording:    {timing['rerun']:>8.2f}s ({100 * timing['rerun'] / total_time:>5.1f}%)"
        )
        _verbose(f"  Total:              {total_time:>8.2f}s")
        _verbose(
            f"  Per-batch average ({batch_size} episodes/batch, {batches_this_session} batches):"
        )
        _verbose(
            f"    Collection: {1000 * timing['collect'] / batches_this_session:.1f}ms"
        )
        _verbose(
            f"    Training:   {1000 * timing['train'] / batches_this_session:.1f}ms"
        )

    # Update run_info with final state
    run_info.status = "completed"
    run_info.finished_at = datetime.now().isoformat()
    run_info.batch_idx = batch_idx
    run_info.episode_count = episode_count
    run_info.curriculum_stage = curriculum_stage
    save_run_info(run_dir, run_info)

    # Clean up
    if collector is not None:
        collector.close()
    set_terminal_title(f"Done: {run_name}")
    set_terminal_progress(-1)  # Clear progress indicator
    if not smoketest:
        notify_completion(run_name)
    wandb.finish()
    env.close()
    if smoketest:
        log_fn(f"Smoketest passed! ({batch_idx} batches, {episode_count} episodes)")
        log.info("Smoketest passed (%d batches, %d episodes)", batch_idx, episode_count)
    else:
        log_fn("Training complete!")
        log.info("Training complete (%d batches, %d episodes)", batch_idx, episode_count)


def run_training(
    app, commands: CommandChannel, smoketest=False, resume=None, num_workers=None, scene_path=None
):
    """
    Entry point for TUI-driven training (called from worker thread).

    Args:
        app: MindSimApp instance
        commands: CommandChannel for TUI commands
        smoketest: Whether to use smoketest config
        resume: Checkpoint resume reference
        num_workers: Worker count override
        scene_path: Override bot scene XML path
    """
    is_biped = scene_path and "biped" in scene_path
    is_walker2d = scene_path and "walker2d" in scene_path
    if smoketest:
        if is_biped:
            cfg = Config.for_biped_smoketest()
        elif is_walker2d:
            cfg = Config.for_walker2d_smoketest()
        else:
            cfg = Config.for_smoketest()
    elif is_biped:
        cfg = Config.for_biped()
    elif is_walker2d:
        cfg = Config.for_walker2d()
    else:
        cfg = Config()

    if scene_path:
        cfg.env.scene_path = scene_path

    dashboard = TuiDashboard(
        app=app,
        total_batches=cfg.training.max_batches,
        algorithm=cfg.training.algorithm,
    )

    _train_loop(
        cfg=cfg,
        dashboard=dashboard,
        smoketest=smoketest,
        resume=resume,
        num_workers_override=num_workers,
        commands=commands,
        app=app,
        log_fn=dashboard.message,
    )


def main(smoketest=False, bot=None, resume=None, num_workers=None, scene_path=None):
    """CLI entry point (headless, no TUI).

    Args:
        smoketest: Run fast end-to-end validation.
        bot: Bot name (e.g. "simplebiped"). Overrides scene_path.
        resume: Resume from checkpoint (local path or wandb artifact ref).
        num_workers: Number of parallel workers for episode collection.
        scene_path: Override bot scene XML path directly.
    """
    is_biped = (bot and "biped" in bot) or (scene_path and "biped" in scene_path)
    is_walker2d = (bot and "walker2d" in bot) or (scene_path and "walker2d" in scene_path)
    if smoketest:
        if is_biped:
            cfg = Config.for_biped_smoketest()
        elif is_walker2d:
            cfg = Config.for_walker2d_smoketest()
        else:
            cfg = Config.for_smoketest()
        print("[SMOKETEST MODE] Running fast end-to-end validation...")
    elif is_biped:
        cfg = Config.for_biped()
    elif is_walker2d:
        cfg = Config.for_walker2d()
    else:
        cfg = Config()

    if scene_path:
        cfg.env.scene_path = scene_path

    robot_name_cli = bot_display_name(bot_name_from_scene_path(cfg.env.scene_path))
    dashboard = AnsiDashboard(
        total_batches=cfg.training.max_batches,
        algorithm=cfg.training.algorithm,
        bot_name=robot_name_cli,
    )

    _train_loop(
        cfg=cfg,
        dashboard=dashboard,
        smoketest=smoketest,
        resume=resume,
        num_workers_override=num_workers,
    )
