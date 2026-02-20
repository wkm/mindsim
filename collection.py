"""
Episode collection and advantage computation.

Contains collect_episode() for running episodes, plus compute_reward_to_go(),
compute_gae(), and log_episode_value_trace() for return/advantage estimation.
"""

import numpy as np
import rerun as rr
import torch

import rerun_logger


def collect_episode(env, policy, device="cpu", log_rerun=False, deterministic=False, *, hierarchy):
    """
    Run one episode and collect data.

    Args:
        env: TrainingEnv instance
        policy: Neural network policy
        device: torch device
        log_rerun: Log episode to Rerun for visualization
        deterministic: If True, use mean actions (no sampling) for evaluation.
                       If False, sample from policy distribution for training.
        hierarchy: RewardHierarchy instance (required).

    Returns:
        episode_data: Dict with observations, actions, rewards, log_probs (if not deterministic), etc.
    """
    # Rerun namespace depends on mode
    ns = "eval" if deterministic else "training"

    observations = []
    sensor_data = []
    actions = []
    log_probs = []  # Only populated when not deterministic
    rewards = []
    distances = []

    obs = env.reset()
    env_config = env.last_reset_config
    has_sensors = env.sensor_dim > 0 and getattr(policy, "sensor_input_size", 0) > 0

    # Reset LSTM hidden state if policy has one
    if hasattr(policy, "reset_hidden"):
        policy.reset_hidden(batch_size=1, device=device)
    done = False
    truncated = False
    total_reward = 0
    steps = 0
    info = {}

    # Accumulate reward component sums for per-episode breakdown
    reward_component_keys = hierarchy.reward_component_keys()
    reward_component_sums = {k: 0.0 for k in reward_component_keys}

    # Accumulate raw reward inputs (physical measures)
    raw_input_keys = [
        "raw_up_z", "raw_forward_vel", "raw_energy",
        "raw_contact_count", "raw_action_jerk",
    ]
    raw_input_sums = {k: 0.0 for k in raw_input_keys}

    # Total path length (sum of per-step distance_moved)
    total_path_length = 0.0

    # Joint velocity sensor indices (for joint_activity)
    joint_vel_indices = []
    for si in env.sensor_info:
        if si["name"].endswith("_vel") and si["dim"] == 1:
            joint_vel_indices.append(si["adr"])
    joint_vel_sum = 0.0  # sum of mean(|joint_vel|) across steps

    # Track trajectory for Rerun
    trajectory_points = []

    # Only record camera if it's being used by the policy and not in walking stage
    show_camera = log_rerun and not getattr(env, "in_walking_stage", False)
    if not getattr(policy, "uses_visual_input", True):
        show_camera = False

    # Set up video encoder for Rerun (H.264 instead of per-frame JPEG)
    video_encoder = None
    if show_camera:
        # Compute actual control frequency from physics config
        action_dt = env.env.model.opt.timestep * env.mujoco_steps_per_action
        control_fps = max(1, int(round(1.0 / action_dt)))
        video_encoder = rerun_logger.VideoEncoder(
            f"{ns}/camera",
            width=env.observation_shape[1],
            height=env.observation_shape[0],
            fps=control_fps,
        )

    while not (done or truncated):
        # Convert observation to torch tensor
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
        sensor_tensor = None
        if has_sensors:
            sensor_tensor = torch.from_numpy(env.current_sensors).unsqueeze(0).to(device)

        # Get action (deterministic or stochastic)
        with torch.no_grad():
            if deterministic:
                action = policy.get_deterministic_action(obs_tensor, sensors=sensor_tensor)
                action = action.cpu().numpy()[0]
                log_prob = None
            else:
                action, log_prob = policy.sample_action(obs_tensor, sensors=sensor_tensor)
                action = action.cpu().numpy()[0]
                log_prob = log_prob.cpu().numpy()[0]

        # Store data
        observations.append(obs)
        if has_sensors:
            sensor_data.append(env.current_sensors.copy())
        actions.append(action)
        if not deterministic:
            log_probs.append(log_prob)

        # Take step
        obs, reward, done, truncated, info = env.step(action)
        rewards.append(reward)
        distances.append(info["distance"])
        total_reward += reward
        for k in reward_component_keys:
            reward_component_sums[k] += info.get(k, 0.0)
        for k in raw_input_keys:
            raw_input_sums[k] += info.get(k, 0.0)
        total_path_length += info.get("distance_moved", 0.0)
        if joint_vel_indices and env.sensor_dim > 0:
            sensor_vals = env.current_sensors
            jv = np.abs([sensor_vals[i] for i in joint_vel_indices])
            joint_vel_sum += float(np.mean(jv))

        # Log to Rerun in real-time
        if log_rerun:
            rr.set_time("step", sequence=steps)
            if video_encoder:
                video_encoder.log_frame(observations[-1])
            # Sensor inputs
            if has_sensors:
                sensor_vals = env.current_sensors
                for si in env.sensor_info:
                    adr, dim = si["adr"], si["dim"]
                    if dim == 1:
                        rr.log(f"{ns}/sensors/{si['name']}", rr.Scalars([sensor_vals[adr]]))
                    else:
                        for d in range(dim):
                            rr.log(
                                f"{ns}/sensors/{si['name']}/{d}",
                                rr.Scalars([sensor_vals[adr + d]]),
                            )
            for i, name in enumerate(env.actuator_names):
                rr.log(f"{ns}/action/{name}", rr.Scalars([action[i]]))
            rr.log(f"{ns}/reward/total", rr.Scalars([reward]))
            rr.log(f"{ns}/reward/cumulative", rr.Scalars([total_reward]))
            rr.log(f"{ns}/distance_to_target", rr.Scalars([info["distance"]]))

            # Log body transforms
            rerun_logger.log_body_transforms(env, namespace=ns)

            # Build and log trajectory
            trajectory_points.append(info["position"])
            if len(trajectory_points) > 1:
                rr.log(
                    f"{ns}/trajectory",
                    rr.LineStrips3D([trajectory_points], colors=[[100, 200, 100]]),
                )

        steps += 1

    # Episode wall time in simulation seconds
    action_dt = env.env.model.opt.timestep * env.mujoco_steps_per_action
    episode_time = steps * action_dt

    # Flush video encoder and log episode summary
    if log_rerun:
        if video_encoder:
            video_encoder.flush()
        rr.log(f"{ns}/episode/total_reward", rr.Scalars([total_reward]))
        rr.log(f"{ns}/episode/final_distance", rr.Scalars([info["distance"]]))
        rr.log(f"{ns}/episode/steps", rr.Scalars([steps]))

    # Compute action statistics (per-actuator by name)
    actions_array = np.array(actions)

    # Determine if episode was a success
    if info.get("in_walking_stage"):
        # Walking stage: success = survived AND moved forward enough
        forward_dist = info.get("forward_distance", 0.0)
        success = bool(truncated and not done and forward_dist >= env.walking_success_min_forward)
    else:
        # Standard stages: success = reached target
        success = bool(done and info["distance"] < env.success_distance)

    result = {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "distances": distances,
        "total_reward": total_reward,
        "steps": steps,
        "final_distance": info["distance"],
        "success": success,
        "env_config": env_config,
        "done": done,
        "truncated": truncated,
        "patience_truncated": info.get("patience_truncated", False),
        "joint_stagnation_truncated": info.get("joint_stagnation_truncated", False),
        "fell": info.get("fell", False),
        "forward_distance": info.get("forward_distance", 0.0),
        "reward_components": reward_component_sums,
        "raw_inputs": {
            "distance_to_target": info["distance"],
            "torso_height": info.get("torso_height", 0.0),
            "up_z": raw_input_sums["raw_up_z"] / max(steps, 1),
            "forward_vel": raw_input_sums["raw_forward_vel"] / max(steps, 1),
            "energy": raw_input_sums["raw_energy"] / max(steps, 1),
            "contact_frac": raw_input_sums["raw_contact_count"] / max(steps, 1),
            "action_jerk": raw_input_sums["raw_action_jerk"] / max(steps, 1),
            # New measures
            "forward_distance": info.get("forward_distance", 0.0),
            "lateral_drift": info.get("lateral_drift", 0.0),
            "total_path_length": total_path_length,
            "avg_speed": total_path_length / (episode_time if episode_time > 0 else 1.0),
            "survival_time": episode_time,
            "joint_activity": joint_vel_sum / max(steps, 1),
        },
    }
    if has_sensors:
        result["sensor_data"] = sensor_data
    # Per-actuator stats keyed by actuator name
    for i, name in enumerate(env.actuator_names):
        motor_actions = actions_array[:, i]
        result[f"{name}_mean"] = float(np.mean(motor_actions))
        result[f"{name}_std"] = float(np.std(motor_actions))

    # Store final observation for GAE bootstrapping on truncated episodes
    if truncated and not done:
        result["final_observation"] = obs
        if has_sensors:
            result["final_sensors"] = env.current_sensors.copy()

    # Only include log_probs for training episodes
    if not deterministic:
        result["log_probs"] = log_probs

    return result


def compute_reward_to_go(rewards, gamma=0.99):
    """
    Compute discounted reward-to-go for a single episode.

    Args:
        rewards: Tensor of per-step rewards (T,)
        gamma: Discount factor

    Returns:
        reward_to_go: Tensor of discounted returns (T,)
    """
    reward_to_go = torch.zeros_like(rewards)
    running_sum = 0
    for t in reversed(range(len(rewards))):
        running_sum = rewards[t] + gamma * running_sum
        reward_to_go[t] = running_sum
    return reward_to_go


def compute_gae(rewards, values, gamma, gae_lambda, next_value=0.0):
    """
    Compute Generalized Advantage Estimation (GAE).

    Args:
        rewards: Tensor of per-step rewards (T,)
        values: Tensor of per-step value estimates (T,)
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        next_value: Bootstrap value for the state after the last step
                    (0 for terminated episodes, V(s_final) for truncated)

    Returns:
        advantages: Tensor of GAE advantages (T,)
        returns: Tensor of GAE returns / value targets (T,)
    """
    T = len(rewards)
    advantages = torch.zeros(T, dtype=rewards.dtype, device=rewards.device)
    gae = 0.0
    for t in reversed(range(T)):
        next_val = next_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_val - values[t]
        gae = delta + gamma * gae_lambda * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


def log_episode_value_trace(
    policy, episode_data, gamma, gae_lambda, device="cpu", namespace="eval"
):
    """
    Run a forward pass on a completed episode to log V(s_t) and A(s_t) to Rerun.

    Gives per-step visibility into the value function's beliefs during the episode.
    """
    obs = torch.from_numpy(np.array(episode_data["observations"])).to(device)
    acts = torch.from_numpy(np.array(episode_data["actions"])).to(device)
    rewards = torch.tensor(episode_data["rewards"], dtype=torch.float32, device=device)
    sensors = None
    if "sensor_data" in episode_data and len(episode_data["sensor_data"]) > 0:
        sensors = torch.from_numpy(np.array(episode_data["sensor_data"])).to(device)

    with torch.no_grad():
        # Use evaluate_actions even for deterministic episodes — we just need values
        # Need dummy actions for the log_prob computation but we only use the values
        _, values, _ = policy.evaluate_actions(obs, acts, sensors=sensors)
        advantages, returns = compute_gae(
            rewards, values, gamma, gae_lambda, next_value=0.0
        )

    values_np = values.cpu().numpy()
    advantages_np = advantages.cpu().numpy()
    returns_np = returns.cpu().numpy()
    cumulative_reward = np.cumsum(episode_data["rewards"])

    for t in range(len(values_np)):
        rr.set_time("step", sequence=t)
        rr.log(f"{namespace}/value/V_s", rr.Scalars([values_np[t]]))
        rr.log(f"{namespace}/value/advantage", rr.Scalars([advantages_np[t]]))
        rr.log(
            f"{namespace}/value/cumulative_reward", rr.Scalars([cumulative_reward[t]])
        )
        rr.log(f"{namespace}/value/gae_return", rr.Scalars([returns_np[t]]))
