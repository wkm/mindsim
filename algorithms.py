"""
Training algorithms: REINFORCE and PPO.

Contains train_step_batched() (REINFORCE) and train_step_ppo() (PPO with GAE).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collection import compute_gae, compute_reward_to_go


def train_step_batched(
    policy, optimizer, episode_batch, gamma=0.99, entropy_coeff=0.01
):
    """
    REINFORCE policy gradient training step on a batch of episodes.

    Computes reward-to-go for all episodes, normalizes advantages across
    the entire batch (so good episodes get positive advantage, bad episodes
    get negative), then takes one optimizer step.

    Includes an entropy bonus to prevent policy collapse: when advantage
    signal is weak (all episodes similar), the entropy term provides a
    non-zero gradient that keeps exploration alive.

    Args:
        policy: Stochastic neural network policy
        optimizer: PyTorch optimizer
        episode_batch: List of episode_data dicts from collect_episode
        gamma: Discount factor for reward-to-go
        entropy_coeff: Weight for entropy bonus (0 = disabled)

    Returns:
        avg_loss: Average loss across batch
        grad_norm: Gradient norm after averaging
        policy_std: Current policy standard deviation
        entropy: Policy entropy value
    """
    optimizer.zero_grad()

    # First pass: compute reward-to-go for all episodes
    all_rtg = []
    for episode_data in episode_batch:
        rewards = torch.tensor(episode_data["rewards"], dtype=torch.float32)
        all_rtg.append(compute_reward_to_go(rewards, gamma))

    # Normalize advantages across the entire batch
    all_rtg_cat = torch.cat(all_rtg)
    batch_mean = all_rtg_cat.mean()
    batch_std = all_rtg_cat.std()
    if batch_std > 1e-8:
        all_rtg = [(rtg - batch_mean) / (batch_std + 1e-8) for rtg in all_rtg]

    # Second pass: compute losses with batch-normalized advantages
    # Weight each episode by its length so every timestep contributes equally,
    # preventing long episodes from dominating the gradient.
    total_steps = sum(len(ep["rewards"]) for ep in episode_batch)
    total_loss = 0.0
    for episode_data, advantage in zip(episode_batch, all_rtg):
        observations = torch.from_numpy(np.array(episode_data["observations"]))
        actions = torch.from_numpy(np.array(episode_data["actions"]))
        sensors = None
        if "sensor_data" in episode_data:
            sensors = torch.from_numpy(np.array(episode_data["sensor_data"]))

        log_probs = policy.log_prob(observations, actions, sensors=sensors)
        # Sum (not mean) within episode, then divide by total batch timesteps
        loss = -torch.sum(advantage * log_probs) / total_steps

        loss.backward()
        total_loss += loss.item()

    avg_loss = total_loss / len(episode_batch)

    # Clip REINFORCE gradients before adding entropy bonus.
    # This prevents large REINFORCE gradients from drowning out the
    # entropy signal — without this, clip_grad_norm_ scales everything
    # together and the small entropy gradient on log_std gets zeroed out.
    total_grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
    total_grad_norm = total_grad_norm.item()

    # Entropy bonus: H(π) = 0.5 * (1 + log(2π)) + log_std per action dim
    # Applied AFTER clipping so the entropy gradient reaches log_std at
    # full strength every step, preventing std collapse.
    if entropy_coeff > 0:
        clamped_log_std = policy.log_std.clamp(max=policy.max_log_std)
        std = torch.exp(clamped_log_std)
        entropy = torch.distributions.Normal(torch.zeros_like(std), std).entropy().sum()
        entropy_loss = -entropy_coeff * entropy
        entropy_loss.backward()
        entropy_val = entropy.item()
    else:
        entropy_val = 0.0

    optimizer.step()

    # Get current policy std for logging
    clamped = policy.log_std.clamp(max=policy.max_log_std)
    policy_std = torch.exp(clamped).detach().cpu().numpy()

    return avg_loss, total_grad_norm, policy_std, entropy_val


def train_step_ppo(
    policy,
    optimizer,
    episode_batch,
    gamma=0.99,
    gae_lambda=0.95,
    clip_epsilon=0.2,
    ppo_epochs=4,
    entropy_coeff=0.01,
    value_coeff=0.5,
    max_grad_norm=0.5,
):
    """
    PPO training step on a batch of episodes.

    Phase 1: Compute GAE advantages (once, before epochs).
    Phase 2: K optimization epochs over the data with clipped surrogate objective.

    Args:
        policy: Stochastic neural network policy with evaluate_actions()
        optimizer: PyTorch optimizer
        episode_batch: List of episode_data dicts from collect_episode
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        clip_epsilon: PPO clip range
        ppo_epochs: Number of optimization passes over the data
        entropy_coeff: Entropy bonus coefficient
        value_coeff: Value loss coefficient
        max_grad_norm: Max gradient norm for clipping

    Returns:
        policy_loss: Average policy loss across epochs
        value_loss: Average value loss across epochs
        entropy: Average entropy across epochs
        grad_norm: Average gradient norm across epochs
        policy_std: Current policy standard deviation
        clip_fraction: Fraction of clipped ratios (last epoch)
        approx_kl: Approximate KL divergence (last epoch)
        explained_variance: How well V(s) predicts returns (1.0 = perfect)
        mean_value: Mean V(s) across batch
        mean_return: Mean GAE return across batch
    """
    device = next(policy.parameters()).device

    # Phase 1: Compute advantages with current policy (no grad)
    episode_data_tensors = []
    all_advantages = []
    all_returns = []

    with torch.no_grad():
        for ep in episode_batch:
            obs = torch.from_numpy(np.array(ep["observations"])).to(device)
            acts = torch.from_numpy(np.array(ep["actions"])).to(device)
            rewards = torch.tensor(ep["rewards"], dtype=torch.float32, device=device)
            old_log_probs = torch.tensor(
                ep["log_probs"], dtype=torch.float32, device=device
            )
            sensors = None
            if "sensor_data" in ep:
                sensors = torch.from_numpy(np.array(ep["sensor_data"])).to(device)

            _, values, _ = policy.evaluate_actions(obs, acts, sensors=sensors)

            # Bootstrap value for truncated episodes
            next_value = 0.0
            if ep.get("truncated") and not ep.get("done") and "final_observation" in ep:
                final_obs = (
                    torch.from_numpy(ep["final_observation"]).unsqueeze(0).to(device)
                )  # (1, H, W, 3)
                dummy_act = torch.zeros(1, policy.num_actions, device=device)
                final_sensors = None
                if "final_sensors" in ep:
                    final_sensors = (
                        torch.from_numpy(ep["final_sensors"]).unsqueeze(0).to(device)
                    )
                _, final_val, _ = policy.evaluate_actions(
                    final_obs, dummy_act, sensors=final_sensors
                )
                next_value = final_val[0].item()

            advantages, returns = compute_gae(
                rewards, values, gamma, gae_lambda, next_value
            )

            ep_tensors = {
                "observations": obs,
                "actions": acts,
                "old_log_probs": old_log_probs,
                "advantages": advantages,
                "returns": returns,
            }
            if sensors is not None:
                ep_tensors["sensors"] = sensors
            episode_data_tensors.append(ep_tensors)

            all_advantages.append(advantages)
            all_returns.append(returns)

    # Value function diagnostics (before normalization)
    all_adv_cat = torch.cat(all_advantages)
    all_ret_cat = torch.cat(all_returns)
    # Explained variance: 1 - Var(returns - values) / Var(returns)
    # values = returns - advantages (before normalization)
    all_val_cat = all_ret_cat - all_adv_cat
    ret_var = all_ret_cat.var()
    explained_variance = (
        1.0 - (all_ret_cat - all_val_cat).var() / (ret_var + 1e-8)
        if ret_var > 1e-8
        else 0.0
    )
    mean_value = all_val_cat.mean().item()
    mean_return = all_ret_cat.mean().item()

    # Normalize advantages across the entire batch
    adv_mean = all_adv_cat.mean()
    adv_std = all_adv_cat.std()
    if adv_std > 1e-8:
        for ep_tensors in episode_data_tensors:
            ep_tensors["advantages"] = (ep_tensors["advantages"] - adv_mean) / (
                adv_std + 1e-8
            )

    # Phase 2: PPO epochs
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_grad_norm = 0.0
    last_clip_fraction = 0.0
    last_approx_kl = 0.0

    for epoch in range(ppo_epochs):
        optimizer.zero_grad()

        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_entropy = 0.0
        epoch_clip_count = 0
        epoch_total_steps = 0
        epoch_kl_sum = 0.0

        total_steps = sum(len(ep["observations"]) for ep in episode_data_tensors)

        # Process episodes sequentially (LSTM hidden state requirement)
        for ep_tensors in episode_data_tensors:
            obs = ep_tensors["observations"]
            acts = ep_tensors["actions"]
            old_lp = ep_tensors["old_log_probs"]
            adv = ep_tensors["advantages"]
            ret = ep_tensors["returns"]
            sensors = ep_tensors.get("sensors")
            T = len(obs)

            new_log_probs, values, entropy = policy.evaluate_actions(
                obs, acts, sensors=sensors
            )

            # PPO clipped surrogate
            ratio = torch.exp(new_log_probs - old_lp)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv
            policy_loss = -torch.min(surr1, surr2).sum() / total_steps

            # Value loss
            value_loss = F.mse_loss(values, ret, reduction="sum") / total_steps

            # Combined loss
            loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy
            loss.backward()

            # Track metrics
            epoch_policy_loss += policy_loss.item() * T
            epoch_value_loss += value_loss.item() * T
            epoch_entropy += entropy.item() * T

            with torch.no_grad():
                clipped = ((ratio - 1.0).abs() > clip_epsilon).float().sum().item()
                epoch_clip_count += clipped
                epoch_total_steps += T
                epoch_kl_sum += (old_lp - new_log_probs).mean().item() * T

        grad_norm = nn.utils.clip_grad_norm_(
            policy.parameters(), max_norm=max_grad_norm
        )
        optimizer.step()

        total_policy_loss += epoch_policy_loss / epoch_total_steps
        total_value_loss += epoch_value_loss / epoch_total_steps
        total_entropy += epoch_entropy / epoch_total_steps
        total_grad_norm += grad_norm.item()

        if epoch == ppo_epochs - 1:
            last_clip_fraction = epoch_clip_count / max(epoch_total_steps, 1)
            last_approx_kl = epoch_kl_sum / max(epoch_total_steps, 1)

    # Average across epochs
    avg_policy_loss = total_policy_loss / ppo_epochs
    avg_value_loss = total_value_loss / ppo_epochs
    avg_entropy = total_entropy / ppo_epochs
    avg_grad_norm = total_grad_norm / ppo_epochs

    # Get current policy std for logging
    clamped = policy.log_std.clamp(max=policy.max_log_std)
    policy_std = torch.exp(clamped).detach().cpu().numpy()

    ev = (
        explained_variance.item()
        if torch.is_tensor(explained_variance)
        else explained_variance
    )

    return (
        avg_policy_loss,
        avg_value_loss,
        avg_entropy,
        avg_grad_norm,
        policy_std,
        last_clip_fraction,
        last_approx_kl,
        ev,
        mean_value,
        mean_return,
    )
