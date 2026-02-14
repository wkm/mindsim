"""
Fast end-to-end smoketest for the training pipeline.

Validates that env, policy, episode collection, training step,
and curriculum all work together. Runs in seconds.
"""

import os
import warnings

import numpy as np
import pytest
import torch

import wandb
from checkpoint import load_checkpoint, save_checkpoint, validate_checkpoint_config
from config import Config
from train import (
    LSTMPolicy,
    TinyPolicy,
    collect_episode,
    compute_gae,
    train_step_batched,
    train_step_ppo,
)
from training_env import TrainingEnv


def _smoketest_config():
    return Config.for_smoketest()


def _biped_smoketest_config():
    return Config.for_biped_smoketest()


def _make_env(cfg):
    return TrainingEnv.from_config(cfg.env)


class TestEnvironment:
    """Test the training environment basics."""

    def test_reset_returns_observation(self):
        cfg = _smoketest_config()
        env = _make_env(cfg)
        obs = env.reset()
        assert obs.shape == (64, 64, 3)
        assert obs.dtype == np.float32
        assert 0.0 <= obs.min() and obs.max() <= 1.0
        env.close()

    def test_step_returns_five_tuple(self):
        cfg = _smoketest_config()
        env = _make_env(cfg)
        env.reset()
        obs, reward, done, truncated, info = env.step([0.5, 0.5])
        assert obs.shape == (64, 64, 3)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(truncated, bool)
        assert "distance" in info
        env.close()

    def test_curriculum_stages(self):
        cfg = _smoketest_config()
        env = _make_env(cfg)
        for stage in [1, 2, 3]:
            env.set_curriculum_stage(stage, 0.5)
            obs = env.reset()
            assert obs.shape == (64, 64, 3)
        env.close()

    def test_episode_terminates(self):
        """Episodes should end within max_episode_steps."""
        cfg = _smoketest_config()
        env = _make_env(cfg)
        env.reset()
        steps = 0
        done = False
        truncated = False
        while not (done or truncated):
            _, _, done, truncated, _ = env.step([0.0, 0.0])
            steps += 1
        assert steps <= cfg.env.max_episode_steps
        env.close()


class TestPolicy:
    """Test policy networks."""

    def test_lstm_policy_forward(self):
        policy = LSTMPolicy(image_height=64, image_width=64, hidden_size=32)
        obs = torch.randn(1, 64, 64, 3)
        policy.reset_hidden(batch_size=1)
        mean, std = policy.forward(obs)
        assert mean.shape == (1, 2)
        assert std.shape == (2,)

    def test_tiny_policy_forward(self):
        policy = TinyPolicy(image_height=64, image_width=64)
        obs = torch.randn(1, 64, 64, 3)
        mean, std = policy.forward(obs)
        assert mean.shape == (1, 2)
        assert std.shape == (2,)

    def test_lstm_sample_action(self):
        policy = LSTMPolicy(image_height=64, image_width=64, hidden_size=32)
        policy.reset_hidden(batch_size=1)
        obs = torch.randn(1, 64, 64, 3)
        action, log_prob = policy.sample_action(obs)
        assert action.shape == (1, 2)
        assert log_prob.shape == (1,)
        assert (action >= -1.0).all() and (action <= 1.0).all()

    def test_lstm_log_prob_sequence(self):
        policy = LSTMPolicy(image_height=64, image_width=64, hidden_size=32)
        T = 5
        obs = torch.randn(T, 64, 64, 3)
        actions = torch.randn(T, 2)
        log_probs = policy.log_prob(obs, actions)
        assert log_probs.shape == (T,)


class TestEpisodeCollection:
    """Test collecting episodes from env with policy."""

    def test_collect_stochastic_episode(self):
        cfg = _smoketest_config()
        env = _make_env(cfg)
        policy = LSTMPolicy(image_height=64, image_width=64, hidden_size=32)

        data = collect_episode(env, policy, deterministic=False)
        assert len(data["observations"]) > 0
        assert len(data["actions"]) == len(data["observations"])
        assert len(data["rewards"]) == len(data["observations"])
        assert len(data["log_probs"]) == len(data["observations"])
        assert isinstance(data["total_reward"], float)
        assert isinstance(data["success"], bool)
        env.close()

    def test_collect_deterministic_episode(self):
        cfg = _smoketest_config()
        env = _make_env(cfg)
        policy = LSTMPolicy(image_height=64, image_width=64, hidden_size=32)

        data = collect_episode(env, policy, deterministic=True)
        assert "log_probs" not in data
        assert len(data["observations"]) > 0
        env.close()


class TestTrainingStep:
    """Test the REINFORCE training step."""

    def test_train_step_produces_loss(self):
        cfg = _smoketest_config()
        env = _make_env(cfg)
        policy = LSTMPolicy(image_height=64, image_width=64, hidden_size=32)
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        # Collect a small batch
        batch = []
        for _ in range(2):
            batch.append(collect_episode(env, policy, deterministic=False))

        loss, grad_norm, policy_std, entropy = train_step_batched(
            policy, optimizer, batch
        )
        assert isinstance(loss, float)
        assert not np.isnan(loss)
        assert grad_norm >= 0
        assert len(policy_std) == 2
        assert entropy > 0  # Entropy should be positive for any non-degenerate policy
        env.close()

    def test_train_step_updates_parameters(self):
        cfg = _smoketest_config()
        env = _make_env(cfg)
        policy = LSTMPolicy(image_height=64, image_width=64, hidden_size=32)
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        # Snapshot params before
        params_before = {name: p.clone() for name, p in policy.named_parameters()}

        batch = [collect_episode(env, policy, deterministic=False) for _ in range(2)]
        train_step_batched(policy, optimizer, batch)

        # At least some params should have changed
        any_changed = False
        for name, p in policy.named_parameters():
            if not torch.equal(p, params_before[name]):
                any_changed = True
                break
        assert any_changed, "Training step should update at least some parameters"
        env.close()


class TestParallelCollection:
    """Test parallel episode collection with torch.multiprocessing."""

    def test_parallel_collect_returns_valid_episodes(self):
        """Parallel workers should produce valid episode data."""
        cfg = _smoketest_config()
        policy = LSTMPolicy(image_height=64, image_width=64, hidden_size=32)

        from parallel import ParallelCollector

        collector = ParallelCollector(
            num_workers=2, env_config=cfg.env, policy_config=cfg.policy
        )

        episodes = collector.collect_batch(
            policy, batch_size=2, curriculum_stage=1, stage_progress=0.5
        )

        assert len(episodes) == 2
        for ep in episodes:
            assert len(ep["observations"]) > 0
            assert len(ep["actions"]) == len(ep["observations"])
            assert len(ep["rewards"]) == len(ep["observations"])
            assert len(ep["log_probs"]) == len(ep["observations"])
            assert isinstance(ep["total_reward"], float)
            assert isinstance(ep["success"], bool)

        collector.close()

    def test_parallel_deterministic_episodes(self):
        """Parallel deterministic (eval) collection should work."""
        cfg = _smoketest_config()
        policy = LSTMPolicy(image_height=64, image_width=64, hidden_size=32)

        from parallel import ParallelCollector

        collector = ParallelCollector(
            num_workers=2, env_config=cfg.env, policy_config=cfg.policy
        )

        episodes = collector.collect_batch(
            policy,
            batch_size=2,
            curriculum_stage=1,
            stage_progress=0.5,
            deterministic=True,
        )

        assert len(episodes) == 2
        for ep in episodes:
            assert "log_probs" not in ep
            assert len(ep["observations"]) > 0

        collector.close()

    def test_parallel_train_step(self):
        """Episodes from parallel collection should work with train_step_batched."""
        cfg = _smoketest_config()
        policy = LSTMPolicy(image_height=64, image_width=64, hidden_size=32)
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        from parallel import ParallelCollector

        collector = ParallelCollector(
            num_workers=2, env_config=cfg.env, policy_config=cfg.policy
        )

        batch = collector.collect_batch(
            policy, batch_size=2, curriculum_stage=1, stage_progress=0.0
        )
        loss, grad_norm, policy_std, entropy = train_step_batched(
            policy, optimizer, batch
        )
        assert isinstance(loss, float)
        assert not np.isnan(loss)

        collector.close()


class TestEndToEnd:
    """Full pipeline integration test."""

    def test_full_pipeline(self):
        """Run a mini training loop: collect, train, advance curriculum."""
        cfg = _smoketest_config()
        env = _make_env(cfg)
        policy = LSTMPolicy(
            image_height=cfg.policy.image_height,
            image_width=cfg.policy.image_width,
            hidden_size=cfg.policy.hidden_size,
        )
        optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.training.learning_rate)

        # Disable wandb
        wandb.init(mode="disabled")

        for batch_idx in range(3):
            env.set_curriculum_stage(1, min(1.0, batch_idx * 0.5))

            # Collect batch
            batch = []
            for _ in range(cfg.training.batch_size):
                batch.append(collect_episode(env, policy, deterministic=False))

            # Train
            loss, grad_norm, policy_std, entropy = train_step_batched(
                policy, optimizer, batch
            )
            assert not np.isnan(loss)

            # Eval
            eval_data = collect_episode(env, policy, deterministic=True)
            assert isinstance(eval_data["success"], bool)

        wandb.finish()
        env.close()


# ── Biped-specific tests ──


class TestBipedEnvironment:
    """Test the biped training environment."""

    def test_reset_returns_observation(self):
        cfg = _biped_smoketest_config()
        env = _make_env(cfg)
        obs = env.reset()
        assert obs.shape == (64, 64, 3)
        assert obs.dtype == np.float32
        env.close()

    def test_actuator_count(self):
        cfg = _biped_smoketest_config()
        env = _make_env(cfg)
        assert env.num_actuators == 6
        assert len(env.actuator_names) == 6
        assert env.action_shape == (6,)
        env.close()

    def test_step_with_six_actions(self):
        cfg = _biped_smoketest_config()
        env = _make_env(cfg)
        env.reset()
        action = [0.0] * 6
        obs, reward, done, truncated, info = env.step(action)
        assert obs.shape == (64, 64, 3)
        assert isinstance(reward, float)
        assert "distance" in info
        assert "torso_height" in info
        env.close()

    def test_fall_detection(self):
        """Extreme actions should eventually trigger fall detection."""
        cfg = _biped_smoketest_config()
        env = _make_env(cfg)
        env.reset()
        fell = False
        for _ in range(cfg.env.max_episode_steps):
            _, _, done, truncated, info = env.step([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
            if info.get("has_fallen", False):
                fell = True
                break
        # Either it fell or the episode ended — both are valid
        assert fell or done or truncated
        env.close()

    def test_biped_rewards_present(self):
        """Biped config should produce upright/alive/energy rewards."""
        cfg = _biped_smoketest_config()
        env = _make_env(cfg)
        env.reset()
        _, _, _, _, info = env.step([0.0] * 6)
        assert "reward_upright" in info
        assert "reward_alive" in info
        assert "reward_energy" in info
        env.close()


class TestBipedPolicy:
    """Test policy networks with 6-action output."""

    def test_lstm_policy_six_actions(self):
        policy = LSTMPolicy(
            image_height=64, image_width=64, hidden_size=32, num_actions=6
        )
        obs = torch.randn(1, 64, 64, 3)
        policy.reset_hidden(batch_size=1)
        mean, std = policy.forward(obs)
        assert mean.shape == (1, 6)
        assert std.shape == (6,)

    def test_tiny_policy_six_actions(self):
        policy = TinyPolicy(image_height=64, image_width=64, num_actions=6)
        obs = torch.randn(1, 64, 64, 3)
        mean, std = policy.forward(obs)
        assert mean.shape == (1, 6)
        assert std.shape == (6,)

    def test_lstm_sample_action_six(self):
        policy = LSTMPolicy(
            image_height=64, image_width=64, hidden_size=32, num_actions=6
        )
        policy.reset_hidden(batch_size=1)
        obs = torch.randn(1, 64, 64, 3)
        action, log_prob = policy.sample_action(obs)
        assert action.shape == (1, 6)
        assert log_prob.shape == (1,)
        assert (action >= -1.0).all() and (action <= 1.0).all()

    def test_lstm_log_prob_sequence_six(self):
        policy = LSTMPolicy(
            image_height=64, image_width=64, hidden_size=32, num_actions=6
        )
        T = 5
        obs = torch.randn(T, 64, 64, 3)
        actions = torch.randn(T, 6)
        log_probs = policy.log_prob(obs, actions)
        assert log_probs.shape == (T,)


class TestBipedEpisodeCollection:
    """Test episode collection with the biped."""

    def test_collect_biped_episode(self):
        cfg = _biped_smoketest_config()
        env = _make_env(cfg)
        policy = LSTMPolicy(
            image_height=64, image_width=64, hidden_size=32, num_actions=6
        )

        data = collect_episode(env, policy, deterministic=False)
        assert len(data["observations"]) > 0
        assert len(data["actions"]) == len(data["observations"])
        # Each action should be 6-dimensional
        assert data["actions"][0].shape == (6,)
        assert isinstance(data["total_reward"], float)
        env.close()


class TestBipedTrainingStep:
    """Test training step with biped episodes."""

    def test_train_step_six_actions(self):
        cfg = _biped_smoketest_config()
        env = _make_env(cfg)
        policy = LSTMPolicy(
            image_height=64, image_width=64, hidden_size=32, num_actions=6
        )
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        batch = [collect_episode(env, policy, deterministic=False) for _ in range(2)]
        loss, grad_norm, policy_std, entropy = train_step_batched(
            policy, optimizer, batch
        )
        assert isinstance(loss, float)
        assert not np.isnan(loss)
        assert grad_norm >= 0
        assert len(policy_std) == 6
        assert entropy > 0
        env.close()


class TestPPO:
    """Test PPO-specific components."""

    def test_evaluate_actions_lstm_shapes(self):
        """evaluate_actions should return correct shapes for LSTMPolicy."""
        policy = LSTMPolicy(image_height=64, image_width=64, hidden_size=32)
        T = 5
        obs = torch.randn(T, 64, 64, 3)
        actions = torch.tanh(torch.randn(T, 2))
        log_probs, values, entropy = policy.evaluate_actions(obs, actions)
        assert log_probs.shape == (T,)
        assert values.shape == (T,)
        assert entropy.shape == ()

    def test_evaluate_actions_tiny_shapes(self):
        """evaluate_actions should return correct shapes for TinyPolicy."""
        policy = TinyPolicy(image_height=64, image_width=64)
        B = 4
        obs = torch.randn(B, 64, 64, 3)
        actions = torch.tanh(torch.randn(B, 2))
        log_probs, values, entropy = policy.evaluate_actions(obs, actions)
        assert log_probs.shape == (B,)
        assert values.shape == (B,)
        assert entropy.shape == ()

    def test_compute_gae_known_values(self):
        """GAE with lambda=1 should equal discounted returns minus values."""
        rewards = torch.tensor([1.0, 1.0, 1.0])
        values = torch.tensor([0.5, 0.5, 0.5])
        gamma = 0.99
        # lambda=1 makes GAE equivalent to full discounted returns - V
        advantages, returns = compute_gae(
            rewards, values, gamma, gae_lambda=1.0, next_value=0.0
        )
        assert advantages.shape == (3,)
        assert returns.shape == (3,)
        # Check last step: delta = 1.0 + 0.99*0 - 0.5 = 0.5
        assert abs(advantages[2].item() - 0.5) < 1e-5
        # returns = advantages + values
        assert torch.allclose(returns, advantages + values)

    def test_compute_gae_with_bootstrap(self):
        """GAE should bootstrap from next_value for truncated episodes."""
        rewards = torch.tensor([1.0, 1.0])
        values = torch.tensor([0.5, 0.5])
        adv_no_boot, _ = compute_gae(rewards, values, 0.99, 0.95, next_value=0.0)
        adv_with_boot, _ = compute_gae(rewards, values, 0.99, 0.95, next_value=10.0)
        # Bootstrapping should increase advantages
        assert adv_with_boot[-1] > adv_no_boot[-1]

    def test_train_step_ppo_produces_loss(self):
        """PPO training step should produce valid losses."""
        cfg = _smoketest_config()
        env = _make_env(cfg)
        policy = LSTMPolicy(image_height=64, image_width=64, hidden_size=32)
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        batch = [collect_episode(env, policy, deterministic=False) for _ in range(2)]
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
        ) = train_step_ppo(policy, optimizer, batch, ppo_epochs=2)

        assert isinstance(policy_loss, float)
        assert isinstance(value_loss, float)
        assert not np.isnan(policy_loss)
        assert not np.isnan(value_loss)
        assert grad_norm >= 0
        assert len(policy_std) == 2
        assert entropy > 0
        assert 0.0 <= clip_fraction <= 1.0
        assert isinstance(explained_variance, float)
        assert isinstance(mean_value, float)
        assert isinstance(mean_return, float)
        env.close()

    def test_train_step_ppo_updates_parameters(self):
        """PPO training step should update at least some parameters."""
        cfg = _smoketest_config()
        env = _make_env(cfg)
        policy = LSTMPolicy(image_height=64, image_width=64, hidden_size=32)
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        params_before = {name: p.clone() for name, p in policy.named_parameters()}
        batch = [collect_episode(env, policy, deterministic=False) for _ in range(2)]
        train_step_ppo(policy, optimizer, batch, ppo_epochs=2)

        any_changed = False
        for name, p in policy.named_parameters():
            if not torch.equal(p, params_before[name]):
                any_changed = True
                break
        assert any_changed, "PPO training step should update at least some parameters"
        env.close()

    def test_ppo_end_to_end_pipeline(self):
        """Full PPO pipeline: collect -> train -> eval cycle."""
        cfg = _smoketest_config()
        env = _make_env(cfg)
        policy = LSTMPolicy(
            image_height=cfg.policy.image_height,
            image_width=cfg.policy.image_width,
            hidden_size=cfg.policy.hidden_size,
        )
        optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.training.learning_rate)

        wandb.init(mode="disabled")

        for batch_idx in range(2):
            env.set_curriculum_stage(1, min(1.0, batch_idx * 0.5))
            batch = [
                collect_episode(env, policy, deterministic=False)
                for _ in range(cfg.training.batch_size)
            ]

            (
                policy_loss,
                value_loss,
                entropy,
                grad_norm,
                policy_std,
                clip_frac,
                approx_kl,
                ev,
                mv,
                mr,
            ) = train_step_ppo(
                policy, optimizer, batch, ppo_epochs=cfg.training.ppo_epochs
            )
            assert not np.isnan(policy_loss)
            assert not np.isnan(value_loss)

            eval_data = collect_episode(env, policy, deterministic=True)
            assert isinstance(eval_data["success"], bool)

        wandb.finish()
        env.close()

    def test_collect_episode_returns_done_truncated(self):
        """collect_episode should return done and truncated fields."""
        cfg = _smoketest_config()
        env = _make_env(cfg)
        policy = LSTMPolicy(image_height=64, image_width=64, hidden_size=32)

        data = collect_episode(env, policy, deterministic=False)
        assert "done" in data
        assert "truncated" in data
        assert isinstance(data["done"], bool)
        assert isinstance(data["truncated"], bool)
        env.close()


class TestCheckpoint:
    """Test checkpoint save/load roundtrip."""

    def test_roundtrip_save_load(self, tmp_path):
        """Save checkpoint, load into fresh policy, verify state matches."""
        cfg = _smoketest_config()
        env = _make_env(cfg)
        policy = LSTMPolicy(
            image_height=cfg.policy.image_height,
            image_width=cfg.policy.image_width,
            hidden_size=cfg.policy.hidden_size,
        )
        optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.training.learning_rate)

        wandb.init(mode="disabled")

        # Do a training step so optimizer has state
        batch = [collect_episode(env, policy, deterministic=False) for _ in range(2)]
        train_step_ppo(policy, optimizer, batch, ppo_epochs=2)

        # Save checkpoint
        ckpt_path = tmp_path / "test_ckpt.pt"
        ckpt = {
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "curriculum_stage": 2,
            "stage_progress": 0.75,
            "mastery_count": 5,
            "batch_idx": 42,
            "episode_count": 1000,
            "config": cfg.to_wandb_config(),
        }
        torch.save(ckpt, ckpt_path)

        # Load into fresh policy
        policy2 = LSTMPolicy(
            image_height=cfg.policy.image_height,
            image_width=cfg.policy.image_width,
            hidden_size=cfg.policy.hidden_size,
        )
        optimizer2 = torch.optim.Adam(
            policy2.parameters(), lr=cfg.training.learning_rate
        )

        loaded = load_checkpoint(str(ckpt_path), cfg)
        policy2.load_state_dict(loaded["policy_state_dict"])
        optimizer2.load_state_dict(loaded["optimizer_state_dict"])

        # Verify state matches
        for (n1, p1), (n2, p2) in zip(
            policy.named_parameters(), policy2.named_parameters()
        ):
            assert n1 == n2
            assert torch.equal(p1, p2), f"Parameter {n1} mismatch after load"

        assert loaded["curriculum_stage"] == 2
        assert loaded["stage_progress"] == 0.75
        assert loaded["mastery_count"] == 5
        assert loaded["batch_idx"] == 42
        assert loaded["episode_count"] == 1000

        wandb.finish()
        env.close()

    def test_architecture_mismatch_raises(self):
        """Loading checkpoint with different hidden_size should raise ValueError."""
        ckpt_config = {
            "policy": {
                "policy_type": "LSTMPolicy",
                "hidden_size": 256,
                "image_height": 64,
                "image_width": 64,
            },
            "training": {},
            "curriculum": {},
            "env": {},
        }
        current_config = {
            "policy": {
                "policy_type": "LSTMPolicy",
                "hidden_size": 32,
                "image_height": 64,
                "image_width": 64,
            },
            "training": {},
            "curriculum": {},
            "env": {},
        }
        with pytest.raises(ValueError, match="Architecture mismatch"):
            validate_checkpoint_config(ckpt_config, current_config)

    def test_policy_type_mismatch_raises(self):
        """Loading checkpoint with different policy_type should raise ValueError."""
        ckpt_config = {
            "policy": {
                "policy_type": "LSTMPolicy",
                "hidden_size": 32,
                "image_height": 64,
                "image_width": 64,
            },
            "training": {},
            "curriculum": {},
            "env": {},
        }
        current_config = {
            "policy": {
                "policy_type": "TinyPolicy",
                "hidden_size": 32,
                "image_height": 64,
                "image_width": 64,
            },
            "training": {},
            "curriculum": {},
            "env": {},
        }
        with pytest.raises(ValueError, match="Architecture mismatch"):
            validate_checkpoint_config(ckpt_config, current_config)

    def test_training_param_change_warns(self):
        """Changing training params should warn but not error."""
        ckpt_config = {
            "policy": {
                "policy_type": "LSTMPolicy",
                "hidden_size": 32,
                "image_height": 64,
                "image_width": 64,
            },
            "training": {"learning_rate": 0.001},
            "curriculum": {},
            "env": {},
        }
        current_config = {
            "policy": {
                "policy_type": "LSTMPolicy",
                "hidden_size": 32,
                "image_height": 64,
                "image_width": 64,
            },
            "training": {"learning_rate": 0.0001},
            "curriculum": {},
            "env": {},
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_checkpoint_config(ckpt_config, current_config)
            assert len(w) == 1
            assert "learning_rate" in str(w[0].message)

    def test_save_checkpoint_creates_file(self, tmp_path, monkeypatch):
        """save_checkpoint should create a local .pt file."""
        cfg = _smoketest_config()
        policy = LSTMPolicy(
            image_height=cfg.policy.image_height,
            image_width=cfg.policy.image_width,
            hidden_size=cfg.policy.hidden_size,
        )
        optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.training.learning_rate)

        wandb.init(mode="disabled")

        # Use tmp_path as working directory so checkpoints/ goes there
        monkeypatch.chdir(tmp_path)

        path = save_checkpoint(
            policy,
            optimizer,
            cfg,
            curriculum_stage=1,
            stage_progress=0.5,
            mastery_count=3,
            batch_idx=10,
            episode_count=100,
            trigger="periodic",
        )
        assert os.path.isfile(path)

        # Verify contents
        ckpt = torch.load(path, weights_only=False)
        assert "policy_state_dict" in ckpt
        assert "optimizer_state_dict" in ckpt
        assert ckpt["curriculum_stage"] == 1
        assert ckpt["batch_idx"] == 10

        wandb.finish()
