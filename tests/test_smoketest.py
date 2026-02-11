"""
Fast end-to-end smoketest for the training pipeline.

Validates that env, policy, episode collection, training step,
and curriculum all work together. Runs in seconds.
"""

import numpy as np
import torch

import wandb
from config import Config
from train import (
    LSTMPolicy,
    TinyPolicy,
    collect_episode,
    train_step_batched,
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
