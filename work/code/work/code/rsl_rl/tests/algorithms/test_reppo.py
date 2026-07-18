# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Relative Entropy Pathwise Policy Optimization."""

from __future__ import annotations

import torch

from rsl_rl.algorithms import REPPO
from rsl_rl.models import ActionValueModel, LayerNormMLPModel
from rsl_rl.storage import RolloutStorage
from tests.conftest import make_obs


def _build_reppo(num_envs: int = 4, num_steps: int = 4) -> REPPO:
    """Build a small CPU REPPO instance."""
    obs = make_obs(num_envs, obs_dim=6)
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = LayerNormMLPModel(
        obs,
        obs_groups,
        "actor",
        2,
        hidden_dims=(32, 32),
        activation="swish",
        distribution_cfg={
            "class_name": "TanhGaussianDistribution",
            "init_std": 0.7,
            "std_type": "log",
        },
    )
    critic = ActionValueModel(
        obs,
        obs_groups,
        "critic",
        2,
        hidden_dims=(32, 32),
        activation="swish",
        num_bins=31,
        vmin=-5.0,
        vmax=5.0,
    )
    storage = RolloutStorage("rl", num_envs, num_steps, obs, [2])
    return REPPO(
        actor,
        critic,
        storage,
        num_learning_epochs=1,
        num_mini_batches=2,
        learning_rate=1e-3,
        desired_kl=0.1,
        optimizer="adam",
    )


def test_tanh_distribution_rsample_is_bounded_and_differentiable() -> None:
    """The REPPO actor distribution should provide bounded pathwise samples."""
    algorithm = _build_reppo()
    obs = make_obs(4, obs_dim=6)
    algorithm.actor(obs, stochastic_output=True)
    action = algorithm._raw_actor.distribution.rsample()
    assert torch.all(action > -1.0)
    assert torch.all(action < 1.0)

    loss = action.sum()
    algorithm.actor_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert any(parameter.grad is not None for parameter in algorithm.actor.parameters())


def test_hl_gauss_targets_are_normalized() -> None:
    """Categorical critic targets should form valid probability distributions."""
    algorithm = _build_reppo()
    targets = torch.tensor([-100.0, -1.0, 0.0, 2.0, 100.0])
    embedded = algorithm._raw_critic.embed_targets(targets)
    assert embedded.shape == (5, algorithm._raw_critic.num_bins)
    assert torch.all(embedded >= 0.0)
    assert torch.allclose(embedded.sum(dim=-1), torch.ones(5), atol=1e-5)


def test_rollout_and_update_changes_both_models() -> None:
    """A complete rollout should produce finite losses and update actor and critic."""
    torch.manual_seed(7)
    algorithm = _build_reppo()
    obs = make_obs(4, obs_dim=6)

    for _ in range(algorithm.storage.num_transitions_per_env):
        algorithm.act(obs)
        next_obs = make_obs(4, obs_dim=6)
        rewards = torch.randn(4)
        dones = torch.zeros(4, dtype=torch.long)
        algorithm.process_env_step(next_obs, rewards, dones, {"time_outs": torch.zeros(4, dtype=torch.bool)})
        obs = next_obs

    algorithm.compute_returns(obs)
    actor_before = [parameter.detach().clone() for parameter in algorithm._raw_actor.parameters()]
    critic_before = [parameter.detach().clone() for parameter in algorithm._raw_critic.parameters()]
    losses = algorithm.update()

    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    assert any(not torch.equal(before, after) for before, after in zip(actor_before, algorithm._raw_actor.parameters()))
    assert any(
        not torch.equal(before, after) for before, after in zip(critic_before, algorithm._raw_critic.parameters())
    )
    assert algorithm.storage.step == 0


def test_timeout_is_recorded_and_excluded_from_terminal_done() -> None:
    """Artificial timeouts should be stored separately from true terminations."""
    algorithm = _build_reppo()
    obs = make_obs(4, obs_dim=6)
    algorithm.act(obs)
    time_outs = torch.tensor([True, False, False, False])
    dones = torch.tensor([1, 0, 1, 0])
    algorithm.process_env_step(obs, torch.zeros(4), dones, {"time_outs": time_outs})

    assert algorithm.storage.truncations[0, 0, 0] == 1
    assert algorithm.storage.dones[0, 0, 0] == 0
    assert algorithm.storage.dones[0, 2, 0] == 1


def test_timeout_stops_td_lambda_recursion_across_auto_reset() -> None:
    """Rewards after an auto-reset must not leak into the preceding episode."""
    algorithm = _build_reppo()
    algorithm.gamma = 1.0
    algorithm.lam = 1.0
    storage = algorithm.storage
    storage.rewards.zero_()
    storage.values.zero_()
    storage.dones.zero_()
    storage.truncations.zero_()

    # Step 1 ends an episode; the large step-2 reward belongs to the reset episode.
    storage.truncations[1] = 1
    storage.rewards[2] = 100.0
    algorithm.compute_returns(make_obs(storage.num_envs, obs_dim=6))

    assert torch.allclose(storage.returns[0], torch.zeros_like(storage.returns[0]))
