# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for shared on-policy timeout-boundary semantics."""

import torch
from tensordict import TensorDict

from rsl_rl.algorithms.transition_handling import compute_gae_returns, compute_td_lambda_returns
from rsl_rl.storage import RolloutStorage


def _obs(value: float = 0.0) -> TensorDict:
    return TensorDict({"policy": torch.tensor([[value]])}, batch_size=[1])


def _transition(step: int, value: float, reward: float, done: bool, valid: bool = True) -> RolloutStorage.Transition:
    transition = RolloutStorage.Transition()
    transition.observations = _obs(float(step))
    transition.actions = torch.tensor([[float(step)]])
    transition.rewards = torch.tensor([reward])
    transition.dones = torch.tensor([done])
    transition.values = torch.tensor([[value]])
    transition.actions_log_prob = torch.zeros(1)
    transition.distribution_params = (torch.zeros(1, 1), torch.ones(1, 1))
    transition.valid = torch.tensor([valid])
    return transition


def _make_boundary_rollout() -> RolloutStorage:
    storage = RolloutStorage("rl", 1, 3, _obs(), [1])
    storage.add_transition(_transition(0, value=10.0, reward=1.0, done=False))
    marked = storage.mark_last_transition_as_bootstrap(torch.tensor([True]), torch.tensor([[20.0]]))
    assert marked.item()
    storage.add_transition(_transition(1, value=20.0, reward=100.0, done=True, valid=False))
    storage.add_transition(_transition(2, value=3.0, reward=2.0, done=False))
    return storage


def test_gae_bootstraps_before_invalid_timeout_and_excludes_it() -> None:
    """GAE should end before the invalid timeout slot and retain the prior bootstrap."""
    storage = _make_boundary_rollout()
    compute_gae_returns(storage, last_values=torch.tensor([[4.0]]), gamma=1.0, lam=0.95)

    assert torch.allclose(storage.returns[:, 0, 0], torch.tensor([21.0, 20.0, 6.0]))
    assert torch.allclose(storage.advantages[:, 0, 0], torch.tensor([11.0, 0.0, 3.0]))
    actions = torch.cat([batch.actions for batch in storage.mini_batch_generator(num_mini_batches=1, num_epochs=1)])
    assert sorted(actions[:, 0].tolist()) == [0.0, 2.0]


def test_td_lambda_uses_the_same_timeout_boundary() -> None:
    """REPPO-style TD-lambda should share PPO's timeout semantics."""
    storage = _make_boundary_rollout()
    compute_td_lambda_returns(storage, last_values=torch.tensor([[4.0]]), gamma=1.0, lam=0.5)
    assert torch.allclose(storage.returns[:, 0, 0], torch.tensor([21.0, 20.0, 6.0]))
