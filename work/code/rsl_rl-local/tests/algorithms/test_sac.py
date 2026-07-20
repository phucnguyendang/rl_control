# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SAC model, timeout, and update smoke tests."""

import math
import torch
from tensordict import TensorDict

from rsl_rl.algorithms import SAC
from rsl_rl.models import SACActorModel, SACCriticModel
from rsl_rl.storage import ReplayBuffer


def _obs(batch_size: int = 4, value: float = 0.0) -> TensorDict:
    return TensorDict(
        {"policy": torch.full((batch_size, 3), value)},
        batch_size=[batch_size],
    )


def _models(batch_size: int = 4) -> tuple[SACActorModel, SACCriticModel]:
    obs = _obs(batch_size)
    groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = SACActorModel(
        obs,
        groups,
        "actor",
        2,
        hidden_dims=(16, 16),
        action_low=[-2.0, -0.5],
        action_high=[1.0, 3.0],
    )
    critic = SACCriticModel(
        obs,
        groups,
        "critic",
        1,
        hidden_dims=(16, 16),
        num_actions=2,
    )
    return actor, critic


def test_actor_respects_vector_action_bounds_and_has_finite_log_prob() -> None:
    """The actor must obey each action dimension's bounds and remain differentiable."""
    actor, _ = _models()
    actions, log_prob = actor.sample_action_logp(_obs())
    assert torch.all(actions >= torch.tensor([-2.0, -0.5]))
    assert torch.all(actions <= torch.tensor([1.0, 3.0]))
    assert torch.isfinite(log_prob).all()
    (actions.mean() + log_prob.mean()).backward()
    assert any(parameter.grad is not None for parameter in actor.parameters())


def test_missing_timeout_observation_is_marked_invalid() -> None:
    """A reset observation must not become the next state of a timeout tuple."""
    actor, critic = _models(batch_size=2)
    replay = ReplayBuffer(2, _obs(2), [2], buffer_size=16)
    sac = SAC(actor, critic, replay, mini_batch_size=2)
    obs = _obs(2)
    sac.act(obs)
    sac.process_env_step(
        next_obs=_obs(2, value=5.0),
        rewards=torch.ones(2),
        dones=torch.tensor([True, False]),
        extras={"time_outs": torch.tensor([True, False])},
    )
    assert replay.valid[:, 0, 0].tolist() == [False, True]
    assert replay.num_valid_transitions == 1


def test_sac_update_is_finite_with_n_step_replay() -> None:
    """A complete twin-Q, actor, and alpha update should produce finite losses."""
    actor, critic = _models(batch_size=2)
    replay = ReplayBuffer(2, _obs(2), [2], buffer_size=32, n_steps=2, gamma=0.99)
    for step in range(4):
        transition = ReplayBuffer.Transition()
        transition.observations = _obs(2, float(step))
        transition.actions = torch.tanh(torch.randn(2, 2))
        transition.rewards = torch.full((2,), 0.1 * step)
        transition.next_observations = _obs(2, float(step + 1))
        transition.dones = torch.zeros(2, dtype=torch.bool)
        transition.bootstrap = torch.zeros(2, dtype=torch.bool)
        transition.valid = torch.ones(2, dtype=torch.bool)
        replay.add_transition(transition)

    sac = SAC(
        actor,
        critic,
        replay,
        num_learning_epochs=1,
        num_mini_batches=2,
        mini_batch_size=4,
        policy_frequency=1,
    )
    losses = sac.update()
    assert set(losses) == {"critic1", "critic2", "actor", "alpha"}
    assert all(math.isfinite(value) for value in losses.values())
