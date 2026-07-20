# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for timeout-safe replay sampling."""

import torch
from tensordict import TensorDict

from rsl_rl.storage import ReplayBuffer


def _obs(value: float, num_envs: int = 1) -> TensorDict:
    return TensorDict(
        {"policy": torch.full((num_envs, 1), value)},
        batch_size=[num_envs],
    )


def _add(
    replay: ReplayBuffer,
    state: float,
    reward: float,
    *,
    done: bool = False,
    bootstrap: bool = False,
    valid: bool = True,
) -> None:
    transition = ReplayBuffer.Transition()
    transition.observations = _obs(state)
    transition.actions = torch.tensor([[state]])
    transition.rewards = torch.tensor([reward])
    transition.next_observations = _obs(state + 1.0)
    transition.dones = torch.tensor([done])
    transition.bootstrap = torch.tensor([bootstrap])
    transition.valid = torch.tensor([valid])
    replay.add_transition(transition)


def test_invalid_timeout_is_not_a_start_and_stops_n_step_window() -> None:
    """Invalid timeout slots must be neither samples nor bridges between episodes."""
    replay = ReplayBuffer(1, _obs(0.0), [1], buffer_size=8, n_steps=3, gamma=1.0)
    _add(replay, 0.0, 1.0)
    _add(replay, 1.0, 100.0, done=True, valid=False)
    _add(replay, 2.0, 10.0)

    envs, positions, chronological = replay._valid_start_indices()  # type: ignore[misc]
    assert positions.tolist() == [0, 2]

    # Force sampling from the transition immediately before the invalid timeout.
    batch = replay._sample_batch((envs[:1], positions[:1], chronological), 1)
    assert torch.isclose(batch.rewards, torch.tensor([[1.0]])).all()
    assert torch.isclose(batch.next_observations["policy"], torch.tensor([[1.0]])).all()
    assert batch.effective_n_steps.item() == 1
    assert torch.isclose(batch.bootstrap_mask, torch.tensor([[1.0]])).all()


def test_terminal_and_timeout_have_different_bootstrap_masks() -> None:
    """True terminals suppress bootstrapping while observed timeouts retain it."""
    terminal = ReplayBuffer(1, _obs(0.0), [1], buffer_size=4, n_steps=1)
    _add(terminal, 0.0, 1.0, done=True, bootstrap=False)
    candidates = terminal._valid_start_indices()
    terminal_batch = terminal._sample_batch(candidates, 1)  # type: ignore[arg-type]
    assert torch.isclose(terminal_batch.bootstrap_mask, torch.tensor([[0.0]])).all()

    timeout = ReplayBuffer(1, _obs(0.0), [1], buffer_size=4, n_steps=1)
    _add(timeout, 0.0, 1.0, done=True, bootstrap=True)
    candidates = timeout._valid_start_indices()
    timeout_batch = timeout._sample_batch(candidates, 1)  # type: ignore[arg-type]
    assert torch.isclose(timeout_batch.bootstrap_mask, torch.tensor([[1.0]])).all()


def test_n_step_return_never_crosses_a_true_terminal() -> None:
    """Rewards from a reset episode must not enter a pre-terminal n-step target."""
    replay = ReplayBuffer(1, _obs(0.0), [1], buffer_size=8, n_steps=3, gamma=0.5)
    _add(replay, 0.0, 2.0)
    _add(replay, 1.0, 4.0, done=True)
    _add(replay, 2.0, 100.0)
    envs, positions, chronological = replay._valid_start_indices()  # type: ignore[misc]
    batch = replay._sample_batch((envs[:1], positions[:1], chronological), 1)
    assert torch.isclose(batch.rewards, torch.tensor([[4.0]])).all()
    assert batch.effective_n_steps.item() == 2
    assert torch.isclose(batch.bootstrap_mask, torch.tensor([[0.0]])).all()
