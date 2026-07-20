# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay storage with timeout-safe, episode-aware n-step sampling."""

from __future__ import annotations

import torch
import warnings
from collections.abc import Generator
from dataclasses import dataclass
from tensordict import TensorDict


class ReplayBuffer:
    """Per-environment circular replay buffer.

    Invalid transitions occupy their chronological slot but are never sampled.
    They therefore act as hard boundaries for n-step returns and prevent a window
    from crossing an auto-reset into the next episode.
    """

    class Transition:
        """Container populated by an off-policy algorithm during collection."""

        def __init__(self) -> None:
            """Initialize an empty transition container."""
            self.observations: TensorDict | None = None
            self.actions: torch.Tensor | None = None
            self.rewards: torch.Tensor | None = None
            self.next_observations: TensorDict | None = None
            self.dones: torch.Tensor | None = None
            self.bootstrap: torch.Tensor | None = None
            self.valid: torch.Tensor | None = None

        def clear(self) -> None:
            """Reset all staged transition fields."""
            self.__init__()

    @dataclass
    class Batch:
        """A sampled replay mini-batch."""

        observations: TensorDict
        actions: torch.Tensor
        rewards: torch.Tensor
        next_observations: TensorDict
        bootstrap_mask: torch.Tensor
        effective_n_steps: torch.Tensor

    def __init__(
        self,
        num_envs: int,
        obs: TensorDict,
        actions_shape: tuple[int, ...] | list[int],
        device: str = "cpu",
        buffer_size: int = 1_000_000,
        n_steps: int = 1,
        gamma: float = 0.99,
    ) -> None:
        """Allocate replay tensors and configure n-step sampling."""
        if num_envs < 1:
            raise ValueError("num_envs must be positive.")
        if buffer_size < num_envs:
            warnings.warn(
                f"buffer_size={buffer_size} is smaller than num_envs={num_envs}; "
                "allocating one transition per environment.",
                RuntimeWarning,
            )
        if n_steps < 1:
            raise ValueError("n_steps must be at least one.")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must lie in [0, 1].")

        self.num_envs = num_envs
        self.device = device
        self.buffer_size = max(buffer_size // num_envs, 1)
        if n_steps > self.buffer_size:
            raise ValueError(f"n_steps ({n_steps}) cannot exceed per-environment replay capacity ({self.buffer_size}).")
        self.n_steps = n_steps
        self.gamma = gamma
        self.step = 0
        self.num_transitions = 0

        self.observations = TensorDict(
            {
                key: torch.zeros(
                    num_envs,
                    self.buffer_size,
                    *value.shape[1:],
                    dtype=value.dtype,
                    device=device,
                )
                for key, value in obs.items()
            },
            batch_size=[num_envs, self.buffer_size],
            device=device,
        )
        self.next_observations = torch.zeros_like(self.observations)
        self.actions = torch.zeros(num_envs, self.buffer_size, *actions_shape, device=device)
        self.rewards = torch.zeros(num_envs, self.buffer_size, 1, device=device)
        self.dones = torch.zeros(num_envs, self.buffer_size, 1, dtype=torch.bool, device=device)
        self.bootstrap = torch.zeros(num_envs, self.buffer_size, 1, dtype=torch.bool, device=device)
        self.valid = torch.zeros(num_envs, self.buffer_size, 1, dtype=torch.bool, device=device)

    def clear(self) -> None:
        """Reset the circular cursor and logical contents."""
        self.step = 0
        self.num_transitions = 0

    def add_transition(self, transition: Transition) -> None:
        """Insert one synchronized vector-environment step."""
        required = (
            transition.observations,
            transition.actions,
            transition.rewards,
            transition.next_observations,
            transition.dones,
            transition.bootstrap,
        )
        if any(value is None for value in required):
            raise ValueError("Replay transition is missing one or more required fields.")

        self.observations[:, self.step].copy_(transition.observations)  # type: ignore[arg-type]
        self.next_observations[:, self.step].copy_(transition.next_observations)  # type: ignore[arg-type]
        self.actions[:, self.step].copy_(transition.actions)  # type: ignore[arg-type]
        self.rewards[:, self.step].copy_(transition.rewards.view(-1, 1))  # type: ignore[union-attr]
        self.dones[:, self.step].copy_(transition.dones.view(-1, 1).bool())  # type: ignore[union-attr]
        self.bootstrap[:, self.step].copy_(transition.bootstrap.view(-1, 1).bool())  # type: ignore[union-attr]
        if transition.valid is None:
            self.valid[:, self.step].fill_(True)
        else:
            self.valid[:, self.step].copy_(transition.valid.view(-1, 1).bool())

        self.step = (self.step + 1) % self.buffer_size
        self.num_transitions = min(self.num_transitions + 1, self.buffer_size)

    def can_sample(self, minimum_transitions: int = 1) -> bool:
        """Return whether at least ``minimum_transitions`` valid entries exist."""
        if minimum_transitions < 1 or self.num_transitions == 0:
            return False
        chronological = self._chronological_indices()
        return int(self.valid[:, chronological].sum().item()) >= minimum_transitions

    @property
    def num_valid_transitions(self) -> int:
        """Number of currently sampleable transitions across all environments."""
        if self.num_transitions == 0:
            return 0
        return int(self.valid[:, self._chronological_indices()].sum().item())

    def mini_batch_generator(
        self,
        num_mini_batches: int,
        mini_batch_size: int,
        num_epochs: int = 1,
    ) -> Generator[Batch, None, None]:
        """Yield randomly sampled, timeout-safe n-step batches."""
        if num_mini_batches < 1 or mini_batch_size < 1 or num_epochs < 1:
            raise ValueError("Replay sampling counts must all be positive.")
        candidates = self._valid_start_indices()
        if candidates is None:
            raise ValueError("Replay buffer does not contain a valid transition.")
        for _ in range(num_epochs):
            for _ in range(num_mini_batches):
                yield self._sample_batch(candidates, mini_batch_size)

    def _chronological_indices(self) -> torch.Tensor:
        if self.num_transitions < self.buffer_size:
            return torch.arange(self.num_transitions, device=self.device)
        return torch.cat((
            torch.arange(self.step, self.buffer_size, device=self.device),
            torch.arange(0, self.step, device=self.device),
        ))

    def _valid_start_indices(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if self.num_transitions == 0:
            return None
        chronological = self._chronological_indices()
        valid = self.valid[:, chronological, 0]
        env_indices, positions = valid.nonzero(as_tuple=True)
        if env_indices.numel() == 0:
            return None
        return env_indices, positions, chronological

    def _sample_batch(
        self,
        candidates: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        mini_batch_size: int,
    ) -> Batch:
        env_candidates, position_candidates, chronological = candidates
        candidate_count = env_candidates.numel()
        sampled = torch.randint(candidate_count, (mini_batch_size,), device=self.device)
        envs = env_candidates[sampled]
        starts = position_candidates[sampled]
        start_physical = chronological[starts]

        rewards = torch.zeros(mini_batch_size, 1, device=self.device)
        effective_n_steps = torch.zeros(mini_batch_size, 1, dtype=torch.long, device=self.device)
        bootstrap_mask = torch.zeros(mini_batch_size, 1, dtype=torch.bool, device=self.device)
        last_physical = start_physical.clone()
        active = torch.ones(mini_batch_size, dtype=torch.bool, device=self.device)
        chronological_length = chronological.numel()

        for offset in range(self.n_steps):
            positions = starts + offset
            exists = positions < chronological_length
            safe_positions = positions.clamp_max(chronological_length - 1)
            physical = chronological[safe_positions]
            row_valid = exists & self.valid[envs, physical, 0]
            include = active & row_valid
            if not include.any():
                break

            rewards[include] += (self.gamma**offset) * self.rewards[envs[include], physical[include]]
            effective_n_steps[include] += 1
            last_physical[include] = physical[include]

            done = self.dones[envs, physical, 0]
            if offset + 1 < self.n_steps:
                next_positions = positions + 1
                next_exists = next_positions < chronological_length
                safe_next = next_positions.clamp_max(chronological_length - 1)
                next_physical = chronological[safe_next]
                next_valid = next_exists & self.valid[envs, next_physical, 0]
            else:
                next_valid = torch.zeros_like(include)

            reached_horizon = offset + 1 == self.n_steps
            stop = include & (done | ~next_valid | reached_horizon)
            terminal_bootstrap = self.bootstrap[envs, physical, 0]
            bootstrap_mask[stop, 0] = torch.where(done[stop], terminal_bootstrap[stop], True)
            active &= ~stop

        observations = TensorDict(
            {key: value[envs, start_physical] for key, value in self.observations.items()},
            batch_size=[mini_batch_size],
            device=self.device,
        )
        next_observations = TensorDict(
            {key: value[envs, last_physical] for key, value in self.next_observations.items()},
            batch_size=[mini_batch_size],
            device=self.device,
        )
        return ReplayBuffer.Batch(
            observations=observations,
            actions=self.actions[envs, start_physical],
            rewards=rewards,
            next_observations=next_observations,
            bootstrap_mask=bootstrap_mask.float(),
            effective_n_steps=effective_n_steps,
        )
