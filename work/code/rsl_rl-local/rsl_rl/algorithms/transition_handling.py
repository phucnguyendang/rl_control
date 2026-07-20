# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared transition-boundary handling for on-policy algorithms."""

from __future__ import annotations

import torch
from collections.abc import Callable
from tensordict import TensorDict

from rsl_rl.storage import RolloutStorage


def configure_timeout_transition(
    storage: RolloutStorage,
    transition: RolloutStorage.Transition,
    next_obs: TensorDict,
    dones: torch.Tensor,
    extras: dict,
    bootstrap_value_fn: Callable[[TensorDict], torch.Tensor],
    device: str,
) -> torch.Tensor:
    """Configure validity and bootstrap metadata for an auto-reset step.

    If the environment supplies ``time_outs_obs`` (or ``final_observation``), the
    timeout transition is retained and bootstraps from that observation. Otherwise
    the timeout-causing transition is marked invalid, while the preceding valid
    transition is turned into a boundary that bootstraps from the current value.
    This prevents learning from a reset observation without requiring changes to
    the simulator.
    """
    num_envs = dones.numel()
    transition.valid = torch.ones(num_envs, dtype=torch.bool, device=device)
    transition.bootstrap = torch.zeros(num_envs, dtype=torch.bool, device=device)
    transition.bootstrap_values = torch.zeros(num_envs, 1, device=device)

    raw_timeouts = extras.get("time_outs")
    if raw_timeouts is None:
        return transition.bootstrap

    timeouts = raw_timeouts.to(device).view(-1).bool() & dones.to(device).view(-1).bool()
    if not timeouts.any():
        return timeouts

    final_obs = extras.get("time_outs_obs", extras.get("final_observation"))
    if final_obs is not None:
        try:
            final_obs = _as_tensordict(final_obs, next_obs).to(device)
            with torch.no_grad():
                bootstrap_values = bootstrap_value_fn(final_obs).detach().view(-1, 1)
        except (KeyError, TypeError, ValueError):
            final_obs = None

    if final_obs is not None:
        transition.bootstrap = timeouts
        transition.bootstrap_values[timeouts] = bootstrap_values[timeouts]
    else:
        # V(s_t) is already present in transition.values. It is the correct next
        # value for the preceding transition, even though s_{t+1}^{final} is lost.
        storage.mark_last_transition_as_bootstrap(timeouts, transition.values)  # type: ignore[arg-type]
        transition.valid = ~timeouts

    return timeouts


def compute_gae_returns(
    storage: RolloutStorage,
    last_values: torch.Tensor,
    gamma: float,
    lam: float,
) -> None:
    """Compute GAE with explicit bootstrap boundaries and invalid-transition gaps."""
    advantage = torch.zeros_like(last_values)
    for step in reversed(range(storage.num_transitions_per_env)):
        explicit_bootstrap = storage.bootstrap[step]
        valid = storage.valid[step]
        done = storage.dones[step].bool()
        terminal = done & ~explicit_bootstrap
        default_next_values = last_values if step == storage.num_transitions_per_env - 1 else storage.values[step + 1]
        next_values = torch.where(
            explicit_bootstrap,
            storage.bootstrap_values[step],
            default_next_values,
        )
        delta = storage.rewards[step] + gamma * (~terminal).float() * next_values - storage.values[step]
        continuation = (~done & ~explicit_bootstrap).float()
        candidate = delta + gamma * lam * continuation * advantage
        advantage = torch.where(valid, candidate, torch.zeros_like(candidate))
        storage.advantages[step] = advantage
        storage.returns[step] = torch.where(valid, advantage + storage.values[step], storage.values[step])


def compute_td_lambda_returns(
    storage: RolloutStorage,
    last_values: torch.Tensor,
    gamma: float,
    lam: float,
) -> None:
    """Compute action-value TD-lambda targets with the shared boundary semantics."""
    recursive_value = last_values
    for step in reversed(range(storage.num_transitions_per_env)):
        explicit_bootstrap = storage.bootstrap[step]
        valid = storage.valid[step]
        done = storage.dones[step].bool()
        default_next_values = last_values if step == storage.num_transitions_per_env - 1 else storage.values[step + 1]
        next_values = torch.where(
            explicit_bootstrap,
            storage.bootstrap_values[step],
            default_next_values,
        )

        terminal_target = storage.rewards[step]
        boundary_target = storage.rewards[step] + gamma * storage.bootstrap_values[step]
        continuing_target = storage.rewards[step] + gamma * ((1.0 - lam) * next_values + lam * recursive_value)
        candidate = torch.where(
            explicit_bootstrap,
            boundary_target,
            torch.where(done, terminal_target, continuing_target),
        )
        recursive_value = torch.where(valid, candidate, torch.zeros_like(candidate))
        storage.returns[step] = torch.where(valid, recursive_value, storage.values[step])


def masked_mean(tensor: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
    """Return a mean that excludes invalid rollout entries."""
    if valid is None:
        return tensor.mean()
    mask = valid.bool()
    while mask.ndim > tensor.ndim and mask.shape[-1] == 1:
        mask = mask.squeeze(-1)
    while mask.ndim < tensor.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(tensor)
    if not mask.any():
        raise ValueError("Mini-batch does not contain any valid transitions.")
    return tensor[mask].mean()


def normalize_valid(tensor: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
    """Normalize a tensor over valid entries and zero invalid entries."""
    if valid is None:
        return (tensor - tensor.mean()) / (tensor.std(unbiased=False) + 1e-8)
    mask = valid.bool()
    while mask.ndim < tensor.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(tensor)
    values = tensor[mask]
    normalized = torch.zeros_like(tensor)
    normalized[mask] = (values - values.mean()) / (values.std(unbiased=False) + 1e-8)
    return normalized


def _as_tensordict(observations: TensorDict | dict, template: TensorDict) -> TensorDict:
    if isinstance(observations, TensorDict):
        return observations
    if isinstance(observations, dict):
        return TensorDict(observations, batch_size=template.batch_size, device=template.device)
    raise TypeError(f"Expected timeout observations to be TensorDict or dict, got {type(observations)}.")
