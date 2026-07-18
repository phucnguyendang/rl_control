# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import math
import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import EmpiricalNormalization, HiddenState
from rsl_rl.utils import resolve_nn_activation


class _LayerNormMLP(nn.Sequential):
    """MLP with LayerNorm after every hidden linear layer."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int | tuple[int, ...] | list[int],
        hidden_dims: tuple[int, ...] | list[int],
        activation: str,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer.")

        processed_hidden_dims = [input_dim if dim == -1 else dim for dim in hidden_dims]
        layers: list[nn.Module] = []
        in_dim = input_dim
        for hidden_dim in processed_hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                resolve_nn_activation(activation),
            ])
            in_dim = hidden_dim

        if isinstance(output_dim, int):
            layers.append(nn.Linear(in_dim, output_dim))
        else:
            output_shape = tuple(output_dim)
            layers.extend([nn.Linear(in_dim, math.prod(output_shape)), nn.Unflatten(-1, output_shape)])

        for index, layer in enumerate(layers):
            self.add_module(str(index), layer)


class LayerNormMLPModel(MLPModel):
    """MLPModel variant matching REPPO's normalized actor architecture."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "swish",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
    ) -> None:
        """Initialize a normalized actor model with an optional output distribution."""
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
        )

        mlp_output_dim = self.distribution.input_dim if self.distribution is not None else output_dim
        self.mlp = _LayerNormMLP(self._get_latent_dim(), mlp_output_dim, hidden_dims, activation)
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)


class ActionValueModel(nn.Module):
    """Action-conditioned categorical Q model used by REPPO.

    The network predicts HL-Gauss logits for Q(s, a). Scalar Q values are decoded
    as the expectation of the categorical distribution.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        action_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (512, 512, 512),
        activation: str = "swish",
        obs_normalization: bool = False,
        num_bins: int = 151,
        vmin: float = -10.0,
        vmax: float = 10.0,
        sigma_ratio: float = 0.75,
        zero_init_scale: float = 40.0,
    ) -> None:
        """Initialize the action-conditioned categorical critic."""
        super().__init__()
        if num_bins < 2:
            raise ValueError("num_bins must be at least 2.")
        if vmax <= vmin:
            raise ValueError(f"vmax ({vmax}) must be greater than vmin ({vmin}).")
        if sigma_ratio <= 0.0:
            raise ValueError(f"sigma_ratio must be positive, got {sigma_ratio}.")

        self.obs_groups = obs_groups[obs_set]
        self.obs_dim = 0
        for obs_group in self.obs_groups:
            if len(obs[obs_group].shape) != 2:
                raise ValueError(
                    f"ActionValueModel only supports 1D observations, got {obs[obs_group].shape} for '{obs_group}'."
                )
            self.obs_dim += obs[obs_group].shape[-1]

        self.action_dim = action_dim
        self.obs_normalization = obs_normalization
        self.obs_normalizer = EmpiricalNormalization(self.obs_dim) if obs_normalization else nn.Identity()
        self.network = _LayerNormMLP(self.obs_dim + action_dim, num_bins, hidden_dims, activation)

        self.num_bins = num_bins
        self.vmin = vmin
        self.vmax = vmax
        self.bin_width = (vmax - vmin) / (num_bins - 1)
        self.sigma = self.bin_width * sigma_ratio
        self.register_buffer(
            "bin_edges",
            torch.linspace(vmin - self.bin_width / 2, vmax + self.bin_width / 2, num_bins + 1),
        )
        self.register_buffer("support", torch.linspace(vmin, vmax, num_bins))

        zero_target = self.embed_targets(torch.zeros(1))
        self.zero_logits = nn.Parameter(zero_init_scale * zero_target)

    def forward(
        self,
        obs: TensorDict,
        actions: torch.Tensor,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        return_logits: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Evaluate Q(s, a), optionally returning the categorical logits."""
        del hidden_state
        if masks is not None:
            raise ValueError("ActionValueModel does not support recurrent trajectory masks.")

        critic_obs = self._get_obs(obs)
        critic_obs = self.obs_normalizer(critic_obs)
        if actions.shape[-1] != self.action_dim:
            raise ValueError(f"Expected action dimension {self.action_dim}, got {actions.shape[-1]}.")
        logits = self.network(torch.cat([critic_obs, actions], dim=-1)) + self.zero_logits
        values = (torch.softmax(logits, dim=-1) * self.support).sum(dim=-1, keepdim=True)
        if return_logits:
            return values, logits
        return values

    def embed_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """Embed scalar regression targets as HL-Gauss categorical targets."""
        if targets.ndim > 0 and targets.shape[-1] == 1:
            targets = targets.squeeze(-1)
        targets = targets.clamp(self.vmin, self.vmax)
        scaled = (self.bin_edges - targets.unsqueeze(-1)) / (self.sigma * math.sqrt(2.0))
        cdf = torch.erf(scaled)
        probabilities = cdf[..., 1:] - cdf[..., :-1]
        probabilities = probabilities.clamp_min(0.0)
        normalizer = (cdf[..., -1:] - cdf[..., :1]).clamp_min(torch.finfo(probabilities.dtype).eps)
        probabilities = probabilities / normalizer
        return probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(probabilities.dtype).eps)

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Return the per-sample HL-Gauss cross-entropy loss."""
        target_distribution = self.embed_targets(targets).detach()
        return -(target_distribution * torch.log_softmax(logits, dim=-1)).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        """Update observation-normalization statistics."""
        if self.obs_normalization:
            self.obs_normalizer.update(self._get_obs(obs))  # type: ignore

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        """Reset recurrent state (no-op for this feedforward model)."""
        del dones, hidden_state

    def get_hidden_state(self) -> HiddenState:
        """Return the recurrent hidden state (always None)."""
        return None

    def _get_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[obs_group] for obs_group in self.obs_groups], dim=-1)
