# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SAC actor and twin-Q critic models for the RSL-RL 5.x model API."""

from __future__ import annotations

import copy
import math
import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.models.reppo_models import _LayerNormMLP
from rsl_rl.modules import MLP, HiddenState


class SACActorModel(MLPModel):
    """Tanh-Gaussian actor with reparameterized samples and bounded actions."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        init_noise_std: float = 1.0,
        layer_norm: bool = False,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        action_low: float | list[float] | torch.Tensor = -1.0,
        action_high: float | list[float] | torch.Tensor = 1.0,
        distribution_cfg: dict | None = None,
    ) -> None:
        """Initialize the bounded actor and its state-dependent distribution."""
        del distribution_cfg  # SAC always requires its own tanh-Gaussian distribution.
        if log_std_max <= log_std_min:
            raise ValueError("log_std_max must be greater than log_std_min.")
        if init_noise_std <= 0.0:
            raise ValueError("init_noise_std must be positive.")
        distribution_cfg = {
            "class_name": "TanhGaussianDistribution",
            "init_std": init_noise_std,
            "std_range": (math.exp(log_std_min), math.exp(log_std_max)),
            "std_type": "log",
            "action_range": (action_low, action_high),
        }
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
        self.output_dim = output_dim

        if layer_norm:
            self.mlp = _LayerNormMLP(
                self._get_latent_dim(),
                self.distribution.input_dim,  # type: ignore[union-attr]
                hidden_dims,
                activation,
            )
            self.distribution.init_mlp_weights(self.mlp)  # type: ignore[union-attr]

        # Small mean-head initialization avoids starting in tanh saturation.
        last_linear = next((module for module in reversed(self.mlp) if isinstance(module, nn.Linear)), None)
        if last_linear is not None:
            nn.init.normal_(last_linear.weight[:output_dim], mean=0.0, std=1e-3)
            nn.init.zeros_(last_linear.bias[:output_dim])
            nn.init.zeros_(last_linear.weight[output_dim:])
            nn.init.constant_(last_linear.bias[output_dim:], math.log(init_noise_std))

    def sample_action_logp(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw a pathwise action sample and its transformed log probability."""
        latent = self.get_latent(obs)
        self.distribution.update(self.mlp(latent))  # type: ignore[union-attr]
        actions = self.distribution.rsample()  # type: ignore[union-attr]
        log_prob = self.distribution.log_prob(actions).unsqueeze(-1)  # type: ignore[union-attr]
        return actions, log_prob


class SACCriticModel(MLPModel):
    """Twin action-value networks with frozen Polyak target copies."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        num_actions: int = 0,
        layer_norm: bool = False,
        distribution_cfg: dict | None = None,
    ) -> None:
        """Initialize online and target twin-Q networks."""
        del distribution_cfg
        if num_actions < 1:
            raise ValueError("num_actions must be positive for an SAC critic.")
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=None,
        )
        self.num_actions = num_actions
        q_input_dim = self.obs_dim + num_actions
        network_class = _LayerNormMLP if layer_norm else MLP
        self.critic1 = network_class(q_input_dim, output_dim, hidden_dims, activation)
        self.critic2 = network_class(q_input_dim, output_dim, hidden_dims, activation)
        self.critic1_target = copy.deepcopy(self.critic1).requires_grad_(False)
        self.critic2_target = copy.deepcopy(self.critic2).requires_grad_(False)
        del self.mlp

    def forward(
        self,
        obs: TensorDict,
        actions: torch.Tensor,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Evaluate the first online Q network."""
        del hidden_state, stochastic_output
        if masks is not None:
            raise ValueError("Recurrent SAC critics are not supported.")
        latent = torch.cat((self.get_latent(obs), actions), dim=-1)
        return self.critic1(latent)

    def evaluate_all_q(self, obs: TensorDict, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate both online Q networks."""
        latent = torch.cat((self.get_latent(obs), actions), dim=-1)
        return self.critic1(latent), self.critic2(latent)

    def evaluate_all_target_q(self, obs: TensorDict, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate both target Q networks."""
        latent = torch.cat((self.get_latent(obs), actions), dim=-1)
        return self.critic1_target(latent), self.critic2_target(latent)

    def init_target_networks(self) -> None:
        """Synchronize targets with online networks."""
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

    @torch.no_grad()
    def soft_update_target_networks(self, tau: float) -> None:
        """Apply a Polyak update to both target networks."""
        for target, online in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target.lerp_(online, tau)
        for target, online in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target.lerp_(online, tau)
