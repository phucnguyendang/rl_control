# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
import torch
import torch.nn as nn
from collections.abc import Iterable
from itertools import chain
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.models import ActionValueModel, MLPModel
from rsl_rl.modules import Distribution
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import compile_model, resolve_callable, resolve_obs_groups, resolve_optimizer


class REPPO:
    """Relative Entropy Pathwise Policy Optimization.

    This implementation ports the robotics-oriented REPPO implementation to the
    model-based API introduced in RSL-RL 5.x. It uses an action-conditioned
    categorical critic, reparameterized actor samples, entropy-temperature tuning,
    and a forward-KL constraint against a frozen behavior actor.
    """

    actor: MLPModel
    critic: ActionValueModel

    def __init__(
        self,
        actor: MLPModel,
        critic: ActionValueModel,
        storage: RolloutStorage,
        num_learning_epochs: int = 8,
        num_mini_batches: int = 64,
        gamma: float = 0.99,
        lam: float = 0.95,
        learning_rate: float = 3e-4,
        max_grad_norm: float = 1.0,
        desired_kl: float = 0.01,
        target_entropy: float = -0.5,
        init_alpha_temp: float = 0.1,
        init_alpha_kl: float = 0.1,
        num_kl_samples: int = 4,
        optimizer: str = "adamw",
        weight_decay: float = 1e-3,
        adam_betas: tuple[float, float] = (0.9, 0.95),
        device: str = "cpu",
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        """Initialize REPPO models, dual variables, storage, and optimizers."""
        if actor.is_recurrent or critic.is_recurrent:
            raise NotImplementedError("This RSL-RL 5.x REPPO port currently supports feedforward actors and critics.")
        if actor.distribution is None or type(actor.distribution).rsample is Distribution.rsample:
            raise ValueError("REPPO requires an actor distribution with reparameterized sampling (rsample).")
        if rnd_cfg is not None:
            raise NotImplementedError("RND is not part of this REPPO port. Set rnd_cfg=None.")
        if symmetry_cfg is not None:
            raise NotImplementedError("Symmetry augmentation is not part of this REPPO port. Set symmetry_cfg=None.")
        if desired_kl <= 0.0:
            raise ValueError("desired_kl must be positive.")
        if init_alpha_temp <= 0.0 or init_alpha_kl <= 0.0:
            raise ValueError("The initial entropy and KL multipliers must be positive.")
        if num_kl_samples < 1:
            raise ValueError("num_kl_samples must be at least 1.")
        if num_learning_epochs < 1 or num_mini_batches < 1:
            raise ValueError("num_learning_epochs and num_mini_batches must both be positive.")
        if not 0.0 <= gamma <= 1.0 or not 0.0 <= lam <= 1.0:
            raise ValueError("gamma and lam must both lie in [0, 1].")
        if max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive.")
        rollout_size = storage.num_envs * storage.num_transitions_per_env
        if rollout_size % num_mini_batches != 0:
            raise ValueError(
                f"Rollout size ({rollout_size}) must be divisible by num_mini_batches ({num_mini_batches})."
            )

        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        if multi_gpu_cfg is None:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1
        else:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]

        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self._raw_actor = self.actor
        self._raw_critic = self.critic
        self._old_actor = copy.deepcopy(self._raw_actor).to(device)
        self._old_actor.eval()
        self._old_actor.requires_grad_(False)

        self.log_alpha_temp = nn.Parameter(torch.log(torch.tensor(init_alpha_temp, device=device)))
        self.log_alpha_kl = nn.Parameter(torch.log(torch.tensor(init_alpha_kl, device=device)))

        optimizer_class = resolve_optimizer(optimizer)
        optimizer_kwargs: dict = {"lr": learning_rate}
        if optimizer.lower() in ("adam", "adamw"):
            optimizer_kwargs["betas"] = adam_betas
        if optimizer.lower() == "adamw":
            optimizer_kwargs["weight_decay"] = weight_decay

        actor_parameters = chain(
            self.actor.parameters(),
            (self.log_alpha_temp, self.log_alpha_kl),
        )
        self.actor_optimizer = optimizer_class(actor_parameters, **optimizer_kwargs)
        self.critic_optimizer = optimizer_class(self.critic.parameters(), **optimizer_kwargs)

        self.storage = storage
        self.transition = RolloutStorage.Transition()
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.gamma = gamma
        self.lam = lam
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.desired_kl = desired_kl
        self.target_entropy = target_entropy * actor.distribution.output_dim
        self.num_kl_samples = num_kl_samples

        # Keep these attributes for the generic RSL-RL runner interface.
        self.rnd = None
        self.symmetry = None
        self.intrinsic_rewards = None

    @property
    def alpha_temp(self) -> torch.Tensor:
        """Positive entropy-temperature multiplier."""
        return self.log_alpha_temp.exp()

    @property
    def alpha_kl(self) -> torch.Tensor:
        """Positive KL-constraint multiplier."""
        return self.log_alpha_kl.exp()

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample rollout actions and record action-conditioned values."""
        self.transition.hidden_states = (self.actor.get_hidden_state(), self.critic.get_hidden_state())
        actions = self.actor(obs, stochastic_output=True)
        self.transition.actions = actions.detach()
        self.transition.values = self.critic(obs, self.transition.actions).detach()
        self.transition.actions_log_prob = self._raw_actor.get_output_log_prob(self.transition.actions).detach()
        self.transition.distribution_params = tuple(
            parameter.detach() for parameter in self._raw_actor.output_distribution_params
        )
        self.transition.observations = obs
        return self.transition.actions

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> None:
        """Store a rollout step with the maximum-entropy reward correction."""
        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs)

        time_outs = extras.get("time_outs", torch.zeros_like(dones, dtype=torch.bool)).to(self.device).bool()
        self.transition.dones = dones.bool() & ~time_outs
        self.transition.truncations = time_outs

        augmented_rewards = rewards.clone()
        augmented_rewards += self.gamma * self.transition.values.squeeze(-1) * time_outs.float()  # type: ignore

        # Match the cvoelcker/rsl_rl port: use the rollout action's log-probability
        # in the soft reward and keep the multiplier fixed for the collected batch.
        augmented_rewards -= (
            self.gamma * self.alpha_temp.detach() * self.transition.actions_log_prob  # type: ignore
        )
        self.transition.rewards = augmented_rewards

        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        self.critic.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        """Compute generalized on-policy Q targets with a backward TD-lambda pass."""
        last_actions = self.actor(obs, stochastic_output=True).detach()
        last_values = self.critic(obs, last_actions).detach()
        recursive_value = last_values

        for step in reversed(range(self.storage.num_transitions_per_env)):
            next_values = (
                last_values if step == self.storage.num_transitions_per_env - 1 else self.storage.values[step + 1]
            )
            # A timeout is bootstrapped in process_env_step, but it must still stop
            # the backwards recursion. Otherwise an auto-reset observation from the
            # next episode leaks into the preceding episode's TD-lambda target.
            episode_boundary = self.storage.dones[step].bool() | self.storage.truncations[step].bool()
            next_is_not_terminal = 1.0 - episode_boundary.float()
            one_step_bootstrap = next_is_not_terminal * self.gamma * next_values
            multi_step_bootstrap = next_is_not_terminal * self.gamma * recursive_value
            recursive_value = (
                self.storage.rewards[step] + (1.0 - self.lam) * one_step_bootstrap + self.lam * multi_step_bootstrap
            )
            self.storage.returns[step] = recursive_value

    def update(self) -> dict[str, float]:
        """Fit the categorical critic first, then optimize the constrained actor."""
        self._old_actor.load_state_dict(self._raw_actor.state_dict())
        self._old_actor.eval()

        critic_totals = {
            "value": 0.0,
            "value_prediction_error": 0.0,
            "target_saturation": 0.0,
            "critic_grad_norm": 0.0,
        }
        actor_totals = {
            "surrogate": 0.0,
            "q_value": 0.0,
            "entropy": 0.0,
            "kl": 0.0,
            "actor_grad_norm": 0.0,
        }

        for batch in self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs):
            metrics = self._update_critic(batch)
            for key in critic_totals:
                critic_totals[key] += metrics[key]

        for batch in self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs):
            metrics = self._update_actor(batch)
            for key in actor_totals:
                actor_totals[key] += metrics[key]

        num_updates = self.num_learning_epochs * self.num_mini_batches
        losses = {key: value / num_updates for key, value in critic_totals.items()}
        losses.update({key: value / num_updates for key, value in actor_totals.items()})
        losses["alpha_temp"] = self.alpha_temp.detach().item()
        losses["alpha_kl"] = self.alpha_kl.detach().item()

        self.storage.clear()
        return losses

    def _update_critic(self, batch: RolloutStorage.Batch) -> dict[str, float]:
        """Perform one HL-Gauss critic update."""
        values, logits = self.critic(batch.observations, batch.actions, return_logits=True)  # type: ignore
        per_sample_loss = self._raw_critic.compute_loss(logits, batch.returns)  # type: ignore
        if batch.truncations is None:
            valid_mask = torch.ones_like(per_sample_loss)
        else:
            valid_mask = 1.0 - batch.truncations.float().squeeze(-1)
        value_loss = (valid_mask * per_sample_loss).mean()

        self.critic_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        if self.is_multi_gpu:
            self._reduce_gradients(self.critic.parameters())
        critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        prediction_error = (values.detach() - batch.returns).abs().mean()  # type: ignore
        target_saturation = (
            (
                (batch.returns <= self._raw_critic.vmin) | (batch.returns >= self._raw_critic.vmax)  # type: ignore
            )
            .float()
            .mean()
        )
        return {
            "value": value_loss.detach().item(),
            "value_prediction_error": prediction_error.item(),
            "target_saturation": target_saturation.item(),
            "critic_grad_norm": torch.as_tensor(critic_grad_norm).detach().item(),
        }

    def _update_actor(self, batch: RolloutStorage.Batch) -> dict[str, float]:
        """Perform one pathwise actor and dual-variable update."""
        critic_grad_state = [parameter.requires_grad for parameter in self.critic.parameters()]
        for parameter in self.critic.parameters():
            parameter.requires_grad_(False)

        try:
            # The first sample updates the stateful distribution object. As in the
            # reference RSL-RL port, it is discarded and replaced with an rsample.
            self.actor(batch.observations, stochastic_output=True)  # type: ignore
            distribution = self._raw_actor.distribution
            actions = distribution.rsample()  # type: ignore
            entropy = -distribution.log_prob(actions)  # type: ignore
            q_values = self.critic(batch.observations, actions).squeeze(-1)  # type: ignore
            primary_actor_loss = -(q_values + self.alpha_temp.detach() * entropy)

            with torch.no_grad():
                self._old_actor(batch.observations, stochastic_output=True)  # type: ignore
                old_distribution = self._old_actor.distribution
                old_actions = old_distribution.sample(torch.Size([self.num_kl_samples]))  # type: ignore
                old_log_prob = old_distribution.log_prob(old_actions)  # type: ignore
            new_log_prob = distribution.log_prob(old_actions)  # type: ignore
            kl = (old_log_prob - new_log_prob).mean(dim=0)

            constrained_actor_loss = torch.where(
                (kl < self.desired_kl).detach(),
                primary_actor_loss,
                self.alpha_kl.detach() * kl,
            ).mean()
            temperature_loss = self.alpha_temp * (entropy.mean() - self.target_entropy).detach()
            kl_multiplier_loss = self.alpha_kl * (self.desired_kl - kl.mean()).detach()
            actor_loss = constrained_actor_loss + temperature_loss + kl_multiplier_loss

            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_parameters = [*self.actor.parameters(), self.log_alpha_temp, self.log_alpha_kl]
            if self.is_multi_gpu:
                self._reduce_gradients(actor_parameters)
            actor_grad_norm = nn.utils.clip_grad_norm_(actor_parameters, self.max_grad_norm)
            self.actor_optimizer.step()
        finally:
            for parameter, requires_grad in zip(self.critic.parameters(), critic_grad_state):
                parameter.requires_grad_(requires_grad)

        return {
            "surrogate": actor_loss.detach().item(),
            "q_value": q_values.detach().mean().item(),
            "entropy": entropy.detach().mean().item(),
            "kl": kl.detach().mean().item(),
            "actor_grad_norm": torch.as_tensor(actor_grad_norm).detach().item(),
        }

    def train_mode(self) -> None:
        """Set train mode for learnable models."""
        self.actor.train()
        self.critic.train()
        self._old_actor.eval()

    def eval_mode(self) -> None:
        """Set evaluation mode for learnable models."""
        self.actor.eval()
        self.critic.eval()
        self._old_actor.eval()

    def save(self) -> dict:
        """Return model, optimizer, and dual-variable states for checkpointing."""
        return {
            "actor_state_dict": self._raw_actor.state_dict(),
            "critic_state_dict": self._raw_critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "log_alpha_temp": self.log_alpha_temp.detach(),
            "log_alpha_kl": self.log_alpha_kl.detach(),
        }

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load selected REPPO checkpoint components."""
        if load_cfg is None:
            load_cfg = {
                "actor": True,
                "critic": True,
                "optimizer": True,
                "dual_variables": True,
                "iteration": True,
            }
        if load_cfg.get("actor"):
            self._raw_actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        if load_cfg.get("critic"):
            self._raw_critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
        if load_cfg.get("optimizer"):
            self.actor_optimizer.load_state_dict(loaded_dict["actor_optimizer_state_dict"])
            self.critic_optimizer.load_state_dict(loaded_dict["critic_optimizer_state_dict"])
        if load_cfg.get("dual_variables"):
            self.log_alpha_temp.data.copy_(loaded_dict["log_alpha_temp"].to(self.device))
            self.log_alpha_kl.data.copy_(loaded_dict["log_alpha_kl"].to(self.device))
        return load_cfg.get("iteration", False)

    def get_policy(self) -> MLPModel:
        """Return the raw actor used for inference and export."""
        return self._raw_actor

    def compile(self, mode: str | None = None) -> None:
        """Compile the current actor and critic when requested."""
        self.actor = compile_model(self._raw_actor, mode)  # type: ignore
        self.critic = compile_model(self._raw_critic, mode)  # type: ignore

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> REPPO:
        """Construct REPPO from an RSL-RL 5.x runner configuration."""
        alg_class: type[REPPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[ActionValueModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])
        actor = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        critic = critic_class(obs, cfg["obs_groups"], "critic", env.num_actions, **cfg["critic"]).to(device)
        print(f"Actor Model: {actor}")
        print(f"Action-Value Model: {critic}")

        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)
        algorithm = alg_class(
            actor,
            critic,
            storage,
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )
        algorithm.compile(cfg.get("torch_compile_mode"))
        return algorithm

    def broadcast_parameters(self) -> None:
        """Broadcast model and dual-variable parameters from rank zero."""
        model_states = [self._raw_actor.state_dict(), self._raw_critic.state_dict()]
        torch.distributed.broadcast_object_list(model_states, src=0)
        self._raw_actor.load_state_dict(model_states[0])
        self._raw_critic.load_state_dict(model_states[1])

        dual_variables = torch.stack([self.log_alpha_temp.detach(), self.log_alpha_kl.detach()])
        torch.distributed.broadcast(dual_variables, src=0)
        self.log_alpha_temp.data.copy_(dual_variables[0])
        self.log_alpha_kl.data.copy_(dual_variables[1])

    def _reduce_gradients(self, parameters: Iterable[nn.Parameter]) -> None:
        """Average a selected set of gradients across all distributed workers."""
        parameter_list = list(parameters)
        gradients = [parameter.grad.view(-1) for parameter in parameter_list if parameter.grad is not None]
        if not gradients:
            return
        flattened_gradients = torch.cat(gradients)
        torch.distributed.all_reduce(flattened_gradients, op=torch.distributed.ReduceOp.SUM)
        flattened_gradients /= self.gpu_world_size

        offset = 0
        for parameter in parameter_list:
            if parameter.grad is None:
                continue
            numel = parameter.numel()
            parameter.grad.data.copy_(flattened_gradients[offset : offset + numel].view_as(parameter.grad))
            offset += numel
