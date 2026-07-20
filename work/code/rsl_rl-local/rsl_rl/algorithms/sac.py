# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Soft Actor-Critic for the RSL-RL 5.x model API."""

from __future__ import annotations

import torch
import torch.nn as nn
from collections.abc import Iterable
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.extensions import RandomNetworkDistillation, Symmetry, resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import SACActorModel, SACCriticModel
from rsl_rl.storage import ReplayBuffer
from rsl_rl.utils import resolve_callable, resolve_obs_groups, resolve_optimizer


class SAC:
    """Entropy-regularized off-policy actor-critic with twin target Q networks."""

    actor: SACActorModel
    critic: SACCriticModel

    def __init__(
        self,
        actor: SACActorModel,
        critic: SACCriticModel,
        replay_buffer: ReplayBuffer,
        replay_buffer_size: int = 1_000_000,
        num_learning_epochs: int = 1,
        num_mini_batches: int = 1,
        mini_batch_size: int = 256,
        actor_learning_rate: float = 1e-3,
        critic_learning_rate: float = 1e-3,
        alpha_learning_rate: float = 1e-3,
        actor_optimizer: str = "adam",
        critic_optimizer: str = "adam",
        auto_alpha: bool = True,
        alpha: float = 0.05,
        tau: float = 0.005,
        gamma: float = 0.998,
        target_entropy_scale: float = 1.0,
        max_grad_norm: float = 1.0,
        policy_frequency: int = 2,
        n_steps: int = 1,
        device: str = "cpu",
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        """Initialize SAC models, replay storage, optimizers, and extensions."""
        del replay_buffer_size, n_steps
        if actor.is_recurrent or critic.is_recurrent:
            raise NotImplementedError("SAC currently supports feedforward actor and critic models only.")
        if num_learning_epochs < 1 or num_mini_batches < 1 or mini_batch_size < 1:
            raise ValueError("SAC update counts and mini_batch_size must be positive.")
        if policy_frequency < 1:
            raise ValueError("policy_frequency must be positive.")
        if alpha <= 0.0:
            raise ValueError("alpha must be positive.")
        if not 0.0 <= gamma <= 1.0 or not 0.0 < tau <= 1.0:
            raise ValueError("gamma must lie in [0, 1] and tau in (0, 1].")

        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        self.gpu_global_rank = 0 if multi_gpu_cfg is None else multi_gpu_cfg["global_rank"]
        self.gpu_world_size = 1 if multi_gpu_cfg is None else multi_gpu_cfg["world_size"]

        self.rnd = RandomNetworkDistillation(device=device, **rnd_cfg) if rnd_cfg else None
        self.symmetry = Symmetry(**symmetry_cfg) if symmetry_cfg else None

        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.replay_buffer = replay_buffer
        self.transition = ReplayBuffer.Transition()
        self.intrinsic_rewards: torch.Tensor | None = None

        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.mini_batch_size = mini_batch_size
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.alpha_learning_rate = alpha_learning_rate
        self.gamma = gamma
        self.tau = tau
        self.max_grad_norm = max_grad_norm
        self.policy_frequency = policy_frequency
        self.update_step = 0
        self.auto_alpha = auto_alpha
        self.target_entropy = -target_entropy_scale * actor.output_dim

        self.log_alpha = nn.Parameter(
            torch.tensor(alpha, device=device).log(),
            requires_grad=auto_alpha,
        )
        self.actor_parameters = [parameter for parameter in self.actor.parameters() if parameter.requires_grad]
        self.critic_parameters = [parameter for parameter in self.critic.parameters() if parameter.requires_grad]
        self.actor_optimizer = resolve_optimizer(actor_optimizer)(self.actor_parameters, lr=actor_learning_rate)
        self.critic_optimizer = resolve_optimizer(critic_optimizer)(self.critic_parameters, lr=critic_learning_rate)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_learning_rate) if auto_alpha else None
        self.critic.init_target_networks()

    @property
    def alpha(self) -> float:
        """Current entropy-temperature value."""
        return self.log_alpha.detach().exp().item()

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample an exploration action and stage its state in the transition."""
        with torch.no_grad():
            actions = self.actor(obs, stochastic_output=True)
        self.transition.observations = obs
        self.transition.actions = actions
        return actions

    def process_env_step(
        self,
        next_obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict,
    ) -> None:
        """Store a step while separating terminal, timeout, and invalid-reset cases."""
        timeouts = extras.get("time_outs", torch.zeros_like(dones, dtype=torch.bool))
        timeouts = timeouts.to(self.device).view(-1).bool() & dones.to(self.device).view(-1).bool()
        true_next_obs = next_obs
        bootstrap = torch.zeros_like(timeouts)
        valid = torch.ones_like(timeouts)

        final_obs = extras.get("time_outs_obs", extras.get("final_observation"))
        if timeouts.any() and final_obs is not None:
            try:
                final_obs = self._as_tensordict(final_obs, next_obs).to(self.device)
                true_next_obs = next_obs.clone()
                for key in true_next_obs:
                    true_next_obs[key][timeouts] = final_obs[key][timeouts]
                bootstrap = timeouts
            except (KeyError, TypeError, ValueError):
                final_obs = None
                true_next_obs = next_obs

        if timeouts.any() and final_obs is None:
            # The returned observation belongs to the reset episode. This tuple has
            # no trustworthy s_{t+1}, so retain it only as a chronological boundary.
            valid = ~timeouts

        self.actor.update_normalization(true_next_obs)
        self.critic.update_normalization(true_next_obs)
        if self.rnd:
            self.rnd.update_normalization(true_next_obs)
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(true_next_obs)
            rewards = rewards + self.intrinsic_rewards
        else:
            self.intrinsic_rewards = None

        self.transition.rewards = rewards
        self.transition.next_observations = true_next_obs
        self.transition.dones = dones
        self.transition.bootstrap = bootstrap
        self.transition.valid = valid
        self.replay_buffer.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)

    def update(self) -> dict[str, float]:
        """Run replay updates and return averaged diagnostics."""
        totals = {"critic1": 0.0, "critic2": 0.0, "actor": 0.0, "alpha": 0.0}
        if self.rnd:
            totals["rnd"] = 0.0
        if self.symmetry:
            totals["symmetry"] = 0.0
        actor_updates = 0
        num_updates = 0

        generator = self.replay_buffer.mini_batch_generator(
            self.num_mini_batches,
            self.mini_batch_size,
            self.num_learning_epochs,
        )
        for batch in generator:
            obs = batch.observations
            actions = batch.actions
            rewards = batch.rewards
            next_obs = batch.next_observations
            bootstrap_mask = batch.bootstrap_mask
            effective_n_steps = batch.effective_n_steps
            original_batch_size = obs.batch_size[0]

            if self.symmetry and self.symmetry.use_data_augmentation:
                obs, actions = self.symmetry.data_augmentation_func(
                    env=self.symmetry.env,
                    obs=obs,
                    actions=actions,
                )
                next_obs, _ = self.symmetry.data_augmentation_func(
                    env=self.symmetry.env,
                    obs=next_obs,
                    actions=None,
                )
                num_aug = int(obs.batch_size[0] / original_batch_size)
                rewards = self._repeat_batch(rewards, num_aug)
                bootstrap_mask = self._repeat_batch(bootstrap_mask, num_aug)
                effective_n_steps = self._repeat_batch(effective_n_steps, num_aug)

            with torch.no_grad():
                next_actions, next_log_prob = self.actor.sample_action_logp(next_obs)
                target_q1, target_q2 = self.critic.evaluate_all_target_q(next_obs, next_actions)
                soft_target = torch.minimum(target_q1, target_q2) - self.log_alpha.exp() * next_log_prob
                discount = torch.pow(
                    torch.as_tensor(self.gamma, device=self.device),
                    effective_n_steps.to(dtype=soft_target.dtype),
                )
                target_q = rewards + discount * bootstrap_mask * soft_target

            q1, q2 = self.critic.evaluate_all_q(obs, actions)
            critic1_loss = nn.functional.mse_loss(q1, target_q)
            critic2_loss = nn.functional.mse_loss(q2, target_q)
            critic_loss = 0.5 * (critic1_loss + critic2_loss)
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            if self.is_multi_gpu:
                self._reduce_gradients(self.critic_parameters)
            nn.utils.clip_grad_norm_(self.critic_parameters, self.max_grad_norm)
            self.critic_optimizer.step()

            new_actions, log_prob = self.actor.sample_action_logp(obs)
            if self.auto_alpha:
                alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad(set_to_none=True)  # type: ignore[union-attr]
                alpha_loss.backward()
                if self.is_multi_gpu:
                    self._reduce_gradients([self.log_alpha])
                self.alpha_optimizer.step()  # type: ignore[union-attr]
            else:
                alpha_loss = torch.zeros((), device=self.device)

            symmetry_loss = torch.zeros((), device=self.device)
            actor_loss = torch.zeros((), device=self.device)
            if self.update_step % self.policy_frequency == 0:
                for parameter in self.critic_parameters:
                    parameter.requires_grad_(False)
                try:
                    q1_new, q2_new = self.critic.evaluate_all_q(obs, new_actions)
                    actor_loss = (self.log_alpha.detach().exp() * log_prob - torch.minimum(q1_new, q2_new)).mean()
                    if self.symmetry:
                        symmetry_loss = self._compute_symmetry_loss(obs, original_batch_size)
                        if self.symmetry.use_mirror_loss:
                            actor_loss = actor_loss + self.symmetry.mirror_loss_coeff * symmetry_loss
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    if self.is_multi_gpu:
                        self._reduce_gradients(self.actor_parameters)
                    nn.utils.clip_grad_norm_(self.actor_parameters, self.max_grad_norm)
                    self.actor_optimizer.step()
                finally:
                    for parameter in self.critic_parameters:
                        parameter.requires_grad_(True)
                actor_updates += 1

            self.critic.soft_update_target_networks(self.tau)

            if self.rnd:
                rnd_loss = self.rnd.compute_loss(batch.observations)
                self.rnd.optimizer.zero_grad(set_to_none=True)
                rnd_loss.backward()
                if self.is_multi_gpu:
                    self._reduce_gradients(self.rnd.parameters())
                self.rnd.optimizer.step()
                totals["rnd"] += rnd_loss.item()

            totals["critic1"] += critic1_loss.item()
            totals["critic2"] += critic2_loss.item()
            totals["actor"] += actor_loss.item()
            totals["alpha"] += alpha_loss.item()
            if self.symmetry:
                totals["symmetry"] += symmetry_loss.item()
            self.update_step += 1
            num_updates += 1

        for key in ("critic1", "critic2", "alpha", "rnd"):
            if key in totals:
                totals[key] /= num_updates
        totals["actor"] /= max(actor_updates, 1)
        if "symmetry" in totals:
            totals["symmetry"] /= max(actor_updates, 1)
        return totals

    def train_mode(self) -> None:
        """Set all learnable components to training mode."""
        self.actor.train()
        self.critic.train()
        if self.rnd:
            self.rnd.train()

    def eval_mode(self) -> None:
        """Set all learnable components to evaluation mode."""
        self.actor.eval()
        self.critic.eval()
        if self.rnd:
            self.rnd.eval()

    def get_policy(self) -> SACActorModel:
        """Return the actor used for inference and export."""
        return self.actor

    def save(self) -> dict:
        """Build a checkpoint dictionary for models and optimizers."""
        saved = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "update_step": self.update_step,
        }
        if self.alpha_optimizer:
            saved["alpha_optimizer_state_dict"] = self.alpha_optimizer.state_dict()
        if self.rnd:
            saved["rnd_state_dict"] = self.rnd.state_dict()
            saved["rnd_optimizer_state_dict"] = self.rnd.optimizer.state_dict()
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Restore selected checkpoint components and report iteration restoration."""
        if load_cfg is None:
            load_cfg = {"actor": True, "critic": True, "optimizer": True, "iteration": True, "rnd": True}
        if load_cfg.get("actor"):
            self.actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        if load_cfg.get("critic"):
            self.critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
        if load_cfg.get("optimizer"):
            self.actor_optimizer.load_state_dict(loaded_dict["actor_optimizer_state_dict"])
            self.critic_optimizer.load_state_dict(loaded_dict["critic_optimizer_state_dict"])
            if self.alpha_optimizer and "alpha_optimizer_state_dict" in loaded_dict:
                self.alpha_optimizer.load_state_dict(loaded_dict["alpha_optimizer_state_dict"])
            self.log_alpha.data.copy_(loaded_dict["log_alpha"].to(self.device))
            self.update_step = loaded_dict.get("update_step", 0)
        if load_cfg.get("rnd") and self.rnd:
            self.rnd.load_state_dict(loaded_dict["rnd_state_dict"], strict=strict)
            self.rnd.optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        return load_cfg.get("iteration", False)

    def clear_storage(self) -> None:
        """Discard all replay contents."""
        self.replay_buffer.clear()

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> SAC:
        """Construct SAC models and replay storage from an RSL-RL configuration."""
        alg_class: type[SAC] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class: type[SACActorModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[SACCriticModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        default_sets = ["actor", "critic"]
        if cfg["algorithm"].get("rnd_cfg") is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        actor = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        critic = critic_class(
            obs,
            cfg["obs_groups"],
            "critic",
            1,
            num_actions=env.num_actions,
            **cfg["critic"],
        ).to(device)
        print(f"SAC Actor Model: {actor}")
        print(f"SAC Critic Model: {critic}")

        replay_buffer = ReplayBuffer(
            num_envs=env.num_envs,
            obs=obs,
            actions_shape=[env.num_actions],
            device=device,
            buffer_size=cfg["algorithm"].get("replay_buffer_size", 1_000_000),
            n_steps=cfg["algorithm"].get("n_steps", 1),
            gamma=cfg["algorithm"].get("gamma", 0.998),
        )
        return alg_class(
            actor,
            critic,
            replay_buffer,
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg.get("multi_gpu"),
        )

    def broadcast_parameters(self) -> None:
        """Broadcast actor, critic, and optional RND state from rank zero."""
        model_states = [self.actor.state_dict(), self.critic.state_dict()]
        if self.rnd:
            model_states.append(self.rnd.predictor.state_dict())
        torch.distributed.broadcast_object_list(model_states, src=0)
        self.actor.load_state_dict(model_states[0])
        self.critic.load_state_dict(model_states[1])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_states[2])

    def _reduce_gradients(self, parameters: Iterable[nn.Parameter]) -> None:
        parameters = list(parameters)
        gradients = [parameter.grad.reshape(-1) for parameter in parameters if parameter.grad is not None]
        if not gradients:
            return
        flat = torch.cat(gradients)
        torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.SUM)
        flat /= self.gpu_world_size
        offset = 0
        for parameter in parameters:
            if parameter.grad is not None:
                count = parameter.numel()
                parameter.grad.copy_(flat[offset : offset + count].view_as(parameter.grad))
                offset += count

    def _compute_symmetry_loss(self, obs: TensorDict, original_batch_size: int) -> torch.Tensor:
        if not self.symmetry.use_data_augmentation:  # type: ignore[union-attr]
            obs, _ = self.symmetry.data_augmentation_func(  # type: ignore[union-attr]
                env=self.symmetry.env,
                obs=obs,
                actions=None,  # type: ignore[union-attr]
            )
        mean_actions = self.actor(obs)
        _, symmetric_actions = self.symmetry.data_augmentation_func(  # type: ignore[union-attr]
            env=self.symmetry.env,  # type: ignore[union-attr]
            obs=None,
            actions=mean_actions[:original_batch_size],
        )
        if mean_actions.shape[0] == original_batch_size:
            return torch.zeros((), device=self.device)
        loss = nn.functional.mse_loss(
            mean_actions[original_batch_size:],
            symmetric_actions.detach()[original_batch_size:],
        )
        return loss if self.symmetry.use_mirror_loss else loss.detach()  # type: ignore[union-attr]

    @staticmethod
    def _as_tensordict(observations: TensorDict | dict, template: TensorDict) -> TensorDict:
        if isinstance(observations, TensorDict):
            return observations
        if isinstance(observations, dict):
            return TensorDict(observations, batch_size=template.batch_size, device=template.device)
        raise TypeError(f"Expected timeout observations to be TensorDict or dict, got {type(observations)}.")

    @staticmethod
    def _repeat_batch(tensor: torch.Tensor, repeats: int) -> torch.Tensor:
        return tensor.repeat(repeats, *([1] * (tensor.ndim - 1)))
