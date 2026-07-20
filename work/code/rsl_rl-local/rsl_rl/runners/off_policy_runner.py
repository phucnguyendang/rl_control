# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Off-policy environment runner."""

from __future__ import annotations

import os
import time
import torch

from rsl_rl.algorithms import SAC
from rsl_rl.env import VecEnv
from rsl_rl.models import SACActorModel
from rsl_rl.utils import check_nan, resolve_callable
from rsl_rl.utils.logger import Logger


class OffPolicyRunner:
    """Collect vectorized experience and train replay-based algorithms."""

    alg: SAC

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        """Construct the algorithm, replay buffer, and logger."""
        self.env = env
        self.cfg = train_cfg
        self.device = device
        self._configure_multi_gpu()

        obs = self.env.get_observations()
        alg_class: type[SAC] = resolve_callable(self.cfg["algorithm"]["class_name"])  # type: ignore
        self.alg = alg_class.construct_algorithm(obs, self.env, self.cfg, self.device)
        self.logger = Logger(
            log_dir=log_dir,
            cfg=self.cfg,
            env_cfg=self.env.cfg,
            num_envs=self.env.num_envs,
            is_distributed=self.is_distributed,
            gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank,
            device=self.device,
        )
        self.current_learning_iteration = 0
        self.start_training = self.cfg.get("start_training", 0)
        self.log_interval = self.cfg.get("log_interval", 1)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Collect experience and run replay updates."""
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length),
            )

        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()
        self.logger.init_logging_writer()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        window_collect_time = 0.0
        window_learn_time = 0.0
        window_iterations = 0
        latest_losses: dict[str, float] = {}

        for it in range(start_it, total_it):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs)
                    next_obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    if self.cfg.get("check_for_nan", True):
                        check_nan(next_obs, rewards, dones)
                    next_obs = next_obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)
                    self.alg.process_env_step(next_obs, rewards, dones, extras)
                    self.logger.process_env_step(rewards, dones, extras, self.alg.intrinsic_rewards)
                    obs = next_obs
            collect_time = time.time() - start

            start = time.time()
            if it >= self.start_training and self.alg.replay_buffer.can_sample():
                latest_losses = self.alg.update()
            else:
                latest_losses = {}
            learn_time = time.time() - start
            self.current_learning_iteration = it

            window_collect_time += collect_time
            window_learn_time += learn_time
            window_iterations += 1
            if it % self.log_interval == 0 or it == total_it - 1:
                collection_size = (
                    self.cfg["num_steps_per_env"] * self.env.num_envs * self.gpu_world_size * window_iterations
                )
                self.logger.log(
                    it=it,
                    start_it=start_it,
                    total_it=total_it,
                    collect_time=window_collect_time,
                    learn_time=window_learn_time,
                    loss_dict=latest_losses,
                    learning_rate=self.alg.actor_learning_rate,
                    action_std=self.alg.get_policy().output_std,
                    rnd_weight=self.alg.rnd.weight if self.alg.rnd else None,
                    alpha=self.alg.alpha,
                    collection_size_override=collection_size,
                )
                window_collect_time = 0.0
                window_learn_time = 0.0
                window_iterations = 0

            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0 and it != 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore[arg-type]

        if self.logger.writer is not None:
            self.save(
                os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt")  # type: ignore[arg-type]
            )
            self.logger.stop_logging_writer()

    def save(self, path: str, infos: dict | None = None) -> None:
        """Save algorithm and iteration state."""
        saved = self.alg.save()
        saved["iter"] = self.current_learning_iteration
        saved["infos"] = infos
        torch.save(saved, path)
        self.logger.save_model(path, self.current_learning_iteration)

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        """Load a checkpoint and return its auxiliary information."""
        loaded = torch.load(path, weights_only=False, map_location=map_location)
        if self.alg.load(loaded, load_cfg, strict):
            self.current_learning_iteration = loaded["iter"]
        return loaded["infos"]

    def get_inference_policy(self, device: str | None = None) -> SACActorModel:
        """Return the deterministic actor interface for inference."""
        self.alg.eval_mode()
        return self.alg.get_policy().to(device)  # type: ignore[return-value]

    def export_policy_to_jit(self, path: str, filename: str = "policy.pt") -> None:
        """Export the actor to TorchScript."""
        model = self.alg.get_policy().as_jit().to("cpu")
        os.makedirs(path, exist_ok=True)
        torch.jit.script(model).save(os.path.join(path, filename))

    def export_policy_to_onnx(self, path: str, filename: str = "policy.onnx", verbose: bool = False) -> None:
        """Export the actor to ONNX."""
        model = self.alg.get_policy().as_onnx(verbose=verbose).to("cpu").eval()
        os.makedirs(path, exist_ok=True)
        torch.onnx.export(
            model,
            model.get_dummy_inputs(),  # type: ignore[attr-defined]
            os.path.join(path, filename),
            export_params=True,
            opset_version=18,
            verbose=verbose,
            input_names=model.input_names,  # type: ignore[attr-defined]
            output_names=model.output_names,  # type: ignore[attr-defined]
        )

    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        """Register another repository for code-state logging."""
        self.logger.git_status_repos.append(repo_file_path)

    def _configure_multi_gpu(self) -> None:
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.cfg["multi_gpu"] = None
            return

        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))
        self.cfg["multi_gpu"] = {
            "global_rank": self.gpu_global_rank,
            "local_rank": self.gpu_local_rank,
            "world_size": self.gpu_world_size,
        }
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(
                f"Device '{self.device}' does not match expected local-rank device 'cuda:{self.gpu_local_rank}'."
            )
        if self.gpu_local_rank >= self.gpu_world_size or self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError("Distributed rank must be smaller than world size.")
        torch.distributed.init_process_group(
            backend="nccl",
            rank=self.gpu_global_rank,
            world_size=self.gpu_world_size,
        )
        torch.cuda.set_device(self.gpu_local_rank)
