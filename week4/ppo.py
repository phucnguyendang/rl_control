from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from cartpole import CartPole, rk4_step

@dataclass
class PPOConfig:
    total_updates: int = 500
    rollout_steps: int = 512
    num_rollouts: int = 8

    ppo_epochs: int = 10
    batch_size: int = 256

    # PPO / GAE.
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    target_kl: float | None = None

    # Optimizer.
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    value_coef: float = 0.5
    entropy_coef: float = 0.001
    max_grad_norm: float = 0.5

    # Episode rules.
    # terminated: cart leaves [-x_limit, x_limit]
    # truncated: episode reaches max_episode_steps
    x_limit: float = 5.0
    max_episode_steps: int | None = 1000

    # Initial-state distribution.
    random_reset: bool = True
    reset_x_range: float = 0.5
    reset_x_dot_range: float = 0.5
    reset_theta_range: float = float(np.pi)
    reset_theta_dot_range: float = 1.0

    # Reward shaping.
    x_penalty: float = 0.03
    x_dot_penalty: float = 0.005
    theta_dot_penalty: float = 0.01
    action_penalty: float = 0.001

    # Applied only on real termination, not on time truncation.
    done_penalty: float = 20.0

    upright_bonus: float = 3.0

    # Bonus condition.
    upright_theta_threshold: float = 0.15
    upright_theta_dot_threshold: float = 0.5
    upright_x_threshold: float = 0.5

    seed: int = 0
    device: str = "cpu"
    print_every: int = 10
    checkpoint_dir: str | None = "checkpoints"
    checkpoint_prefix: str = "ppo_cartpole"


class Critic(nn.Module):
    def __init__(self, env: CartPole):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(env.state_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class Actor(nn.Module):
    def __init__(self, env: CartPole):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(env.state_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        self.log_std = nn.Parameter(torch.zeros(1))
        self.force_limit = float(env.params.force_limit)
        self.eps = 1e-6

    def get_dist(self, state: torch.Tensor) -> torch.distributions.Normal:
        mean = self.net(state)
        log_std = self.log_std.expand_as(mean)
        std = log_std.exp()
        return torch.distributions.Normal(mean, std)

    def _squash(self, raw_action: torch.Tensor, limit: float) -> torch.Tensor:
        return torch.tanh(raw_action) * limit

    def _atanh(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, -1.0 + self.eps, 1.0 - self.eps)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    def _log_prob_from_raw(
        self,
        dist: torch.distributions.Normal,
        raw_action: torch.Tensor,
    ) -> torch.Tensor:
        squashed = torch.tanh(raw_action)
        log_prob_raw = dist.log_prob(raw_action)

        # action = force_limit * tanh(raw_action)
        # da / draw = force_limit * (1 - tanh(raw_action)^2)
        log_abs_det_jacobian = torch.log(
            self.force_limit * (1.0 - squashed.pow(2)) + self.eps
        )

        log_prob = log_prob_raw - log_abs_det_jacobian
        return log_prob.sum(dim=-1)

    @torch.no_grad()
    def sample_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.get_dist(state)
        raw_action = dist.sample()
        action = self._squash(raw_action, self.force_limit)
        log_prob = self._log_prob_from_raw(dist, raw_action)
        return action, log_prob

    def evaluate_action(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.get_dist(state)

        normalized_action = action / self.force_limit
        raw_action = self._atanh(normalized_action)
        log_prob = self._log_prob_from_raw(dist, raw_action)

        # Entropy of the base Gaussian. This is a useful exploration regularizer,
        # although it is not the exact entropy of the squashed policy.
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy

    @torch.no_grad()
    def mean_action(self, state: torch.Tensor) -> torch.Tensor:
        dist = self.get_dist(state)
        return self._squash(dist.mean, self.force_limit)


class Agent(nn.Module):
    def __init__(self, env: CartPole):
        super().__init__()
        self.actor = Actor(env)
        self.critic = Critic(env)


def is_done_state(state: np.ndarray, x_limit: float) -> bool:
    return abs(float(state[0])) > x_limit


def reset_env(env: CartPole, cfg: PPOConfig) -> np.ndarray:
    if not cfg.random_reset:
        return env.reset()

    rng = getattr(env, "rng", np.random.default_rng(cfg.seed))
    state0 = np.array(
        [
            rng.uniform(-cfg.reset_x_range, cfg.reset_x_range),
            rng.uniform(-cfg.reset_x_dot_range, cfg.reset_x_dot_range),
            rng.uniform(-cfg.reset_theta_range, cfg.reset_theta_range),
            rng.uniform(-cfg.reset_theta_dot_range, cfg.reset_theta_dot_range),
        ],
        dtype=float,
    )

    return env.reset(state0)


def reset_rollout_states(env: CartPole, cfg: PPOConfig) -> np.ndarray:
    return np.stack([reset_env(env, cfg) for _ in range(cfg.num_rollouts)], axis=0)


def calculate_rewards(
    states: np.ndarray,
    actions: np.ndarray,
    terminateds: np.ndarray,
    cfg: PPOConfig,
) -> np.ndarray:
    state_arr = np.asarray(states, dtype=float)
    action_arr = np.asarray(actions, dtype=float)
    if action_arr.shape[-1:] == (1,):
        action_arr = np.squeeze(action_arr, axis=-1)
    action_arr = np.broadcast_to(action_arr, state_arr.shape[:-1])
    terminated_arr = np.asarray(terminateds, dtype=bool)

    x = state_arr[..., 0]
    x_dot = state_arr[..., 1]
    theta = state_arr[..., 2]
    theta_dot = state_arr[..., 3]

    reward = np.cos(theta)
    reward -= cfg.x_penalty * x**2
    reward -= cfg.x_dot_penalty * x_dot**2
    reward -= cfg.theta_dot_penalty * theta_dot**2
    reward -= cfg.action_penalty * action_arr**2

    near_upright = (
        (np.abs(theta) < cfg.upright_theta_threshold)
        & (np.abs(theta_dot) < cfg.upright_theta_dot_threshold)
        & (np.abs(x) < cfg.upright_x_threshold)
    )

    reward = np.where(near_upright, reward + cfg.upright_bonus, reward)
    reward = np.where(terminated_arr, reward - cfg.done_penalty, reward)
    return np.asarray(reward, dtype=float)


def calculate_reward(
    state: np.ndarray,
    action: np.ndarray,
    terminated: bool,
    cfg: PPOConfig,
) -> float:
    return float(calculate_rewards(state, action, np.asarray(terminated), cfg))


def collect_rollout_batch(
    env: CartPole,
    agent: Agent,
    cfg: PPOConfig,
) -> Dict[str, torch.Tensor | np.ndarray]:
    agent.eval()

    device = torch.device(cfg.device)
    num_rollouts = cfg.num_rollouts

    states = []
    actions = []
    old_log_probs = []
    rewards = []
    terminateds = []
    truncateds = []
    episode_ends = []
    values = []
    next_values = []

    episode_returns = []
    episode_lengths = []
    episode_terminateds = []
    episode_truncateds = []

    current_ep_return = np.zeros(num_rollouts, dtype=float)
    current_ep_len = np.zeros(num_rollouts, dtype=np.int64)

    rollout_states = reset_rollout_states(env, cfg)

    with torch.no_grad():
        for _ in range(cfg.rollout_steps):
            state_tensor = torch.as_tensor(
                rollout_states,
                dtype=torch.float32,
                device=device,
            )

            value_tensor = agent.critic(state_tensor).squeeze(-1)
            action_tensor, log_prob_tensor = agent.actor.sample_action(state_tensor)
            action_np = action_tensor.cpu().numpy()

            # action is already inside [-force_limit, force_limit] because of tanh.
            next_states = rk4_step(
                rollout_states,
                action_np,
                params=env.params,
                wrap=True,
            )

            terminated = np.abs(next_states[:, 0]) > cfg.x_limit
            if cfg.max_episode_steps is None:
                truncated = np.zeros(num_rollouts, dtype=bool)
            else:
                truncated = (~terminated) & (
                    current_ep_len + 1 >= cfg.max_episode_steps
                )
            episode_end = terminated | truncated

            reward = calculate_rewards(next_states, action_np, terminated, cfg)

            next_state_tensor = torch.as_tensor(
                next_states,
                dtype=torch.float32,
                device=device,
            )
            next_value_tensor = agent.critic(next_state_tensor).squeeze(-1)
            next_value_tensor = torch.where(
                torch.as_tensor(terminated, dtype=torch.bool, device=device),
                torch.zeros_like(next_value_tensor),
                next_value_tensor,
            )

            states.append(rollout_states.copy())
            actions.append(action_np.copy())
            old_log_probs.append(log_prob_tensor.cpu().numpy().copy())
            rewards.append(reward.copy())
            terminateds.append(terminated.astype(float))
            truncateds.append(truncated.astype(float))
            episode_ends.append(episode_end.astype(float))
            values.append(value_tensor.cpu().numpy().copy())
            next_values.append(next_value_tensor.cpu().numpy().copy())

            current_ep_return += reward
            current_ep_len += 1

            if np.any(episode_end):
                for rollout_idx in np.flatnonzero(episode_end):
                    episode_returns.append(float(current_ep_return[rollout_idx]))
                    episode_lengths.append(float(current_ep_len[rollout_idx]))
                    episode_terminateds.append(float(terminated[rollout_idx]))
                    episode_truncateds.append(float(truncated[rollout_idx]))

                    current_ep_return[rollout_idx] = 0.0
                    current_ep_len[rollout_idx] = 0
                    next_states[rollout_idx] = reset_env(env, cfg)

            rollout_states = next_states

    # Current unfinished episode statistic. This is logged separately and is not
    # mixed into mean_episode_return.
    partial_episode_return = current_ep_return.copy()
    partial_episode_length = current_ep_len.astype(float).copy()

    batch = {
        "states": torch.as_tensor(np.asarray(states), dtype=torch.float32, device=device),
        "actions": torch.as_tensor(np.asarray(actions), dtype=torch.float32, device=device),
        "old_log_probs": torch.as_tensor(
            np.asarray(old_log_probs),
            dtype=torch.float32,
            device=device,
        ),
        "rewards": torch.as_tensor(np.asarray(rewards), dtype=torch.float32, device=device),
        "terminateds": torch.as_tensor(
            np.asarray(terminateds),
            dtype=torch.float32,
            device=device,
        ),
        "truncateds": torch.as_tensor(
            np.asarray(truncateds),
            dtype=torch.float32,
            device=device,
        ),
        "episode_ends": torch.as_tensor(
            np.asarray(episode_ends),
            dtype=torch.float32,
            device=device,
        ),
        "values": torch.as_tensor(np.asarray(values), dtype=torch.float32, device=device),
        "next_values": torch.as_tensor(
            np.asarray(next_values),
            dtype=torch.float32,
            device=device,
        ),
        "episode_returns": np.asarray(episode_returns, dtype=np.float32),
        "episode_lengths": np.asarray(episode_lengths, dtype=np.float32),
        "episode_terminateds": np.asarray(episode_terminateds, dtype=np.float32),
        "episode_truncateds": np.asarray(episode_truncateds, dtype=np.float32),
        "partial_episode_return": np.asarray(partial_episode_return, dtype=np.float32),
        "partial_episode_length": np.asarray(partial_episode_length, dtype=np.float32),
    }

    return batch


def compute_gae(
    batch: Dict[str, torch.Tensor | np.ndarray],
    cfg: PPOConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rewards = batch["rewards"]
    terminateds = batch["terminateds"]
    episode_ends = batch["episode_ends"]
    values = batch["values"]
    next_values = batch["next_values"]

    assert isinstance(rewards, torch.Tensor)
    assert isinstance(terminateds, torch.Tensor)
    assert isinstance(episode_ends, torch.Tensor)
    assert isinstance(values, torch.Tensor)
    assert isinstance(next_values, torch.Tensor)

    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(rewards.shape[1], dtype=rewards.dtype, device=rewards.device)

    n_steps = rewards.shape[0]
    for t in reversed(range(n_steps)):
        # terminated=True means true terminal failure, so no bootstrap.
        # truncated=True still bootstraps through next_values[t].
        not_terminal = 1.0 - terminateds[t]
        delta = rewards[t] + cfg.gamma * not_terminal * next_values[t] - values[t]

        # Do not let GAE leak across either episode boundaries or the end of
        # the rollout batch.
        if t == n_steps - 1:
            continuation = torch.zeros_like(gae)
        else:
            continuation = 1.0 - episode_ends[t]

        gae = delta + cfg.gamma * cfg.gae_lambda * continuation * gae
        advantages[t] = gae

    returns = advantages + values
    advantages = (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + 1e-8
    )

    return returns, advantages


class PPODataloader:
    def __init__(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        batch_size: int,
    ):
        self.states = states
        self.actions = actions
        self.old_log_probs = old_log_probs
        self.returns = returns
        self.advantages = advantages
        self.batch_size = batch_size

    def __iter__(self):
        n = self.states.shape[0]
        indices = torch.randperm(n, device=self.states.device)

        for start in range(0, n, self.batch_size):
            idx = indices[start : start + self.batch_size]
            yield (
                self.states[idx],
                self.actions[idx],
                self.old_log_probs[idx],
                self.returns[idx],
                self.advantages[idx],
            )


def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    with torch.no_grad():
        var_y = torch.var(y_true)
        if var_y.item() < 1e-8:
            return 0.0
        return float((1.0 - torch.var(y_true - y_pred) / var_y).item())


def make_empty_history() -> Dict[str, list]:
    return {
        "update": [],
        "mean_step_reward": [],
        "mean_episode_return": [],
        "mean_episode_length": [],
        "num_episodes": [],
        "num_terminated_episodes": [],
        "num_truncated_episodes": [],
        "terminated_rate": [],
        "truncated_rate": [],
        "episode_end_rate": [],
        "partial_episode_return": [],
        "partial_episode_length": [],
        "mean_abs_x": [],
        "mean_abs_theta": [],
        "upright_fraction": [],
        "actor_loss": [],
        "value_loss": [],
        "entropy": [],
        "approx_kl": [],
        "clip_fraction": [],
        "explained_variance": [],
    }


def append_history(
    history: Dict[str, list],
    update: int,
    batch: Dict[str, torch.Tensor | np.ndarray],
    stats: Dict[str, float],
    cfg: PPOConfig,
) -> None:
    states = batch["states"]
    rewards = batch["rewards"]
    terminateds = batch["terminateds"]
    truncateds = batch["truncateds"]
    episode_ends = batch["episode_ends"]
    episode_returns = batch["episode_returns"]
    episode_lengths = batch["episode_lengths"]
    episode_terminateds = batch["episode_terminateds"]
    episode_truncateds = batch["episode_truncateds"]
    partial_episode_return = batch["partial_episode_return"]
    partial_episode_length = batch["partial_episode_length"]

    assert isinstance(states, torch.Tensor)
    assert isinstance(rewards, torch.Tensor)
    assert isinstance(terminateds, torch.Tensor)
    assert isinstance(truncateds, torch.Tensor)
    assert isinstance(episode_ends, torch.Tensor)
    assert isinstance(episode_returns, np.ndarray)
    assert isinstance(episode_lengths, np.ndarray)
    assert isinstance(episode_terminateds, np.ndarray)
    assert isinstance(episode_truncateds, np.ndarray)
    assert isinstance(partial_episode_return, np.ndarray)
    assert isinstance(partial_episode_length, np.ndarray)

    # Clean definition: mean_episode_return only uses completed episodes.
    # A completed episode is either terminated or truncated.
    if len(episode_returns) > 0:
        mean_episode_return = float(np.mean(episode_returns))
        mean_episode_length = float(np.mean(episode_lengths))
    else:
        mean_episode_return = float("nan")
        mean_episode_length = float("nan")

    flat_states = states.reshape(-1, states.shape[-1])
    theta = flat_states[:, 2]
    x = flat_states[:, 0]
    upright = (
        (theta.abs() < cfg.upright_theta_threshold)
        & (x.abs() < cfg.upright_x_threshold)
    )

    history["update"].append(update)
    history["mean_step_reward"].append(float(rewards.reshape(-1).mean().item()))
    history["mean_episode_return"].append(mean_episode_return)
    history["mean_episode_length"].append(mean_episode_length)
    history["num_episodes"].append(int(len(episode_returns)))
    history["num_terminated_episodes"].append(int(np.sum(episode_terminateds)))
    history["num_truncated_episodes"].append(int(np.sum(episode_truncateds)))
    history["terminated_rate"].append(float(terminateds.reshape(-1).mean().item()))
    history["truncated_rate"].append(float(truncateds.reshape(-1).mean().item()))
    history["episode_end_rate"].append(float(episode_ends.reshape(-1).mean().item()))
    history["partial_episode_return"].append(float(np.mean(partial_episode_return)))
    history["partial_episode_length"].append(float(np.mean(partial_episode_length)))
    history["mean_abs_x"].append(float(x.abs().mean().item()))
    history["mean_abs_theta"].append(float(theta.abs().mean().item()))
    history["upright_fraction"].append(float(upright.float().mean().item()))
    history["actor_loss"].append(float(stats["actor_loss"]))
    history["value_loss"].append(float(stats["value_loss"]))
    history["entropy"].append(float(stats["entropy"]))
    history["approx_kl"].append(float(stats["approx_kl"]))
    history["clip_fraction"].append(float(stats["clip_fraction"]))
    history["explained_variance"].append(float(stats["explained_variance"]))


def save_checkpoint(
    path: Path,
    agent: Agent,
    optimizer: optim.Optimizer,
    cfg: PPOConfig,
    history: Dict[str, list],
    update: int,
    ep_return: float,
    best_ep_return: float | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "update": update,
            "ep_return": ep_return,
            "best_ep_return": best_ep_return,
            "config": asdict(cfg),
            "agent_state_dict": agent.state_dict(),
            "actor_state_dict": agent.actor.state_dict(),
            "critic_state_dict": agent.critic.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
        },
        path,
    )


def train(env: CartPole, cfg: PPOConfig | None = None) -> Tuple[Agent, Dict[str, list]]:
    if cfg is None:
        cfg = PPOConfig()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device(cfg.device)
    agent = Agent(env).to(device)

    optimizer = optim.Adam(
        [
            {"params": agent.actor.parameters(), "lr": cfg.actor_lr},
            {"params": agent.critic.parameters(), "lr": cfg.critic_lr},
        ]
    )

    history = make_empty_history()
    best_ep_return = float("-inf")
    checkpoint_dir = Path(cfg.checkpoint_dir) if cfg.checkpoint_dir else None

    for update in range(1, cfg.total_updates + 1):
        batch = collect_rollout_batch(env, agent, cfg)
        returns, advantages = compute_gae(batch, cfg)

        states_rollout = batch["states"]
        actions_rollout = batch["actions"]
        old_log_probs_rollout = batch["old_log_probs"]

        assert isinstance(states_rollout, torch.Tensor)
        assert isinstance(actions_rollout, torch.Tensor)
        assert isinstance(old_log_probs_rollout, torch.Tensor)

        states = states_rollout.reshape(-1, states_rollout.shape[-1])
        actions = actions_rollout.reshape(-1, actions_rollout.shape[-1])
        old_log_probs = old_log_probs_rollout.reshape(-1)
        returns_flat = returns.reshape(-1)
        advantages_flat = advantages.reshape(-1)

        loader = PPODataloader(
            states=states,
            actions=actions,
            old_log_probs=old_log_probs,
            returns=returns_flat,
            advantages=advantages_flat,
            batch_size=cfg.batch_size,
        )

        agent.train()

        actor_losses = []
        value_losses = []
        entropies = []
        approx_kls = []
        clip_fracs = []

        stop_update_early = False

        for _epoch in range(cfg.ppo_epochs):
            for (
                batch_states,
                batch_actions,
                batch_old_log_probs,
                batch_returns,
                batch_advantages,
            ) in loader:
                values_pred = agent.critic(batch_states).squeeze(-1)
                value_loss = nn.functional.mse_loss(values_pred, batch_returns)

                new_log_probs, entropy = agent.actor.evaluate_action(
                    batch_states,
                    batch_actions,
                )

                log_ratio = new_log_probs - batch_old_log_probs
                ratio = torch.exp(log_ratio)

                surrogate1 = ratio * batch_advantages
                surrogate2 = torch.clamp(
                    ratio,
                    1.0 - cfg.clip_epsilon,
                    1.0 + cfg.clip_epsilon,
                ) * batch_advantages

                actor_loss = -torch.min(surrogate1, surrogate2).mean()
                entropy_mean = entropy.mean()

                loss = (
                    actor_loss
                    + cfg.value_coef * value_loss
                    - cfg.entropy_coef * entropy_mean
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    approx_kl = (batch_old_log_probs - new_log_probs).mean()
                    clip_fraction = (
                        (ratio - 1.0).abs() > cfg.clip_epsilon
                    ).float().mean()

                actor_losses.append(float(actor_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy_mean.item()))
                approx_kls.append(float(approx_kl.item()))
                clip_fracs.append(float(clip_fraction.item()))

                if cfg.target_kl is not None and approx_kl.item() > cfg.target_kl:
                    stop_update_early = True
                    break

            if stop_update_early:
                break

        with torch.no_grad():
            values_after_update = agent.critic(states).squeeze(-1)
            ev = explained_variance(values_after_update, returns_flat)

        stats = {
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
            "clip_fraction": float(np.mean(clip_fracs)) if clip_fracs else 0.0,
            "explained_variance": ev,
        }

        append_history(history, update, batch, stats, cfg)

        ep_return = history["mean_episode_return"][-1]
        if checkpoint_dir is not None:
            if np.isfinite(ep_return) and ep_return > best_ep_return:
                best_ep_return = ep_return
                save_checkpoint(
                    checkpoint_dir / f"{cfg.checkpoint_prefix}_best.pt",
                    agent,
                    optimizer,
                    cfg,
                    history,
                    update,
                    ep_return,
                    best_ep_return,
                )

            best_for_save = None if not np.isfinite(best_ep_return) else best_ep_return
            save_checkpoint(
                checkpoint_dir / f"{cfg.checkpoint_prefix}_last.pt",
                agent,
                optimizer,
                cfg,
                history,
                update,
                ep_return,
                best_for_save,
            )

        if cfg.print_every and update % cfg.print_every == 0:
            print(
                f"update={update:04d} | "
                f"ep_return={history['mean_episode_return'][-1]: .3f} | "
                f"ep_len={history['mean_episode_length'][-1]: .1f} | "
                f"num_ep={history['num_episodes'][-1]} | "
                f"term_ep={history['num_terminated_episodes'][-1]} | "
                f"trunc_ep={history['num_truncated_episodes'][-1]} | "
                f"partial_ret={history['partial_episode_return'][-1]: .3f} | "
                f"partial_len={history['partial_episode_length'][-1]: .0f} | "
                f"step_reward={history['mean_step_reward'][-1]: .3f} | "
                f"|x|={history['mean_abs_x'][-1]: .3f} | "
                f"|theta|={history['mean_abs_theta'][-1]: .3f} | "
                f"upright={history['upright_fraction'][-1]: .3f} | "
                f"actor_loss={stats['actor_loss']: .3f} | "
                f"value_loss={stats['value_loss']: .3f} | "
                f"kl={stats['approx_kl']: .5f}"
            )

    return agent, history


@torch.no_grad()
def run_policy(
    env: CartPole,
    agent: Agent,
    max_steps: int = 1000,
    x_limit: float = 5.0,
    device: str = "cpu",
    deterministic: bool = True,
    state0: np.ndarray | None = None,
    cfg: PPOConfig | None = None,
    use_random_reset: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    agent.eval()

    if use_random_reset:
        if cfg is None:
            raise ValueError("cfg must be provided when use_random_reset=True")
        state = reset_env(env, cfg)
        reward_cfg = cfg
    elif state0 is not None:
        state = env.reset(state0)
        reward_cfg = cfg if cfg is not None else PPOConfig(x_limit=x_limit, device=device)
    else:
        state = env.reset()
        reward_cfg = cfg if cfg is not None else PPOConfig(x_limit=x_limit, device=device)

    states = [state.copy()]
    actions = []
    rewards = []

    for _ in range(max_steps):
        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        if deterministic:
            action_tensor = agent.actor.mean_action(state_tensor)
        else:
            action_tensor, _ = agent.actor.sample_action(state_tensor)

        action_np = action_tensor.squeeze(0).cpu().numpy()
        next_state = env.step(action_np, method="rk4", clip_force=False, wrap=True)
        terminated = is_done_state(next_state, x_limit)
        reward = calculate_reward(next_state, action_np, terminated, reward_cfg)

        states.append(next_state.copy())
        actions.append(action_np.copy())
        rewards.append(reward)

        state = next_state
        if terminated:
            break

    return np.asarray(states), np.asarray(actions), np.asarray(rewards)
