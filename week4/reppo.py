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
class REPPOConfig:
    total_updates: int = 500
    rollout_steps: int = 512
    num_rollouts: int = 8

    # Kept close to ppo.py. In REPPO, this means update epochs.
    ppo_epochs: int = 10
    batch_size: int = 256

    # Discount and lambda. In REPPO, gae_lambda is used as TD-lambda for Q targets.
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # HL Gauss hyper params
    q_bin = 10
    target_q_std = q_bin * 2
    q_value_max: float = 4000
    q_value_min: float = -1000
    num_q_bins = (q_value_max - q_value_min) // q_bin + 1

    # REPPO KL / entropy constraints.
    # actor_loss = -Q(s, a_pi) + alpha * log_pi(a_pi|s) while KL < target_kl,
    # otherwise actor_loss = beta * KL(old || new).
    target_kl: float = 0.02
    target_entropy: float = 1.0  # Base-Gaussian entropy target for 1D action.
    initial_alpha: float = 0.001
    initial_beta: float = 1.0
    alpha_lr: float = 1e-3
    beta_lr: float = 1e-3
    min_log_multiplier: float = -20.0
    max_log_multiplier: float = 10.0

    # Optimizer.
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
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
    # theta = 0 is upright, so cos(theta) gives:
    # upright -> +1, downward -> -1.
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

    # Misc.
    seed: int = 0
    device: str = "cpu"
    print_every: int = 10
    checkpoint_dir: str | None = "checkpoints_reppo"
    checkpoint_prefix: str = "reppo_cartpole"


class QCritic(nn.Module):
    def __init__(self, env: CartPole, cfg: REPPOConfig):
        super().__init__()
        self.force_limit = float(env.params.force_limit)

        self.net = nn.Sequential(
            nn.Linear(env.state_dim + 1, 64),
            nn.LayerNorm(64),
            nn.SiLU(),

            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.SiLU(),

            nn.Linear(64, cfg.num_q_bins),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        normalized_action = action / self.force_limit
        x = torch.cat([state, normalized_action], dim=-1)
        return self.net(x)
    
class Actor(nn.Module):
    """Gaussian policy with tanh squashing to respect force limits.

    raw_action ~ Normal(mean, std)
    action = tanh(raw_action) * force_limit

    log_prob includes the tanh change-of-variable correction.
    """

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

    def rsample_action(
        self,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.get_dist(state)
        raw_action = dist.rsample()
        action = self._squash(raw_action, self.force_limit)
        log_prob = self._log_prob_from_raw(dist, raw_action)

        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy

    def evaluate_action(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.get_dist(state)

        normalized_action = action / self.force_limit
        raw_action = self._atanh(normalized_action)
        log_prob = self._log_prob_from_raw(dist, raw_action)

        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy

    @torch.no_grad()
    def mean_action(self, state: torch.Tensor) -> torch.Tensor:
        dist = self.get_dist(state)
        return self._squash(dist.mean, self.force_limit)

class Agent(nn.Module):
    def __init__(self, env: CartPole, cfg: REPPOConfig):
        super().__init__()
        self.actor = Actor(env)
        self.critic = QCritic(env, cfg)

def is_done_state(state: np.ndarray, x_limit: float) -> bool:
    return abs(float(state[0])) > x_limit


def reset_env(env: CartPole, cfg: REPPOConfig) -> np.ndarray:
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


def reset_rollout_states(env: CartPole, cfg: REPPOConfig) -> np.ndarray:
    return np.stack([reset_env(env, cfg) for _ in range(cfg.num_rollouts)], axis=0)


def calculate_rewards(
    states: np.ndarray,
    actions: np.ndarray,
    terminateds: np.ndarray,
    cfg: REPPOConfig,
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
    cfg: REPPOConfig,
) -> float:
    return float(calculate_rewards(state, action, np.asarray(terminated), cfg))


def collect_rollout_batch(
    env: CartPole,
    agent: Agent,
    cfg: REPPOConfig,
) -> Dict[str, torch.Tensor | np.ndarray]:
    agent.eval()

    device = torch.device(cfg.device)
    num_rollouts = cfg.num_rollouts

    states = []
    actions = []
    old_log_probs = []
    rewards = []
    next_states_for_targets = []
    terminateds = []
    truncateds = []
    episode_ends = []

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

            # Keep the true transition next_state before reset. This is needed for
            # bootstrapping Q at rollout/time-limit boundaries.
            transition_next_states = next_states.copy()

            states.append(rollout_states.copy())
            actions.append(action_np.copy())
            old_log_probs.append(log_prob_tensor.cpu().numpy().copy())
            rewards.append(reward.copy())
            next_states_for_targets.append(transition_next_states)
            terminateds.append(terminated.astype(float))
            truncateds.append(truncated.astype(float))
            episode_ends.append(episode_end.astype(float))

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
        "next_states": torch.as_tensor(
            np.asarray(next_states_for_targets),
            dtype=torch.float32,
            device=device,
        ),
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
        "episode_returns": np.asarray(episode_returns, dtype=np.float32),
        "episode_lengths": np.asarray(episode_lengths, dtype=np.float32),
        "episode_terminateds": np.asarray(episode_terminateds, dtype=np.float32),
        "episode_truncateds": np.asarray(episode_truncateds, dtype=np.float32),
        "partial_episode_return": np.asarray(partial_episode_return, dtype=np.float32),
        "partial_episode_length": np.asarray(partial_episode_length, dtype=np.float32),
    }

    return batch

def q_bin_centers(
    cfg: REPPOConfig,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return torch.linspace(
        cfg.q_value_min,
        cfg.q_value_max,
        cfg.num_q_bins,
        device=device,
        dtype=dtype,
    )


def normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))


def scalar_to_hl_gauss_targets(
    targets: torch.Tensor,
    cfg: REPPOConfig,
) -> torch.Tensor:
    targets = targets.clamp(cfg.q_value_min, cfg.q_value_max)

    centers = q_bin_centers(cfg, targets.device, targets.dtype)
    bin_width = centers[1] - centers[0]

    edges = torch.empty(
        cfg.num_q_bins + 1,
        device=targets.device,
        dtype=targets.dtype,
    )
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * bin_width
    edges[-1] = centers[-1] + 0.5 * bin_width

    sigma = max(float(cfg.target_q_std), 1e-6)

    z = (edges.unsqueeze(0) - targets.unsqueeze(-1)) / sigma
    cdf = normal_cdf(z)

    probs = cdf[:, 1:] - cdf[:, :-1]
    probs = probs.clamp_min(1e-12)
    probs = probs / probs.sum(dim=-1, keepdim=True)

    return probs


def hl_gauss_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    cfg: REPPOConfig,
) -> torch.Tensor:
    target_probs = scalar_to_hl_gauss_targets(targets, cfg)
    log_probs = nn.functional.log_softmax(logits, dim=-1)
    return -(target_probs * log_probs).sum(dim=-1).mean()


def q_value_from_logits(
    logits: torch.Tensor,
    cfg: REPPOConfig,
) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    centers = q_bin_centers(cfg, logits.device, logits.dtype)
    return (probs * centers).sum(dim=-1)

@torch.no_grad()
def compute_reppo_q_targets(
    batch: Dict[str, torch.Tensor | np.ndarray],
    agent: Agent,
    cfg: REPPOConfig,
    alpha: float,
) -> torch.Tensor:
    """Compute fixed on-policy soft TD-lambda Q targets for one rollout batch."""
    rewards = batch["rewards"]
    old_log_probs = batch["old_log_probs"]
    next_states = batch["next_states"]
    terminateds = batch["terminateds"]
    episode_ends = batch["episode_ends"]

    assert isinstance(rewards, torch.Tensor)
    assert isinstance(old_log_probs, torch.Tensor)
    assert isinstance(next_states, torch.Tensor)
    assert isinstance(terminateds, torch.Tensor)
    assert isinstance(episode_ends, torch.Tensor)

    n_steps, num_rollouts = rewards.shape
    soft_rewards = rewards - float(alpha) * old_log_probs

    flat_next_states = next_states.reshape(-1, next_states.shape[-1])
    next_actions, _ = agent.actor.sample_action(flat_next_states)
    q_bootstrap_logits = agent.critic(flat_next_states, next_actions)
    q_bootstrap = q_value_from_logits(q_bootstrap_logits, cfg)
    q_bootstrap = q_bootstrap.reshape(n_steps, num_rollouts)

    targets = torch.zeros_like(rewards)
    g = q_bootstrap[-1]

    for t in reversed(range(n_steps)):
        not_terminal = 1.0 - terminateds[t]

        # Do not let the lambda recursion leak across either real episode
        # boundaries, time truncations, or the end of the rollout batch. Still
        # bootstrap one step at non-terminal boundaries.
        if t == n_steps - 1:
            continuation = torch.zeros_like(g)
        else:
            continuation = 1.0 - episode_ends[t]

        mixed_next_q = continuation * (
            (1.0 - cfg.gae_lambda) * q_bootstrap[t] + cfg.gae_lambda * g
        ) + (1.0 - continuation) * q_bootstrap[t]

        g = soft_rewards[t] + cfg.gamma * not_terminal * mixed_next_q
        targets[t] = g

    return targets


class REPPODataloader:
    def __init__(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        q_targets: torch.Tensor,
        batch_size: int,
    ):
        self.states = states
        self.actions = actions
        self.old_log_probs = old_log_probs
        self.q_targets = q_targets
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
                self.q_targets[idx],
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
        "alpha": [],
        "beta": [],
        "log_alpha": [],
        "log_beta": [],
        "q_mean": [],
        "q_target_mean": [],
    }


def append_history(
    history: Dict[str, list],
    update: int,
    batch: Dict[str, torch.Tensor | np.ndarray],
    stats: Dict[str, float],
    cfg: REPPOConfig,
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
    history["alpha"].append(float(stats["alpha"]))
    history["beta"].append(float(stats["beta"]))
    history["log_alpha"].append(float(stats["log_alpha"]))
    history["log_beta"].append(float(stats["log_beta"]))
    history["q_mean"].append(float(stats["q_mean"]))
    history["q_target_mean"].append(float(stats["q_target_mean"]))


def _safe_log(x: float, eps: float = 1e-12) -> float:
    return float(np.log(max(float(x), eps)))


def _coef_from_log(log_value: float, cfg: REPPOConfig) -> float:
    clipped = np.clip(log_value, cfg.min_log_multiplier, cfg.max_log_multiplier)
    return float(np.exp(clipped))

def _update_multipliers_paper_style(
    log_alpha: float,
    log_beta: float,
    entropy: float,
    kl: float,
    cfg: REPPOConfig,
) -> Tuple[float, float]:
    """Update log multipliers using the REPPO paper's Eq. 11 and Eq. 12 form.

    log_alpha <- log_alpha - lr_alpha * exp(log_alpha) * (H - target_entropy)
    log_beta  <- log_beta  - lr_beta  * exp(log_beta)  * (KL - target_kl)
    """
    alpha = _coef_from_log(log_alpha, cfg)
    beta = _coef_from_log(log_beta, cfg)

    log_alpha = log_alpha - cfg.alpha_lr * alpha * (entropy - cfg.target_entropy)
    log_beta = log_beta - cfg.beta_lr * beta * (kl - cfg.target_kl)

    log_alpha = float(np.clip(log_alpha, cfg.min_log_multiplier, cfg.max_log_multiplier))
    log_beta = float(np.clip(log_beta, cfg.min_log_multiplier, cfg.max_log_multiplier))
    return log_alpha, log_beta


def save_checkpoint(
    path: Path,
    agent: Agent,
    actor_optimizer: optim.Optimizer,
    critic_optimizer: optim.Optimizer,
    cfg: REPPOConfig,
    history: Dict[str, list],
    update: int,
    ep_return: float,
    best_ep_return: float | None,
    log_alpha: float,
    log_beta: float,
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
            "actor_optimizer_state_dict": actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": critic_optimizer.state_dict(),
            "log_alpha": log_alpha,
            "log_beta": log_beta,
            "alpha": _coef_from_log(log_alpha, cfg),
            "beta": _coef_from_log(log_beta, cfg),
            "history": history,
        },
        path,
    )


def train(env: CartPole, cfg: REPPOConfig | None = None) -> Tuple[Agent, Dict[str, list]]:
    if cfg is None:
        cfg = REPPOConfig()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device(cfg.device)
    agent = Agent(env, cfg).to(device)

    actor_optimizer = optim.Adam(agent.actor.parameters(), lr=cfg.actor_lr)
    critic_optimizer = optim.Adam(agent.critic.parameters(), lr=cfg.critic_lr)

    log_alpha = _safe_log(cfg.initial_alpha)
    log_beta = _safe_log(cfg.initial_beta)

    history = make_empty_history()
    best_ep_return = float("-inf")
    checkpoint_dir = Path(cfg.checkpoint_dir) if cfg.checkpoint_dir else None

    for update in range(1, cfg.total_updates + 1):
        alpha = _coef_from_log(log_alpha, cfg)
        beta = _coef_from_log(log_beta, cfg)

        batch = collect_rollout_batch(env, agent, cfg)
        q_targets = compute_reppo_q_targets(batch, agent, cfg, alpha=alpha).detach()

        states_rollout = batch["states"]
        actions_rollout = batch["actions"]
        old_log_probs_rollout = batch["old_log_probs"]

        assert isinstance(states_rollout, torch.Tensor)
        assert isinstance(actions_rollout, torch.Tensor)
        assert isinstance(old_log_probs_rollout, torch.Tensor)

        states = states_rollout.reshape(-1, states_rollout.shape[-1])
        actions = actions_rollout.reshape(-1, actions_rollout.shape[-1])
        old_log_probs = old_log_probs_rollout.reshape(-1)
        q_targets_flat = q_targets.reshape(-1)

        loader = REPPODataloader(
            states=states,
            actions=actions,
            old_log_probs=old_log_probs,
            q_targets=q_targets_flat,
            batch_size=cfg.batch_size,
        )

        agent.train()

        actor_losses = []
        value_losses = []
        entropies = []
        approx_kls = []
        kl_clip_fracs = []
        q_means = []
        q_target_means = []

        for _epoch in range(cfg.ppo_epochs):
            for (
                batch_states,
                batch_actions,
                batch_old_log_probs,
                batch_q_targets,
            ) in loader:
                # 1) Q critic update on fixed on-policy TD-lambda targets.
                q_logits = agent.critic(batch_states, batch_actions)
                value_loss = hl_gauss_loss(q_logits, batch_q_targets, cfg)

                with torch.no_grad():
                    q_pred = q_value_from_logits(q_logits, cfg)

                critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(agent.critic.parameters(), cfg.max_grad_norm)
                critic_optimizer.step()

                # 2) Actor update through the frozen Q function. We freeze critic
                # parameters, but do not detach q_pi: actor still needs dQ/da.
                for p in agent.critic.parameters():
                    p.requires_grad_(False)

                new_actions, log_probs, entropy = agent.actor.rsample_action(batch_states)
                q_pi_logits = agent.critic(batch_states, new_actions)
                q_pi = q_value_from_logits(q_pi_logits, cfg)
                
                new_log_probs_old_actions, _ = agent.actor.evaluate_action(
                    batch_states,
                    batch_actions,
                )
                approx_kl = (batch_old_log_probs - new_log_probs_old_actions).mean()
                entropy_mean = entropy.mean()

                alpha = _coef_from_log(log_alpha, cfg)
                beta = _coef_from_log(log_beta, cfg)

                if approx_kl.detach().item() < cfg.target_kl:
                    actor_loss = (-q_pi + alpha * log_probs).mean()
                    kl_clip_fraction = torch.zeros((), dtype=torch.float32, device=device)
                else:
                    actor_loss = beta * approx_kl
                    kl_clip_fraction = torch.ones((), dtype=torch.float32, device=device)

                actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(agent.actor.parameters(), cfg.max_grad_norm)
                actor_optimizer.step()

                for p in agent.critic.parameters():
                    p.requires_grad_(True)

                # 3) Update alpha and beta in log space using current constraint errors.
                with torch.no_grad():
                    entropy_scalar = float(entropy_mean.item())
                    kl_scalar = float(approx_kl.item())
                    log_alpha, log_beta = _update_multipliers_paper_style(
                        log_alpha=log_alpha,
                        log_beta=log_beta,
                        entropy=entropy_scalar,
                        kl=kl_scalar,
                        cfg=cfg,
                    )

                actor_losses.append(float(actor_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy_mean.item()))
                approx_kls.append(float(approx_kl.item()))
                kl_clip_fracs.append(float(kl_clip_fraction.item()))
                q_means.append(float(q_pred.detach().mean().item()))
                q_target_means.append(float(batch_q_targets.detach().mean().item()))

        with torch.no_grad():
            q_after_update_logits = agent.critic(states, actions)
            q_after_update = q_value_from_logits(q_after_update_logits, cfg)
            ev = explained_variance(q_after_update, q_targets_flat)

        alpha = _coef_from_log(log_alpha, cfg)
        beta = _coef_from_log(log_beta, cfg)

        stats = {
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
            "clip_fraction": float(np.mean(kl_clip_fracs)) if kl_clip_fracs else 0.0,
            "explained_variance": ev,
            "alpha": alpha,
            "beta": beta,
            "log_alpha": log_alpha,
            "log_beta": log_beta,
            "q_mean": float(np.mean(q_means)) if q_means else 0.0,
            "q_target_mean": float(np.mean(q_target_means)) if q_target_means else 0.0,
        }

        append_history(history, update, batch, stats, cfg)

        ep_return = history["mean_episode_return"][-1]
        if checkpoint_dir is not None:
            if np.isfinite(ep_return) and ep_return > best_ep_return:
                best_ep_return = ep_return
                save_checkpoint(
                    checkpoint_dir / f"{cfg.checkpoint_prefix}_best.pt",
                    agent,
                    actor_optimizer,
                    critic_optimizer,
                    cfg,
                    history,
                    update,
                    ep_return,
                    best_ep_return,
                    log_alpha,
                    log_beta,
                )

            best_for_save = None if not np.isfinite(best_ep_return) else best_ep_return
            save_checkpoint(
                checkpoint_dir / f"{cfg.checkpoint_prefix}_last.pt",
                agent,
                actor_optimizer,
                critic_optimizer,
                cfg,
                history,
                update,
                ep_return,
                best_for_save,
                log_alpha,
                log_beta,
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
                f"q_loss={stats['value_loss']: .3f} | "
                f"kl={stats['approx_kl']: .5f} | "
                f"H={stats['entropy']: .3f} | "
                f"alpha={stats['alpha']: .5f} | "
                f"beta={stats['beta']: .5f}"
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
    cfg: REPPOConfig | None = None,
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
        reward_cfg = cfg if cfg is not None else REPPOConfig(x_limit=x_limit, device=device)
    else:
        state = env.reset()
        reward_cfg = cfg if cfg is not None else REPPOConfig(x_limit=x_limit, device=device)

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
