# Author: Basil M. Alzboun

"""
DOCUMENTED DEVIATION from Master Prompt V3 Section 12
=======================================================
Section 12 specifies Stable-Baselines3 (a PyTorch-based library) as the
preferred PPO implementation. This sandbox has ~1 CPU core, no GPU, and
insufficient free disk space to install PyTorch (verified: `pip install
torch` fails with "No space left on device" after partially writing
~7 GB of wheel files). Per Master Prompt V3 Section 0 ("if you identify a
genuine... experimentally impossible specification, STOP that component,
explain the issue, and propose the smallest scientifically justified
correction") and Section 86, the smallest correction that preserves the
scientific content (a real, gradient-based, clipped-surrogate policy
optimizer, trained online, with all of PPO's defining hyperparameters) is:

    A from-scratch NumPy implementation of PPO with a linear-Gaussian
    policy and a linear value baseline, trained with the standard
    clipped-surrogate objective and GAE advantages, using finite-difference-free
    analytic gradients (the policy is linear + Gaussian, so the score-function
    gradient of the clipped surrogate has a closed form; no autodiff library
    is required).

This is a real optimizer that really trains via real gradient steps -- it
is not a mock, a lookup table, or a scripted policy. It is a strictly
smaller-capacity substitute for a deep PPO agent, which is an honest,
documented, and load-bearing limitation of the reported synthetic-environment
results (see docs/DEVIATIONS.md and the Scientific Audit Report).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PPOHyperparameters:
    """Recorded verbatim per Section 12's requirement list."""

    learning_rate: float = 0.05
    gamma: float = 0.97
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.001
    value_coef: float = 0.5
    batch_size: int = 32
    rollout_length: int = 60  # one episode for this environment's max_steps
    n_epochs: int = 4
    total_env_steps: int = 6000  # REDUCED from main_v2.tex's 2e6 -- see docs/DEVIATIONS.md
    network_architecture: str = "linear_gaussian_policy+linear_value_baseline"
    optimizer: str = "vanilla_gradient_ascent_on_clipped_surrogate"
    gradient_clip_norm: float = 5.0
    log_std_init: float = -0.5


@dataclass
class LinearGaussianPolicy:
    """pi(a|s) = N(W_mu @ s + b_mu, diag(exp(log_std)^2)).

    Parameters are the flat vector theta = [W_mu.flatten(), b_mu, log_std,
    W_v.flatten(), b_v] -- this flat vector plays the role of the
    learner's parameters theta_t in the paper's SCM, and is exactly what
    diagnostics.py measures (update magnitude, gradient interference,
    representation drift proxies, etc.).
    """

    obs_dim: int
    act_dim: int
    W_mu: np.ndarray = field(default=None)
    b_mu: np.ndarray = field(default=None)
    log_std: np.ndarray = field(default=None)
    W_v: np.ndarray = field(default=None)
    b_v: np.ndarray = field(default=None)

    def __post_init__(self) -> None:
        """Allow constructing an "empty" policy (all None) that will
        immediately have `set_flat_params` called on it -- e.g. when
        making a structurally-typed copy target. Allocate zero-shaped
        arrays so `.size`/`.shape`/`.reshape` are always well-defined."""
        if self.W_mu is None:
            self.W_mu = np.zeros((self.act_dim, self.obs_dim))
        if self.b_mu is None:
            self.b_mu = np.zeros(self.act_dim)
        if self.log_std is None:
            self.log_std = np.zeros(self.act_dim)
        if self.W_v is None:
            self.W_v = np.zeros((1, self.obs_dim))
        if self.b_v is None:
            self.b_v = np.zeros(1)

    @classmethod
    def initialize(cls, obs_dim: int, act_dim: int, rng: np.random.Generator, log_std_init: float) -> "LinearGaussianPolicy":
        return cls(
            obs_dim=obs_dim,
            act_dim=act_dim,
            W_mu=rng.normal(0, 0.1, size=(act_dim, obs_dim)),
            b_mu=np.zeros(act_dim),
            log_std=np.full(act_dim, log_std_init),
            W_v=rng.normal(0, 0.1, size=(1, obs_dim)),
            b_v=np.zeros(1),
        )

    def act(self, obs: np.ndarray, rng: np.random.Generator, deterministic: bool = False) -> Tuple[np.ndarray, float]:
        mean = self.W_mu @ obs + self.b_mu
        std = np.exp(self.log_std)
        if deterministic:
            action = mean
        else:
            action = mean + std * rng.normal(size=self.act_dim)
        logp = -0.5 * np.sum(((action - mean) / std) ** 2 + 2 * self.log_std + np.log(2 * np.pi))
        return action, float(logp)

    def logp(self, obs: np.ndarray, action: np.ndarray) -> float:
        mean = self.W_mu @ obs + self.b_mu
        std = np.exp(self.log_std)
        return float(-0.5 * np.sum(((action - mean) / std) ** 2 + 2 * self.log_std + np.log(2 * np.pi)))

    def value(self, obs: np.ndarray) -> float:
        return float((self.W_v @ obs + self.b_v).item())

    def flat_params(self) -> np.ndarray:
        return np.concatenate([self.W_mu.ravel(), self.b_mu, self.log_std, self.W_v.ravel(), self.b_v])

    def set_flat_params(self, flat: np.ndarray) -> None:
        i = 0
        n = self.W_mu.size
        self.W_mu = flat[i:i + n].reshape(self.W_mu.shape); i += n
        n = self.b_mu.size
        self.b_mu = flat[i:i + n]; i += n
        n = self.log_std.size
        self.log_std = flat[i:i + n]; i += n
        n = self.W_v.size
        self.W_v = flat[i:i + n].reshape(self.W_v.shape); i += n
        n = self.b_v.size
        self.b_v = flat[i:i + n]; i += n

    def copy(self) -> "LinearGaussianPolicy":
        new = LinearGaussianPolicy(self.obs_dim, self.act_dim)
        new.set_flat_params(self.flat_params().copy())
        return new


def _compute_gae(rewards: List[float], values: List[float], last_value: float, gamma: float, lam: float) -> Tuple[np.ndarray, np.ndarray]:
    values = values + [last_value]
    advantages = np.zeros(len(rewards))
    gae = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = advantages + np.array(values[:-1])
    return advantages, returns


def collect_rollout(env_factory: Callable, policy: LinearGaussianPolicy, rng: np.random.Generator, rollout_length: int):
    env = env_factory()
    obs = env.reset(rng)
    obs_buf, act_buf, logp_buf, rew_buf, val_buf = [], [], [], [], []
    infos = []
    for _ in range(rollout_length):
        action, logp = policy.act(obs, rng)
        value = policy.value(obs)
        next_obs, reward, done, info = env.step(action, rng)
        obs_buf.append(obs); act_buf.append(action); logp_buf.append(logp)
        rew_buf.append(reward); val_buf.append(value)
        infos.append(info)
        obs = next_obs
        if done:
            obs = env.reset(rng)
    last_value = policy.value(obs)
    return obs_buf, act_buf, logp_buf, rew_buf, val_buf, last_value, infos


def ppo_update(
    policy: LinearGaussianPolicy,
    obs_buf, act_buf, logp_old_buf, advantages, returns,
    hp: PPOHyperparameters,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """One PPO clipped-surrogate update via analytic score-function
    gradients on the linear-Gaussian policy (no autodiff needed).

    `rng` controls epoch-shuffling order. If None, a fresh unseeded
    generator is used (non-deterministic -- only acceptable for ad hoc
    exploration, never for a run whose reproducibility is claimed;
    `train_policy` below always passes its own seeded `rng`)."""
    if rng is None:
        rng = np.random.default_rng()
    n = len(obs_buf)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    total_grad_norm = 0.0
    total_ratio_clipped_frac = 0.0

    for _ in range(hp.n_epochs):
        idx = rng.permutation(n)
        grad_flat = np.zeros_like(policy.flat_params())
        clipped_count = 0
        for i in idx:
            obs = obs_buf[i]; action = act_buf[i]
            logp_old = logp_old_buf[i]; adv = advantages[i]; ret = returns[i]

            mean = policy.W_mu @ obs + policy.b_mu
            std = np.exp(policy.log_std)
            logp_new = policy.logp(obs, action)
            ratio = np.exp(logp_new - logp_old)

            unclipped = ratio * adv
            clipped_ratio = np.clip(ratio, 1 - hp.clip_range, 1 + hp.clip_range)
            clipped = clipped_ratio * adv
            use_clipped = clipped < unclipped
            if use_clipped:
                clipped_count += 1
                surrogate_grad_scale = 0.0 if (ratio < 1 - hp.clip_range or ratio > 1 + hp.clip_range) else ratio * adv
            else:
                surrogate_grad_scale = ratio * adv

            # d logp / d(W_mu, b_mu): standard Gaussian score function
            d_action = (action - mean) / (std ** 2)
            dW_mu = np.outer(d_action, obs) * surrogate_grad_scale
            db_mu = d_action * surrogate_grad_scale
            d_log_std = (((action - mean) ** 2) / (std ** 2) - 1.0) * surrogate_grad_scale

            value_pred = policy.value(obs)
            value_err = (value_pred - ret)
            dW_v = -hp.value_coef * value_err * obs.reshape(1, -1)
            db_v = np.array([-hp.value_coef * value_err])

            entropy_grad_log_std = hp.entropy_coef * np.ones_like(policy.log_std)

            g = np.concatenate([
                dW_mu.ravel(), db_mu, d_log_std + entropy_grad_log_std,
                dW_v.ravel(), db_v,
            ])
            grad_flat += g

        grad_flat /= n
        norm = np.linalg.norm(grad_flat)
        if norm > hp.gradient_clip_norm:
            grad_flat = grad_flat * (hp.gradient_clip_norm / norm)
        total_grad_norm += norm
        total_ratio_clipped_frac += clipped_count / n

        new_params = policy.flat_params() + hp.learning_rate * grad_flat
        policy.set_flat_params(new_params)

    return dict(
        mean_grad_norm=total_grad_norm / hp.n_epochs,
        mean_clip_fraction=total_ratio_clipped_frac / hp.n_epochs,
        last_grad_flat=grad_flat,
    )


def train_policy(
    env_factory: Callable,
    policy: LinearGaussianPolicy,
    rng: np.random.Generator,
    hp: PPOHyperparameters,
) -> Tuple[LinearGaussianPolicy, Dict]:
    """Run PPO for hp.total_env_steps environment steps. Returns the
    (mutated in-place) policy and a training-curve log."""
    n_updates = max(1, hp.total_env_steps // hp.rollout_length)
    returns_log = []
    last_grad = None
    for _ in range(n_updates):
        obs_buf, act_buf, logp_buf, rew_buf, val_buf, last_value, infos = collect_rollout(
            env_factory, policy, rng, hp.rollout_length
        )
        advantages, returns = _compute_gae(rew_buf, val_buf, last_value, hp.gamma, hp.gae_lambda)
        stats = ppo_update(policy, obs_buf, act_buf, logp_buf, advantages, returns, hp, rng=rng)
        last_grad = stats["last_grad_flat"]
        returns_log.append(float(np.sum(rew_buf)))
    return policy, dict(returns_log=returns_log, final_grad=last_grad, n_updates=n_updates)
