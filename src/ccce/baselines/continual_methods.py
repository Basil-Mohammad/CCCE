"""
Continual-learning baselines (Master Prompt V3, Section 14).

Implements, at reduced scale (see docs/DEVIATIONS.md for the NumPy-PPO
substitution this all sits on top of):

    naive       : sequential fine-tuning, no protection at all
    replay      : interleave a buffer of transitions from previous tasks
                  into each training rollout batch (Lin 1992 / Lopez-Paz &
                  Ranzato 2017 in spirit)
    ewc         : quadratic penalty toward previous-task parameters,
                  weighted by a diagonal Fisher-information proxy
                  (Kirkpatrick et al. 2017)
    distillation: soft-target regression of the new policy's action means
                  toward the old policy's action means on old-task states
                  (Li & Hoiem 2017 in spirit)
    upgd        : per-parameter utility-gated perturbation (Elsayed &
                  Mahmood 2024 in spirit): protect high-utility weights,
                  perturb low-utility ones

These are documented, simplified, RL-adapted re-implementations of the
published ideas, not the original authors' code -- Section 14 requires
"verified independently" and "not deliberately weak"; the adaptations
required for this linear-Gaussian-policy RL setting are documented per
function below.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from ccce.learners.numpy_ppo import LinearGaussianPolicy, PPOHyperparameters, collect_rollout, ppo_update, _compute_gae


@dataclass
class EWCState:
    """Fisher-diagonal proxy + anchor parameters, accumulated across tasks."""

    anchor_params: Optional[np.ndarray] = None
    fisher_diag: Optional[np.ndarray] = None
    lambda_ewc: float = 400.0

    def penalty_grad(self, current_params: np.ndarray) -> np.ndarray:
        if self.anchor_params is None or self.fisher_diag is None:
            return np.zeros_like(current_params)
        return -self.lambda_ewc * self.fisher_diag * (current_params - self.anchor_params)

    def update_after_task(self, policy: LinearGaussianPolicy, grad_samples: List[np.ndarray]) -> None:
        """Approximate the Fisher diagonal as the empirical variance of
        per-sample gradients on this task (a standard, documented
        diagonal-Fisher proxy)."""
        grads = np.stack(grad_samples, axis=0)
        new_fisher = np.mean(grads ** 2, axis=0)
        if self.fisher_diag is None:
            self.fisher_diag = new_fisher
        else:
            self.fisher_diag = self.fisher_diag + new_fisher  # accumulate across tasks, standard online-EWC choice
        self.anchor_params = policy.flat_params().copy()


@dataclass
class ReplayBuffer:
    capacity: int = 2000
    obs: List[np.ndarray] = field(default_factory=list)
    act: List[np.ndarray] = field(default_factory=list)
    logp: List[float] = field(default_factory=list)
    adv: List[float] = field(default_factory=list)
    ret: List[float] = field(default_factory=list)

    def add(self, obs, act, logp, adv, ret, rng: Optional[np.random.Generator] = None):
        if rng is None:
            rng = np.random.default_rng()  # documented fallback; train_with_replay always passes its seeded rng
        for o, a, lp, ad, r in zip(obs, act, logp, adv, ret):
            if len(self.obs) >= self.capacity:
                idx = rng.integers(0, self.capacity)
                self.obs[idx], self.act[idx], self.logp[idx] = o, a, lp
                self.adv[idx], self.ret[idx] = ad, r
            else:
                self.obs.append(o); self.act.append(a); self.logp.append(lp)
                self.adv.append(ad); self.ret.append(r)

    def sample(self, n: int, rng: np.random.Generator):
        if not self.obs:
            return [], [], [], np.array([]), np.array([])
        idx = rng.integers(0, len(self.obs), size=min(n, len(self.obs)))
        return (
            [self.obs[i] for i in idx], [self.act[i] for i in idx], [self.logp[i] for i in idx],
            np.array([self.adv[i] for i in idx]), np.array([self.ret[i] for i in idx]),
        )


def train_naive(env_factory, policy, rng, hp: PPOHyperparameters) -> Dict:
    """Baseline 1: naive sequential fine-tuning (no protection at all)."""
    from ccce.learners.numpy_ppo import train_policy
    _, log = train_policy(env_factory, policy, rng, hp)
    return log


def train_with_replay(env_factory, policy, rng, hp: PPOHyperparameters, buffer: ReplayBuffer) -> Dict:
    """Baseline 2: interleave replayed transitions from `buffer` into every
    update (Lin 1992-style experience replay, adapted for on-policy PPO by
    mixing replayed advantages/returns into the same gradient batch)."""
    n_updates = max(1, hp.total_env_steps // hp.rollout_length)
    returns_log = []
    last_grad = None
    for _ in range(n_updates):
        obs_buf, act_buf, logp_buf, rew_buf, val_buf, last_value, infos = collect_rollout(
            env_factory, policy, rng, hp.rollout_length
        )
        advantages, returns = _compute_gae(rew_buf, val_buf, last_value, hp.gamma, hp.gae_lambda)

        r_obs, r_act, r_logp, r_adv, r_ret = buffer.sample(hp.rollout_length // 2, rng)
        if r_obs:
            obs_buf = obs_buf + r_obs
            act_buf = act_buf + r_act
            logp_buf = logp_buf + r_logp
            advantages = np.concatenate([advantages, r_adv])
            returns = np.concatenate([returns, r_ret])

        stats = ppo_update(policy, obs_buf, act_buf, logp_buf, advantages, returns, hp, rng=rng)
        buffer.add(obs_buf[: hp.rollout_length], act_buf[: hp.rollout_length],
                   logp_buf[: hp.rollout_length], advantages[: hp.rollout_length].tolist(),
                   returns[: hp.rollout_length].tolist(), rng=rng)
        last_grad = stats["last_grad_flat"]
        returns_log.append(float(np.sum(rew_buf)))
    return dict(returns_log=returns_log, final_grad=last_grad, n_updates=n_updates)


def train_with_ewc(env_factory, policy, rng, hp: PPOHyperparameters, ewc_state: EWCState) -> Dict:
    """Baseline 3: EWC-penalized training (Kirkpatrick et al. 2017)."""
    n_updates = max(1, hp.total_env_steps // hp.rollout_length)
    returns_log = []
    last_grad = None
    per_sample_grads_for_fisher = []
    for _ in range(n_updates):
        obs_buf, act_buf, logp_buf, rew_buf, val_buf, last_value, infos = collect_rollout(
            env_factory, policy, rng, hp.rollout_length
        )
        advantages, returns = _compute_gae(rew_buf, val_buf, last_value, hp.gamma, hp.gae_lambda)
        stats = ppo_update(policy, obs_buf, act_buf, logp_buf, advantages, returns, hp, rng=rng)

        # Apply the EWC penalty as an additional parameter-space correction
        # after the PPO step (documented composition of the two updates).
        penalty_grad = ewc_state.penalty_grad(policy.flat_params())
        policy.set_flat_params(policy.flat_params() + hp.learning_rate * penalty_grad)

        last_grad = stats["last_grad_flat"]
        per_sample_grads_for_fisher.append(stats["last_grad_flat"])
        returns_log.append(float(np.sum(rew_buf)))

    ewc_state.update_after_task(policy, per_sample_grads_for_fisher)
    return dict(returns_log=returns_log, final_grad=last_grad, n_updates=n_updates)


def train_with_distillation(env_factory, policy, teacher_policy, rng, hp: PPOHyperparameters, distill_coef: float = 0.1) -> Dict:
    """Baseline 4: policy distillation toward the pre-update ("teacher")
    policy's action means, regularizing drift (Li & Hoiem 2017 in spirit,
    adapted from classification logits to continuous action means)."""
    n_updates = max(1, hp.total_env_steps // hp.rollout_length)
    returns_log = []
    last_grad = None
    for _ in range(n_updates):
        obs_buf, act_buf, logp_buf, rew_buf, val_buf, last_value, infos = collect_rollout(
            env_factory, policy, rng, hp.rollout_length
        )
        advantages, returns = _compute_gae(rew_buf, val_buf, last_value, hp.gamma, hp.gae_lambda)
        stats = ppo_update(policy, obs_buf, act_buf, logp_buf, advantages, returns, hp, rng=rng)

        # Distillation correction: pull W_mu, b_mu toward the teacher's.
        flat = policy.flat_params()
        teacher_flat = teacher_policy.flat_params()
        n_mu = policy.W_mu.size + policy.b_mu.size
        flat[:n_mu] = flat[:n_mu] + distill_coef * (teacher_flat[:n_mu] - flat[:n_mu])
        policy.set_flat_params(flat)

        last_grad = stats["last_grad_flat"]
        returns_log.append(float(np.sum(rew_buf)))
    return dict(returns_log=returns_log, final_grad=last_grad, n_updates=n_updates)


def train_with_upgd(env_factory, policy, rng, hp: PPOHyperparameters, utility_ema: Optional[np.ndarray] = None, decay: float = 0.9, perturb_scale: float = 0.01) -> Dict:
    """Baseline 5: Utility-gated Perturbed Gradient Descent (Elsayed &
    Mahmood 2024 in spirit): parameters with high running utility (here,
    approximated as |parameter * gradient|, a standard saliency proxy) are
    updated normally and protected from perturbation; low-utility
    parameters are perturbed to preserve plasticity."""
    n_updates = max(1, hp.total_env_steps // hp.rollout_length)
    returns_log = []
    last_grad = None
    flat = policy.flat_params()
    if utility_ema is None:
        utility_ema = np.zeros_like(flat)

    for _ in range(n_updates):
        obs_buf, act_buf, logp_buf, rew_buf, val_buf, last_value, infos = collect_rollout(
            env_factory, policy, rng, hp.rollout_length
        )
        advantages, returns = _compute_gae(rew_buf, val_buf, last_value, hp.gamma, hp.gae_lambda)
        stats = ppo_update(policy, obs_buf, act_buf, logp_buf, advantages, returns, hp, rng=rng)
        grad = stats["last_grad_flat"]

        flat = policy.flat_params()
        utility = np.abs(flat * grad)
        utility_ema = decay * utility_ema + (1 - decay) * utility
        # Normalize to [0, 1] utility scores this step.
        u_norm = (utility_ema - utility_ema.min()) / (utility_ema.max() - utility_ema.min() + 1e-8)
        protect_factor = u_norm  # 1 = fully protected, 0 = fully perturbable
        perturb = rng.normal(0, perturb_scale, size=flat.shape) * (1 - protect_factor)
        policy.set_flat_params(flat + perturb)

        last_grad = grad
        returns_log.append(float(np.sum(rew_buf)))
    return dict(returns_log=returns_log, final_grad=last_grad, n_updates=n_updates, utility_ema=utility_ema)
