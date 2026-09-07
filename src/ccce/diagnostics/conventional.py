# Author: Basil M. Alzboun

"""
Conventional continual-learning diagnostics (Master Prompt V3, Section 26).
Definitions are fixed here, before any confirmation-stage experiment is run
(the module is imported, unmodified, by both discovery and confirmation
scripts) -- this file *is* the pre-registration of these definitions.

    BWT                   : backward transfer, Q_i(after task j) - Q_i(after task i)
    Gradient Interference : cosine similarity between the parameter-update
                             gradient of the target task and a probe gradient
                             recomputed on the prior task's data (negative
                             cosine similarity = interference)
    Representation Drift  : here, since our policy is linear, "representation"
                             is the induced action-mean mapping W_mu; drift is
                             the normalized Frobenius distance between W_mu
                             before and after the update (primary metric,
                             chosen before confirmation per Section 26)
    Plasticity            : ratio of gradient norm on a *fresh* probe task to
                             the gradient norm at initialization (a shrinking
                             ratio across tasks indicates loss of plasticity,
                             in the spirit of Dohare et al. 2024)
    Task Similarity       : cosine similarity between task target/goal
                             vectors, defined independently of CCCE
    Update Magnitude      : L2 norm of the flat parameter update
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ccce.learners.numpy_ppo import LinearGaussianPolicy


def backward_transfer(q_before_new_task: float, q_after_new_task: float) -> float:
    """BWT = Q_i(after learning task j) - Q_i(right after task i itself).
    Here both arguments are the same scalar competence measure evaluated
    at two different checkpoints; the caller supplies the correct pair."""
    return q_after_new_task - q_before_new_task


def gradient_interference(grad_target: np.ndarray, grad_probe: np.ndarray) -> float:
    """Cosine similarity between two gradient vectors. Negative values
    indicate the two tasks' gradients point in conflicting directions
    (interference); values near +1 indicate aligned (transferring)
    gradients."""
    num = float(np.dot(grad_target, grad_probe))
    denom = float(np.linalg.norm(grad_target) * np.linalg.norm(grad_probe)) + 1e-12
    return num / denom


def representation_drift(policy_before: LinearGaussianPolicy, policy_after: LinearGaussianPolicy) -> float:
    """Primary metric (fixed before confirmation, Section 26): normalized
    Frobenius distance between the action-mean weight matrices."""
    diff = policy_after.W_mu - policy_before.W_mu
    norm_before = np.linalg.norm(policy_before.W_mu) + 1e-8
    return float(np.linalg.norm(diff) / norm_before)


def plasticity_ratio(grad_norm_fresh_probe: float, grad_norm_at_init: float) -> float:
    """>1 = no apparent loss of plasticity relative to init; <1 = shrinking
    gradient responsiveness, consistent with loss-of-plasticity phenomena."""
    return float(grad_norm_fresh_probe / (grad_norm_at_init + 1e-12))


def task_similarity(goal_a: np.ndarray, goal_b: np.ndarray) -> float:
    """Defined independently of CCCE (Section 26): cosine similarity of
    the two tasks' goal vectors."""
    num = float(np.dot(goal_a, goal_b))
    denom = float(np.linalg.norm(goal_a) * np.linalg.norm(goal_b)) + 1e-12
    return num / denom


def update_magnitude(params_before: np.ndarray, params_after: np.ndarray) -> float:
    return float(np.linalg.norm(params_after - params_before))


@dataclass
class DiagnosticsBundle:
    bwt: float
    gradient_interference: float
    representation_drift: float
    plasticity: float
    task_similarity: float
    update_magnitude: float
