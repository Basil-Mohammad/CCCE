# Author: Basil M. Alzboun

"""
Flat-parameter utilities for Stable-Baselines3 policies (needed to
implement EWC directly on top of SB3, since SB3 has no built-in
continual-learning methods). Mirrors the flat-parameter interface of
`ccce.learners.numpy_ppo.LinearGaussianPolicy` so the same EWC logic
(Fisher-diagonal proxy via parameter-delta-squared, normalized penalty
with gradient clipping) can be applied to a real torch policy network.
"""
from __future__ import annotations

import numpy as np
import torch


def get_flat_params(model) -> np.ndarray:
    with torch.no_grad():
        return torch.cat([p.data.flatten() for p in model.policy.parameters()]).cpu().numpy()


def set_flat_params(model, flat: np.ndarray) -> None:
    flat_t = torch.as_tensor(flat, dtype=torch.float32)
    i = 0
    with torch.no_grad():
        for p in model.policy.parameters():
            n = p.numel()
            p.data.copy_(flat_t[i:i + n].view(p.shape))
            i += n


def apply_ewc_penalty(
    model, anchor_params: np.ndarray, fisher_diag: np.ndarray,
    lambda_ewc: float, max_penalty_norm: float = 5.0,
) -> None:
    """Same normalized penalty formula as
    `ccce.baselines.continual_methods.EWCState.penalty_grad`, applied
    directly to the real torch policy's parameters, with an ADDITIONAL
    safeguard not needed in the ~19-parameter NumPy-policy case: the step
    is also capped at half the current anchor-distance, so it can never
    overshoot past the anchor regardless of the parameter-space's scale.

    This was found necessary empirically: `max_penalty_norm=5.0` was
    calibrated for the small NumPy linear-Gaussian policy (~19
    parameters, where post-burst parameter movement and the anchor
    distance are both O(1-10)). Applied unchanged to a real SB3 neural
    network (~9,000+ parameters, where a single short training burst
    moves the flat parameter vector by only ~0.1-0.2 in L2 norm), the
    fixed 5.0 clip is drastically larger than the actual gap, causing the
    "corrective" step to blow straight through the anchor and land
    further away than before the correction (observed: gap 0.166 -> 4.83
    after a naive un-adapted penalty step). The distance-relative cap
    below prevents this class of overshoot in any parameter-space scale.
    """
    current = get_flat_params(model)
    gap = current - anchor_params
    gap_norm = np.linalg.norm(gap)
    fisher_max = np.max(fisher_diag) + 1e-12
    fisher_normalized = fisher_diag / fisher_max
    raw_grad = -lambda_ewc * fisher_normalized * gap
    norm = np.linalg.norm(raw_grad)
    effective_cap = min(max_penalty_norm, 0.5 * gap_norm)
    if norm > effective_cap and norm > 0:
        raw_grad = raw_grad * (effective_cap / norm)
    set_flat_params(model, current + raw_grad)
