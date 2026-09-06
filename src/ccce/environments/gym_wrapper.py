"""
Gymnasium-compatible wrapper around SyntheticPointMassEnv (Master Prompt
V3 Section 9's "standard, well-established implementations" requirement
for the *learning algorithm* side -- Stable-Baselines3 -- applied to our
own custom environment; distinct from Section 9's separate requirement
to *also* validate against standard benchmark environments like
HalfCheetah, which is handled in `environments/mujoco_validation.py`).

This wrapper makes NO change to the underlying dynamics in
`synthetic_env.py`; it only adapts the reset()/step() API and exposes
`observation_space`/`action_space`, as required by `gymnasium.Env`.
"""
from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ccce.environments.synthetic_env import InterventionSpec, SyntheticPointMassEnv, TaskSpec


class GymSyntheticPointMassEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, task: TaskSpec, intervention: Optional[InterventionSpec] = None):
        super().__init__()
        self._inner_factory = lambda: SyntheticPointMassEnv(task, intervention=intervention)
        self._inner = self._inner_factory()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self._np_random_local = np.random.default_rng()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._np_random_local = np.random.default_rng(seed)
        self._inner = self._inner_factory()
        obs = self._inner.reset(self._np_random_local)
        return obs.astype(np.float32), {}

    def step(self, action):
        obs, reward, done, info = self._inner.step(np.asarray(action, dtype=np.float64), self._np_random_local)
        terminated = bool(done)
        truncated = False
        return obs.astype(np.float32), float(reward), terminated, truncated, info

    def render(self):
        return None

    def close(self):
        pass
