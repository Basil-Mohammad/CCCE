# Author: Basil M. Alzboun

"""
Level 2: synthetic low-dimensional continuous-control environment
(Master Prompt V3, Section 8; main_v2.tex Section 5.2).

State:      s_t = [x_t, xdot_t, y_t, ydot_t]
Action:     a_t = [a_x, a_y] in [-1, 1]^2
Dynamics:   p_{t+1} = p_t + dt * v_t
            v_{t+1} = rho(z) * v_t + dt * a_t + eps_t
Causal factors z = [z1, z2, z3, z4]:
    z1: position-dynamics scaling (affects effective damping baseline)
    z2: velocity-dynamics damping rho(z)
    z3: obstacle density/response (collision penalty & termination)
    z4: recovery-dynamics perturbation magnitude (mid-episode push)

The agent observes only [x, xdot, y, ydot] (and the goal, folded into the
reward/competence computation) -- it never observes z directly, matching
the requirement that "the agent must NOT receive the ground-truth causal
graph" (Section 8).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class TaskSpec:
    """T_j = (g_j, w_j, C_j) per main_v2.tex Section 5.3 / Table 12."""

    task_id: str
    name: str
    goal: np.ndarray  # g_j, shape (2,)
    z1_position_dynamics: float = 1.0
    z2_velocity_damping: float = 0.98
    z3_obstacle_density: float = 0.0  # 0 = none, higher = denser
    z4_recovery_perturbation: float = 0.0  # magnitude of mid-episode push
    obstacles: Tuple[Tuple[float, float, float], ...] = field(default_factory=tuple)  # (x, y, radius)
    lambda_v: float = 0.01
    lambda_a: float = 0.01
    lambda_c: float = 1.0
    max_steps: int = 60
    success_radius: float = 0.3


def default_task_set() -> Dict[str, TaskSpec]:
    """The six tasks specified in main_v2.tex Table 12, with concrete
    numeric obstacle/perturbation parameters chosen for this
    implementation (documented here since the manuscript left exact
    numeric obstacle placement unspecified by design -- Section 86)."""
    return {
        "T1": TaskSpec(
            "T1", "Reach-Center", goal=np.array([0.0, 0.0]),
            z1_position_dynamics=1.0, z2_velocity_damping=0.98, z3_obstacle_density=0.0,
            z4_recovery_perturbation=0.0, obstacles=(),
        ),
        "T2": TaskSpec(
            "T2", "Reach-Offset-A", goal=np.array([3.0, 3.0]),
            z1_position_dynamics=1.0, z2_velocity_damping=0.98, z3_obstacle_density=1.0,
            z4_recovery_perturbation=0.0, obstacles=((1.5, 1.5, 0.3),),
        ),
        "T3": TaskSpec(
            "T3", "Obstacle-Weave", goal=np.array([5.0, 0.0]),
            z1_position_dynamics=1.0, z2_velocity_damping=0.97, z3_obstacle_density=2.0,
            z4_recovery_perturbation=0.0,
            obstacles=((1.5, 0.5, 0.3), (3.0, -0.5, 0.3), (4.0, 0.5, 0.3)),
        ),
        "T4": TaskSpec(
            "T4", "Recovery-Push", goal=np.array([0.0, 5.0]),
            z1_position_dynamics=1.0, z2_velocity_damping=0.98, z3_obstacle_density=0.0,
            z4_recovery_perturbation=0.6, obstacles=(),
        ),
        "T5": TaskSpec(
            "T5", "Combined-Hazard", goal=np.array([4.0, -4.0]),
            z1_position_dynamics=1.0, z2_velocity_damping=0.96, z3_obstacle_density=2.0,
            z4_recovery_perturbation=0.8,
            obstacles=((2.0, -2.0, 0.3), (3.0, -3.5, 0.3)),
        ),
        "T6": TaskSpec(
            "T6", "Precision-Dock", goal=np.array([1.0, 1.0]),
            z1_position_dynamics=1.0, z2_velocity_damping=0.98, z3_obstacle_density=0.0,
            z4_recovery_perturbation=0.0, obstacles=(), success_radius=0.15,
        ),
    }


@dataclass
class InterventionSpec:
    """A do(C=c) intervention applied at evaluation time (Section 18)."""

    causal_variable: str  # "z1" | "z3" | "z4"
    value: float


class SyntheticPointMassEnv:
    """Gymnasium-style API (reset/step) without depending on the
    `gymnasium` package, since it is not installed in this sandbox
    (documented deviation; the API surface is intentionally
    Gymnasium-compatible for a future drop-in swap)."""

    def __init__(self, task: TaskSpec, dt: float = 0.1, intervention: Optional[InterventionSpec] = None):
        self.task = task
        self.dt = dt
        self.intervention = intervention
        self._z1 = task.z1_position_dynamics
        self._z2 = task.z2_velocity_damping
        self._z3 = task.z3_obstacle_density
        self._z4 = task.z4_recovery_perturbation
        if intervention is not None:
            if intervention.causal_variable == "z1":
                self._z1 = intervention.value
            elif intervention.causal_variable == "z2":
                self._z2 = intervention.value
            elif intervention.causal_variable == "z3":
                # Additive obstacle-density scaling: intervention value
                # scales the *effective* number of active obstacles.
                self._z3_scale = intervention.value
            elif intervention.causal_variable == "z4":
                self._z4 = intervention.value
        self._z3_scale = getattr(self, "_z3_scale", 1.0)

        self.pos = np.zeros(2)
        self.vel = np.zeros(2)
        self.t = 0
        self.collided = False
        self._pushed = False

    def reset(self, rng: np.random.Generator) -> np.ndarray:
        self.pos = rng.normal(0, 0.1, size=2)
        self.vel = np.zeros(2)
        self.t = 0
        self.collided = False
        self._pushed = False
        return self._obs()

    def _obs(self) -> np.ndarray:
        return np.array([self.pos[0], self.vel[0], self.pos[1], self.vel[1]], dtype=float)

    def _active_obstacles(self):
        n_active = int(round(len(self.task.obstacles) * self._z3_scale))
        n_active = max(0, min(n_active, len(self.task.obstacles)))
        return self.task.obstacles[:n_active]

    def step(self, action: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, float, bool, Dict]:
        action = np.clip(action, -1.0, 1.0)

        # Mid-episode recovery perturbation (z4): a one-time velocity kick
        # at the episode midpoint, if this task/intervention has z4 > 0.
        if not self._pushed and self.t == self.task.max_steps // 2 and self._z4 > 0:
            push_dir = rng.normal(size=2)
            push_dir = push_dir / (np.linalg.norm(push_dir) + 1e-8)
            self.vel += self._z4 * push_dir
            self._pushed = True

        noise = rng.normal(0, 0.01, size=2)
        self.pos = self.pos + self.dt * self.vel * self._z1
        self.vel = self._z2 * self.vel + self.dt * action + noise

        collision = False
        for (ox, oy, r) in self._active_obstacles():
            if np.linalg.norm(self.pos - np.array([ox, oy])) < r:
                collision = True
                self.collided = True

        self.t += 1
        dist = np.linalg.norm(self.pos - self.task.goal)
        reward = (
            -(dist ** 2)
            - self.task.lambda_v * float(np.sum(self.vel ** 2))
            - self.task.lambda_a * float(np.sum(action ** 2))
            - self.task.lambda_c * float(collision)
        )
        success = dist < self.task.success_radius
        done = self.t >= self.task.max_steps or success
        info = dict(collision=collision, success=success, distance=dist)
        return self._obs(), float(reward), bool(done), info

    def rollout_metadata(self) -> Dict:
        return dict(z1=self._z1, z2=self._z2, z3_scale=self._z3_scale, z4=self._z4)
