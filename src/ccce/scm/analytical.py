# Author: Basil M. Alzboun

"""
Level 1: Analytical structural causal model (Master Prompt V3, Section 7).

Purpose (verbatim from the spec): "exact causal ground truth... verify
intervention semantics, potential outcomes, CCCE oracle, estimator bias,
estimator variance, identification."

Design
------
We use a linear-Gaussian SCM over a parameter vector theta in R^d, which
plays the role of the learner's parameters theta_t in main_v2.tex's SCM
(H_t -> L_t -> U_t -> theta_{t+1} -> pi -> tau_i -> Y_i). A learning
protocol L_j = (T_j, A_j, B_j, eta_j, xi_j) pulls theta toward a
task-specific target vector w_j at an effective rate eta_j (a scalar
function of algorithm and budget: eta_j = gain(A_j) * (1 - exp(-B_j/tau))):

    theta_j = (1 - eta_j) * theta_{j-1} + eta_j * w_j + U_j,   U_j ~ N(0, sigma_U^2 I)

This is a linear-Gaussian recursion, so E[theta_j] is *exactly* computable
in closed form by iterating the deterministic mean recursion (no Monte
Carlo needed for the mean).

The competence outcome for competence i, under intervention do(C = c), is

    Y_i(theta, c) = a_i + b_i^T theta + c * (h_i^T theta) + xi_Y,   xi_Y ~ N(0, sigma_Y^2)

The b_i^T theta term is the "nominal" (c=0) sensitivity of competence i to
parameters; the c * (h_i^T theta) term is an *interaction* between the
intervention and the parameters. This interaction term is what makes a
"hidden counterfactual vulnerability" analytically constructible: choosing
b_i nearly orthogonal to (E[theta_T] - E[theta_C]) while h_i is *not*
orthogonal to it produces RC_i ~ 0 (nominal difference vanishes) while
CCCE_i(c) for c != 0 does not.

Because E[theta_j] is exact and Y_i is linear in theta given c, the
population CCCE is available in closed form:

    CCCE_i(c | H) = E[Y_i(theta_T, c)] - E[Y_i(theta_C, c)]
                  = (b_i + c * h_i)^T ( E[theta_T] - E[theta_C] )

This module provides that closed form (`exact_ccce`) *and* a Monte Carlo
sampler (`sample_theta_trajectory`, `sample_outcome`) used only by the
*estimator* (src/ccce/estimator), never by the oracle. The oracle
(src/ccce/oracle/analytical_oracle.py) calls `exact_ccce` directly and
therefore has zero Monte Carlo error by construction -- it is the
"ground truth" against which the estimator is checked, and it does not
share code with the estimator (Reviewer-2 attack #11 in main_v2.tex).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class LearningProtocolParams:
    """Analytical stand-in for L_j = (T_j, A_j, B_j, eta_j, xi_j).

    `target` is the task's pull direction w_j (stands in for T_j and, via
    its magnitude/direction, for A_j's inductive bias). `gain` and
    `budget` compose into the effective learning rate eta_j; keeping them
    separate (rather than a single scalar) preserves the manuscript's
    requirement that the algorithm and the budget are distinct components
    of the protocol, even though in this analytical model they enter
    through one scalar effective rate.
    """

    name: str
    target: np.ndarray  # w_j, shape (d,)
    gain: float  # algorithm-dependent learning gain, > 0
    budget: float  # training budget (e.g., normalized steps), >= 0
    tau: float = 1.0  # budget saturation time-constant

    @property
    def effective_rate(self) -> float:
        eta = self.gain * (1.0 - np.exp(-self.budget / self.tau))
        return float(np.clip(eta, 0.0, 1.0))


@dataclass
class NoUpdateProtocol:
    """L_0: the no-update protocol. eta = 0 by construction."""

    name: str = "L0_no_update"

    @property
    def effective_rate(self) -> float:
        return 0.0

    target: np.ndarray = field(default=None)  # unused when eta=0


@dataclass
class CompetenceLinearForm:
    """Linear-Gaussian competence outcome model for one competence i.

    Y_i(theta, c) = a_i + b_i^T theta + c * (h_i^T theta) + noise
    """

    competence_id: str
    a_i: float
    b_i: np.ndarray  # shape (d,), nominal sensitivity
    h_i: np.ndarray  # shape (d,), intervention-interaction sensitivity
    sigma_y: float
    valid_intervention_support: Sequence[float]  # e.g. (-2,-1,0,1,2)

    def is_supported(self, c: float) -> bool:
        return any(np.isclose(c, s) for s in self.valid_intervention_support)


@dataclass
class AnalyticalSCM:
    """The full Level-1 analytical SCM."""

    dim: int
    sigma_u: float
    theta0_mean: np.ndarray  # shape (d,), E[theta_0]
    competences: Dict[str, CompetenceLinearForm]

    # ---- Exact (closed-form) mean recursion -----------------------------
    def exact_theta_mean_trajectory(
        self, protocols: Sequence[LearningProtocolParams]
    ) -> List[np.ndarray]:
        """Return [E[theta_0], E[theta_1], ..., E[theta_J]] exactly."""
        means = [np.array(self.theta0_mean, dtype=float)]
        mean = means[0]
        for proto in protocols:
            eta = proto.effective_rate
            target = proto.target if proto.target is not None else mean
            mean = (1.0 - eta) * mean + eta * target
            means.append(mean.copy())
        return means

    def exact_ccce(
        self,
        competence_id: str,
        c: float,
        history_means: Sequence[np.ndarray],
        protocol_T: LearningProtocolParams,
        protocol_C: LearningProtocolParams,
    ) -> float:
        """Exact CCCE_i(c | H) = (b_i + c h_i)^T (E[theta_T] - E[theta_C]).

        `history_means` is [.., E[theta_{j-1}]] i.e. the mean trajectory
        up to (and including) the shared history H_{j-1}; both protocols
        are applied starting from history_means[-1].
        """
        comp = self.competences[competence_id]
        theta_prev = history_means[-1]

        eta_T, eta_C = protocol_T.effective_rate, protocol_C.effective_rate
        target_T = protocol_T.target if protocol_T.target is not None else theta_prev
        target_C = protocol_C.target if protocol_C.target is not None else theta_prev

        mean_T = (1.0 - eta_T) * theta_prev + eta_T * target_T
        mean_C = (1.0 - eta_C) * theta_prev + eta_C * target_C
        delta = mean_T - mean_C

        return float((comp.b_i + c * comp.h_i) @ delta)

    def exact_rc(
        self,
        competence_id: str,
        history_means: Sequence[np.ndarray],
        protocol_T: LearningProtocolParams,
    ) -> float:
        """Exact actual-retention-change RC_i = Q_i(L_T) - Q_i(L_0), at c=0."""
        no_update = NoUpdateProtocol()
        return self.exact_ccce(competence_id, 0.0, history_means, protocol_T, no_update)

    # ---- Monte Carlo sampling (estimator side only) ----------------------
    def sample_theta_trajectory(
        self,
        protocols: Sequence[LearningProtocolParams],
        theta0: np.ndarray,
        rng: np.random.Generator,
    ) -> List[np.ndarray]:
        """Draw one stochastic realization of the theta_j trajectory."""
        traj = [np.array(theta0, dtype=float)]
        theta = traj[0]
        for proto in protocols:
            eta = proto.effective_rate
            target = proto.target if proto.target is not None else theta
            noise = rng.normal(0.0, self.sigma_u, size=self.dim)
            theta = (1.0 - eta) * theta + eta * target + noise
            traj.append(theta.copy())
        return traj

    def sample_outcome(
        self, competence_id: str, theta: np.ndarray, c: float, rng: np.random.Generator
    ) -> float:
        comp = self.competences[competence_id]
        noise = rng.normal(0.0, comp.sigma_y)
        return float(comp.a_i + comp.b_i @ theta + c * (comp.h_i @ theta) + noise)
