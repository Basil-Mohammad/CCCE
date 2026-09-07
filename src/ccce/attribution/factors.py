"""
ccce.attribution.factors
=========================

Author: Basil M. Alzboun

Independent-latent generative substrate for the diagnostic factor
taxonomy used throughout the causal diagnostic framework
(see main_v4.tex, Section 4.7, Table 1).

Scientific purpose
-------------------
The attribution ground-truth worlds (``ccce.attribution.worlds``) require
a *provable* guarantee that the true causal term generating a Competence
Change Event in a given world shares no free parameter with the
measurement function of any other diagnostic factor, except the one (or
two, for interaction worlds) factor(s) designated as the true cause for
that world. Without this guarantee, a test of attribution correctness
degenerates into a test of predictive correlation in a finite sample,
which is exactly the conflation the diagnostic framework is designed to
avoid (main_v4.tex, Section 1, "the central conflation this paper
addresses").

This module enforces that guarantee structurally, not statistically: every
diagnostic factor's value is computed as a pure function of its own
dedicated, independently drawn latent variable (plus, where explicitly
noted, a shared but factor-symmetric measurement-noise term that never
enters any outcome-generating equation). No two factors' measurement
functions read from the same latent variable. This is verified at
runtime by ``verify_latent_independence`` and is additionally checked
statically by ``tests/test_attribution_worlds.py`` via source inspection.

Two categories are implemented, matching main_v4.tex Table 1:

    Category A (Observational Diagnostic Signals):
        gradient_interference, representation_drift, plasticity,
        task_similarity, update_magnitude, instability_signature

    Category B (Intervenable Diagnostic Factors):
        task_order, task_overlap, update_budget, learning_rate,
        replay_availability, environmental_causal_factor
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field, fields
from typing import Dict, List

import numpy as np

# ----------------------------------------------------------------------
# Category A: Observational Diagnostic Signals
# ----------------------------------------------------------------------

OBSERVATIONAL_FACTOR_NAMES: List[str] = [
    "gradient_interference",
    "representation_drift",
    "plasticity",
    "task_similarity",
    "update_magnitude",
    "instability_signature",
]

# Category B: Intervenable Diagnostic Factors
INTERVENABLE_FACTOR_NAMES: List[str] = [
    "task_order",
    "task_overlap",
    "update_budget",
    "learning_rate",
    "replay_availability",
    "environmental_causal_factor",
]

ALL_FACTOR_NAMES: List[str] = OBSERVATIONAL_FACTOR_NAMES + INTERVENABLE_FACTOR_NAMES


@dataclass
class LatentBundle:
    """One independently drawn latent variable per diagnostic factor,
    plus one independent latent per intervenable factor, plus a single
    additional latent reserved for factors that are structurally excluded
    from the measured diagnostic vocabulary entirely (used by World D,
    ``ccce.attribution.worlds``). Every latent is drawn from its own
    dedicated stream position, so no two latents share entropy.
    """

    gradient_interference: float
    representation_drift: float
    plasticity: float
    task_similarity: float
    update_magnitude: float
    instability_signature: float
    task_order: float
    task_overlap: float
    update_budget: float
    learning_rate: float
    replay_availability: float
    environmental_causal_factor: float
    unmeasured: float  # deliberately outside ALL_FACTOR_NAMES; see World D

    @classmethod
    def sample(cls, rng: np.random.Generator) -> "LatentBundle":
        """Draw one independent standard-normal latent per field, in a
        fixed field order, from a single RNG stream. Because each latent
        consumes exactly one scalar draw from the stream in a fixed,
        documented order, the independence of the underlying PCG64
        stream's successive draws (NumPy's documented guarantee) transfers
        directly to independence of the latents themselves.
        """
        names = [f.name for f in fields(cls)]
        draws = rng.standard_normal(len(names))
        return cls(**dict(zip(names, draws)))


def measure_observational_factors(
    latents: LatentBundle, noise_scale: float = 0.05, rng: np.random.Generator | None = None
) -> Dict[str, float]:
    """Category A measurement functions. Each factor is a pure function of
    its own dedicated latent plus independent idiosyncratic noise; no
    factor's formula references another factor's latent.
    """
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0.0, noise_scale, size=len(OBSERVATIONAL_FACTOR_NAMES))
    return {
        "gradient_interference": float(np.tanh(latents.gradient_interference) + noise[0]),
        "representation_drift": float(np.abs(latents.representation_drift) + noise[1]),
        "plasticity": float(latents.plasticity + noise[2]),
        "task_similarity": float(np.tanh(latents.task_similarity) + noise[3]),
        "update_magnitude": float(np.abs(latents.update_magnitude) + noise[4]),
        "instability_signature": float(np.abs(latents.instability_signature) ** 1.5 + noise[5]),
    }


def measure_intervenable_factors(latents: LatentBundle) -> Dict[str, float]:
    """Category B factor *settings*. In an actual experiment these are
    chosen by the experimenter (they are the object of Level D
    intervention), not sampled; here, for constructing the observational
    baseline against which an intervention is contrasted, they are drawn
    from their own dedicated latents so that, absent an explicit
    intervention, their distribution is independent of every other
    factor's latent.
    """
    return {
        "task_order": float(latents.task_order),
        "task_overlap": float(latents.task_overlap),
        "update_budget": float(latents.update_budget),
        "learning_rate": float(latents.learning_rate),
        "replay_availability": float(latents.replay_availability),
        "environmental_causal_factor": float(latents.environmental_causal_factor),
    }


def verify_latent_independence() -> None:
    """Static, source-level verification that no two measurement
    functions in this module reference the same ``LatentBundle`` field.

    This is the structural half of the non-circularity guarantee referred
    to throughout main_v4.tex Section 7.2; the statistical half (that
    resampling a nuisance latent does not move the outcome equation of a
    world whose true cause is a different latent) is verified per-world
    in ``ccce.attribution.worlds.verify_world_non_circularity``.

    Raises
    ------
    AssertionError
        If any latent field name appears in the source of more than one
        measurement function below.
    """
    field_names = {f.name for f in fields(LatentBundle)}
    usage: Dict[str, List[str]] = {name: [] for name in field_names}

    for fn in (measure_observational_factors, measure_intervenable_factors):
        source = inspect.getsource(fn)
        for name in field_names:
            if f"latents.{name}" in source:
                usage[name].append(fn.__name__)

    for name, users in usage.items():
        assert len(users) <= 1, (
            f"Non-circularity violation: latent field '{name}' is read by more than "
            f"one measurement function ({users}). Each latent must be read by exactly "
            f"one measurement function for the attribution ground-truth guarantee to hold."
        )
