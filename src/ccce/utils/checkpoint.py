"""
Checkpointing utilities (Master Prompt V3, Section 6).

A checkpoint directory contains, at minimum:
    model_state.pkl        (policy parameters; .pt naming from the spec
                             assumed a torch backend, which is unavailable
                             in this sandbox -- see docs/DEVIATIONS.md)
    optimizer_state.pkl
    environment_state.pkl
    rng_state.pkl
    metadata.json
    metrics.json

Every task-boundary checkpoint (T1_end, T2_end, ...) and every periodic
step checkpoint is written through `save_checkpoint`, and a run is resumed
through `load_checkpoint`. Resumption restores the RNG bundle exactly, so
the *distribution* of subsequent randomness is reproduced even though
bitwise trajectory identity additionally depends on the determinism of the
learner update itself (documented per-component).
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from ccce.utils.rng import RNGBundle


@dataclass
class CheckpointPaths:
    root: Path

    @property
    def model_state(self) -> Path:
        return self.root / "model_state.pkl"

    @property
    def optimizer_state(self) -> Path:
        return self.root / "optimizer_state.pkl"

    @property
    def environment_state(self) -> Path:
        return self.root / "environment_state.pkl"

    @property
    def rng_state(self) -> Path:
        return self.root / "rng_state.pkl"

    @property
    def metadata(self) -> Path:
        return self.root / "metadata.json"

    @property
    def metrics(self) -> Path:
        return self.root / "metrics.json"


def save_checkpoint(
    root: Path,
    model_state: Dict[str, Any],
    optimizer_state: Dict[str, Any],
    environment_state: Dict[str, Any],
    rng_bundle: RNGBundle,
    metadata: Dict[str, Any],
    metrics: Dict[str, Any],
) -> CheckpointPaths:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    paths = CheckpointPaths(root)

    with open(paths.model_state, "wb") as f:
        pickle.dump(model_state, f)
    with open(paths.optimizer_state, "wb") as f:
        pickle.dump(optimizer_state, f)
    with open(paths.environment_state, "wb") as f:
        pickle.dump(environment_state, f)
    with open(paths.rng_state, "wb") as f:
        pickle.dump(rng_bundle.state_dict(), f)
    with open(paths.metadata, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    with open(paths.metrics, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    return paths


def load_checkpoint(root: Path) -> Dict[str, Any]:
    root = Path(root)
    paths = CheckpointPaths(root)
    if not paths.metadata.exists():
        raise FileNotFoundError(f"No checkpoint metadata found at {paths.metadata}")

    with open(paths.model_state, "rb") as f:
        model_state = pickle.load(f)
    with open(paths.optimizer_state, "rb") as f:
        optimizer_state = pickle.load(f)
    with open(paths.environment_state, "rb") as f:
        environment_state = pickle.load(f)
    with open(paths.rng_state, "rb") as f:
        rng_state = pickle.load(f)
    with open(paths.metadata) as f:
        metadata = json.load(f)
    with open(paths.metrics) as f:
        metrics = json.load(f)

    return dict(
        model_state=model_state,
        optimizer_state=optimizer_state,
        environment_state=environment_state,
        rng_state=rng_state,
        metadata=metadata,
        metrics=metrics,
    )


def restore_rng_bundle(top_level_seed: int, rng_state: Dict[str, bytes]) -> RNGBundle:
    bundle = RNGBundle(top_level_seed=top_level_seed)
    bundle.load_state_dict(rng_state)
    return bundle
