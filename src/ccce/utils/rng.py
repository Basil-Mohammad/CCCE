# Author: Basil M. Alzboun

"""
Random-number-stream management.

Master Prompt V3, Section 59 ("Randomness Control") requires separate,
non-shared RNG streams for each of: training, environment, evaluation,
oracle, and bootstrap. This module enforces that separation: it is
impossible to obtain two streams that share state, and every stream is
seeded deterministically from a single top-level experiment seed via
NumPy's recommended `SeedSequence` spawning mechanism (which is designed
exactly for this "one seed -> many independent streams" use case).
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Dict

import numpy as np

STREAM_NAMES = (
    "training_rng",
    "environment_rng",
    "evaluation_rng",
    "oracle_rng",
    "bootstrap_rng",
)


@dataclass
class RNGBundle:
    """A bundle of independent, named RNG streams derived from one seed.

    Streams are derived via `numpy.random.SeedSequence.spawn`, which
    guarantees (to the standard of the NumPy PCG64 bit generator) that the
    resulting streams are statistically independent of one another, even
    though they originate from a single top-level seed. This is the
    mechanism, not a single shared `np.random.default_rng(seed)` reused
    across components.
    """

    top_level_seed: int
    _streams: Dict[str, np.random.Generator] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        seed_seq = np.random.SeedSequence(self.top_level_seed)
        children = seed_seq.spawn(len(STREAM_NAMES))
        for name, child_seq in zip(STREAM_NAMES, children):
            self._streams[name] = np.random.Generator(np.random.PCG64(child_seq))

    def stream(self, name: str) -> np.random.Generator:
        if name not in self._streams:
            raise KeyError(
                f"Unknown RNG stream '{name}'. Valid streams: {STREAM_NAMES}. "
                "Reusing an undeclared stream is disallowed by design (Section 59)."
            )
        return self._streams[name]

    @property
    def training(self) -> np.random.Generator:
        return self.stream("training_rng")

    @property
    def environment(self) -> np.random.Generator:
        return self.stream("environment_rng")

    @property
    def evaluation(self) -> np.random.Generator:
        return self.stream("evaluation_rng")

    @property
    def oracle(self) -> np.random.Generator:
        return self.stream("oracle_rng")

    @property
    def bootstrap(self) -> np.random.Generator:
        return self.stream("bootstrap_rng")

    def state_dict(self) -> Dict[str, bytes]:
        """Serialize all stream states (for checkpointing, Section 6)."""
        return {name: pickle.dumps(gen.bit_generator.state) for name, gen in self._streams.items()}

    def load_state_dict(self, state: Dict[str, bytes]) -> None:
        for name, blob in state.items():
            bg_state = pickle.loads(blob)
            self._streams[name].bit_generator.state = bg_state


def paired_evaluation_seed(base_seed: int, replicate: int) -> int:
    """Derive a deterministic per-replicate seed for *paired* evaluation.

    Used when L_T and L_C must see the *same* evaluation randomness
    (common random numbers, Section 22/59) for a given replicate index.
    Both arms of a paired comparison call this with the same
    (base_seed, replicate) and therefore get the identical evaluation seed.
    """
    seq = np.random.SeedSequence([base_seed, replicate, 0xC0FFEE])
    return int(seq.generate_state(1, dtype=np.uint32)[0])
