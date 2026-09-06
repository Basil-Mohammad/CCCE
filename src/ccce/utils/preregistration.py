"""
Pre-registration-like lock mechanism (Master Prompt V3, Section 57).

Before a confirmation-stage experiment is run, a manifest describing every
locked design choice (seeds, task sequences, interventions, controls,
budget, thresholds) is written to disk and hashed. The hash is logged
alongside the eventual results, so a reader can verify after the fact that
the locked design was not altered between registration and execution.
This module does not (and cannot, in a single-process script) prevent a
determined author from editing the file before "unlocking" it -- what it
provides is an auditable paper trail, not a cryptographic guarantee.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict


def write_confirmation_manifest(path: Path, design: Dict[str, Any]) -> str:
    """Write the locked design to `path` and return its SHA-256 hash.
    Must be called BEFORE any confirmation-stage result is inspected."""
    path.parent.mkdir(parents=True, exist_ok=True)
    canonical = json.dumps(design, sort_keys=True, indent=2, default=str)
    with open(path, "w") as f:
        f.write(canonical)
    return hashlib.sha256(canonical.encode()).hexdigest()


def verify_confirmation_manifest(path: Path, expected_hash: str) -> bool:
    """Verify the on-disk manifest still matches the hash recorded at
    registration time (i.e., it was not edited after locking)."""
    with open(path) as f:
        canonical = f.read()
    actual_hash = hashlib.sha256(canonical.encode()).hexdigest()
    return actual_hash == expected_hash
