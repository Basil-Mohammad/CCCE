"""
Run provenance and hardware detection (Master Prompt V3, Sections 3, 53, 54).

Every run manifest records: experiment_id, run_id, seed, algorithm,
environment, task_sequence, hyperparameter_hash, git_commit,
software_versions, hardware, timestamp. Hardware-dependent parameters
(number of workers, device) are auto-detected and *logged*, never silently
substituted for a scientifically important hyperparameter.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict


def git_commit_hash(repo_root: str = ".") -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "UNKNOWN_NO_GIT_REPO"


def package_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {"python": sys.version.split()[0]}
    for pkg in ("numpy", "scipy", "pandas", "matplotlib", "pytest", "yaml"):
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "NOT_INSTALLED"
    # Explicitly document the absence of GPU-dependent packages rather than
    # silently omitting them (Section 72: "record exact versions").
    for pkg in ("torch", "gymnasium", "stable_baselines3", "mujoco"):
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except Exception:
            # Broad `except Exception` (not just ImportError) is deliberate:
            # a partially-installed GPU package can fail with OSError/etc.
            # rather than a clean ImportError, and provenance logging must
            # never crash a run over an optional, absent dependency.
            versions[pkg] = "NOT_INSTALLED_SANDBOX_CONSTRAINT"
    return versions


def hardware_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "cpu_count_logical": multiprocessing.cpu_count(),
        "gpu_available": False,
        "gpu_name": None,
        "cuda_available": False,
    }
    try:
        import torch  # noqa: F401

        info["gpu_available"] = torch.cuda.is_available()
        if info["gpu_available"]:
            info["gpu_name"] = torch.cuda.get_device_name(0)
        info["cuda_available"] = torch.cuda.is_available()
    except Exception:
        info["torch_status"] = "NOT_INSTALLED_SANDBOX_CONSTRAINT"
    return info


def hyperparameter_hash(hyperparams: Dict[str, Any]) -> str:
    canonical = json.dumps(hyperparams, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def build_manifest(
    experiment_id: str,
    run_id: str,
    seed: int,
    algorithm: str,
    environment: str,
    task_sequence: str,
    intervention_grid: Any,
    control_protocol: str,
    hyperparameters: Dict[str, Any],
    repo_root: str = ".",
) -> Dict[str, Any]:
    """Build the exact manifest schema required by Section 54."""
    return {
        "experiment_id": experiment_id,
        "run_id": run_id,
        "seed": seed,
        "algorithm": algorithm,
        "environment": environment,
        "task_sequence": task_sequence,
        "intervention_grid": intervention_grid,
        "control_protocol": control_protocol,
        "hyperparameters": hyperparameters,
        "hyperparameter_hash": hyperparameter_hash(hyperparameters),
        "git_commit": git_commit_hash(repo_root),
        "python_version": sys.version.split()[0],
        "package_versions": package_versions(),
        "hardware": hardware_info(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
