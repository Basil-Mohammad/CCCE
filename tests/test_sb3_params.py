import numpy as np
import pytest
from stable_baselines3 import PPO

from ccce.environments.gym_wrapper import GymSyntheticPointMassEnv
from ccce.environments.synthetic_env import default_task_set
from ccce.learners.sb3_params import get_flat_params, set_flat_params, apply_ewc_penalty

TASKS = default_task_set()


def _make_model(seed=0):
    env = GymSyntheticPointMassEnv(TASKS["T1"])
    return PPO("MlpPolicy", env, verbose=0, n_steps=64, batch_size=32, seed=seed)


def test_flat_params_roundtrip():
    model = _make_model()
    flat = get_flat_params(model)
    set_flat_params(model, flat)
    assert np.allclose(flat, get_flat_params(model))


def test_training_changes_flat_params():
    model = _make_model()
    before = get_flat_params(model)
    model.learn(total_timesteps=64)
    after = get_flat_params(model)
    assert not np.allclose(before, after)


def test_ewc_penalty_never_overshoots_anchor_small_gap():
    model = _make_model(seed=0)
    anchor = get_flat_params(model)
    model.learn(total_timesteps=64)
    before_dist = np.linalg.norm(get_flat_params(model) - anchor)
    fisher = np.ones_like(anchor)
    apply_ewc_penalty(model, anchor, fisher, lambda_ewc=300.0)
    after_dist = np.linalg.norm(get_flat_params(model) - anchor)
    assert after_dist < before_dist


def test_ewc_penalty_never_overshoots_anchor_large_gap():
    model = _make_model(seed=1)
    anchor = get_flat_params(model)
    model.learn(total_timesteps=2000)
    before_dist = np.linalg.norm(get_flat_params(model) - anchor)
    fisher = np.ones_like(anchor)
    apply_ewc_penalty(model, anchor, fisher, lambda_ewc=300.0)
    after_dist = np.linalg.norm(get_flat_params(model) - anchor)
    assert after_dist < before_dist


def test_ewc_penalty_zero_gap_is_noop():
    model = _make_model(seed=2)
    anchor = get_flat_params(model)
    fisher = np.ones_like(anchor)
    apply_ewc_penalty(model, anchor, fisher, lambda_ewc=300.0)
    after = get_flat_params(model)
    assert np.allclose(after, anchor, atol=1e-6)
