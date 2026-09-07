# Author: Basil M. Alzboun

"""
Unit + integration tests for the Level-2 synthetic environment pipeline
(Master Prompt V3, Sections 35, 43).
"""
import numpy as np
import pytest

from ccce.environments.synthetic_env import SyntheticPointMassEnv, InterventionSpec, default_task_set
from ccce.learners.numpy_ppo import LinearGaussianPolicy, PPOHyperparameters, train_policy, collect_rollout
from ccce.competence.contract import evaluate_competence, envelope_satisfied, DEFAULT_ENVELOPE
from ccce.diagnostics.conventional import (
    gradient_interference, representation_drift, plasticity_ratio, task_similarity, update_magnitude,
)
from ccce.baselines.continual_methods import EWCState, ReplayBuffer, train_with_replay, train_with_ewc


TASKS = default_task_set()


def test_task_set_has_six_tasks_with_documented_overlap():
    assert set(TASKS.keys()) == {"T1", "T2", "T3", "T4", "T5", "T6"}
    # T1 and T2 share z1/z2 baseline dynamics; T5 combines z3 and z4 (design intent).
    assert TASKS["T5"].z3_obstacle_density > 0 and TASKS["T5"].z4_recovery_perturbation > 0


def test_env_reset_and_step_shapes():
    env = SyntheticPointMassEnv(TASKS["T1"])
    rng = np.random.default_rng(0)
    obs = env.reset(rng)
    assert obs.shape == (4,)
    next_obs, reward, done, info = env.step(np.array([0.1, -0.1]), rng)
    assert next_obs.shape == (4,)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert "success" in info and "collision" in info


def test_intervention_z1_changes_dynamics():
    rng = np.random.default_rng(0)
    env_baseline = SyntheticPointMassEnv(TASKS["T1"])
    env_intervened = SyntheticPointMassEnv(TASKS["T1"], intervention=InterventionSpec("z1", 2.0))
    env_baseline.reset(np.random.default_rng(1))
    env_intervened.reset(np.random.default_rng(1))
    env_baseline.pos = np.array([1.0, 1.0]); env_baseline.vel = np.array([1.0, 0.0])
    env_intervened.pos = np.array([1.0, 1.0]); env_intervened.vel = np.array([1.0, 0.0])
    ob1, *_ = env_baseline.step(np.zeros(2), np.random.default_rng(2))
    ob2, *_ = env_intervened.step(np.zeros(2), np.random.default_rng(2))
    assert not np.allclose(ob1, ob2), "z1 intervention should change position update"


def test_episode_terminates_within_max_steps():
    env = SyntheticPointMassEnv(TASKS["T3"])
    rng = np.random.default_rng(0)
    env.reset(rng)
    steps = 0
    done = False
    while not done and steps < 1000:
        _, _, done, _ = env.step(rng.uniform(-1, 1, size=2), rng)
        steps += 1
    assert steps <= TASKS["T3"].max_steps


# --------------------------------------------------------------------------
# PPO learner
# --------------------------------------------------------------------------
def test_policy_flat_params_roundtrip():
    rng = np.random.default_rng(0)
    policy = LinearGaussianPolicy.initialize(4, 2, rng, log_std_init=-0.5)
    flat = policy.flat_params()
    policy2 = LinearGaussianPolicy(4, 2)
    policy2.set_flat_params(flat)
    assert np.allclose(flat, policy2.flat_params())


def test_training_changes_parameters():
    rng = np.random.default_rng(0)
    policy = LinearGaussianPolicy.initialize(4, 2, rng, log_std_init=-0.5)
    before = policy.flat_params().copy()
    hp = PPOHyperparameters(total_env_steps=300, rollout_length=60, n_epochs=2)
    train_policy(lambda: SyntheticPointMassEnv(TASKS["T1"]), policy, rng, hp)
    after = policy.flat_params()
    assert not np.allclose(before, after), "training should change parameters"


def test_training_is_deterministic_given_same_seed():
    hp = PPOHyperparameters(total_env_steps=180, rollout_length=60, n_epochs=2)

    rng1 = np.random.default_rng(42)
    policy1 = LinearGaussianPolicy.initialize(4, 2, rng1, log_std_init=-0.5)
    train_policy(lambda: SyntheticPointMassEnv(TASKS["T1"]), policy1, rng1, hp)

    rng2 = np.random.default_rng(42)
    policy2 = LinearGaussianPolicy.initialize(4, 2, rng2, log_std_init=-0.5)
    train_policy(lambda: SyntheticPointMassEnv(TASKS["T1"]), policy2, rng2, hp)

    assert np.allclose(policy1.flat_params(), policy2.flat_params()), (
        "Same seed must produce identical trajectories (reproducibility, Section 3)"
    )


# --------------------------------------------------------------------------
# Competence evaluation
# --------------------------------------------------------------------------
def test_competence_outcome_vector_in_valid_range():
    rng = np.random.default_rng(0)
    policy = LinearGaussianPolicy.initialize(4, 2, rng, log_std_init=-0.5)
    outcome = evaluate_competence(TASKS["T1"], policy, n_episodes=3, rng=rng)
    vec = outcome.as_vector()
    assert np.all(vec >= -0.01) and np.all(vec <= 2.01)  # 'time' term can slightly exceed 1 by construction


def test_envelope_check_runs_without_error():
    rng = np.random.default_rng(0)
    policy = LinearGaussianPolicy.initialize(4, 2, rng, log_std_init=-0.5)
    outcome = evaluate_competence(TASKS["T1"], policy, n_episodes=3, rng=rng)
    result = envelope_satisfied(outcome, DEFAULT_ENVELOPE)
    assert isinstance(result, bool)


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------
def test_gradient_interference_range():
    g1 = np.array([1.0, 0.0])
    g2 = np.array([1.0, 0.0])
    g3 = np.array([-1.0, 0.0])
    assert gradient_interference(g1, g2) == pytest.approx(1.0)
    assert gradient_interference(g1, g3) == pytest.approx(-1.0)


def test_representation_drift_zero_for_identical_policy():
    rng = np.random.default_rng(0)
    p1 = LinearGaussianPolicy.initialize(4, 2, rng, log_std_init=-0.5)
    p2 = p1.copy()
    assert representation_drift(p1, p2) == pytest.approx(0.0)


def test_representation_drift_positive_after_training():
    rng = np.random.default_rng(0)
    p1 = LinearGaussianPolicy.initialize(4, 2, rng, log_std_init=-0.5)
    p_before = p1.copy()
    hp = PPOHyperparameters(total_env_steps=180, rollout_length=60, n_epochs=2)
    train_policy(lambda: SyntheticPointMassEnv(TASKS["T1"]), p1, rng, hp)
    assert representation_drift(p_before, p1) > 0.0


def test_task_similarity_identical_goals_is_one():
    assert task_similarity(np.array([1.0, 1.0]), np.array([2.0, 2.0])) == pytest.approx(1.0)


def test_update_magnitude_nonneg():
    a = np.array([0.0, 0.0])
    b = np.array([3.0, 4.0])
    assert update_magnitude(a, b) == pytest.approx(5.0)


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------
def test_ewc_penalty_zero_before_any_task():
    ewc = EWCState()
    grad = np.ones(5)
    assert np.allclose(ewc.penalty_grad(np.ones(5)), np.zeros(5))


def test_ewc_state_updates_after_task():
    rng = np.random.default_rng(0)
    policy = LinearGaussianPolicy.initialize(4, 2, rng, log_std_init=-0.5)
    ewc = EWCState()
    hp = PPOHyperparameters(total_env_steps=180, rollout_length=60, n_epochs=2)
    train_with_ewc(lambda: SyntheticPointMassEnv(TASKS["T1"]), policy, rng, hp, ewc)
    assert ewc.anchor_params is not None
    assert ewc.fisher_diag is not None
    assert ewc.fisher_diag.shape == policy.flat_params().shape


def test_replay_buffer_add_and_sample():
    buf = ReplayBuffer(capacity=10)
    obs = [np.zeros(4) for _ in range(5)]
    act = [np.zeros(2) for _ in range(5)]
    logp = [0.0] * 5
    adv = [1.0] * 5
    ret = [1.0] * 5
    buf.add(obs, act, logp, adv, ret)
    assert len(buf.obs) == 5
    s_obs, s_act, s_logp, s_adv, s_ret = buf.sample(3, np.random.default_rng(0))
    assert len(s_obs) == 3


def test_replay_training_runs_end_to_end():
    rng = np.random.default_rng(0)
    policy = LinearGaussianPolicy.initialize(4, 2, rng, log_std_init=-0.5)
    buf = ReplayBuffer(capacity=500)
    hp = PPOHyperparameters(total_env_steps=180, rollout_length=60, n_epochs=2)
    log = train_with_replay(lambda: SyntheticPointMassEnv(TASKS["T1"]), policy, rng, hp, buf)
    assert "returns_log" in log
    assert len(buf.obs) > 0
