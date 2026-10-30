from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env

import gym_insurance
from gym_insurance.v1 import BENFORD, InsurEnv, benford_distance, first_digit


def claims(amounts=(10.0, 20.0, 30.0)):
    return pd.DataFrame({"claim_id": [str(i) for i in range(len(amounts))],
                         "claim_date": pd.date_range("2017-01-01", periods=len(amounts)),
                         "amount": amounts})


@pytest.mark.parametrize("amount,digit", [(0.00012, 1), (0.9, 9), (10.0, 1), (234.5, 2), (99999, 9), (1e-20, 1)])
def test_first_significant_digit(amount, digit):
    assert first_digit(amount) == digit


@pytest.mark.parametrize("amount", [0, -1, np.nan, np.inf])
def test_bad_amounts_rejected(amount):
    with pytest.raises(ValueError):
        InsurEnv(claims([amount]), 100)


def test_hand_calculated_distance_and_reward_no_double_counting():
    env = InsurEnv(claims([10., 20., 30.]), 100)
    env.reset()
    # A cohort of one leading 1 has TV = 1 - log10(2).
    _, r1, _, _, _ = env.step(1)
    assert r1 == pytest.approx(np.log10(2))
    assert env.distance == pytest.approx(1 - np.log10(2))
    assert env.counterfactual_delta() == pytest.approx(BENFORD[1])
    _, r2, _, _, _ = env.step(1)
    assert r2 == pytest.approx(BENFORD[1])
    assert env.counts.tolist() == [1, 1, 0, 0, 0, 0, 0, 0, 0]
    assert env.distance == pytest.approx(1 - BENFORD[0] - BENFORD[1])


def test_primary_reward_telescopes_and_rejection_has_no_reward():
    env = InsurEnv(claims(), 100)
    env.reset()
    env.step(1)
    _, reward, _, _, _ = env.step(0)
    assert reward == 0
    env.step(1)
    assert env.total_reward == pytest.approx(1 - env.distance)


def test_signed_counterfactual_is_explicit_separate_ablation():
    env = InsurEnv(claims([10., 20.]), 100, reward_mode="signed_counterfactual")
    env.reset()
    env.step(1)
    _, reward, terminated, _, info = env.step(0)
    assert reward == pytest.approx(-BENFORD[1])
    assert env.counts.tolist() == [1, 0, 0, 0, 0, 0, 0, 0, 0]
    assert info["reward_mode"] == "signed_counterfactual"
    assert terminated


def test_next_observation_is_next_case_with_updated_budget():
    env = InsurEnv(claims(), 100, amount_scale=10)
    initial, _ = env.reset()
    nxt, _, terminated, truncated, _ = env.step(1)
    assert initial[0] == 1
    assert nxt[0] == 2
    assert nxt[1] == 9
    assert nxt[2] == pytest.approx(.9)
    assert nxt[5] == 1  # selected frequency of digit 1
    assert not terminated and not truncated
    assert env.observation_space.contains(nxt)


def test_infeasible_accept_advances_to_later_affordable_case():
    env = InsurEnv(claims([100., 10.]), 20)
    env.reset()
    nxt, reward, terminated, truncated, info = env.step(1)
    assert reward == 0 and not info["feasible"]
    assert not terminated and not truncated
    assert nxt[0] == 10
    _, _, terminated, _, _ = env.step(1)
    assert terminated and env.accepted == 1 and env.spent == 10


def test_exact_decimal_budget_and_terminal_precedence():
    env = InsurEnv(claims([.1, .2, .01]), .3, max_steps=2)
    env.reset()
    env.step(1)
    obs, _, terminated, truncated, _ = env.step(1)
    assert terminated and not truncated
    assert env.spent == .3 and env.remaining_budget == 0
    assert np.array_equal(obs, np.zeros_like(obs))
    with pytest.raises(RuntimeError):
        env.step(0)


def test_configured_horizon_truncates_not_terminates():
    env = InsurEnv(claims(), 100, max_steps=1)
    env.reset()
    _, _, terminated, truncated, _ = env.step(0)
    assert truncated and not terminated


def test_reject_all_distance_is_explicit_empty_sentinel():
    env = InsurEnv(claims(), 100)
    env.reset()
    for _ in range(3):
        env.step(0)
    assert env.distance == 1 and env.total_reward == 0 and env.accepted == 0


def test_no_input_mutation_and_reset_restores_state():
    frame = claims()
    snapshot = frame.copy(deep=True)
    env = InsurEnv(frame, 100)
    env.reset(); env.step(1); env.reset()
    pd.testing.assert_frame_equal(frame, snapshot)
    assert env.position == 0 and env.accepted == 0 and env.remaining_budget == 100


def test_local_seed_shuffle_deterministic_without_global_rng_mutation():
    np.random.seed(1234)
    before = np.random.get_state()
    env = InsurEnv(claims(np.arange(1, 30, dtype=float)), 100, shuffle=True)
    env.reset(seed=42); order = env.order.copy()
    env.reset(seed=42)
    assert np.array_equal(order, env.order)
    env.reset(seed=43)
    assert not np.array_equal(order, env.order)
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]


def test_no_hidden_file_writes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    env = InsurEnv(claims(), 100)
    env.reset(); env.step(1); env.reset()
    assert list(Path(tmp_path).iterdir()) == []


@pytest.mark.parametrize("budget", [0, -1, np.inf, np.nan])
def test_bad_budgets(budget):
    with pytest.raises(ValueError):
        InsurEnv(claims(), budget)


def test_duplicate_ids_and_bad_features():
    frame = claims(); frame.loc[1, "claim_id"] = "0"
    with pytest.raises(ValueError): InsurEnv(frame, 100)
    frame = claims(); frame["feature"] = [1, np.nan, 3]
    with pytest.raises(ValueError): InsurEnv(frame, 100, feature_columns=["feature"])


def test_gymnasium_checker_and_registration():
    env = InsurEnv(claims(), 100)
    check_env(env, skip_render_check=True)
    registered = gym.make("InsurEnv-v1", data=claims(), budget=100)
    assert registered.reset(seed=4)[0].shape == (23,)


def test_count_validation_and_action_validation():
    with pytest.raises(ValueError): benford_distance(np.ones(8))
    with pytest.raises(ValueError): benford_distance(-np.ones(9))
    env = InsurEnv(claims(), 100)
    with pytest.raises(RuntimeError): env.step(1)
    env.reset()
    with pytest.raises(ValueError): env.step(2)
