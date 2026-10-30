import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gym_insurance import benchmark as b


def config():
    cfg = json.loads((Path(__file__).parents[1] / "configs/psr-v1.json").read_text())
    cfg.update(train_end="2017-12-31", validation_end="2018-12-31", test_end="2019-12-31",
               episode_size=4, seeds=[0, 1], budget_fractions=[.1, .25, .5])
    cfg["dqn"].update(total_steps=32, batch_size=4, warmup_steps=4, replay_size=32,
                      train_frequency=1, target_update_steps=8, hidden_sizes=[8, 8])
    return cfg


def fixture(tmp_path, cfg=None, test_multiplier=1):
    cfg = cfg or config()
    rows = []
    for year in (2017, 2018, 2019):
        for i in range(9):
            rows.append({"claim_id": f"{year}-{i}", "claim_date": f"{year}-01-{i + 1:02d}",
                         "amount": (i + 1) * 10 * (test_multiplier if year == 2019 else 1)})
    source = tmp_path / "input.csv"; pd.DataFrame(rows).to_csv(source, index=False)
    cfgfile = tmp_path / "config.json"; b.write_json(cfgfile, cfg)
    out = tmp_path / "run"
    b.freeze(cfgfile, out)
    return source, out


def test_prepare_train_only_budgets_and_temporal_split(tmp_path):
    source, out = fixture(tmp_path, test_multiplier=1e6)
    result = b.prepare(source, out)
    assert result["train_median_window_total"] == 180  # sums100 and260
    assert result["budgets"] == [18, 45, 90]
    assert result["amount_scale_train_median"] == 50
    assert result["splits"]["test"]["tail_rows_excluded"] == 1
    for split, year in [("train", 2017), ("validation", 2018), ("test", 2019)]:
        assert set(b.load_split(out, split).claim_date.dt.year) == {year}


def test_freeze_required_and_cannot_be_silently_rewritten(tmp_path):
    with pytest.raises(FileNotFoundError): b.prepare(tmp_path / "absent.csv", tmp_path)
    source, out = fixture(tmp_path)
    with pytest.raises(FileExistsError): b.freeze(tmp_path / "config.json", out)
    lock = b.read_json(out / "frozen_protocol.json")
    lock["config"]["dqn"]["learning_rate"] = 2
    b.write_json(out / "frozen_protocol.json", lock)
    with pytest.raises(ValueError, match="modified"): b.prepare(source, out)


def test_duplicate_identity_not_silently_dropped(tmp_path):
    source, out = fixture(tmp_path)
    frame = pd.read_csv(source); frame.loc[1, "claim_id"] = frame.loc[0, "claim_id"]
    frame.to_csv(source, index=False)
    with pytest.raises(ValueError, match="unique"): b.prepare(source, out)


def test_no_engineered_full_history_features_permitted():
    cfg = config(); cfg["feature_columns"] = ["full_history_mean"]
    with pytest.raises(ValueError, match="amount-only"): b.validate_config(cfg)


def test_amount_threshold_uses_current_observable_amount_at_source_precision():
    from gym_insurance.v1 import InsurEnv
    frame = pd.DataFrame({"claim_id": ["a"], "claim_date": ["2017-01-01"], "amount": [50.0000001]})
    env = InsurEnv(frame, 100, amount_scale=50)
    obs, _ = env.reset()
    assert obs[0] == 1  # A float32 rounding edge must not silently change the rule.
    assert b.policy_action("amount_threshold", env, np.random.default_rng(0), 50, None, obs) == 0


def test_prepared_split_integrity(tmp_path):
    source, out = fixture(tmp_path); b.prepare(source, out)
    p = out / "prepared/test.csv"; p.write_text(p.read_text() + "tampered")
    with pytest.raises(ValueError, match="changed"): b.prepared(out)


def test_derived_prepared_metadata_is_bound_to_digest(tmp_path):
    source, out = fixture(tmp_path); b.prepare(source, out)
    metadata = b.read_json(out / "prepared.json")
    metadata["budgets"][0] = 99999
    b.write_json(out / "prepared.json", metadata)
    with pytest.raises(ValueError, match="metadata changed"): b.prepared(out)


def test_baseline_end_to_end_validation_before_test_and_aggregate_report(tmp_path):
    cfg = config(); cfg["policies"].remove("dqn")
    source, out = fixture(tmp_path, cfg); b.prepare(source, out)
    b.train(out)
    with pytest.raises(ValueError, match="validation"): b.evaluate(out, "test")
    validation = b.evaluate(out, "validation")
    test = b.evaluate(out, "test")
    report = b.report(out)
    assert len(validation) == len(test) == 2 * 3 * 2 * 5 * 2
    assert test[test.policy == "reject_all"].terminal_distance.eq(1).all()
    assert test[test.policy == "reject_all"].claim_coverage.eq(0).all()
    assert test.budget_utilization.le(1).all()
    primary = test[test.reward_mode == "potential_improvement"]
    np.testing.assert_allclose(primary.total_reward, 1 - primary.terminal_distance)
    assert report["paired_comparisons"] == []
    assert "claim_id" not in (out / "report.json").read_text()
    with pytest.raises(FileExistsError): b.evaluate(out, "test")


def test_real_pytorch_training_saving_evaluation(tmp_path):
    cfg = config(); cfg["seeds"] = [0]; cfg["budget_fractions"] = [.5]
    cfg["reward_modes"] = ["potential_improvement"]
    source, out = fixture(tmp_path, cfg); b.prepare(source, out)
    jobs = b.train(out)
    assert len(jobs) == 1 and jobs[0]["updates"] > 0
    assert jobs[0]["last_100_loss_mean"] >= 0
    assert jobs[0]["implementation"] == "native_pytorch_dqn"
    b.evaluate(out, "validation"); b.evaluate(out, "test")
    result = b.report(out)
    assert result["paired_comparisons"][1]["paired_windows_per_seed"] == 2
    # Resume uses a verified checkpoint rather than training a hidden replacement.
    assert b.train(out)[0]["model_sha256"] == jobs[0]["model_sha256"]


def test_paired_statistics_use_seed_year_cells_not_rows():
    rows = []
    for seed in range(2):
        for year in (2020, 2021):
            for block in range(3):
                for policy, distance in [("dqn", .2), ("myopic", .3)]:
                    rows.append(dict(split="test", reward_mode="potential_improvement", budget_index=0,
                                     seed=seed, year=year, episode_id=f"{year}-{block}", policy=policy,
                                     terminal_distance=distance))
    result = b.paired_summary(pd.DataFrame(rows), bootstraps=50)[0]
    assert result["independent_training_seeds"] == 2
    assert result["calendar_year_clusters"] == 2
    assert result["paired_windows_per_seed"] == 6
    assert result["difference_equal_year_mean"] == pytest.approx(-.1)


def test_missing_paired_window_is_rejected():
    rows = [dict(split="test", reward_mode="potential_improvement", budget_index=0,
                 seed=0, year=2020, episode_id=episode, policy=policy, terminal_distance=.2)
            for episode, policy in [("2020-0", "dqn"), ("2020-1", "dqn"), ("2020-0", "myopic")]]
    with pytest.raises(ValueError, match="missing paired"): b.paired_summary(pd.DataFrame(rows))


def test_prespecified_controls_are_illustrative_and_label_blind(tmp_path):
    source, out = fixture(tmp_path)
    result = b.controls(out)
    assert result["identical_decisions_across_opposite_label_worlds"]
    benign = next(r for r in result["results"] if r["fixture"] == "all_legitimate_rounded_digit1" and r["policy"] == "myopic")
    assert benign["rejected_legitimate_fraction"] > .9
    assert benign["synthetic_fraud_prevalence"] == 0
    assert benign["rejected_synthetic_fraud_fraction"] is None


def test_swapped_checkpoint_pairs_cannot_be_reused_or_evaluated(tmp_path):
    import shutil
    cfg = config(); cfg["seeds"] = [0, 1]; cfg["budget_fractions"] = [.5]
    cfg["reward_modes"] = ["potential_improvement"]
    source, out = fixture(tmp_path, cfg); b.prepare(source, out); b.train(out)
    first = out / "models" / b.model_name("potential_improvement", 0, 0)
    second = out / "models" / b.model_name("potential_improvement", 0, 1)
    shutil.copyfile(first, second)
    shutil.copyfile(first.with_suffix(".json"), second.with_suffix(".json"))
    with pytest.raises(ValueError, match="identity"): b.train(out)
    with pytest.raises(ValueError, match="identity"): b.evaluate(out, "validation")
