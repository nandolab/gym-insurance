"""Export verified aggregates and explicit equal-year seed variation after completion.

This postprocessor does not train, tune, choose checkpoints, or change the protocol.
"""
import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from gym_insurance import benchmark as b

METRICS = ["terminal_distance", "claim_coverage", "value_coverage", "budget_utilization", "total_reward"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    study, out = Path(args.study), Path(args.output)
    if not (study / "test_evaluation.json").is_file():
        raise SystemExit("final test evaluation is not complete")
    # Revalidate protocol, prepared data, checkpoints' recorded evaluation bindings,
    # hashes and the exact evaluation matrix before exporting any conclusions.
    report = b.report(study)
    lock, prepared = b.prepared(study)
    jobs = b.read_json(study / "training.json")["jobs"]
    expected_jobs = (len(lock["config"]["seeds"]) * len(prepared["budgets"]) * len(lock["config"]["reward_modes"]))
    if len(jobs) != expected_jobs:
        raise ValueError("incomplete training job manifest")
    expected_ids = {(mode, budget, seed) for mode in lock["config"]["reward_modes"]
                    for budget in range(len(prepared["budgets"])) for seed in lock["config"]["seeds"]}
    if {(j["reward_mode"], j["budget_index"], j["seed"]) for j in jobs} != expected_ids:
        raise ValueError("training job identities are missing or duplicated")
    for job in jobs:
        b.validate_job(job, lock, prepared, job["reward_mode"], job["budget_index"], job["seed"])
        model = study / "models" / b.model_name(job["reward_mode"], job["budget_index"], job["seed"])
        if b.sha256(model) != job["model_sha256"]:
            raise ValueError("trained checkpoint changed after evaluation")
    frames = [pd.read_csv(study / f"{split}_metrics.csv") for split in ("validation", "test")]
    frame = pd.concat(frames, ignore_index=True)
    year_keys = ["split", "reward_mode", "budget_index", "policy", "seed", "year"]
    per_year = frame.groupby(year_keys)[METRICS].mean().reset_index()
    seed_keys = ["split", "reward_mode", "budget_index", "policy", "seed"]
    per_seed = per_year.groupby(seed_keys)[METRICS].mean().reset_index()
    variation = per_seed.groupby(seed_keys[:-1])[METRICS].agg(["mean", "std", "min", "max"])
    variation.columns = ["_".join(col) for col in variation.columns]
    variation = variation.reset_index().fillna(0)
    out.mkdir(parents=True, exist_ok=True)
    per_year.to_csv(out / "gym-seed-year-means.csv", index=False)
    per_seed.to_csv(out / "gym-seed-equal-year-means.csv", index=False)
    variation.to_csv(out / "gym-seed-variation.csv", index=False)
    b.write_json(out / "gym-paired-comparisons.json", {
        "protocol_sha256": lock["protocol_sha256"],
        "comparisons": report["paired_comparisons"],
        "estimand": "DQN-minus-myopic mean terminal distance; first average windows within seed/year, then weight years equally",
        "interval": "crossed bootstrap of seed and calendar-year cells; conditional descriptive 95% interval; only two held-out year clusters"})
    copies = {
        "frozen_protocol.json": "gym-frozen-protocol.json",
        "prepared.json": "gym-prepared-manifest.json",
        "training.json": "gym-training.json",
        "report.json": "gym-report.json",
        "summary.csv": "gym-summary.csv",
        "REPORT.md": "gym-report.md",
        "synthetic_controls.json": "gym-synthetic-controls.json",
        "synthetic_controls.csv": "gym-synthetic-controls.csv",
        "validation_evaluation.json": "gym-validation-evaluation.json",
        "test_evaluation.json": "gym-test-evaluation.json",
    }
    for source, target in copies.items():
        shutil.copyfile(study / source, out / target)
    source_root = Path(__file__).resolve().parents[1]
    wheel = source_root / "artifacts/wheels/gym_insurance-0.2.0.dev0-py3-none-any.whl"
    if wheel.exists():
        shutil.copyfile(wheel, out / wheel.name)
    status_path = out / "gym-engineering-status.json"
    status = b.read_json(status_path) if status_path.exists() else {}
    status.update({"phase": "completed_local_historical_benchmark",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "trained_models": len(jobs),
        "evaluation_run_window_rows": {split: len(f) for split, f in zip(("validation", "test"), frames)},
        "protocol_sha256": lock["protocol_sha256"], "source_sha256": prepared["source_sha256"],
        "frozen_code_sha256": lock["code_sha256"], "current_code_sha256": b.code_hash(),
        "frozen_code_unchanged": lock["code_sha256"] == b.code_hash(),
        "per_seed_weighting": "equal years after averaging windows within each seed/year; seed std is training-seed variation, not population uncertainty",
        "export_postprocessor_sha256": b.sha256(__file__),
        "main_report_weighting": "window-weighted descriptive aggregate mean/std; not the paired equal-year estimand"})
    b.write_json(status_path, status)
    print(json.dumps({"status": status["phase"], "models": status["trained_models"],
                      "rows": status["evaluation_run_window_rows"],
                      "protocol_sha256": lock["protocol_sha256"], "output": str(out.resolve())}, indent=2))


if __name__ == "__main__":
    main()
