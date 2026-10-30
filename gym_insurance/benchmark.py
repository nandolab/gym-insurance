"""Frozen-protocol preparation, training, evaluation and aggregate reporting CLI."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import itertools
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .v1 import InsurEnv, REWARD_MODES

POLICIES = ("reject_all", "accept_feasible", "random", "amount_threshold", "myopic", "dqn")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def code_hash():
    root = Path(__file__).parent
    return json_hash({str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*.py"))})


def environment_manifest(include_torch=True):
    names = ["numpy", "pandas", "gymnasium"] + (["torch"] if include_torch else [])
    packages = {}
    for name in names:
        module = importlib.import_module(name)
        packages[name] = {"version": str(getattr(module, "__version__", "unknown")),
                          "resolved_path": str(module.__file__)}
    return {"python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "packages": packages,
            "device": "cpu", "shared_installations_modified": False}


def validate_config(cfg):
    if cfg["schema_version"] != 1:
        raise ValueError("unsupported protocol schema")
    if not cfg["seeds"] or len(set(cfg["seeds"])) != len(cfg["seeds"]):
        raise ValueError("seeds must be a nonempty unique list")
    if not set(cfg["policies"]).issubset(POLICIES) or not cfg["policies"]:
        raise ValueError("invalid policies")
    if not set(cfg["reward_modes"]).issubset(REWARD_MODES) or not cfg["reward_modes"]:
        raise ValueError("invalid reward modes")
    if cfg["episode_size"] < 2 or not 0 < min(cfg["budget_fractions"]) <= max(cfg["budget_fractions"]) <= 1:
        raise ValueError("invalid episode size or budget fractions")
    if len(set(cfg["budget_fractions"])) != len(cfg["budget_fractions"]):
        raise ValueError("budget fractions must be unique")
    if pd.Timestamp(cfg["train_end"]) >= pd.Timestamp(cfg["validation_end"]):
        raise ValueError("train_end must precede validation_end")
    if cfg.get("test_end") and pd.Timestamp(cfg["validation_end"]) >= pd.Timestamp(cfg["test_end"]):
        raise ValueError("validation_end must precede test_end")
    if cfg.get("feature_columns", []):
        raise ValueError("protocol v1 is amount-only: history/other features require a new protocol")
    dqn = cfg["dqn"]
    if dqn["gamma"] != 1.0:
        raise ValueError("this finite-horizon potential protocol fixes gamma=1")
    for key in ("total_steps", "batch_size", "warmup_steps", "replay_size", "train_frequency", "target_update_steps"):
        if int(dqn[key]) < 1:
            raise ValueError(f"{key} must be positive")
    if dqn["replay_size"] < max(dqn["batch_size"], dqn["warmup_steps"]):
        raise ValueError("replay buffer is smaller than batch/warmup")


def freeze(config_path, output):
    cfg = read_json(config_path)
    validate_config(cfg)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "frozen_protocol.json"
    if path.exists():
        raise FileExistsError("protocol already frozen; use a new output directory for a revision")
    lock = {"config": cfg, "config_sha256": json_hash(cfg), "code_sha256": code_hash(),
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "test_inspection": "protocol frozen before prepare/evaluate; no test-based tuning allowed"}
    lock["environment"] = environment_manifest("dqn" in cfg["policies"])
    lock["protocol_sha256"] = json_hash(lock)
    write_json(path, lock)
    return lock


def locked(output):
    output = Path(output)
    lock = read_json(output / "frozen_protocol.json")
    recorded = lock["protocol_sha256"]
    if json_hash({k: v for k, v in lock.items() if k != "protocol_sha256"}) != recorded:
        raise ValueError("frozen protocol was modified")
    if json_hash(lock["config"]) != lock["config_sha256"] or code_hash() != lock["code_sha256"]:
        raise ValueError("configuration/code changed after freeze; create a new versioned run")
    return lock


def episodes(frame, size):
    result = []
    # No window crosses years or splits. Incomplete tails are explicitly excluded.
    for year, group in frame.groupby(frame["claim_date"].dt.year, sort=True):
        group = group.sort_values(["claim_date", "claim_id"], kind="stable")
        for start in range(0, len(group) - size + 1, size):
            result.append((f"{year}-{start // size:05d}", group.iloc[start:start + size].reset_index(drop=True)))
    return result


def prepare(input_path, output, id_column="claim_id", date_column="claim_date", amount_column="amount"):
    output = Path(output)
    lock = locked(output)  # Must precede opening input data.
    cfg = lock["config"]
    if (output / "prepared.json").exists():
        raise FileExistsError("prepared data already exists; refusing to overwrite")
    raw = pd.read_csv(input_path, dtype={id_column: "string"}, usecols=[id_column, date_column, amount_column])
    frame = raw.rename(columns={id_column: "claim_id", date_column: "claim_date", amount_column: "amount"})
    if frame["claim_id"].isna().any() or frame["claim_id"].str.strip().eq("").any() or frame["claim_id"].duplicated().any():
        raise ValueError("input requires unique nonempty claim IDs; resolve duplicates in provenance step")
    frame["claim_date"] = pd.to_datetime(frame["claim_date"], errors="raise", format="ISO8601")
    if frame["claim_date"].isna().any():
        raise ValueError("invalid or missing cohort dates")
    frame["amount"] = pd.to_numeric(frame["amount"], errors="raise")
    if not np.isfinite(frame["amount"]).all() or (frame["amount"] <= 0).any():
        raise ValueError("amount must be finite and positive; exclusions belong in provenance step")
    frame = frame.sort_values(["claim_date", "claim_id"], kind="stable").reset_index(drop=True)
    train_end, val_end = pd.Timestamp(cfg["train_end"]), pd.Timestamp(cfg["validation_end"])
    masks = {"train": frame.claim_date <= train_end,
             "validation": (frame.claim_date > train_end) & (frame.claim_date <= val_end),
             "test": frame.claim_date > val_end}
    outside = 0
    if cfg.get("test_end"):
        outside = int((frame.claim_date > pd.Timestamp(cfg["test_end"])).sum())
        masks["test"] &= frame.claim_date <= pd.Timestamp(cfg["test_end"])
    splits = {name: frame.loc[mask].copy() for name, mask in masks.items()}
    counts = {}
    train_windows = episodes(splits["train"], cfg["episode_size"])
    if not train_windows:
        raise ValueError("no full training windows for frozen episode_size")
    median_total = float(np.median([f.amount.sum() for _, f in train_windows]))
    budgets = [median_total * x for x in cfg["budget_fractions"]]
    amount_scale = float(splits["train"].amount.median())
    prepared_dir = output / "prepared"
    prepared_dir.mkdir(exist_ok=True)
    hashes = {}
    for name, split in splits.items():
        windows = episodes(split, cfg["episode_size"])
        if not windows:
            raise ValueError(f"no full {name} windows for frozen episode_size")
        target = prepared_dir / f"{name}.csv"
        split.to_csv(target, index=False, date_format="%Y-%m-%d")
        hashes[name] = sha256(target)
        counts[name] = {"rows": len(split), "full_windows": len(windows),
                        "rows_in_windows": len(windows) * cfg["episode_size"],
                        "tail_rows_excluded": len(split) - len(windows) * cfg["episode_size"],
                        "years": sorted(int(y) for y in split.claim_date.dt.year.unique())}
    manifest = {"protocol_sha256": lock["protocol_sha256"], "source_sha256": sha256(input_path),
                "source_path": str(Path(input_path).resolve()), "source_columns": {"id": id_column, "date": date_column, "amount": amount_column},
                "date_semantics": cfg["date_semantics"], "amount_availability": cfg["amount_availability"],
                "splits": counts, "split_sha256": hashes, "outside_test_end_excluded": outside,
                "train_median_window_total": median_total, "budgets": budgets,
                "amount_scale_train_median": amount_scale,
                "amount_threshold_train_median": amount_scale,
                "features": "current amount plus within-episode selected cohort/budget state only; no full-history features"}
    manifest["prepared_sha256"] = json_hash(manifest)
    write_json(output / "prepared.json", manifest)
    return manifest


def prepared(output):
    output = Path(output)
    lock = locked(output)
    metadata = read_json(output / "prepared.json")
    if metadata.get("prepared_sha256") != json_hash({k: v for k, v in metadata.items() if k != "prepared_sha256"}):
        raise ValueError("prepared metadata changed")
    if metadata["protocol_sha256"] != lock["protocol_sha256"]:
        raise ValueError("prepared data protocol mismatch")
    for split, digest in metadata["split_sha256"].items():
        if sha256(output / "prepared" / f"{split}.csv") != digest:
            raise ValueError(f"prepared {split} data changed")
    return lock, metadata


def load_split(output, split):
    return pd.read_csv(Path(output) / "prepared" / f"{split}.csv", dtype={"claim_id": "string"}, parse_dates=["claim_date"])


def model_name(mode, budget_index, seed):
    return f"{mode}-b{budget_index}-s{seed}.pt"


def validate_job(job, lock, metadata, mode, budget_index, seed):
    expected = {"seed": seed, "reward_mode": mode, "budget_index": budget_index,
                "budget": metadata["budgets"][budget_index], "config": lock["config"]["dqn"],
                "observation_size": 23, "implementation": "native_pytorch_dqn",
                "protocol_sha256": lock["protocol_sha256"], "source_sha256": metadata["source_sha256"],
                "prepared_sha256": metadata["prepared_sha256"]}
    for key, value in expected.items():
        if job.get(key) != value:
            raise ValueError(f"checkpoint job identity mismatch: {key}")


def train(output):
    from . import dqn
    output = Path(output)
    lock, metadata = prepared(output)
    cfg = lock["config"]
    windows = episodes(load_split(output, "train"), cfg["episode_size"])
    (output / "models").mkdir(exist_ok=True)
    jobs = []
    for mode in cfg["reward_modes"]:
        for b, budget in enumerate(metadata["budgets"]):
            for seed in cfg["seeds"]:
                if "dqn" not in cfg["policies"]:
                    continue
                model_path = output / "models" / model_name(mode, b, seed)
                sidecar = model_path.with_suffix(".json")
                if model_path.exists() and sidecar.exists():
                    job = read_json(sidecar)
                    validate_job(job, lock, metadata, mode, b, seed)
                    if job["model_sha256"] != sha256(model_path):
                        raise ValueError("existing model mismatch; refusing reuse")
                else:
                    def make_env(index):
                        return InsurEnv(windows[index][1], budget, amount_scale=metadata["amount_scale_train_median"], reward_mode=mode)
                    q, job = dqn.train(make_env, len(windows), 23, seed, cfg["dqn"])
                    job.update({"reward_mode": mode, "budget_index": b, "budget": budget,
                                "protocol_sha256": lock["protocol_sha256"], "source_sha256": metadata["source_sha256"],
                                "prepared_sha256": metadata["prepared_sha256"]})
                    dqn.save(model_path, q, job)
                    job["model_sha256"] = sha256(model_path)
                    write_json(sidecar, job)
                jobs.append(job)
                print(f"trained/reused {mode} budget={b} seed={seed}", flush=True)
    write_json(output / "training.json", {"protocol_sha256": lock["protocol_sha256"], "prepared_sha256": metadata["prepared_sha256"], "jobs": jobs})
    return jobs


def policy_action(name, env, rng, threshold, q, obs):
    if name == "reject_all":
        return 0
    if name == "accept_feasible":
        return 1
    if name == "random":
        return int(rng.integers(2))
    if name == "amount_threshold":
        return int(env.current_amount <= threshold)
    if name == "myopic":
        return int(env.counterfactual_delta() > 0)
    if name == "dqn":
        from .dqn import predict
        return predict(q, obs)
    raise ValueError(f"unknown policy {name}")


def evaluate(output, split="validation"):
    from . import dqn
    output = Path(output)
    lock, metadata = prepared(output)
    cfg = lock["config"]
    if split not in ("validation", "test"):
        raise ValueError("evaluation split must be validation or test")
    if split == "test" and not (output / "validation_metrics.csv").is_file():
        raise ValueError("run frozen validation evaluation before opening test evaluation")
    target = output / f"{split}_metrics.csv"
    if target.exists():
        raise FileExistsError("evaluation exists; refusing to overwrite held-out evidence")
    windows = episodes(load_split(output, split), cfg["episode_size"])
    results = []
    for mode in cfg["reward_modes"]:
        for b, budget in enumerate(metadata["budgets"]):
            for seed in cfg["seeds"]:
                q = None
                if "dqn" in cfg["policies"]:
                    path = output / "models" / model_name(mode, b, seed)
                    sidecar = read_json(path.with_suffix(".json"))
                    validate_job(sidecar, lock, metadata, mode, b, seed)
                    if sidecar["model_sha256"] != sha256(path):
                        raise ValueError("model integrity mismatch")
                    q, model_meta = dqn.load(path)
                    validate_job(model_meta, lock, metadata, mode, b, seed)
                for policy in cfg["policies"]:
                    for window_number, (window_id, frame) in enumerate(windows):
                        rng = np.random.default_rng(np.random.SeedSequence([seed, window_number]))
                        env = InsurEnv(frame, budget, amount_scale=metadata["amount_scale_train_median"], reward_mode=mode)
                        obs, _ = env.reset(seed=seed)
                        started = perf_counter()
                        done = False
                        while not done:
                            action = policy_action(policy, env, rng, metadata["amount_threshold_train_median"], q, obs)
                            obs, _, terminated, truncated, _ = env.step(action)
                            done = terminated or truncated
                        results.append({"split": split, "reward_mode": mode, "budget_index": b,
                            "initial_budget": budget, "seed": seed, "policy": policy,
                            "episode_id": window_id, "year": int(window_id[:4]),
                            "terminal_distance": env.distance, "total_reward": env.total_reward,
                            "accepted_count": env.accepted, "claim_coverage": env.accepted / len(frame),
                            "value_coverage": env.spent / float(frame.amount.sum()),
                            "budget_utilization": env.spent / budget, "steps": env.position,
                            "episode_rows": len(frame), "runtime_seconds": perf_counter() - started,
                            "empty_selected_cohort": env.accepted == 0})
                print(f"evaluated {split} {mode} budget={b} seed={seed}", flush=True)
    pd.DataFrame(results).to_csv(target, index=False)
    write_json(output / f"{split}_evaluation.json", {
        "protocol_sha256": lock["protocol_sha256"], "source_sha256": metadata["source_sha256"],
        "prepared_sha256": metadata["prepared_sha256"],
        "metrics_sha256": sha256(target), "rows": len(results),
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "no_tuning": "all hyperparameters fixed before preparation; no model selection on validation/test"})
    return pd.DataFrame(results)


def paired_summary(frame, bootstraps=1000):
    """Paired DQN-minus-myopic differences, crossed seed/year cluster bootstrap."""
    keys = ["split", "reward_mode", "budget_index", "seed", "episode_id", "year"]
    left = frame[frame.policy == "dqn"]
    right = frame[frame.policy == "myopic"]
    if left.empty or right.empty:
        return []
    matched = left.merge(right, on=keys, suffixes=("_dqn", "_myopic"), validate="one_to_one", how="outer", indicator=True)
    if not matched["_merge"].eq("both").all():
        raise ValueError("missing paired DQN/myopic window rows")
    matched["difference"] = matched.terminal_distance_dqn - matched.terminal_distance_myopic
    result = []
    for (split, mode, budget), group in matched.groupby(["split", "reward_mode", "budget_index"]):
        cells = group.groupby(["seed", "year"]).difference.mean().unstack("year")
        values = cells.to_numpy()
        if not np.isfinite(values).all():
            raise ValueError("unpaired seed/year cells")
        rng = np.random.default_rng(1729)
        draws = [float(values[np.ix_(rng.integers(values.shape[0], size=values.shape[0]),
                                     rng.integers(values.shape[1], size=values.shape[1]))].mean())
                 for _ in range(bootstraps)]
        result.append({"split": split, "reward_mode": mode, "budget_index": int(budget),
            "comparison": "dqn_minus_myopic_terminal_distance; negative favors DQN",
            "difference_equal_year_mean": float(values.mean()),
            "conditional_95_percent_interval": [float(x) for x in np.quantile(draws, [.025, .975])],
            "independent_training_seeds": values.shape[0], "calendar_year_clusters": values.shape[1],
            "paired_windows_per_seed": int(group.episode_id.nunique()),
            "limitation": "few historical year clusters; conditional descriptive interval, not fraud efficacy or broad population inference"})
    return result


def report(output):
    output = Path(output)
    lock, metadata = prepared(output)
    frames = []
    for split in ("validation", "test"):
        path = output / f"{split}_metrics.csv"
        if path.exists():
            evaluation = read_json(output / f"{split}_evaluation.json")
            if evaluation["metrics_sha256"] != sha256(path) or evaluation["protocol_sha256"] != lock["protocol_sha256"] or evaluation.get("prepared_sha256") != metadata["prepared_sha256"]:
                raise ValueError("evaluation integrity mismatch")
            checked = pd.read_csv(path)
            unique_keys = ["split", "reward_mode", "budget_index", "seed", "policy", "episode_id"]
            if checked.duplicated(unique_keys).any():
                raise ValueError("duplicate evaluation rows")
            expected = (len(lock["config"]["reward_modes"]) * len(metadata["budgets"]) * len(lock["config"]["seeds"])
                        * len(lock["config"]["policies"]) * metadata["splits"][split]["full_windows"])
            if len(checked) != expected or set(checked.policy) != set(lock["config"]["policies"]):
                raise ValueError("incomplete evaluation matrix")
            window_ids = [name for name, _ in episodes(load_split(output, split), lock["config"]["episode_size"])]
            expected_keys = set(itertools.product([split], lock["config"]["reward_modes"],
                range(len(metadata["budgets"])), lock["config"]["seeds"], lock["config"]["policies"], window_ids))
            if set(checked[unique_keys].itertuples(index=False, name=None)) != expected_keys:
                raise ValueError("evaluation matrix has unexpected or missing identities")
            frames.append(checked)
    if not frames:
        raise ValueError("no completed evaluations")
    frame = pd.concat(frames, ignore_index=True)
    columns = ["terminal_distance", "claim_coverage", "value_coverage", "budget_utilization", "runtime_seconds", "total_reward"]
    summary = frame.groupby(["split", "reward_mode", "budget_index", "policy"])[columns].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index().fillna(0)
    summary.to_csv(output / "summary.csv", index=False)
    result = {"protocol_sha256": lock["protocol_sha256"], "source_sha256": metadata["source_sha256"],
              "summary": summary.to_dict("records"), "paired_comparisons": paired_summary(frame),
              "interpretation": ["First-digit distribution proxy only; no independently labeled fraud outcomes.",
                  "Policy-date cohorts are not verified claim-arrival sequences. Amounts are already PAID indemnities: this is retrospective record selection, not prospective payout approval.",
                  "Coverage and budget use must accompany distance; rejecting everyone has sentinel distance 1.",
                  "Primary reward telescopes to 1 minus terminal distance. Signed counterfactual is a separate ablation.",
                  "Repeated deterministic baseline seeds are paired controls, not additional independent evidence.",
                  "No clinical, underwriting, fraud-loss or production validation is established."]}
    write_json(output / "report.json", result)
    lines = ["# Frozen InsurEnv-v1 benchmark", "", f"Protocol: `{lock['protocol_sha256']}`", "",
             "This is a historical first-digit proxy experiment, with no labeled fraud outcomes.", "",
             "## Interpretation", ""] + [f"- {x}" for x in result["interpretation"]]
    lines += ["", "## Paired DQN versus myopic comparison", ""]
    for item in result["paired_comparisons"]:
        lo, hi = item["conditional_95_percent_interval"]
        lines.append(f"- {item['split']}, {item['reward_mode']}, budget {item['budget_index']}: distance difference {item['difference_equal_year_mean']:.4f}; conditional interval [{lo:.4f}, {hi:.4f}], {item['independent_training_seeds']} seeds and {item['calendar_year_clusters']} years.")
    lines += ["", "Detailed aggregate metrics: summary.csv. No claim-level identities appear in report artifacts."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def controls(output):
    from .controls import run
    output = Path(output)
    lock = locked(output)
    target = output / "synthetic_controls.json"
    if target.exists():
        raise FileExistsError("synthetic controls already recorded")
    cfg = lock["config"].get("synthetic_controls", {"seed": 20261022, "rows": 256})
    result = run(seed=cfg["seed"], rows=cfg["rows"])
    result["protocol_sha256"] = lock["protocol_sha256"]
    write_json(target, result)
    pd.DataFrame(result["results"]).to_csv(output / "synthetic_controls.csv", index=False)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze"); f.add_argument("--config", required=True); f.add_argument("--output", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--input", required=True); p.add_argument("--output", required=True)
    p.add_argument("--id-column", default="claim_id"); p.add_argument("--date-column", default="claim_date"); p.add_argument("--amount-column", default="amount")
    for command in ("train", "evaluate", "report", "controls"):
        p = sub.add_parser(command); p.add_argument("--output", required=True)
        if command == "evaluate": p.add_argument("--split", choices=["validation", "test"], default="validation")
    args = parser.parse_args(argv)
    try:
        if args.command == "freeze": result = freeze(args.config, args.output)
        elif args.command == "prepare": result = prepare(args.input, args.output, args.id_column, args.date_column, args.amount_column)
        elif args.command == "train": result = {"trained_jobs": len(train(args.output))}
        elif args.command == "evaluate": result = {"evaluation_rows": len(evaluate(args.output, args.split))}
        elif args.command == "controls": result = controls(args.output)
        else: result = {"paired_comparisons": report(args.output)["paired_comparisons"]}
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
