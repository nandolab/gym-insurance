"""Run the actual pipeline on deterministic SYNTHETIC data, with short DQN training."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from gym_insurance import benchmark as b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists():
        raise SystemExit("choose a fresh output directory")
    out.mkdir(parents=True)
    rng = np.random.default_rng(20261022)
    rows = []
    for year in (2017, 2018, 2019):
        for i in range(48):
            rows.append({"claim_id": f"SYNTHETIC-{year}-{i}",
                         "claim_date": f"{year}-01-{i // 2 + 1:02d}",
                         "amount": round(float(10 ** rng.uniform(1, 3)), 2)})
    pd.DataFrame(rows).to_csv(out / "SYNTHETIC.csv", index=False)
    cfg = b.read_json(Path(__file__).resolve().parents[1] / "configs/psr-v1.json")
    cfg.update(experiment="SYNTHETIC-smoke-not-research-results", episode_size=12,
               train_end="2017-12-31", validation_end="2018-12-31", test_end="2019-12-31", seeds=[0])
    cfg["dqn"].update(total_steps=128, warmup_steps=16, batch_size=16,
                      replay_size=256, train_frequency=1, target_update_steps=32)
    b.write_json(out / "smoke_config.json", cfg)
    b.freeze(out / "smoke_config.json", out)
    b.controls(out)
    b.prepare(out / "SYNTHETIC.csv", out)
    b.train(out)
    b.evaluate(out, "validation")
    b.evaluate(out, "test")
    b.report(out)
    print("Synthetic smoke completed; these are software checks, not empirical research claims.")


if __name__ == "__main__":
    main()
