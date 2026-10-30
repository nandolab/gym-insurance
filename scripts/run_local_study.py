"""Run/resume the reviewed local-snapshot protocol; never overwrite holdout results."""
import argparse
import json
from pathlib import Path

from gym_insurance import benchmark as b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--config", default="configs/psr-v1.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if b.sha256(args.input) != args.expected_source_sha256.lower():
        raise SystemExit("source hash differs from reviewed provenance")
    out = Path(args.output)
    if not (out / "frozen_protocol.json").exists():
        lock = b.freeze(args.config, out)
    else:
        lock = b.locked(out)
        if b.json_hash(b.read_json(args.config)) != lock["config_sha256"]:
            raise SystemExit("provided configuration differs from frozen protocol")
    print("FROZEN_PROTOCOL", lock["protocol_sha256"], flush=True)
    if not (out / "prepared.json").exists():
        metadata = b.prepare(args.input, out)
    else:
        _, metadata = b.prepared(out)
        if metadata["source_sha256"] != args.expected_source_sha256.lower():
            raise SystemExit("prepared source differs from reviewed provenance")
    print("PREPARED", json.dumps({"splits": metadata["splits"], "budgets": metadata["budgets"]}), flush=True)
    if not (out / "synthetic_controls.json").exists():
        b.controls(out)
    b.train(out)
    for split in ("validation", "test"):
        if not (out / f"{split}_metrics.csv").exists():
            b.evaluate(out, split)
    b.report(out)
    print("COMPLETED", str(out.resolve()), flush=True)


if __name__ == "__main__":
    main()
