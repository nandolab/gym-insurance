"""Prespecified illustrative controls; no empirical claims about PSR fraud."""
import numpy as np
import pandas as pd

from .v1 import InsurEnv


def run(seed=20261022, rows=256):
    rng = np.random.default_rng(seed)
    variable_amounts = np.round(10 ** rng.uniform(1, 3, size=rows), 2)
    independent_labels = rng.random(rows) < .2
    fixtures = [
        ("all_legitimate_rounded_digit1", np.full(rows, 100.), np.zeros(rows, dtype=bool)),
        ("labels_independent_of_digits", variable_amounts, independent_labels),
        ("opposite_label_world_same_observables", variable_amounts, ~independent_labels),
    ]
    records, selections = [], {}
    policies = ("accept_feasible", "myopic", "random", "amount_threshold")
    for name, amounts, labels in fixtures:
        data = pd.DataFrame({"claim_id": [f"SYNTHETIC-{i}" for i in range(rows)],
                             "claim_date": pd.date_range("2017-01-01", periods=rows), "amount": amounts})
        threshold = float(np.median(amounts))
        for policy in policies:
            # Nonbinding budget isolates the proxy/label distinction in this control.
            env = InsurEnv(data, float(amounts.sum()) + 1., amount_scale=threshold)
            env.reset(seed=seed)
            policy_rng = np.random.default_rng(seed + 1)
            for i in range(rows):
                if policy == "accept_feasible": action = 1
                elif policy == "myopic": action = int(env.counterfactual_delta() > 0)
                elif policy == "random": action = int(policy_rng.integers(2))
                else: action = int(amounts[i] <= threshold)
                _, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated: break
            selected = np.array([r["accepted"] for r in env.records], dtype=bool)
            selections[(name, policy)] = selected
            records.append({"fixture": name, "policy": policy, "synthetic_rows": rows,
                            "terminal_distance": env.distance, "claim_coverage": float(selected.mean()),
                            "synthetic_fraud_prevalence": float(labels.mean()),
                            "selected_synthetic_fraud_fraction": float(labels[selected].mean()) if selected.any() else None,
                            "rejected_synthetic_fraud_fraction": float((~selected)[labels].mean()) if labels.any() else None,
                            "rejected_legitimate_fraction": float((~selected)[~labels].mean()) if (~labels).any() else None})
    paired_identical = all(np.array_equal(selections[("labels_independent_of_digits", p)],
                                          selections[("opposite_label_world_same_observables", p)]) for p in policies)
    return {"seed": seed, "rows_per_fixture": rows, "results": records,
            "identical_decisions_across_opposite_label_worlds": paired_identical,
            "interpretation": "Constructed examples only. Identical observed amounts and policy decisions admit different true labels. This demonstrates non-identifiability without labels, not that Benford is empirically invalid for PSR fraud."}
