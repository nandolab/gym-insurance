# InsurEnv

Reproducible reinforcement-learning research on **budget-constrained selection
of historical paid indemnity records** using an unlabeled digit-distribution proxy.

The corrected `InsurEnv-v1` implementation is a development preview. It does not
establish fraud-detection accuracy, prevented fraud, financial savings or suitability
for operational claim approvals. Package release, source/data rights confirmation
and independent scientific review remain open.

## Start here

- [Corrected v1 development branch](https://github.com/nandolab/gym-insurance/tree/refactor/insurenv-v1-consolidation)
- [Research protocol and environment contract](docs/PROTOCOL.md)
- [Reproduction instructions](docs/REPRODUCING.md)
- [Roadmap](ROADMAP.md) and [tracked issues](https://github.com/nandolab/gym-insurance/issues)
- [Contribution guide](CONTRIBUTING.md)
- [License status](LICENSE-STATUS.txt)

`main` is the public landing branch. The development branch contains the corrected
package and tests until its review is completed. Historical code remains available
in Git history and the `archive/feature-enhance-env-20261022` tag. Old experiments
and corrected v1 results are different protocols and must not be combined.

## Install the development preview

```sh
git clone --branch refactor/insurenv-v1-consolidation https://github.com/nandolab/gym-insurance.git
cd gym-insurance
python -m venv .venv
# Activate .venv for your shell, then:
python -m pip install -e ".[research,test]"
python -m pytest
python -m scripts.synthetic_smoke --output artifacts/my-first-smoke
```

Use Python 3.10 or newer as declared by the candidate package; the CI matrix shows
the combinations actually exercised. Always choose a new smoke output directory.
Synthetic data tests the pipeline and is not empirical fraud evidence.

## Research scope

The PSR study uses policy-contract dates as cohort ordering, not observed claim
arrival times. The amount field describes indemnity already paid. A virtual budget
constrains retrospective record inclusion. There are no independent fraud labels.

V1 optimizes first-digit total variation. Its primary reward measures actual
before/after improvement; the separately named signed-counterfactual reward is an
ablation. Baselines receive the same observations, ordered windows and budgets.
Training-only calibration and temporal splits are explicit in the saved protocol.

The fixed 12,000-step DQN candidate did not beat myopic proxy selection in the
recorded local study. The myopic policy selected only about 5.5–6.8% of records,
so a lower proxy distance does not establish business superiority. Only two held-out
policy years support the conditional descriptive uncertainty. Review/reproduction
is tracked separately from source availability.

## Repository layout

```text
gym_insurance/       Versioned environment and benchmark implementation (v1 branch)
configs/            Explicit experiment configuration (v1 branch)
tests/              Deterministic correctness and integrity tests (v1 branch)
scripts/            Synthetic workflow and research helpers (v1 branch)
docs/               Protocol, reproduction and project guidance
.github/            Issue templates, PR guidance and CI
ROADMAP.md          Ordered work and acceptance gates
```

Raw records, private reviews/correspondence, manuscripts, models, runtime caches
and local archives are excluded from public changes. Do not attach them to issues.
There is no automatic manuscript submission or package publishing workflow.
