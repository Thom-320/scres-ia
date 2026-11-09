# SCRES: supply-chain simulation and policy evaluation

**When does an apparent reinforcement-learning gain survive a fair comparison?**

SCRES contains a Python/SimPy reconstruction of a thirteen-operation supply-chain
model, Gymnasium environments, policy experiments and evaluation tools. It is
research by Thomas Chisica with Alexander Garrido, building on the MFSC model
described in Garrido-Rios (2017). The reconstruction and its extensions should not
be confused with complete behavioral replication of the source model.

The most useful entry point is the **evaluation work**, not a training curve:
which decisions each policy is allowed to make, how calibration is separated
from evaluation, and which conclusions remain after stronger controls.

## Where to start

| What to inspect | Why it matters |
| --- | --- |
| [Track B same-contract verdict](docs/TRACK_B_SAME_CONTRACT_CHALLENGE_VERDICT_2026-07-10.md) | Across 10 PPO checkpoints and 60 held-out streams, a calibration-selected constant policy with the same decision rights outperformed canonical PPO. Earlier gains depended on a restricted comparator family. |
| [Later Program Q evidence status](docs/PROGRAM_Q_CANONICAL_EVIDENCE_STATUS_2026-08-25.md) | A separate contract found feedback value over open-loop calendars, but no material neural premium over the evaluated classical controllers; the per-product guardrail failed. This is not a deployment-safety claim. |
| [Reproducibility and artifact map](docs/REPRODUCIBILITY.md) | Maps implementations, configurations and stored artifacts. Read each artifact with its dated verdict. |

These are **different experiments with different contracts**, not estimates of one
RL effect. Large custody bundles are not all on GitHub, so this is not a complete
raw-artifact package for every historical run.

## Contribution and scope

- Simulator reconstruction and explicit environment/action interfaces.
- Policy-comparison experiments with separate calibration and held-out streams.
- Automated tests, source-model checks and tools for trajectory export.
- Documentation of comparator mismatches, claim revisions and collaboration interfaces.

There is no general claim here that PPO beats strong classical policies, that
resilience has been universally learned, or that an intended manuscript is a
published paper. Source attribution, physics fidelity, endpoints and decision
rights are part of the scientific question.

## Repository map

| Location | Purpose |
| --- | --- |
| [`supply_chain/`](supply_chain/) | DES model, configuration and Gymnasium interfaces |
| [`scripts/`](scripts/) | Evaluation, diagnostics, export and historical experiment runners |
| [`tests/`](tests/) | Implementation and claim-integrity checks |
| [`contracts/`](contracts/) | Explicit experiment specifications |
| [`results/`](results/) | Tracked result records and audit summaries |
| [`docs/`](docs/) | Dated methods, verdicts and provenance |

## Setup and checks

Python 3.11 is the documented compatibility target. From a fresh clone:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Inspect the relevant test files before choosing a test scope. The full suite is
not a promise of a quick smoke run:

```bash
python -m pytest tests/
```

To inspect the static-simulation command without launching an experiment:

```bash
python run_static.py --help
```

Simulation outputs and validation reports are written under `outputs/`; do not
overwrite a frozen experiment directory. Training and large replay commands are
deliberately not the default getting-started path. Their original configurations
remain in the [artifact map](docs/REPRODUCIBILITY.md) and Git history; running them
does not reopen a stopped scientific claim.

## Interpreting older documentation

The repository retains a long experimental history. In particular:

- For **Track B**, use the [same-contract verdict](docs/TRACK_B_SAME_CONTRACT_CHALLENGE_VERDICT_2026-07-10.md), not the earlier restricted-frontier wins.
- For **Program Q**, use the [August 25 reconciliation](docs/PROGRAM_Q_CANONICAL_EVIDENCE_STATUS_2026-08-25.md), not an older reservation record saying its seeds were unopened.
- The [repository source-of-truth record](docs/REPOSITORY_SOURCE_OF_TRUTH.md) describes the Program O boundary and a later addendum. It is not a substitute for the subsequent Program Q adjudication.
- [Older claim registries](docs/CLAIMS_REGISTRY_Q1_DEFENSE_2026-07-01.md), training plots and contributor handoffs remain provenance; they can contain superseded instructions or interpretations.

Scientific changes require an explicit question, comparable decision rights,
defined outcomes and a documented evaluation protocol. A passing test or a
positive learning curve is not sufficient evidence of adaptive superiority.
