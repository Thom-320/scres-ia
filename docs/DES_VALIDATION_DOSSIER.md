# DES validation dossier for Program L(e-1)

**Contract:** `garrido_learning_v1`  
**Physical reference:** `garrido_proxy_v1`  
**Status:** Gate 0 v3 PASS; Gate 1 v3 complete; Gate 2 v3 terminal stop. No
powered PPO authorization.

## Claim boundary

The physical environment is a thesis-grounded reconstruction with audited
physical causality and a disclosed attribution proxy. The failed
`garrido_reference_v2` distribution gate is not relabelled. Attribution-sensitive
Garrido claims remain bounded by
`supply_chain/data/garrido_proxy_v1_freeze_2026-07-10.json`.

## Runtime freeze

Observed local scientific runtime on 2026-07-10:

| Component | Version |
|---|---:|
| Python | 3.11.15 |
| Gymnasium | 1.3.0 |
| Stable-Baselines3 | 2.9.0 |
| NumPy | 2.4.6 |
| SimPy | 4.1.2 |
| PyTorch | 2.12.1 |
| psutil | 7.2.2 |

- `requirements-pinned.txt` SHA256:
  `a081502d88ec8275084c88e6571284886f70a9add48df858a57eaa937b54150b`
- `garrido_proxy_v1` SHA256:
  `3d7aaa14263596cd68dce8a79f1f04bcb8fca9639c10cd9ac8160586fda4ed96`

The historical DQN transfer test can fail intermittently on macOS inside
`psutil.virtual_memory()` (`host_statistics64 ... array not large enough`). Program
L uses on-policy PPO and does not invoke a DQN replay buffer. The full
confirmatory runtime must nevertheless record a clean `psutil` probe before launch;
an unresolved failure is a platform limitation, not a scientific result.

## Frozen environment behavior

Program L adds a wrapper and does not redefine Track A/B/C:

- categorical S1/S2/S3 action;
- one-week symmetric activation delay;
- common S1 physical warm-up across policies within each strategic buffer;
- treatment begins at endogenous physical warm-up completion;
- strategic buffer chosen at reset and invariant during the campaign;
- no physical state carried across reset;
- no privileged risk/regime/forecast observation;
- risk-free warm-up followed by a materialized, hash-addressed post-warm-up
  risk calendar replayed identically across policies;
- fixed calibration-only observation normalization;
- operational reward separated from Excel ReT evaluation.

The base wrapper now passes the complete proxy fields into `MFSCSimulation`,
including recovery-period mode, R24 attribution window, Op9 dispatch policy and
downstream transport mode. Defaults remain backward compatible.

## Gate 0 checks

| Check | Required artifact/test | Current status |
|---|---|---|
| Mass conservation | existing Tier-1 proxy audit + regression suite | inherited PASS; rerun required before confirmatory |
| Risk liveness R11-R24/R3 | existing liveness audit | inherited PASS; R3 excluded from training |
| Same seed/tape/actions | exact trajectory equality | implemented and passing in `test_l_program_env.py` |
| Action purity | only `assembly_shifts` and thesis batch coupling may change | implemented and passing |
| Buffer invariance | targets identical after every weekly step | implemented and passing |
| One-week shift lag | requested at k, effective at k+1 | implemented and passing |
| Reset isolation | post-warm-up state reproduces under the same seed | implemented and passing |
| Observation non-privilege | no regime/risk/future/forecast fields | implemented and passing |
| Normalizer freeze | immutable stats and field lock | implemented and passing |
| Campaign hash | serialized tape hash verification | implemented and passing |
| System TTR | risk clustering, baseline, recovery and censoring | implemented; synthetic test passing |
| Excel ReT identity | all L runners use `ret_excel` | test and runner guard PASS |
| Demand CRN | common S1 start; compare post-warm-up schedules | PASS under S1-vs-S3 policy test |
| Risk-calendar CRN | materialize after risk-free warm-up, then replay | PASS; 60 unique calibration tape hashes |
| Prefix replay | bitwise identity before branch | PASS across 600 states / 1,800 branches |

## Metric semantics

- `ret_excel`: primary Garrido Excel ReT.
- `service_loss_auc_ration_hours`: exact post-promise order lateness area.
- `late_backlog_hours`: incremental dense version of the same quantity-hour
  concept used for training.
- `total_backlog_hours`: total order-wait quantity-hours used for training.
- `rpj_mean`, `rpj_p95`: order-attribution recovery periods. Historical
  `ttr_mean/ttr_p95` aliases remain only for artifact compatibility.
- `system_ttr_*`: recovery of weekly fill/backlog after compound risk clusters.

System TTR clusters events separated by less than 168 hours. Its pre-onset
baseline is the median of four weeks. Recovery requires two consecutive weekly
observations with fill at least 95% of baseline and backlog no more than 105% of
baseline. Open clusters are right-censored.

## Gate 0 promotion rule

Gate 0 v3 passed locally. Gate 2 v3 subsequently returned
`STOP_NO_DEPLOYABLE_ADAPTIVE_HEADROOM`; therefore Program L remains blocked before
powered PPO regardless of Gate-0 readiness. See
`docs/L_PROGRAM_GATE2_VERDICT_2026-07-10.md`.
