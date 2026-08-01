# Garrido–WRAP–SCRES+AI implementation status

**Contract:** `garrido_wrap_scres_ai_v1`  
**Overall status:** `HOLD_WRAP_BEHAVIORAL_FIDELITY`  
**Claim status:** `DEVELOPMENT_ONLY`

This report records what the implementation currently establishes. It does not promote
the development artifacts into paper claims.

## Source and WRAP audit

Command:

```bash
.venv/bin/python scripts/audit_garrido_wrap_sources.py
```

Artifact: `results/garrido_wrap_source_audit/result.json`

- The repository design contains all 90 published Cf rows.
- The two genuine raw workbooks cover Cf1–Cf20.
- Cf21–Cf90 are regenerated from the published thesis design, not author-delivered data.
- `Rsult_1.xlsx` contains Cf1–Cf12 plus aggregate sheets and remains secondary.
- The ReT endpoint remains provisional because the exact `ΣBt` semantics are unresolved.
- The existing endogenous behavioral fidelity gate remains held.

## Q1 result

Command:

```bash
.venv/bin/python scripts/build_garrido_fig5_surrogate.py \
  --output results/garrido_wrap_q1/result.json \
  --sesoi-r2 0.05
```

Artifact: `results/garrido_wrap_q1/result.json`

The driver-to-ReT version of Figure 5 is an identity and is not a learning task. In the
valid held-out task `(rho, risk design) -> ReT`:

| Model | Mean R² | Difference from linear | Paired 95% CI |
|---|---:|---:|---:|
| Linear | 0.9697 | — | — |
| Backprop/MLP | 0.9863 | +0.0166 | [0.0048, 0.0283] |
| KAN | 0.9913 | +0.0216 | [0.0042, 0.0390] |

Neither neural model reaches the preregistered `SESOI_R2 = 0.05`. The machine-readable
decision is:

```text
NO_GO_NEURAL_PREMIUM_IN_WRAP_PANEL
selected_model_before_gates = linear_null
promotion_eligible = false
```

This answers Q1 for the current WRAP panel as: neural networks are a theoretically
compatible family, but the panel does not demonstrate a practically meaningful neural
premium over a linear rule.

## Q2 implementation and smoke

Runner: `scripts/run_garrido_wrap_closed_loop.py`  
Artifact: `results/garrido_wrap_q2_smoke_2016h/result.json`

The runner implements the between-run interface:

```text
choose Cf -> run strict WRAP DES -> observe ReT/service -> update learner -> choose next Cf
```

The smoke used one R1r campaign, six selections, a 2,016-hour development horizon, and
the linear null. OFAT/no-update selected `Cf1…Cf6`; the active retained/reset arms selected
`Cf1, Cf2, Cf3, Cf31, Cf34, Cf36`. The retained/reset difference was zero in this single
campaign, which is expected as a smoke result and is not an H4 confirmation.

The runner includes retained, reset, no-update, and OFAT arms, same-Cf seeds, explicit
oracle provenance, and a hard hold status. It refuses a neural learner unless Q1 has
produced an explicitly promoted model or `--allow-development-learner` is supplied.

## Verification

```text
tests/test_garrido_wrap_scres_ai_contract.py       7 passed
tests/test_garrido_excel_ret.py                    18 passed
tests/test_ret_metric_invariants.py                2 passed, 2 xfailed (pre-existing)
tests/test_garrido_replication_harness.py          included in 33 passed
tests/test_thesis_faithful_lane.py                 included in 33 passed
ruff                                              passed
```

The next scientific step is not to tune the learner. It is to close or formally preserve
the WRAP behavioral and `ΣBt` metric holds, then run Q2 with virgin evaluation campaigns
and enough matched seeds for a retained-versus-reset interval.

