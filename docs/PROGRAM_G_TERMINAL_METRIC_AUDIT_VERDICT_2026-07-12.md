# Program G terminal metric audit verdict — 2026-07-12

## Verdict

**`STOP_PROGRAM_G_NO_ROBUST_ADAPTIVE_VALUE_UNDER_STYLIZED_CONTRACT`.**

The corrective audit was frozen in commits `2818352` and `36bf0aa` before opening
calibration tapes `1040001+` (n=200) or the locked terminal test `1050001+` (n=400).
It corrected the 144-hour compressed-week defect, used the canonical ReT ledger with the
current order included in cumulative backlog/unattended counters, marked horizon-pending
orders lost, and added quantity-weighted ReT.

The full-contract periodic comparator selected on calibration was `ABAB`. No observable
policy passed all preregistered guardrails. In the locked test, cover minus ABAB was:

| Endpoint | Mean paired delta | CI95 | Required direction |
|---|---:|---:|---:|
| canonical ReT per order | -0.02317 | [-0.02816, -0.01893] | positive |
| quantity-weighted ReT | -0.00695 | [-0.00869, -0.00548] | non-negative |
| attended orders | -3.51 | [-4.15, -2.95] | non-negative |
| worst-CSSU fill | -0.1734 | [-0.2033, -0.1466] | non-negative |
| unfulfilled rations at horizon | +3,370 | [+2,831, +3,971] | non-positive |

MPC also failed every substantive guardrail. The two depth-3 trees collapsed to the same
tested behavior as cover and failed identically. Resource rights were equal by construction.

## Allowed claim

> Within the stylized Program G spatial order adapter, persistent tempo and imperfect advance
> signals did not yield robust observable adaptive value over the best periodic same-contract
> schedule after correcting calendar and ReT-ledger defects.

## Forbidden claims

- This is not a full Op1–Op13 DES validation.
- It is not proof that all spatial MFSC control or all RL is impossible.
- The already-opened G5 service-loss result is not virgin confirmation under canonical ReT.
- Cobb-Douglas-inspired rankings are construct sensitivities and cannot rescue the STOP.
- No further PPO, reward, horizon, signal-quality, or metric search is authorized in Program G.

Machine artifact: `results/program_g/terminal_metric_audit/verdict.json`.
