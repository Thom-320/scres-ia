# Garrido risk-headroom sensitivity — terminal development verdict

**Date:** 2026-07-16

**Verdict:** `DEVELOPMENT_NO_DOOR_UNDER_TESTED_FRONTIER`

The source-faithful risk sensitivity completed all 4,860 frozen evaluations:
45 profiles, 18 Track-A constant postures, and six development seeds over the
ten-year R1/R2 horizon. The producer exited zero, the session watcher reached a
terminal empty PGID/SID, the remote and retrieved checksums match, and an
independent recomputation returned `PASS_GARRIDO_RISK_AUDIT` with no failures.

## What changed and what did not

The Garrido current/increased profiles and the frozen 1.5x/2x impact
sensitivities materially change physical system performance. For example, the
best attainable mean canonical ReT spans about 0.484–0.819 across the R2
frequency profiles. This is a genuine risk sensitivity result.

What does not change materially is the value of tailoring the tested Track-A
constant posture to the risk regime:

| Quantity | Maximum observed value | Gate |
|---|---:|---:|
| `H_profile_raw` | 0.00024068 | 0.01 |
| raw paired LCB95 | 0.00012982 | >0, but magnitude still 41.5x below gate |
| `H_profile_safe` | 0.00006931 | 0.01 |
| safe paired LCB95 | 0.00000000 | >0 required |
| passing doors | 0 | at least 1 required |

At the timing contract's load-bearing `R2_frequency`, resource cap 0.5:

- `H_profile_raw = 0.00024068`;
- `H_profile_safe = 0.00005042`, CI95 `[0, 0.00035070]`;
- the robust comparator is `f1_S1`;
- all resource and service guardrails are non-inferior;
- the door still fails by roughly two orders of magnitude.

R3 was not scaled and was excluded from the category sensitivity screen. The
primary endpoint remained `ret_excel_request_snapshot_v2`; no Cobb-Douglas or
temporal metric was used to select a posture.

## Consequence for endogenous timing

The frozen timing contract required a passing R2/profile door before accessing
seeds `7460001–7460048`. Because the independently audited count is zero, those
seeds remain unopened and the restricted timing experiment is closed without
execution. Running it anyway would violate the preregistered sequence and turn
risk intensity into a post-hoc rescue.

This null is bounded to the tested family: common strategic buffer fraction plus
assembly shifts. It does not prove that risks are irrelevant, that endogenous
timing can never matter, or that all domain-plausible decision rights are inert.
It shows that risk stress alone does not create material regime-tailoring
headroom for this Track-A frontier.

## Claim boundary

- `H_profile_safe` is a development regime-tailoring diagnostic, not H_PI and
  not H_obs;
- no timing ceiling was estimated;
- no learner is authorized;
- Paper 2 is not confirmed;
- Paper 3 remains blocked;
- a new route requires a genuinely new preregistered mechanism justified by
  domain facts, not a larger stress multiplier chosen after this result.

## Custody

- executed source commit: `01c1d9db480193e2919e9ade74252da5df4a4ccd`;
- contract SHA-256: `57310d977377acaf400979f7770833bf51b5e2ab15c6a8c62e354fa186c5478e`;
- result SHA-256: `e4a3d4a05d604234ac9ef0c34811e96870e0daf8747bda60385f0e0eb2ac2d35`;
- raw-row SHA-256: `5e2b6d7f03583641ec538ccb6149c442825f1417d1a7ee9837d25c272a1d8de7`;
- retrieved package SHA-256: `16d42277483defbbc722a6232eac4f1fe8c41e748df846ba24467505ddf7a55c`;
- evidence directory: `results/garrido_risk_headroom_sensitivity_v1/`.
