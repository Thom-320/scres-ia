# Program O rho90/share90 causal diagnostic

## Verdict

`TECHNICAL_INVALIDATION_REQUIRES_CORRECTIVE_REPEAT`

The sealed validation execution cannot support either a PASS or the previously reported terminal STOP. Two exact implementation defects affected the frozen gate:

1. The contract said that the strongest open-loop calendar was selected on burned development data and frozen before validation. The executor instead selected a new point comparator on the sealed validation block in every cell.
2. The simultaneous inference used one unstandardized maximum deviation across 69 estimands with incompatible units. Service-loss AUC drove a critical value of `23,651,319.39`, which was then subtracted from unit-scale ReT and placebo contrasts.

Neither defect changes the physical trajectories, policy actions, raw matrices, metrics, mass ledgers, or resource equality. They invalidate the adjudication layer.

## Comparator drift was outcome-determinative

| Cell | Frozen development static | Validation-selected static | ΔReT vs frozen static | Favorable vs frozen static | Reported favorable |
|---|---:|---:|---:|---:|---:|
| rho75/share90 | 39599 | 27391 | 0.08642 | 44/48 | 45/48 |
| rho90/share75 | 39599 | 39663 | 0.06903 | 45/48 | 44/48 |
| rho90/share90 | 31743 | 49151 | 0.09205 | **46/48** | **26/48** |

The earlier 26/48 STOP therefore does not measure performance against the comparator frozen before validation. It arose because the sealed tapes were used to choose a different C-heavy calendar `[2,3,3,3,3,3,3,3]` that was near-optimal on the C-dominant subset.

## What caused the apparent 22-tape failure

The 22 reported non-favorable tapes were not explained by a scheduler, metric, CRN, mass, or physical-replay defect:

- emitted calendars reconstructed exactly;
- every stored per-tape delta reconstructed exactly;
- physical replay failures: 0/1,423;
- unique scheduled-resource vectors: 1;
- action and state-counterfactual audits passed.

They were campaigns with a mean first-half C share of `0.905`, versus `0.279` in the favorable group. On those tapes the validation-selected C-heavy static calendar was at the 96.9th percentile of all 65,536 calendars, versus the 31.9th percentile in the favorable group. Static-calendar percentile correlated `−0.943` with policy-minus-static ReT.

This is primarily a comparator-selection/label-regime interaction, not commitment delay:

- current-week action mismatch was slightly lower, not higher, in the non-favorable group;
- next-week mismatch was also lower;
- phase shifts and regime transitions were smaller;
- belief error differed little;
- wrong-product positive inventory was higher and amplified losses, but was secondary to the near-oracle C-heavy comparator.

The visible distribution was genuinely split under the validation-selected comparator: mean `0.0829`, median `0.0276`, lower quartile `−0.0615`, upper quartile `0.2242`. This split is descriptive only because the comparator was not the frozen one.

## Corrected descriptive inference

A studentized one-sided max-t audit replaced the invalid raw-scale maximum. It is descriptive and cannot promote the burned result.

Against the validation-selected comparator:

- all three primary simultaneous LCBs exceeded `0.01`;
- all 27 information-placebo LCBs were positive;
- CVaR10 non-inferiority failed simultaneously in all three cells.

Against the frozen development comparators:

- primary simultaneous LCBs were `0.0564`, `0.0429`, and `0.0557`;
- all 27 placebo LCBs were positive;
- CVaR10 non-inferiority remained unresolved in rho75/share90 and rho90/share75 (`−0.0288` and `−0.0235` LCBs).

Thus the diagnostic identifies a real technical invalidation but does not retrospectively establish H_obs. A clean corrective validation must decide the tail guardrail prospectively.

## Decision

- Previous terminal STOP: retracted as an adjudication result.
- H_obs confirmed: no.
- Learner authorized: no.
- Corrective repeat on a fresh seed block: licensed.
- Policy, cells, physics, metric, placebos, favorable threshold, and guardrails: unchanged.
- Corrective changes allowed: explicit frozen comparator indices and studentized simultaneous inference only.

If that corrective run fails any frozen gate, Program O closes without another rescue. If it passes, it establishes classical H_obs and only then licenses a separately frozen learner gate.
