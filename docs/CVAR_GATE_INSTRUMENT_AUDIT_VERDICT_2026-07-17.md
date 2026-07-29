# CVaR gate instrument audit — verdict (2026-07-17)

**Contract:** `contracts/cvar_gate_instrument_audit_v1.json` (frozen before execution)
**Script:** `scripts/audit_cvar_gate_instrument.py` · **Result:** `results/program_o/cvar_gate_instrument_audit_v1/result.json`
**Data:** burned corrective-validation tapes 7430001–48 only. No sealed or virgin seed touched.

## Status

```
INSTRUMENT_DEFECT_CERTIFIED_PENDING_INDEPENDENT_ADJUDICATION
```

## Findings (every number from the populated result)

- **M0 — machinery identity: PASS.** The audit replica reproduces the published simultaneous
  critical value exactly (2.8358 vs 2.8358) and the published per-cell CVaR LCBs within 5e-4,
  using the original seed and 10,000 resamples. Everything below runs on certified-identical
  machinery.
- **A3 — trivial-equivalence control: pass probability 0.0** (0/600 tape-resampled worlds). A
  policy whose true tail effect is EXACTLY ZERO — with the real noise structure of this design —
  never passes the margin-0 joint gate. A calibrated non-inferiority instrument should pass a
  truly equivalent policy with high probability; this one passes it never. (The degenerate
  literal self-comparison, delta ≡ 0, passes via the zero-variance exact bound — an analytic
  edge case, not a usable pass path.)
- **A1 — power: the minimum TRUE CVaR10 improvement for 80% pass probability at margin 0 is
  ≈ +0.079** — more than twice the observed tail point estimates (+0.035/+0.020) and larger than
  the confirmed MEAN effect itself. The gate demanded tail *superiority at clairvoyant scale*
  under the label "non-inferiority".
- **A2 — positive controls: both oracles PASS in all 3 cells** (per-tape mean-oracle and
  tail-oracle over the full 65,536 frontier; metric-family-only, a-fortiori direction disclosed).
  The gate is not infinitely impossible — it is passable exactly by policies with
  clairvoyant-scale tail improvements, which is what a superiority test admits.
- **A4 — margins (report-only, grid frozen in the contract):** power curves at −0.01 and −0.02
  are in the result JSON; no margin is applied retroactively to any verdict.

## What this does and does not change

- `STOP_PROGRAM_O_AFTER_CORRECTIVE_VALIDATION` is **unchanged** (its `no_post_failure_changes`
  covers thresholds and guardrails; this audit examined the instrument, not the verdict).
- Per the frozen decision rule, the defect certification is handed to the **independent auditor**,
  who alone adjudicates — together with Garrido's written M2 answer — whether a new learner
  contract (`program_o_ret_learner_v1`) with a properly calibrated, prospectively frozen
  acceptance rule (explicit margin, demonstrated power, canonical lens) may open on fresh sealed
  tapes. We do not adjudicate: we are the conflicted party.
- The manuscript gains the precise sentence: *the failed component was a zero-margin joint
  tail-safety gate that the instrument audit certifies as a de-facto superiority test —
  unpassable by genuinely non-inferior policies at this design's sample size (trivial-equivalent
  pass probability 0.0; 80%-power threshold ≈ +0.079).*
