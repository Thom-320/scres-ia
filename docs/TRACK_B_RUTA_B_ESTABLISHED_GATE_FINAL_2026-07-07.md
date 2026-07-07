# Ruta B under the established causal gate — final verdict (2026-07-07)

## Question

Does the Ruta B checkpoint (live auxiliary belief loss, R22+R24 lead-2w λ=0.25,
5 seeds × 60k, `v10_no_forecast`, Case C) show ANY causal-prevention signal
under the validated counterfactual methodology — the same
`scripts/audit_track_b_risk_event_counterfactual.py` that produced the original
null on the reactive baseline
(`docs/TRACK_B_CASE_C_CAUSAL_COUNTERFACTUAL_VERDICT_2026-07-06.md`)?

This is the closing test after the retraction: the naive inlined gate (episode-
median calm action, no exclusion halo) was shown to produce ~70-90% false-
positive rates on a reactive baseline and on a permuted-label control
(`docs/TRACK_B_RUTA_B_COUNTERFACTUAL_GATE_AUDIT_2026-07-07.md`).

## Incident note: first attempt ran the wrong environment

The first VPS run of this comparison
(`outputs/experiments/track_b_established_gate_{reactive,ruta_b}_2026-07-07`,
now marked `INVALID_WRONG_ENV.md`) silently ignored
`--enabled-risks/--faithful/--risk-*-by-id`: the script was synced to the host
but its dependency `scripts/audit_track_b_prevention_mechanism.py` (whose
`env_kwargs()` resolves those flags) was stale. Detection: `mean_R_full ≈
0.0056` (all-risks branch-composition scale) instead of Case C's ~0.48, and
R24 events/episode ≈ 29 instead of ~90 (freq ×3 missing). The script now
echoes `resolved_env_kwargs` and `mean_R_full_overall` into `summary.json` so
this failure mode is self-evident in the output.

## Valid run (this verdict's basis)

- `outputs/experiments/track_b_established_gate_reactive_case_c_2026-07-07/`
- `outputs/experiments/track_b_established_gate_ruta_b_case_c_2026-07-07/`
- Sanity gates passed: `mean_R_full_overall` = 0.48116 (reactive) / 0.48109
  (Ruta B); `resolved_env_kwargs` shows `enabled_risks=(R22,R23,R24)`,
  `R24 freq ×3`, `R22/R23 impact ×1.5`, `year_basis=thesis`. Pair counts
  (274/127/377) match the valid 2026-07-06 reactive run exactly.
- Protocol: 5 seeds × 12 episodes, `max_steps=104`, pre-window [-4,-1],
  policy-specific calm action with (-4,+8) exclusion halo and
  lowest-intensity-quartile fallback.

## Results (recomputed per seed from raw CSV)

Pooled positive-pair rates and mean deltas:

| Policy | Risk | Pairs | Positive | Rate | Mean Δ ReT Excel |
|---|---|---:|---:|---:|---:|
| reactive PPO | R22 | 274 | 21 | 7.7% | +0.0000658 |
| reactive PPO | R23 | 127 | 14 | 11.0% | +0.000266 |
| reactive PPO | R24 | 377 | 14 | 3.7% | -0.0000012 |
| Ruta B | R22 | 274 | 32 | 11.7% | +0.000110 |
| Ruta B | R23 | 127 | 18 | 14.2% | +0.000196 |
| Ruta B | R24 | 377 | 28 | 7.4% | +0.0000203 |

Per-seed rates: reactive 3-12%, Ruta B 3-19% — every cell far below the ≥50%
promotion bar, medians ≈ 0, deltas one to two orders of magnitude below the
artifact-gate deltas. Interpretation column: `sin_senal_causal_clara` /
`pre_acciones_reducen_ReT_o_costo_de_oportunidad` throughout.

## Verdict

**Null, definitively.** Under sound counterfactual methodology and the correct
Case C environment, Ruta B shows no causal-prevention signal — its rates sit
in the same noise band as the reactive baseline (Ruta B is 2-4pp above
reactive on R22/R24, which is far inside the band that every negative result
this session has occupied, and nothing like the 50-90% the flawed gate
produced). The retraction of the preventive claim is confirmed with correct
numbers.

What remains standing about Ruta B (unaffected by this verdict):

- ReT parity with the reactive baseline (+10.05% vs best static) at ~55% of
  its resource cost (0.396 vs 0.719) — the efficiency/regularization result,
  currently being attributed via the control ladder (λ=0 head, constant-label,
  permuted-label arms).

## Chain of documents

1. Apparent positive: `TRACK_B_RUTA_B_LIVE_AUX_CASE_C_CONFIRM_VERDICT_2026-07-06.md` (erratum added)
2. Controls + retraction: `TRACK_B_RUTA_B_COUNTERFACTUAL_GATE_AUDIT_2026-07-07.md`
3. This closure: correct-env run of the trusted gate → null on both policies.

Next (per the post-retraction roadmap): preventive-headroom ceiling tests
(forced-prep response surface + clairvoyant PPO) decide whether ANY prevention
work continues in this environment.
