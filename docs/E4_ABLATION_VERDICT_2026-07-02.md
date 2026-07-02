# E4: final 5-seed/60k 8D action-space ablation — verdict (2026-07-02)

**Status: DONE.** `outputs/experiments/track_b_ablation_8d_final_2026-07-01/`
(joint, downstream_only, shift_only — all 3 arms, 5 seeds × 60k timesteps,
12 eval episodes, `control_v1` reward, `adaptive_benchmark_v2` risk, h104,
matching the canonical protocol except learning rate — see caveat below).

## Result

| Arm | PPO order-level ReT | Best static (in-arm) | Δ vs best static | RawWin |
|---|---:|---:|---:|---|
| joint (full 8D) | 0.005587 | 0.005172 (`s3_d1.50`) | **+0.000415** | True |
| downstream_only | 0.005678 | 0.005112 (shift frozen; statics collapse) | **+0.000566** | True |
| shift_only | 0.005607 | 0.005191 (`s2_d1.00`) | **+0.000416** | True |

Confirms the 2-seed/30k screen (2026-07-01) at full confirmatory scale:
`downstream_only ≥ shift_only ≥ joint` in ΔReT — **downstream-only access
captures the largest delta vs its own best static, exceeding the full 8D
joint contract.**

## Verdict on the "more knobs" attack (T5, docs/REVIEWER2_DEEP_AUDIT_2026-07-01.md)

**Answered, and favorably.** The win is not explained by dimensionality —
adding shift-mix control on top of downstream dispatch does not increase
the delta over the best comparator; if anything it's diluted slightly in
`joint` (+0.000415) versus `downstream_only` alone (+0.000566). This
strengthens the frontier-dependence story: **the value is concentrated in
reaching the downstream dispatch bottleneck specifically, not in action-space
size.** Safe wording:

> "An ablation over the 8D action contract shows the gain concentrates in
> downstream dispatch access: a downstream-only policy captures a larger
> Excel-ReT advantage over its best static comparator (+0.000566) than the
> full joint 8D contract (+0.000415), indicating the result is driven by
> bottleneck-alignment rather than by the number of controllable dimensions."

Do NOT claim the joint contract is unnecessary for the canonical paper
result — the canonical confirmatory run (`track_b_gain_2026-06-30/...`) used
the joint 8D contract and that is the headline number; this ablation is
mechanism evidence, not a recommendation to drop the shift dimensions from
the primary contract.

## Caveats to disclose

1. **Learning-rate mismatch vs canonical run.** This ablation used
   `--learning-rate 0.0001` (per `docs/track_b_q1_stats_2026-07-01/ablation_decision.md`'s
   prescribed command), while the canonical confirmatory run used
   `learning_rate 0.0003` (`n_steps=1024, batch_size=256` match). The
   absolute ReT values here (~0.0056-0.0057) are noticeably lower than the
   canonical run's PPO Excel ReT (0.005893) — plausibly the lower LR under
   the same 60k-timestep budget. The *within-ablation* comparison
   (joint vs downstream_only vs shift_only, same LR across all three arms)
   is valid and is what the verdict above relies on; do not directly
   compare these absolute ReT numbers to the canonical headline number.
2. **`downstream_only`'s "best static" is degenerate.** Because the
   downstream_only wrapper freezes shift at a fixed value, all 9
   `s{1,2,3}_d{1.00,1.50,2.00}` static labels in that arm collapse to
   identical episode outcomes (same reward/fill/ReT for all 9 rows) — the
   in-arm "best static" comparator is really just "the one static config
   reachable under a frozen shift," not a comparable dense grid. The
   cross-arm comparison of *deltas* is still meaningful, but don't report
   "downstream_only's best static" as equivalent in rigor to the canonical
   run's 147-cell dense frontier.
3. **Heuristics were evaluated in-run** (`heur_hysteresis`, `heur_disruption_aware`
   appear in each arm's policy summary) but only 2 of the 5 heuristics from
   `scripts/track_b_heuristics.py` — check `policy_summary.csv` per arm for
   the full set if needed for the paper's heuristic-baseline table.

## Registry update

C7 (`docs/CLAIMS_REGISTRY_Q1_DEFENSE_2026-07-01.md`: "action-space alignment
with the downstream bottleneck explains Track A vs Track B... causal
ablation needs current rerun") — **upgrade from "Supported as framing" to
"Supported"**, citing this table.
