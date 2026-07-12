# Reviewer-#2 pre-submission fixes — applied 2026-07-02

Source: four external Reviewer-#2 verdicts + literature/positioning review
(user-supplied 2026-07-02). All P0 wording fixes and both zero-compute
evidence upgrades are now IN the manuscript
(`docs/manuscript_current/submission/elsevier/`, compiles clean, 35 pp).

## Applied (manuscript edits)

1. **Dispatch-inclusive cost sensitivity added (§4.3, Table `tab:dispatch_cost`).**
   The single biggest attack ("dispatch is free") is now a *win*: from
   `docs/track_b_q1_stats_2026-07-02_final_10seed/dispatch_cost_sensitivity.csv`,
   PPO is nominally cheaper even at λ=0 in the 10-seed pool but CI overlaps
   zero; it becomes significantly cheaper (CI95 fully negative) at λ≥0.025.
   Mechanism: PPO
   expedites selectively (mean m10/m12 ≈ 1.30/1.27) vs static 2.00/1.50
   held permanently. Also in abstract + discussion. Registry C18.
2. **E3 corrected and reframed as "Cross-regime stress evaluation" (§4.7).**
   Found during this pass: the frozen matrix selected per-cell "best
   static" by a secondary criterion, flipping severe/h104 marginally
   positive and inflating h52 deltas. Recomputed against best in-cell
   static by the primary metric with seed-clustered CI95
   (`e3_per_cell_seed_ci.csv`): 4/6 positive (all with CI95>0, 5/5 seeds),
   severe negative at BOTH horizons (−0.000060, −0.000075) = boundary
   *regime*, disclosed in a footnote. Table 7 now carries CIs. fig5
   regenerated. Abstract/summary/conclusion updated; registry C11
   corrected (obsolete majority-win and inflated severe-cell values must
   never be cited again).
3. **Comparator scope renamed everywhere**: "dense downstream-dispatch
   static grid/frontier (shift × Op10 × Op12)"; Methods states upstream
   dims held at canonical settings and points to Track A frontier as the
   upstream bound.
4. **Cost index renamed** "shift-utilization cost index" (Methods §3.3,
   Table 4, discussion); "resource-efficient" → "shift-cost-dominant".
5. **ReT ceiling interpretation** (§4.3): ceiling 0.5/72 ≈ 0.006944;
   gain = 78.7% → 84.9% of ceiling ≈ +6.1 pp; tails lead the operational
   reading. Also one sentence in abstract.
6. **E4 softened**: §4.4 retitled "Action-space decomposition (mechanism
   ablation)"; "concentrates specifically" → sufficiency wording
   ("downstream access is sufficient and strongest; shift-only also
   captures part; not the only useful surface"); footnote now also
   discloses the degenerate downstream-only static comparator; summary
   and discussion updated; "causal" purged except as "not causal proof".
7. **Privileged-obs table fixed (metric mixing)**: was Excel-ReT rows +
   order-level masked row. Now two explicit columns (Excel / order-level)
   for all five rows (masked PPO Excel 0.005841 extracted from the VPS
   E2 artifacts).
8. **Reward sweep (§4.6) reframed**: "does not depend" → exploratory
   screen + chronology (canonical fixed before the confirmatory audit;
   screen run after as robustness); all 18 outcomes incl. 6 non-promoted
   reported.
9. **Track A**: boundary table added (`tab:track_a_boundary`: PPO/BC lose
   by −7.0e−6; oracle headroom +0.000176..+0.000296 unconverted);
   cross-lane metric-scale comparability note (0.155 vs 0.0059 never
   compared across lanes); "no publishable" → "no statistically and
   operationally material".
10. **New Methods §3.5 "Statistical inference"**: CRN (Law 2015), 16
    distinct scenarios → episode CIs descriptive, seed = inferential
    unit, top-12 winner's-curse recheck, secondary panel no-multiplicity
    note. PPO hyperparameters now stated (MlpPolicy 64×64 separate
    heads, lr 3e-4, n_steps 1024, batch 256, 10 epochs, γ 0.99, GAE 0.95,
    clip 0.2).
11. **Appendix A added (main.tex)**: top-12 static robustness table
    (all 12 deltas positive, CI95>0, delta grows down-ranking). C19.
12. **Dispatch operational meaning** (§3.2): expediting authority
    (frequency/vehicles/priority release), does not alter demand/risk/
    upstream physics; costs via sensitivity, not calibrated monetary.
13. **Reproducibility statement** (§3.4 end): repo, branch, artifact
    tracing, seeds.
14. **Bib**: `perezpena2025` → `badakhshan2024` (text said Perez-Pena,
    entry was Badakhshan/Mustafee/Bahadori); added Hopp & Spearman
    (Factory Physics), Law (Simulation Modeling & Analysis, CRN),
    Maggiar et al. 2025 (arXiv 2507.22040). "Dominant hybrid paradigm"
    → "most common hybrid pairing" (intro + discussion).
15. **Strong-baselines paragraph** (§2.5) citing Gijsbrechts/Stranieri/
    Maggiar: the bar is dense strong-comparator families, not naive rules.
16. **Table 1** caption: conceptual positioning matrix, not systematic
    coding. **Limitations**: PPO-only wording → "not the mechanism
    tested; SAC/TD3 natural extensions". Conclusion: "value appears" →
    "is observed in this benchmark".
17. **3×3 upstream static bound added after the initial fix pass.** The
    local bound varies Op3/Op9 quantity multipliers at the best downstream
    cell (S2, Op10×2.00, Op12×1.50). Best bound policy ReT `0.0056120`;
    PPO remains higher at `0.0056660`, seed-paired delta `+0.0000540`
    CI95 `[+0.0000424,+0.0000656]`. This supports "dense
    downstream-dispatch grid plus local upstream bound," not an exhaustive
    eight-dimensional static frontier claim.
18. **10-seed canonical Track B expansion completed and promoted (§4.3).**
    `docs/track_b_q1_stats_2026-07-02_final_10seed/` merges seeds 1-5 with
    the VPS seed expansion 6-10 and CRN-pairs all ten seeds against the
    exact headline static comparator. Excel ReT delta `+0.000438`, pooled
    CI95 `[+0.000409,+0.000468]`, seed-clustered CI95
    `[+0.000421,+0.000458]`; order-level ReT delta `+0.000426`; all ten
    seed-level deltas positive.

## Deliberately NOT done (per reviewer docs' own "don't" lists)

- No H4 reframing (the VPS run is positive at small effect size, but remains
  a non-gating future-work extension).
- No SAC/TD3 runs before submission (response-letter ammunition only).
- No new reward sweeps; no Track A reopening; no architecture zoo.

## Remaining compute upgrades (P1/P2, in order of value)

1. **Per-cell dense frontier for current/h104 + increased/h104** —
   upgrades E3 from stress screen toward real generalization. `current/h104`
   is complete and remains positive against the full 147-cell per-cell
   dense frontier (`+0.000244`, CI95 `[+0.000219,+0.000268]` vs
   `S3_op10_2.00_op12_1.50`). `increased/h104` is still running and must
   land before the full E3 dense-frontier upgrade is closed.

## Addendum 2026-07-02 (equations & Garrido-source pass)

Deep-read of Garrido's draft (v.0), the 2024 ICCL AI-SCRES paper, the
2024 IJPR factory-resilience paper, and the 2017 thesis (metric chapter
+ risk tables), reconciled with code ground truth:

- **Equations added (all numbered):** POMDP tuple (eq:pomdp, promoted),
  exact `control_v1` reward (eq:control_v1; also FIXED a prose error --
  the reward has a shift-utilization term 0.06(s-1), not a
  "shift-switching cost"), the order-level Garrido/Excel ReT case
  formula (eq:ret, thesis Eq. 5.5 weights 1/0.5/0 + workbook
  operationalization incl. backorder cap 60 and denominator j, audited
  47,546 orders / 0 mismatches), CVaR05 (eq:cvar), C(lambda_d) dispatch
  cost (symbol renamed from lambda to avoid the GAE collision).
- **Ceiling substantiated empirically:** canonical ledger shows 99.64%
  of orders in the recovery branch and min RP_j = 72.0 h exactly across
  ALL policies (48 h lead time + one 24 h dispatch cycle) -> per-order
  cap 0.5/72 = 0.006944 observed to the digit.
- **New tables:** complete thesis risk architecture R11-R3 (upgraded
  Codex's R11-R14 representative table), 8D track_b_v1 action-contract
  decode table.
- **New figure fig8:** ReT branch-logic order timeline (redrawn from
  thesis Fig. 5.5).
- **KAN verdict (deep-read):** Garrido et al. 2024 name KAN exactly
  twice, as an unranked, uncited, property-free alternative alongside
  backprop NN and RL/sim-opt; no KAN-specific argument exists in that
  paper. Manuscript future-work wording corrected to say "named ...
  without ranking them or arguing for KAN-specific properties".
  Decision: do NOT implement KAN for Paper 1; candidate for future
  interpretable policy distillation.
- **Fixed in Codex's parallel equation block:** "anticipation period" ->
  autonomy period (thesis term; 'anticipation' is on the banned list),
  fill-rate denominator Q_j -> running order index j (per
  ret_thesis.py:125) + backorder cap 60, "calibrated minimum recovery
  period" -> empirical 72 h floor with ledger evidence.
- Dispatch-cost sensitivity re-verified on the 10-seed pool (Codex's
  recompute): lambda_d=0 delta -0.00198 matches my independent check;
  significantly cheaper from lambda_d >= 0.025 (was 0.075 on 5-seed).
