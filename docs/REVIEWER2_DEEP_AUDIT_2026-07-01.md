# Reviewer #2 Deep Audit — Q1 Readiness (2026-07-01)

Produced by a 9-agent adversarial audit of the actual repo artifacts, code, manuscript, and
literature (5 evidence auditors → 3 hostile referees → editor synthesis), plus a full read of
Garrido, Pongutá & Adarme (2024, LNCS 15168). This supersedes the three external reviews where
they conflict: those reviewed the *claims registry and plans*; this audit reviewed the *code,
run artifacts, and the actual LaTeX draft*.

## Verdict

**The science is closer to Q1-ready than the external reviews say. The manuscript and one
unexamined confound are much further away.**

- The headline Track B result survives every statistical attack we could construct — including
  the seed-clustering and winner's-curse attacks the external reviews flagged as open. Those are
  now **neutralized with zero new compute** (numbers below).
- But there is one attack **none of the three external reviews found** that can kill the science
  itself (T3: privileged observation), one that can void the validation anchor (T6: fidelity-gate
  flow mode), and the actual draft in `docs/manuscript_current/submission/elsevier/` is one full
  era behind the evidence and asserts claims your own `effect_sizes.csv` falsifies (T1).

---

## Part 0 — What changed since the external reviews were written

| Item | External reviews said | Actual state (verified on disk) |
|---|---|---|
| Fidelity gap (buffer saturation) | not discussed | **RESOLVED** — paired H2/H3 gate PASSED (H2 29/30, H3 27/30 family-scenarios, 300 episodes); `docs/FIDELITY_GAP_BUFFER_SATURATION_2026-06-28.md` |
| 8D ablation | "not run" | **Screen DONE** (Jul 1, 2 seeds/30k): downstream_only `0.0057` ≥ joint `0.0055` ≥ shift_only `0.0056`; 5 heuristics all lose to PPO. **5-seed final never launched**; fixed-shift / no-risk arms **not implemented** in `scripts/run_track_b_ablation.py:65-69` |
| Seed-clustered inference | "must do, result may weaken" | **Done in this audit from existing data — result HOLDS** (see Part 2) |
| Reward×obs sweep | "running" | 18-cell Kaggle sweep complete, 14 cells promoted (`track_b_adaptive_sweep_kaggle_2026-07-01_v6`) |
| v9 confirm | not known | **Half-broken**: candidate 1 (ReT_excel_plus_cvar, v9) wins ReT/tail but **fails the pre-registered cost cap** (0.924 > 0.70); candidate 2 (control_v1, v9) **never ran** — indentation bug at `kaggle/track_b_adaptive_confirm_v9/scresia_track_b_adaptive_confirm_v9.py:239-242` (`run()` nested inside `if alpha is not None:`). Quarantine v9; v7 stays canonical. |
| Registry C1 number clash | not noticed | Resolved — C1 cited the `order_level_ret_mean` row (0.005666/0.005251) mislabeled as Excel ReT; Excel ReT is 0.005893/0.005466. Same run, same CSV. **Registry fixed 2026-07-01.** |

---

## Part 1 — THE go/no-go attack (new; not in any external review)

### T3. Privileged observation: the agent reads the simulator's ground truth

Observation v7 (inherited from v6) includes:
- a **one-hot of the TRUE generative disruption regime** (`nominal/strained/pre_disruption/disrupted/recovery`),
  built directly from `self.adaptive_regime` — `supply_chain/supply_chain.py:1395-1398`;
- **48h/168h "forecasts" computed from the TRUE regime transition matrix**
  (`ADAPTIVE_BENCHMARK_TRANSITIONS`, `supply_chain/config.py:557-585`) with only σ=0.05 noise —
  `supply_chain/supply_chain.py:1729-1746`.

Every comparator is an **open-loop constant**. The 5 heuristics in `scripts/track_b_heuristics.py`
read only backlog/fill indices — none conditions on the regime one-hot or forecasts — and they
exist only in the 2-seed screen.

**The reviewer sentence:** *"A lookup table with five constants — one per observed regime —
would exploit exactly this signal with zero learning. Beating open-loop constants with an
oracle-informed feedback controller shows feedback beats open-loop, not that RL adds value."*

**Go/no-go experiment (E1, eval-only, 1-2 days):** fit the best constant `(shift, op10, op12)`
per observed regime on training-seed episodes; evaluate CRN-paired under the canonical protocol
alongside the 5 heuristics and a forecast-threshold rule. Report the gap decomposition
**constant → regime-table → tuned heuristic → PPO** as the paper's central figure.
- If PPO clearly beats the regime table → the claim survives and is *stronger*.
- If the regime table matches PPO → the paper has no RL result; reframe as
  feedback-vs-learning decomposition and target COR, not IJPR.

**Companion (E2, one retrain, 5 seeds/60k):** PPO with the regime one-hot + forecast fields
masked. Shows the win doesn't reduce to reading the oracle. Add a table classifying all 52 v7
dims as simulator-privileged vs plausibly-measurable.

### T6. Fidelity-gate flow-mode flag (do this FIRST — hours)

The gate manifest (`outputs/benchmarks/garrido_static_fidelity_stress/paired_h2_h3_full_cf1_30_thesis_1rep_2026_06_28/manifest.json`)
records `raw_material_flow_mode='legacy_validated'` — which `docs/RET_GARRIDO2024_AUDIT_2026-06-18.md`
(F2) previously labeled the **inert open-loop bug mode** vs faithful `kit_equivalent_order_up_to`.
Either mode semantics changed after 06-18, or the PASS ran in a non-faithful configuration and the
paper's validation anchor is void. Reconcile against git history; if wrong, rerun the 300-episode
gate (hours). Also: add a binomial/sign test to the H2/H3 counts (Garrido used binomial tests),
disclose the gate passes on **ReT sign only** (fill 0-6/10; H1 'missing'), horizon 260 weekly
steps ≈ 5.4 thesis-years (not 20).

---

## Part 2 — Good news: attacks already neutralized (put these IN the paper)

Recomputed in this audit from `episode_metrics.csv` (no new runs):

1. **Seed-level inference (kills the pseudo-replication attack):** per-seed mean deltas
   0.000408 / 0.000414 / 0.000429 / 0.000434 / 0.000447 — **all 5 positive**; seed-level paired
   t(4)=60.8, p=4.4e-7, CI95 [+0.000407, +0.000446]; seed-clustered bootstrap CI
   [+0.000414, +0.000438]; **60/60 pairs positive** (min +0.0000779); **all 16 distinct CRN
   scenarios positive** (min mean +0.000217). Report seed-level as PRIMARY inference; keep the
   pooled n=60 bootstrap as secondary; disclose that the sliding eval windows mean only 16
   distinct CRN scenarios.
2. **Winner's curse quantified harmless:** comparator is argmax of 147 statics on the same test
   episodes (real defect — disclose it), but PPO beats **every top-12 static** with fully
   positive CIs; the delta *grows* down the ranking (+0.000427 vs 2nd → +0.000465 vs 12th) and
   the top-12 spread (3.8e-5) is ~10× smaller than the PPO delta. Selection bias cannot flip
   the sign. Add this table.
3. **CRN is real and correct:** `eval_seed = seed + 50000 + episode` shared across all policies;
   train seeds (1-5) disjoint from eval seeds (50001-50016).
4. **Statics got MORE tuning than PPO:** one modest PPO config vs a 147-cell tuned grid; and the
   grid's dispatch range (up to 2.5×) exceeds PPO's action cap (2.0×) — asymmetry FAVORS statics.
   Free defense; say it.
5. **All 10 "suspicious" citations from the external reviews are REAL** (web-verified: Ghasemloo
   2605.27556, ReflectiChain 2606.10359, Pan 2506.21872, Maggiar 2507.22040, Temizöz 2411.00515,
   Kotecha 2410.18631, Stranieri 2501.10895, Che 2409.03740, MORSE 2509.06490, Garrido IJPR DOI).
6. **Attacks that do NOT bite** (don't over-invest): recurrence (RecurrentPPO ran 500k×5 and
   lost — registered), H4 overclaim (registry defers it; draft contains zero H4 claims), reward
   hacking (agent never observes Excel ReT — frame the reward/metric split as an anti-gaming
   design), sim provenance (formula-exact ReT over 47,546 rows beats nearly all RL-for-SCM
   benchmarks).

---

## Part 3 — Blocking list (beyond T3/T6)

### T1. The draft contradicts your own data (fatal, editing-only)
`docs/manuscript_current/submission/elsevier/` (all mtimes 2026-06-24):
- claims **"57% assembly-hours reduction vs best static"** (04_results.tex:100-102,137-138;
  05_discussion.tex:33-34) — `effect_sizes.csv` says cost index PPO 0.682 vs 0.667, CI spans 0,
  **win=False** (PPO is slightly *costlier*);
- abstract headline **"fill rate 1.000, zero backorders"** — current flow fill is **0.9613**;
- **"7D"** throughout — contract is 8D;
- §4.6 asserts cross-scenario generalization (+9.2/+31.0/+27.6pp) from the retired 500k era —
  registry C11 says generalization is **not closed**;
- frozen-contract block (03_methodology.tex:110-120: ReT_seq_v1, κ=0.20, 500k, seeds 11-55,
  20 eps) is wrong on every element vs the canonical run's `summary.json`;
- instruction-voice scaffolding still in the body (03_methodology.tex lines 4, 44-45, 84, 92, 129);
- 0 figures (3 Pareto PNGs sit unused), no CRN mention, no dense-grid definition, no MDP tuple.
**Every number in the paper must be rebuilt from `docs/track_b_q1_stats_2026-07-01/`.**

### T2. Citation apparatus (desk-screen level; one editing day)
- **One `\cite` command in the entire manuscript** (goldratt1984goal); `\nocite{*}` at main.tex:48;
  two conflicting `\bibliographystyle` lines (:50 elsarticle-num, :52 plain). As compiled, the
  paper has a one-entry bibliography.
- **Integrity-level:** DOI 10.1080/17477778.2025.2500393 (Kogler & Maxera, J.Simulation) is in
  `references.bib` TWICE under different invented authorship (`rolf2025jsim` = "Rolf, Benjamin
  and others") and §2.2 cites it as **two independent reviews**. Delete `rolf2025jsim`, rewrite §2.2.
- `temizoz2025` has the wrong DOI (correct: 10.1016/j.ejor.2025.01.026, EJOR 324(1):104-117);
  `ding2026` DOI + author list unverified vs SSRN 5609791; 7 "and others" placeholders;
  `garrido2024ijpr` coauthor is "Fabian Ponguta", not "Edwin Pongutá".
- **Zero SCRES-theory anchors:** no Ponomarov & Holcomb 2009, Wieland & Durach 2021,
  Hosseini/Ivanov/Dolgui 2019, no Ivanov ripple/viability. A resilience paper with no resilience
  definition cite is desk-reject bait at IJPR/IJPE.
- Missing anchor-paper cross-check cites (reviewers WILL check against Garrido 2024's own list):
  von Rueden, Zhang IJPR 2024, Chan 2022, Saisridhar Triple-R, Bruckler 2024, Ivanov 2019,
  Moosavi & Hosseini 2021, Fattahi 2020, Pires Ribeiro & Barbosa-Povoa 2018, Taghizadeh 2021,
  Greasley & Edwards 2021, Eryarsoy 2022, **Levitt & March 1988** (the "Alzheimer effect" IS
  organizational forgetting — this cite is load-bearing).
- Missing for claims made: Tamar/Chow CVaR-RL (tail claims), Oroojlooyjadid Beer Game, Madeka 2022.
- Named competitors to cite & differentiate: Ding et al. 2026 (MARL reconfiguration, IJPE),
  Kotecha & del Rio Chanona 2025 + MORSE, Rolf et al. 2024 IJPR, Lu et al. 2025, Stranieri
  2501.10895 (ally: PPO does not universally dominate), Bussieweke/Mula 2025, Aboutorab 2024.

### T4. Single-cell win in a bespoke regime
`adaptive_benchmark_v2` multiplies exactly the downstream risks the new actuators control
(R22×1.35, R23×1.15, R24×1.25, surge×1.20 — `supply_chain/config.py:632-641`) and the win exists
only at that cell (h104, control_v1, v7). No PPO eval at current/increased/severe, no h260.
**Fix (E3, eval-only):** run the canonical checkpoints through `scripts/eval_track_b_cross_scenario.py`
at current/increased/severe × h52/h104, CRN-paired vs per-regime dense statics. Print the
multiplier table in the paper and call the regime a *designed sustained-disruption stress cell*.
This decides whether the title keeps "frontier-dependent" or gets scoped.

### T5. Ablation + comparator scope
- The 5-seed/60k final ablation prescribed by `ablation_decision.md` was never launched — the one
  **mandatory new training run** (E4). Own the screen's `downstream_only ≥ joint` pattern: it
  strengthens "lever authority, not lever count".
- The "dense static frontier" is dense in only **3 of 8 action dims** (shift, op10, op12;
  `scripts/run_track_b_dense_crn_static.py:50-56`). Either bound the best-8D-constant with a small
  eval-only 3×3 grid over op3_q×op9_q at the best downstream cell, or rewrite the comparator claim
  as "best static over the shift×op10×op12 family" (and cite the Track A dense-CRN null for the
  inventory dims).

### T7. Statistical presentation (pure reanalysis, ~2 days)
- Seed-level primary inference + 16-scenario disclosure + top-12 robustness (Part 2).
- **CVaR05 has three inconsistent definitions** — one is literally the p05 percentile mislabeled
  (`scripts/audit_track_b_mechanism.py:195`); `effect_sizes.csv` has NO CVaR row despite the
  registry requiring one. Pick one definition (conditional mean of lowest 5% of per-episode Excel
  ReT), one comparator (dense S2_op10_2.00_op12_1.50), add the row with CI.
- Drop/rescope Cohen's d=2.87 (CRN-paired pooled deltas make d uninterpretable; a referee will
  laugh at d=6.17 on fill). Use the ceiling framing instead.
- One comparator everywhere: dense S2_op10_2.00_op12_1.50 (decision.json uses s2_d1.50;
  mechanism_audit.json uses s3_d2.00 — unify or label).
- `order_ctj_p99` and `order_dpj_p99` rows are byte-identical — verify aliasing before publishing
  a 12-metric panel with a duplicated row. Designate `order_ret_excel` as the single pre-specified
  primary endpoint; the rest are descriptive secondaries (all survive Bonferroni×12 anyway — say so).

### M1. Metric degeneracy (reframe, no compute)
In this regime Excel ReT is effectively 0.5/RPj with hard ceiling 0.5/72 = **0.006944**
(= ledger p95 = p99 for BOTH policies). The +0.000426 delta = **~6.1% of ceiling** (PPO at 84.8%
vs static 78.7%) ≈ six points more orders fulfilled under recovery. State the ceiling wherever
the delta appears; lead the results with the service/tail panel (CTj p99 8113→1207 h; service-loss
AUC 1.13M→113k; backlog 10,929→309); defend Excel ReT as the thesis-continuity metric, never as
field-general. Also: the ReT>1/38-rows quirk belongs to the Garrido raw-workbook lane only —
canonical ledger has 0 such rows (say this; it preempts a spurious attack).

### M2. Cost accounting (likely an UNCLAIMED WIN)
`assembly_cost_index` = mean shift level / 3 — prices shift labor only; dispatch is free in both
reward and cost index; cost CVaR95 favors the static (0.877 vs 0.667) — disclose both. BUT PPO
uses *less* mean dispatch than the best static (1.30/1.27 vs 2.00/1.50). An offline dispatch-cost
sensitivity (charge c per multiplier-step, sweep c, from existing ledgers) likely makes PPO
strictly cheaper on a dispatch-inclusive cost — currently unclaimed. Compute it.

### M3. Forking paths → weaponize
~27 registered lanes, ~11 rewards, obs v1-v9, 4 contracts, PPO/RecurrentPPO/DQN/BC. Consolidate
into ONE dated exhibit (lanes, configs, outcomes, artifacts, pre-stated gates quoted verbatim from
`TRACK_B_TOP_TIER_AUDIT_2026-06-30.md`) + the exploratory→confirmatory→dense-CRN ladder. This is
the strongest anti-fishing defense in the repo and doubles as a methodology contribution.

### M4. Framing correction (more interesting than the current one)
"RL wins iff the action space reaches the bottleneck" is (a) Theory-of-Constraints folklore and
(b) **contradicted in-house**: Track A had measured oracle headroom (+0.0003..+0.0028) that PPO,
BC, and BC+PPO all failed to convert (best fair attempt missed by 7e-6). The defensible and
*stronger* claim: **action-space coverage of the binding constraint is necessary but not
sufficient; we quantify both directions in a thesis-validated testbed.** The unconverted-headroom
result is a genuinely novel negative finding — most papers can't even measure headroom.

---

## Part 4 — Three sentences that must never appear

1. *"PPO reaches perfect fill (1.000) and reduces assembly hours by 57% versus the best static."*
   (Falsified by your own `effect_sizes.csv`.)
2. *"RL adds SCRES value when and only when the action space reaches the binding bottleneck."*
   (Universal law from one cell; refuted in-house by Track A headroom; ToC restated.)
3. *"The agent learns to anticipate disruptions."* (`anticipation_claim_allowed=false`; no lead/lag
   artifacts; and the obs contains the true regime + forecasts — any anticipation claim converts
   directly into the privileged-observation kill point.)

## Part 5 — Journal ladder

**IJPR (primary).** Published Garrido 2024 IJPR; hosts the DES+RL genre (Rolf, Stranieri, Zhang,
Saisridhar). Anchor is an asset. Wants: 8-10k words, figures, generalization matrix, dynamic-rule
baselines at scale, complete bibliography. Realistic: major revision → ~35-45% eventual accept.

Ladder: **IJPR → IJPE → Computers & OR → Computers & IE.**
Do NOT submit to EJOR (no methodological novelty — vanilla PPO+MLP), Omega/IJOPM (no managerial/
SCRES theory yet), M&SOM (single non-stylized case). Decision rule: if E1 shows the
regime-conditioned table ≈ PPO, skip IJPR and go to COR with the honest decomposition framing.

## Part 6 — Execution order (~1 week wall-clock; 1 mandatory training run + 1 small retrain)

| # | Task | Cost | De-risks |
|---|---|---|---|
| E6 | Reconcile `raw_material_flow_mode` in fidelity-gate manifest vs git history; rerun 300-ep gate if bug mode; add binomial tests | hours | T6 (do FIRST — a bad answer changes everything) |
| E1 | **Go/no-go:** regime-conditioned static table + forecast rule + 5 heuristics under canonical 5-seed CRN protocol | 1-2 d eval-only | T3, baselines |
| E2 | Obs-masked PPO (v7 minus regime one-hot + forecasts), 5 seeds/60k | ~1 d compute | T3 |
| E3 | Cross-regime matrix: canonical checkpoints @ current/increased/severe × h52/h104, CRN vs per-regime dense statics | 1-2 d eval-only | T4, title scope |
| E4 | `track_b_ablation_8d_final` 5 seeds/60k (joint/downstream_only/shift_only), heuristics folded in | 1 training run | T5 |
| E5 | Reanalysis bundle: seed-level stats, top-12 table, unified CVaR05 + CI row, dispatch-cost sweep, ceiling framing, 3×3 inventory-constant grid | ~2 d analyst, 0 compute | T7, M1, M2 |
| E7 | Learning curves + entropy figures from SB3 logs; one 120k plateau run; wire 3 Pareto PNGs + topology schematic | 1 d | M6 |
| — | Full manuscript rewrite to dense-CRN canon + bibliography repair (T1, T2) | 2-3 d writing | T1, T2 |

**Hold for the response letter (do NOT gate submission):** SAC/TD3 at the canonical cell; Track A
PPO on adaptive_benchmark_v2 (completes the 2×2: action space × regime — separates regime
observability from frontier effects); Track B h260; mechanism lead/lag audit; H4 retained-vs-reset.

## Part 7 — Repo hygiene before any supplement/OSF release

- Scrub/retitle `docs/WIN_CONFIRMED_2026-06-29.md` (per-op "win" falsified by rich-metrics rerun,
  PROMISING_LANES lane 1g — a stale WIN_CONFIRMED title in the supplement is reviewer catnip).
- Archive or label `outputs/experiments/track_b_dense_frontier_2026-07-01` (different undocumented
  regime; same cell scores 0.005550 there vs 0.002291 in the canonical grid — if a reviewer finds
  two "dense frontiers" with 2.4× different values, the CRN story collapses).
- Registry C1 label: **fixed 2026-07-01** (order_level_ret_mean vs order_ret_excel).
- Fix the v9 kernel indentation bug for the record; keep all v9 numbers out of the paper
  (candidate 1 fails its own cost gate 0.924 > 0.70; candidate 2 never ran).
- Rename paper contributions C1-C4 (collide with registry claim IDs C1-C14).
- Disclose calibrated-to-target parameters (delay=54h, R14 period=72h) in the deviations table.
- Bib: delete `rolf2025jsim`; fix temizoz2025 DOI; verify ding2026; replace 7 "and others"
  placeholders; fix Ponguta/Garrido-Ríos name forms.
