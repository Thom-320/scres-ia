# Reproducing Garrido's 90-configuration design — results, 2026-07-29

**Status:** `DEVELOPMENT_REPRODUCTION_NO_CLAIM`. Development runs only. No confirmation
universe opened, no network trained.

Runner: `scripts/reproduce_garrido_configurations.py`.
Design: `supply_chain/garrido_thesis_design.py` (thesis Tables 6.11–6.23).
Artifact: `results/garrido_reproduction/reproduction.json`.

## Why this was possible without the missing data

Garrido delivered three workbooks. Two are the thesis data — `Raw_data1+Re.xlsx` (Cf1–Cf10)
and `Raw_data2+Re.xlsx` (Cf11–Cf20), verified row-for-row against the degrees of freedom
published in Tables 6.26 and 6.27, 18 of 20 exact. The third, `Rsult_1.xlsx`, corresponds to
no published table and is a different run set entirely. **Cf21–Cf90 were never delivered, and
Cf31–Cf90 are exactly the buffer and shift scenarios — hypotheses H2 and H3.**

They were not needed. The design is fully published, and `config.py` already reproduces
Table 6.16 (buffer ladder) and Table 6.20 (capacity by shifts) **cell for cell**. The
`risk_overrides` mechanism reproduces Table 6.12 for all nine risks at both levels, also
exactly. So the 70 missing configurations were regenerated rather than requested.

## Part 1 — Validation against the delivered data (Cf1–Cf20)

Ten-year horizon, only the family's risks enabled (confirmed by the workbook column sets),
`risk_overrides` set to `increased` for each `+` in the design matrix.

| | our mean ReT | thesis mean ReT | ratio | our n | thesis n |
|---|---|---|---|---|---|
| R1r (Cf1–Cf10) | 0.0038–0.0056 | 0.0052–0.0087 | 0.49–0.94 | ~2,845 | 2,061–4,420 |
| R2r (Cf11–Cf20) | 0.405–0.669 | 0.116–0.395 | 1.56–3.94 | ~2,845 | 1,956–2,277 |

**What reproduces.** The order-of-magnitude separation between risk families is reproduced:
ReT under R2r is roughly a hundred times ReT under R1r, in both the thesis and our model.
That separation is the structural signature of the metric — the `APj/LT` branch dominating
under rare high-impact risks — and we recover it without tuning.

**What does not.** Two discrepancies, both real and both worth stating plainly.

*Level.* Under R1r we run about 25% low; under R2r we run 2–4x high. Since our ReT
implementation was previously verified against these same workbooks at the formula level
(0/47,546 cell mismatches), the divergence is in the DES trajectory, not the metric —
specifically in the distribution of autotomy periods relative to a fixed 48-hour lead time.

*Order count — resolved, and it is not a fidelity gap.* The right comparison is generated
orders against generated orders. **Max `j` in his 10-year sheets is 2,834–2,841 against our
2,845 — agreement to 0.4%.** What differs is how many of those orders carry a ReT value: only
**72.7%–75.7%** of his are scored. So demand generation matches; the *scored population* does
not. That is the same policy-dependent censoring documented in
`AUTHORITY_LADDER_SCREEN_FINDINGS_2026-07-28.md` §3, now confirmed in the author's own data.
Comparing our all-order mean to his scored-subset mean therefore compares different
populations, which is part of the level gap above and must not be read as a physics
difference.

*Horizon heterogeneity in his data.* Cf1 and Cf2 ran to ~161,190 h — **19.84 years**, not the
10 that Table 6.13 prescribes — which is what makes their 4,241 and 4,420 rows sit so far
above every other configuration (1,956–2,278). CF5 is separately anomalous (4,241 rows,
21 columns where siblings have 22, and 4,241 is exactly CF1's count). **Cf1, Cf2 and Cf5 are
quarantined from level comparison** until their true horizons are reconciled.

**Scope limit that matters.** 19 of 20 thesis seeds were recovered from the workbooks and used
here, but our RNG is not Simulink's: seed 375 does not reproduce Garrido's stream. **This is a
distributional comparison, not a per-order replication, and no per-order agreement should be
claimed.**

## Part 2 — The 70 configurations that were never delivered (Cf21–Cf90)

The thesis reuses one seed across each `(Cf_b, Cf_b+30, Cf_b+60)` triple, which makes the
buffer and shift effects **paired contrasts**: same seed, same risk pattern, only the decision
right changes. That pairing is verified in the artifact. For Cf21–Cf30 the thesis seeds were
not recoverable, so a deterministic fallback keyed on the base index is used — internally
paired, but not tied to Garrido's realisations.

**Corrected 2026-07-29 after adversarial review — two defects of mine were found and fixed.**

*H2 was mis-instrumented.* Scenario II is a **replenishment policy**, not an initial
endowment. §6.7.3: "independently of the occurrence of the above risks, every t = 168, 336,
504, 672, or 1,344 hours, the level of I_tS is replenished in the quantities of raw material
and rations indicated in Table 6.16." `_inventory_buffer_replenishment` returns immediately
unless `inventory_replenishment_period` is set, and the runner passed only `initial_buffers`.
The first H2 numbers measured a one-off injection. Fixed; the effect roughly **quintuples**
under R1r.

*The "7/10" for shifts was forced by the design, not noise.* Exactly **three of the ten** H3
configurations in every family carry S = 1 — identical to their baseline — so their deltas are
zero by construction (verified: 0 non-zero deltas among those 3 in each family, confirming the
pairing is exact). The correct denominator is the seven genuinely treated pairs.

| risk family | baseline | + buffers (H2) | Δ | | pairs | + shifts (H3, S>1) | Δ | | pairs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R1r | 0.004716 | 0.005255 | +0.000540 | **+11.45%** | 10/10 | 0.005064 | +0.000504 | **+11.06%** | 7/7 |
| R2r | 0.529626 | 0.662908 | +0.133282 | **+25.17%** | 10/10 | 0.648247 | +0.096419 | **+17.47%** | 7/7 |
| R3 | 0.968878 | 0.991324 | +0.022446 | **+2.32%** | 10/10 | 0.983136 | +0.015272 | **+1.58%** | 7/7 |

### Evaluation windows differ, by about one percent

`compute_episode_metrics` starts after each run's own warm-up, and buffers shorten it (Cf3
960 h vs Cf33 73 h), so a paired contrast does not evaluate an identical interval. Quantified:
the window is **98.81% of horizon without buffers and 99.91% with**, a 1.1-point difference.
That does not account for an 11–25% effect. It is also not an instrumentation error: warm-up is
endogenous in the thesis too (his own values range 823–1,225 h across configurations, and
Table 6.16 stock legitimately shortens the time until the first Q = 5,000 reaches Op9).
Equalising it would be less faithful, not more. Disclosed, not removed.

### All six moderation hypotheses reproduce in direction

The thesis concludes for H2a/H2b/H2c that "when on-hand inventory buffers were increased by
the indicated levels, the MFSC's measure of resilience (ReT) **increased**" in the presence of
R1r, R2r and R3 respectively, at 99% confidence; and for H3a/H3b/H3c the same for short-term
manufacturing capacity, at 95% (H3a, H3c) and 99% (H3b).

Every one of the six is positive in our reproduction.

A convergent detail worth noting: the thesis reports the **shift** hypotheses at *lower*
confidence than the buffer ones (95% vs 99% for the R1r and R3 cases). Our pair counts show
the same asymmetry from the opposite direction — buffers improve 9–10 of 10 pairs, shifts only
7 of 10 in every family. The weaker hypothesis in the thesis is the noisier one here too,
which is not something the design forced.

## What this establishes, and what it does not

**Establishes.** The published design can be executed end-to-end in our DES; the risk-family
structure of ReT is reproduced; and the two moderation effects Garrido tested — buffers and
capacity — are recovered in direction across all three risk categories, including the sixty
configurations he never sent.

**Does not establish.** Numerical agreement with the thesis. The level gap under R2r (2–4x)
and the constant order count are unresolved and are the two things to chase next. Nor does
this validate the metric: the `APj/LT` branch remains unbounded, and every ReT figure above
inherits that property.

**Bearing on the authority ladder.** State this in **relative** terms; the earlier "500x"
framing compared absolute deltas across baselines that themselves differ by a factor of a
hundred, and was misleading. On the corrected instrument the buffer right is worth **+11.45%**
under R1r, **+25.17%** under R2r and **+2.32%** under R3 — roughly an elevenfold spread across
families, with R3 additionally compressed against a ceiling (baseline 0.969). The absolute R1r
effect is `+0.00054`, still the 1e-4 order we have been calling "no headroom" in the screens,
which is worth noting; but the defensible claim is the relative one: **the value of a decision
right depends strongly on which risk family is enabled**, and it is largest exactly where the
disruptions are rare and severe.

Two caveats on that claim, both open. Our throughput sits 8% below his validated figure and our
scored population differs from his by ~27%, so the *levels* are ours, not his. And every ReT
above inherits the unbounded `APj/LT` branch.

## Outstanding — not blockers, but named

1. **Table 6.20 traces.** `shifts` updates `op3_q` and `batch_size`; whether every Q and ROP
   for Op3, Op4, Op7 and Op8 is materialised exactly as published has not been verified against
   operational traces. H3 is a plausible full-DES analogue until it is.
2. **Inference unit.** The figures above are per-configuration means over one tape each.
   Thousands of correlated orders are not independent replications; repeating over multiple
   matched tapes and inferring at run level is the right next step.
3. **Custody.** These files are untracked inside an unfinished merge.
