# 3. Results

> **Supersedes `03_results_draft.md` (draft v1, 2026-07-18)**, which was written around the
> Program O conversion ladder before that programme closed
> (`STOP_PROGRAM_O_AFTER_CORRECTIVE_VALIDATION`). That draft is retained unedited as a record.
>
> Every number below carries its artifact and file digest. The claim lock and evidence registry
> govern admissibility; internal confirmation counts are not manuscript evidence.

---

## 3.1 Scope of the demand process, stated up front

All results below hold **within the thesis-inherited demand process**, `U(2400, 2600)` rations
placed every 24 h on six operating days per week (Garrido-Ríos, 2017, Table 6.4).

We characterise it rather than assert it. Over 12 episodes of ~954 complete weeks
(`results/demand_process/result.json`, `cb4f88398c4f93a4`): 65,835 regular orders fall exactly
within the contract bounds (min 2400, max 2600, mean 2500.21); realised **weekly demand has a
coefficient of variation of 7.1 %**, and 24.8 % of weeks already exceed single-shift assembly
capacity. Weekly demand is **not i.i.d.**: lag-1 autocorrelation is −0.228 (SE 0.0038) against an
i.i.d. band of ±0.065, decaying through −0.170 at lag 2 to −0.015 at lag 4.

Two consequences we state rather than bury. First, the process is neither static nor negligible in
variability, so the common characterisation of this benchmark as "almost deterministic demand" is
not accurate at the weekly decision cadence. Second, the memory it does carry is **negative, weak
and undirected** — consistent with anti-clustering of contingent surges — which is not the kind of
state a conditioned controller can exploit. Whether the findings below survive strongly non-stationary
or seasonal demand is **not established by the inherited-process results**. A bounded development
sensitivity is reported separately after the transfer results and cannot regrade Confirmation 2.

The source-faithful seasonal generator produces weekly CV **0.177** and lag-12 ACF **0.839** in
`results/demand_seasonal_engine/result.json`. Its original 12-episode sampler test and forecast
correlation test failed, so these numbers establish trajectory structure only. Holt-Winters is a
researcher-defined observable extension, not a repair of Garrido's equation; the amended protocol
defines forecast scoring at t+1 against naive and seasonal-naive baselines, synthetic phase tests
and a shuffled placebo. No forecast-skill claim is made here before that bounded sensitivity run.

---

## 3.2 The reconstructed DES reproduces Garrido's physical interventions (Confirmation 1)

Before asking what a learner can add, we establish that the simulator reproduces the effects the
source thesis reports. `garrido_h2_h3_confirmation_v1` (`bc375d3021b64d10`; completion receipt
`d4305bcf6bf5209d`, `COMPLETE_VALID_CONFIRMATION_AGGREGATE`) opened 12 confirmation tape roots
generated before any row was read, with development roots left unopened, and evaluated 1,080 rows
across six preregistered panels under Holm correction.

**All six panels confirm**, in direction and with every subsidiary gate met:

| Panel | Δ flow fill rate | LCB95 | Favourable tapes | Δ lost orders | Holm *p* |
|---|---:|---:|---:|---:|---:|
| R1r · H2 buffer | +0.0875 | +0.0870 | 12/12 | −180.2 | 6.6 × 10⁻¹⁷ |
| R1r · H3 shift | +0.0675 | +0.0670 | 12/12 | −143.0 | 1.9 × 10⁻¹⁵ |
| R2r · H2 buffer | +0.0919 | +0.0900 | 12/12 | −227.4 | 1.3 × 10⁻¹⁵ |
| R2r · H3 shift | +0.0619 | +0.0608 | 12/12 | −148.1 | 4.8 × 10⁻¹⁶ |
| R3 · H2 buffer | +0.0170 | +0.0167 | 12/12 | −36.5 | 4.8 × 10⁻¹⁷ |
| R3 · H3 shift | +0.0124 | +0.0121 | 12/12 | −28.4 | 3.2 × 10⁻¹⁵ |

Concordance was required across four endpoints simultaneously — delivered rations, flow fill rate,
full-ledger ReT and unresolved orders — and **generated orders were exactly zero in all 12 tapes of
every panel**, so no panel improved its score by moving demand rather than serving it.

Two features deserve emphasis. The effect is an order of magnitude smaller under R3 than under the
recurrent-risk families, which is a direction the source predicts but a magnitude it does not
quantify. And **lost orders fall in every panel**, so the physical improvement is not purchased by
abandoning claimants.

> **Claim boundary, taken verbatim from the artifact:** *"Confirmation applies only to H2/H3
> resource interventions in the frozen thesis-grounded reconstructed DES; it does not establish
> learner, feedback, or architectural value."*

This section establishes targeted physical correspondence. It establishes nothing about learning,
and it is not used as evidence for retention or architecture.

---

## 3.3 The literal Fig. 5 mapping is an identity

Garrido et al. (2024) propose a neuron whose dendrites are the four SCRES drivers `d_i`, weighted
by `ρ`, with the resilience metric as its axon. Taken literally, that mapping is not a learning
problem.

Fitting the drawn network on the driver table (`results/garrido_fig5_surrogate/result.json`,
`58d4c8a071cec86a`, falsifier `f1_task_A_is_an_identity`) recovers **R² = 1.0 with a maximum
identity error of 3.22 × 10⁻¹⁵**. The identified coefficients are 0.999999999999968 (`Re_RPj`) and
0.9999999999999998 (`Re_FRt`); the remaining driver columns are identically zero and therefore
carry no identifiable coefficient. Status: `IDENTITY_NOT_A_LEARNING_TASK`.

ReT *is* the sum of the driver contributions supplied to the neuron, so a perfect fit here is an
algebraic tautology and must never be reported as evidence of learning. This is not a criticism of
the proposal; it relocates where the learning is. The task that carries genuine uncertainty is not
predicting ReT from aggregated drivers — it is **deciding which configuration to run next**. That
reframing motivates everything that follows.

---

## 3.4 In development, stateful search occupies the top of the ladder

We compare fifteen deployable search procedures plus an oracle under a common budget of 24
evaluations across six ordered risk contexts (`results/search_ladder_v5/result.json`,
`f648a1da5aefaf2f`; 12 seeds, replay of a previously opened surface). The primary metric is the
area under the normalised regret curve; lower is better.

| Rank | Method | AUC regret | Retains state |
|---:|---|---:|---|
| 1 | `ucb1_transfer` | 0.04502 | ✔ |
| 2 | `neuron_memory` | 0.05203 | ✔ |
| 3 | `ofat_transfer` | 0.06274 | ✔ |
| 4 | `lookahead_kg_transfer` | 0.08018 | ✔ |
| 5 | `gp_ei_transfer` | 0.08390 | ✔ |
| 6 | `thompson_transfer` | 0.08908 | ✔ |
| 7–15 | `ucb1`, `ofat`, `gp_ei`, `thompson`, `lhs_local`, **`neuron_reset`**, `lookahead_kg`, `random`, `annealing` | 0.0966 – 0.1742 | ✘ |

The rank display is diagnostic. The adjudicated development/replay estimand is the within-family
paired contrast in `results/retention_contrasts/result.json` (`RETENTION_LOWERS_REGRET_IN_6_OF_6_FAMILIES`):

| Family | Memoryless − retained AUC | n |
|---|---:|---:|
| neuron | +0.06070 [+0.04568, +0.07953] | 12 |
| UCB1 | +0.05153 [+0.03583, +0.06593] | 12 |
| OFAT | +0.03750 [+0.02920, +0.04675] | 12 |
| KG | +0.03461 [+0.02610, +0.04315] | 12 |
| GP-EI | +0.02271 [+0.01276, +0.03410] | 12 |
| Thompson | +0.01985 [+0.01022, +0.02956] | 12 |

Positive values mean that retaining search state lowered regret. The artifact is a sealed-tape
reanalysis with no new seeds and no prospective adjudication; context-level effects cannot be
recovered because contexts were averaged before storage.

Six intervals computed from the same twelve tapes are six correlated looks, not six adjudications,
so the six contrasts are treated as **one inferential family**
(`results/retention_simultaneous/result.json`). A single bootstrap index matrix is shared across all
six, preserving their correlation; the resulting max-*T* simultaneous critical value is **2.591**
against a 1.906 marginal reference. All six simultaneous lower bounds remain above zero, Holm
rejects all six, and the sign of every bound holds in **40 of 40** resampling seeds — a stability
check this project has learned to run, since `ofat_lcb_reconciliation` measured a bound whose sign
depended on the resampling seed.

The same artifact reports the endpoint the deployment argument actually implies. `search_ladder_v5`
stores, alongside AUC, the **simple regret of the recommendation carried forward at budget 24**, and
under that endpoint the picture is weaker and differently ordered: all six point estimates keep
their sign, but only **one of six** retains a simultaneous lower bound above zero, and the family
ordering is not preserved — `lookahead_kg` leads under final regret and is fourth under AUC. Holm
and max-*T* also disagree there (four rejections against one bound above zero), which at n = 12 is
itself the finding. We report both and choose neither. Section 5 returns to this: a paper that
argues only the final recommendation is deployed, and then scores the area under the regret curve,
owes the reader that comparison.

This is **development evidence on previously opened tapes** and adjudicates nothing. It motivates
the mechanism; it does not confirm it.

The surface property that constrains this comparison, `H_regime`, is reported **with its metric
named**, because the name alone covers two statistics (`results/h_regime_crosswalk/result.json`).
On the `ret_excel_risk_conditional` surface of the 288 grid over twelve seeds it is **0.003802**,
reproduced here to the last digit against `surface_gates_v2`; on the Cobb-Douglas index
reconstructed from `aggregates.json` over six seeds it is **0.0** with a single configuration
optimal in every context. Those are different surfaces, and neither figure transfers to the other.

We draw no bar comparison from either, because the statistic is not invariant to the declared
utility scale: a strictly increasing rescaling that leaves every ordering untouched moves the
ret_excel figure from 0.003802 to 0.010776 on the 288 grid and from 0.028294 to 0.067539 on the
extended one. The crosswalk's `f3` demonstrates this rather than asserting it, rescaling the surface
and requiring every ordinal statistic back bit-identical while *H* moves.

What the same caches support is ordinal, and no monotone transform can move it: contextual rankings
exist and are strongly but not perfectly aligned, with mean pairwise rank correlation **+0.844** on
the 288 grid and **+0.909** on the extended, and top-25 configuration sets overlapping **91.7%** and
**23.5%** respectively. Retention here is buying **the avoided cost of rediscovering a near-common
good configuration**, not regime-tailored adaptation. Search-transfer value and operational
adaptation value are distinct quantities, and only the first is measured.

---

## 3.5 Prospective transfer is carrier-specific (Confirmation 2)

The ladder cannot distinguish transferred structure from a method simply revisiting configurations
that worked before. `grid_transfer_confirmation_v2` (`7bc33823ccd90b5e`, `run_role: CONFIRMATION`,
`scope: CONFIRMATION_ON_RESERVED_VIRGIN_BLOCK`) separates them by expanding the design space from
288 to 4,608 configurations — sixteen-fold — and scoring each family against **two** comparators:
a cold start, and a **state-blind replay of its own visit marginals**. The second preserves each
method's marginal sampling frequencies while destroying the retained structure, so only transferred
structure can beat it. n = 60.

| Family | vs cold start (mean, LCB95) | **vs state-blind marginal replay** (mean, LCB95, UCB95) |
|---|---|---|
| **`ucb1`** | +0.05744, +0.04989 | **+0.03073, +0.01990, +0.04256** ✔ |
| `neuron` | +0.05439, +0.04290 | **−0.01178, −0.01849, −0.00484** ✘ |
| `gp` | +0.01433, +0.00879 | −0.02160, −0.03051, −0.01227 ✘ |
| `ofat` | +0.01422, +0.00800 | −0.02467, −0.03258, −0.01666 ✘ |

**Every family beats a cold start.** The preregistered confirmatory arm — factorized UCB search —
outperformed both cold start and a state-blind replay of its own search marginals. Prespecified
secondary analyses found no corresponding advantage for the evaluated neural, GP-EI or OFAT
carriers; the neural contrast fell on the wrong side of zero. The secondary negative is not a
confirmatory negative. Frozen verdict: `GRID_TRANSFER_CONFIRMED__UCB1`.

A post-hoc re-read of the same prospective artifact as twelve arms rather than four within-family
contrasts found that the four cold-start arms occupied ranks 9–12, while the four lowest-regret
arms were mutually indistinguishable under Holm correction. Three of those four were
frequency-matched replays of a carrier's visit marginals; retention beat its own marginal replay
in one family and lost distinguishably in three. This re-read does not select a new winner and was
not preregistered.

This is the central result. It licenses a narrower and more useful statement than the ladder alone
would support:

> **Under simultaneous inference, retaining search state lowered development regret in all six
> matched families, and the prespecified factorized-UCB arm met the prospective transfer criterion
> beyond cold start and state-blind marginal replay. Secondary carrier contrasts showed no
> corresponding neural advantage under this contract. Re-read as twelve arms, however, the four
> lowest-regret arms are indistinguishable and three of them discard the carrier entirely — so what
> reliably transfers is a first-order visit distribution, not a sequential search procedure.**

The mechanism that transfers is a **factorised statistic over design levels**, not a learned
distributed representation. Whether other neural carriers transfer under this contract is not
established; what is established is that the one evaluated here does not.

---

## 3.6 Predictive fit does not imply search quality

Search efficiency and supervised fit are separate estimands.

At matched parameter budgets (KAN 532, MLP 529) over the same six contexts
(`results/surrogate_architecture_bakeoff/result.json`, `f96e5b6ff0489932`), the KAN search arm has
higher AUC regret than its matched MLP by **+0.01037, CI95 [+0.00302, +0.01893]**, *p* = 0.0012,
with lower AUC regret being better. This bakeoff adjudicates search; its fit evidence comes from
different contracts and is not combined with this number.

The best searcher of the seven-architecture bake-off is the **five-parameter neuron** (AUC 0.0520),
ahead of the matched MLP (0.0885), the KAN (0.0989), a spline-polynomial (0.0975), gradient-boosted
trees (0.1083) and a Matérn GP (0.1138).

Separately, an architecture-specific gain requires the surface's curvature to exceed the noise
obscuring it. On a deliberately curved surface (`results/headroom/buffer_prediction_premium/result.json`,
`54bf5fa2594262bd`; 1,530 episodes, seed-grouped cross-validation, six falsifiers), curvature
recomputed in situ is **0.0763** against **0.3174** of unexplained episode-level variance. Held-out
R² is 0.6826 for a linear model, 0.7163 for the KAN (+0.034 [−0.079, +0.146]) and **0.5548 for the
backpropagation MLP — worse than a straight line** (−0.128 [−0.316, +0.060]). Neither reaches the
preregistered SESOI of 0.05.

Predictive fit, terminal ceiling capture and sequential-search efficiency are three distinct
quantities, and improving the first does not deliver the second or third.

---

## 3.7 What these results do not establish

Stated as prohibitions rather than hedges:

- **No learning by the physical chain.** The loop closes *between* simulation runs. No routine of a real
  organisation was updated and the physical chain retains nothing across a campaign.
- **No within-episode adaptive control.** The retained state selects the next configuration; it
  does not act on the event stream inside a replication.
- **No architecture-specific transfer advantage.** None is confirmed anywhere in this study, and the one carrier tested
  prospectively fails the marginal-replay contrast.
- **No regime-tailored adaptation claim in either direction.** `H_regime` is reported per metric
  and never against a bar: it is not invariant to the declared utility scale. What is measured is
  ordinal — contextual rankings are strongly aligned — and retention buys rediscovery cost, not
  tailoring.
- **No claim beyond the tested demand and risk process**, whose realised properties are given in
  §3.1.

The original conceptual draft's hypotheses H1–H4 are reconciled against the estimands finally
identified in **Appendix A**. They are not the spine of this study: H3 is not supported, H1 rests on
a redefined restricted-horizon estimand, and presenting them as a passing family would misrepresent
a sequence in which the estimands were identified after the hypotheses were written.
