# 3. Results

> **Supersedes `03_results_draft.md` (draft v1, 2026-07-18)**, which was written around the
> Program O conversion ladder before that programme closed
> (`STOP_PROGRAM_O_AFTER_CORRECTIVE_VALIDATION`). That draft is retained unedited as a record.
>
> Every number below carries its artifact and file digest. Evidence grade follows
> `GARRIDO_Q1_Q2_CLAIM_FREEZE_2026-08-07` and its amendments 1–3: **two prospective confirmations
> are usable here**; a third (`gsa_confirmation`, `1f487d91900e2ea4`) exists, ran on a *repurposed*
> virgin block, and was downgraded by its own corrective to a one-bit calendar choice — it is
> declared in the census and enters no claim.

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
state a conditioned policy can exploit. Whether the findings below survive strongly non-stationary
or seasonal demand is **not established here** and is the subject of a separate study.

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

This section validates the instrument. It establishes nothing about learning, and we do not use it
to.

---

## 3.3 The literal Fig. 5 neuron has nothing to learn

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

**The six methods that retain state occupy the six top positions**, and the same neural
approximator falls from rank 2 to rank 12 when its memory is removed. Paired against
`neuron_memory` (n = 12), the retention contrast within that family is **+0.0607 [LCB95 +0.0457]**,
while `ucb1_transfer` is **−0.0070 [−0.0243, +0.0141]** and `ofat_transfer` **+0.0107 [+0.0000356,
+0.0217]** — the latter excludes zero by roughly thirty-six millionths and establishes no practical
importance.

This is **development evidence on previously opened tapes** and adjudicates nothing. It motivates
the mechanism; it does not confirm it.

We also report the surface property that constrains this comparison: the value of knowing the
regime is `H_regime = 0.00380` [LCB95 ≈ 0] against a preregistered bar of 0.05, so the reference
gate returns `NON_SEPARABLE_BUT_CONTEXT_INVARIANT`. Retention here is buying **the avoided cost of
rediscovering a near-common good configuration**, not regime-tailored adaptation. Search-transfer
value and operational adaptation value are distinct quantities, and only the first is measured.

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

**Every family beats a cold start. Only `ucb1_transfer` beats its own marginal replay.** The neural
carrier fails that contrast with its confidence interval entirely on the unfavourable side, as do
the Gaussian-process and one-factor-at-a-time carriers. Frozen verdict:
`GRID_TRANSFER_CONFIRMED__UCB1`.

This is the central result. It licenses a narrower and more useful statement than the ladder alone
would support:

> **State retention ranked above memoryless search during development, but prospective transfer was
> carrier-specific: factor-level UCB1 outperformed both cold start and a state-blind replay of its
> own search marginals, whereas the neural carrier did not.**

The mechanism that transfers is a **factorised statistic over design levels**, not a learned
distributed representation. Whether other neural carriers transfer under this contract is not
established; what is established is that the one evaluated here does not.

---

## 3.6 Predictive fit does not imply search quality

If the neural carrier's difficulty were insufficient capacity, better function approximation should
help. It does not.

At matched parameter budgets (KAN 532, MLP 529) over the same six contexts
(`results/surrogate_architecture_bakeoff/result.json`, `f96e5b6ff0489932`), the KAN attains better
supervised fit on held-out partitions yet **searches worse**: `kan − mlp_matched` = **+0.01037,
CI95 [+0.00302, +0.01893]**, *p* = 0.0012, with lower AUC regret being better — the interval lies
entirely against the KAN. Status: `KAN_SEARCHES_WORSE_THAN_A_MATCHED_MLP`.

The best searcher of the seven-architecture bake-off is the **five-parameter neuron** (AUC 0.0520),
ahead of the matched MLP (0.0885), the KAN (0.0989), a spline-polynomial (0.0975), gradient-boosted
trees (0.1083) and a Matérn GP (0.1138).

Separately, a neural premium requires the surface's curvature to exceed the noise obscuring it. On
a deliberately curved surface (`results/headroom/buffer_prediction_premium/result.json`,
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

- **No organisational learning.** The loop closes *between* simulation runs. No routine of a real
  organisation was updated and the physical chain retains nothing across a campaign.
- **No within-episode adaptive control.** The retained state selects the next configuration; it
  does not act on the event stream inside a replication.
- **No neural premium.** None is confirmed anywhere in this study, and the one carrier tested
  prospectively fails the marginal-replay contrast.
- **No regime-tailored adaptation.** `H_regime` = 0.0038 fails its 0.05 bar; retention buys
  rediscovery cost, not tailoring.
- **No claim beyond the tested demand and risk process**, whose realised properties are given in
  §3.1.

The original conceptual draft's hypotheses H1–H4 are reconciled against the estimands finally
identified in **Appendix A**. They are not the spine of this study: H3 is not supported, H1 rests on
a redefined restricted-horizon estimand, and presenting them as a passing family would misrepresent
a sequence in which the estimands were identified after the hypotheses were written.
