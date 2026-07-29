# Porting Garrido's Cobb–Douglas resilience index to the MFSC DES

**Source:** Garrido, Pongutá & García-Reyes (2024), *Zero-inventory plans, constant workforce,
or hybrid approach? Analysing pure production strategies for enhancing factory resilience with
demand variability*, International Journal of Production Research, DOI
10.1080/00207543.2024.2425771. Index defined in §3.4, Equations (2)–(6).

**Status:** implemented development sensitivity, not economically calibrated and
not authorized for primary selection.

## 1. What the paper actually specifies

Starting from the Cobb–Douglas production function `P(L,C) = μ·L^k·C^(1−k)` (Eq. 2), with
efficiency parameter μ = 1, the factory resilience index over five output variables is

    R(ζ, ε, φ, τ, κ̇) = (ζ^a)·(1/ε^b)·(φ^c)·(1/τ^d)·(1/κ̇^n)          (Eq. 3)

linearised as

    R = a·ln ζ − b·ln ε + c·ln φ − d·ln τ − n·ln κ̇                    (Eq. 4)

with the fitted exponents

    R = 0.024·ln ζ − 0.026·ln ε + 0.04·ln φ − 0.06·ln τ − 0.1771·ln κ̇  (Eq. 5)

and finally squashed to (0,1) because "the R index is meaningless for negative values":

    R = 1 / (1 + exp(−(0.024·ln ζ − 0.026·ln ε + 0.04·ln φ
                       − 0.06·ln τ − 0.1771·ln κ̇)))                    (Eq. 6)

R is dimensionless on (0,1); 0 is the lowest and 1 the highest factory resilience.

### The five variables and their signs

| symbol | definition in the paper | direction | rationale given |
|---|---|---|---|
| ζ | `Σ I_t / T` — average accumulated **inventory** | **raises** R | higher inventory ⇒ shorter lead times ⇒ greater responsiveness to demand variability |
| ε | `Σ B_t / T` — average accumulated **backorders** | lowers R | higher ε ⇒ greater lead times ⇒ lower responsiveness |
| φ | `Σ U_t / T` — average **spare production capacity** | **raises** R | same responsiveness argument |
| τ | `Σ (NR_t / min{GR_{t+v}, Θ_t}) / T` — average **time to meet net requirements** | lowers R | longer τ ⇒ lower responsiveness |
| κ̇ | `7·κ(S_ij) / Σ_ij κ(S_ij)` — normalised **total cost** deviation | lowers R | higher cost ⇒ less efficient, less agile facility |

with `κ(S_ij) = Σ[c_p·P_t + c_h·H_t + c_ℓ·L_t + c_u·U_t + c_i·I_t + c_b·B_t + c_o·O_t]`,
i.e. costs of regular production, hiring, firing, marginal capacity, holding, backorders and
overtime.

### How the exponents were obtained — this is the part that must NOT be copied

The paper is explicit (§3.4, after Table 3):

> "the highest values of ζ, ε, φ, τ and κ̇(S_ij) were identified after 10,000 simulation runs.
> And second, each function argument was equated to 1/5. For example, in the case of ζ,
> ζ^max ≈ 3,612, from which a·ln 3,612 = 0.20, resulting in a = 0.024."

So the exponents are **not preference weights**. They are scale normalisers:

    exponent_x = 0.20 / ln(x_max)

where `x_max` is the largest value that variable reached across 10,000 runs **of their APP
Monte-Carlo model**. Each of the five terms is thereby made to contribute at most 1/5 at its
own observed maximum, which is what makes five quantities in incompatible units comparable.

**Consequence for us.** `0.024 / 0.026 / 0.04 / 0.06 / 0.1771` encode `ζ_max ≈ 3,612` and the
corresponding maxima of a 36-week aggregate-production-planning model with ~800-unit demand.
Our DES carries inventories in the millions. Copying those five numbers would silently rescale
every term by orders of magnitude and produce a meaningless index. **The correct port
re-derives the exponents with Garrido's own rule, from our own maxima.**

## 2. Why this index is the right answer to a problem we measured

Two independent defects in `ret_excel`, both quantified on 2026-07-28 (see
`AUTHORITY_LADDER_SCREEN_FINDINGS_2026-07-28.md` §3):

1. Under the thesis risk regime `ret_excel` goes **non-monotone across shift postures** — it
   ranks shift 1 first while shift 3 delivers 18% more rations, zero lost orders and +14.6
   fill points — because the omitted-order fraction is policy-dependent (18.60% vs 3.91%) and
   all cases collapse into the `recovery` bucket.
2. Shifts carry **no cost** in the ReT objective, so nothing restrains always choosing shift 3.

The C–D index addresses both, and not by coincidence:

- It reads **spare capacity φ, backorders ε and time-to-fulfil τ directly from the physical
  ledger**, with no case classification and no order-visibility filter. There is nothing in
  Eq. (6) that can censor a subset of orders.
- It carries an explicit **cost term κ̇** that prices exactly the levers ReT leaves free —
  regular production, overtime, hiring/firing and marginal capacity. Shift 3 stops being free.

That is why the fallback Garrido authorised is principled here rather than convenient: we have
measured the failure his alternative was designed to survive.

## 3. Port specification

### 3.1 Variable mapping — DES to index

| index term | MFSC DES source | note |
|---|---|---|
| ζ | time-weighted average of all ration + raw-material containers over the horizon | must include in-transit once conservative sourcing lands, else buffers look free |
| ε | time-weighted average backorder quantity (`backorder_qty` path) | already tracked |
| φ | installed capacity minus realised throughput per period, averaged | needs a spare-capacity recorder; `assembly_shifts` sets installed capacity |
| τ | mean time to meet net requirements | closest existing quantity is `ttr_mean`; the paper's ratio form must be reproduced, not substituted |
| κ̇ | normalised total cost across the compared policy set | **see 3.3 — this term is relative, not absolute** |

### 3.2 Exponent calibration — frozen before any comparison

1. Run the calibration sweep (development tapes only, disjoint from any evaluation block).
2. Record `x_max` for each of ζ, ε, φ, τ, κ̇ over that sweep.
3. Set `exponent_x = 0.20 / ln(x_max)`, exactly Garrido's rule.
4. **Freeze the five exponents and the five maxima in a contract file, with the sweep's hash,
   before evaluating a single policy.** Re-deriving them later, after seeing results, would
   convert a scale normaliser into a tuned preference weight.

### 3.3 The cost term is relative — a genuine hazard

`κ̇(S_ij) = 7·κ(S_ij) / Σ_ij κ(S_ij)` normalises each strategy's cost by the sum over **the
seven substrategies being compared** (the 7 is the count of that set). So κ̇, and therefore R,
is **not a per-policy absolute quantity** — it depends on the comparison set. Adding or
removing a policy changes every other policy's R.

Two consequences, both binding:

- The comparison set must be **declared and frozen before evaluation**, and the generalised
  normaliser is `|S|·κ(s) / Σ_{s∈S} κ(s)`.
- R values are **not comparable across different comparison sets**, and never across papers.
  Any table must state its set.

### 3.4 What must be reported alongside

R never replaces the canonical metric. Every screen reports, side by side:

    ret_excel                (canonical, censored — the thesis Excel replication)
    ret_excel_full_ledger    (same formula, uncensored)
    R_cobb_douglas           (this index, with its frozen comparison set)

plus the five raw components ζ, ε, φ, τ, κ̇, so a reader can see which term drives a movement.

## 4. The trap this must not walk into

The project has already been burned once by exactly this manoeuvre. In **Program G** the
adaptive rule won under a Cobb–Douglas-style proxy and **lost under Garrido's ReT**: the
corrected result was `cover − ABAB = −0.021`, CI95 `[−0.027, −0.015]`, with ABAB also winning
on quantity-weighted ReT and worst-CSSU fill. The lesson recorded then stands: **metric choice
can manufacture adaptive value.**

So the governing rules here are:

- **Triangulate, never select.** All three metrics are reported for every policy, always. A
  result that appears only under R is reported as metric-dependent, not as a win.
- **Freeze before you look.** Exponents, maxima, comparison set and the cost coefficients
  `c_p, c_h, c_ℓ, c_u, c_i, c_b, c_o` are all frozen in a contract before evaluation.
- **Do not inherit `TrackCEconomicsWrapper`'s lambdas** as κ costs. They were frozen for the
  Track B 11D contract, charge only three buffers, and are a scalarisation of a different
  objective. κ needs its own costed ledger.
- **Declare the provenance.** R is Garrido's published index, ported with re-derived scale
  exponents — not the thesis ReT, and not a metric of our invention. State it that way.

## 5. Implementation and remaining calibration boundary

Completed:

1. Periodic spare-capacity recorder from Table 6.20 installed capacity minus realised
   production.
2. Direct Algorithm-2 implementation of gross requirements, net requirements and τ.
3. Seven separate unpriced cost-component means persisted for every episode, allowing
   repricing without replay.
4. Garrido's published `c=1` assumption preserved only as a replication baseline.
5. Frozen one-factor relative-price sensitivity for holding, backorders and spare
   capacity at 0.5×, 2× and 5×.

Still open:

1. A signed domain cost vector with units, price year and source. Until it exists,
   `c=1` is not an MFSC economic calibration.
2. Procurement/injection does not appear among Garrido's seven κ terms and therefore
   remains an explicit physical resource/Pareto axis; it is not silently assigned a
   price.
3. Overtime is structurally absent in this DES and remains zero.

Contract:
`contracts/cobb_douglas_economic_sensitivity_v1.json`.
Artifact:
`results/cobb_douglas/economic_sensitivity_v1/result.json`.

The current result makes the limitation material: the R1r winner is stable across
the frozen relative-price grid, but the R2r winner changes between greedy PI and
MPC. Therefore no scalar economic winner is reported for R2r, and Cobb-Douglas
cannot select a checkpoint or architecture.
