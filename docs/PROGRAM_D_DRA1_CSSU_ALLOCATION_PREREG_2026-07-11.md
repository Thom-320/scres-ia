# Program D — DRA-1: Two-CSSU spatial-allocation discovery gate (PREREGISTRATION, frozen 2026-07-11)

> **2026-07-11 correction (pre-experiment, no DRA-1 run yet):** two amendments after
> the ReT-metric review — (1) primary endpoint is now `ret_excel_visible_v1`
> (`docs/RET_EXCEL_VISIBLE_V1_CONTRACT_2026-07-11.md`), NOT the full-ledger
> `ret_excel`; because the visible ledger omits lost orders (verified: 660 visible vs
> 875 placed on a probe, mean 0.00645 vs 0.00486), it has a "shed-to-win" incentive,
> so the **lost-order guardrail is strictly binding for every headline**. (2) The CSSU
> split is framed honestly as OUR structural extension (see §1), not a demonstrable
> thesis omission.

Status: **frozen before any DRA-1 code/experiment.** Authorized only after D1
reached `STOP_NO_STATE_DEPENDENT_RATIONING_HEADROOM` (368f381) — the *temporal*
zero-sum lever has no deployable dynamic value; DRA-1 tests the *spatial* zero-sum
lever, the committee's #1 family. Primary metric: **`ret_excel_visible_v1`** (Garrido
workbook-visible ledger; see contract doc), co-primary **`service_loss_auc`**, with
**lost orders as a binding guardrail** (the visible ledger omits lost orders, so a
policy must never win by shedding). Mandatory sensitivity: `ret_excel_visible_clipped_0_1`
(the un-clipped recovery branch can exceed 1). Never `ret_thesis`/`order_level_ret_mean`/
`Rsult_1` Re column. No PPO/KAN/RNN/SAC/TD3/MARL before the confirmatory gate.

## 0. Why DRA-1 is different from the five stopped programs
All prior levers (buffer, shift, dispatch-rate, downstream-reserve, D1 rationing)
were either monotone knobs or a temporal reordering of a SINGLE queue — no
competing recipients. DRA-1 introduces the one structural feature none of them had:
**two spatially distinct sinks (CSSU-A, CSSU-B) competing for a fixed daily
capacity, under LOCALIZED disruptions.** The optimal split can genuinely reverse
with state (serve the stressed/undisrupted node), which is the state-contingency
the structural criterion requires. This is the last untested decision-right family
below the CDC.

## 1. Fidelity basis (verified first-hand 2026-07-11)
- **The two-CSSU split is OUR structural extension, not a demonstrable thesis
  omission.** The thesis is ambiguous: Fig 6.2 labels Op11 "Combat Service Support
  Units (2)" and the text mentions combat brigades containing CSSUs, but Fig 6.1 and
  the modelled operations 10–12 treat them AGGREGATED — which our reconstruction
  faithfully mirrors with one `rations_cssu` container. Splitting into A/B is a
  proposed extension to create a spatial allocation decision; it must be presented as
  such and **requires Garrido face-validation before the virgin confirmatory** — never
  sold as reproducing an explicit thesis division.
- **Garrido marks the relevant disruptions per order in his own workbooks**
  (Raw_data2, CF11–CF20): columns `R21_1..R21_5, R22_1..R22_4, R23, R24`. So
  R22 (LOC/arc attacks) and R23 (attacks on forward logistics units = the CSSUs)
  are per-order-attributable disruptions in his ground truth — localizing them to
  drive spatial allocation is FAITHFUL, not invented.
- **Our ReT gate already handles them.** `order_has_ret_risk_indicator` is
  event-driven (keys on `event.risk_id`), not hardcoded to R11–R14. Verified:
  with R21/R22/R23/R24 enabled, the per-order gate fires (R22→87, R23→92, R21→36,
  R24→300 orders on a 52-wk probe). No gate change needed for R2r.
- **Downstream Q**: Fig 6.2/§6.3 **2,400–2,600** is the primary `downstream_q_source`;
  Table 6.20 (2,000–2,500) is a named sensitivity, never averaged.
- **Scale caveat (binding for the paper)**: Garrido's baseline workbook mean ReT
  ≈ 1.0 (near-saturation); risks-on regimes score far lower. DRA-1 reports
  RELATIVE comparisons (policy vs best constant on the same cohort/regime); absolute
  ReT is NOT claimed comparable to Garrido's baseline configs.

## 2. Risk-editing authority (Garrido-authorized) as the TREATMENT — with a placebo
Garrido authorized enabling/disabling and changing frequency/impact of each risk.
DRA-1 uses this ONLY to create identifiable spatial state-dependence, never to fit a
win:
- **Treatment regime** (frozen before results): a localized high-threat scenario —
  R23 directed at ONE CSSU at a time and R22 directed at ONE lane at a time, at a
  pre-registered elevated frequency/impact justified as a "localized intense-threat"
  military scenario (not tuned to results). R21 (natural disasters) may co-occur;
  R3 excluded (OOD only); R24 (contingent demand) localized to one CSSU.
- **Shuffled-localization PLACEBO (V_info identification, committee requirement):**
  an arm with the SAME total R22/R23 intensity but the destination/lane label
  shuffled (uninformative). If the allocation policy beats the best constant ONLY
  under true localization and NOT under the placebo, that isolates the value of
  *knowing where the disruption is* — a clean value-of-information result. If the
  policy beats the constant equally under placebo, the "win" is not informational
  and DRA-1 STOPS.

## 3. Minimal physical contract (built only after this prereg + Garrido sign-off)
- Split `rations_cssu` → `rations_cssu_A`, `rations_cssu_B`; per-CSSU backlog,
  in-transit, and lane state (Op10A/Op10B inbound, Op12A/Op12B outbound).
- **Conserve total demand and total daily dispatch capacity**: q_{A,t}+q_{B,t} ≤
  min{C_t, I_SB,t}; no action increases total throughput — it only allocates it.
- Aggregate demand unchanged; each order destination-tagged A/B (symmetric baseline
  shares; R24 creates temporary asymmetric mission demand at one CSSU).
- Localized R22 hits one lane; localized R23 destroys one CSSU; both preserve the
  thesis-native onset/duration distributions, only the LOCATION is the treatment.
- No partial fulfilment / eviction / capacity / demand / risk as *learned actions*
  in v1 beyond the allocation + service rule defined below.

## 4. Action set, epoch, observations
- **Actions (9)**: allocation share α_A ∈ {0.25, 0.50, 0.75} (α_B = 1−α_A) ×
  service rule ∈ {SPT_FULL (thesis), FIFO_PARTIAL, R24_AGE_PARTIAL}. Enumerable; no
  RL needed to search 9.
- **Epoch**: decide every 24 h immediately before the daily Op9→CSSU dispatch;
  action active one day; one-step latency; transport PT 24 h.
- **Observations (deployable only)**: SB inventory; CSSU-A/B inventory; A/B days of
  cover; A/B backlog qty & count; A/B max/mean age; R24 share per queue; A/B
  in-transit + ETA; A/B route/CSSU status (up/down); recent 7-day demand & fill per
  CSSU; previous allocation; operational-week phase. EXCLUDED: future risk
  type/onset/duration, latent regime, oracle demand, retrospective fields.

## 5. Gate protocol (same discipline as D1-v2; guardrails binding)
- **Liveness**: split conserves mass (per-node balance residual < 1e-6); allocation
  actually moves stock to A vs B; localized R22/R23 actually degrade the targeted
  node/lane. Fail-closed.
- **Static frontier**: all 9 constant policies, paired CRN, 60 calibration tapes
  (localized regime), 30 select / 30 validate; best-admissible constant = highest
  select ReT within service-loss ≤1% and lost ≤2% vs the thesis SPT_FULL/50-50
  reference; guardrails binding.
- **Exact-prefix branching**: replay to identical state (bitwise state + exogenous
  hash asserts, fail-closed), branch the 9 actions 24 h then revert to best
  constant; cohort = orders open at branch ∪ new orders in horizon (backlog NOT
  excluded); horizons 72 h + 28 d; label by 28 d ReT, tie-break service-loss →
  lost → stable order. Cluster inference BY TAPE (not by state).
- **Observable tree**: depth-3, 5 folds grouped by tape, deployable features only,
  evaluated sequentially on held-out tapes.
- **Virgin confirmatory**: freeze code/tree/features/analysis/manifest/git-SHA, then
  ONE opening of 40 virgin tapes. Runner aborts if virgin tapes open before freeze.

## 6. Promotion / stop (frozen)
Promote to sequential control ONLY if ALL: liveness+mass-balance pass; ≥2 allocation
levels each optimal in ≥15% of branch states, none >85%; oracle 28 d service-loss
≥5% lower than best constant with CI95 lower >0; ΔReT co-directional CI95 lower ≥0;
lost-order relative increase CI95 upper ≤2%; tree recovers ≥50% oracle & beats best
constant CI95>0 on held-out; virgin: ΔReT CI95>0, service-loss ≥3% CI95>0, positive
in ≥3/4 strata & ≥70% tapes, CVaR10 not worse >2%; **worst-CSSU floor**:
min(ReT_A, ReT_B) not down >0.01 vs best constant (unless a lower mission priority is
pre-specified); **and the advantage survives ONLY under true localization, not the
shuffled placebo** (V_info). Else emit `STOP_NO_OBSERVABLE_SPATIAL_HEADROOM`.

Binding guardrails (any breach ⇒ stop): gain from shedding orders/backlog eviction;
gain from more total transport; 72 h benefit becoming 28 d damage; benefit requiring
future information; benefit equal under the shuffled-localization placebo.

## 7. Claim ladder (pre-specified)
| Result | Allowed claim |
|---|---|
| STOP at any gate | 6th boundary result → the thesis-complete decision surface below the CDC exposes no deployable dynamically-convertible frontier (strong "Discovering Decision Rights Before RL" paper) |
| branching passes, tree fails | latent spatial headroom, not observable-deployable |
| tree passes, placebo not separated | apparent adaptive value but no information value |
| tree passes AND placebo-separated AND virgin-confirmed | spatial allocation is a genuine state-contingent decision with observable, resource-equivalent value — THEN and only then consider PPO/L(t−1) |

## 8. Non-claims / out of scope
No routing/modes (no defensible geography/fleet params); no lateral transshipment in
v1 (mixing allocation + lateral confounds identification); no product mix; no PPO
before the tree beats the best constant; no L(t−1) before a state-contingent channel
is demonstrated; no averaging the two downstream-Q ranges; Op1/Op2 forbidden.

## 9. Provenance
Depends on the CSSU-split refactor (to be built under its own commit with
liveness/replay-identity/mass-balance/no-privileged-obs tests) + this frozen prereg +
Garrido's face-validation of the A/B split and the localized-threat regime. Committee
verdicts + reconciliation: `docs/external_assessments/`. D1 terminal result:
`docs/PROGRAM_D_D1_V2_BRANCHING_VERDICT_2026-07-11.md`.
