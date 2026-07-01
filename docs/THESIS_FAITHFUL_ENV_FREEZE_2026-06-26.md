# Thesis-Faithful Environment Freeze (2026-06-26)

Frozen DES environment for the **1-to-1 PPO-vs-static comparison** (Idea 1). Every policy —
static (Garrido's own configs) and dynamic (PPO) — is evaluated in **this** environment under
common random numbers. Machine-readable freeze:
`supply_chain/data/thesis_faithful_env_freeze_2026-06-26.json`. Source of truth in code:
`supply_chain.config.THESIS_FAITHFUL_PROTOCOL`.

## Frozen configuration

| Field | Value | Why |
| --- | --- | --- |
| year_basis | `thesis` (8064 h/yr) | thesis calendar |
| horizon_hours | 161280 (20 yr) | thesis run length |
| warmup_trigger | `op9_arrival` | thesis warm-up (≈838 h fill pipeline) |
| downstream_q_source | `figure_6_2` | reproduces Table 6.11; `table_6_20` is a sensitivity lane |
| r14_defect_mode | `thesis_strict_op6` | defects rework at Op6 (Table 6.6b) |
| risk_occurrence_mode | `thesis_window` | reproduces Table 6.11 frequencies |
| raw_material_flow_mode | `kit_equivalent_order_up_to` (m=2.0) | best match to Table 6.10/6.1 |
| stochastic_pt | `false` | deterministic PT (faithful); stochastic is a fine-tuned-env knob |
| risk_freq/impact/demand multipliers | 1.0 | no modulation in the faithful lane |

## ReT-fidelity fixes (the four that landed the R1 scale)

These corrected the endogenous ReT from ~0.20 (≈30× too high) to Garrido's scale:

1. **`demand_on_hand_fulfillment_delay = 54 h`** — orders incur the delivery cycle (no instant
   CTj=0 orders; ~all orders late like Garrido).
2. **`GARRIDO_R14_RET_PERIOD_HOURS = 72 h`** — R14-only orders contribute 72 h to AP/RP/DP
   (Garrido's R14-only RPj median is exactly 72 h), so they score 0.5/72≈0.007, not 0.5/1=0.5.
3. **Unfulfilled orders score 0** (`compute_ret_per_order_excel_formula`) — lost/pending orders
   no longer fall into the high no-risk fill-rate branch (Garrido's lost orders score ~0.002).
4. **Excel fill-rate `Bt` capped at `BACKORDER_QUEUE_CAP=60`** — matches Garrido's capped backlog
   ledger; our cumulative count grew to thousands and crushed the fill-rate branch.

## PPO action / observation (Track A)

- Action: `Discrete(18)` = 6 inventory levels `{0,168,336,504,672,1344}` × 3 shifts `{1,2,3}`.
- Observation: `v4`. Decision cadence: weekly (168 h).

## Verification (2026-06-26)

- **R1 (CF1-10):** DES mean ReT **0.0044** vs Excel **0.0063** (ratio 0.71×) — same scale.
- **Forensic lane** (`excel_order_tape`+`excel_risk_tape`): ReT MAE **0.0038** — reproduces Excel.
- **Faithful-lane tests:** 49 passed (`test_thesis_faithful_lane`, `test_clean_metrics`,
  `test_garrido_excel_ret`, `test_garrido_replication_harness`).

## Known limitations (documented, not hidden)

- **R2/CF11-20 endogenous:** under the R2-only gate the DES under-attributes R22/R23/R24 to orders
  → R2 ReT does not match Garrido. Does **not** affect the training regimes (R14 ubiquitous there →
  no no-risk orders). Stays a documented limitation.
- **CTj/RPj extreme tail:** endogenous p99 ≈ 2× Garrido's; affects CVaR/p95 tail metrics, not the
  mean ReT.
- **Excel ReT is a quirky metric:** used as an **evaluation bar**, not a training target (Reward A
  trains on a dense operational signal; `ReT_excel_delta` direct-training is an optional sensitivity).

The fine-tuned headroom environment (Idea 2) extends THIS freeze with pre-registered realism knobs;
see the PPO experiment protocol.
