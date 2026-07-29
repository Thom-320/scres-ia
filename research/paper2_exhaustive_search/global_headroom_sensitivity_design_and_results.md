# Global headroom sensitivity: design and audited results

Date: 2026-07-13

## Existing global result

Program I is the repository's completed historical sensitivity analysis of adaptive value rather than raw outcome. It uses the stylized `ret_order`/full-ledger order adapter, not the governing full-DES `ret_excel_request_snapshot_v2`, so it is construct-sensitivity evidence rather than a current-task H_PI/H_obs screen. Its most important within-contract result is interaction-specific:

- scarcity and concurrent spatial demand move `H_obs`;
- information quality, signal lead/persistence and risk magnitude are inert in that frozen contract;
- the located region reproduces on two fresh seed blocks with mean observable advantage about `0.0100-0.0114`;
- its LCB95 remains `0.0072-0.0087`, below the current 0.01 LCB requirement;
- worst-CSSU fill deteriorates about `0.126-0.128`, so the region fails the simultaneous fairness guardrail.

This is a boundary result, not a candidate cell. Increasing risk magnitude or signal quality inside the same spatial contract is a blocked reparameterization.

The complete historical design is now reconstructed in
`results/headroom_gsa/all_cells_reconstruction.json`: 56 Morris cells, 40 GP
search cells and one located-region confirmation, using the original burned
`3,000,001+` tapes. It reproduces every stored aggregate verdict exactly and
retains every null cell. `global_sensitivity_portfolio_inventory.json`
content-addresses this reconstruction together with the 320-row full-DES
Morris design, 140-row pilot, 4,320-row local-branch table and 64-cell
corrective atlas. This makes the executed portfolio complete as reporting; it
does not turn noncanonical or restricted screens into governing ceilings.

The concurrently committed 64-cell atlas is not additional canonical evidence.
Corrective replay reconstructs the omitted `1000*cell_index` seed convention
and reproduces all 64 stored statistics. Its two reported `H_obs>=0.01` cells
used `compute_order_level_ret_excel_formula`, and both fail lost-order
non-inferiority: mean deltas `+0.4792` (CI95 `[+0.1875,+0.8333]`) and `+0.1667`
(CI95 `[+0.0208,+0.3542]`), with no tape improving. Direct visible-ledger
optimization is not a rescue or a boundary: all 64 cells select `HOLD^4`, whose
zero visible rows score 1.0 while all 48 generated orders are lost. Its apparent
64/64 `H_PI=0` is therefore an unguarded sparse-ledger shed-to-win degeneracy,
not the project's guardrail-constrained H_PI ceiling. Paired replay of all 81
fixed sequences proves that changing `r22_prob` from 0 to 0.30 changes R22
arrays but not fixed-sequence scores because `simulate_orders` ignores
`tape.r22`; the belief policy may still change its sequence through its shadow
state. See
`results/paper2_search/voi_ceiling_atlas_corrective_audit.json`.

The full-DES Program-I local branching result is also quantitative:

| Family | Eight-week restricted oracle mean | CI95 | Resource-equivalent? | Verdict |
|---|---:|---:|---|---|
| production posture | `0.00003951` | `[0.00003255,0.00004674]` | no | too small, horizon unstable |
| Op9 dispatch cadence | `0.00001127` | `[0.00000832,0.00001470]` | no | too small, horizon unstable |
| Op10/Op12 cadence split | `0.00001045` | `[0.00000739,0.00001420]` | yes | too small, horizon unstable |

These are ceilings only for their versioned local action portfolios.

## Frozen response for any newly validated candidate

No new numerical global screen is authorized while every fresh family is either domain-blocked or `active_for_bound`. If Garrido supplies the missing facts, the screen must be frozen before results as follows.

### Responses

For every cell and common tape block compute:

1. causal action liveness per action dimension;
2. action-optimal support and ranking reversal;
3. resource-restricted `H_PI` through the canonical aggregator;
4. `H_obs` for a strong non-neural policy;
5. resource-adjusted `H_obs` when separately budgeted resources differ; allocation destinations within one validated fungible team are not separate purchases;
6. lost-order, quantity-ReT, worst-node, backlog-age and tail deltas;
7. signal-placebo rollout contrasts;
8. fixed-calendar/phase replacement contrasts.

Do not screen raw `ReT(policy)`.

### Frozen cheap-to-expensive order

1. Exact zero-liveness and dominance tests.
2. Affected-order influence closure using the exact canonical numerator and visible denominator.
3. Morris screening of paired adaptive-value responses.
4. Sobol/Latin-hypercube variance decomposition over explicitly sourced input distributions.
5. Exact enumeration/DP/MILP or information-relaxation bound at surviving cells.
6. Observable belief/MPC/tree conversion.
7. Adjacent-cell stability and null/placebo tests.
8. Learner only after every pre-learner gate passes.

The unclipped workbook metric is not globally bounded by one and its visible-row denominator can change. Therefore `affected rows / all rows` is not a primary-metric ceiling. It is allowed only as a clipped, fixed-denominator sensitivity. The primary ceiling must optimize the canonical numerator/denominator directly.

### Cell selection

The selection rule is the least-favorable plausible connected region satisfying every pre-learner gate. The maximizer is never selected simply because it has the largest headroom. All screened cells, including nulls, remain in the result artifact.

## Family-specific parameter routing

- Product mix: no parameter grid until BOM, setup, substitution, demand shares and signal timing are signed off.
- Reservation/multimodal transport: no grid until shared fleet, payload/turnaround, booking/dwell and signal data are signed off.
- Alternate-route recourse: the existing v1 grid is retrospective development evidence only. Do not rescreen until the route/fleet/signal facts are signed off; then freeze a true resource-restricted full-horizon oracle, full calendar/classical frontier, componentwise vehicle ledger and route-removal null before opening any result.
- Censored demand: no grid until system-of-record censoring and active-report resource are confirmed.
- Storage: no grid until actual capacity/overflow/relocation data are confirmed.
- Inspection: no grid until Op7 authority, effort-to-sensitivity/specificity, persistent lot state/leading signal, qualified-output consequences and AQL are confirmed. Native no-action inspection has exact incremental `H_PI=H_obs=0`; exogenous per-unit/lot latent draws must be CRN-coupled and R14 metric attribution must be independent of the inspection action.
- Component-specific R13/Op4 kit balancing: no grid until named component delays, BOM, component inventory/pipeline observability, finite mixed-load lift and expedition/substitution rules are confirmed. The current aggregate kernel has exact `H_PI=H_obs=0` for this absent action.
- R14 detected-lot disposition: no grid until discretion, rework time/capacity/yield, replacement material/lead, partial-release/quality rules and persistent risk attribution are confirmed. Current fixed unit-yield same-rate rework weakly dominates discard.
- R21/R3 restoration sequencing: no grid until Garrido supplies Maintenance Battalion team count/skills, base/travel/activation, operation-specific repair work, preemption, damage-assessment lead/error, and whether 120/672 hours is team work or autonomous downtime. Under the current independent parallel-recovery physics the action is absent, so `H_PI=H_obs=0`; the historical restoration-order-invariance claim is not a machine-certified ceiling on a scarce-team extension.
- Mission loadout: no grid until authority, pack/mass limits, pre-departure mission/resupply signals, sealed consumption-order semantics and unused-pack return rules are confirmed. In the current DES the action is absent, so `H_PI=H_obs=0` exactly for this family.
- Integrated M/T/R: do not sweep the already chosen 80% efficacy or 0.85 signal. First certify the existing full-horizon frontier/PI ceiling, validate that one real team is fungible across targets, and settle reserve issue/replenishment semantics.

The implemented effect quotient is the frozen execution route for that existing cell: 30,765,821 calibration and 57,918,762 locked effect executions. It changes compute cost only; it does not alter the preregistered cell or authorize a parameter search.

This routing prevents selecting physical parameters from learner performance.

The thesis-supported current-to-increased occurrence cells for R11–R24/R3 may
be used only after a decision contract is independently licensed. They change
occurrence frequency, not observation or action authority. Program I already
shows that magnitude/frequency changes alone do not create the required
scarcity-by-concurrency feedback mechanism in its frozen contract.
