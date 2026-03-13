# Garrido Thesis Alignment Audit

## Scope

Source of truth audited against the repo:

- [`docs/WRAP_Theses_Garrido_Rios_2017.pdf`](/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/docs/WRAP_Theses_Garrido_Rios_2017.pdf)

Code paths audited:

- [`supply_chain/config.py`](/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/supply_chain/config.py)
- [`supply_chain/supply_chain.py`](/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/supply_chain/supply_chain.py)
- [`supply_chain/env_experimental_shifts.py`](/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/supply_chain/env_experimental_shifts.py)
- [`supply_chain/external_env_interface.py`](/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/supply_chain/external_env_interface.py)
- [`supply_chain/dkana.py`](/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/supply_chain/dkana.py)

## Executive Verdict

The repo is **largely aligned** with Garrido at the level of:

- 13-operation MFSC topology
- baseline process times, quantities, reorder points, and risks
- warm-up logic
- scenario-II inventory buffers
- scenario-III short-term capacity by shifts

The repo is **not aligned** with Garrido at the level of:

- thesis output data structure
- order-level resilience accounting
- lost/unattended orders (`Ut`)
- strict action/observation variables from the thesis

The strongest mismatch is this:

- the DES now implements Garrido's **pending-order list with memory**, capped at 60 delayed orders, scheduled by **SPT**, with contingent orders prioritized
- but the wrappers and exported metrics still do **not** reproduce Garrido's full SDM / order-level resilience columns (`Bt`, `Ut`, `APj`, `RPj`, `DPj`) as first-class outputs

That is the core reason David is right to be suspicious about reducing everything to scalar summaries.

## What Matches Garrido

### 1. MFSC topology and operation semantics

The 13-operation chain in the thesis is reproduced in the repo:

- Thesis operations and parameters are described on pages 84-89 of the PDF:
  - Op1-Op12 process definitions
  - demand at Op13
  - Figure 6.2 initial configuration
- The repo mirrors those operations in [`supply_chain/config.py`](/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/supply_chain/config.py) and [`supply_chain/supply_chain.py`](/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/supply_chain/supply_chain.py#L102).

Examples of direct alignment:

- Op1: `PT=672`, `Q=12`, `ROP=4032`
- Op2: `PT=24`, `Q=190000`, `ROP=672`
- Op3/Op4: `PT=24`, `Q=15500`, `ROP=168`
- Op7/Op8: `Q=5000`, `ROP=48` at `S=1`
- Op9-Op12: `PT=24/0/24`, `Q=2400..2600`, `ROP=24`

These are consistent with thesis pages 84-88 and Table 6.20 on page 109.

### 2. Assembly capacity and shift scenarios

The repo matches Garrido's short-term capacity scenario:

- Thesis Table 6.20, page 109
- Repo implementation in [`supply_chain/config.py`](/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/supply_chain/config.py#L379)

Direct matches:

- `S=1`: `op3_q=15500`, `op7_q=5000`, `op7_rop=48`
- `S=2`: `op3_q=31000`, `op7_q=5000`, `op7_rop=24`
- `S=3`: `op3_q=47000`, `op7_q=7000`, `op7_rop=24`

The RL shift action is therefore grounded in Garrido's scenario-III capacity design, even if the RL wrapper uses a continuous signal to choose `S`.

### 3. Inventory buffer scenario

The repo matches Garrido's strategic inventory reserve scenario:

- Thesis Table 6.16, page 107
- Repo implementation in [`supply_chain/config.py`](/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/supply_chain/config.py#L365)

Matches:

- Op3 and Op5 raw material buffers:
  - `15360`, `30720`, `46080`, `61440`, `122880`
- Op9 ration buffers:
  - `15750`, `31500`, `47250`, `63000`, `126000`

### 4. Current and increased risk levels

The repo matches Garrido's current and increased risk coding:

- Thesis Table 6.12, page 105
- Repo implementation in [`supply_chain/config.py`](/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/supply_chain/config.py)

Examples:

- `R11`: current `U(1,168)`, increased `U(1,42)`
- `R12`: current `p=1/11`, increased `p=4/11)`
- `R13`: current `p=1/10`, increased `p=4/10)`
- `R14`: current `p=3/100`, increased `p=8/100)`
- `R21`: current `U(1,16128)`, increased `U(1,4032)`
- `R22`: current `U(1,4032)`, increased `U(1,1344)`
- `R23`: current `U(1,8064)`, increased `U(1,1344)`
- `R24`: current `U(1,672)`, increased `U(1,336)`
- `R3`: current `U(1,161280)`, increased `U(1,80640)`

Important note:

- repo levels `severe` and `severe_extended` are **not** in Garrido
- they are repo extrapolations for RL benchmarking

### 5. Warm-up period

The repo matches Garrido's warm-up concept:

- Thesis Section 6.8.2, page 112
- Garrido: first batch of `Q=5000` rations arriving at Op9, deterministic estimate `838.8` hours
- Repo: [`supply_chain/config.py`](/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/supply_chain/config.py#L435) sets `estimated_deterministic_hrs = 838.8`

## What Only Partially Matches

### 1. CSSUs are aggregated

The thesis describes Op11 as two CSSUs.

- Thesis page 87 says Op11 is handled by CSSUs and Table 6.20 reflects that structure.
- The repo keeps `num_units=2` for Op11 in config, but the simulation state is aggregated into a single buffer/container.

Verdict:

- conceptually aligned
- structurally simplified

### 2. Assembly PT values

The thesis table prints `PT=0` for Op5-Op7 in Table 6.20, but the narrative on pages 85-87 defines the per-ration processing time as `1 / λ = 0.003125 hours`.

The repo uses:

- `0.0031201248...`, i.e. `1 / 320.5`

Verdict:

- aligned with the narrative and Table 6.3 capacity logic
- not literally equal to the rounded `0` shown in Table 6.20

### 3. Demand and backorder service representation

The thesis is order-centric.

- ReT is defined per order `j` in Chapter 5
- SDM in Table 6.25 / Equation 6.4 uses `j, OPTj, OATj, CTj, LTj, ΣBt, ΣUt, APj, RPj, DPj, Rcr/Op`

The repo is mainly step-centric.

- [`supply_chain/supply_chain.py`](/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/supply_chain/supply_chain.py#L314) exposes a 15-d aggregate observation
- [`supply_chain/env_experimental_shifts.py`](/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/supply_chain/env_experimental_shifts.py) derives reward and diagnostics at step level

Verdict:

- useful RL abstraction
- not the same data model as Garrido's original simulation output

## What Does Not Match Garrido

### 1. Backorders, lost orders, and scheduling rule

This gap is now **partially closed in the DES core**.

Garrido, page 98:

- backorders are pending demands not yet delivered
- they accumulate in a pending-order list
- the list is capped at 60 delayed orders
- overflow becomes lost or unattended orders `Ut`
- service rule is SPT
- contingent demand has priority over regular demand

The repo now implements that logic in the base DES:

- delayed orders enter an explicit `pending_backorders` queue
- the queue is capped at 60
- overflow increments `total_unattended_orders`
- backlog fulfillment is contingent-first and then SPT by order size
- delayed orders are removed from the queue only when theatre inventory can satisfy them in full

Code evidence:

- [`supply_chain/supply_chain.py`](/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/supply_chain/supply_chain.py)

Remaining mismatch:

- the wrappers still expose aggregate RL summaries such as `new_backorder_qty`, `pending_backorder_qty`, and `backorder_rate`
- the repo still does **not** emit Garrido's full SDM columns `Bt`, `Ut`, `APj`, `RPj`, `DPj` as the primary simulation output

### 2. Action variables are not Garrido's original experimental variables

Garrido's simulated factors are:

- risk level `Rcr`
- inventory buffers `It,S`
- short-term manufacturing capacity `S`

Our RL action space is:

- `op3_q multiplier`
- `op9_q multiplier`
- `op3_rop multiplier`
- `op9_rop multiplier`
- `assembly_shifts`

Code:

- [`supply_chain/env_experimental_shifts.py`](/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/supply_chain/env_experimental_shifts.py#L647)

Verdict:

- `assembly_shifts` is thesis-grounded
- the four continuous inventory-control signals are a repo extension for RL
- therefore the action space is **not equivalent** to Garrido's original experimental design

### 3. Observation variables are not Garrido's thesis variables

The thesis output variables are order/resilience variables.

Our observation is an engineered control state:

- inventories
- fill rate
- backorder rate
- downtime flags
- time fraction
- pending batch
- contingent demand proxy
- plus `v2`/`v3` temporal features

Code:

- [`supply_chain/supply_chain.py`](/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/supply_chain/supply_chain.py#L314)
- [`supply_chain/external_env_interface.py`](/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/supply_chain/external_env_interface.py#L11)

Verdict:

- useful for RL
- not thesis-identical

### 4. ReT in the repo is an approximation, not Garrido's original order-level computation

Garrido's ReT is order-based and conditioned on `APj`, `RPj`, `DPj-RPj`, and `FRt`.

The repo:

- approximates `ReT_thesis` at step level
- also uses `control_v1` as operational reward for RL

Code:

- [`supply_chain/env_experimental_shifts.py`](/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/supply_chain/env_experimental_shifts.py#L288)

Verdict:

- appropriate as a repo benchmark lane
- not a literal recreation of Garrido's original resilience computation pipeline

## Observation and Action Audit

### Observation audit

`v1`, `v2`, and `v3` are repo-specific observation contracts, not thesis-native variables.

- `v1`: 15 dims, aggregate control state
- `v2`: adds previous-step demand/backorder/disruption
- `v3`: adds normalized cumulative backorder and disruption history

These are valid RL/POMDP mitigations, but they should be described as:

- observed control state for the RL agent
- not as Garrido's original simulation state vector

### Action audit

The action wrapper is a policy layer over thesis parameters.

It is defensible to say:

- action dim 5 is derived from Garrido-relevant control levers

It is not defensible to say:

- Garrido's original model used this RL action space

## DKANA Audit

### What David was right about

For DKANA, scalar summaries are too weak when the underlying phenomenon is structured.

In particular:

- cumulative disruption is naturally per-operation
- backorder state is naturally queue-like and order-aware

So David is right in spirit: scalar-only context was too compressed.

### What is now implemented for DKANA

Without changing the benchmark RL contracts (`v1` and `v2` remain intact), the repo now supports:

- `v3` observation contract with normalized cumulative memory
- extended `state_constraint_context` for external models

Current DKANA-ready export now includes:

- `observations.npy` with `v3` = 20 dims
- `state_constraint_context.npy` with 45 dims

Those 45 dims include:

- previous 22 scalar feasibility/context features
- 3 queue-state scalars:
  - `pending_backorders_count`
  - `pending_backorder_qty`
  - `unattended_orders_total`
- 7-d vector `cum_backorder_rate_*` by inventory node
- 13-d vector `cum_disruption_fraction_op1..op13`

Code:

- [`supply_chain/env_experimental_shifts.py`](/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/supply_chain/env_experimental_shifts.py#L270)
- [`supply_chain/external_env_interface.py`](/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/supply_chain/external_env_interface.py#L54)
- [`supply_chain/dkana.py`](/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/supply_chain/dkana.py#L10)

Important limitation:

- the backorder vector is sparse by construction because the current DES only materializes unmet demand at the theatre sink
- this is better than a single scalar, but it is still **not a full Garrido-style pending-order state**

## What Must Change for Strict Garrido Fidelity

If the goal is to reproduce Garrido more faithfully, the next major implementation should be:

1. derive order-level variables `OPTj, OATj, CTj, LTj, Bt, Ut, APj, RPj, DPj`
2. expose an SDM-like thesis output bundle, not just RL aggregates
3. make the queue state visible in any thesis-faithful validation scripts

Until that exists, no one should claim:

- the repo exactly reproduces Garrido's backorder dynamics

## Practical Recommendation

For the paper and the codebase:

- describe the DES core as thesis-aligned
- describe the RL environment as a controlled abstraction layered on top of the DES
- describe `v3` and the DKANA vector context as POMDP/external-model enhancements
- be explicit that backorder queue dynamics remain a partial mismatch versus Garrido

That framing is accurate and defensible.
