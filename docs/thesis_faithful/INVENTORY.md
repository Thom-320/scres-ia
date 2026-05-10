# Thesis Extraction Inventory

This inventory records the thesis items that the `thesis_1to1` lane must keep
auditable. The PDF and `thesis.txt` remain the primary sources; older repo docs
are secondary evidence only.

## Core Timing And Capacity

| Item | Thesis source | Extracted value | Repo constant |
|---|---|---:|---|
| Simulation horizon | Section 6.8.1 | `161,280h` | `SIMULATION_HORIZON` |
| Thesis year | Table 6.2 | `8,064h` | `HOURS_PER_YEAR_THESIS` |
| Shift length | Table 6.2 | `8h` | `HOURS_PER_SHIFT` |
| Assembly rate | Table 6.3 | `320.5 rations/hour` | `ASSEMBLY_RATE` |
| EC1 | Table 6.3 | `738,432 rations/year` | capacity check |
| Warm-up | Section 6.8.2 | first `Q=5,000` arrival at Op9 | `warmup_trigger="op9_arrival"` |

## Demand And Validation

| Item | Thesis source | Extracted value | Repo constant |
|---|---|---:|---|
| Regular demand | Table 6.4 | `U(2400,2600)`, daily, 6 days/week | `DEMAND` |
| Lead-time promise | Section 6.8.2 | `48h` | `LEAD_TIME_PROMISE` |
| Table 6.10 ECS avg | Table 6.10 | `767,591.5` | `VALIDATION_TABLE_6_10` |
| Table 6.10 RMSE | Table 6.10 | `87,918` | `VALIDATION_TABLE_6_10["RMSE"]` |

## Downstream Q Discrepancy

The thesis contains a useful tension:

- Figure 6.2 and Section 6.3.3: `U(2400,2600)`
- Table 6.20: `U(2000,2500)`

The lane defaults to `figure_6_2` because it matches the demand model and the
current repo baseline. The launcher exposes `--downstream-q-source table_6_20`
so this discrepancy can be tested directly instead of hidden in code.

## Scenario Families

| Family | Thesis source | Harness scenario |
|---|---|---|
| Initial configuration `Cf0` | Figure 6.2 | `cf0` |
| Current stochastic risks | Tables 6.6-6.8, 6.12 | `stochastic_current` |
| Inventory buffers `It,S` | Table 6.16 | `inventory_168`, `inventory_336`, `inventory_504`, `inventory_672`, `inventory_1344` |
| Manufacturing capacity `S` | Table 6.20 | `capacity_s1`, `capacity_s2`, `capacity_s3` |

## Risk Model

| Risk | Current level source | Current value |
|---|---|---|
| R11 | Table 6.12 | occurrence `U(1,168)`, recovery `Exp(mean=2h)` |
| R12 | Table 6.12 | `B(n=12,p=1/11)`, `168h` delay per contract |
| R13 | Table 6.12 | `B(n=12,p=1/10)`, `24h` delay per delivery |
| R14 | Table 6.12 | `B(n=2564,p=3/100)`, defects returned to Op6 for reprocessing |
| R21 | Table 6.12 | occurrence `U(1,16128)`, recovery `Exp(mean=120h)` |
| R22 | Table 6.12 | occurrence `U(1,4032)`, recovery `Exp(mean=24h)` |
| R23 | Table 6.12 | occurrence `U(1,8064)`, recovery `Exp(mean=120h)` |
| R24 | Table 6.12 | occurrence `U(1,672)`, contingent demand `U(2400,2600)` |
| R3 | Table 6.12 | occurrence `U(1,161280)`, recovery `672h` |

## Output Fields

The lane exports order-level traces aligned with Table 6.25:

- `Cfi`
- `j`
- `OPTj`
- `OATj`
- `CTj`
- `LTj`
- backorder/lost/contingent flags
- `APj`
- `RPj`
- `DPj`
- risk event trace by seed

## Factorial Design Coverage

The factorial harness encodes Cf1-Cf90:

- Cf1-Cf30: risk level matrices for R1r, R2r, and R3.
- Cf31-Cf60: inventory-period matrices `It,S` paired to the same risk blocks.
- Cf61-Cf90: capacity matrices `S` paired to the same risk blocks.

The risk source row for Cf31-Cf60 is `Cfi - 30`; the risk source row for
Cf61-Cf90 is `Cfi - 60`. This keeps Cf85 paired with Cf25, not Cf55.
