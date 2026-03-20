# Project Documentation: MFSC DES + RL for Supply Chain Resilience

**Last updated:** 2026-03-18

---

## 1. Project Overview and Motivation

This project rebuilds the Military Food Supply Chain (MFSC) discrete-event simulation from Garrido-Rios' 2017 PhD thesis (originally MATLAB/Simulink) in Python using SimPy, wraps it in Gymnasium for reinforcement learning, and trains agents with Stable-Baselines3.

**Research question:** Can an RL agent learn shift-control and inventory policies that improve supply chain resilience under disruption, compared to static configurations?

**Publication target:** Q1/Q2 journal (IJPR primary, IEEE TAI secondary, EJOR stretch). Contribution framed as *reward design and auditing for operational resilience control under disruptions*, not architectural novelty.

---

## 2. Thesis Summary (Garrido-Rios 2017)

**Title:** "A Mixed-Method Study on the Effectiveness of a Buffering Strategy in the Relationship between Risks and Resilience"

**Institution:** University of Warwick, Warwick Manufacturing Group (WMG)

**Core contributions:**
1. Unified definition of SCRes: "The adaptive ability of the supply chain/network to respond to/react to/resist to unexpected operational disruptions, recover from them and return to the original/desired state."
2. Novel measure of resilience (ReT) based on the *tail autotomy effect* (TAE) from biology
3. ReT integrates four sub-indicators: autotomy period (APj), recovery period (RPj), disruption period (DPj), and fill rate (FRt)
4. Simulation of 90 configurations under 3 scenarios testing 9 hypotheses about buffering strategies

**ReT formula (Eq. 5.5):**
```
           | Re_max * (APj / LT)           during autotomy (CT = LT despite risks)
           | Re     * (1 / RPj)            during recovery (CT > LT, recovering)
ReT(j) =  | Re_min * (DPj - RPj) / CTj    during non-recovery (CT > LT, not recovering)
           | 1 - (Bt + Ut) / Dt            during non-disruption (fill rate)
```
Where Re_max = 1, Re = 1, Re_min = 0.

**Key findings:**
- Inventory buffers positively moderate risks-resilience relationship (99% confidence)
- Short-term manufacturing capacity (shifts) positively moderates for R1r and R3 (95%+)
- Staff preference: inventory buffers over capacity increases

---

## 3. Architecture

### Data Flow

```
config.py (thesis params)
    |
    v
MFSCSimulation (SimPy DES engine, hourly assembly granularity)
    |
    +---> run() for batch validation
    |
    +---> step(action, step_hours) for RL
              |
              v
       MFSCGymEnv (4-dim actions)        -- base env
       MFSCGymEnvShifts (5-dim actions)  -- shift control env
              |
              v
       train_agent.py (PPO via SB3, VecNormalize, Monitor)
              |
              v
       outputs/ (models, curves, JSON, CSV)
```

### Supply Chain Topology (13 Operations)

```
[Op1: MLA]          Contracting (biannual, PT=672h)
    |
    v
[Op2: Suppliers]    Raw material delivery (monthly, PT=24h, 12 suppliers x 190,000 units)
    |
    v
[Op3: WDC]          Reception & storage (weekly, PT=24h, Q=15,500/rm)
    |
    v
[Op4: LOC]          Transport WDC->AL (PT=24h)
    |
    v
[Op5: Pre-assembly] ---+
[Op6: Assembly]     ---+-- Assembly Line (lambda=320.5 rations/hr, hourly granularity)
[Op7: QC & Ship]   ---+   Batch=5,000 rations
    |
    v
[Op8: LOC]          Transport AL->SB (PT=24h)
    |
    v
[Op9: SB]           Receipt & dispatch (daily, PT=24h, Q=U(2400,2600))
    |
    v
[Op10: LOC]         Transport SB->CSSU (daily, PT=24h)
    |
    v
[Op11: CSSUs]       Cross-docking (PT=0, 2 units)
    |
    v
[Op12: LOC]         Transport CSSU->Theatre (daily, PT=24h)
    |
    v
[Op13: Theatre]     Demand sink (U(2400,2600) rations/day, 6 days/week)
```

### Risk Mapping

```
Category 1 - Operational (R1r):
  R11: Workstation breakdowns       -> Op5, Op6    (U(1,168), Exp(2))
  R12: Contract delays              -> Op1         (B(12, 1/11))
  R13: Raw material shortages       -> Op2         (B(12, 1/10))
  R14: Defective products           -> Op7         (B(2564, 3/100))

Category 2 - Disasters/Attacks (R2r):
  R21: Natural disasters            -> Op3,5,6,7,9 (U(1,16128), Exp(120))
  R22: LOC destruction              -> Op4,8,10,12  (U(1,4032), Exp(24))
  R23: Forward unit destruction     -> Op11         (U(1,8064), Exp(120))
  R24: Contingent demand surge      -> Op13         (U(1,672), +U(2400,2600))

Category 3 - Black-swan (R3):
  R3:  Black-swan event             -> Op5,6,7,9   (U(1,161280), fixed 672h)
```

---

## 4. DES Model Specification

### Parameters

| Constant | Value | Source |
|----------|-------|--------|
| Assembly rate (lambda) | 320.5 rations/hr | Table 6.3 |
| Hours per shift | 8 | Table 6.2 |
| Operating days/week | 6 (Mon-Sat) | Section 6.3 |
| Batch size (S=1,2) | 5,000 rations | Table 6.20 |
| Batch size (S=3) | 7,000 rations | Table 6.20 |
| Simulation horizon | 161,280 hrs (20 yrs) | Section 6.8.1 |
| Warmup | ~838.8 hrs (deterministic) | Section 6.8.2 |
| Backorder queue | Cap 60, SPT, contingent priority | Section 6.5.4 |
| Year basis (thesis) | 336 days = 8,064 hrs | Table 6.2 |

### Timing and Granularity

- **Assembly line:** Hourly ticks (improvement over thesis Simulink)
- **Procurement:** Biannual (Op1), monthly (Op2), weekly (Op3/Op4)
- **Distribution:** Daily (Op9-Op12)
- **Demand:** Daily, 6 days/week
- **RL step:** 168 hrs (weekly) by default

### Risk Levels

| Level | Description | Source |
|-------|-------------|--------|
| `current` | Thesis '-' column in Table 6.12 | Table 6.12 |
| `increased` | Thesis '+' column in Table 6.12 | Table 6.12 |
| `severe` | Extrapolated (2x frequency of increased) | Repo extension |
| `severe_extended` | Severe + scaled disruption magnitudes | Repo extension |

---

## 5. Gymnasium Environment Specification

### Base Env: MFSCGymEnv (env.py)

- **Observation:** 15-dim continuous (inventory levels, fill/backorder rates, disruption flags, time)
- **Action:** 4-dim [-1, 1] mapped to [0.5, 2.0] multipliers via `1.25 + 0.75 * action`
  - [0]: op3_q multiplier (WDC dispatch quantity)
  - [1]: op9_q multiplier (SB dispatch quantity)
  - [2]: op3_rop multiplier (WDC dispatch frequency)
  - [3]: op9_rop multiplier (SB dispatch frequency)
- **Reward modes:** `proxy`, `rt_v0`

### Shift Control Env: MFSCGymEnvShifts (env_experimental_shifts.py)

- **Observation:** 15-dim (v1), 18-dim (v2), or 20-dim (v3)
  - v2 adds: prev_step_demand, prev_step_backorder, prev_step_disruption
  - v3 adds: cumulative_backorder_rate, cumulative_disruption_fraction
- **Action:** 5-dim [-1, 1]
  - [0-3]: Same inventory multipliers as base env
  - [4]: Shift selector: <-0.33 -> S=1, [-0.33,0.33) -> S=2, >=0.33 -> S=3
- **Reward modes:** `ReT_thesis`, `rt_v0`, `control_v1`, `control_v1_pbrs`

### Observation Vector Detail (v1, 15-dim)

| Index | Field | Normalization |
|-------|-------|--------------|
| 0 | raw_material_wdc | / 1e6 |
| 1 | raw_material_al | / 1e6 |
| 2 | rations_al | / 1e5 |
| 3 | rations_sb | / 1e5 |
| 4 | rations_cssu | / 1e5 |
| 5 | rations_theatre | / 1e5 |
| 6 | fill_rate | [0, 1] |
| 7 | backorder_rate | [0, 1] |
| 8 | assembly_line_down | {0, 1} |
| 9 | any_loc_down | {0, 1} |
| 10 | op9_down | {0, 1} |
| 11 | op11_down | {0, 1} |
| 12 | time_fraction | [0, 1] |
| 13 | pending_batch / batch_size | [0, 1] |
| 14 | contingent_demand / 2600 | [0, ...] |

---

## 6. Reward Functions

### ReT_thesis (Garrido Eq. 5.5 Step-Level Approximation)

Maps the thesis's order-level ReT to step-level metrics:
- **Case 1 (No disruption):** `Re = fill_rate`
- **Case 2 (Autotomy):** `Re = 1 - disruption_fraction` (FR >= 0.95 despite disruption)
- **Case 3 (Recovery):** `Re = 1 / (1 + disruption_fraction)` (FR < 0.95, disruption present)
- **Case 4 (Non-recovery):** `Re = 0` (high disruption AND low fill rate)

Step reward: `ReT_step - delta * (S - 1)`

**Status:** Reporting-only metric. Not used as training objective (collapses to S1).

### control_v1 (Primary Benchmarking Reward)

```
reward = -(w_bo * service_loss + w_cost * shift_cost + w_disr * disruption_fraction)
```

Where:
- `service_loss = new_backorder_qty / new_demanded` (ration-weighted)
- `shift_cost = S - 1` (linear: S=1 free, S=2 costs 1, S=3 costs 2)
- `disruption_fraction = disruption_hours / (step_hours * 13)`

Default weights: `w_bo=1.0, w_cost=0.06, w_disr=0.0`

### control_v1_pbrs (PBRS Extension)

Adds potential-based reward shaping: `F = gamma * phi(s') - phi(s)`
- Cumulative variant: `phi(s) = -alpha * max(0, tau - FR) / tau`
- Step-level variant: `phi(s) = -alpha * prev_step_backorder_norm`

Preserves optimal policy per Ng et al. (1999).

---

## 7. Benchmark Results Summary

### Preliminary Results (Frozen)

**PPO + ReT_thesis:** Misaligned -- agent collapses to S1 (cheapest shift, poor service). Not suitable as training reward.

**PPO + control_v1 (50k steps):** Weight-sensitive. PPO outperforms static baselines only in a narrow weight region.

**PPO + control_v1 (500k steps, 5 seeds):**
- Under `increased` risk: Competitive with best static baseline
- Under `severe` risk: Slight advantage over best static (p~0.19)

**Three-part bottleneck identified:**
1. Reward alignment (ReT_thesis -> control_v1)
2. Reward sensitivity (narrow optimal weight region)
3. Regime dependence (gains only under higher stress)

### Static Baselines

| Config | Shifts | Fill Rate | Throughput |
|--------|--------|-----------|------------|
| S1 | 1 | Baseline | ~738k/yr |
| S2 | 2 | Higher | ~1.48M/yr |
| S3 | 3 | Highest | ~2.22M/yr |

### Heuristic Baselines

- **HeuristicHysteresis:** Deadband shift control based on fill rate thresholds
- **HeuristicDisruptionAware:** Reactive shift + inventory based on disruption signals
- **HeuristicTuned:** Combined, grid-searchable

---

## 8. Known Gaps and Limitations

### DES Discrepancies vs Thesis (from Audit 2026-03-18)

1. **[MEDIUM] R12/Op1-Op2 coupling missing** -- R12 delays Op1 but doesn't gate Op2 supplier deliveries
2. **[MINOR] Op8 event-triggered** -- ships on material availability, not 48h schedule
3. **[MINOR] R14 defects discarded** -- thesis says reprocess, code removes
4. **[MINOR] Warmup trigger fires early** -- on production vs Op9 receipt
5. **[LOW] PT rounding** -- thesis internal inconsistency (0.003125 vs 1/320.5)

### Thesis Limitations (Acknowledged)

- Single product assumption (21 ration types -> 1)
- No cost factor in ReT
- Single MFSC unit of analysis
- 2 CSSUs aggregated into 1 node
- Deterministic processing times in thesis (stochastic_pt is a repo extension)

### RL-Specific Limitations

- Partial observability (15-dim obs doesn't capture full state)
- Weekly step size may miss within-week dynamics
- VecNormalize running statistics can drift across training
- PPO MLP architecture may not capture temporal dependencies

---

## 9. What's Next

### Immediate (Phase 2 Experiments)

1. **RecurrentPPO comparison:** LSTM-based policy (sb3-contrib) to address POMDP
2. **Frame-stacking:** VecFrameStack n=4 to provide temporal context to MLP
3. **Tuned heuristic baseline:** Grid-search over hysteresis parameters
4. **10-seed benchmark expansion:** From 5 to 10 seeds for statistical robustness

### Manuscript Writing (Current Cycle)

1. Section 4.2: Frozen results from 500k benchmarks
2. Section 4.3: Algorithm comparison (PPO vs SAC vs RecurrentPPO vs heuristic)
3. Section 5: Discussion, limitations, future work

### Deferred (Future Work)

- DKANA integration (causal self-attention policy architecture)
- SAC comparison
- PBRS systematic evaluation
- Multi-product extension
- Cost factor integration

---

## 10. File-by-File Reference

### Core Package: supply_chain/

| File | Purpose | Lines |
|------|---------|-------|
| `config.py` | Single source of truth for all simulation parameters | ~500 |
| `supply_chain.py` | SimPy DES engine (13 ops, hourly assembly, step API) | ~1,030 |
| `env.py` | MFSCGymEnv: base Gymnasium wrapper (4-dim actions) | ~178 |
| `env_experimental_shifts.py` | MFSCGymEnvShifts: shift control env (5-dim actions, 4 reward modes) | ~780 |
| `external_env_interface.py` | ExternalEnvSpec + run_episodes() for external model integration | ~472 |
| `dkana.py` | DKANA-compatible pipeline and policy architecture | ~varies |

### Entry Points (Root)

| File | Purpose |
|------|---------|
| `run_static.py` | Deterministic/stochastic baselines, validates against thesis |
| `train_agent.py` | PPO training pipeline (SB3, VecNormalize, Monitor) |
| `validation_report.py` | Dual-basis validation tables |

### Benchmark Scripts (scripts/)

| File | Purpose |
|------|---------|
| `benchmark_control_reward.py` | Multi-seed PPO/SAC vs static/heuristic under control_v1 |
| `benchmark_delta_sweep_static.py` | Delta parameter sensitivity for shift cost |
| `benchmark_ret_ablation_static.py` | ReT formula variant ablation |
| `benchmark_minimal_shift_control.py` | Quick multi-seed PPO vs static |
| `formal_evaluation.py` | End-to-end evaluation of pre-trained models |
| `export_trajectories_for_david.py` | Export trajectory .npy files for DKANA |
| `build_dkana_dataset.py` | Convert trajectories to DKANA-ready windows |

### Tests (tests/)

Pytest suite covering: env contracts, reward components, benchmark aggregation, DKANA pipeline, delta sweep, ReT ablation, formal evaluation.

### Documentation (docs/)

| File/Dir | Purpose |
|----------|---------|
| `WRAP_Theses_Garrido_Rios_2017.pdf` | Original thesis |
| `DKANA_INTEGRATION_GUIDE.md` | External model integration guide |
| `briefs/` | Research briefs (5 files): alignment audit, control reward memo, preliminary results, repo cleanup, shift control audit |
| `manuscript_notes/` | Writing guidance (8 files): claim language, frozen results, strategy, execution backlog, writeup backlog |
| `artifacts/` | Benchmark artifacts and analysis outputs |
| `meeting_packages/` | Garrido meeting materials |

---

*This document was generated 2026-03-18. For the latest parameter values, always refer to `config.py`.*
