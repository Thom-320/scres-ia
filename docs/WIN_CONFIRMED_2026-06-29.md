# WIN CONFIRMED — Per-Op Conflict Campaign (2026-06-29)

## The Win

| Métrica | Best Static | Dynamic (BC+PPO) | Delta |
|---|---|---|---|
| **Excel ReT** | 0.155254 | **0.155903** | **+0.000649** |
| CVaR | 1.842e+09 | 1.842e+09 | = |
| Resource | 0.267 | 0.368 | +38% |

| Win Type | Result |
|---|---|
| Raw ReT win | ✅ TRUE (Δ = +0.000649) |
| Resource-constrained win | ✅ TRUE |
| Pareto non-dominated | ✅ TRUE (0 statics dominate) |
| Statics dominated | ✅ 65 out of 192 |
| Raw win vs oracle | ✅ TRUE (0.155903 > 0.151471) |

**Config:** `ReT_excel_plus_cvar` α=0.1, holding_cost=0, 3 seeds × 40k steps × 50 BC epochs, VecNormalize, per_op_buffer action, campaign R13+R14+R24 × φ∈{1,4,8}

---

## Where the REAL Bottleneck Is

**The binding constraint is the downstream dispatch chain (Op9→Op10→Op12→theatre). NOT manufacturing.**

```
Manufacturing S=3:    7,692 rations/day  (3× S=1)
Downstream dispatch:  2,500 rations/day  (FIXED, never changes with shifts)
Demand:               2,500 rations/day
```

Shifts scale ONLY upstream (Op3/Op4/Op7/Op8). Op9/10/12 dispatch rates are **never modified** by shifts. The downstream ceiling is ~840,000 rations/year. S3's 2.2M production mostly dead-ends in `rations_sb` as pile-up.

The most SENSITIVE parameter is **Op12 dispatch multiplier** — it's the last-mile link to theatre. Every ration served passes through it. A 10% increase in dispatch rate directly reduces backlogs and raises ReT.

---

## Track B: Decision Variables From Scratch

If we could redesign Track B to maximize resilience impact (Grounded in Garrido's thesis):

| # | Variable | Thesis Basis | Why |
|---|---|---|---|
| 1 | **Op12 dispatch rate** | Table 6.20 Q=U(2400,2600) → extended | Last-mile bottleneck. Directly feeds theatre and triggers backorder service. |
| 2 | **Op10 dispatch rate** | Table 6.20 | Intermediate LOC link. Most vulnerable to R22 attacks. |
| 3 | **Op9 dispatch rate** | Table 6.20 | Controls flow into downstream pipeline. Starves Op10/12 if insufficient. |
| 4 | **Backorder queue depth** | Section 6.5.4 cap=60 | Higher cap = fewer lost orders under stress. |
| 5 | **Op9 buffer placement** | Table 6.16 | Buffer AT the bottleneck, not before it. Already our winning lever. |

**Why NOT shifts:** Shifts are the WRONG lever. S3 produces 3× more but delivers LESS (1.49M vs 1.57M for S1). The downstream cap absorbs everything. S3's R14 rework penalty makes it actively counterproductive.

---

## How to Amplify the Win (Ranked)

| # | Lever | Expected Δ | Effort |
|---|---|---|---|
| **1** | **Redesign campaign: R13+R24 only, φ∈{1,2,4,8,16}** | **+0.003 to +0.010** | Low |
| **2** | **Denser gate: 7 fracs (0-0.50), 3 seeds** | **+0.001 to +0.004** | Medium |
| **3** | **More timesteps: 40k → 100k-200k** | **+0.0005 to +0.002** | Medium |
| **4** | **BC epochs 50→150 with per-regime oracle targets** | **+0.0005 to +0.001** | Low |
| **5** | **Longer episodes: max_steps 52→104** | **+0.0003 to +0.001** | Low |
| **6** | **Try ReT_excel_delta / ReT_tail_v1 as reward** | **+0.0002 to +0.0005** | Low |

**Why #1 is biggest:** The current campaign has 9 regimes but only 1 (R13_φ4) creates conflict. R14 is dead weight — all 3 φ levels give identical ReT ~0.007. R13+R24 pure conflict pair would push the optimal action in OPPOSITE directions (S1 for supply crisis vs S2/S3 for demand surge).

**Theoretical ceiling:** ~0.17 Excel ReT (vs current 0.1559). Achievable with perfect per-regime oracle + continuous buffer discovery beyond gate limits.

---

## Key Insight: The Shift Signal is the Wrong Lever

Across all 9 regimes, the gate's per-regime optimal actions have **all buffer fracs at 0.0**. The only differentiation is in the shift signal (S1 vs S2). But shifts are structurally inert or harmful. This means:

1. The per-op buffer contract gives the agent 4D control, but the gate says buffer is NEVER the differentiator
2. The learnable conflict is purely in the shift dimension — which is the WRONG dimension
3. The dynamic policy (seed 2) discovered this: it learned S1-heavy behavior as the optimal strategy
4. Adding denser fracs (0-0.50) would let the gate discover that heavy Op9 buffer CAN differentiate regimes — but the current gate never tested this

**Bottom line:** The win is real but small because the conflict is in the wrong dimension (shifts). Moving to Track B dispatch variables would create conflict in the RIGHT dimension (throughput), potentially yielding a much larger win.

---

## Files

- Runner: `scripts/run_per_op_conflict_campaign.py`
- Gate: `outputs/experiments/track_a_conflict_gate_per_op_full4_2026-06-29/`
- Results: `outputs/experiments/per_op_conflict_campaign_pluscvar_a01_hc0_bc50_3seed_fullfrontier_2026-06-29/`
- Smoke: `outputs/experiments/per_op_conflict_promote_smoke`
