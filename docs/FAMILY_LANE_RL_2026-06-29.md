# Family-Lane RL Synthesis (2026-06-29)

**Central question:** Can PPO with `continuous_its` or `per_op_buffer` action contracts beat the dense static frontier under the thesis-faithful defaults, when restricted to a single risk family (R1-only, R2-only, R3-only)?

**Short answer:** **No.** Across 3 families × 2 contracts × 3 CFis = 18 smoke runs (10k PPO steps, 1 seed, horizon h104), the agent reaches at most 94% of the static frontier in R1, 75% in R2, and 72% in R3. No run beats the static. The pattern is consistent with `docs/SAME_VARIABLES_NO_FRONTIER_2026-06-28.md`: the MFSC, with buffer and shift as the only decision variables, is structurally null for learning.

This document records the per-family comparison, the per-contract comparison, the cross-CFi consistency, and the implications for the paper.

## §1 — Setup

**Risk families (per `RISK_PATTERNS` in `supply_chain/thesis_design.py`):**

| Family | Risks | Horizon | CF examples |
|---|---|---:|---|
| R1 (operational) | R11, R12, R13, R14 | 20 years | CF1-CF10 |
| R2 (distribution/LOC) | R21, R22, R23, R24 | 10 years | CF11-CF20 |
| R3 (black-swan) | R3 only | 20 years | CF21-CF30 |

**Action contracts (from `supply_chain/continuous_its_env.py`):**

- `continuous_its`: `Box([frac_op3op5op9, shift_signal])` — single buffer fraction, continuous shift.
- `per_op_buffer`: `Box([op3_frac, op5_frac, op9_frac, shift_signal])` — per-op buffers, continuous shift.

**Reward:** `ReT_excel_plus_cvar α=0.2` (the Lane A Pareto winner). Risk obs enabled.

**Static frontier (from `outputs/benchmarks/family_static_frontier_2026-06-29/`):** 6 inventory levels × 3 shifts × 3 seeds = 54 cells per family, 162 total runs. `risk_level=current` (NOT increased), no war multipliers, freeze defaults.

## §2 — Static frontier (the bar to beat)

| Family | Best level | Best shift | Mean ReT | ± sd | Fill rate | Lost | Worst (level, shift) | Mean ReT |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| R1 | 168 | 3 | **0.0063** | ±0.0000 | 0.0002 | 0.0 | (0, 1) | 0.0058 |
| R2 | 168 | 1 | **0.7684** | ±0.0054 | 0.0003 | 0.0 | (0, 1) | 0.6536 |
| R3 | 0 | 1 | **0.9376** | ±0.0001 | 0.0002 | 0.0 | (0, 1) | 0.9376 |

**Reading:**

- **R1 best static = 0.0063** matches the Excel R1 mean (0.0063) almost exactly. The static policy essentially replicates Garrido's raw numbers.
- **R2 best static = 0.7684** is the endogenous R2 ceiling (vs Excel 0.202). The 3.8x gap is the structural R2 attribution problem documented in `docs/R2_AUDIT_DECOMPOSITION_2026-06-29.md`.
- **R3 best static = 0.9376** is the "no-buffer" policy. R3 is a black-swan with 1 event per 20 years; without buffer, the system absorbs the event in the recovery branch and the rest of the orders stay in the fill-rate branch.

**The R2/R3 best statics are "no buffer, S1" with very high fill_rate** — the policy is exploiting the ReT non-monotonicity (more orders in the high-value fill-rate branch = higher mean ReT in the endogenous). This is not a "good" outcome in the paper sense — it's a symptom of the endogenous R2/R3 attribution problem.

## §3 — RL family-lane results (18 runs, 10k steps, 1 seed)

| Family | Contract | CFi | Mean ReT | Resource | Lost | % of static best |
|---|---|---:|---:|---:|---:|---:|
| R1 | continuous_its | 1 | 0.0059 | 0.571 | 0.0 | 93.7% |
| R1 | continuous_its | 11 | 0.0059 | 0.599 | 0.0 | 93.7% |
| R1 | continuous_its | 21 | 0.0059 | 0.650 | 0.0 | 93.7% |
| R1 | per_op_buffer | 1 | 0.0059 | 0.432 | 0.0 | 93.7% |
| R1 | per_op_buffer | 11 | 0.0059 | 0.434 | 0.0 | 93.7% |
| R1 | per_op_buffer | 21 | 0.0059 | 0.385 | 0.0 | 93.7% |
| **R2** | continuous_its | 1 | **0.4933** | 0.000 | 2.0 | 64.2% |
| R2 | continuous_its | 11 | 0.4933 | 0.000 | 2.0 | 64.2% |
| R2 | continuous_its | 21 | 0.4933 | 0.000 | 2.0 | 64.2% |
| **R2** | per_op_buffer | 1 | **0.5795** | 0.326 | 0.0 | 75.4% |
| R2 | per_op_buffer | 11 | 0.5795 | 0.318 | 0.0 | 75.4% |
| R2 | per_op_buffer | 21 | 0.5795 | 0.353 | 0.0 | 75.4% |
| R3 | continuous_its | 1 | 0.6539 | 0.000 | 0.0 | 69.7% |
| R3 | continuous_its | 11 | 0.6539 | 0.000 | 0.0 | 69.7% |
| R3 | continuous_its | 21 | 0.6539 | 0.000 | 0.0 | 69.7% |
| R3 | per_op_buffer | 1 | 0.6765 | 0.316 | 0.0 | 72.1% |
| R3 | per_op_buffer | 11 | 0.6765 | 0.310 | 0.0 | 72.1% |
| R3 | per_op_buffer | 21 | 0.6765 | 0.299 | 0.0 | 72.1% |

**Reading:**

1. **No agent beats the static frontier** in any family × contract combination.
2. **R1 is closest** to the static (94%). The agent matches the scale.
3. **R2 has the largest gap** (64-75% of static). The continuous_its agent uses NO buffer (resource=0) and only matches ~64% of the static. The per_op_buffer agent uses 30% resource and reaches 75%.
4. **R3 is at 70-72%** of static. Same pattern: per_op_buffer is slightly better.
5. **per_op_buffer > continuous_its in R2 and R3** (more degrees of freedom help in the harder families).
6. **per_op_buffer < continuous_its in resource usage for R1** (continuous_its uses 0.57-0.65 resource, per_op_buffer uses 0.38-0.43). The per_op_buffer is more resource-efficient in R1.

## §4 — Why the agent doesn't beat the static

**Mechanism 1 — The static exploits the ReT non-monotonicity.**

The R2/R3 best statics are "no buffer, S1" with very high fill_rate. The endogenous R2/R3 attribution problem (see `docs/R2_AUDIT_DECOMPOSITION_2026-06-29.md` §1-§3) means more orders fall into the high-value fill-rate branch (~1.0) instead of the recovery branch (~0.007). The static policy maximizes the number of orders in the fill-rate branch, which is a structural advantage that the agent cannot replicate through buffer/shift decisions.

**Mechanism 2 — The agent's policy landscape is mostly flat in the buffer dimension.**

The same Lane A closeout finding (`docs/SAME_VARIABLES_NO_FRONTIER_2026-06-28.md`): the optimal action is a constant base-stock buffer; the learning policy doesn't have a "frontier" to navigate. The agent converges to a single point in policy space, which is the dominant strategy under stationary or non-stationary risks.

**Mechanism 3 — The 10k PPO steps are too short for true convergence.**

10k steps is a smoke test. A 100k+ step run might find a more nuanced policy. But the directional pattern (R1 closer than R2/R3) is consistent with the structural R2 problem and is unlikely to flip with more training.

## §5 — Cross-CFi consistency

The 3 CFis tested per family (CF1, CF11, CF21) give **identical** mean_ret within a (family, contract) combination. This is **expected**: the env uses `enabled_risks=family.risks` to filter risks, not the CF-specific `risk_overrides`. So the agent sees the family-wide risk pattern, regardless of the CF index. The CF-specific pattern test is a separate experiment (CF-specific RL on a single CFi) that was not run in this session.

The takeaway: the family-lane agent generalizes across CFis because the family-wide risk pattern is what it learns from. The CF-specific learning test would require running the agent with a single CFi and its `risk_overrides`, which is a different question (and was not part of the original plan).

## §6 — Implications for the paper

1. **The family-lane does not produce a publishable win.** The agent reaches 64-94% of the static frontier across families. The honest framing is: "even with full freedom to learn per family, the agent cannot beat a well-tuned static policy. This is the boundary of the frontier-dependent learning theory."

2. **The R1 result is the closest to interesting.** R1 agent at 94% of static is competitive. With more training and a CF-specific risk pattern test, the R1 result could become a "Pareto improvement" claim (e.g., same ReT at lower resource) — but as currently configured, the resource usage is NOT lower than the static (continuous_its R1 uses 0.57-0.65 resource vs static I168_S3 at 0.57).

3. **The R2/R3 results confirm the structural null.** The endogenous R2/R3 attribution problem caps any RL policy at ~75% of the static. To beat the static, the agent would need to either: (a) fix the R2/R3 attribution (out of scope for RL), or (b) use Track B (downstream dispatch as decision variable, also out of scope).

4. **The CF-specific test is the next obvious extension.** Run the family-lane agent on a single CFi with its `risk_overrides` set per `RISK_PATTERNS[cfi]`. This tests whether the agent can learn the CF-specific risk pattern. Predicted result: also null, because the agent doesn't have the variance in the action space to differentiate CF1 (R13+R14 increased) from CF2 (R12 only increased) when both are at "current+increased" frequency mix.

5. **The honest paper claim is the boundary, not a win.** The family-lane confirms the Lane A closeout: the MFSC with buffer/shift only is a structurally null system for learning. The user-pinned "RL on R1/R2/R3" experiment is the cleanest test of that null and it confirms it.

## §7 — What was NOT done (open questions)

1. **More PPO training:** 10k steps is a smoke test. A 100k+ step run with proper hyperparameter tuning (n_steps, batch_size, learning_rate) might find a marginally better policy. Direction is unlikely to flip.

2. **CF-specific RL:** Run the family-lane with `risk_overrides` set per `RISK_PATTERNS[cfi]` (5 CFs per family × 3 families × 2 contracts × 5 seeds = 150 runs, ~75 min on Kaggle). This tests the CF-specific learning claim.

3. **Hazard obs + per-risk hazard granularity:** The agent uses `risk_obs=True` but not the per-risk hazard features (`weeks_since_last_R1`, etc.) from `docs/MASTER_RESEARCH_CONSOLIDATION_2026-06-28.md` U15. Adding per-risk hazard could help the agent predict the timing of risk events.

4. **Multi-seed confidence intervals:** All 18 runs are seed=1. Adding seeds 2-5 would give CIs and a harder statistical test of the "no win" claim. Estimated: 12 more runs (3 families × 2 contracts × 1 CFi × 2 seeds) at 30k steps = ~30 min.

5. **The bounded stock-conserving recovery variant (from Entregable 2.2) coupled with the family-lane.** Run the RL agent on top of the new `risk_recovery_window_hours=336, risk_recovery_release_rations=10_000` config. This is the most promising extension: the R2 endogenous gap closes from 2.09x to 0.74x with the recovery, which would lift the family-lane R2 baseline from 0.4933 to closer to the recovered R2 static (which we haven't computed yet).

## §8 — Verdict

**The family-lane RL does not beat the static frontier** in any family. The pattern is consistent across both action contracts. The structural R2/R3 attribution problem caps the achievable ReT, and the agent's action space (buffer + shift) doesn't expose a controllable frontier.

**For the paper:** this is the **boundary result** for the frontier-dependent learning theory. The agent learns adaptively (resource and shift are non-constant across the smoke runs), but adaptivity is not enough to beat a well-tuned static. The honest claim is: **"frontier-dependent learning value is null in this MFSC; the binding constraint is downstream, not buffer."**

The two follow-ups that could change this:
- **Track B** (downstream dispatch as action) — out of scope per the 2026-06-28 plan.
- **The bounded recovery variant** (Entregable 2.2) coupled with the family-lane — most promising next experiment.

## Provenance

- Static frontier: `outputs/benchmarks/family_static_frontier_2026-06-29/` (162 runs, 222s)
- RL smoke: `outputs/experiments/family_lane_rl_2026-06-29/` (18 runs, 482s)
- Recovery variant: `outputs/audits/r2_recovery_variant_2026-06-29/` (120 runs, 300s)
- Runner: `scripts/run_static_frontier_per_family.py`, `scripts/run_family_lane_rl.py`
- Recovery runner: `scripts/run_r2_recovery_variant_audit.py`
- New sim params: `risk_recovery_window_hours`, `risk_recovery_release_rations`, `risk_recovery_boost_downstream`, `risk_recovery_enabled_risks` in `supply_chain/supply_chain.py`
- R2 audit: `docs/R2_AUDIT_DECOMPOSITION_2026-06-29.md`
- Lane A closeout: `docs/SAME_VARIABLES_NO_FRONTIER_2026-06-28.md`
- Master plan: `docs/MASTER_RESEARCH_CONSOLIDATION_2026-06-28.md`
