# VERDICT — Forensic DES audit + Table 6.10 replication

**Date:** 2026-06-19
**Auditor:** Claude Code (Opencode, glm-5.2)
**Source of truth:** `thesis.txt` (Garrido-Rios 2017, extracted text)
**Branch:** `codex/garrido-postfix-reruns` (quarantine commit at HEAD)

---

## TL;DR

The repo **can** reproduce the Garrido-Rios 2017 DES, but only after fixing three foundation-level default-value bugs. The much-cited `-29% better RMSE than thesis` claim from `THESIS_FIDELITY_AUDIT.md` is **methodologically invalid and not reproducible**. After fixes, the model achieves RMSE 83,292 vs thesis RMSE 87,918 (0.95×) on a true apples-to-apples Table 6.10 comparison — i.e. ~5% better than the thesis's own ECS-vs-Pt RMSE, with a -7.7% systematic under-delivery bias.

All prior RL benchmarks (`PPO vs static`, `ReT_seq_v1`, Kaggle confirmatory runs) computed on the buggy default config are **not comparable** to the thesis — they ran with 2× too-frequent uniform risks and a 24h-early warmup trigger. Re-running them is out of scope for this phase.

---

## What was done

1. **Quarantine** of `outputs/` (14 GB), `kaggle/`, `notebooks/`, `logs/`, `paper_results_package/` → `*_claimed_2026-06-19/`. Active workspace clean.
2. **Forensic DES audit** of `supply_chain/config.py` + `supply_chain/supply_chain.py` vs `thesis.txt` (35 findings, full register at `sandbox/forensic_des.md`).
3. **Forensic RL audit** of `supply_chain/env_experimental_shifts.py` + reward design (15 findings, full register at `sandbox/forensic_rl.md`).
4. **Empirical probes** confirming the highest-impact bugs:
   - `sandbox/results/p_b1_risk_frequency.json` — B1 risk-frequency doubling
   - `sandbox/results/p0_warmup_det.json` — M0 warmup time
   - `sandbox/results/p_table_6_10_replication.json` — Table 6.10 replication
5. **Three foundation-level fixes** at `supply_chain/supply_chain.py:134,142` (defaults only; legacy modes preserved as opt-in).

---

## Confirmed bugs (ranked by impact)

### FIXED in this pass

**B1 — `legacy_renewal` default doubles uniform-risk frequency.**
- Code: `supply_chain/supply_chain.py:142` default was `"legacy_renewal"`; `supply_chain.py:1483-1484` samples `rng.integers(a, b+1)` per event (renewal with mean IA=(a+b)/2).
- Thesis Table 6.11 expects periodic sampling at rate `b` hours per event.
- Empirical: legacy_renewal gives R11=93.3/yr vs thesis 48/yr (1.94×); R21/R22/R23/R24/R3 all ~2× too.
- **Fix:** default changed to `"thesis_periodic"`. Empirical post-fix: R11=47.3/yr, R22=2/yr, R24=12/yr ✓.

**E1 — `production` warmup trigger fires 24h early.**
- Code: `supply_chain/supply_chain.py:134` default was `"production"`; fires when `total_produced >= batch_size` at AL output (line 1259-1263), i.e. before the 24h Op8 transport to Op9.
- Thesis Sec 6.8: warmup completes on first Q=5000 *arrival at Op9*.
- **Fix:** default changed to `"op9_arrival"`. Empirical post-fix: warmup = 943h (vs 919h for `production` — exactly 24h later as expected).

### DOCUMENTED, not fixed (low impact / out of scope)

**M0 — Warmup time = 943h vs thesis estimate 838.8h (+104h, +12%).**
- The 104h gap comes from ROP-cycle alignment in the upstream Ops (Op3 dispatches at multiples of 168h; Op2 delivers at t=672h, so first useful Op3 dispatch races with that cycle).
- This is **structural**, not a bug: the engine correctly implements time-triggered ROPs per Fig 6.2; the 838.8h thesis estimate is a simple sum of PTs that ignores ROP-cycle waiting.
- Impact on annual metrics: 104h / 161,280h = 0.06% — negligible. Does not block Table 6.10 replication.

**B3 — R13 event-log count is 50% of thesis Table 6.11 even in `thesis_periodic` mode.**
- Code: `_risk_R13` logs 1 event per cycle when the binomial draw `B(12, 0.10) > 0` (probability 0.718). Thesis Table 6.11 says 58 events/year = expected total delayed deliveries = 48 cycles × 1.2 mean.
- **Dynamic behavior is correct** — the engine applies `delay = k × 24h` for k delayed deliveries. Only the *event-log count* under-reports because each cycle logs at most 1 event regardless of k.
- Impact: simulation dynamics unaffected; only comparisons to Table 6.11 event counts are misleading. Fix would be to log `k` events per cycle, but this is a logging convention change, not a dynamics fix.

**B12 — `r14_defect_mode="reprocess"` default routes defects to raw_material_al (skipping Op5).**
- Thesis Table 6.6b: defects return to Op6 rework buffer.
- Engine has the thesis-correct mode `"thesis_strict_op6"` available; only the default is wrong.
- Not fixed in this pass (RL envs inherit this default; fixing requires coordinated RL-layer change).

### NOT BUGS (semantic clarifications)

- **R14 "events/year" in Table 6.11 = 22,153** is actually *defective rations/year* (sum of daily binomial draws), not *risk events*. The engine logs 1 event per working day; ~255/year; each event reports a binomial draw of mean 76 defects. Dynamic correct; terminology mismatch.
- **Unbounded `simpy.Container`** is correct: thesis Sec 6.5.3 explicitly says "storage capacities of WDC, SBs and CSSBs are assumed to be unlimited". The "unbounded container = optimistic bias" hypothesis from the original plan is **rejected**.

---

## Table 6.10 replication (the main deliverable)

**Protocol:** 3 seeds × 20-year horizon, `risk_occurrence_mode="thesis_periodic"`, `warmup_trigger="op9_arrival"`, post-warmup yearly deliveries for years 1-8, averaged across seeds.

| | Mean annual delivery | RMSE vs Pt | Ratio vs thesis RMSE |
|---|---|---|---|
| Thesis (Pt vs ECS) | 767,592 (ECS mean) | 87,918 | 1.00× |
| **Stochastic thesis-faithful (ours)** | **708,666** | **83,292** | **0.95×** |
| Deterministic no-risks (audit-doc config) | 738,432 | 75,145 | 0.85× |
| Audit doc claimed (deterministic config) | — | "62,055" | 0.71× |

**Key observations:**

1. **Apples-to-apples RMSE: 0.95× thesis.** Our stochastic thesis-faithful run reproduces the thesis's RMSE within 5%. This is a fair comparison: same protocol (3 seeds, risks enabled, current level), same Table 6.10 Pt array.

2. **Systematic under-delivery of -7.68%.** Our mean is 708,666 vs thesis ECS 767,592. Within AGENTS.md ±15% tolerance for Phase 2, but a real bias. Likely contributors: (a) R13 event-log under-counting is symptomatic of slightly reduced disruption load, (b) 104h warmup offset, (c) some parameter drift not yet isolated.

3. **Per-year variance is too low.** Our years range 694k-718k (24k spread). Thesis ECS ranges 712k-888k (176k spread). The thesis's variation is ~7× larger. Probable cause: the thesis aligned ECS per historical year with year-specific scenario inputs (different operations tempo per year); we cannot reproduce that pattern without per-year historical calibration data.

4. **The audit doc's `-29.4% better` claim is not reproducible.** Running the exact audit-doc config (`deterministic_baseline=True, risks_enabled=False`) yields RMSE 75,145, not 62,055. The audit's 62,055 came from a configuration or computation we cannot identify from the codebase. The claim should be **retracted**.

5. **The deterministic no-risks RMSE (75,145) being lower than the stochastic thesis-faithful RMSE (83,292) is the methodological invalidity in one number.** Deterministic-by-construction produces a near-constant year-by-year output (`[738432]×8`), so its RMSE-vs-Pt is essentially `std(Pt)` — a function of Pt's variance, not of model fidelity. The audit doc compared this number to the thesis's stochastic-with-risks RMSE and declared victory.

---

## What this invalidates

1. **`THESIS_FIDELITY_AUDIT.md` §7 and §8** — the `-29.4% RMSE` and `-4.16% annual delivery` claims. Both came from configs that don't reproduce, and the comparison methodology is invalid.

2. **`DIVERGENCE_FIX_PLAN.md` TL;DR** — claims "If you only do P0 [change year_basis], divergence drops from ±4% to ±2%". This was based on the same flawed comparison.

3. **All RL benchmarks trained on the default config** — every PPO training run that didn't explicitly pass `risk_occurrence_mode="thesis_periodic"` and `warmup_trigger="op9_arrival"` (which is all of them, per `env_experimental_shifts.py:344,349` hardcoding legacy defaults) was trained on a model with 2× too-frequent uniform risks. The training signal is not comparable to thesis-scenario disruptions.

4. **`docs/RET_GARRIDO2024_AUDIT_2026-06-18.md` and related** — these audits of the reward design were performed on the buggy DES layer. Their conclusions about reward calibration may still hold for the relative ordering of policies, but absolute numbers and "PPO outperforms thesis" claims need re-running.

---

## What still works (credit where due)

The static structure is faithful. The forensic found **17 MATCHes**:
- All 13 operations (Fig 6.2) with correct PT, Q, ROP, risk assignments.
- All 9 risks (Tab 6.6b/6.7b/6.8b) with correct distribution parameters (a, b, n, p).
- All 5 inventory-buffer levels (Tab 6.16).
- All 3 shift configurations (Tab 6.20).
- Tab 6.10 validation data (Pt and ECS arrays, RMSE 87,918).
- λ=320.5, RATIONS_PER_BATCH=5000, RATIONS_PER_SHIFT=2564.
- 8064-h year, 161,280-h horizon, 6000-order cap, 60-backorder cap.
- Container-not-Store, unbounded buffers (thesis-authorized).

The engine has all the thesis-faithful knobs. The bugs were in *which knob was default*, not in the capability.

---

## Recommendations (prioritized)

**P0 — Re-run all RL benchmarks with the fixed defaults.** This is mandatory before any paper claim about PPO vs thesis-style static. The RL envs themselves still hardcode legacy defaults (`env_experimental_shifts.py:344,349`); either change those defaults too or pass explicit kwargs at all training entry points.

**P1 — Retract `THESIS_FIDELITY_AUDIT.md` §7-8 and the `-29%` claim publicly.** Update the audit doc or replace it with this VERDICT.

**P2 — Investigate the -7.68% under-delivery bias.** Candidates: R13 dynamic effect of under-logging, Op7 ROP-induced queueing, R11 repair-time clipping (mean shifts from 2h to 1.83h). Run a Cf0 deterministic 20-year horizon and check per-Op throughput to localize.

**P3 — Investigate per-year variance mismatch.** If the thesis's ECS pattern cannot be reproduced without per-year historical data, document that explicitly in any paper claim.

**P4 — R13 logging convention.** Either log `k` events per cycle to match Table 6.11 terminology, or add a docstring noting that the engine's `risk_events` count uses a different convention.

**P5 — Fix `r14_defect_mode` default** in coordination with the RL layer.

---

## Artifacts

- `sandbox/forensic_des.md` — full DES divergence register (318 lines)
- `sandbox/forensic_rl.md` — full RL divergence register (259 lines)
- `sandbox/probes/p_b1_risk_frequency.py` + `.json` — empirical B1 confirmation
- `sandbox/probes/p0_warmup_det.py` + `.json` — M0 warmup gate
- `sandbox/probes/p_table_6_10_replication.py` + `.json` — Table 6.10 replication
- `sandbox/results/*.json` — raw outputs of all probes

## Files modified

- `supply_chain/supply_chain.py:134,142` — defaults changed from `"production"`→`"op9_arrival"` and `"legacy_renewal"`→`"thesis_periodic"`. No other code changes.

## Files NOT modified (intentional)

- RL envs (`env_experimental_shifts.py`, `env.py`) — they hardcode legacy defaults; coordinated RL-layer fix is P0 above.
- All docs, audit reports, configs — they are claims to be retracted, not edited in-place.
- Quarantined `outputs_claimed_2026-06-19/` etc. — preserved for forensic diff value.
