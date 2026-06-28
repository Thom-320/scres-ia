# Action propagation + decision cadence (2026-06-27)

Answers three user questions: (1) do the agent's choices actually reflect in the env, and with what
latency? (2) how long should a step be / how often should the agent decide? (3) would finer (daily)
observations/decisions help, hurt cost, or hurt speed?

## 1. Do actions propagate? YES — verified empirically
`scripts/audit_action_propagation.py` (war φ4/ψ1.5, h52). Excel trace:
`outputs/audits/action_propagation_2026-06-27/action_propagation_trace.xlsx` (step-change + const-frac0
+ const-frac1 sheets — you can SEE each week's commanded frac → target set → container level → outcome).

| check | result |
|---|---|
| **target set** | INSTANT — `sim.inventory_buffer_targets == frac×I1344` the same step |
| **buffer RAISE** (frac 0→1, wk10) | INSTANT — rations container 3,379 → 108,490 same step (target 126,000; the gap is in-week consumption). `_set_targets` calls `container.put(shortfall)` at decision time; lead=0 in this path |
| **buffer LOWER** (frac 1→0, wk30) | target cleared INSTANTLY; physical stock drifts down only as fast as consumption (slow — see below). You can't un-buy stock, which is physically correct |
| **outcome changes** | frac=0 vs frac=1 (same seed): `ret_excel` **0.00197 vs 0.00237** (+20%), total_inventory 4.40M vs 5.86M. The buffer choice DOES move the resilience outcome |
| **shift** (S1 vs S3) | delivered_rations 705,853 vs 715,860 — moves, but weakly |

**Verdict: actions propagate; the agent is in control of the environment.** No artifact there.

## 2. The real constraint is action AUTHORITY, not propagation or cadence
The audit surfaced the decisive fact: **`fill_rate ≈ 0.0064 regardless of buffer (frac 0 vs 1)**.**
Buffer moves upstream inventory + ReT (recovery), but **cannot move delivered fill** — fill is
**downstream-bottlenecked** (op9/op10/op12 throughput caps; the long-standing F11 finding). Shift moves
delivered only ~1%. So Track A's levers (buffer, shift) have **limited authority over the binding
constraint**. This — not propagation lag and not decision cadence — is why every Track-A win evaporates
at multi-seed. The lever the agent holds is real but weak on the metric that's saturated.

## 3. Step length / decision cadence — recommendation
Current: weekly (`step_size_hours=168`); risks are hourly; buffer top-up is instant.

- **Buffer cadence:** because the raise is instant (no lead time in the wrapper path), cadence is NOT
  gated by a buffer lead time here. But since the buffer barely moves fill, faster buffer decisions are
  unlikely to unlock anything on fill. Weekly is fine for the strategic buffer.
- **Finer obs/decisions (daily, 24h) — EMPIRICALLY TESTED (cadence-ceiling, no training):**
  best achievable static over a 1-yr horizon, weekly vs daily:
  - **fill ceiling: NO change** (0.0064 → 0.0064) — bottleneck-bound; finer cadence cannot help fill.
  - **ReT ceiling: +35%** (0.00194 → 0.00261) — finer cadence DOES raise achievable ReT.
  The ReT gain is mostly an env mechanic: at daily cadence the strategic buffer is **replenished every
  24h instead of every 168h**, so backorders clear faster → higher Garrido ReT. So your intuition is
  right *for ReT* (the primary metric), wrong for fill. Costs of true daily DECISIONS: ~7× steps/episode
  (h52→h365), ~7× slower, harder credit assignment.
- **Cheap capture FAILED (tested):** weekly decisions + 24h `replenishment_period` stays at 0.00194 —
  it does NOT recover the daily 0.00261. So the ReT gain comes from the immediate buffer top-up *at each
  DECISION* (`_set_targets`→`_top_up_inventory_buffer` every step), not the periodic replenishment
  process. The benefit cannot be decoupled from decision cadence — to get +35% ReT you must actually
  decide daily (≈7× cost). `replenishment_period` is now a wrapper param but does not substitute.
- **Open question (queued training test):** daily cadence raises the ReT ceiling for BOTH static and
  dynamic, so it raises the bar too. The win question is whether finer decisions add *headroom*
  (dynamic−static gap) — more decision points = more chances to time the buffer to the hazard. A
  daily-cadence training run (1–2 seeds, h≈364, resource-charged) tests this; queued after the
  preventive-Pareto confirms.
- **Two-timescale idea (worth a cheap test):** slow buffer (weekly) + fast shift/capacity (daily), since
  capacity can ramp faster than strategic stock. But shift has weak authority too, so expect a small
  effect.
- **Recommendation:** keep weekly as the primary cadence. Run ONE cheap daily-cadence ablation (1 seed,
  h365) to confirm finer cadence does NOT unlock fill — if the bottleneck binds (expected), the answer is
  "cadence is not the lever; action authority is." The honest path to a bigger win is **Track B
  (downstream control)**, where the agent's actions reach the binding constraint, OR the **resource-aware
  Pareto / efficiency win on Track A** (same resilience at lower charged resource), which does not need to
  move the saturated fill.

## Bottom line for the paper
The simulator faithfully reflects the agent's decisions (instant target, same-step raise, outcome
changes). The reason Track A can't dominate on raw fill is structural action-authority (downstream
bottleneck), not a control or propagation bug — which strengthens, not weakens, the honest story:
the win lives where the lever has authority (Track B) or on the efficiency/Pareto axis (charged
resources), not in faster or finer Track-A buffering.
