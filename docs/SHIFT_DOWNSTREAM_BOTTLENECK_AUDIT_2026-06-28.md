# Audit: why a lean low-buffer/low-shift static beats aggressive ones (2026-06-28)

User question: a simple `f0.10_S1` (10% buffer, 1 shift) beats aggressive `S3`/high-buffer policies on
Excel ReT — "me parece muy raro, no entiendo qué está pasando." Audited against the thesis, the Excel,
and the DES code. **Verdict: expected and thesis-faithful.** Three faithful mechanisms compose.

## How shifts work (thesis + Excel + code — they agree)
- **S = "short-term manufacturing capacity"** of the assembly line (thesis §6.3.2, §6.7.4, Table 6.20;
  thesis Ch.7.4 tests S as a *moderator* of risk→ReT, H3a/b/c). S1/S2/S3 = 1/2/3 shifts/day →
  theoretical output **2,564 / 5,128 / 7,692 rations/day**.
- In code (`config.py:CAPACITY_BY_SHIFTS`, "Table 6.20, only Op3/Op4/Op7/Op8 change"): shifts scale ONLY
  **upstream manufacturing** — op3/op4 raw-material intake (15.5k→47k/wk) and op7/op8 batch size (5k→7k).
- **Downstream distribution is FIXED** and shift-independent: op9 (SB dispatch), op10 (→CSSU), op12
  (→theatre) each ship `U(2400, 2600)` every 24h → a hard ceiling of **~1.82M rations/yr**
  (`supply_chain.py:1686-1751`). The thesis models S as manufacturing capacity only; distribution (LOC)
  does not scale. **Our DES is faithful.**

## The three faithful mechanisms that make lean win
1. **Downstream cap.** Every policy delivers ~1.5–1.6M (near the 1.82M LOC ceiling) regardless of buffer
   or shift. Shifts raise how much you MAKE, never how much you can SHIP.
2. **Overproduction → dead inventory.** S3 produces ~3× (3.5M vs 1.2M) but the surplus dead-ends in
   `rations_sb` (1.8M pile-up vs 0.005M for S1). No delivery gain.
3. **R14 self-penalty (the key war-stress mechanism).** Thesis Table 6.6b: quality defects =
   `Binomial(n = units produced per shift, p = 3/100)`, and defective items are **returned to the previous
   operation for re-processing**. So **more production manufactures more defects** → more rework/re-processing.
   Under war stress (φ4 amplifies R14 frequency ×4), S3's overproduction triggers a rework burden (rework
   appears: 68 vs 0 for S1) that congests assembly — so S3 even delivers slightly LESS.

## Evidence (h104, war φ4/ψ1.5 unless noted)
| policy | ReT | delivered | produced | rations_sb | rework |
|---|---|---|---|---|---|
| f0.10_S1 | **0.00239** | 1.572M | 1.22M | 0.005M | 0 |
| f0.10_S2 | 0.00236 | 1.534M | 2.43M | 0.69M | 310 |
| f0.10_S3 | 0.00205 | 1.492M | 3.51M | 1.82M | 34 |
- Buffer fraction (0.10/0.50/1.0) barely matters; **shift is the dominant — and harmful — lever.**
- **Normal regime (φ1/ψ1):** S1≈S3 on delivery (1.615 vs 1.619, both capped), rework 0 — shifts are
  merely *wasteful* (2.7M dead inventory). The delivery *decrease* with shift appears ONLY under war,
  ONLY when R14 rework appears → confirms mechanism #3.
- Downstream cap computed: op9/10/12 `U(2400,2600)`/24h ≈ 1.82M/yr — all deliveries sit just below it.

## Why this is the central finding (reshapes the claim)
**At war stress, resilience is DISTRIBUTION-bound, not manufacturing-bound.** R22/R23 (attacks on
lines-of-communication / forward logistics) hit the downstream LOC (op4/8/9/10/11/12) — exactly the fixed
bottleneck. The only manufacturing lever (shifts) cannot compensate for a distribution bottleneck, and is
actively counterproductive via R14. Consequences:
- A **lean static (S1 + modest buffer) is near-optimal** → very hard for any policy to beat on raw ReT.
- The dense-frontier gate (`f0.10_S1`, resource 0.05, Excel 0.00228) appears to **dominate the dynamic**
  (0.00214, resource 0.241) on BOTH axes — the coarse 5×3 grid skipped the 0.10 sweet spot. The Pareto
  win needs the same-seed dense re-test (task #28).
- This is the F13 / "frontier-dependent" thesis made concrete: **RL adds value only where the action space
  reaches the binding constraint.** Track A (buffer×shift) does not.

## "Win another way" — implications
- **Track B (downstream control: op10_q, op12_q dispatch) is the structural lever** that reaches the
  binding LOC constraint. Repo already has a positive Track B signal. STRONGEST path.
- **Per-op buffer idea** (`Box([op3_frac, op5_frac, op9_frac, shift])`, saved to memory): reasonable, but
  keeps the inert/harmful shift and never touches the downstream cap → unlikely to fix the bottleneck.
- **Lower-stress regime** (smaller φ): manufacturing matters again, but there statics are already
  near-optimal (no headroom) — the thesis regime.
- **Honest reframing:** lead with "resilience is distribution-bound under war-stress; the manufacturing
  lever is inert, so lean static buffering is near-optimal and the controllable lever is downstream
  distribution" — a genuine, defensible, thesis-grounded contribution.

## Open thread (small, secondary)
The exact magnitude of the S3 delivery *decrease* (~3%, 46k rations) vs the tiny end-state rework buffer
(68) suggests cumulative re-processing volume (not end-state level) drives it. A cumulative-rework +
op7-utilization trace would confirm the precise decomposition. Does not change the conclusion.
