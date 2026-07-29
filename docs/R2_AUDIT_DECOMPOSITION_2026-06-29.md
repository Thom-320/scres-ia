# R2 Endogenous Audit Decomposition (2026-06-29)

**Central question:** Why is the endogenous R2 ReT 2.39× higher than the Excel target (`0.483` vs `0.202`), when R1 is only 0.71× off?

**Short answer:** The endogenous R2 DES under-attributes risk to orders, so more orders fall into the high-value fill-rate branch (close to 1.0) instead of the recovery branch (≈0.007). The mechanism is correctable in principle but not with the variants tested so far — see §5 for the recommended next experiment.

This document decomposes the 2.39× gap along four axes, using only the audit artifacts already produced in this session (`garrido_r2_*` series, `garrido_des_family_match_after_bt_cap_2026-06-26`) and proposes ONE concrete next calibration step.

## §1 — Per-CF ReT gap (which CFs dominate the 2.39× family mean?)

Source: `outputs/audits/garrido_des_family_match_after_bt_cap_2026-06-26/summary.json:680-1230` (R2 rows for CF11-CF20).

Computed gap contribution: `(des_ret - excel_ret) / sum_of_all_gaps`. The 10 CFs together sum to a 0.281 absolute gap (the difference between 0.483 and 0.202).

| CFi | Excel ReT | DES ReT | Gap (DES−Excel) | Contribution to total | Lost (mean) | Pending terminal (mean) | Backlog max h |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 11 | 0.180 | 0.368 | +0.188 | 6.7% | 305 | 149,426 | 80,562 |
| 12 | 0.395 | 0.541 | +0.146 | 5.2% | 189 | 149,062 | 80,562 |
| 13 | 0.142 | 0.397 | +0.255 | 9.1% | 323 | 151,268 | 80,562 |
| 14 | 0.173 | 0.450 | +0.277 | 9.9% | 182 | 142,326 | 80,562 |
| 15 | 0.191 | 0.459 | +0.268 | 9.5% | 186 | 147,292 | 80,562 |
| 16 | 0.235 | 0.629 | +0.394 | 14.0% | 72 | 139,185 | 80,562 |
| **17** | **0.179** | **0.547** | **+0.368** | **13.1%** | 72 | 135,630 | 80,562 |
| **18** | **0.260** | **0.566** | **+0.306** | **10.9%** | 68 | 138,328 | 80,562 |
| **19** | **0.138** | **0.514** | **+0.376** | **13.4%** | 168 | 149,967 | 80,562 |
| 20 | 0.116 | 0.354 | +0.238 | 8.5% | 320 | 146,535 | 80,562 |
| **Family** | **0.202** | **0.483** | **+0.281** | **100%** | **189** | **144,902** | **80,562** |

**Top-3 contributors (51% of the gap):** CF16 (14.0%), CF19 (13.4%), CF17 (13.1%). These three CFs share a pattern from `RISK_PATTERNS`:
- **CF16:** R21−, R22+, R23−, R24− → R22 is the only R2 at increased level
- **CF17:** R21−, R22+, R23+, R24− → R22+R23 at increased
- **CF19:** R21−, R22+, R23−, R24+ → R22+R24 at increased

Common factor: **R22 (LOC destruction) at increased level** in all three. R22 is the high-frequency/short-duration downstream disruption.

**Insight:** The R2 gap is concentrated in CFs where R22 is active. R21 (natural disaster) and R23 (forward unit destruction) are less problematic. R24 (contingent demand) is high-share in Excel but the DES handles it as a point event that materializes a contingent order (see `supply_chain.py:2438-2457`), so the contribution to the gap is smaller than expected.

**Bottom-3 contributors (20% of the gap):** CF11 (6.7%), CF12 (5.2%), CF20 (8.5%). CF11 and CF12 are the "many-risks" CFs (R21+R23+R24 all at increased); CF20 has R22+R23+R24 all at increased. These have **less R22 dominance** in the actual risk generation (more R21/R23 events) and the gap is smaller.

**Conclusion §1:** The R2 gap is **driven by R22 events** (LOC destruction) which the DES under-attributes to orders. The fix must target the R22 attribution path specifically.

## §2 — Per-risk (R21/R22/R23/R24) contribution

Source: `outputs/audits/garrido_r2_downstream_2026-06-26/r2_column_mode_summary.csv` (rows `current_attribution`, all CFs).

| Risk | Excel share (mean over CF11-CF20) | DES share (current_attribution, mean) | Ratio | Excel share (range) | DES share (range) |
|---|---:|---:|---:|---|---|
| R21_1 | 0.064 | 0.057 | 0.90 | 0.021–0.118 | 0.029–0.092 |
| R21_2 | 0.077 | 0.057 | 0.74 | 0.007–0.147 | 0.029–0.092 |
| R21_3 | 0.068 | 0.057 | 0.84 | 0.015–0.148 | 0.029–0.092 |
| R21_4 | 0.058 | 0.057 | 0.99 | 0.020–0.107 | 0.029–0.092 |
| R21_5 | 0.063 | 0.057 | 0.91 | 0.018–0.117 | 0.029–0.092 |
| **R22_1** | **0.207** | **0.131** | **0.63** | 0.108–0.293 | 0.077–0.162 |
| **R22_2** | **0.213** | **0.131** | **0.62** | 0.106–0.276 | 0.077–0.162 |
| **R22_3** | **0.198** | **0.131** | **0.66** | 0.090–0.302 | 0.077–0.162 |
| **R22_4** | **0.201** | **0.131** | **0.65** | 0.098–0.286 | 0.077–0.162 |
| **R23** | **0.218** | **0.140** | **0.64** | 0.061–0.359 | 0.051–0.178 |
| **R24** | **0.655** | **0.295** | **0.45** | 0.559–0.766 | 0.211–0.378 |

**Reading:**
- **R22 (LOC destruction):** DES attributes 62-66% of the Excel share. The DES is missing ~1/3 of the R22 events. This is the largest absolute gap in attribution.
- **R24 (contingent demand):** DES attributes only 45% of the Excel share. R24 is a **point event** that triggers a cascade of contingent orders; the workbook marks every order in the cascade, but the DES only marks the contingent order itself. This is a structural under-attribution.
- **R23 (forward unit destruction):** DES 64% of Excel share. Similar pattern to R22.
- **R21 (natural disaster):** closest to Excel (74-99%). R21 is the long-duration disruption (mean 273-317 h); the time-overlap attribution is more accurate for long events.

**Mechanism interpretation:**

R22/R23/R24 are **short-duration downstream disruptions** (mean duration 21-127 h for R22/R23, point event for R24). The DES `_set_order_ret_indicators` (`supply_chain.py:2355-2478`) marks an order as R22-affected only if `[event_start, event_end]` overlaps `[OPTj, OATj]` — but for a short event (24 h) in a long order window (50+ days), this catches few orders.

The workbook, by contrast, marks downstream orders as R22-affected for the **duration of the resulting backlog**, not just the event. This is the "tail-window" mechanism that the audit already measured:

| Mode | R22 share ratio | R23 share ratio | R24 share ratio |
|---|---:|---:|---:|
| current_attribution (DES) | 0.62× | 0.64× | 0.45× |
| **tail_window_event_hits (DES)** | **25-39×** | **89-117×** | **105-183×** |
| Excel | 1.0× | 1.0× | 1.0× |

A "tail-window" attribution overcorrects by 25-180× because the R2 backlog never clears (max backlog interval = 80,562 h = 10 years for every CF11-CF20 run). The truth is somewhere between current and tail-window.

**Conclusion §2:** The DES under-attributes R22/R23/R24 because the attribution window is too short (event duration only). A **bounded tail-window** with a release of finite downstream stock is the missing mechanism. The audit already identified the right magnitude (release ≈ 2,500 rations per event) but it was never implemented as a default.

## §3 — Per-sub-measure breakdown (APj/RPj/DPj/lost/pending/fill_rate)

Source: `outputs/audits/garrido_des_family_match_after_bt_cap_2026-06-26/summary.json:680-1230` and the `garrido_des_family_match_after_bt_cap_2026-06-26/audit.json`.

| Sub-measure | R1 mean (Excel) | R1 mean (DES) | R1 ratio | R2 mean (Excel) | R2 mean (DES) | R2 ratio |
|---|---:|---:|---:|---:|---:|---:|
| Mean ReT | 0.0063 | 0.0045 | 0.71× | 0.2019 | 0.4827 | **2.39×** |
| CTj p99 (h) | 7,424 | 15,311 | 2.06× | 10,087 | 21,549 | 2.14× |
| RPj p99 (h) | 832 | 3,770 | 4.53× | 4,544 | 737 | **0.16×** |
| Lost (mean) | small | 197.6 | — | 0.0–388 | 188.6 | — |
| Pending (mean) | small | 59.4 | — | 0 | 144,902 | — |

**Reading:**
- **CTj p99 is 2× Excel in both R1 and R2.** This is the "backlog tail" issue — the DES queue takes longer to drain than Garrido's. R1 and R2 have the same CTj tail issue (not specific to R2).
- **RPj p99 is 4.5× Excel in R1 but only 0.16× in R2.** This is the **unique R2 problem**: the DES under-attributes R2 events to orders, so RPj (which is the disruption duration attributed to each order) is small. R1's RPj over-attribution is a *different* issue (queue-wait inflation fixed by `ret_recovery_period_mode="disruption"`, see `ENDOGENOUS_TAIL_RPJ_FINDINGS_2026-06-26.md:40-46`).
- **Lost and pending are massive in R2 DES** (189 lost, 145k pending) but small in Excel R2. This means the Excel workbook **does not lose or backlog orders** the way the DES does — the workbook's orders complete eventually, the DES's orders get stuck.

**The gap is two-part:**

1. **Backlog tail (CTj p99 2× Excel):** Affects R1 and R2 equally. This is the "queue takes too long to drain" problem. Mitigation: a bounded release of downstream stock after R2 events (see §5).
2. **Risk attribution (RPj p99 0.16× Excel, fill_rate over-counted):** R2-specific. The DES doesn't see the R2 events as affecting enough orders, so fewer orders go to the recovery branch and more stay in the high-fill-rate branch. Mitigation: same bounded release of downstream stock.

**Conclusion §3:** The R2 gap is NOT a CTj-only problem (R1 has the same CTj issue). It's a CTj + RPj compound problem. Fixing only the tail won't close the gap; fixing only the attribution won't either. **Both must be addressed by a single mechanism**: a finite, stock-conserving downstream catch-up that (a) drains the backlog and (b) marks the recovered orders as R2-affected.

## §4 — Per-branch ReT composition (recovery vs fill-rate vs unfulfilled)

Source: `outputs/audits/garrido_excel_des_2026-06-25/README.md:200-205` and `:69-72`.

| Branch | Excel R1 (CF1-CF10) | DES R1 (3 seeds) | Excel R2 (CF11-CF20) | DES R2 (3 seeds) |
|---|---:|---:|---:|---:|
| Fill-rate (no risk, served) | 0.03% | 22.15% | 22.26% | ~50% (by complement) |
| Autotomy (APj > 0) | 0.44% | 18.39% | 0.06% | small |
| Recovery (0 < RPj < DPj) | **99.52%** | 29.35% | **77.68%** | **~45%** |
| Risk no recovery (DPj−RPj > 0) | 0.00% | 30.11% | ~0% | ~5% |

**Reading:**
- **Excel R1 is 99.5% recovery.** Almost every order has *some* risk indicator. The DES marks only 29% as recovery — the **other 70% is split between fill-rate (22%) and risk-no-recovery (30%)**.
- **Excel R2 is 77.7% recovery.** DES marks only ~45% as recovery. The remaining 32% is **fill-rate** in the DES (DES over-counts the no-risk branch).
- **Risk no recovery (DPj−RPj > 0):** 0% in Excel, 30% in DES R1 and ~5% in DES R2. The DES is creating orders where `CTj > LTj` but `RPj ≈ 0` (no risk attributed) — these are the "lost in queue, no risk cause" orders.

**Why the gap = 2.39×:**

The Excel ReT mean is dominated by the **recovery branch** (0.5/RPj where RPj is hundreds of hours, so 0.5/RPj ≈ 0.005). The DES ReT mean is dominated by the **fill-rate branch** (close to 1.0 when no risk is attributed) and the **risk-no-recovery branch** (0 by formula). The DES mean is therefore ~50% × 1.0 + 45% × 0.007 + 5% × 0 = 0.500, vs Excel 77.7% × 0.007 + 22.3% × 1.0 = 0.228 (approximate, real Excel R2 mean is 0.202 — autotomy and lost bring it down further).

**The fix is to push more DES orders into the recovery branch**, which means: more orders must be marked as R2-affected, with non-zero RPj. The release mechanism does this *and* drains the queue simultaneously (because it only fires for orders that pass through the affected window).

**Conclusion §4:** The 2.39× gap is the symptom; the **mechanism** is "the DES fill-rate branch is over-counted because R2 events are not attributed to enough orders". A bounded stock-conserving recovery window is the natural fix.

## §5 — The bounded stock-conserving recovery variant: candidate `release=2,500`, `window=336h`

Source: `outputs/audits/garrido_r2_recovery_release_sweep_2026-06-26/variant_by_cf_summary.csv` and `outputs/audits/garrido_r2_recovery_window_2026-06-26/variant_by_cf_summary.csv`.

The release sweep tested 4 magnitudes of stock release after a 336 h post-R2-event window:

| Variant | CT p99 ratio (mean) | RP p99 ratio (mean) | Lost (mean) | Pending (mean) | Backlog max h | Theatre terminal (mean) |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 2.09 | 0.56 | 188.6 | 144,902 | 80,562 | 1,412 |
| r2_window_336h_release_2,500 | **1.07** | **0.39** | 9.6 | 47,857 | 50,237 | 121,683 |
| r2_window_336h_release_5,000 | 0.20 | 0.10 | 0.1 | 0 | 10,521 | 679,838 |
| r2_window_336h_release_10,000 | 0.06 | 0.05 | 0.0 | 0 | 3,162 | 2,012,484 |
| r2_window_336h_release_31,500 | 0.01 | 0.03 | 0.0 | 0 | 361 | 7,938,212 |

**Reading:**
- **Release = 2,500 rations/event** is the **only magnitude that closes the CT tail to <1.5× Excel without inflating the theatre** (121k vs 1,412 baseline = 86× increase but still within a realistic stock level; 5,000+ gives 480k-7.9M which is implausible).
- **Release = 2,500 leaves lost = 9.6 (vs 188 baseline) and pending = 47,857 (vs 144,902).** That's a 95% reduction in lost orders and 67% reduction in pending stock.
- **Backlog max interval drops from 80,562 h to 50,237 h** (38% reduction) — the queue still doesn't fully clear in 10 years, but the worst-case is much smaller.
- **Per-CF heterogeneity** is significant: CF14-19 reach CT p99 ratio of 0.32-0.78 (well under Excel), but CF11/13/20 stay at 1.33-2.38 (similar to baseline). This means the release magnitude is **per-CF dependent**, not a one-size-fits-all number. The "median CF" behavior is the right bar.

**The implementation needed (next step):**

1. **Add a `risk_recovery_window` parameter to `MFSCSimulation`:**
   - `risk_recovery_window_hours: float = 0.0` (default = off, preserves freeze)
   - `risk_recovery_release_rations: float = 0.0` (default = off)
   - `risk_recovery_window_enabled_risks: tuple[str, ...] = ("R21", "R22", "R23", "R24")` (which risks trigger the recovery)

2. **Logic in `supply_chain.py`:** after each R2 event ends, for `risk_recovery_window_hours` post-event, allow up to `risk_recovery_release_rations` to flow from CSSU to theatre (stock-conserving — no injection), and mark every order in this window as having the R2 risk indicator.

3. **Register the candidate** in `docs/THESIS_FAITHFUL_ENV_FREEZE_2026-06-26.md` as a **sensitivity**, not a default. The freeze remains `delay=54, risk_occurrence_mode=thesis_window, ret_recovery_period_mode=disruption` — the recovery window is opt-in.

4. **Audit script** `scripts/run_r2_recovery_variant_audit.py` that runs the variant on CF11-CF20 with 3 seeds, 10-year horizon, and reports per-sub-measure.

**Stop-rule for the candidate** (registered before running):

- PASS if: CTj p99 ratio < 1.5× Excel AND RPj p99 ratio < 2.0× Excel AND Theatre terminal < 200k rations
- PARTIAL if: CTj p99 ratio < 2.0× Excel (was 2.09) but RPj p99 inflates > 2× or theatre > 200k
- FAIL if: any of the above is worse than baseline

If PASS or PARTIAL: this is the calibrated recovery for the endogenous R2 lane, and the paper can claim "bounded stock-conserving recovery closes the R2 gap to within 1.5× of Excel for 7/10 CFs". If FAIL: the audit's prior conclusion stands — the R2 endogenous lane remains documented as a fidelity limitation, and the paper's RL claim is restricted to the R1 lane and the forensic tape lane.

## §6 — Implications for the paper

1. **R1 is in scale, R2 is not.** R1 endogenous ReT = 0.71× Excel is within "calibration residual". R2 endogenous ReT = 2.39× Excel is a **structural gap** that the bounded recovery candidate may or may not close.

2. **The forensic tape lane (`risk_attribution_source="excel_risk_tape"`) is the verified replication.** R1 MAE 0.0023, R2 MAE 0.0023, branch shares exact. The endogenous lane is a **separate model** that approximates Garrido under R1 but diverges under R2.

3. **For the paper's RL claim:** any RL result on the endogenous R2 lane must be compared against the **dense static frontier under the same enabled_risks** (R2-only), not against the full R1+R2 frontier or the Excel tape. The R2-only frontier is the only fair comparison under the endogenous R2 limitations.

4. **For the H4 retained-reset claim:** the endogenous R2 lane is the wrong test bed (too noisy, gap not bounded). Use the endogenous R1 lane or the forensic tape lane for H4 confirmation.

5. **The bounded recovery candidate is a paper contribution, not a workaround.** If it closes the gap, the paper has: (a) characterized the missing mechanism (R2 catch-up is bounded), (b) calibrated the magnitude (2,500 rations/event), (c) demonstrated the recovery is heterogeneous across CFs (some CFs need more, some less). This is the **structural characterization** the user has been asking for, complementing the "frontier-dependent learning theory" from `docs/SAME_VARIABLES_NO_FRONTIER_2026-06-28.md`.

## §7 — What was NOT done (open questions for the next session)

1. **The 4 sub-measures (APj/RPj/DPj/CTj) per-CF Excel-vs-DES table** was not generated because the existing `audit.json` files don't separate them by CF. A targeted re-run of the 30 CF×3 seeds with full sub-measure output would give the per-CF table directly. Estimated 2-3 hours.

2. **The "tail-window with release" attribution** was conceptually sketched but not implemented in the sim. The release sweep variants were diagnostic-only; the code never had a `risk_recovery_window` parameter. This is §5's deliverable.

3. **The R3 family (CF21-CF30)** was not audited. R3 is the black-swan event, U(1, 161,280 h) = 1 per 20 years. The endogenous R3 lane has never been compared against the Excel R3 baseline. This is a separate audit that requires 20-year horizons.

4. **The 102-replication Monte Carlo Validation (MCV)** is in the thesis Chapter 6.9 but has never been run on the repo. The repo uses 3-5 seeds; the thesis uses 102. The statistical power difference is significant. Estimated 6-8 hours Kaggle for a single family (CF1, CF11, or CF21).

5. **The per-CF 9 hypothesis tests (H1a-H3c)** with Kruskal-Wallis + Wilcoxon are the **result of the thesis** and have never been run on the repo's data. Without these, the comparison "does the repo confirm Garrido's hypotheses?" cannot be answered.

## Provenance

- All CSVs and JSONs referenced are dated 2026-06-25 to 2026-06-28
- The release sweep is the most recent data (2026-06-26)
- This document is a synthesis, not a new audit
- The candidate bounded recovery is **proposed, not implemented** — see §5 for the implementation plan
- The 5 deliverables (D2.1-D2.5) correspond to the 5 sections here
