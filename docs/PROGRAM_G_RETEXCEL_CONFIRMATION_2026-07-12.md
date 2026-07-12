# Program G — ret_excel confirmation REVERSES the G5 win — 2026-07-12

Status: **Under the project's PRIMARY metric `ret_excel`, the G1–G5 observable adaptive win does NOT
hold. Program G joins the boundary results: clairvoyant headroom exists, but no observable policy
converts it — blind alternation is near-optimal.** This is the honest counterpart to the G5 proxy
result, which was disclosed as service-loss, not ret_excel. The port used the project's real
order-level machinery (`compute_order_level_ret_excel_visible_ledger` /
`compute_ret_per_order_excel_formula`) on emitted `OrderRecord`s via a disclosed daily adapter
(`supply_chain.program_g.simulate_orders`) — no new physics in the shared 12-op DES.

## Result on VIRGIN tapes (region = 12 surge-1.50 cells, 150–200 tapes)

| Quantity (ret_excel full ledger, unfulfilled = 0) | Value |
|---|---:|
| Clairvoyant ret_excel oracle (best of 81 seq/tape) | 0.5654 |
| Best static ABAB (frozen, maximizes ret_excel on train) | 0.5475 |
| Cover heuristic (service-loss-trained, the G5 winner) | 0.5115 |
| ret_excel-NATIVE depth-3 tree (fit on ret_excel-oracle actions) | 0.5359 |

- **Clairvoyant ret_excel headroom is REAL: oracle − static = +0.0179, CI95 [0.0136, 0.0222] > 0**
  (comparable to DRA-2b's +0.022). The spatial mechanism genuinely contains headroom under the thesis
  metric too.
- **No observable policy converts it.** cover − static = **−0.036** [−0.047, −0.026]; ret_excel-native
  tree − static = **−0.012** [−0.0155, −0.008]. Both are WORSE than blind alternation.
- **Shed-to-win guardrail (DRA-1 lesson) fires**: the cover policy attends **3.3 fewer orders** (39.7 vs
  42.9 of 48); its higher single-tape visible score was a shedding artifact, gone across 200 tapes.

## Why the headline flipped (service-loss → ret_excel)
The G1–G5 win was measured in **service-loss (unmet ration-mass)**, which rewards CONCENTRATING the
convoy on the starving CSSU. The thesis's **ret_excel is order-level resilience** (fill-rate /
cumulative-backlog per order): it rewards keeping BOTH CSSU order queues moving, so **blind ABAB
alternation attends more orders on time and is near-optimal**, while the concentration policy lets one
queue grow and attends fewer orders. The adaptive-value verdict is **metric-dependent**: a ration-mass
proxy manufactured an apparent adaptive win that the order-level resilience metric does not support.

## Honest conclusion
- Under **ret_excel** (primary), Program G is a **boundary result**, matching DRA-2b/Program E:
  `physical authority → clairvoyant headroom (+0.018) → observable conversion FAILS`. Even a
  ret_excel-native observable tree cannot beat blind alternation.
- Under **service-loss** (proxy), an observable cover rule does convert ~0.68–0.73 — a real but
  metric-specific effect.
- **Methodological contribution**: the "is RL warranted" answer depends on the resilience definition;
  a mass-based proxy can flatter adaptivity. This STRENGTHENS the "when NOT to train" manuscript and
  adds a metric-sensitivity result, rather than delivering a neural/adaptive win.

## For Garrido
Program G's shared-transport spatial mechanism has genuine clairvoyant headroom even in ret_excel, but
**it is not observably or learnably convertible under the thesis resilience metric** — the best fixed
alternating schedule is near-optimal. The apparent adaptive win seen earlier was an artifact of the
service-loss proxy. Net: the project's central finding ("adaptive/RL value is scarce in this MFSC once
you use the right same-contract comparator and the thesis metric") holds across D, DRA-1/2/2b, E, F,
AND now G. Program G does not block Paper 1; it adds a metric-dependence result.

## Limits / provenance
Stylized weekly→daily order adapter (R22 off → no-risk fill-rate branch), disclosed; Option-A convoy;
surge-1.50 region. `results/program_g/retexcel/verdict.json`; oracle/native-tree numbers reproduced in
this session's transcript. Supersedes the G5 headline where they conflict: **the G5 win is proxy-only.**
