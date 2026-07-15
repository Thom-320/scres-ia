# Program O — audited state-rich fit disposition

**Date:** 2026-07-15

**Frozen-gate outcome:** `STOP_RESOURCE_OR_GUARDRAIL_CONFOUND`

**Claim boundary:** full-DES H_PI remains established; H_obs is **not** established;
no learner, Paper 2 claim, or Paper 3 study is authorized.

This document corrects the earlier terminal synthesis against the retrieved, hashed
VPS result. It does not treat an unexecuted placebo as evidence and does not upgrade
the finite development screen into an exhaustion theorem.

## 1. Custody and provenance

- Run: `program-o-state-rich-fit-v1-20260715`.
- Scientific result commit: `041dcef31bda032a3fe9c6eac22739ec45f092dd`.
- Frozen contract SHA-256:
  `97885cf771656cf1829899af9e9dcdd09dc93c2c9aa4e9f683555a3d43c8680d`.
- Parent result SHA-256:
  `2de51be686539ccc801566a2886d7b0ec27ff1e30a8694eb5a0a2f15fc7a9cd0`.
- Retrieved result SHA-256:
  `d67ac97a359a307ca632b6a13493e3ff5940a97e9440a6bc4b7d77c08a147875`.
- Remote checksum-manifest SHA-256:
  `d20220537b5034717dee8b459d8e0c0bafa59f4b0e0c78b3d89f4a52ea8d8ab1`.
- Producer exited 0; stderr was empty; the watcher reached terminal state with an
  empty process group.
- Only burned fit seeds `7420001–7420048` were used. The result records
  `validation_seed_accessed: false`; sealed seeds `7420049–7420096` were not
  present in the retrieved run.

The retrieved result and custody manifests are under
`results/program_o/state_rich_comparator_fit_v1/`.

## 2. What the fit actually showed

The frozen family contained ten finite controllers evaluated in four connected
cells (40 controller-cell rows): base-stock, max-pressure/hysteresis,
min-cost-flow, belief-MPC H3/H4, and approximate belief-DP H3/H4.

- All 40 rows produced state-varying action trajectories.
- Development deltas against the full 65,536-calendar open-loop frontier ranged
  from `+0.02448` to `+0.10752` ReT.
- Metric guardrails passed in **33/40**, not 40/40.
- The stronger state-counterfactual certificate passed in **8/40**: the two
  belief-MPC configurations in all four cells. Other controllers failed at least
  one product-channel counterfactual.
- Reserved capacity and gross production were equal by construction.

These are development signals only. They do not establish H_obs because the
frozen resource gate failed everywhere.

## 3. Load-bearing STOP

`strict_actual_use_pass` is false in **40/40** rows. For every controller-cell
row, the actual-use-matched open-loop set is empty (`eligible_calendar_count: 0`):
no one of the 65,536 calendars used at least as much actual downstream freight as
the controller under the frozen comparison rule.

Relative to the best-ReT open-loop comparator, the controllers used an additional
168–818 downstream vehicle-hours and 3.5–17.04 loaded departures on average,
depending on controller and cell. Consequently:

- no controller passed the pre-placebo rule;
- no cell passed;
- no connected component existed;
- no configuration was selected;
- the sealed validation block remains closed.

This establishes the frozen verdict `STOP_RESOURCE_OR_GUARDRAIL_CONFOUND`. It does
not establish that the observable signal is worthless under every possible freight
accounting convention.

## 4. Important correction: information placebos were not run

The earlier certificate incorrectly interpreted `information_placebos_pass: false`
as evidence that delayed, masked, swapped, and cross-tape state matched the true
state. In the result, `information_placebos` is `null` because the preregistered
pre-placebo gate failed first. Therefore no empirical conclusion about incremental
operational-state information can be drawn from those placebos.

In particular, the claims that “state adds nothing,” “value saturates at belief,”
and “state-rich increment is approximately zero” are **retracted as unsupported**.
The correct conclusion is narrower: the tested controllers showed positive
development ReT deltas, but none satisfied the frozen actual-transport frontier;
some also failed metric or state-counterfactual guardrails.

## 5. Scientific disposition

- Program O retains its custody-verified full-DES perfect-information result:
  safe H_PI `0.15151`, simultaneous LCB95 `0.11562`, with the exact fungible null.
- Program O does **not** establish H_obs under the frozen state-rich gate.
- Do not open `7420049–7420096`.
- Do not train or freeze a learner.
- Paper 2 and Paper 3 remain unauthorized.

Any reconsideration of fixed-reserved versus pay-per-use freight is a new governance
decision requiring a new, prospectively frozen resource estimand. It cannot be used
to relabel this stopped run after the fact. Domain validation of the two-product
non-fungibility construct remains useful but is not a substitute for this gate.
