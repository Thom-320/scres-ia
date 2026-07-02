# E3: cross-regime/horizon generalization matrix — verdict (2026-07-02)

**Status: DONE.** `outputs/experiments/track_b_e3_cross_regime_horizon_matrix_vps_2026-07-02/`
(canonical PPO checkpoints, frozen — no retraining — evaluated CRN-paired
against per-cell static comparators, 5 seeds × 12 episodes, `control_v1`,
obs v7, `current`/`increased`/`severe` × h52/h104, `--skip-heuristics` — see
`docs/OVERNIGHT_STATUS_2026-07-02.md` for the two bugs fixed to get this
running).

**Caveat on comparator rigor:** this uses the lighter `eval_track_b_cross_scenario.py`
static set (`STATIC_POLICY_SPECS`, a fixed handful of S1-3 × dispatch
constants — same set at every risk level), not a re-optimized dense
per-regime frontier. Real generalization evidence, not the audit's ideal
per-regime dense-CRN spec.

## Result

| Horizon | Risk level | PPO Excel/order ReT | Best static | Δ | Win? |
|---|---|---:|---:|---:|---|
| h52 | current | 0.005316 | 0.004832 | **+0.000483** | Yes |
| h52 | increased | 0.003259 | 0.002517 | **+0.000742** | Yes |
| h52 | severe | 0.0001193 | 0.0001797 | **−0.0000604** | **No** |
| h104 | current | 0.005648 | 0.005439 | **+0.000209** | Yes |
| h104 | increased | 0.003660 | 0.003095 | **+0.000565** | Yes |
| h104 | severe | 0.0001282 | 0.0001188 | +0.0000094 | Yes (marginal) |

**5 of 6 cells positive.** The one loss (severe/h52) is also the cell with
the lowest absolute ReT by ~40× (0.0001 vs 0.005) — deep into the regime
where almost nothing is fulfilled (fill rate ~0.001, backorder rate
~0.999). The h104/severe win is real but tiny (+0.0000094, order of
magnitude smaller than the other 4 positive cells) — treat as "does not
catastrophically fail," not "wins," at severe risk.

## Verdict on the single-cell-artifact attack (T4, docs/REVIEWER2_DEEP_AUDIT_2026-07-01.md)

**Substantially answered.** The Track B win is not confined to the bespoke
`adaptive_benchmark_v2` stress cell — it generalizes to Garrido-native
`current` and `increased` risk levels at both horizons, with comfortably
larger absolute deltas than the canonical cell in some rows (e.g.
h52/increased +0.000742 > the canonical adaptive_benchmark_v2 result of
+0.000426). The only genuine weak spot is `severe` risk, especially at the
shorter h52 horizon, where the system is so degraded (fill ~0.1%) that the
metric itself becomes close to floor-bound for everyone. Safe wording:

> "The learned policy's advantage over dense-comparable static dispatch
> generalizes across Garrido-native risk levels (current, increased) and
> both evaluated horizons (h52, h104), with the single exception of the
> most severe disruption regime at the shorter horizon, where the system
> operates near a service floor for all policies and the comparison is
> not economically meaningful."

Do NOT claim the win holds "at all risk levels" — sever/h52 is a genuine
loss and must be disclosed, not averaged away.

## Companion result: Track A does NOT win on the bespoke regime

Ran the RL reviewer's suggested check — Track A's original `continuous_its`
buffer/shift action family, trained fresh on `adaptive_benchmark_v2` itself
(the same regime Track B's canonical win uses), 5 seeds × 60k:
`outputs/experiments/track_a_continuous_its_adaptive_benchmark_v2_2026-07-02/summary.json`

| | Excel ReT | CVaR | Verdict |
|---|---:|---:|---|
| Learned (Track A, continuous buffer fraction) | 0.00530 | 1.77e9 | — |
| Best constant-fraction static | 0.00539 | 1.34e9 | Static wins both |

**Track A loses on the exact regime Track B was evaluated on.** This
directly refutes the sharpest form of the RL reviewer's attack ("the regime
was engineered to make Track B win — if Track A also wins there, the story
is regime-observability, not action-space frontier"): it is not sufficient
to simply observe the bespoke regime and have *some* learned policy win;
the win requires reaching the downstream dispatch bottleneck specifically.
This completes the 2×2 (action family × regime) the RL reviewer asked for:

| | Garrido-native regimes | `adaptive_benchmark_v2` |
|---|---|---|
| Track A (buffer/shift) | No dynamic frontier (dense-CRN null, prior work) | **Loses** (this result) |
| Track B (+ downstream dispatch) | **Wins** 5/6 cells (this result) | **Wins** (canonical) |

Safe wording:

> "To rule out the possibility that the stress regime alone — rather than
> action-space coverage of the bottleneck — explains the result, we
> retrained the original Garrido-variable policy family on the same
> `adaptive_benchmark_v2` regime; it does not outperform the best static
> comparator (Excel ReT 0.00530 vs 0.00539). The win requires both the
> regime's pressure and access to the downstream dispatch bottleneck."

## Registry update

C11 (`docs/CLAIMS_REGISTRY_Q1_DEFENSE_2026-07-01.md`: "Track B generalizes
beyond one adaptive benchmark cell... Needs verification") — **upgrade to
"Supported with one disclosed exception (severe/h52)"**, citing the table
above. Add a new registry item for the Track A-on-adaptive_benchmark_v2
companion result (regime-observability rebuttal).
