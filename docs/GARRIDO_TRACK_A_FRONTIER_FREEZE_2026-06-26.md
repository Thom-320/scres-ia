# Garrido Track A Frontier Freeze

Date: 2026-06-26

## Frozen Choice

Primary Track A frontier for the next retained-vs-reset pilot:

- Reward mode: `ReT_garrido2024`
- Primary outcome: `cd_sigmoid_index`
- Risk frequency multiplier: `1.0`
- Risk impact multiplier: `1.0`
- Shift cost: `0.5`
- Kappa train fraction: `0.2`
- Grid gate: joint `6x3` inventory x shift
- Calibration: current repo file `supply_chain/data/ret_garrido2024_calibration.json`

Machine-readable freeze:
`supply_chain/data/garrido_track_a_frontier_freeze_2026-06-26.json`

## Metric Consensus

The Cobb-Douglas freeze below remains the cost-aware Track A frontier lane, but
it is no longer the only paper-facing outcome. The canonical evaluation target
for "beat Garrido" is a panel:

- Clean resilience/service: order-level fill rate, lost/unattended orders,
  backorder quantity, service-loss area, p95/CVaR service loss, and recovery
  diagnostics.
- Continuity with Garrido 2017: `mean_ret_excel_formula`, reported and audited,
  but not optimized as a standalone objective.
- Resource efficiency: shift-hours, extra shift-hours, strategic buffer target
  units/unit-hours, and Cobb-Douglas/resource-cost indices.

Reason: the static baseline panel shows that raw Excel ReT can be
non-monotone and even anti-correlated with operational service. In the 20-year
panel, `original_S1_I0` has higher Excel ReT than the buffered policies while
losing many more orders and delivering much lower fill. Therefore Excel ReT is
kept as a faithful continuity metric, not as the single optimization target.

Consensus baseline hierarchy:

- Strong efficient frontier to beat: `I168_S1` in the static panel, with
  `I168_S2` retained as the severe-regime capacity sensitivity.
- Expensive thesis-style comparator: `I1344_S3`; useful for showing resource
  waste, but not sufficient as the only baseline for a final win claim.
- Original thesis/control comparator: `original_S1_I0`.

Next gate before more PPO tuning: add a simple threshold heuristic to the same
panel/horizon/seeds. If a deterministic rule can match `I168_S1` service with
less resource, it becomes the stronger comparator that RL must beat.

## Workbook Fidelity Audit

Audit artifact:

- `outputs/audits/garrido_workbook_fidelity_2026-06-26/audit_report.md`

Result:

- `Raw_data1+Re.xlsx`: primary order-level target for `CF1-CF10`.
- `Raw_data2+Re.xlsx`: primary order-level target for `CF11-CF20`.
- `Rsult_1.xlsx`: secondary aggregate/distribution workbook for `APj`, `RPj`,
  `DPj`, and `Re`; not a one-to-one trajectory replay target.
- Extraction gate: `47,546` raw rows, `0` formula mismatches, `CF2` present.

The remaining ReT divergence is not a formula-replication failure. It is a
branch-composition/fidelity issue: in the raw Excel targets, `CF1-CF10` are
almost entirely risk-active recovery rows (`~99.5%` recovery, median positive
`RPj ~100 h`). In our DES, `original_S1_I0` still produces a visible
risk-inactive/fill-rate branch (`~8.4%`) and a much fatter positive `RPj` tail
(`p95 ~13,944 h` in the current-regime two-seed audit). That can make raw
Excel-ReT means diverge from service quality even though the formula itself is
exactly reproduced.

Implication: before claiming thesis-level behavioral equivalence, keep auditing
RPj/backlog tail and risk attribution for no-buffer policies. For policy
optimization, report Excel ReT for continuity but decide wins on the full
service/resource panel.

Follow-up fix:

- Code path fixed: `demand_on_hand_fulfillment_delay` now applies to orders
  served through the pending-backorder path as well as direct on-hand orders.
  Previously, a queued order could be reserved and finalized in the same
  timestamp, producing `CTj=0` despite the configured delay.
- Audit artifacts:
  - `outputs/audits/garrido_workbook_fidelity_delay48_2026-06-26/audit_report.md`
  - `outputs/audits/garrido_workbook_fidelity_delay100_2026-06-26/audit_report.md`
- `48.00744 h` removes instant orders but makes buffered policies recover at
  `RPj~48 h`, below the `Raw_data1` target.
- `100 h` is a useful sensitivity for matching the median positive `RPj` in the
  `CF1-CF10`/R1-style lane: `I168_S1` current moves to Excel ReT `~0.0050`,
  recovery share `~99.95%`, median positive `RPj=100 h`, close to `Raw_data1`
  family means of ReT `~0.0063`, recovery share `~99.53%`, and median positive
  `RPj~99.8 h`.
- Remaining blocker: no-buffer policies still have a long RPj tail and lost
  orders (`original_S1_I0` p95 positive `RPj~1656 h` under delay `100 h`).
  That points to backlog-flow/capacity mechanics, not the on-hand delay path.

Default decision after the Claude/Codex cross-check:

- Verification artifact:
  `outputs/audits/garrido_delay54_claude_verification_2026-06-26/verification.json`
- The paper-facing Garrido/Track A default is now
  `demand_on_hand_fulfillment_delay=54.0 h`, defined in
  `THESIS_FAITHFUL_PROTOCOL`. It is also the default in `MFSCSimulation`,
  `MFSCGymEnvShifts`, the static panel, the dynamic-vs-static runner, and the
  Garrido Excel replication runners; legacy instant fulfilment must now be
  requested explicitly with `demand_on_hand_fulfillment_delay=0.0`.
- Reason: it is the smallest calibrated value that removes `CTj=0` and crosses
  the thesis `LT=48 h` cliff. In the verification sweep, `delay=0` produced
  many instant orders and ReT `~0.099`; `delay=48` removed instant orders but
  left many orders exactly on-time and ReT `~0.245`; `delay=54` produced
  `0%` instant, about `91%` late plus the remaining unfulfilled/lost tail, and
  ReT `~0.007`, matching the `Raw_data1`/`CF1-CF10` order of magnitude
  (`~0.0063`).
- `delay=100 h` remains a sensitivity for matching median `RPj~100 h`, but it
  is no longer the default because it imposes a larger artificial delivery lag
  than needed to reproduce Garrido's late-order structure.
- The heavy-tail caveat remains: the current DES p99 CT/RP tail is still much
  heavier than Garrido's `CF1` tail. In the local check, `delay=54` with
  `increased` risk gave p99 `~20,304 h` at a 10-year horizon and `~20,632 h`
  at a 20-year horizon, versus Garrido `CF1` p99 around `6,628 h`. That is the
  next fidelity blocker.

## Threshold Heuristic Gate

Heuristic artifact:

- `outputs/benchmarks/garrido_dynamic_vs_static/threshold_heuristic_5seed_104w_2026-06-26/summary.json`

Setup: no PPO training, `5` seeds, `104` weeks, `current/increased/severe`,
same Track A `Discrete(18)` action surface. Two rules were tested:

- `heuristic_threshold_lean`: default `S1/I0`, switch to `S2/I168` under
  backlog/service-loss/disruption stress.
- `heuristic_threshold_buffer`: default `S1/I168`, switch to `S2/I168` under
  stress.

Result against the efficient frontier:

| Regime | Heuristic | Frontier | Delta CD | Delta fill | Delta extra shift | Delta buffer |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| current | threshold_buffer | S1/I168 | -0.0302 | -0.0034 | +2,956.8 | 0.0 |
| increased | threshold_buffer | S1/I168 | -0.0204 | +0.1029 | +14,380.8 | 0.0 |
| severe | threshold_buffer | S2/I168 | -0.0103 | +0.0021 | -168.0 | 0.0 |

Conclusion: the simple reactive threshold rule is useful as a comparator, but
it does not beat the efficient frontier. It improves service in `increased` by
buying many extra shift-hours, and in `severe` it approximately ties service
with only a tiny shift saving. The next dynamic-policy work must either
anticipate regime transitions earlier or explicitly optimize resource use; a
plain backlog-reactive rule is not enough.

## Why This Wins

This is the most defensible next lane because it is parsimonious and efficient:
it does not require extra risk-frequency tuning, it keeps the optimal inventory
interior (`I168`), and the fast control lever moves from `S1` in calm regimes to
`S2` under severe disruption. That gives Track A dynamic headroom without
making the paper defend permanent high-capacity operation.

The train-shaped alternative, `cd_train_index`, is retained as a sensitivity.
It produced a stronger S1-to-S3 frontier when `risk_frequency_multiplier=2.0`,
but that uses extra risk tuning and a less efficient severe-regime shift choice.

## Evidence

Primary gate:

- `outputs/experiments/garrido2024_joint_grid_2026-06-26/summary.json`
- Metric: `ret_garrido2024_sigmoid_total`
- Regime argmax: `current=S1_I168`, `increased=S1_I168`, `severe=S2_I168`
- Interior optimum: pass
- Robust regime-dependent shift flip: pass

Codex/Claude comparison:

- `outputs/experiments/garrido_cd_joint_claude_codex_comparison_2026-06-26/comparison.md`

## Primary Pilot Result

Pilot artifact:

- `outputs/benchmarks/retention_transfer/garrido_cd_sigmoid_phi1_pilot_2026-06-26/transfer.json`

Result under the frozen primary lane:

- Reward mode: `ReT_garrido2024`
- Outcome: `cd_sigmoid_index`
- Frozen env: `risk_frequency_multiplier=1.0`, `risk_impact_multiplier=1.0`,
  `ret_g24_shift_cost=0.5`, `ret_g24_kappa_train_frac=0.2`
- Seeds: 10
- Blocks per seed: 24
- MEMORY retained-minus-reset: `-0.0004`, CI95 `[-0.0053, +0.0046]`
- Total retained-minus-frozen: `+0.0022`, CI95 `[-0.0009, +0.0054]`

Interpretation: the frozen cost-aware Track A lane has a legitimate static
frontier, but this pilot does not yet show a retained-memory advantage. Treat
this as a clean pilot null, not as the final powered result.

## Dynamic-vs-Static Pilot

Runner:

- `scripts/compare_garrido_dynamic_vs_static.py`

Clean pilot artifact:

- `outputs/benchmarks/garrido_dynamic_vs_static/pilot_3seed_512t_52w_clean_2026-06-26/summary.json`
- Dominance-plane rerun with full static baselines and resource metrics:
  `outputs/benchmarks/garrido_dynamic_vs_static/dominance_3seed_512t_52w_v3_2026-06-26/summary.json`

This is the first same-action-surface Track A comparison: PPO is trained on
`Discrete(18)` and evaluated against the frozen efficient static frontier
(`S1_I168` for current/increased; `S2_I168` for severe).

Result:

| Regime | PPO CD | Static CD | Delta CD | Delta Excel ReT |
| --- | ---: | ---: | ---: | ---: |
| current | 0.6332 | 0.7300 | -0.0967 | +0.0520 |
| increased | 0.6162 | 0.7026 | -0.0864 | -0.0489 |
| severe | 0.5849 | 0.6598 | -0.0749 | -0.2663 |

Interpretation: the current PPO pilot does not beat the efficient static
frontier under the primary Cobb-Douglas outcome. In `current`, PPO improves
Excel ReT but does so with much weaker fill/service and lower cost-aware CD, so
it is not a defensible win. The learned policy also avoids `S2` and jumps
between `S1` and `S3`, while the severe static optimum is `S2_I168`. The next
training fix should target efficient S2 usage before spending on a powered run.

Dominance-plane read:

- PPO has one resource-only Pareto win against the expensive thesis-style
  `static_S3_I1344` baseline in `current`: CD `+0.0275`, Excel ReT `+0.0489`,
  extra shift-hours `-11648`, buffer target units `-255436`.
- That is not a strict service-safe dominance win: fill is lower by `-0.331`
  and p95 service loss is worse (`0.890` vs `0.334`).
- No strict service-resource dominance rows pass yet.

Therefore the paper claim is not ready as "dynamic beats Garrido" on this
pilot. The useful signal is narrower: the runner now exposes the right Pareto
plane, and the next reward/curriculum change must preserve Excel ReT and tail
service while reducing resource use.

## S2 Reward/Curriculum Screen

The first reward-only severe screen showed that `ReT_garrido2024`,
`ReT_garrido2024_raw`, and `ReT_garrido2024_train` did not reliably learn the
efficient `S2` action. `control_v1` learned `S2` usage but lost too much
service/Excel ReT.

The key experimental correction was to initialize PPO with the same strategic
buffer family used by the static policies (`static_S1_I168`) before learning
starts. This removes the unfair default-start disadvantage where PPO began from
`S1/I0` while static baselines began with their strategic buffers already
installed.

Current best candidate:

- Reward: `ReT_seq_v1`
- PPO initial decision: `static_S1_I168`
- Evaluation: unchanged (`cd_sigmoid_mean`, Excel ReT, fill, p95/CVaR, shift
  hours, buffer units)
- 10-seed artifact:
  `outputs/benchmarks/garrido_dynamic_vs_static/dominance_retseq_i168_10seed_2048t_52w_2026-06-26/summary.json`

10-seed result:

| Regime | PPO CD | PPO Excel | PPO Fill | PPO S1/S2/S3 |
| --- | ---: | ---: | ---: | ---: |
| current | 0.6863 | 0.0300 | 0.946 | 60.0 / 16.2 / 23.8 |
| increased | 0.6595 | 0.2382 | 0.597 | 41.5 / 25.1 / 33.4 |
| severe | 0.6422 | 0.3102 | 0.399 | 33.5 / 26.1 / 40.5 |

Dominance read:

- Resource-Pareto wins vs expensive `static_S3_I1344` in all three regimes.
- Strict service-resource win vs `static_S3_I1344` in `current` only.
- Against the efficient frontier, PPO is close but does not dominate:
  - current vs `S1_I168`: Excel `-0.0002`, fill `+0.013`, but more resources.
  - increased vs `S1_I168`: Excel `+0.0054`, but fill `-0.032` and more
    resources.
  - severe vs `S2_I168`: Excel `-0.0036`, fill approximately tied, but more
    resources.

Next action: tune the resource side of this candidate, not the service side.
The candidate now preserves much more Excel/service than earlier PPO runs; the
remaining gap is excess buffer and severe shift usage versus the efficient
static frontier.

## Garrido 2017 Alignment

The 2017 thesis does not contain a cost-aware static optimum. It tests three
separate simulation scenarios: increased risk frequency, increased inventory
buffers, and increased short-term manufacturing capacity. In Scenario II the
inventory experiment keeps `S=1` fixed; in Scenario III the capacity experiment
keeps zero inventory at the critical storage points. The thesis therefore does
not estimate the joint `inventory x shift` optimum used by this Track A gate.

Under the 2017 ReT/Excel logic, higher buffers and higher capacity generally
increase resilience because cost is excluded. Section 8.5.2 explicitly names
the non-inclusion of cost as a limitation, and Section 8.6.2 proposes future
work on an optimum level of SCRes that includes cost. The frozen Cobb-Douglas
lane should therefore be described as Garrido-aligned future work / 2024-style
cost-aware extension, not as the exact 2017 ReT formula.

## Caveat

The exact severe shift (`S2` vs `S3`) is sensitive to the Cobb-Douglas variant
and calibration. The current freeze uses the repo calibration for
reproducibility. Re-derived faithful calibration is a declared sensitivity, not
part of this pilot freeze.

## Pilot Command

```bash
python scripts/retention_transfer.py \
  --label garrido_cd_sigmoid_phi1_pilot_2026-06-26 \
  --reward-mode ReT_garrido2024 \
  --outcome cd_sigmoid_index \
  --risk-frequency-multiplier 1.0 \
  --risk-impact-multiplier 1.0 \
  --ret-g24-shift-cost 0.5 \
  --ret-g24-kappa-train-frac 0.2 \
  --seeds 8201,8202,8203,8204,8205,8206,8207,8208,8209,8210 \
  --n-blocks 24 \
  --max-steps 12 \
  --train-per-block 150 \
  --rho-disruption 0.85 \
  --regime-seed 909 \
  --mask-preset direct_disruption_blind \
  --learning-starts 50 \
  --buffer-size 10000
```
