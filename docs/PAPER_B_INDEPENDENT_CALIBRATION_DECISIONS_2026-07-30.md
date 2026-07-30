# Paper B — independent calibration decisions

**Status:** `METHOD_DECISION_FREEZE_NO_EXTERNAL_INPUT_REQUIRED`.

These decisions remove two collaborator-dependent blockers without pretending that
missing domain data were observed. Garrido may still face-validate them, but execution
and the bounded methodological claims do not wait for that response. The machine-readable
freeze is
`contracts/paper_b_independent_calibration_v1.json`.

## What the sources actually determine

The thesis determines the service commitment and the ReT classification:

- Section 6.8.2 says that finished-product availability at Op9 allows troops to be
  supplied within a **pre-set lead-time of 48 hours**.
- Section 5.5.1 and Algorithm 1 define autotomy when a risk intersects the order and
  **`CTj = LTj`**.
- The thesis does **not** specify a fixed 54-hour fulfillment delay.

The 2024 factory-resilience paper determines a reproducible cost assumption, not an
economic calibration:

- Section 3.1 assumption (6) sets all seven coefficients to **`c = 1`** to isolate the
  index from cost parameters.
- Section 5 explicitly studies what happens when cost parameters vary. Therefore
  `c = 1` is a published baseline, but not evidence that one unit of inventory, spare
  capacity, production, backlog, hiring, firing, and overtime has the same MFSC cost.

The relevant pages were checked both as extracted text and as rendered PDF pages:
thesis printed page 111 (PDF page 112), and journal printed pages 7 and 11
(PDF pages 8 and 12).

## Decision 1 — the 54-hour delay

`54 h` remains unchanged in every historical reproduction and frozen experiment. It is
labelled **provisional aggregate-reproduction anchor**, because that is what it is: the
smallest tested value beyond `LT=48` that reproduced the raw-workbook ReT order of
magnitude.

It is not promoted to a physical fact. Prospective robustness work uses the complete,
predeclared signed-slack grid around `LT=48`:

| signed slack `delay − LT` | −6 | −1 | 0 | +1 | +6 | +12 |
|---:|---:|---:|---:|---:|---:|---:|
| delay (h) | 42 | 47 | 48 | 49 | 54 | 60 |

All rows must be reported. A delay cannot be chosen because it creates greater oracle,
controller, or neural headroom. A universal claim requires directional consistency over
the grid; otherwise the result is a delay-by-policy interaction.

The prospective ledger also separates two semantics that had been conflated:

- thesis autotomy: `CTj = LTj`, within a declared numeric tolerance;
- operational on-time service: `CTj <= LTj`.

The historical Excel-compatible column remains untouched.

## Decision 2 — κ and costs

We do not invent pesos, dollars, or exchange rates. We also do not keep Paper B blocked
waiting for them.

The primary resource comparison is a **physical Pareto front**, with service and resource
axes reported separately: fill, lost orders, unresolved backorder, delivered rations,
strategic injection, terminal stock, and shift/capacity use. Exploratory service floors
are swept and are never called deployment thresholds.

Cobb-Douglas remains a secondary sensitivity:

1. reproduce Garrido's published `c=1` baseline;
2. reprice the active static terms `c_p`, `c_u`, `c_i`, and `c_b` over one-factor
   multipliers `0.5, 1, 2, 5`;
3. report ranking stability and the physical Pareto front;
4. do not select a policy from Cobb-Douglas.

`c_h`, `c_l`, and `c_o` are structurally unidentified in fixed-shift static postures:
there is no within-run hiring/firing or overtime signal to price. A monetary vector is
required only for a monetary claim, an economic deployment recommendation, or promotion
of Cobb-Douglas to a policy-selection endpoint.

The executable grid is
`contracts/cobb_douglas_economic_sensitivity_v2.json`; its repricing artifact is
`results/cobb_douglas/economic_sensitivity_v2/result.json`. No DES trajectory was
replayed or changed. Over the frozen eight-arm panel:

- R1r retains the same Cobb-Douglas winner in all 13 scenarios;
- R2r does not: the winner changes between `v2_greedy_pi_best_found_v2` and
  `v2_replay_mpc_v2` when holding or backorder prices move.

The PI label remains `GREEDY_PI_BEST_FOUND_NOT_EXACT_CEILING`, and neither arm may be
selected from this sensitivity. The result establishes price-dependent ranking in R2r,
not an economic winner.

## What the 648-posture screen establishes

The complete static domain is now enumerated: `6^3 × 3 = 648` buffer/shift postures.
That closes the static coverage gap. It does not close the adaptive joint-controller
question, and the artifact stores means rather than per-root rows, so it has no paired
confidence intervals.

The varying ReT shift winners establish a **buffer × capacity interaction** and rule out
a context-free claim such as “more shifts always help.” They do not, by themselves,
prove that ReT fails to measure capacity: a legitimate production system may have
substitutable inventory and capacity.

Conversely, Cobb-Douglas choosing S1 in all 216 buffer vectors is not independent
validation. That unanimity is conditional on `c=1`, and the economic sensitivity already
shows that rankings can move with relative prices.

## Consequence

Neither missing item remains a human-input blocker:

- 54 h is a historical calibration anchor plus a prospective, non-selective sensitivity;
- κ is a published-baseline sensitivity plus a primary physical Pareto analysis.

Optional feedback from Garrido can improve face validity later. It cannot retroactively
select a delay, a cost vector, a controller, or an endpoint after results are visible.
