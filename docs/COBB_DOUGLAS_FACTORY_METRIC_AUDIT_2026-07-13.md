# Cobb-Douglas factory-resilience and canonical ReT audit

Date: 2026-07-13
Machine record: [`metric_governance_audit.json`](../research/paper2_exhaustive_search/metric_governance_audit.json)

## Verdict

**2026-07-14 supersession:** the source-aligned canonical development endpoint
is now `ret_excel_request_snapshot_v2`. The earlier
`ret_excel_visible_v1` OAT-derived `Bt/Ut` ledger is quarantined as
metric-development only and cannot establish a Paper-2 null, positive or
ceiling. The fresh workbook audit still establishes its narrow result: all 20
raw `CF` sheets and 47,546 formula-bearing rows replay with zero mismatches and
maximum absolute error `0.0` when workbook `Bt/Ut` are supplied. It did not
validate reconstruction of those ledgers from `OATj`.

The v2 aggregator now accepts request-time `Bt/Ut` snapshots and reproduces the
same 47,546 cells exactly. This is provisional pending confirmation of
same-timestamp Simulink ordering and re-score of all eligible tapes and
comparators. Lost orders and the other physical/resource outcomes remain
simultaneous guardrails.

Garrido, Pongutá and García-Reyes (2024) supplies a legitimate second
construct, but it is a **factory-level aggregate-production-planning index**,
not the thesis order-level MFSC ReT. In the current Paper 2 contract it may be a
frozen secondary construct-sensitivity measure; it cannot rescue a quarantined
v1 result or substitute for a future v2 Paper-2 result.

## What the 2024 index measures

The paper studies a hypothetical make-to-stock factory for 36 weeks. Demand
variability is the sole uncertainty. Seven fixed pure APP substrategies are
compared: three zero-inventory/chase rules, two constant-workforce rules and two
hybrid rules. Its five variables are average inventory `zeta`, backorders
`epsilon`, spare capacity `phi`, time/effort to meet net requirements `tau`, and
normalized total cost `kappa_dot`.

The published log score is

```text
0.024 ln(zeta) - 0.026 ln(epsilon) + 0.040 ln(phi)
- 0.060 ln(tau) - 0.1771 ln(kappa_dot)
```

and the reported index is its sigmoid. The coefficients were fitted so that
each term contributed 0.2 at the maximum observed in 10,000 simulations of the
paper's factory. They are therefore not universal MFSC weights. The paper's own
result is also a warning for adaptive-headroom work: fixed match-chase strategy
`S12` dominates the other pure strategies, with reported median index
`0.55784`. The paper does not test sequential feedback, learning, or retained
learning.

## Three repository objects that must not be conflated

1. `supply_chain.program_g.metrics_all` uses the published exponents only as a
   distribution-construct probe. Its `phi` and `kappa` are fixed at one and its
   two-CSSU variables are not the paper's factory calibration. Program G
   correctly labels this “Cobb-Douglas-inspired.”
2. `MFSCGymEnvShifts._compute_ret_garrido2024` maps five APP-like variables into
   the full DES. That mapping is useful experimentally, but the default file
   named `ret_garrido2024_calibration_faithful_2026-06-26.json` stores
   DES-rebalanced exponents but activates `variance_log` balancing, under which
   standardized log variables replace the published coefficient calculation;
   it also applies a post-hoc de-saturation constant. It is **not the published
   Equation 5/6 index**. The
   DES cost ledger also omits hiring, firing and overtime variables that the
   APP paper includes.
3. The published equation itself can be reproduced exactly when supplied with
   positive, valid factory components and the paper's strategy-normalized cost.
   Mapping MFSC state into those components remains a researcher adaptation and
   needs its own validation and calibration declaration. The isolated
   `supply_chain.factory_resilience` module now provides this formula-only
   implementation without changing any historical runtime or result.

Historical artifact names and results are preserved for provenance; this audit
narrows their claim labels rather than rewriting them retrospectively.

## Does Cobb-Douglas expose headroom?

The honest answer is “a small exploratory action surface, but no established
adaptive or learned win.” The prior clean `phi=2, psi=1` Track-A screen reported
about `0.00816` adapted-CD sigmoid oracle-minus-robust-fixed spread. That is not
a certified `H_PI`: the comparator is not the complete full-horizon open-loop
frontier, the endpoint is not canonical ReT, and the tapes are not virgin.

The current PPO evidence is negative against the best static CD policy:

| Regime | PPO minus best static CD |
|---|---:|
| current | -0.083453 |
| increased | -0.012111 |
| severe | -0.015638 |

The apparent `phi=4, psi=1.5` war-stress win was invalidated. Corrected PPO CD
was `-0.0418` below the best static policy, while flow fill, lost rate and Excel
ReT were all materially worse. Program G independently shows the broader
failure mode: the CD-preferred allocation raised the index while reducing
canonical order ReT, attended orders and worst-CSSU fill.

Consequently:

- CD is useful as a prespecified construct-sensitivity and factory-APP outcome.
- It does not supply current Paper 2 headroom under the exact success standard.
- A separate factory-resilience paper is possible, but requires a new frozen
  APP contract, complete calendar/classical comparators and a validated cost and
  variable ledger. It must not be presented as thesis ReT.
- No learner or Paper 3 work is authorized by the CD evidence.

## Risk changes

Editing thesis risk parameters or introducing a realistic risk is allowed only
through a frozen intervention ledger: operational meaning, affected operations,
units, conservation/resource effects, source or exact Garrido sign-off question,
plausible range, pre-action observation timing, null mechanism-removal cell and
claim limit. Risk magnitude may not be increased until a preferred policy or
metric wins. In particular, the old `phi4/psi1.5` stress cell remains exploratory
and cannot be relabeled thesis-faithful.
