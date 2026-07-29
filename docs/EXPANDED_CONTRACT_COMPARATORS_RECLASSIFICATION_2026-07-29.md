# Reclassification of commit 5413cdd — the step-3 instrument is not valid

**New status:** `DEVELOPMENT_INSTRUMENT_DIAGNOSTIC_NOT_VALID_MPC_OR_DDMRP_ADJUDICATION`

Adversarial review found defects in `supply_chain/expanded_contract_controllers.py` and
`scripts/run_expanded_contract_comparators.py` that invalidate the adjudications in
`EXPANDED_CONTRACT_STRUCTURED_COMPARATORS_2026-07-29.md`. Every one was verified against the
code before conceding. The numbers in `results/expanded_contract_comparators/result.json` are
real; what they support is much narrower than what I wrote.

The hardened 12-tape / 5-scenario run was killed rather than allowed to finish, because
scaling a defective instrument only produces tighter intervals around the wrong estimand.

## Defects, verified

**1. The MPC is not an MPC.** `ReceedingHorizonMPC.act(self, sim, epoch)` never reads `sim`.
It replans on the committed action prefix alone, so it is open-loop replanning, not
state-feedback control. Worse than the review stated: `make_rollout` is called with
`scenario_seed = t + 10_000*(k+1)`, a *different* exogenous stream from the episode being
controlled, so candidates were scored against futures unrelated to the realised trajectory.
A controller that cannot see the state cannot demonstrate that state-contingent control is
worthless.

**2. The static incumbent is the wrong one.** `level_targets()` sets all three nodes from a
single rung, so only 6 of the contract's 6^3 = 216 postures were searched. The buffer gate
had already found the best posture is **heterogeneous** — `op3_rm` 61,440, `op5_rm` **0**,
`op9_rations` 126,000 — which is absent from that set. So "MPC loses to the best static" is
unsupported: it lost to the best *homogeneous* static, and the true incumbent is stronger
still. My claim that "the static incumbent sits inside the MPC's candidate set" is false for
the real incumbent, which also removes the search-failure argument I built on it.

**3. "Identical admissible set" is false.** Static and MPC choose among 6 rungs; DDMRP emits
unbounded continuous targets and injected 26–32x more material. The arms did not share an
action domain, so the comparison is not rights-matched. I asserted the opposite in both the
document and the commit message.

**4. The DDMRP is stylised, not complete.** `window_days` is declared and never used — ADU is
the last interval's consumption, not a rolling window. Net flow position is on-hand only, with
no on-order pipeline and no qualified demand spikes. `total_order_fulfilled` is used as the
consumption proxy for all three nodes, including the two raw-material nodes. And replenishment
is still the exogenous unpriced top-up.

**5. Mean flatness does not bound adaptive value — this was my central error.** I argued that
because the static ladder is flat above I168 to within 1.6e-04, "the spread of the admissible
set bounds what any adaptive policy over that set can buy". That is a fallacy. Two actions can
have nearly identical *marginal means* and still swap ranking as a function of state;
adaptation exploits the state-conditional ranking, not the marginal spread. The argument
proves nothing about state-contingent control.

**6. Custody is too coarse.** Only per-arm means and sign counts were persisted. Per-tape rows,
chosen actions, DDMRP targets, rollout values and states were discarded, so paired intervals
cannot be computed and the mechanism cannot be audited.

## What the run does still support

As exploratory development evidence only:

- A non-zero buffer is worth a great deal against `I0` (−0.000884 in R1r, −0.135289 in R2r).
- Among the six *homogeneous* postures, `I168` had the best mean `ret_excel`.
- The stylised DDMRP produced a higher fill rate while consuming vastly more material.
- `ret_excel`, `ret_excel_full_ledger` and `ret_thesis` do not induce the same ranking.
- A nominal two-scenario open-loop replanner did not beat `I168`.

## What must not be claimed from it

- that DDMRP was beaten;
- that this is the expanded-contract MPC of Garrido's step 3;
- that dynamic buffer control has no value;
- that the neural residual is closed on this contract.

## The correct instrument for the bound I wanted

Tape-level adaptive headroom was already measured, on the full 216-posture heterogeneous set,
by `results/authority_ladder/buffer_gate/screen_result.json`:

    H_PI      = 1.158e-04      (clairvoyant per-tape posture choice minus best fixed posture)
    LCB95     = 7.886e-05
    grid span = 1.204e-03
    tapes with a strictly better posture = 11

That is the honest statement: choosing the posture *per tape with perfect foresight* buys
1.16e-04 over the best fixed posture, against a grid span an order of magnitude larger. It
bounds tape-level posture selection. It does **not** bound epoch-level adaptation within a
tape, which is a strictly larger space and remains unmeasured.

## Required before re-running at 12 tapes x 5 scenarios

1. Enumerate all 216 heterogeneous postures and freeze the true incumbent.
2. Make each rollout replay the realised episode to the current epoch under the *same* seed,
   verified by a state hash before branching, then branch the future.
3. Give the MPC the per-node action space, not the coupled rung.
4. Persist per-epoch action, state, target, estimated value and scenario for every tape.
5. Complete the DDMRP: per-node ADU on a real window, on-order, qualified spikes.
6. Project DDMRP onto the same action domain, or compare under a resource budget / Pareto front.
7. Measure action-ranking reversals across states and a per-tape dynamic ceiling.
8. Compute paired intervals from the persisted rows.

## Bearing on the metric question

Defect 4 aside, the finding that DDMRP delivers a **higher fill rate and a worse ReT** is not
an artefact of the stylisation — the same inversion appears for shift postures in
`AUTHORITY_LADDER_SCREEN_FINDINGS_2026-07-28.md` §3, where ReT ranks shift 1 first while shift
3 delivers 18% more rations with zero lost orders. Two metrics that disagree with physical
service in the same direction, twice, is a property of the endpoint rather than of the
controllers. This is the concrete case for the Cobb–Douglas index the PI proposed and Garrido
authorised: it reads spare capacity, backorders and time-to-fulfil straight off the physical
ledger, with no case classification and no order-visibility filter, and it prices the resources
that ReT leaves free. Spec in `COBB_DOUGLAS_RESILIENCE_PORT_SPEC_2026-07-28.md`.
