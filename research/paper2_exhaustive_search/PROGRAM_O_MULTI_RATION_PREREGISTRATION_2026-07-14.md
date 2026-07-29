# Program O — multi-ration shared-line product-mix gate

Status: **frozen before the Stage-A result**. No Program-O stochastic tape has
been opened. No learner or Paper 3 work is authorized.

## Why this family is open

The thesis states that the real MFSC produces 21 ration types selected for
nutritional and climatic requirements, while the executable DES deliberately
compresses them to one Cold Weather Combat Ration. Restoring product identity
is therefore a domain-anchored omitted dimension. The thesis does not provide
the numerical BOM, setup, substitution, product-share or forecast parameters;
those are disclosed researcher-introduced physics and sharply limit any claim.

This is not a request for Garrido to find headroom. We specify a falsifiable
contract ourselves. Later expert review may assess face validity, but it is not
a prerequisite for the development search.

## Necessary mechanism

Two nonfungible product classes share fixed Op5–Op7 capacity. Each weekly
action assigns the next three existing 5,000-ration batches between products;
it does not buy capacity, inventory, shifts or transport. A persistent product
mix and an imperfect pre-commitment warning can reverse the ranking of the
allocation action. Setup and partial substitution can make a wrong commitment
persist. Total demand, gross capacity and downstream transport remain fixed.

The null cell makes the two labels physically interchangeable: identical BOM
and rate, zero setup, complete substitution, 50/50 label-independent demand and
an uninformative warning. Relabeling must then be bit-identical in aggregate.

## Frozen gate sequence

1. **Affected-order ceiling.** Use only the already-open Garrido Excel rows and
   the canonical `ret_excel_request_snapshot_v2` implementation. Calculate the
   smallest fraction of visible rows that would have to improve perfectly to
   raise mean ReT by 0.01. This is metric liveness, not policy headroom. Proceed
   only if at least one workbook requires no more than 10% of rows.
2. **Corrected exact transducer.** If Gate 1 clears, freeze the two-product
   transducer before opening seeds 7400001+. It must serve backlog from later
   inventory, freeze pending setup targets, conserve all components and use
   the canonical aggregator. Enumerate all `4^8=65,536` calendars.
3. **Full-DES H_PI.** Only if the transducer has exact null identity and
   material resource-matched H_PI, implement product-tagged lots through the
   real Op10–Op12 path. Freeze a connected cell-selection and simultaneous
   validation rule before opening the full-DES validation block.
4. **H_obs.** Only after validated full-DES H_PI, compare a deployable belief
   policy directly with the best complete open-loop calendar, base-stock,
   hysteresis, min-cost flow, rolling MILP/MPC and exact/approximately bounded
   belief DP.
5. **Learner.** Prohibited until all pre-learner gates, guardrails and the
   action-trajectory certificate pass.

## Claim boundary

A positive transducer or full-DES result would establish adaptive value only
inside the disclosed two-product extension. It would not show that the chosen
numerical parameters describe the real MFSC. If an interpretable belief/MPC
policy wins and a learner does not improve on it, the result is adaptive
production planning with no neural incremental value.

The complete machine contract is
`contracts/program_o_multi_ration_product_mix_v1.json`.
