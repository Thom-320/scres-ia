# ChatGPT Pro audit entrypoint — Q-R1 retained learning

Status: external adversarial audit surface. This file changes no contract, result, or historical verdict.

## North star and claim boundary

North star: demonstrate causally that retaining decision-relevant knowledge across campaigns improves cold-start supply-chain resilience after a complete physical reset.

The final neural claim would require a retained learner or hybrid to beat the best retained structured controller under identical causal history, memory rights, physical resets, actions, and resources. A win only against a reset MPC is not a neural premium.

All Q-R1 work on this branch is exploratory or burned prospective replication. No learner training is authorized by the current artifacts.

## Start here

1. `contracts/q_r1_retained_learning_discovery_v1.json`
2. `contracts/q_r1_cold_start_replication_v1.json`
3. `results/q_r1/discovery_terminal_summary_v1.json`
4. `results/q_r1/cold_start_replication_v1/adjudication.json`
5. `supply_chain/q_r1_retained_learning.py`
6. `supply_chain/program_t_joint_belief.py`
7. `supply_chain/program_t_full_des_mpc.py`
8. `scripts/run_q_r1_d1_demand_memory.py`
9. `scripts/run_q_r1_d3_residual_bound.py`
10. `scripts/run_q_r1_cold_start_replication.py`

## Frozen result to reproduce, not merely quote

The fresh burned replication used 24 histories, roots `7570801–7570824`, with 12 campaigns per history.

- retained binary context, persistence 0.90: mean early canonical ReT delta `+0.0226361`; history-clustered descriptive LCB95 `+0.0190475`;
- persistence 0.75: `+0.0137836`; LCB95 `+0.0093491`;
- iid: exactly `0`;
- shuffled history: `-0.0045399`;
- wrong history: `-0.0648259`;
- favorable pairs at persistence 0.90: `41.67%`;
- worst-product delta: `-0.0177425`;
- lost-order delta: `0`;
- maximum unresolved-order delta: `+3`.

The frozen adjudication is `STOP_RETAINED_EFFECT_NOT_REPLICATED_RESIDUAL_ONLY`. It records `d4_authorized=false` and `learner_training_authorized=false`. Do not reinterpret the positive ReT mechanism as a safe or learned premium.

## High-priority hypotheses the audit must try to falsify

These are suspected failure modes, not accepted conclusions.

1. **Off-policy common continuation.** The cold-start construction may concatenate the retained two-week prefix with weeks 3–8 of a reset-MPC calendar computed on a different physical trajectory, instead of applying the same continuation policy to each arm's actual state. Determine whether this creates avoidable unresolved orders or biases early ReT.
2. **Incomplete ledger pressure.** `early_ret_2w` uses the canonical visible request-snapshot ReT. Determine exactly how unresolved orders enter its denominator and whether a policy can improve the score by postponing completion without recording lost demand.
3. **D3 action-library contamination.** Determine whether the privileged D3 selector maximizes over shuffled and wrong-history candidate calendars. If so, recompute bounds for: all candidates; causal non-placebo candidates; deployable retained/reset candidates. Do not describe a candidate-library ceiling as history-learnable value.
4. **Weak structured comparator.** Q-R1 uses a scenario H3 MPC with four belief particles in important paths. Test convergence of actions and values with exact or stratified integration, larger scenario counts, H3/H8 where feasible, robust and constraint-aware modes. Distinguish compute infeasibility from lack of value.
5. **Metric/guardrail mismatch.** Determine whether `max_unresolved_orders_delta <= 0` is the intended operational constraint or an accidental maximum-over-pairs gate. Estimate mean, tail, and clustered uncertainty for unresolved orders, while retaining the hard no-lost-demand rule.
6. **Artificially small decision contract.** Quantify what `count4`, weekly dwell, eight decisions, and common continuation remove. Preserve Program Q's exact `4^8` frontier, but assess a prospective extension separately rather than claiming Q exhausts dynamic policies.
7. **Risk relevance.** Existing Garrido-native risks may damage the system without changing the relative value of product-mix actions. Audit risk ownership and direct-SimPy execution. Only recommend a researcher-defined risk if a learner-blind sensitivity design shows a physically connected region with oracle headroom, action reversals, and structured observable conversion.
8. **Exact-model advantage.** The demand generator and exact joint Bayesian model may be too well matched for a learned encoder to add value. Quantify model misspecification before recommending a GRU.

## Required invariants

Do not recommend relaxing these to manufacture a win:

- no future, latent regime, true parameter, tape, or seed in a deployable actor;
- learner and MPC receive the same causal history and memory rights;
- identical physical initial state, demand, shocks, resources, mass, and decisions;
- risk/environment selection is learner-blind;
- historical Q/O/R0/Q2 verdicts remain immutable;
- new hypotheses use fresh prospective contracts;
- canonical ReT remains primary, with full ledger, unresolved/lost orders, worst-product service, recovery, and tails reported transparently.

## Minimal reproduction commands

Run in a clean environment from the audited commit. Record dependency versions and elapsed time.

```bash
python3 -m pytest -q tests/test_program_t_joint_belief.py tests/test_q_r1_retained_learning.py tests/test_q_r1_cold_start_replication.py
python3 scripts/run_q_r1_cold_start_replication.py --help
python3 scripts/run_q_r1_d3_residual_bound.py --help
```

Before rerunning burned experiments, verify each runner's CLI and output path. Do not overwrite committed artifacts. Prefer an isolated temporary output directory or a new branch.

## Required audit output

Return:

1. an executive verdict with the strongest currently supported claim;
2. a reproduced-results table and any mismatches;
3. P0/P1/P2 findings with exact file and line evidence;
4. an assumption register separating bugs, arbitrary choices, defensible invariants, and unknowns;
5. a metric audit from raw order ledger to canonical ReT and every guardrail;
6. comparator-fairness and leakage audits;
7. a claim ladder: supported, near-supported, unsupported;
8. the three highest-value next experiments, with prospective estimands, controls, stop rules, expected compute, and what each failure actually closes;
9. exact code changes needed before any learner training;
10. a recommendation among: repair Q-R1 conversion, run learner-blind risk sensitivity, expand the decision contract prospectively, or publish Program Q without another training campaign.

Be adversarial. Do not accept prose, filenames, verdict labels, or hashes as proof. Trace values from code and raw rows, execute the feasible pipeline, and explicitly identify any result that cannot be reproduced.
