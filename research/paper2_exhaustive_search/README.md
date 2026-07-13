# Paper 2 adaptive-headroom exhaustive search

## Current scientific status

`OPEN_ACTIVE_BOUND_REQUIRED` — Paper 2 is not confirmed and Paper 3 is not authorized.

The only numerically active family is the integrated M/T/R bottleneck contract. Three retrospective null tapes have exact equality only for selected terminal metrics and exogenous hashes; this is not an action/full-state policy-equivalence proof. Its current evidence is insufficient: the full-horizon open-loop frontier contains 11,184,811 feasible calendars, the resource-restricted PI ceiling is not computed, and reserve ration issue/replenishment semantics are unresolved. The frozen one-team-week budget itself is equal; M/T/R are allocation destinations.

An exact tape-specific physical-effect quotient now reduces the required DES executions from 2,013,265,980 to 88,684,583 (22.70×), or about 32.68 serial CPU-days at the latest measured full-run rate. This is an executable acceleration, not a frontier or H_PI result. The row therefore remains `active_for_bound`, not falsified, promotable or terminal.

All other new mechanism families are either formally reduced, quantitatively below the practical gate on a versioned contract, or blocked on explicit domain facts. Because disclosed researcher extensions are open-ended, this package does not claim a universal boundary over every imaginable future extension.

## Artifact index

- `phase0_failure_taxonomy.json`: A–K3 plus bottleneck failure reconstruction.
- `source_reconstruction.md` and `source_extraction_index.json`: thesis, Garrido paper and v0 extraction provenance.
- `primary_source_literature_review.md`: mechanism-oriented primary-source review.
- `approach_registry.json`: live family states, ceilings and state-change evidence.
- `candidate_intervention_ledger.json`: operational meaning, provenance, conservation, null regime and claim limits.
- `decision_right_catalog_coverage.json`: one-to-one routing audit of all 32 catalogued Op3--Op13 decision rights; 13 are exact current-kernel zero/action-equivalent and no overlooked source-native executable family remains.
- `global_headroom_sensitivity_design_and_results.md`: frozen selection logic and all available screen results.
- `prelearner_contract.json`: gates that currently prohibit learner training.
- `comparator_completeness_audit.md`: comparator coverage and missing bounds.
- `action_trajectory_audit.md`: K3 reproduction and bottleneck replacement controls.
- `results/paper2_bottleneck/effect_quotient_audit.json`: exact state-effect quotient counts and collision validation; no H_PI result yet.
- `contracts/paper2_bottleneck_full_horizon_bound_v1.json`: frozen retrospective protocol for the exact 11,184,811-calendar M/T/R frontier; seed 1110001 is excluded from final inference because it was used to develop the acceleration.
- `results/paper2_bottleneck/loose_canonical_upper_bound.json`: rigorous locked-tape bound showing why the un-clipped sparse ReT ledger defeats a cheap affected-order ceiling.
- `results/paper2_bottleneck/signal_mapping_audit.json`: exact 27-policy memoryless signal screen; calibration selects constant M.
- `boundary_scope_proof.md` and `boundary_verification.json`: machine-checked scope and nonterminal verdict.
- `garrido_decision_questions.md`: minimal falsifiable domain questions that could reopen blocked mechanisms.
- `paper2_paper3_status.md` and `paper_facing_claims_table.md`: paper-facing claim boundaries.
- `reproducibility_manifest.json`: inputs, hashes, environment, commands and tape status.
- `rejected_concurrent_artifacts.md`: explicit exclusion of a locally generated but mathematically invalid constants-only “PI ceiling.”
- `concurrent_boundary_commit_audit.json`: machine-readable supersession of commit `a91890bf`'s terminal boundary claim; the commit is useful exploratory evidence but fails Return B while M/T/R remains unbounded.
- `seed_burn_ledger_correction.json`: fail-closed burn record, including conservative exclusion of the unmanifested 7.2M atlas block opened by `a91890bf`.
- `results/paper2_search/voi_ceiling_atlas_corrective_audit.json`: exact replay of all 64 concurrent-atlas cells. It reproduces all stored full-order statistics, proves that both source-positive cells fail lost-order non-inferiority, and exposes an unguarded sparse-visible `HOLD^4` degeneracy plus fixed-sequence R22 score inertness. The raw 64/64 zero is explicitly not a project H_PI ceiling.
- `toy_screen_adversarial_audit.md`: why concurrent F8/F10 proxy toys are negative exploratory diagnostics, not canonical evidence.
- `PROGRAM_L_ROUTE_RECOURSE_VERDICT.md` and `results/paper2_search/program_l_corrective_audit.json`: corrective rejection of the concurrent route-recourse terminal claim. The tested heuristic is a development null; the introduced family remains domain-blocked and unbounded.

Run the machine checks with:

```bash
PYTHONPATH=. .venv/bin/python scripts/verify_paper2_exhaustion.py
PYTHONPATH=. .venv/bin/python scripts/audit_decision_right_catalog_coverage.py
PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_paper2_exhaustive_search_registry.py tests/test_paper2_bottleneck.py tests/test_replenish_ret.py
```
