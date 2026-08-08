# Paper 2 release scope

This file is the executable boundary for the Paper 2 release gate. It is subordinate to
`papers/PORTFOLIO_CLAIM_LOCK.json` for ownership, `papers/paper2/claim_lock.json` for wording and
`research/seed_custody_registry.json` for seed custody.

## In scope

- `results/grid_transfer_confirmation_v2/result.json` and its base/extended surface caches.
- The downstream transfer chain and its six falsifiers.
- `results/retention_contrasts/result.json` as a development/replay reanalysis.
- The six thesis-derived comparative panels, Figure 5 identity, the matched KAN/MLP search
  comparison, and the separately contracted KAN latent result.
- `results/demand_process/result.json` and the bounded seasonal-demand sensitivity, always with
  their stated scope and failed falsifiers where applicable.
- Source closure declared by the cited cache manifests, including `supply_chain/supply_chain.py`.

## Required release checks

1. Certificate A and B from `scripts/verify_frozen_path_equivalence_v2.py` are sealed.
2. `tools/verify_sealed_payload.py` reproduces the seal from a clean clone.
3. Every cited number resolves to one row in `papers/paper2/claim_lock.json`.
4. No forbidden wording or retracted figure appears in the manuscript or compiled bundle.
5. The release marker is green; exclusions outside this closure require a tracked issue and
   `xfail(strict=True)`, never a silent skip.

## Explicit exclusions

The following are not Paper 2 claims: organisational learning, within-episode adaptive control,
DES validation, Simulink reproduction, a universal UCB superiority claim, a neural premium, a
policy-level carrier claim, or a claim about demand processes outside the evaluated contract.

The historical retained-learning Paper 3 and the negative headroom dossier remain separate
governance objects. Their status does not authorize new Paper 2 seeds or experiments.
