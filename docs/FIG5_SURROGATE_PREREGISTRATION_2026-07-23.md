# Fig-5 surrogate study — preregistered endpoints (before any training)

**Date:** 2026-07-23. **Status:** endpoints frozen before the first fit. Methodological study on
burned data; no seed custody; no claim beyond search-cost comparisons.

**Question (Garrido 2024, Fig 5, executed):** can a supervised surrogate (drivers → ReT) replace
the costly exhaustive configuration search — graded against EXACT answer keys?

**Families (exactly the paper's three):** backprop MLP (scaffold `scripts/run_bc_de_sweetspot.py:89-120`);
KAN (scale `scripts/run_kan_scres_demo.py`, pykan 0.2.8, LBFGS); simulation-optimization
(`supply_chain/program_u_static_search.py` `cma_es_search`/`bayesian_optimization_search`/random at
matched eval budgets).

**Datasets:** (D1) 48 burned campaigns × 65,536 calendars regenerated via
`simulate_full_des_frontier(include_q_r1_metrics=True)` (~2.3 s/campaign), features = weekly counts,
κ, retained prior, campaign index, initial regime; label = `early_ret_complete_cohort`.
(D2) thesis-native replication: the 3,096-row dispatch-screen table (posture, profile, seed → ret_excel).

**Frozen endpoints (all exact by construction):**
1. **Regret** of surrogate-argmax vs the true optimum, per campaign (mean, median, q95).
2. **Top-k recall** (k = 10, 100) of the true top-k inside surrogate top-k.
3. **Sample efficiency**: simulator evals needed to reach regret ≤ ε ∈ {0.005, 0.01} vs the
   65,536 exhaustive baseline — surrogate-guided (fit on n, propose argmax, verify, refit) vs
   CMA-ES vs BO vs random, identical eval budgets n ∈ {64, 256, 1024, 4096}.
4. **Transfer**: train on 36 campaigns, test regret on 12 held-out (split by history root).
5. **R² comparison** MLP vs KAN vs linear on identical splits (replicating the 147-cell demo at scale).

**Verdict language:** the surrogate competes against SEARCH COST, never against the structured
controller. No neural-premium claim; the exact answer keys are the referee.

**Self-checks:** dataset spot-replays reproduce frontier values at 1e-9; split by root (no leakage);
seeds for train/test splits fixed at 20260723.
