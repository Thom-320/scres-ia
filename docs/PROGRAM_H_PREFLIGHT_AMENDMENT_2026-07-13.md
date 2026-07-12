# Program H preflight amendment — frozen before 1060001

This amendment corrects two specification details without changing Program G or selecting any
parameter from Program H outcomes.

1. The inherited 12-cell Program G terminal region used `r22_weekly_prob=0.05`; Program H uses
   that exact value. The contract phrase "R22 off" was stale and is superseded.
2. A numerical QMDP approximation is reported as a diagnostic, not automatically as a rigorous
   upper bound. The rigorous information-relaxation ceiling is the exact 81-sequence tape oracle,
   which observes the complete tape. Only a mathematically verified relaxation may be labeled an
   upper bound on the best observable policy.

Frozen O0 algorithms before data:

- exact Bayes marginal over tempo using the frozen signal likelihood; the augmented dwell phase
  is retained conceptually, but over a four-week episode and minimum dwell four the realized tempo
  is constant within the horizon;
- regret-weighted fitted-Q policy: one ExtraTrees regressor per action, 300 trees, minimum leaf 5,
  fixed seed 20260713, trained on exact counterfactual terminal ReT for all observable prefixes;
- belief-MPC: two-week expected-demand lookahead with exact action enumeration and Bayes posterior;
- belief rollout: remaining-horizon enumeration under posterior-mean demand;
- no hyperparameter or architecture sweep.

If all tested belief-aware policies fail the locked gate while the rigorous full-information
ceiling remains material, the allowed conclusion is policy failure with unresolved formal
information sufficiency—not a false proof that `J*_obs <= J_static`. Program H still terminates
because it is the final computational extension.
