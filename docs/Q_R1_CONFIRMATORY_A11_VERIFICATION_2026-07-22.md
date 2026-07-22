# Q-R1 successor confirmatory — independent A11 verification (2026-07-22)

Canonical confirmatory ran on VPS ovh-agent-lab (commit 983c197, ~9.3h). I pulled the 4 raw shard
result.json from the VPS and ran the FROZEN merge + adjudicate scripts on my machine.

## Integrity (verified)
4 shards: claim=PROSPECTIVE_CONFIRMATION, commit 983c1971c1ae, dirty=False, contract_sha ac9ddd46d0e2,
planner ret_proxy_scenario_h4_p16_stratified, 5184 rows each; exact frozen coverage 7572001-7572032;
no overwrites. Merge: 20736 rows, single commit, no duplicates.

## VERDICT (independent, frozen scripts): STOP_REPAIRED_Q_R1_NO_RETAINED_INFORMATION_PASS
learner_training_authorized = False. Both estimands (prefix_natural_replanning, sustained_control) identical.

The retained effect is REAL, LARGE, and passes magnitude + dose:
- early_ret_complete_cohort κ=.90 **+0.06563 (LCB95 +0.05244)**; visible +0.06043 (LCB +0.04847) —
  both far above the 0.01 SESOI, LCB>0 (visible_pass & complete_pass = True).
- dose PERFECT: κ.90 +0.06563 > κ.75 +0.03408 > iid +0.00000 (exact).
- anti-oracle: wrong_posterior -0.10158.

Fails two gates:
1. mechanism_pass=False — SOLELY the delayed_posterior placebo +0.00591 (>0; rule = all placebos ≤0).
   shuffled -0.00425 (ok), wrong -0.10158 (ok). A minor stale-timing leak.
2. worst_product_lcb_ge_minus_0p02=False — worst-product delta mean -0.02161, CI95 [-0.0365, -0.0077];
   LCB -0.0365 < -0.02. Unresolved mean +0.0938 (max +20), lost 0. A real SERVICE trade-off: the
   retained arm lifts mean ReT partly by favoring the majority product, degrading the minority fill.

## Interpretation
Retained decision knowledge CAUSALLY improves mean cold-start ReT prospectively — large, dose-responsive,
anti-oracle-correct = Garrido cumulative learning CONFIRMED on fresh sealed seeds. But NOT jointly-safe
(worst-product) and a minor delayed-timing leak. Same shape as Program O (mean conversion real,
jointly-safe not established). Verdict correctly STOPS; no learner. This independent run of the frozen
scripts on the VPS raw outputs is the canonical verified verdict; Codex's adjudication was not yet pushed
at verification time and should match.
