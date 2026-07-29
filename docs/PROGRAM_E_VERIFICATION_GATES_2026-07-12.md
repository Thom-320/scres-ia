# Program E verification gates (frozen by verifier BEFORE the build, 2026-07-12)

Role: Codex builds Program E (`docs/PROGRAM_E_ORACLE_TO_POLICY_PREREGISTRATION_2026-07-12.md`,
contract `program_e_policy_realizability_v1.json`) end-to-end; this freezes the checks the
verifier will run and the pass/fail lines, so the audit cannot move goalposts (V4→V5 lesson).
The prereg is endorsed as sound; these are the independent machine audits per stage.

## Frozen audit checks
- **No-privileged-observation.** Assert the policy observation vector contains none of:
  future risk/onset/duration, future demand, latent regime, oracle labels, any
  retrospective/future outcome. Fail-closed test on the env.
- **Action mask is real (not silent-HOLD).** MaskablePPO must receive a mask that forbids
  DISPATCH when convoy-away / route-down / empty-staging; verify an invalid DISPATCH is
  masked, NOT silently remapped to HOLD (which pollutes gradients/entropy). Machine check.
- **Comparator frozen on VALIDATION, not test.** The deterministic best static AND the
  convex-mixture weights are selected once on validation within each learner seed's
  departures+unavailable-hours envelope, then frozen; assert they are NOT reselected on the
  virgin tapes (that would make the baseline a retrospective oracle).
- **Tape disjointness + hashing.** 900001–900080 / 910001–910020 / 920001–920060 are
  disjoint from each other and from all DRA-2/2b/D1 universes; hashes recorded; virgin
  opened once, only after E3 passes and weights/tree/heuristic/mixture/analysis are frozen.
- **PPO-once discipline.** One config, 10 seeds, no architecture/reward sweep; if E1
  baselines fail, PPO is DIAGNOSTIC-ONLY and does not auto-open the virgin test.
- **Resource envelope executed, not prose.** η and ΔReT computed against the enveloped
  comparator; verify a "win" is not bought with more departures/unavailable-hours.
- **Two claim levels honored.** Learning claim (ΔReT CI95>0, η≥0.50, service non-inferior,
  lost not up, ≥8/10 seeds, ≥70% tapes, tail ok) vs STRONG managerial claim (+ service ≥5%).
  Verify the 5% is NOT lowered and NOT conflated with the learning claim.
- **Inference by seed×tape.** Two-way cluster bootstrap; CRN-paired; per-family; CVaR95.
- **Reported whether PPO wins/ties/loses.** No outcome is relabeled a failure; the paper
  is written regardless. persistent/reset (L(e−1)) only after a confirmed OOS PPO win.

Standing: DRA-1, DRA-2, DRA-2b stay closed. Program E is the last experimental campaign
before the manuscript freeze. The corrected closing claim (from my overclaim) stands:
"DRA-2b did not authorize PPO under its rule; it did NOT prove RL cannot convert the
headroom — Program E measures H_obs / η."
