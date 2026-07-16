# Program O — terminal corrective outcome certificate

**Original date:** 2026-07-15

**Corrected through:** 2026-07-16
**Status:** `STOP_PROGRAM_O_AFTER_CORRECTIVE_VALIDATION`

This certificate supersedes the earlier `ACTIVE_FIXED_CLOCK_PHYSICAL_PREFLIGHT`
version. Program O completed its prospective corrective classical-control
validation and is closed. The result establishes a reproducible mean canonical
ReT advantage for the frozen belief-MPC, but it does not establish the project's
joint safe classical-H_obs contract because simultaneous CVaR10 non-inferiority
failed in two of the three frozen cells.

## Evidence chain

| Stage | Result | Claim boundary |
|---|---:|---|
| Full-DES safe perfect-information ceiling | 0.15151; simultaneous LCB95 0.11562 | H_PI only |
| Fully fungible null | exactly 0 | causal mechanism control |
| Corrective rho75/share90 ΔReT | 0.09852; simultaneous LCB95 0.06595 | mean classical advantage |
| Corrective rho90/share75 ΔReT | 0.07347; simultaneous LCB95 0.04303 | mean classical advantage |
| Corrective rho90/share90 ΔReT | 0.09974; simultaneous LCB95 0.05860 | mean classical advantage |
| Information placebos | all 27 pass; minimum simultaneous LCB95 0.00716 | observable/state-dependent signal |
| Physical replay | 1,451 episodes; zero failures; one resource vector | equal scheduled resources |
| rho75/share90 ΔCVaR10 | 0.03502; simultaneous LCB95 **-0.00858** | fails joint safety |
| rho90/share75 ΔCVaR10 | 0.01954; simultaneous LCB95 **-0.01551** | fails joint safety |

The point estimates favor the MPC, but the preregistered gate required every
simultaneous guardrail lower bound to be non-negative. The two negative CVaR10
bounds therefore determine the verdict.

## Scientific interpretation

Program O supports the bounded statement that operational state feedback can
improve mean canonical ReT over a development-frozen full-horizon open-loop
comparator under the two-product researcher extension. The advantage survived
information placebos, resource matching, action-trajectory checks, state
counterfactuals, and physical replay.

It does **not** support any of the following:

- joint safe classical H_obs under the frozen contract;
- a learner-entry decision;
- Paper 2 learned value;
- Paper 3 retained learning.

The corrective contract forbids another controller, cell deletion, metric,
physics, comparator, threshold, or guardrail change. A new learned policy cannot
be introduced as a rescue of this result.

## Custody

- executed commit: `7a05d448f4d788a19385a0c65c842b8663ed8391`;
- scientific source commit: `14b559cc0c88b8c186673077403d7c4253337cae`;
- result SHA-256: `3d3ff5b37510a993582cfc82b5414868da5cec2f99eda5da1df58013af389877`;
- retrieved manifest SHA-256: `635f855228af404165484f8ca1732cd13114a01c16e8ce5e996efdebbe8b938e`;
- seed block: `7430001–7430048`, opened once and burned;
- independent audit:
  `results/program_o/fixed_clock_hobs_corrective_validation_v1/independent_audit_v1.json`.

## Final claim boundary

`classical_primary_ret_advantage_confirmed: true` ·
`safe_joint_h_obs_contract_confirmed: false` ·
`program_o_closed: true` ·
`second_rescue_forbidden: true` ·
`learner_authorized: false` ·
`paper2_confirmed: false` ·
`paper3_authorized: false`.
