# Repository source of truth

**Effective date:** 2026-07-17
**Scientific status:** `CURRENT_IMPLEMENTED_PORTFOLIO_EXHAUSTED_NO_LEARNER_AUTHORIZED`

This document is the repository-level claim boundary. It supersedes the former
Track-A/Track-B narrative that presented `ReT_seq_v1` and stress-regime gains as
the current paper contribution.

## Binding headline

No tested learner has established deployable adaptive value under the current
contracts. No learner is authorized. Paper 2 is not confirmed, and Paper 3
remains blocked.

The strongest implemented mechanism is Program O, a nonfungible product-mix
extension of the full DES. It established a large, custody-verified
full-information ceiling, and its corrective observable validation established
a reproducible mean canonical-ReT advantage with genuine state-dependent
actions. It did **not** satisfy the frozen joint tail-safety contract.

The terminal label is:

`STOP_PROGRAM_O_AFTER_CORRECTIVE_VALIDATION`

## Program O evidence

### Full-information ceiling

The full-DES translation established:

- safe `H_PI = 0.1515137892`;
- simultaneous safe LCB95 `= 0.1156159089`;
- exact fungible-null `H_PI = 0`;
- 25,177 direct horizon-8 parity episodes;
- equal production and reserved downstream resources.

This is a physical opportunity ceiling, not observable adaptation and not a
learner result. The authoritative compact record is
`results/program_o/full_des_hpi_translation_v1/validation_custody_verdict_v1.json`.

### Observable corrective validation

The corrective validation used fresh seeds `7430001-7430048`, frozen
development-selected full-frontier comparators, and studentized one-sided
max-t inference. Mean canonical-ReT passed in all connected cells:

| Cell | Mean delta ReT | Simultaneous LCB95 | Favorable tapes |
|---|---:|---:|---:|
| rho75/share90 | 0.09852 | 0.06595 | 44/48 |
| rho90/share75 | 0.07347 | 0.04303 | 42/48 |
| rho90/share90 | 0.09974 | 0.05860 | 46/48 |

All 27 information-placebo contrasts passed. Physical equality passed across
1,451 replays with zero failures. Action trajectories and state
counterfactuals passed in every cell.

The frozen joint contract nevertheless failed because simultaneous CVaR10
non-inferiority did not clear zero in two cells:

- rho75/share90: LCB95 `-0.0085776`;
- rho90/share75: LCB95 `-0.0155069`.

All other guardrails passed. The point estimates favored the controller, but
the preregistration required every guardrail LCB to be non-negative. The
contract forbids a second rescue, cell deletion, threshold relaxation,
controller change, or metric change.

The authoritative records are:

- `docs/PROGRAM_O_CORRECTIVE_HOBS_VALIDATION_VERDICT_2026-07-15.md`;
- `results/program_o/fixed_clock_hobs_corrective_validation_v1/independent_audit_v1.json`.

### Why a corrective run existed

The first prospective block, seeds `7420049-7420096`, was opened exactly once.
Its automatic adjudication was retracted because the executor reselected the
comparator on validation tapes and used an invalid unstandardized simultaneous
critical value across heterogeneous estimands. The burned trajectories and
custody remained valid. A single corrective validation was authorized to test
the same scientific contract with those adjudication defects repaired.

The first-run records are retained as historical evidence, not as the current
terminal verdict:

- `docs/PROGRAM_O_FIXED_CLOCK_HOBS_VALIDATION_VERDICT_2026-07-15.md`;
- `results/program_o/fixed_clock_hobs_validation_v1/independent_audit_v1.json`.

## Current claim boundary

It is accurate to claim:

- material full-information product-mix headroom in the full DES;
- exact collapse of that headroom under the fungible null;
- observable, state-dependent mean canonical-ReT improvement over the frozen
  full open-loop comparator;
- failure to establish joint tail-safe classical `H_obs` under the frozen
  familywise contract.

It is not accurate to claim:

- safe joint `H_obs > 0` under the project contract;
- learned adaptive superiority;
- Paper 2 confirmation;
- Paper 3 authorization;
- a global impossibility theorem outside the implemented and preregistered
  portfolio.

The portfolio-level machine-readable boundary is
`research/paper2_exhaustive_search/paper2_current_boundary_certificate_20260716.json`.

## Guardrail taxonomy and the conditional ReT-centred route (2026-07-17)

`docs/GUARDRAIL_TAXONOMY_AUDIT_2026-07-17.md` classifies every terminal closure: Class A
(canonical-construct failure), Class B (identification guards — resources, placebos, anti-shed —
non-negotiable under any construct reading), Class C (deployability preferences beyond the source
construct, e.g. tail/CVaR non-inferiority). **CVaR appears as a closure reason only in the
Program O corrective validation — exactly one candidate reopening.** The route to a learner is
strictly sequential and prospectively gated:

1. Garrido's written, dated answer to **M2** (operational acceptance criterion: mean canonical
   ReT vs binding worst-theatre/worst-campaign requirements; two-sided wording, frozen
   interpretation rule);
2. the frozen **`contracts/cvar_gate_instrument_audit_v1.json`** (power / oracle control /
   trivial-equivalence control of the zero-margin gate; burned tapes only; report-only margins);
3. adjudication by the **independent auditor** — never by the implementing side;
4. only then a new frozen learner contract (`program_o_ret_learner_v1`) on fresh sealed tapes,
   with Class-B guards intact and CVaR as secondary reporting.

`STOP_PROGRAM_O_AFTER_CORRECTIVE_VALIDATION` remains immutable under every branch of this route.

## Metric and domain status

Program O used `ret_excel_request_snapshot_v2` as its frozen canonical primary
endpoint. Garrido face validation remains necessary to identify the intended
same-timestamp ordering of `sumBt`/`sumUt` and to establish how representative
the nonfungible product classes are of the MFSC. Those answers may refine
construct validity or justify a genuinely new preregistered contract. They do
not retrospectively reopen Program O.

## Historical lanes

Track A, Track B, Track B-P, Track C, Programs D through N, and older
`ReT_seq_v1` benchmark bundles remain useful provenance and bounded evidence.
They are not the current positive paper claim. Earlier gains must retain their
original comparator and contract qualifiers.

## Publication and execution authorization

- Current defensible paper route: a boundary paper separating physical
  headroom, observable mean conversion, and joint tail-safe deployability.
- New simulation: only after a genuinely new mechanism is justified and
  preregistered with new physics, observations, comparators, and seeds.
- Learner: not authorized under current contracts.
- Paper 3: blocked until a future contract establishes learned adaptive value.

## Provenance scope of this reconciliation

This small reconciliation changes the remote claim state and publishes compact
audited summaries with immutable hashes. It intentionally does not add the raw
calendar matrices or large custody bundles to Git history. Those remain
external custody artifacts identified by SHA-256 in the included audit files.
Accordingly, this PR makes the terminal claim boundary reviewable from GitHub;
it is not by itself a complete raw-data replication package.

## Document hierarchy

When documents disagree, use:

1. this file;
2. `paper2_current_boundary_certificate_20260716.json`;
3. the Program O corrective independent audit and verdict;
4. the full-DES HPI custody verdict;
5. dated historical verdicts and older claim registries.
