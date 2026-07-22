# Submission A execution board — eight-week ceiling

Start: 2026-07-22. Binding submission ceiling: 2026-09-16.

This board controls publication work only. Q-R1 cannot change the Program Q
claim, enter this manuscript, or delay the submission ceiling.

| Due | Deliverable | Owner | State | Acceptance evidence |
|---|---|---|---|---|
| 2026-07-22 | Frozen scientific package, manuscript, six tables, four figures and review PDF | Thom | DONE | `source_of_truth.json`; `generated/generated_files.sha256`; `output/pdf/submission_a_program_q_draft.pdf` |
| 2026-07-22 | Hardware-specific action-latency benchmark and removal of unmeasured amortization claim | Thom | DONE | `results/program_q/latency_benchmark_v1/result.json`; commit `a543f65` |
| 2026-07-22 | Two frozen confirmatory falsification probes: action-trajectory feedback and modal/phase-only/frequency-matched replacements | Thom | DONE | `results/program_q/confirmation_v1_20260718/artifacts/confirmation/evaluation/result.json`: `trajectory_audits`, `replacement_controls`, and both corresponding integrity gates |
| 2026-07-29 | Author order, affiliations, corresponding author and CRediT approved | Thom + all authors | BLOCKED_HUMAN | Signed/dated update to `AUTHORSHIP_AND_PORTFOLIO.md` |
| 2026-07-29 | Garrido face validation: ReT, researcher extension, worst-product/unresolved and closed-loop wording | Garrido | BLOCKED_HUMAN | Written response to `GARRIDO_FACE_VALIDATION_REQUEST.md` |
| 2026-08-05 | Security and public-release review of military-model wording and assets | Thom + designated reviewer | TODO | Signed checklist entry plus any redaction commit |
| 2026-08-05 | Funding, acknowledgments, conflicts and AI-assistance disclosure | Thom + all authors | BLOCKED_HUMAN | Final manuscript declarations and metadata |
| 2026-08-19 | Independent scientific and language review with every actionable comment adjudicated | Independent reviewer + Thom | TODO | Review log referencing manuscript lines and resolution commits |
| 2026-08-26 | Clean-room reproduction from pinned environment | Reproducibility reviewer | TODO | Fresh clone log, test report, generated-file hash match and PDF build receipt |
| 2026-09-02 | Immutable release/tag and archival evidence bundle | Thom | TODO | Git tag, release URL, bundle SHA-256 and DOI |
| 2026-09-09 | C&IE formatting, cover letter and submission metadata finalized | Thom + corresponding author | TODO | Placeholder-free files and author approval |
| 2026-09-16 | Submit to Computers & Industrial Engineering | Corresponding author | TODO | Editorial-system submission receipt |

## Binding claim boundary

- Supported: RecurrentPPO beats the complete 65,536-calendar open-loop frontier
  in the specified contract and is practically equivalent to the reselected
  structured feedback family.
- Not supported: neural premium, product-level safety, accumulated learning,
  active-risk improvement, or superiority over all possible static or dynamic
  policies.
- Latency is secondary and hardware-specific. The measured structured family
  was faster; no neural compute advantage is claimed.
- The two probes above are checks inside the single sealed Program Q
  confirmation. They are not two additional experiments or independent
  replications, and no exploratory probe is promoted retrospectively.

## Escalation rule

Any item still blocked seven days before its due date is escalated to the
corresponding author. Q-R1, comparator v2, new risks and new learners are never
valid reasons to move the 2026-09-16 ceiling.
