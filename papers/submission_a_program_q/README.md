# Submission A - Program Q

This directory is the only manuscript source for Submission A.

Scientific base: `f2dfe356c179bd16f4b89b26e8ed3b19d69f5a71`.

Binding interpretation:

- feedback value over the complete 65,536-calendar static frontier: supported;
- practical equivalence to the best tested structured controller: supported;
- neural premium: not supported;
- worst-product or deployment safety: not established;
- retained learning and Garrido-native active-risk learning: outside this paper.

## Deterministic build

The compact frontier file was derived once from the frozen 144-file calibration
matrix after verifying every source hash:

```bash
python scripts/extract_program_q_static_frontier.py \
  --calibration-root /path/to/calibration_run \
  --output papers/submission_a_program_q/data/static_frontier_summary.npz \
  --manifest-output papers/submission_a_program_q/data/static_frontier_summary_manifest.json
```

Generate the source of truth, six tables, and four figures:

```bash
python scripts/build_submission_a_program_q.py
```

Compile `main.tex` with the repository's LaTeX build workflow. The generator
verifies the frozen evidence hashes before producing outputs; it does not rerun
the simulator or change a scientific verdict.

## Submission blockers, not scientific blockers

- confirm final author order, affiliations, and corresponding author;
- obtain Garrido's written face validation of the ReT interpretation and the
  boundary between the thesis model and the researcher-defined extension;
- archive the release bundle and insert its DOI;
- perform journal-format and language review on the final anonymous PDF.
