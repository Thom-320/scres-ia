# Submission PDF

`submission_a_program_q_draft.pdf` is the rendered manuscript generated from
`main.tex` after rebuilding all evidence tables and figures from the frozen
Program Q artifacts. It is a review draft, not a submitted version: author
metadata, Garrido face validation, archival DOI, and journal-format QA remain
open.

The archived PDF is compiled by `scripts/build_submission_a_pdf.py` with a
fixed source-date epoch. Rebuilding twice from one commit must produce the same
SHA-256 digest.
