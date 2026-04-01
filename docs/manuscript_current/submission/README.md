# Submission Package

This directory contains the assembly outputs for the current manuscript.

Files:

- `manuscript_full.md`: concatenated manuscript source for rapid review
- `elsevier/main.tex`: LaTeX submission draft
- `elsevier/sections/*.tex`: Pandoc-generated section fragments
- `elsevier/references.bib`: bibliography snapshot used by the submission draft

Notes:

- `main.tex` uses `elsarticle.cls` when it is available in the local TeX installation.
- If `elsarticle.cls` is unavailable, the draft falls back to `article` so the manuscript still compiles for internal review.
- In-text citations remain plain-text prose at this stage; `\nocite{*}` is used so the bibliography prints for review builds.
