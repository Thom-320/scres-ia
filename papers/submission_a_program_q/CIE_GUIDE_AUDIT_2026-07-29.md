# C&IE guide-for-authors audit — 2026-07-29 UTC

Authoritative page:
<https://www.sciencedirect.com/journal/computers-and-industrial-engineering/publish/guide-for-authors>

This audit covers submission mechanics only. It does not change Program Q
science or authorize submission without the coauthors.

| Current C&IE requirement | Evidence/status |
|---|---|
| Double-anonymized review | Anonymous `main.tex`; separate `TITLE_PAGE.tex` prepared. |
| Separate title page with authors, affiliations, acknowledgements, declaration and corresponding-author contact | Template prepared; human fields remain `PENDING`. |
| Editable LaTeX sources | `main.tex`, `references.bib`, editable table sources and separate figures are present. |
| Abstract no longer than 250 words | Elsevier-class abstract: 215 words by repository audit. |
| 1–7 English keywords | Six keywords. |
| 3–5 separate highlights, each at most 85 characters | Five highlights in `HIGHLIGHTS.txt`; maximum line content is below 85 characters. |
| Graphical abstract | Encouraged, not required; intentionally omitted to avoid adding a nonessential figure before submission. |
| Separate generative-AI declaration before references | Included in `main.tex`; author-confirmation wording in `GENERATIVE_AI_DISCLOSURE_DRAFT.md`. |
| Competing-interest declaration tool output uploaded as `.doc/.docx` | Human-blocked; title-page/checklist placeholders identify the required output. |
| Funding sources and sponsor role disclosed | Human-blocked. |
| Research data deposited and cited, or non-sharing explanation | RC1 GitHub custody release exists; final archival DOI and anonymous-review routing remain pending. |
| Required data-availability statement | Present in the anonymized manuscript; final DOI remains pending. |
| Every reference cited in both directions | Must be rechecked on the final tagged source after approved edits. |
| Separate artwork with captions | Four separate PNG figures and manuscript captions are present. |
| One corresponding author with full email, postal address and phone | Human-blocked. |
| Spelling, grammar and independent language review | Review packet prepared; independent reviewer pending. |

## Double-anonymization boundary

The administrative files in this directory contain author names, email-message
IDs, branch names, and a public GitHub URL. They must not be uploaded as
anonymous supplementary material. The anonymous submission bundle is built by
`scripts/build_submission_a_cie_review_bundle.py`, which includes only the
manuscript source, bibliography, highlights, tables, and figures and fails if
known identifying tokens appear in its text files.

The public RC1 release is custody evidence, not the anonymous-review upload.
The corresponding author must decide with the journal whether the final DOI is
provided to editors only, via an anonymous repository link, or made visible
after review. That decision cannot be inferred from the repository.

## Current official journal snapshot

- CiteScore: 13.5
- Impact Factor: 7.3
- submission to first decision: 4 days
- submission to decision after review: 90 days
- submission to acceptance: 208 days
- acceptance to online publication: 7 days

These descriptive values can change and are not scientific evidence.
