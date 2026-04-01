# Paper Writing Checklist

Source of truth for turning the current experimental package into a submission-ready paper.
Updated: 2026-04-01.

This checklist assumes:

- the experimental backbone is frozen around `Track A` negative evidence and `Track B` positive evidence;
- Claude is owning the visual package;
- Codex is owning manuscript text, statistics, bibliography, provenance, and repo-paper alignment.

---

## 0. Ground Truth Before Writing

- [x] `Track A` negative result is frozen as thesis-faithful evidence
- [x] `Track B` positive result is frozen as the main positive lane
- [x] matched `5D vs 7D` ablation exists and closes the action-contract question inside the benchmark family
- [x] literature positioning is mature enough to write from
- [ ] all paper-facing notes point to the same final paper claim

Final paper claim:

> In this MFSC DES benchmark, RL does not beat strong static baselines under the thesis-faithful control contract, but it does beat them after a minimal action-space extension that exposes the active downstream constraint; therefore, RL effectiveness depends critically on action-space alignment with the operational bottleneck.

---

## 1. Writing Artifacts We Still Need

### A. Manuscript core

- [ ] final title fixed
- [ ] one-line claim fixed
- [ ] contribution list `C1-C4` frozen
- [x] abstract drafted
- [x] introduction drafted
- [x] related work drafted
- [x] methodology drafted
- [x] results drafted
- [x] discussion drafted
- [x] conclusion drafted
- [ ] limitations paragraph drafted

### B. Figures and tables

- [ ] system/control figure finalized
- [ ] main comparison table finalized
- [ ] matched ablation table or figure finalized
- [ ] learning-curve figure finalized
- [ ] shift-mix or utilization figure finalized
- [ ] appendix reproducibility table finalized

### C. Reproducibility and citation package

- [x] `.bib` file created inside the repo
- [x] journal template selected and added to workflow
- [x] paper-facing manifests pin exact commit hashes instead of `HEAD`
- [ ] reproducibility appendix drafted
- [ ] file-to-claim artifact map drafted

---

## 2. Ownership Split

### Claude visual

- [ ] build the paper-facing visual package
- [ ] produce publication-grade figures
- [ ] produce styled tables if they are being rendered visually
- [ ] keep labels, metric names, and captions aligned with the frozen Track A / Track B language

### Codex

- [x] kill or supersede the stale manuscript framing in the repo
- [x] write the manuscript text backbone
- [x] compute and report formal statistics
- [x] create the bibliography package inside the repo
- [x] fix provenance and reproducibility gaps
- [ ] align README / docs / manuscript notes with the final paper claim

---

## 3. Codex Critical Path

### C1. Freeze the manuscript language

- [x] replace stale `control_v1`-centric framing in manuscript draft files
- [x] rewrite the paper around `Track A` negative evidence + `Track B` positive evidence
- [x] ensure the matched `5D vs 7D` ablation is described as supportive causal evidence, not universal proof
- [x] remove or quarantine obsolete notes that still imply the old storyline

Primary files to replace or supersede:

- `docs/manuscript_draft/section_3_3_rl_formulation.md`
- `docs/manuscript_draft/section_4_2_hybrid_results.md`
- `docs/manuscript_notes/paper_writeup_backlog.md`

### C2. Write the minimum viable manuscript

- [x] `00_title_claim_abstract.md`
- [x] `01_introduction.md`
- [x] `02_related_work.md`
- [x] `03_methodology.md`
- [x] `04_results.md`
- [x] `05_discussion.md`
- [x] `06_conclusion.md`

Recommended writing order:

1. title + one-line claim + contributions + abstract
2. methodology
3. results
4. related work
5. introduction
6. discussion + conclusion

### C3. Formal statistics

- [x] compute pairwise tests for Track B PPO vs `s2_d1.00`
- [x] compute pairwise tests for Track B PPO vs `s3_d2.00`
- [x] report effect sizes using a justified non-parametric or parametric measure
- [x] freeze wording for Track A negative controls separately from Track B positive result
- [x] add a compact stats table ready for the paper

Minimum output:

- one machine-readable summary file
- one manuscript-ready table
- one short methods paragraph describing the inferential procedure

### C4. Bibliography and citations

- [ ] create `references.bib` in-repo
- [x] create `references.bib` in-repo
- [ ] normalize citation keys
- [ ] ensure Ding et al. (2026) is entered with verified metadata
- [ ] convert the current literature notes into citeable manuscript references
- [ ] remove placeholder or ghost citations from the old draft

### C5. Provenance and reproducibility

- [x] replace `code_ref: "HEAD"` in paper-facing launchers with resolved git hashes
- [ ] verify every cited artifact bundle has a manifest, command, and summary
- [ ] add a concise reproducibility appendix tied to the frozen contracts
- [ ] document exact commands for Track A, Track B, and the matched ablation

Primary files to patch:

- `scripts/run_track_b_benchmark.py`
- `scripts/run_track_b_smoke.py`
- `scripts/run_paper_benchmark.py`

---

## 4. Deliverables By Session

### Session 1

- [ ] freeze title
- [ ] freeze one-line claim
- [ ] freeze `C1-C4`
- [x] draft abstract
- [x] create manuscript file structure

### Session 2

- [x] draft methodology
- [ ] patch provenance hashes
- [ ] create `.bib`

### Session 3

- [ ] compute formal statistics
- [x] draft results around the frozen figures/tables

### Session 4

- [x] draft related work
- [x] draft introduction

### Session 5

- [x] draft discussion
- [x] draft conclusion
- [ ] assemble reproducibility appendix

---

## 5. Stop Rules

Do not spend more time on:

- additional reward redesign for the main paper
- new primary experiments unless strictly needed for statistical replication
- speculative architecture framing (`DKANA`, `KAN`, `GNN`) as a core paper contribution
- “first RL for SCM” novelty claims
- literature planning documents that do not end in manuscript text

---

## 6. Definition of Done

The paper package is ready for Garrido review only when all of the following are true:

- [ ] manuscript sections exist in prose, not notes
- [ ] visuals are frozen
- [ ] tables are frozen
- [ ] statistics are frozen
- [ ] bibliography is in-repo and clean
- [ ] provenance is pinned to exact commits
- [ ] reproducibility appendix exists
- [ ] README and source-of-truth docs match the manuscript claim

At that point, the repo has crossed from `experiment-complete` to `paper-ready`.
