# References checklist — the ~10 canonical citations missing from `references.bib`

**Date of verification:** 2026-08-24 · **Verifier:** Claude (this session) ·
**Target bib:** `docs/manuscript_current/submission/elsevier/references.bib` (37 entries)

Every DOI / arXiv ID below was resolved **before** being listed. Nothing here is transcribed from
memory. The rule applied: an entry is `VERIFIED` only if a registry API returned the record and its
**title, authors, year, container, volume and pages matched** what the entry claims; anything that
did not resolve, or that resolved to metadata differing from the claim, is `POR_VERIFICAR` with the
discrepancy stated.

## Which registry, and why it matters

- **Crossref** (`api.crossref.org`) registers journal articles and most conference proceedings.
- **DataCite** (`api.datacite.org`) registers arXiv's `10.48550/arXiv.*` DOIs. **Crossref returns
  HTTP 404 for all of them by design** — that 404 is not evidence the DOI is wrong. All three arXiv
  DOIs below returned 404 from Crossref and 200 with full matching metadata from DataCite.
- **export.arxiv.org** was queried for the arXiv entries to recover the venue from the authors'
  own `arxiv:comment` field. It rate-limited (HTTP 429) on two of three; those two are marked
  `venue POR_VERIFICAR` while their DOIs remain firm on DataCite.

## Status of the ten

| # | Entry | In bib? | DOI / ID | Verified against | Status |
|---|---|---|---|---|---|
| 1 | Ng, Harada & Russell 1999 | **already present** (`ng1999`) | none | Crossref bibliographic search | `NO_DOI` — see note |
| 2 | Wiewiora 2003 | missing | `10.1613/jair.1190` | Crossref 200 | **VERIFIED** |
| 3 | Yu et al. 2022 (MAPPO) | missing | `10.52202/068431-1787` + `10.48550/arXiv.2103.01955` | Crossref 200 / DataCite 200 | **VERIFIED** |
| 4 | Rashid et al. 2018 (QMIX) | missing | `10.48550/arXiv.1803.11485` | DataCite 200 | **VERIFIED** (venue `POR_VERIFICAR`) |
| 5 | Altman 1999 (CMDP) | missing | `10.1201/9781315140223` | Crossref 200 | `POR_VERIFICAR` — year mismatch, see note |
| 6 | Achiam et al. 2017 (CPO) | missing | `10.48550/arXiv.1705.10528` | DataCite 200 + arXiv API 200 | **VERIFIED** |
| 7 | Clark & Scarf 1960 | missing | `10.1287/mnsc.6.4.475` | Crossref 200 | **VERIFIED** |
| 8 | Gallego & Moon 1993 | missing | `10.1057/jors.1993.141` | Crossref 200 | **VERIFIED** |
| 9 | Sterman 1989 | missing | `10.1287/mnsc.35.3.321` | Crossref 200 | **VERIFIED** |
| 10 | Eckman, Henderson & Shashaani 2023 (SimOpt) | missing | `10.1287/ijoc.2023.1273` | Crossref 200 | **VERIFIED** |

**Nine to add. Ng 1999 is already in the bib** — the source reports list it as a gap, but
`references.bib:255` already carries `@inproceedings{ng1999}`. Verify its page range (278–287)
against the ICML 1999 proceedings before dispatch; that is the one field no API confirmed.

---

## Verified metadata, as returned by the registry

Exactly what the API said, so a later reader can diff the BibTeX against it.

**2 · Wiewiora 2003** — Crossref `10.1613/jair.1190` → *"Potential-Based Shaping and Q-Value
Initialization are Equivalent"*, Journal of Artificial Intelligence Research, 2003, vol. 19,
pp. 205–208, `journal-article`, AI Access Foundation. Author returned as `Wiewiora, E.`
(initial only in the record; the full given name **Eric** is *not* confirmed by the API — use the
initial, or confirm from the PDF).

**3 · Yu et al. 2022 (MAPPO)** — two independent records, both matching:
- Crossref `10.52202/068431-1787` → *"The Surprising Effectiveness of PPO in Cooperative
  Multi-Agent Games"*, **Advances in Neural Information Processing Systems 35**, 2022,
  pp. 24611–24624, `proceedings-article`; event *36th Conference on Neural Information Processing
  Systems (NeurIPS 2022)*, New Orleans, 2022-11-28/12-09. Authors: Yu, Velu, Vinitsky, Gao, Wang,
  Bayen, Wu.
- DataCite `10.48550/arXiv.2103.01955` → same title, `Preprint`, arXiv, publicationYear 2021, same
  seven authors.
- *Note:* the Crossref record gives the container as NeurIPS 35 without distinguishing the
  **Datasets & Benchmarks** track. If the track matters for the citation, confirm it from the
  proceedings page — the API does not state it.

**4 · Rashid et al. 2018 (QMIX)** — DataCite `10.48550/arXiv.1803.11485` → *"QMIX: Monotonic Value
Function Factorisation for Deep Multi-Agent Reinforcement Learning"*, `Preprint`, arXiv,
publicationYear 2018. Authors: Rashid, Samvelyan, de Witt, Farquhar, Foerster, Whiteson.
`POR_VERIFICAR`: the **venue**. Crossref has no ICML 2018 (PMLR v80) record, and `export.arxiv.org`
returned HTTP 429 before the `arxiv:comment` field could be read. A JMLR 21(178), 2020 journal
version of this work also exists and was **not** verified here — decide which version to cite and
confirm it from the publisher page.

**5 · Altman 1999 (CMDP)** — Crossref `10.1201/9781315140223` → *"Constrained Markov Decision
Processes"*, `monograph`, **Routledge**, issued **2021**, author Altman, Eitan.
`POR_VERIFICAR`: this DOI is the **2021 Routledge reissue**, not the 1999 CRC Press first edition.
The original 1999 edition appears to carry no DOI. Two honest options — pick one, do not blend
them: (a) cite the 1999 edition with `publisher = {CRC Press}, year = {1999}` and **no DOI**; or
(b) cite the reissue with the DOI and `year = {2021}`. Citing "Altman 1999" *with* this DOI would
attach a verified identifier to an unverified year.

**6 · Achiam et al. 2017 (CPO)** — DataCite `10.48550/arXiv.1705.10528` → *"Constrained Policy
Optimization"*, `Preprint`, arXiv, publicationYear 2017. Authors: Achiam, Held, Tamar, Abbeel.
`export.arxiv.org` returned 200 with `published 2017-05-30T10:07:31Z` and the authors' own comment
**"Accepted to ICML 2017"** — the venue is confirmed from the record, not from memory.

**7 · Clark & Scarf 1960** — Crossref `10.1287/mnsc.6.4.475` → *"Optimal Policies for a
Multi-Echelon Inventory Problem"*, Management Science, 1960, vol. 6, no. 4, pp. 475–490,
`journal-article`, INFORMS. Authors: Clark, Andrew J.; Scarf, Herbert.

**8 · Gallego & Moon 1993** — Crossref `10.1057/jors.1993.141` → *"The Distribution Free Newsboy
Problem: Review and Extensions"*, Journal of the Operational Research Society, 1993, vol. 44,
no. 8, pp. 825–834, `journal-article`. Authors: Gallego, Guillermo; Moon, Ilkyeong.

**9 · Sterman 1989** — Crossref `10.1287/mnsc.35.3.321` → *"Modeling Managerial Behavior:
Misperceptions of Feedback in a Dynamic Decision Making Experiment"*, Management Science, 1989,
vol. 35, no. 3, pp. 321–339, `journal-article`, INFORMS. Author: Sterman, John D.

**10 · Eckman, Henderson & Shashaani 2023** — Crossref `10.1287/ijoc.2023.1273` → *"SimOpt: A
Testbed for Simulation-Optimization Experiments"*, INFORMS Journal on Computing, 2023, vol. 35,
no. 2, pp. 495–508, `journal-article`, INFORMS. Authors: Eckman, David J.; Henderson, Shane G.;
Shashaani, Sara.

**1 · Ng, Harada & Russell 1999** — no Crossref record. A bibliographic query for the exact title
returns only later derivative works (`10.1016/j.neucom.2017.05.090`, `10.1613/jair.3384`), which
confirms the absence rather than the citation. ICML 1999 proceedings (Morgan Kaufmann) predate
routine DOI assignment; **no DOI is the correct state**, not a gap. Resolve the page range against
DBLP or the proceedings volume.

---

## Ready-to-paste BibTeX

House style matches the existing file: two-space indent, aligned `=`, braces protecting
capitalisation in titles. Nine entries; `ng1999` is not repeated.

```bibtex
% ── Reward shaping / potential-based invariance ──────────────

@article{wiewiora2003,
  author  = {Wiewiora, E.},
  title   = {Potential-Based Shaping and {Q}-Value Initialization are Equivalent},
  journal = {Journal of Artificial Intelligence Research},
  volume  = {19},
  pages   = {205--208},
  year    = {2003},
  doi     = {10.1613/jair.1190},
}

% ── Multi-agent RL: the primary sources our comparators cite ─

@inproceedings{yu2022mappo,
  author    = {Yu, Chao and Velu, Akash and Vinitsky, Eugene and Gao, Jiaxuan and
               Wang, Yu and Bayen, Alexandre and Wu, Yi},
  title     = {The Surprising Effectiveness of {PPO} in Cooperative Multi-Agent Games},
  booktitle = {Advances in Neural Information Processing Systems 35 (NeurIPS 2022)},
  pages     = {24611--24624},
  year      = {2022},
  doi       = {10.52202/068431-1787},
}

@misc{rashid2018qmix,
  author       = {Rashid, Tabish and Samvelyan, Mikayel and de Witt, Christian Schroeder and
                  Farquhar, Gregory and Foerster, Jakob and Whiteson, Shimon},
  title        = {{QMIX}: Monotonic Value Function Factorisation for Deep Multi-Agent
                  Reinforcement Learning},
  year         = {2018},
  eprint       = {1803.11485},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  doi          = {10.48550/arXiv.1803.11485},
}

% ── Constrained MDPs / safe RL: the guardrail formalism ──────

@book{altman1999,
  author    = {Altman, Eitan},
  title     = {Constrained {M}arkov Decision Processes},
  publisher = {CRC Press},
  year      = {1999},
  % NO DOI on the 1999 edition. The verified DOI 10.1201/9781315140223 belongs to the
  % 2021 Routledge reissue; do not attach it to year 1999.
}

@misc{achiam2017cpo,
  author        = {Achiam, Joshua and Held, David and Tamar, Aviv and Abbeel, Pieter},
  title         = {Constrained Policy Optimization},
  year          = {2017},
  note          = {Accepted to ICML 2017},
  eprint        = {1705.10528},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.1705.10528},
}

% ── Classical inventory theory: the structure our comparators exploit ──

@article{clark1960,
  author  = {Clark, Andrew J. and Scarf, Herbert},
  title   = {Optimal Policies for a Multi-Echelon Inventory Problem},
  journal = {Management Science},
  volume  = {6},
  number  = {4},
  pages   = {475--490},
  year    = {1960},
  doi     = {10.1287/mnsc.6.4.475},
}

@article{gallego1993,
  author  = {Gallego, Guillermo and Moon, Ilkyeong},
  title   = {The Distribution Free Newsboy Problem: Review and Extensions},
  journal = {Journal of the Operational Research Society},
  volume  = {44},
  number  = {8},
  pages   = {825--834},
  year    = {1993},
  doi     = {10.1057/jors.1993.141},
}

% ── Behavioural OM: why open-loop human policy is the wrong baseline ──

@article{sterman1989,
  author  = {Sterman, John D.},
  title   = {Modeling Managerial Behavior: Misperceptions of Feedback in a Dynamic
             Decision Making Experiment},
  journal = {Management Science},
  volume  = {35},
  number  = {3},
  pages   = {321--339},
  year    = {1989},
  doi     = {10.1287/mnsc.35.3.321},
}

% ── Simulation-optimisation testbeds: the methodological anchor ──

@article{eckman2023simopt,
  author  = {Eckman, David J. and Henderson, Shane G. and Shashaani, Sara},
  title   = {{SimOpt}: A Testbed for Simulation-Optimization Experiments},
  journal = {INFORMS Journal on Computing},
  volume  = {35},
  number  = {2},
  pages   = {495--508},
  year    = {2023},
  doi     = {10.1287/ijoc.2023.1273},
}
```

---

## Where each one earns its place in the manuscript

Not padding — each answers a specific referee objection, and the section that needs it is named.

| Entry | Answers | Section |
|---|---|---|
| `ng1999`, `wiewiora2003` | *"Your terminal-only reward could have been shaped; and if you shape it, does the policy ranking survive?"* We use terminal ReT with **no** shaping; these two are the citation for why that is the conservative choice and what invariance would require if a shaped arm is ever added. | Methods §2.4 |
| `yu2022mappo`, `rashid2018qmix` | *"Why not MARL?"* — and the primary sources for the algorithms Ding 2026 and Kong 2026 use. Cite the source, not only its users. | Related work; Discussion (scope limits) |
| `altman1999`, `achiam2017cpo` | *"Your equity guardrail is a hand-rolled gate — why not a constrained MDP?"* The CMDP formalism and its policy-optimisation instantiation are the principled alternative we deliberately did not take, and we should say why. | Methods §2.3; Conclusion §5.3 |
| `clark1960`, `gallego1993` | *"Your 'classical controllers' are ad hoc."* They are not: base-stock echelon structure and distribution-free newsvendor bounds are the theory the ten comparators instantiate. | Methods §2.3 (comparator family) |
| `sterman1989` | *"Why is the open-loop frontier the right null?"* Because human open-loop policy under feedback delay is the documented failure mode; the enumerated frontier is its strongest possible form. | Introduction; Discussion §4.2 |
| `eckman2023simopt` | *"Is your protocol reproducible by anyone else?"* The current standard testbed for simulation-optimisation experiments; positions our benchmark relative to it. | Methods §2.7; Conclusion §5.4 |

---

## Two things to settle before this bib ships

1. **This bib may be the wrong file.** The header of
   `docs/manuscript_current/submission/elsevier/references.bib` reads *"References for: Action Space
   Alignment with Operational Constraints Determines RL Effectiveness for SCRES"* — a different
   (Paper 1) title. The CIE submission needs either a new bib for Paper 2, or a merge with
   `/home/ubuntu/scres-sources/registry/bibliografia_paper.bib` (119 entries, DOIs pre-verified
   against Crossref/DataCite). **None of the nine entries above is in either file**; adding them to
   only one will leave the other short. Decide which is canonical first.

2. **The CIE-specific citations are a separate list.** Guzmán et al. 2026
   (`10.1016/j.cie.2026.112044`), Habibi et al. 2023 (`10.1016/j.cie.2023.109531`), Tian et al. 2024
   (`10.1016/j.cie.2023.109829`), Park & Lee 2025 (`10.1016/j.cie.2025.111312`), Sriprateep et al.
   2026 (`10.1016/j.cie.2025.111583`), Ding et al. 2026 (`10.1016/j.ijpe.2026.109995`), Kaynov et
   al. 2024 (`10.1016/j.ijpe.2023.109088`), Gijsbrechts et al. 2022 (`10.1287/msom.2021.1064`) and
   Rolf et al. 2022 (`10.1080/00207543.2022.2140221`) are cited in the cover letter and must appear
   in the manuscript. **All nine resolved against Crossref on 2026-08-24** with matching title,
   container, year, volume and article number. They are in the registry bib but not in
   `references.bib`.

## Reproducing this verification

```bash
# Crossref (journal articles, proceedings)
curl -s -A "mailto:YOUR_EMAIL" "https://api.crossref.org/works/10.1287/mnsc.6.4.475" | jq '.message | {title, "container-title", issued, volume, page, author}'

# DataCite (arXiv 10.48550/* DOIs -- Crossref 404s on these BY DESIGN)
curl -s "https://api.datacite.org/dois/10.48550%2FarXiv.1803.11485" | jq '.data.attributes | {titles, publicationYear, creators}'

# arXiv (venue, from the authors' own comment field; rate-limits aggressively)
curl -s "https://export.arxiv.org/api/query?id_list=1705.10528" | grep -E "<title>|arxiv:comment|journal_ref"
```
