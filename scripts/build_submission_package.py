#!/usr/bin/env python3
"""Build a submission package from the current manuscript workspace.

Outputs:

- `docs/manuscript_current/submission/manuscript_full.md`
- `docs/manuscript_current/submission/elsevier/main.tex`
- `docs/manuscript_current/submission/elsevier/sections/*.tex`
- `docs/manuscript_current/submission/README.md`

The generated `main.tex` prefers `elsarticle.cls` when available and falls back
to `article` for local review builds.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
MANUSCRIPT_DIR = REPO / "docs" / "manuscript_current"
SUBMISSION_DIR = MANUSCRIPT_DIR / "submission"
ELSEVIER_DIR = SUBMISSION_DIR / "elsevier"
SECTION_FILES = [
    "01_introduction.md",
    "02_related_work.md",
    "03_methodology.md",
    "04_results.md",
    "05_discussion.md",
    "06_conclusion.md",
]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def extract_title(path: Path) -> str:
    text = read_text(path)
    match = re.search(r"\*\*(.+?)\*\*", text)
    if not match:
        raise ValueError("Could not extract title")
    return match.group(1).strip()


def extract_abstract(path: Path) -> str:
    text = read_text(path)
    marker = "## Abstract"
    if marker not in text:
        raise ValueError("Could not locate abstract")
    abstract = text.split(marker, 1)[1].strip()
    return abstract.split("\n\n", 1)[0].strip()


def strip_status_lines(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if line.startswith("Status:"):
            continue
        lines.append(line)
    return "\n".join(lines).strip() + "\n"


def assemble_markdown(title: str, abstract: str) -> str:
    parts = [
        f"# {title}",
        "",
        "## Abstract",
        "",
        abstract,
        "",
    ]
    for filename in SECTION_FILES:
        parts.append(strip_status_lines(read_text(MANUSCRIPT_DIR / filename)).rstrip())
        parts.append("")
    return "\n".join(parts).strip() + "\n"


def markdown_to_latex(md_path: Path, tex_path: Path) -> None:
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["pandoc", str(md_path), "--from", "gfm", "--to", "latex"],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    tex_path.write_text(result.stdout, encoding="utf-8")


def write_main_tex(title: str, abstract: str, out_path: Path) -> None:
    content = rf"""\IfFileExists{{elsarticle.cls}}{{%
\documentclass[preprint,12pt]{{elsarticle}}
\newif\ifuseelsevier
\useelseviertrue
}}{{%
\documentclass[12pt]{{article}}
\newif\ifuseelsevier
\useelsevierfalse
}}
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage{{lmodern}}
\usepackage{{hyperref}}
\usepackage{{amsmath,amssymb}}
\usepackage{{booktabs}}
\usepackage{{longtable}}
\usepackage{{array}}
\usepackage{{graphicx}}

\begin{{document}}
\ifuseelsevier
\begin{{frontmatter}}
\title{{{title}}}
\begin{{abstract}}
{abstract}
\end{{abstract}}
\begin{{keyword}}
Supply chain resilience \sep reinforcement learning \sep discrete-event simulation \sep military supply chain \sep bottleneck alignment
\end{{keyword}}
\end{{frontmatter}}
\else
\title{{{title}}}
\author{{}}
\date{{}}
\maketitle
\begin{{abstract}}
{abstract}
\end{{abstract}}
\fi

\input{{sections/01_introduction.tex}}
\input{{sections/02_related_work.tex}}
\input{{sections/03_methodology.tex}}
\input{{sections/04_results.tex}}
\input{{sections/05_discussion.tex}}
\input{{sections/06_conclusion.tex}}

\nocite{{*}}
\ifuseelsevier
\bibliographystyle{{elsarticle-num}}
\else
\bibliographystyle{{plain}}
\fi
\bibliography{{references}}

\end{{document}}
"""
    out_path.write_text(content, encoding="utf-8")


def write_submission_readme() -> None:
    content = """# Submission Package

This directory contains the assembly outputs for the current manuscript.

Files:

- `manuscript_full.md`: concatenated manuscript source for rapid review
- `elsevier/main.tex`: LaTeX submission draft
- `elsevier/sections/*.tex`: Pandoc-generated section fragments
- `elsevier/references.bib`: bibliography snapshot used by the submission draft

Notes:

- `main.tex` uses `elsarticle.cls` when it is available in the local TeX installation.
- If `elsarticle.cls` is unavailable, the draft falls back to `article` so the manuscript still compiles for internal review.
- In-text citations remain plain-text prose at this stage; `\\nocite{*}` is used so the bibliography prints for review builds.
"""
    (SUBMISSION_DIR / "README.md").write_text(content, encoding="utf-8")


def main() -> None:
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    ELSEVIER_DIR.mkdir(parents=True, exist_ok=True)
    (ELSEVIER_DIR / "sections").mkdir(parents=True, exist_ok=True)

    title_source = MANUSCRIPT_DIR / "00_title_claim_abstract.md"
    title = extract_title(title_source)
    abstract = extract_abstract(title_source)

    manuscript_full = SUBMISSION_DIR / "manuscript_full.md"
    manuscript_full.write_text(assemble_markdown(title, abstract), encoding="utf-8")

    for filename in SECTION_FILES:
        md_path = MANUSCRIPT_DIR / filename
        tex_path = ELSEVIER_DIR / "sections" / filename.replace(".md", ".tex")
        markdown_to_latex(md_path, tex_path)

    write_main_tex(title, abstract, ELSEVIER_DIR / "main.tex")
    (ELSEVIER_DIR / "references.bib").write_text(
        read_text(MANUSCRIPT_DIR / "references.bib"),
        encoding="utf-8",
    )
    write_submission_readme()

    print(f"Wrote {manuscript_full}")
    print(f"Wrote {ELSEVIER_DIR / 'main.tex'}")
    print(f"Wrote {ELSEVIER_DIR / 'references.bib'}")


if __name__ == "__main__":
    main()
