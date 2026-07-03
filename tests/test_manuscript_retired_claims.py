from __future__ import annotations

import re
from pathlib import Path

SECTIONS_DIR = Path("docs/manuscript_current/submission/elsevier/sections")

# Mirrors docs/CLAIMS_REGISTRY_Q1_DEFENSE_2026-07-01.md's "Retired Or Unsafe
# Claims" table. Keep these two in sync: if a claim is retired there, its
# banned phrase belongs here too, so a regression can't silently creep back
# into the manuscript between reviewer rounds.
RETIRED_CLAIM_PATTERNS = {
    "Track B is thesis-faithful.": r"track\s*b\s+is\s+thesis-faithful",
    "Track B uses the retired seven-dimensional label.": r"seven-dimensional",
    "Perfect-fill or fill=1.000 is the headline.": r"fill\s*=\s*1\.000",
    "Strictly Pareto-dominates on all metrics.": r"dominates\s+on\s+all\s+metrics",
    "PPO anticipates risks.": r"ppo\s+anticipates",
    "H4 is proven.": r"h4\s+is\s+proven",
    "Track A preventive/coarse-frontier wins are publishable.": (
        r"track\s*a\s+preventive.{0,40}publishable"
    ),
}


def _manuscript_text() -> str:
    assert SECTIONS_DIR.is_dir(), f"expected manuscript sections at {SECTIONS_DIR}"
    return "\n".join(
        p.read_text(encoding="utf-8") for p in sorted(SECTIONS_DIR.glob("*.tex"))
    ).lower()


def test_no_retired_claims_in_manuscript() -> None:
    text = _manuscript_text()
    hits = {
        claim: pattern
        for claim, pattern in RETIRED_CLAIM_PATTERNS.items()
        if re.search(pattern, text)
    }
    assert not hits, (
        "Retired/unsafe claim(s) reappeared in the manuscript "
        f"(see docs/CLAIMS_REGISTRY_Q1_DEFENSE_2026-07-01.md): {hits}"
    )
