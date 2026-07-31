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
    # 2026-07-09 provenance repairs (independent assessment, verified):
    "Invented +-15% validation threshold.": r"pm\s*15\\?%\s*.{0,30}threshold",
    "Stale h104 delta from the weaker-comparator duplicate run (table use).": (
        r"\+0\.000244\s*\[\+0\.000209"
    ),
    "Stale h104 increased delta from the duplicate run (table use).": (
        r"\+0\.000623\s*\[\+0\.000584"
    ),
    "p99 described as worst-case.": r"worst-case\s+(1\\?%|resupply|performance|bound)",
    "E4 arms described as single-lever isolation.": (
        r"only\s+op10/op12\s+dispatch\s+controllable"
    ),
    "Validated digital twin language.": r"validated\s+digital\s+twin",
    # 2026-07-10 identified factorial (Blocker 2) retired the E4 reading:
    "Downstream dispatch as strongest lever (retired by identified factorial).": (
        r"strongest\s+(observed\s+)?lever"
    ),
    "Only-when-dispatch framing (refuted by the upstream_shift arm).": (
        r"only\s+when\s+the\s+(action\s+contract|controllable\s+interface)"
    ),
    "Regardless-of-algorithm/reward universality language.": (
        r"regardless\s+of\s+(reward\s+design\s+or\s+)?algorithm"
    ),
}


# A retired phrase is banned as an ASSERTION, not as a word. The manuscript is expected to
# say what it is NOT -- `04_mfsc_case.tex` reads "not as a validated digital twin", which is
# the disclaimer this registry wants, and a substring match flagged it as a regression. A
# check that fires on correct text is worse than no check: it trains the reader to expect a
# red bar, and a REAL regression then hides behind the same failure.
#
# So a hit is discounted when a negation cue governs it: the cue must sit in the WINDOW
# characters before the match, with no sentence boundary in between (a full stop ends the
# negation's reach, so "...is not supported. The model is a validated digital twin." still
# fails, as it must).
NEGATION_WINDOW = 48
NEGATION_CUES = re.compile(
    r"\b(?:not|never|neither|nor|no|without|rather\s+than|instead\s+of|"
    r"cannot|can't|isn't|aren't|does\s+not|do\s+not|avoid|reject(?:s|ed)?|"
    r"stop(?:s|ped)?\s+short\s+of)\b")
SENTENCE_BOUNDARY = re.compile(r"[.;:!?]|\\\\|\n\s*\n")


def _is_disclaimed(text: str, start: int) -> bool:
    """True when a negation cue governs the match beginning at `start`."""
    window = text[max(0, start - NEGATION_WINDOW):start]
    cues = list(NEGATION_CUES.finditer(window))
    if not cues:
        return False
    # Only the LAST cue can govern; anything before it may belong to an earlier clause.
    tail = window[cues[-1].end():]
    return not SENTENCE_BOUNDARY.search(tail)


def find_asserted_claims(text: str) -> dict[str, str]:
    """Retired claims ASSERTED in `text`, ignoring occurrences that are disclaimed."""
    hits: dict[str, str] = {}
    for claim, pattern in RETIRED_CLAIM_PATTERNS.items():
        for m in re.finditer(pattern, text):
            if not _is_disclaimed(text, m.start()):
                hits[claim] = pattern
                break
    return hits


def _manuscript_text() -> str:
    assert SECTIONS_DIR.is_dir(), f"expected manuscript sections at {SECTIONS_DIR}"
    return "\n".join(
        p.read_text(encoding="utf-8") for p in sorted(SECTIONS_DIR.glob("*.tex"))
    ).lower()


def test_no_retired_claims_in_manuscript() -> None:
    hits = find_asserted_claims(_manuscript_text())
    assert not hits, (
        "Retired/unsafe claim(s) reappeared in the manuscript "
        f"(see docs/CLAIMS_REGISTRY_Q1_DEFENSE_2026-07-01.md): {hits}"
    )


def test_disclaimer_is_not_a_regression_but_an_assertion_is() -> None:
    """The negation exemption must not become a way to smuggle the claim back in."""
    disclaimed = "we present it as a throughput check, not as a validated digital twin."
    asserted = "the environment is a validated digital twin of the case study."
    assert not find_asserted_claims(disclaimed)
    assert find_asserted_claims(asserted) == {
        "Validated digital twin language.": r"validated\s+digital\s+twin"}
    # A negation in an EARLIER sentence must not reach across the boundary.
    assert find_asserted_claims(
        "this is not supported. the model is a validated digital twin.")
    # Nor across a LaTeX line break inside one paragraph.
    assert find_asserted_claims(r"h4 is not proven \\ the model is a validated digital twin")


def test_every_pattern_still_fires_on_its_own_claim() -> None:
    """A regex that matches nothing protects nothing -- catch a typo'd pattern here."""
    for claim, pattern in RETIRED_CLAIM_PATTERNS.items():
        probe = {
            "Track B is thesis-faithful.": "track b is thesis-faithful",
            "Track B uses the retired seven-dimensional label.": "seven-dimensional action",
            "Perfect-fill or fill=1.000 is the headline.": "fill = 1.000",
            "Strictly Pareto-dominates on all metrics.": "ppo dominates on all metrics",
            "PPO anticipates risks.": "ppo anticipates the disruption",
            "H4 is proven.": "h4 is proven by the frontier",
            "Track A preventive/coarse-frontier wins are publishable.":
                "track a preventive results are publishable",
            "Invented +-15% validation threshold.": r"a \pm 15\% fidelity threshold",
            "Stale h104 delta from the weaker-comparator duplicate run (table use).":
                "+0.000244 [+0.000209, +0.000280]",
            "Stale h104 increased delta from the duplicate run (table use).":
                "+0.000623 [+0.000584, +0.000662]",
            "p99 described as worst-case.": "the worst-case resupply time",
            "E4 arms described as single-lever isolation.":
                "only op10/op12 dispatch controllable",
            "Validated digital twin language.": "a validated digital twin",
            "Downstream dispatch as strongest lever (retired by identified factorial).":
                "the strongest observed lever",
            "Only-when-dispatch framing (refuted by the upstream_shift arm).":
                "only when the action contract exposes it",
            "Regardless-of-algorithm/reward universality language.":
                "holds regardless of reward design or algorithm",
        }[claim]
        assert re.search(pattern, probe), f"pattern for {claim!r} no longer fires"
