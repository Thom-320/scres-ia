#!/usr/bin/env python3
"""Withdrawn seals, in one machine-readable place instead of seven field names.

WHY THIS EXISTS. The repo rule is that a withdrawn result is KEPT and LABELLED, never deleted. It
has been followed -- but the label was written wherever the author of the day happened to put it.
A scan of `results/*/result*.json` finds the same relation under `supersedes`, `predecessor`,
`predecessors`, `superseded_null`, `supersedes_for_multiplicity`,
`transform_family_supersedes_the_ceiling` and a free-text `retraction`. Nothing can read that. So a
reviewer asking "is this artifact still live?" has to know the vocabulary, and an agent asking the
same question five weeks from now does not.

Normalising costs nothing and buys the check that matters: a superseded artifact must not be cited
as live evidence. That check runs in `tests/test_supersession_registry.py`.

PARTIAL SUPERSESSION IS ITS OWN RELATION, and it is not a formality. `retention_simultaneous`
supersedes `retention_contrasts` FOR MULTIPLICITY ONLY -- the per-family estimates stand, the
simultaneous reading replaces the marginal one. A registry with only a live/dead bit would either
kill a claim the manuscript legitimately cites, or wave through the exact over-reading the
simultaneous run was built to prevent.

WHAT THIS IS NOT. It does not adjudicate. Every edge here is one an artifact or a dated document
ALREADY declared; this script finds them and gives them one shape. An edge that appears only in
CURATED must name the document that declared it, and that document must exist on disk.

Development tooling. Reads artifacts, writes one JSON. No seeds, no science.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

OUT = Path("research/supersession_registry.json")
CLAIM_LOCK = Path("papers/paper2/claim_lock.json")

#: Relations, with the reading rule attached to each. The rule is the point: "superseded" alone does
#: not say whether the old number may still be quoted.
RELATIONS = {
    "SUPERSEDED_BY_FAILED_REPLICATION": (
        "a preregistered replication on fresh seeds did not reproduce it; the artifact is retained "
        "as the record of what was believed, and its number may not be cited as a live estimate"),
    "SUPERSEDED_BY_CORRECTIVE_RERUN": (
        "a defect was found and the corrected run replaces it; the old number may be cited only as "
        "the defective one, next to the defect"),
    "PREDECESSOR_IN_A_VERSION_CHAIN": (
        "the successor is the citable version; the predecessor is retained for provenance and is "
        "not independent evidence"),
    "SUPERSEDED_IN_PART": (
        "one component is replaced and the rest stands; the artifact stays citable, but only "
        "together with its successor, and never for the replaced component"),
    "VOIDED_OF_OBJECT": (
        "the artifact is internally sound but the thing it measured was withdrawn, so it makes no "
        "claim about anything; it is neither positive nor negative evidence"),
    "RELABELLED_BLOCKED_INSTRUMENT": (
        "the run completed but the instrument was defective; retained, and never evidence"),
    "SUPERSEDED_BY_A_WIDENED_COMPARATOR_CLASS": (
        "the result is arithmetically intact and its advantage over the comparators it HAD is "
        "real; a later run widened the comparator class and the advantage did not survive it. The "
        "number may be cited only against the narrow class it beat, never as a premium"),
    "SUPERSEDED_BY_FAILED_PREMISE": (
        "the artifact is a DESIGN, not a measurement, and the evidence it was built on does not "
        "say what its author claimed. Nothing inside it was run or refuted; its reason to exist "
        "was. It is retained as the record of a proposal and may never be cited as a live design"),
    "LINEAGE_NOT_SUPERSESSION": (
        "the successor was BUILT ON this artifact but answers a different question, so nothing is "
        "withdrawn; recorded for provenance and never enforced against a citation"),
}


def lineage_stem(artifact: str) -> str:
    """The result directory with any trailing `_vN` removed.

    A bare `predecessor` field means two different things in this tree. In `monotone_transform_
    family_v4` it means v3 was replaced. In `citable_risk_attitudes` it means the risk-attitude run
    was BUILT ON the transform family, which answers a different question and is not withdrawn by
    anything. The field name cannot tell them apart, so the stem does: same lineage is a version
    chain, different lineage is provenance.

    This under-claims on purpose. `monotone_transform_ceiling -> monotone_transform_family_v2` is a
    real chain that this rule files as lineage, because a registry that invents a supersession is
    worse than one that misses a rename.
    """
    stem = artifact.split("/")[1] if "/" in artifact else artifact
    while stem.rsplit("_", 1)[-1].startswith("v") and stem.rsplit("_", 1)[-1][1:].isdigit():
        stem = stem.rsplit("_", 1)[0]
    return stem

#: Field names actually found in the tree, mapped to (relation, direction). `direction` says which
#: side of the edge the HOSTING artifact sits on: "successor" means the host supersedes what it
#: names; "successor_of_named" is the same thing read the other way for `predecessor`-style fields.
HARVEST = {
    "supersedes": ("SUPERSEDED_BY_CORRECTIVE_RERUN", "host_is_successor"),
    "predecessor": ("PREDECESSOR_IN_A_VERSION_CHAIN", "host_is_successor"),
    "predecessors": ("PREDECESSOR_IN_A_VERSION_CHAIN", "host_is_successor"),
    "predecessor_artifact": ("PREDECESSOR_IN_A_VERSION_CHAIN", "host_is_successor"),
    "supersedes_for_multiplicity": ("SUPERSEDED_IN_PART", "host_is_successor"),
    "transform_family_supersedes_the_ceiling": ("SUPERSEDED_IN_PART", "host_is_superseded"),
}

#: Edges declared in a dated document rather than inside an artifact. Each MUST name that document,
#: and the document must exist -- otherwise this list becomes a place to assert supersession by
#: writing it down, which is the failure mode the registry exists to remove.
CURATED: list[dict] = [
    {
        "superseded": "results/ceiling_null_diagnostic/result.json",
        "successor": "results/expanded_signal_search/result.json",
        "relation": "SUPERSEDED_BY_FAILED_REPLICATION",
        "evidence": "docs/RETRACTACION_TECHO_CLARIVIDENTE_2026-08-08.md",
        "why": (
            "the clairvoyant gap that passed its interaction null at p=0.0132 on twelve reused "
            "tapes measured +0.024054 on 48 virgin seeds against a null whose MEAN is +0.026641, "
            "p=0.7482 -- below its own null, which is the Jensen bias of a minimum over 27 noisy "
            "options and nothing else. The run was not defective; twelve tapes were not enough"),
        "retained": True,
    },
    {
        "superseded": "results/signal_search/result.json",
        "successor": "results/expanded_signal_search/result.json",
        "relation": "VOIDED_OF_OBJECT",
        "evidence": "docs/RETRACTACION_TECHO_CLARIVIDENTE_2026-08-08.md",
        "why": (
            "it searched for an observable signal capturing a ceiling that no longer replicates. "
            "Its negative is neither strengthened nor weakened; it stops being a statement about "
            "anything"),
        "retained": True,
    },
    {
        "superseded": "results/headroom/cd_surface_prediction_premium/result.json",
        "successor": "results/program_n/gate_b_confirmation_v3/result.json",
        "relation": "SUPERSEDED_IN_PART",
        "evidence": "docs/CORRECCION_TECHO_SUPERFICIE_CD_2026-08-09.md",
        "why": (
            "two components go and the rest stands. The NOT_CAPTURED half is superseded: on a "
            "virgin block the repaired fit captures the margin, mlp +0.1081 [+0.0601, +0.1561] "
            "against the best classical arm of its information class. And its "
            "train_cell_mean_comparator was never a ceiling -- neural arms beat it in all four "
            "Gate B runs including this one -- so '+0.0625 of available margin' has no upper "
            "bound to subtract from and may not be quoted. What stands is the finding itself: "
            "there was margin over the linear baseline and its networks did not take it"),
        "retained": True,
    },
    {
        "superseded": "results/program_n/gate_b_confirmation_v2/result.json",
        "successor": "results/program_n/gate_b_confirmation_v3/result.json",
        "relation": "RELABELLED_BLOCKED_INSTRUMENT",
        "evidence": "docs/ENMIENDA_F2_SIN_CLAUSULA_DE_ORDEN_2026-08-09.md",
        "why": (
            "f2 demanded the classical ranking be preserved exactly and burned the block on a "
            "0.0153 swap whose own paired CI was [-0.0388, +0.0082] -- a sign test on a quantity "
            "that straddles zero. The defect is the falsifier's, not the data's, and the repaired "
            "form was shown to pass on this very artifact. Retained as development evidence; it "
            "is not a confirmation and its seeds are spent"),
        "retained": True,
    },
    {
        "superseded": "results/track_b_nonneural/result.json",
        "successor": "results/program_n/gate_a2_track_b/result.json",
        "relation": "SUPERSEDED_BY_A_WIDENED_COMPARATOR_CLASS",
        "evidence": "docs/CORRECCION_TECHO_SUPERFICIE_CD_2026-08-09.md",
        "why": (
            "NEURAL_PREMIUM_LIKELY_IN_TRACK_B rested on beating a constant and a threshold rule. "
            "Gate A2 added a linear feedback policy and an EWMA rule on the same block, paired by "
            "seed: the MLP still beats the threshold rule (+0.472 [+0.275, +0.658], 37/48) and "
            "both history placebos, so its memory does something -- but linear feedback beats the "
            "MLP (-0.559 [-0.748, -0.386], 7/48 favourable). The +1.60 headline also shrinks to "
            "+0.47 once paired by seed. The premium was real against the class we had"),
        "retained": True,
    },
    {
        "superseded": "contracts/program_x_o_scale_amortized_control_v2.json",
        "successor": "contracts/program_x_o_scale_amortized_control_v1.json",
        "relation": "SUPERSEDED_BY_FAILED_PREMISE",
        "evidence": "docs/RETRACTACION_CONTENTION_V1_Y_ENMIENDA_X_2026-08-12.md",
        "why": (
            "v2 gave the latent regime a minimum dwell on the stated ground that contention_v1 was "
            "the one place a learner beat a belief planner, at +0.0136 [LCB95 +0.0124]. That "
            "number appears in no artifact. The real contrast is +0.011477 [+0.009135] against a "
            "SESOI of 0.010, so it does not clear, and the artifact says so: "
            "AUDIT_STOPS_CORRECTLY_BUT_POSITIVE_DIRECTION_NOT_DEMONSTRATED. The direction also "
            "inverts -- the learner's edge over the true-model arm is LARGER in the min_dwell=1 "
            "cell (+0.019374) than in the min_dwell=4 one (+0.010323) -- min_dwell is confounded "
            "with rho across the only two cells, and v2's own G4b gate already fails on existing "
            "data: the true-model filter beats the first-order one by 0.001154, an order of "
            "magnitude under the SESOI. v1 is the live contract again"),
        "retained": True,
    },
]


def digest(path: Path) -> dict:
    if not path.exists():
        return {"exists": False, "file_sha256": None, "claim_status": None, "self_sha256": None}
    raw = path.read_bytes()
    out = {"exists": True, "file_sha256": sha256(raw).hexdigest()}
    try:
        doc = json.loads(raw)
        out["claim_status"] = doc.get("claim_status") if isinstance(doc, dict) else None
        out["self_sha256"] = doc.get("self_sha256") if isinstance(doc, dict) else None
    except json.JSONDecodeError:
        out["claim_status"] = out["self_sha256"] = None
    return out


def as_paths(value) -> list[str]:
    """A declared edge may be a path, a list of paths, or an object carrying one under `path`."""
    if isinstance(value, str):
        return [value] if value.endswith(".json") else []
    if isinstance(value, list):
        return [p for v in value for p in as_paths(v)]
    if isinstance(value, dict):
        for key in ("path", "artifact"):
            if isinstance(value.get(key), str):
                return [value[key]]
    return []


def harvest(root: Path) -> list[dict]:
    edges = []
    for path in sorted(root.glob("results/*/result*.json")):
        try:
            doc = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(doc, dict):
            continue
        for field, (relation, direction) in HARVEST.items():
            if field not in doc:
                continue
            named = as_paths(doc[field])
            why = doc[field].get("why") if isinstance(doc[field], dict) else None
            for other in named:
                host = str(path.relative_to(root))
                pair = ((other, host) if direction == "host_is_successor" else (host, other))
                actual = relation
                if (relation == "PREDECESSOR_IN_A_VERSION_CHAIN"
                        and lineage_stem(pair[0]) != lineage_stem(pair[1])):
                    actual = "LINEAGE_NOT_SUPERSESSION"
                edges.append({
                    "superseded": pair[0], "successor": pair[1], "relation": actual,
                    "evidence": f"{host}#{field}", "why": why,
                    "retained": bool(doc[field].get("retained", True))
                    if isinstance(doc[field], dict) else True,
                    "source": "HARVESTED",
                })
    return edges


def cited_artifacts(root: Path) -> tuple[dict[str, list[str]], dict[str, set[str]]]:
    """Which claim ids cite which artifact, and which artifacts each claim drags along with it.

    The second half is what makes SUPERSEDED_IN_PART enforceable: a claim citing a partly
    superseded artifact must name a companion claim whose artifact IS the successor.
    """
    lock = root / CLAIM_LOCK
    if not lock.exists():
        return {}, {}
    claims = json.loads(lock.read_text()).get("claims", [])
    by_id = {c["claim_id"]: c.get("artifact", "") for c in claims}
    cited: dict[str, list[str]] = {}
    companions: dict[str, set[str]] = {}
    for claim in claims:
        cited.setdefault(claim.get("artifact", ""), []).append(claim["claim_id"])
        companions[claim["claim_id"]] = {
            by_id.get(other, "") for other in (claim.get("must_be_cited_with") or [])}
    return cited, companions


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    root: Path = args.root
    out_path = args.output or (root / OUT)

    problems: list[str] = []
    edges = harvest(root)
    for entry in CURATED:
        edge = dict(entry, source="CURATED")
        doc = root / entry["evidence"]
        if not doc.exists():
            problems.append(
                f"{entry['superseded']}: curated edge cites {entry['evidence']}, which is missing "
                "-- a curated edge without its declaring document is an assertion, not a record")
        edges.append(edge)

    cited, companions = cited_artifacts(root)
    rows = []
    for edge in edges:
        if edge["relation"] not in RELATIONS:
            problems.append(f"{edge['superseded']}: unknown relation {edge['relation']!r}")
        sup, suc = digest(root / edge["superseded"]), digest(root / edge["successor"])
        if not sup["exists"]:
            problems.append(
                f"{edge['superseded']}: superseded artifact is GONE. The rule is retain and label; "
                "a missing one cannot be audited and cannot be un-deleted")
        if not suc["exists"]:
            problems.append(f"{edge['successor']}: successor artifact missing")
        claims = cited.get(edge["superseded"], [])
        # A live citation of a fully superseded artifact is the defect this registry exists to
        # catch. SUPERSEDED_IN_PART is exempt from THAT check by relation -- the artifact stays
        # citable -- but not from checking. Its constraint is that the successor travels with it,
        # and an exemption nobody verifies is how a partial supersession quietly becomes none.
        if claims and edge["relation"] not in ("SUPERSEDED_IN_PART",
                                               "LINEAGE_NOT_SUPERSESSION"):
            problems.append(
                f"{edge['superseded']}: superseded ({edge['relation']}) but cited as live evidence "
                f"by {', '.join(claims)} in {CLAIM_LOCK}")
        if claims and edge["relation"] == "SUPERSEDED_IN_PART":
            for claim_id in claims:
                if edge["successor"] not in companions.get(claim_id, set()):
                    problems.append(
                        f"{claim_id}: cites {edge['superseded']}, which is SUPERSEDED_IN_PART by "
                        f"{edge['successor']}, without a `must_be_cited_with` pointing at it -- the "
                        "partial supersession is unenforced and reads as no supersession")
        rows.append({
            **edge,
            "reading_rule": RELATIONS.get(edge["relation"]),
            "superseded_digest": sup, "successor_digest": suc,
            "cited_by_claim_lock": claims,
        })

    rows.sort(key=lambda r: (r["superseded"], r["successor"]))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            cwd=root).stdout.strip()
    payload = {
        "schema_version": "supersession_registry_v1",
        "generated_at_commit": commit,
        "policy": ("a withdrawn result is retained and labelled, never deleted; this file is the "
                   "single machine-readable index of which seals are withdrawn and how far"),
        "relations": RELATIONS,
        "n_edges": len(rows),
        "n_curated": sum(1 for r in rows if r["source"] == "CURATED"),
        "problems": problems,
        "edges": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")

    print(f"{'superseded':52}{'relation':34}fuente")
    for r in rows:
        print(f"{r['superseded']:52}{r['relation']:34}{r['source']}")
    if problems:
        print("\nPROBLEMAS:")
        for p in problems:
            print("  -", p)
    print(f"\n  -> {out_path}  ({len(rows)} aristas, {len(problems)} problemas)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
