"""The supersession registry, and the controls that prove its checks can fire.

Every test below that asserts "no problems" is paired with one that reintroduces the defect and
demands the problem appear. A guardrail nobody has watched fail is a guardrail nobody has tested --
this repo shipped a real data leak under a hardcoded `passed: True`, and the lesson was not "be
careful", it was "make the control part of the suite".
"""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_supersession_registry_v1 as reg  # noqa: E402

REGISTRY = ROOT / "research/supersession_registry.json"


@pytest.fixture(scope="module")
def registry() -> dict:
    assert REGISTRY.exists(), "run scripts/build_supersession_registry_v1.py"
    return json.loads(REGISTRY.read_text())


def test_registry_is_clean(registry):
    assert registry["problems"] == [], registry["problems"]
    assert registry["schema_version"] == "supersession_registry_v1"
    assert registry["n_edges"] > 0


def test_every_relation_carries_a_reading_rule(registry):
    """"Superseded" alone does not say whether the old number may still be quoted."""
    for edge in registry["edges"]:
        assert edge["relation"] in reg.RELATIONS
        assert edge["reading_rule"], edge["superseded"]


def test_no_withdrawn_artifact_was_deleted(registry):
    """The rule is retain and label. A missing file cannot be audited or un-deleted."""
    for edge in registry["edges"]:
        assert edge["superseded_digest"]["exists"], edge["superseded"]


def test_todays_retraction_is_recorded(registry):
    edges = {e["superseded"]: e for e in registry["edges"]}
    ceiling = edges["results/ceiling_null_diagnostic/result.json"]
    assert ceiling["relation"] == "SUPERSEDED_BY_FAILED_REPLICATION"
    assert ceiling["successor"] == "results/expanded_signal_search/result.json"
    assert (ROOT / ceiling["evidence"]).exists()
    assert edges["results/signal_search/result.json"]["relation"] == "VOIDED_OF_OBJECT"


def test_a_curated_edge_without_its_document_is_rejected(tmp_path, monkeypatch):
    """CONTROL. Otherwise CURATED becomes a place to assert supersession by writing it down."""
    monkeypatch.setattr(reg, "CURATED", [{
        "superseded": "results/signal_search/result.json",
        "successor": "results/expanded_signal_search/result.json",
        "relation": "VOIDED_OF_OBJECT",
        "evidence": "docs/A_DOCUMENT_THAT_WAS_NEVER_WRITTEN.md",
        "why": "asserted, not recorded", "retained": True,
    }])
    out = tmp_path / "r.json"
    monkeypatch.setattr(sys, "argv", ["x", "--root", str(ROOT), "--output", str(out)])
    reg.main()
    problems = json.loads(out.read_text())["problems"]
    assert any("A_DOCUMENT_THAT_WAS_NEVER_WRITTEN" in p for p in problems), problems


def test_a_fully_superseded_artifact_cited_as_live_evidence_is_rejected(tmp_path, monkeypatch):
    """CONTROL, and the check the whole registry exists for.

    `retention_contrasts` really is cited by the claim lock and really is superseded -- in part.
    Downgrading the exemption must make the problem appear; if it does not, the check is dead and
    the clean run above means nothing.
    """
    monkeypatch.setattr(reg, "HARVEST", dict(
        reg.HARVEST, supersedes_for_multiplicity=("SUPERSEDED_BY_CORRECTIVE_RERUN",
                                                  "host_is_successor")))
    out = tmp_path / "r.json"
    monkeypatch.setattr(sys, "argv", ["x", "--root", str(ROOT), "--output", str(out)])
    reg.main()
    problems = json.loads(out.read_text())["problems"]
    assert any("retention_contrasts" in p and "cited as live evidence" in p
               for p in problems), problems


def test_partial_supersession_must_be_enforced_by_a_companion_claim(tmp_path, monkeypatch):
    """CONTROL. This one caught a real gap: H_regime quoted the transform family's numbers while
    no row cited that artifact, so the partial supersession read as none."""
    lock = json.loads((ROOT / "papers/paper2/claim_lock.json").read_text())
    stripped = {"claims": [{k: v for k, v in c.items() if k != "must_be_cited_with"}
                           for c in lock["claims"]]}
    fake_root = tmp_path / "root"
    (fake_root / "papers/paper2").mkdir(parents=True)
    (fake_root / "papers/paper2/claim_lock.json").write_text(json.dumps(stripped))
    cited, companions = reg.cited_artifacts(fake_root)
    assert companions["H_REGIME_MUST_BE_LABELLED_BY_METRIC"] == set(), (
        "with must_be_cited_with removed there is no companion, which is the defect state")
    assert "results/h_regime_crosswalk/result.json" in cited


def test_lineage_is_not_read_as_supersession():
    """`predecessor` means "replaced" inside one lineage and "built on" across two."""
    assert reg.lineage_stem("results/monotone_transform_family_v4/result.json") == \
        reg.lineage_stem("results/monotone_transform_family_v3/result.json")
    assert reg.lineage_stem("results/citable_risk_attitudes/result.json") != \
        reg.lineage_stem("results/monotone_transform_family_v4/result.json")


def test_lineage_edges_are_recorded_not_dropped(registry):
    """Under-claiming a relation is fine; losing the edge is not."""
    lineage = [e for e in registry["edges"] if e["relation"] == "LINEAGE_NOT_SUPERSESSION"]
    assert lineage, "the citable_risk_attitudes -> transform-family edges must still be recorded"
