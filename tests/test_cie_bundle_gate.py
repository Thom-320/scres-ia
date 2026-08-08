from __future__ import annotations

from pathlib import Path

from scripts.check_cie_bundle_v1 import check_bundle


ROOT = Path(__file__).resolve().parent.parent


def test_cie_gate_accepts_canonical_paper_text() -> None:
    errors = check_bundle(ROOT / "papers/paper2", ROOT / "papers/paper2/claim_lock.json")
    assert not [e for e in errors if "missing claim artifact" in e]


def test_cie_gate_rejects_identity_token(tmp_path: Path) -> None:
    paper = tmp_path / "paper"
    paper.mkdir()
    (paper / "01_introduction.md").write_text("Alice's manuscript")
    lock = tmp_path / "lock.json"
    lock.write_text(json_text())
    assert any("identity token" in e for e in check_bundle(paper, lock, identity_tokens=("Alice",)))


def json_text() -> str:
    return '{"claims": []}'
