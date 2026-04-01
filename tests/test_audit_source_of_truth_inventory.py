from __future__ import annotations

from pathlib import Path

from scripts.audit_source_of_truth_inventory import (
    choose_reference_label,
    extract_doc_references,
    recommended_action,
)


def test_extract_doc_references_normalizes_bundle_and_non_bundle(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path
    doc = repo_root / "docs" / "note.md"
    bundle_dir = repo_root / "docs" / "artifacts" / "control_reward" / "paper_run"
    bundle_dir.mkdir(parents=True)
    bundle_summary = bundle_dir / "summary.json"
    bundle_summary.write_text("{}", encoding="utf-8")
    seed_note = (
        repo_root
        / "docs"
        / "artifacts"
        / "control_reward"
        / "control_reward_500k_seed_inference"
        / "seed_inference.md"
    )
    seed_note.parent.mkdir(parents=True)
    seed_note.write_text("note", encoding="utf-8")
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(
        "`docs/artifacts/control_reward/paper_run` and "
        "`docs/artifacts/control_reward/control_reward_500k_seed_inference/seed_inference.md`",
        encoding="utf-8",
    )

    refs = extract_doc_references(doc, repo_root)

    assert len(refs) == 2
    bundle_ref = next(ref for ref in refs if ref["bundle_root"] is not None)
    assert Path(bundle_ref["bundle_root"]).name == "paper_run"
    note_ref = next(ref for ref in refs if ref["bundle_root"] is None)
    assert note_ref["reference_type"] == "non_bundle_reference"


def test_choose_reference_label_prefers_kappa_lane() -> None:
    rows = [
        {"label": "paper_control_v1_500k", "audit_status": "auditable"},
        {"label": "paper_ret_seq_k020_500k", "audit_status": "auditable"},
    ]

    assert choose_reference_label(rows) == "paper_ret_seq_k020_500k"


def test_recommended_action_demotes_historical() -> None:
    audited_row = {"audit_status": "historical_artifact"}

    assert recommended_action(audited_row, None) == "demote_to_historical"
