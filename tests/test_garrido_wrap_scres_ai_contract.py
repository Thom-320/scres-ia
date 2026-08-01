from __future__ import annotations
import json
from pathlib import Path

from openpyxl import Workbook
import pytest

from scripts.audit_garrido_wrap_sources import build_audit
from scripts.build_garrido_fig5_surrogate import (
    paired_difference_summary,
    q1_decision,
)
from scripts.run_garrido_wrap_closed_loop import (
    BetweenRunLearner,
    campaign_groups,
    candidate_table,
    load_oracle,
    run_arm,
)


def _mini_workbook(path: Path, sheet_name: str) -> None:
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = sheet_name
    worksheet.append(["Cfi", sheet_name, None, "Q", "j", "OPTj", "OATj", "CTj"])
    worksheet.append([1, sheet_name, None, 1, 1, 0, 1, 1])
    workbook.save(path)


def test_source_audit_exposes_coverage_and_metric_hold(tmp_path: Path) -> None:
    paths = {
        "garrido_2024_scres_ai": tmp_path / "scres.pdf",
        "garrido_2024_factory_resilience": tmp_path / "factory.pdf",
        "v0_pdf": tmp_path / "v0.pdf",
        "v0_docx": tmp_path / "v0.docx",
        "raw_data1": tmp_path / "raw1.xlsx",
        "raw_data2": tmp_path / "raw2.xlsx",
        "rsult_1": tmp_path / "rsult.xlsx",
        "wrap_thesis": tmp_path / "wrap.pdf",
    }
    for key, path in paths.items():
        if key == "raw_data1":
            _mini_workbook(path, "CF1")
        elif key == "raw_data2":
            _mini_workbook(path, "CF11")
        elif key == "rsult_1":
            _mini_workbook(path, "Cf1")
        else:
            path.write_bytes(key.encode("utf-8"))

    payload = build_audit(source_paths=paths)

    assert payload["claim_status"] == "DEVELOPMENT_SOURCE_AUDIT"
    assert payload["cf_coverage"]["author_delivered_cf"] == [1, 11]
    assert payload["cf_coverage"]["repo_design_count"] == 90
    assert payload["metric_contract"]["sumBt_status"] == "UNRESOLVED_SOURCE_SEMANTICS"
    assert payload["gate_status"]["behavioral_fidelity"] == "HOLD_WRAP_BEHAVIORAL_FIDELITY"


def test_paired_q1_rule_requires_sesoi_and_positive_ci() -> None:
    task = {
        "per_fold": {
            "linear": [0.50, 0.52, 0.51, 0.49, 0.50],
            "backprop": [0.56, 0.58, 0.57, 0.55, 0.56],
            "kan": [0.51, 0.52, 0.50, 0.51, 0.52],
        }
    }
    summary = paired_difference_summary(task, "backprop", sesoi=0.05)
    assert summary["mean_difference"] == pytest.approx(0.06)
    assert summary["passes_sesoi_and_ci"] is True

    decision = q1_decision(task, task, sesoi_r2=0.05)
    assert decision["decision"] == "PASS_Q1_NEURAL_PREMIUM"
    assert decision["selected_model_before_gates"] == "backprop"
    assert decision["promotion_eligible"] is False


def test_q1_rule_rejects_small_neural_gain() -> None:
    task = {
        "per_fold": {
            "linear": [0.50, 0.52, 0.51, 0.49, 0.50],
            "backprop": [0.51, 0.53, 0.52, 0.50, 0.51],
            "kan": [0.50, 0.52, 0.51, 0.49, 0.50],
        }
    }
    decision = q1_decision(task, task, sesoi_r2=0.05)
    assert decision["decision"] == "NO_GO_NEURAL_PREMIUM_IN_WRAP_PANEL"
    assert decision["selected_model_before_gates"] == "linear_null"


def test_retained_and_reset_state_have_distinct_campaign_semantics() -> None:
    retained = BetweenRunLearner("linear", seed=1, retained=True)
    reset = BetweenRunLearner("linear", seed=1, retained=False)
    retained.observe(1, 0.1)
    reset.observe(1, 0.1)

    retained.start_campaign()
    reset.start_campaign()

    assert retained.observations == [(1, 0.1)]
    assert reset.observations == []

    no_update = BetweenRunLearner("linear", seed=1, retained=True, update=False)
    no_update.observe(1, 0.1)
    assert no_update.observations == []


def test_between_run_arm_never_uses_unobserved_outcomes() -> None:
    candidates = candidate_table(fallback_seed_base=900_000)
    campaigns = [("R1r", [1, 2, 3])]
    oracle = {1: 0.1, 2: 0.9, 3: 0.2}
    calls: list[int] = []

    def fake_simulate(candidate):
        calls.append(candidate.cf)
        return {"cf": candidate.cf, "ret_excel": oracle[candidate.cf]}

    arm = run_arm(
        "retained",
        candidates,
        campaigns,
        budget=2,
        learner_kind="linear",
        learner_seed=7,
        retained=True,
        update=True,
        horizon_hours=None,
        oracle=oracle,
        simulate=fake_simulate,
    )

    assert len(calls) == 2
    assert len(set(calls)) == 2
    assert all(record["cf"] in calls for record in arm["records"])
    assert arm["campaigns"][0]["oracle_best"] == pytest.approx(0.9)


def test_q2_dry_run_contract_can_load_development_oracle(tmp_path: Path) -> None:
    path = tmp_path / "oracle.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "test",
                "claim_status": "DEVELOPMENT_DRIVER_TABLE",
                "rows": [{"cf": 1, "ret_excel": 0.5}],
            }
        )
    )
    values, metadata = load_oracle(path)
    assert values == {1: 0.5}
    assert metadata["claim_status"] == "DEVELOPMENT_DRIVER_TABLE"


def test_campaign_order_is_reproducible() -> None:
    candidates = candidate_table(fallback_seed_base=900_000)
    first = campaign_groups(candidates, order="shuffled", max_campaigns=None, seed=42)
    second = campaign_groups(candidates, order="shuffled", max_campaigns=None, seed=42)
    assert first == second
