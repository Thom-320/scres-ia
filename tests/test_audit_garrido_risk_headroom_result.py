from pathlib import Path

from scripts.audit_garrido_risk_headroom_result import (
    approximately_equal,
    audit,
)


def test_numeric_recursive_comparison_is_tight() -> None:
    assert approximately_equal({"x": [1.0]}, {"x": [1.0 + 1e-14]}) == []
    assert approximately_equal({"x": [1.0]}, {"x": [1.1]})


def test_missing_custody_package_fails_closed(tmp_path: Path) -> None:
    contract = tmp_path / "contract.json"
    contract.write_text("{}")
    result = audit(
        tmp_path / "missing-run",
        contract_path=contract,
        expected_source_commit="deadbeef",
    )
    assert result["status"] == "FAIL_GARRIDO_RISK_AUDIT"
    assert "missing_remote_files_sha256" in result["failures"]
