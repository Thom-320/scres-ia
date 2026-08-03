from pathlib import Path

from scripts.adjudicate_g3_obs_contract_scope import build_adjudication


ROOT = Path(__file__).resolve().parents[1]


def test_scope_adjudication_preserves_mismatch_and_supplements_f2():
    payload = build_adjudication(
        ROOT / "results/headroom/g3_obs_conversion_v2/result.json",
        ROOT / "results/headroom/g3_obs_conversion_v2/f2_audit_result.json",
        ROOT / "docs/PREREGISTRO_G3_OBS_CONVERSION_OBSERVABLE_2026-08-01.md",
        ROOT / "docs/PREREGISTRO_G3_OBS_V2_POTENCIA_2026-08-02.md",
        ROOT / "docs/CONTRATO_ADJUDICACION_ALCANCE_G3_OBS_2026-08-02.md",
    )

    assert payload["claim_status"] == "SOURCE_ARTIFACT_PRESERVED_SCOPE_MISMATCH_NOT_PROMOTABLE"
    assert payload["promotion_status"] == "BLOCKED_NO_RETROACTIVE_RESEAL_AND_NO_CONTRACT_CONFORMITY"
    assert payload["scope_checks"]["source_seal_matches_legacy_contract"] is True
    assert payload["scope_checks"]["source_seal_matches_intended_v2"] is False
    assert payload["scope_checks"]["source_execution_matches_legacy_declared_scope"] is False
    assert payload["scope_checks"]["source_execution_matches_v2_fields"] is True
    assert payload["supplemental_f2_audit"]["all_cells_passed"] is True
    assert payload["custody"]["retroactive_reseal"] is False
    assert payload["custody"]["new_seeds_opened"] is False
    assert payload["custody"]["des_rerun"] is False


def test_scope_adjudication_prohibits_reclassification_as_v2():
    payload = build_adjudication(
        ROOT / "results/headroom/g3_obs_conversion_v2/result.json",
        ROOT / "results/headroom/g3_obs_conversion_v2/f2_audit_result.json",
        ROOT / "docs/PREREGISTRO_G3_OBS_CONVERSION_OBSERVABLE_2026-08-01.md",
        ROOT / "docs/PREREGISTRO_G3_OBS_V2_POTENCIA_2026-08-02.md",
        ROOT / "docs/CONTRATO_ADJUDICACION_ALCANCE_G3_OBS_2026-08-02.md",
    )

    assert "The source run was executed or sealed under the v2 contract." in payload["prohibited_claims"]
    assert payload["scope_checks"]["source_execution_is_v2_confirmatory"] is False
