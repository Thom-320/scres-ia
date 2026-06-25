from __future__ import annotations

from pathlib import Path


def test_paper_contract_is_falsifiable_not_directional_win() -> None:
    text = Path("docs/PAPER_CONTRACT_2026-06-24.md").read_text(encoding="utf-8")

    assert "Primary Hypothesis" in text
    assert "The null result is admissible" in text
    assert "No outcome direction is required" in text
    assert "held-out evaluation result" in text
    assert "retune the environment" in text
    assert "claim is supported only if" not in text
    assert "directional win" not in text.lower()


def test_paper_contract_separates_reference_from_extension() -> None:
    text = Path("docs/PAPER_CONTRACT_2026-06-24.md").read_text(encoding="utf-8")

    assert "thesis-anchored experimental platform" in text
    assert "`des_reference_v1`" in text
    assert "`learning_extension_v1`" in text
    assert "Default decision cadence: `block`" in text
    assert "Track B downstream-control experiments are outside" in text


def test_paper_contract_freezes_learning_regime_and_dose_response() -> None:
    text = Path("docs/PAPER_CONTRACT_2026-06-24.md").read_text(encoding="utf-8")
    env_spec = Path("docs/environment_spec.md").read_text(encoding="utf-8")

    for document in (text, env_spec):
        normalized = document.lower()
        assert "d/d_rho" in normalized
        assert (
            "persistent disruption regime" in normalized
            or "persistent campaign-phase" in normalized
        )
        assert "stochastic ration demand" in normalized
        assert (
            "training/calibration" in normalized
            or "training scenario tapes" in normalized
        )
