from __future__ import annotations

import json
from pathlib import Path

from scripts.run_garrido_risk_headroom_sensitivity import (
    CF_R1,
    CF_R2,
    R1,
    R2,
    build_profiles,
    make_env,
)


def test_exact_thesis_frequency_profiles_are_present() -> None:
    by_id = {profile["id"]: profile for profile in build_profiles()}

    assert by_id["R1_current"]["overrides"] == {risk: "current" for risk in R1}
    assert by_id["R2_current"]["overrides"] == {risk: "current" for risk in R2}
    for cf, signs in {**CF_R1, **CF_R2}.items():
        risks = R1 if cf in CF_R1 else R2
        expected = {
            risk: ("increased" if sign == "+" else "current")
            for risk, sign in zip(risks, signs)
        }
        assert by_id[cf]["overrides"] == expected


def test_black_swan_is_absent_from_every_profile() -> None:
    for profile in build_profiles():
        assert "R3" not in profile["enabled"]
        assert "R3" not in profile["overrides"]
        assert "R3" not in profile["impact"]


def test_impact_profiles_change_only_the_declared_risk() -> None:
    by_id = {profile["id"]: profile for profile in build_profiles()}
    for risk in ("R11", "R21", "R22", "R23", "R24"):
        profile = by_id[f"impact_{risk}_psi2"]
        assert profile["impact"] == {risk: 2.0}
        assert set(profile["overrides"].values()) == {"current"}


def test_contract_keeps_canonical_metric_and_ten_year_horizon() -> None:
    contract = json.loads(
        Path("contracts/garrido_risk_headroom_sensitivity_v1.json").read_text()
    )
    assert contract["metric"]["primary"] == "ret_excel_request_snapshot_v2"
    assert contract["development"]["max_steps"] == 520
    assert contract["learner_authorized"] is False
    assert contract["paper2_confirmed"] is False
    assert contract["paper3_authorized"] is False


def test_screen_uses_independent_per_risk_rng_streams() -> None:
    profile = next(item for item in build_profiles() if item["id"] == "Cf1")
    env = make_env(profile, seed=7, max_steps=1)
    try:
        assert env.unwrapped.sim.risk_rng_mode == "per_risk"
        assert env.unwrapped.sim.strict_exogenous_crn is True
    finally:
        env.close()
