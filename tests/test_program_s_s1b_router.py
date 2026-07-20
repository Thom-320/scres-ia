from supply_chain.program_s_execution_router import route_risk_mask


def test_r14_is_direct_until_action_dependence_is_certified() -> None:
    route = route_risk_mask(
        mask="PRODUCTION_QUALITY_SURGE",
        risks=("R11", "R14", "R24"),
        r14_probability_multiplier=1.0,
        certified_masks={"PRODUCTION_QUALITY_SURGE": True},
        r14_action_dependence_certificate=False,
    )
    assert route.engine == "direct_simpy"


def test_certified_action_independent_mask_uses_transducer() -> None:
    route = route_risk_mask(
        mask="LOC_SURGE",
        risks=("R22", "R24"),
        r14_probability_multiplier=1.0,
        certified_masks={"LOC_SURGE": True},
    )
    assert route.engine == "certified_transducer"


def test_uncertified_mask_fails_closed_to_direct_simpy() -> None:
    route = route_risk_mask(
        mask="LOC_SURGE",
        risks=("R22", "R24"),
        r14_probability_multiplier=1.0,
        certified_masks={"LOC_SURGE": False},
    )
    assert route.engine == "direct_simpy"

