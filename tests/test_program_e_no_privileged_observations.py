from supply_chain.dra2_policy_env import OBSERVATION_KEYS


def test_program_e_observation_contract_has_no_privileged_fields():
    forbidden = ("future", "regime", "oracle", "repair_duration", "next_risk", "ret_excel")
    assert not any(token in key for key in OBSERVATION_KEYS for token in forbidden)
