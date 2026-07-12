from __future__ import annotations

from supply_chain.external_env_interface import (
    ACTION_FIELDS_TRACK_B_V1,
    get_track_b_env_spec,
)


def test_track_b_action_field_order_locked() -> None:
    """Dim->node mapping is a paper-facing claim, not just an implementation
    detail. A silent reorder would misattribute the CDC (Op3) authority to a
    different node without any test failing elsewhere.
    """
    assert ACTION_FIELDS_TRACK_B_V1 == (
        "op3_q_multiplier_signal",
        "op9_q_multiplier_signal",
        "op3_rop_multiplier_signal",
        "op9_rop_multiplier_signal",
        "op5_q_multiplier_signal",
        "assembly_shift_signal",
        "op10_q_multiplier_signal",
        "op12_q_multiplier_signal",
    )


def test_track_b_cdc_dims_are_op3_only() -> None:
    """Locks which dims constitute "authority over the CDC itself" (Op3), the
    exact distinction the post_cdc_only ablation (dims 0 and 2) depends on.
    """
    fields = ACTION_FIELDS_TRACK_B_V1
    cdc_dims = [i for i, name in enumerate(fields) if name.startswith("op3_")]
    assert cdc_dims == [0, 2]


def test_track_b_env_spec_action_fields_match_contract() -> None:
    spec = get_track_b_env_spec()
    assert spec.action_fields == ACTION_FIELDS_TRACK_B_V1
    assert len(spec.action_fields) == 8
