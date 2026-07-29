from scripts.audit_paper2_bottleneck_effect_quotient import (
    count_effect_words,
    effect_codes,
    full_calendar_count,
)
from supply_chain.paper2_bottleneck import CONTEXTS, materialize_tape


def test_full_calendar_count_matches_corrective_audit():
    assert full_calendar_count() == 11_184_811


def test_null_event_tape_collapses_every_calendar_to_one_effect_word():
    tape = materialize_tape(1_098_001, CONTEXTS[0], "disposable_unit", weeks=24)
    tape["base_events"] = []
    assert effect_codes(tape) == [(0, 0, 0)] * 24
    assert count_effect_words(tape) == 1


def test_all_three_event_types_make_current_action_distinguishable():
    tape = materialize_tape(1_098_002, CONTEXTS[0], "disposable_unit", weeks=24)
    tape["base_events"] = [
        {"onset_hours": 2.0, "risk_id": "R11"},
        {"onset_hours": 3.0, "risk_id": "R22"},
        {"onset_hours": 4.0, "risk_id": "R24"},
    ]
    assert effect_codes(tape)[0] == (1, 2, 4)
