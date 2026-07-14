import json
from pathlib import Path

from scripts.screen_program_o_exact_transducer import (
    complete_calendars,
    make_tape,
    simulate,
)


ROOT = Path(__file__).resolve().parent.parent
CONTRACT = json.loads(
    (ROOT / "contracts/program_o_exact_transducer_v1.json").read_text()
)


def test_complete_binary_frontier_and_resource_identity():
    calendars = complete_calendars()
    assert len(calendars) == 2**8 == CONTRACT["complete_open_loop_calendars"]
    tape = make_tape(1, persistence=0.9, dominant_share=0.9)
    first = simulate(tape, calendars[0], CONTRACT, complete_substitution=False)
    last = simulate(tape, calendars[-1], CONTRACT, complete_substitution=False)
    assert first["production_batches"] == last["production_batches"] == 24
    assert first["production_quantity"] == last["production_quantity"] == 120000.0


def test_fungible_null_is_exact_for_every_calendar():
    tape = make_tape(2, persistence=0.9, dominant_share=0.9)
    results = [
        simulate(tape, calendar, CONTRACT, complete_substitution=True)
        for calendar in complete_calendars()
    ]
    keys = (
        "ret",
        "ret_full",
        "visible_rows",
        "unfulfilled_orders",
        "unfulfilled_quantity",
        "worst_product_fill",
        "ending_inventory",
    )
    assert all(
        tuple(row[key] for key in keys) == tuple(results[0][key] for key in keys)
        for row in results
    )


def test_tape_is_deterministic_and_total_demand_is_fixed():
    first = make_tape(3, persistence=0.75, dominant_share=0.75)
    second = make_tape(3, persistence=0.75, dominant_share=0.75)
    assert first == second
    assert len(first.order_products) == 8 * 6
    assert set(first.order_products) <= {"P_C", "P_H"}
