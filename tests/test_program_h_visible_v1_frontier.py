import numpy as np

from scripts.bootstrap_program_h_visible_v1_frontier import bootstrap
from scripts.recompute_program_h_visible_v1_frontier import (
    FIELDS,
    SEQUENCES,
    evaluate_block,
    solve_comparator,
    solve_guardrailed_pi_relaxation,
    tape,
)


def test_visible_v1_repair_enumerates_complete_frontier_and_guardrailed_lps():
    calibration = evaluate_block([tape(index, 1_060_001) for index in range(2)])
    locked = evaluate_block([tape(index, 1_070_001) for index in range(3)])

    assert len(SEQUENCES) == 81
    assert set(calibration) == set(FIELDS)
    assert all(values.shape == (2, 81) for values in calibration.values())
    assert all(values.shape == (3, 81) for values in locked.values())

    comparator = solve_comparator(calibration, worst_fill_margin=0.02)
    weights = comparator["weights"]
    assert np.isclose(weights.sum(), 1.0)
    assert comparator["feasible_deterministic_count"] >= 1

    comparator_rows = {field: values @ weights for field, values in locked.items()}
    pi = solve_guardrailed_pi_relaxation(
        locked, comparator_rows, worst_fill_margin=0.02
    )
    assert pi["solver_objective"] + 1e-10 >= comparator_rows["ret_visible"].mean()
    assert pi["nonzero_weight_count"] >= 3

    values = bootstrap(
        locked,
        comparator["deterministic_index"],
        n_resamples=3,
        seed=20260714,
        worst_fill_margin=0.02,
    )
    assert len(values) == 3
    assert all(np.isfinite(value) for value in values)
