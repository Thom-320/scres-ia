import numpy as np

from scripts.recompute_program_j_visible_v1_frontier import (
    FIELDS,
    REFERENCE,
    SEQUENCES,
    solve_pi,
    solve_static_mixture,
)


def test_program_j_repair_uses_complete_3_to_8_frontier_and_guardrailed_lps():
    assert len(SEQUENCES) == 3**8 == 6561
    assert REFERENCE in SEQUENCES
    rng = np.random.default_rng(20260714)
    arrays = {
        "ret_visible": rng.uniform(0.0, 0.01, size=(2, len(SEQUENCES))),
        "ret_quantity": np.full((2, len(SEQUENCES)), 0.5),
        "ret_cvar05": np.full((2, len(SEQUENCES)), 0.001),
        "lost_orders": np.zeros((2, len(SEQUENCES))),
        "service_loss_auc": np.full((2, len(SEQUENCES)), 10.0),
        "flow_fill_rate": np.full((2, len(SEQUENCES)), 0.75),
        "executed_pm_hours": np.full((2, len(SEQUENCES)), 192.0),
        "corrective_hours": np.zeros((2, len(SEQUENCES))),
        "mass_residual": np.zeros((2, len(SEQUENCES))),
    }
    assert set(arrays) == set(FIELDS)

    static = solve_static_mixture(arrays)
    weights = static["weights"]
    assert np.isclose(weights.sum(), 1.0)
    comparator = {field: rows @ weights for field, rows in arrays.items()}
    pi = solve_pi(arrays, comparator)

    assert pi["objective"] + 1e-12 >= comparator["ret_visible"].mean()
    assert (
        pi["expected"]["corrective_hours"].mean()
        <= comparator["corrective_hours"].mean() + 1e-9
    )
    assert static["feasible_deterministic_count"] == len(SEQUENCES)
