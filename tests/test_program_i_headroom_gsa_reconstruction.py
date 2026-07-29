import numpy as np

from scripts.reconstruct_program_i_headroom_gsa_cells import (
    Headroom,
    NAMES,
    reconstruct_morris,
)


def test_morris_reconstruction_reports_every_cell_and_both_targets():
    def evaluate(theta):
        values = np.asarray([theta[name] for name in NAMES], dtype=float)
        return Headroom(
            H_PI=float(values.sum()),
            H_obs=float(values @ np.arange(1, len(values) + 1)),
            eta=0.0,
            n=1,
        )

    rows, summaries = reconstruct_morris(evaluate)

    assert len(rows) == 8 * (len(NAMES) + 1) == 56
    assert [row["cell_index"] for row in rows] == list(range(56))
    assert set(summaries) == {"H_PI", "H_obs"}
    assert set(summaries["H_PI"]) == set(NAMES)
    assert all(len(row["theta"]) == len(NAMES) for row in rows)
    assert all(summary["mu_star"] >= 0 for target in summaries.values() for summary in target.values())
