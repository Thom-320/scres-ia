import numpy as np

from scripts.screen_program_o_fixed_clock_hobs_validation import (
    decode_calendar,
    joint_bootstrap,
)


def test_decode_calendar_is_inverse_of_base4_indexing():
    assert decode_calendar(0) == (0,) * 8
    assert decode_calendar(65535) == (3,) * 8
    assert decode_calendar(39599) == (2, 1, 2, 2, 2, 2, 3, 3)


def test_joint_bootstrap_reselects_full_available_frontier_and_is_simultaneous():
    cells = ("rho75_share90", "rho90_share75", "rho90_share90")
    panels = {}
    rows = {}
    placebos = {}
    for offset, cell_id in enumerate(cells):
        # Four synthetic calendars are enough to exercise exact re-selection;
        # the production run supplies all 65,536 columns without pruning.
        base = np.linspace(0.4, 0.6, 48)[:, None]
        ret = np.hstack(
            [base, base + 0.01, base + 0.02, base + 0.03 + offset * 0.001]
        )
        panel = {
            "ret_visible": ret,
            "ration_ret_visible": ret,
            "ret_full": ret,
            "quantity_ret_full": ret,
            "ret_visible_cvar10": ret,
            "worst_product_fill": ret,
            "omitted_rows": 1.0 - ret,
            "omitted_quantity": 1.0 - ret,
            "lost_orders": 1.0 - ret,
            "lost_quantity": 1.0 - ret,
            "unresolved_orders": 1.0 - ret,
            "unresolved_quantity": 1.0 - ret,
            "max_backlog_age": 1.0 - ret,
            "service_loss_auc": 1.0 - ret,
        }
        # Fill unused matrix keys with a harmless copy.
        from supply_chain.program_o_full_des_transducer import MATRIX_KEYS

        for key in MATRIX_KEYS:
            panel.setdefault(key, ret)
        panels[cell_id] = panel
        rows[cell_id] = {"calendar_indices": [3] * 48}
        contrast_index = [2] * 48
        placebos[cell_id] = {
            "total_information": {
                "placebos": {
                    "no_state": {"calendar_indices": contrast_index}
                }
            }
        }

    result = joint_bootstrap(
        panels=panels,
        rows=rows,
        placebo_rows=placebos,
        cells=cells,
        resamples=100,
    )
    assert result["static_reselected_each_resample"] is True
    assert result["estimand_count"] > len(cells)
    assert all(
        np.isfinite(row["simultaneous_lcb95"])
        for row in result["estimates"].values()
    )
