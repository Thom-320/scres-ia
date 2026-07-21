from scripts.run_q2_garrido_native_loc_factorial import (
    PRODUCT_CELLS,
    RISK_LEVELS,
    connected_components,
    make_native_cell,
)


def test_exact_garrido_native_levels_only():
    assert set(RISK_LEVELS.values()) == {(1.0, 1.0), (3.0, 1.0), (1.0, 2.0), (3.0, 2.0)}
    for product in PRODUCT_CELLS:
        for r22, r24 in RISK_LEVELS.values():
            cell = make_native_cell(r22, r24, product)
            assert cell.mask == "LOC_SURGE"
            assert cell.coupling == "independent"
            assert cell.phi_by_risk == {"R22": r22, "R24": r24}
            assert cell.psi_by_risk == {"R22": 1.0, "R24": 1.0}


def test_connectivity_requires_one_axis_native_move():
    product = "rho75_share90"
    rows = [
        {"level_id": level, "product_cell": product, "cell_pass": level != "r22_increased__r24_increased"}
        for level in RISK_LEVELS
    ]
    components = connected_components(rows)
    assert len(components[0]) == 3


def test_product_cells_are_not_treated_as_physical_neighbors():
    rows = [
        {"level_id": "r22_current__r24_current", "product_cell": product, "cell_pass": True}
        for product in PRODUCT_CELLS
    ]
    assert sorted(map(len, connected_components(rows))) == [1, 1, 1]
