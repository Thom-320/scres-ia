from __future__ import annotations

import math

from scripts.calibrate_cd_exponents import calibrate_from_rows


def test_calibrate_from_rows_uses_garrido_maxima_rule() -> None:
    rows = [
        {
            "zeta_avg": 100.0,
            "epsilon_avg": 50.0,
            "phi_avg": 20.0,
            "tau_avg": 10.0,
            "average_cost": 40.0,
        },
        {
            "zeta_avg": 200.0,
            "epsilon_avg": 80.0,
            "phi_avg": 30.0,
            "tau_avg": 12.0,
            "average_cost": 60.0,
        },
    ]

    payload = calibrate_from_rows(rows)

    assert payload["kappa_ref"] == 50.0
    assert payload["maxima"]["zeta"] == 200.0
    assert payload["maxima"]["epsilon"] == 80.0
    assert payload["maxima"]["phi"] == 30.0
    assert payload["maxima"]["tau"] == 12.0
    assert payload["maxima"]["kappa_dot"] == 1.2
    assert payload["a_zeta"] == payload["target_contribution"] / math.log(200.0)
    assert payload["b_epsilon"] == payload["target_contribution"] / math.log(80.0)
    assert payload["c_phi"] == payload["target_contribution"] / math.log(30.0)
    assert payload["d_tau"] == payload["target_contribution"] / math.log(12.0)
    assert payload["n_kappa"] == payload["target_contribution"] / math.log(1.2)
