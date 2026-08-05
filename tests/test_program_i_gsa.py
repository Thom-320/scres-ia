"""Program I GSA acceptance tests (per the 2026-07-12 dictamen).

- Sobol/Morris recover known indices on Ishigami (estimator correctness).
- A 'very sensitive but constant-optimal' headroom must NOT promote (sensitivity != headroom).
- A 'ranking-reversal' headroom (observable conversion > 0) must promote past the gate.
- Dependent-input caveat: the stylized Program-I factors are INDEPENDENT, so standard Sobol is
  valid here; dependent-input indices are only needed for the deferred full-DES lane (Q/ROP/capacity).
"""
import numpy as np

from supply_chain.gsa import gp_locate, sobol_indices, morris_screen, ishigami, ISHIGAMI_BOUNDS


def test_sobol_recovers_ishigami_indices():
    s = sobol_indices(ishigami, ISHIGAMI_BOUNDS, N=1024, seed=1)
    # analytic: S1 ~ [0.314, 0.442, 0.0]; ST ~ [0.558, 0.442, 0.244]
    assert abs(s["x1"]["S1"] - 0.314) < 0.08
    assert abs(s["x2"]["S1"] - 0.442) < 0.08
    assert s["x3"]["S1"] < 0.08                      # x3 has ~0 first-order
    assert s["x3"]["ST"] > 0.12                      # but non-zero total (pure interaction)
    assert s["x1"]["interaction_gap"] > 0.10         # x1<->x3 interaction detected


def test_morris_ranks_active_factors_above_inert():
    # f depends only on x1; x2 is inert -> mu_star(x1) >> mu_star(x2)
    bounds = [(0.0, 1.0, "x1"), (0.0, 1.0, "x2")]
    m = morris_screen(lambda x: 3.0 * x[0], bounds, r=20, seed=0)
    assert m["x1"]["mu_star"] > 10 * max(m["x2"]["mu_star"], 1e-9)


def test_gp_locate_returns_history_and_declares_grid_snapping():
    grid = np.array([[x, y] for x in (0.0, 0.5, 1.0) for y in (0.0, 0.5, 1.0)])
    result = gp_locate(
        lambda point: 1.0 - (point[0] - 0.75) ** 2 - (point[1] - 0.25) ** 2,
        [(0.0, 1.0), (0.0, 1.0)],
        n_init=3,
        n_iter=4,
        seed=11,
        grid=grid,
    )

    assert result["grid_anchored"] is True
    assert result["n_init"] == 3
    assert result["n_eval"] == len(result["history"]) == 7
    assert len({tuple(row["x"]) for row in result["history"]}) == result["n_eval"]
    assert all(row["source"] in {"lhs_snap", "ei_snap"} for row in result["history"])
    assert result["duplicate_policy"].startswith("skip_duplicate_grid_points")


def _gate(H_obs_samples, worst_fill_delta=0.0, delta_min=0.01):
    """Program-I promotion gate: LCB95(H_obs) >= delta_min AND fairness guardrail holds."""
    x = np.asarray(H_obs_samples)
    rng = np.random.default_rng(0)
    lcb = np.percentile([rng.choice(x, len(x), True).mean() for _ in range(2000)], 2.5)
    return bool(lcb >= delta_min and worst_fill_delta > -0.02)


def test_sensitive_but_constant_optimal_does_not_promote():
    # Clairvoyant headroom exists (output is sensitive) but the observable policy loses:
    # H_obs samples centered at -0.02 -> gate must STOP (this is the Program G/H anchor case).
    rng = np.random.default_rng(1)
    H_obs = rng.normal(-0.02, 0.01, 300)
    assert _gate(H_obs) is False


def test_ranking_reversal_promotes_only_if_fair():
    # Observable conversion is positive AND stable -> passes the H_obs gate...
    rng = np.random.default_rng(2)
    H_obs = rng.normal(0.03, 0.008, 300)
    assert _gate(H_obs, worst_fill_delta=0.0) is True
    # ...but if it wins by starving a node (worst-fill delta -0.13), the fairness guardrail STOPS it
    # (the actual Program-I GP-located region behaves exactly this way).
    assert _gate(H_obs, worst_fill_delta=-0.13) is False
