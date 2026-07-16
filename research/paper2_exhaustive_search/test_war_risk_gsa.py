"""Validate the GSA core against ANALYTIC ground truth before trusting it on the DES.

The Ishigami function is the canonical benchmark for variance-based sensitivity because its
Sobol indices are known in closed form -- and, crucially for us, it contains a factor (x3) whose
MAIN effect is exactly zero but whose TOTAL effect is 0.244: x3 matters *only through its
interaction with x1*. A one-factor-at-a-time screen perturbing x3 around a nominal point sees
(almost) nothing and declares it irrelevant. That is precisely the failure mode of the completed
Garrido risk screen, reproduced here on a function where we know the right answer.

Run: .venv/bin/python -m pytest research/paper2_exhaustive_search/test_war_risk_gsa.py -q
"""

from __future__ import annotations

import numpy as np

from research.paper2_exhaustive_search.war_risk_gsa import (
    morris_screening,
    sobol_indices,
    prim_box,
)

A_COEF, B_COEF = 7.0, 0.1
ISHIGAMI_BOUNDS = [(-np.pi, np.pi)] * 3
ISHIGAMI_NAMES = ["x1", "x2", "x3"]


def ishigami(x: np.ndarray) -> float:
    return float(
        np.sin(x[0]) + A_COEF * np.sin(x[1]) ** 2 + B_COEF * (x[2] ** 4) * np.sin(x[0])
    )


def _ishigami_analytic() -> dict[str, np.ndarray]:
    pi = np.pi
    v1 = 0.5 * (1 + B_COEF * pi**4 / 5) ** 2
    v2 = A_COEF**2 / 8
    v13 = 8 * B_COEF**2 * pi**8 / 225
    var = A_COEF**2 / 8 + B_COEF * pi**4 / 5 + B_COEF**2 * pi**8 / 18 + 0.5
    return {
        "S1": np.array([v1 / var, v2 / var, 0.0]),
        "ST": np.array([(v1 + v13) / var, v2 / var, v13 / var]),
    }


def test_sobol_matches_ishigami_analytic():
    truth = _ishigami_analytic()
    res = sobol_indices(ishigami, ISHIGAMI_BOUNDS, names=ISHIGAMI_NAMES, n=4096, seed=1)
    assert np.allclose(res.S1, truth["S1"], atol=0.05), f"S1={res.S1} vs {truth['S1']}"
    assert np.allclose(res.ST, truth["ST"], atol=0.05), f"ST={res.ST} vs {truth['ST']}"


def test_sobol_detects_pure_interaction_that_oat_is_blind_to():
    """THE money test: x3 has S3 == 0 (no main effect) but ST3 ~ 0.244 (pure interaction).

    We also simulate an OAT probe of x3 around the nominal point and show it reports ~nothing --
    demonstrating concretely that the OAT design used by the completed risk screen would have
    declared such a factor irrelevant.
    """
    res = sobol_indices(ishigami, ISHIGAMI_BOUNDS, names=ISHIGAMI_NAMES, n=4096, seed=2)
    i3 = res.names.index("x3")

    # variance-based: no main effect, but substantial total effect => interaction mass
    assert res.S1[i3] < 0.05, f"expected ~0 main effect for x3, got {res.S1[i3]}"
    assert res.ST[i3] > 0.15, f"expected substantial total effect for x3, got {res.ST[i3]}"
    assert res.interaction[i3] > 0.15, "interaction mass (ST-S1) must be detected for x3"
    assert not res.additive(), "Ishigami is strongly non-additive; additive() must be False"

    # OAT probe at the nominal point x1=x2=x3=0: vary ONLY x3 across its whole range.
    nominal = np.zeros(3)
    oat_vals = []
    for v in np.linspace(-np.pi, np.pi, 21):
        p = nominal.copy()
        p[2] = v
        oat_vals.append(ishigami(p))
    oat_range = float(np.max(oat_vals) - np.min(oat_vals))
    assert oat_range < 1e-9, (
        "OAT at this nominal point sees x3 as EXACTLY inert (sin(x1)=0 kills the x3 term), "
        f"range={oat_range} -- yet its true total effect is {res.ST[i3]:.3f}. "
        "This is the OAT blindness the war-scenario probe exists to repair."
    )


def test_additive_function_has_no_interaction_mass():
    """If the response is additive, ST ~= S1 for every factor => the OAT null GENERALISES.

    This is the pre-registered 'best outcome for the negative': it would upgrade the certificate
    from 'negative along the axes' to 'negative including interactions'.

    CALIBRATION: ST-S1 is a difference of noisy estimators and has a positive noise floor even at
    zero true interaction (measured: ~0.038 at n=1024, ~0.031 at n=4096, ~0.001-0.012 at n=16384).
    n must therefore be large enough that the floor sits well BELOW the decision tolerance --
    otherwise an 'additive' verdict is just noise. This test pins that calibration.
    """
    def additive(x: np.ndarray) -> float:
        return float(3.0 * x[0] + 2.0 * x[1] - 1.0 * x[2])

    truth_S1 = np.array([9 / 14, 4 / 14, 1 / 14])  # Var(c_i x_i)/Var(f) for uniform inputs
    res = sobol_indices(additive, [(0.0, 1.0)] * 3, names=["a", "b", "c"], n=16384, seed=3)

    # estimator is converged at this n
    assert np.allclose(res.S1, truth_S1, atol=0.03), f"S1={res.S1} vs analytic {truth_S1}"
    # true interaction is exactly zero; the estimate must sit under the calibrated tolerance
    assert np.all(res.interaction <= 0.03), f"interaction={res.interaction} exceeds calibrated tol"
    assert res.additive(tol=0.03), "additive response must be flagged additive at calibrated n/tol"
    assert abs(res.S1.sum() - 1.0) < 0.05, f"first-order indices should sum to ~1, got {res.S1.sum()}"


def test_additivity_verdict_is_noise_at_insufficient_n():
    """GUARD: at small n the interaction floor rivals the tolerance -- an 'additive' claim there
    would be reading noise as a result (the same class of error as reading a default flag as
    data). This test documents the floor so the real run cannot be under-powered by accident."""
    def additive(x: np.ndarray) -> float:
        return float(3.0 * x[0] + 2.0 * x[1] - 1.0 * x[2])

    floors = [
        max(sobol_indices(additive, [(0.0, 1.0)] * 3, n=1024, seed=s).interaction.max() for s in (1, 2, 3))
    ]
    # at n=1024 the floor is materially above the strict tolerance we use at n>=16384
    assert floors[0] > 0.015, (
        f"expected a measurable noise floor at n=1024, got {floors[0]}; if this drops, re-calibrate "
        "the additivity tolerance in war_risk_gsa.SobolResult.additive"
    )


def test_morris_ranks_influence_and_flags_nonadditivity():
    res = morris_screening(ishigami, ISHIGAMI_BOUNDS, names=ISHIGAMI_NAMES, r=40, levels=4, seed=4)
    # x1 and x2 dominate influence; all three have some effect via mu_star ordering
    assert res.mu_star[0] > 0 and res.mu_star[1] > 0
    # sigma is large for the interacting/non-linear factors -> flagged
    flagged = res.interacting(ratio=0.5)
    assert "x1" in flagged, f"x1 interacts with x3 and must be flagged; flagged={flagged}"
    assert res.n_evaluations == 40 * (3 + 1)


def test_prim_recovers_a_known_box():
    """Scenario discovery: plant a region where the target is 1 and check PRIM finds it."""
    rng = np.random.default_rng(0)
    X = rng.random((3000, 3))
    # target is positive only where x0 > 0.7 AND x1 > 0.6  (an INTERACTION region)
    y = ((X[:, 0] > 0.7) & (X[:, 1] > 0.6)).astype(float)
    box = prim_box(X, y, names=["r22", "r24", "noise"], peel_alpha=0.06, min_support=0.03)
    assert box.density > 0.85, f"box should be dense in positives, got {box.density}"
    assert "r22" in box.restricted and "r24" in box.restricted, f"restricted={box.restricted}"
    assert "noise" not in box.restricted, "irrelevant factor must not be restricted"
    lo0, _ = box.limits[0]
    lo1, _ = box.limits[1]
    assert lo0 > 0.5 and lo1 > 0.4, f"box lower limits should approach the true thresholds: {box.limits}"


def test_prim_finds_nothing_when_target_is_noise():
    """Guard against false discovery: pure noise must not yield a dense box."""
    rng = np.random.default_rng(1)
    X = rng.random((2000, 3))
    y = (rng.random(2000) < 0.1).astype(float)  # 10% positives, unrelated to X
    box = prim_box(X, y, names=["a", "b", "c"], peel_alpha=0.05, min_support=0.2)
    assert box.density < 0.35, f"noise must not produce a dense box, got {box.density}"
