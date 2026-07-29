import math

import pytest

from supply_chain.factory_resilience import (
    FactoryResilienceComponents,
    PUBLISHED_EXPONENTS,
    compute_published_factory_resilience,
    normalize_strategy_cost,
    published_log_score,
    published_raw_product,
    published_sigmoid_index,
)


def test_published_coefficients_are_exactly_frozen():
    assert PUBLISHED_EXPONENTS == {
        "zeta": 0.024,
        "epsilon": 0.026,
        "phi": 0.040,
        "tau": 0.060,
        "kappa_dot": 0.1771,
    }


def test_published_equations_are_reproduced_without_des_mapping():
    components = FactoryResilienceComponents(
        zeta=3612.0,
        epsilon=80.0,
        phi=120.0,
        tau=4.0,
        kappa_dot=1.25,
    )
    expected = (
        0.024 * math.log(3612.0)
        - 0.026 * math.log(80.0)
        + 0.040 * math.log(120.0)
        - 0.060 * math.log(4.0)
        - 0.1771 * math.log(1.25)
    )
    assert published_log_score(components) == pytest.approx(expected, abs=1e-15)
    assert published_raw_product(components) == pytest.approx(math.exp(expected))
    assert published_sigmoid_index(components) == pytest.approx(
        1.0 / (1.0 + math.exp(-expected))
    )
    payload = compute_published_factory_resilience(components)
    assert payload["formula_scope"] == "published_equations_3_to_6_only"
    assert payload["published_exponents"] == PUBLISHED_EXPONENTS


def test_published_index_has_the_declared_monotonic_directions():
    base = FactoryResilienceComponents(100.0, 20.0, 30.0, 2.0, 1.0)
    baseline = published_sigmoid_index(base)
    assert published_sigmoid_index(
        FactoryResilienceComponents(110.0, 20.0, 30.0, 2.0, 1.0)
    ) > baseline
    assert published_sigmoid_index(
        FactoryResilienceComponents(100.0, 20.0, 35.0, 2.0, 1.0)
    ) > baseline
    assert published_sigmoid_index(
        FactoryResilienceComponents(100.0, 25.0, 30.0, 2.0, 1.0)
    ) < baseline
    assert published_sigmoid_index(
        FactoryResilienceComponents(100.0, 20.0, 30.0, 3.0, 1.0)
    ) < baseline
    assert published_sigmoid_index(
        FactoryResilienceComponents(100.0, 20.0, 30.0, 2.0, 1.2)
    ) < baseline


def test_strategy_cost_normalization_uses_the_complete_strategy_ensemble():
    costs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    assert normalize_strategy_cost(4.0, costs) == pytest.approx(1.0)
    normalized = [normalize_strategy_cost(cost, costs) for cost in costs]
    assert sum(normalized) / len(normalized) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "components",
    [
        FactoryResilienceComponents(0.0, 1.0, 1.0, 1.0, 1.0),
        FactoryResilienceComponents(1.0, -1.0, 1.0, 1.0, 1.0),
        FactoryResilienceComponents(1.0, 1.0, float("nan"), 1.0, 1.0),
    ],
)
def test_nonpositive_or_nonfinite_components_fail_closed(components):
    with pytest.raises(ValueError):
        published_log_score(components)


def test_strategy_cost_normalization_rejects_an_unbound_cost():
    with pytest.raises(ValueError, match="member"):
        normalize_strategy_cost(4.0, [1.0, 2.0, 3.0])
