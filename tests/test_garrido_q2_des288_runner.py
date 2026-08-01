from __future__ import annotations

import numpy as np

from scripts.run_garrido_q2_des288_v1 import (
    CONTEXT_ORDER,
    DEFAULT,
    FACTOR_NAMES,
    SERVICE_FIRST_V2_COMPONENTS,
    _paired,
    _falsifiers,
    search,
    selected_configs,
)


def _synthetic_surface(configs, contexts, seeds):
    surface = {}
    for context in contexts:
        for seed in seeds:
            rows = []
            for i, config in enumerate(configs):
                flow = 0.50 + 0.001 * i + 0.00001 * (seed - seeds[0])
                backorder = 1_000.0 + i
                rows.append(
                    {
                        "config": dict(config),
                        "context": context,
                        "seed": seed,
                        "service_key": [flow, flow, -backorder, 0.1],
                        "claimant_fills": {},
                        "demanded_by_claimant": {},
                        "delivered_by_claimant": {},
                        "cssu_total_demanded": 0.0,
                        "cssu_total_delivered": 0.0,
                        "drivers": [float(i), 0.0, 0.0, 0.0],
                        "panel": {
                            "flow_fill_rate": flow,
                            "backorder_qty_final": backorder,
                            "ret_excel_visible_clipped_0_1": 0.1,
                            "demanded_rations": 0.0,
                            "delivered_rations": 0.0,
                        },
                    }
                )
            surface[(context, seed)] = rows
    return surface


def test_smoke_surface_keeps_ofat_proposals_and_full_surface_has_288_configs():
    assert len(selected_configs(None)) == 288
    smoke = selected_configs(8)
    assert len(smoke) >= 8
    for name, levels in {
        key: tuple(values) for key, values in {
            "buffer_hours": (0.0, 168.0, 336.0, 504.0, 672.0, 1344.0),
            "shifts": (1, 2, 3),
            "op9_rop": (12.0, 24.0, 36.0, 48.0),
            "op12_rop": (12.0, 24.0, 36.0, 48.0),
        }.items()
    }.items():
        assert all(
            any(dict(candidate, **{name: level}) == candidate for candidate in smoke)
            or any(candidate[name] == level for candidate in smoke)
            for level in levels
        )
    assert DEFAULT["op9_rop"] == 24.0
    assert tuple(DEFAULT[name] for name in FACTOR_NAMES)


def test_synthetic_q2_contract_falsifiers_pass_without_running_des():
    configs = selected_configs(8)
    contexts = (CONTEXT_ORDER[0],)
    seeds = [7_100_001, 7_100_002]
    surface = _synthetic_surface(configs, contexts, seeds)
    results = {
        strategy: [
            search(
                strategy,
                seed,
                np.random.default_rng(90_000 + repeat),
                surface,
                configs,
                contexts,
                4,
            )
            for repeat, seed in enumerate(seeds)
        ]
        for strategy in ("ofat", "random", "no_update", "retained", "reset")
    }
    falsifiers = _falsifiers(
        surface=surface,
        results=results,
        configs=configs,
        contexts=contexts,
        seeds=seeds,
        budget=4,
        rng=np.random.default_rng(20260801),
    )
    assert all(check["passed"] for check in falsifiers.values())


def test_zero_budget_retained_and_reset_have_identical_trace_contract():
    configs = selected_configs(8)
    contexts = (CONTEXT_ORDER[0],)
    seeds = [7_100_001]
    surface = _synthetic_surface(configs, contexts, seeds)
    retained = search("retained", seeds[0], np.random.default_rng(1), surface, configs, contexts, 0)
    reset = search("reset", seeds[0], np.random.default_rng(1), surface, configs, contexts, 0)
    assert retained["per_context"] == reset["per_context"]


def test_primary_orientation_is_reset_minus_retained_for_faster_memory():
    """The preregistered primary is positive when retained needs fewer runs."""
    results = {
        "retained": [
            {"per_context": {"R1r": {"runs_to_oracle": 2.0}}},
            {"per_context": {"R1r": {"runs_to_oracle": 3.0}}},
        ],
        "reset": [
            {"per_context": {"R1r": {"runs_to_oracle": 4.0}}},
            {"per_context": {"R1r": {"runs_to_oracle": 5.0}}},
        ],
    }
    delta = _paired(
        results,
        "retained",
        "reset",
        ("R1r",),
        "runs_to_oracle",
        rng=np.random.default_rng(3),
        n_boot=100,
        sign="b_minus_a",
    )
    assert delta["mean"] == 2.0
    assert delta["lcb95"] > 0.0
