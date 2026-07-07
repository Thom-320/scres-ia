from scripts.audit_track_b_prevention_mechanism import (
    aggregate_policy_windows,
    infer_event_anchor,
    window_label,
)


def _rows(policy: str, baseline: float, pre: float, post: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for step, rel in enumerate(range(-6, 10)):
        if -4 <= rel <= -1:
            intensity = pre
        elif 1 <= rel <= 8:
            intensity = post
        else:
            intensity = baseline
        rows.append(
            {
                "policy": policy,
                "step": step,
                "relative_week": rel,
                "event_window": window_label(rel),
                "action_intensity": intensity,
            }
        )
    return rows


def _cf(policy: str, pre_delta: float, post_delta: float) -> list[dict[str, object]]:
    return [
        {"policy": policy, "reset_window": "pre", "delta_ret_excel": pre_delta},
        {"policy": policy, "reset_window": "event", "delta_ret_excel": 0.0},
        {"policy": policy, "reset_window": "post", "delta_ret_excel": post_delta},
    ]


def test_window_label_boundaries() -> None:
    assert window_label(-4) == "pre"
    assert window_label(-1) == "pre"
    assert window_label(0) == "event"
    assert window_label(8) == "post"
    assert window_label(9) == "baseline"


def test_static_policy_classifies_as_no_clear_signal() -> None:
    summary = aggregate_policy_windows(
        _rows("static", baseline=0.4, pre=0.4, post=0.4),
        _cf("static", pre_delta=0.0, post_delta=0.0),
        [],
    )
    assert summary[0]["classification"] == "sin señal clara"


def test_preventive_policy_needs_pre_activation_and_positive_pre_delta() -> None:
    summary = aggregate_policy_windows(
        _rows("preventive", baseline=0.3, pre=0.5, post=0.5),
        _cf("preventive", pre_delta=0.001, post_delta=0.0),
        [],
    )
    assert summary[0]["classification"] == "preventiva"


def test_reactive_policy_needs_post_activation_and_positive_post_delta() -> None:
    summary = aggregate_policy_windows(
        _rows("reactive", baseline=0.3, pre=0.3, post=0.55),
        _cf("reactive", pre_delta=0.0, post_delta=0.001),
        [],
    )
    assert summary[0]["classification"] == "reactiva"


def test_infer_event_anchor_prefers_regime_transition_over_forecast() -> None:
    rows = [
        {"step": 0, "forecast_168h": 0.9, "regime_pre_disruption": 0, "regime_disrupted": 0},
        {"step": 3, "forecast_168h": 0.1, "regime_pre_disruption": 1, "regime_disrupted": 0},
    ]
    assert infer_event_anchor(rows) == 3
