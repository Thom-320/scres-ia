from scripts.run_q_r1_cold_start_replication import adjudicate


def _contrast(mean, lcb=0.02, fill=0.0):
    return {
        "mean_early_ret_delta": mean,
        "history_clustered_lcb95": lcb,
        "favorable_fraction": 0.8,
        "mean_worst_product_delta": fill,
        "max_unresolved_orders_delta": 0.0,
        "max_lost_orders_delta": 0.0,
        "max_resource_error": 0.0,
        "n_pairs": 10,
    }


def test_adjudication_requires_both_retention_and_residual(monkeypatch):
    values = {
        (0.5, "retained_posterior"): _contrast(0.0),
        (0.75, "retained_posterior"): _contrast(0.012),
        (0.9, "retained_posterior"): _contrast(0.018),
        (0.9, "shuffled_posterior"): _contrast(0.0),
        (0.9, "wrong_posterior"): _contrast(-0.01),
    }
    monkeypatch.setattr(
        "scripts.run_q_r1_cold_start_replication.retained_contrast",
        lambda _rows, kappa, arm: values[(kappa, arm)],
    )
    result = adjudicate(
        {"rows": []},
        {"primary_persistent": {"history_clustered_lcb95": 0.015, "action_divergence": 0.2}},
    )
    assert result["d4_authorized"] is True
    assert result["learner_training_authorized"] is False


def test_guardrail_failure_blocks_retained_pass(monkeypatch):
    values = {
        (0.5, "retained_posterior"): _contrast(0.0),
        (0.75, "retained_posterior"): _contrast(0.012),
        (0.9, "retained_posterior"): _contrast(0.018, fill=-0.03),
        (0.9, "shuffled_posterior"): _contrast(0.0),
        (0.9, "wrong_posterior"): _contrast(-0.01),
    }
    monkeypatch.setattr(
        "scripts.run_q_r1_cold_start_replication.retained_contrast",
        lambda _rows, kappa, arm: values[(kappa, arm)],
    )
    result = adjudicate(
        {"rows": []},
        {"primary_persistent": {"history_clustered_lcb95": 0.015, "action_divergence": 0.2}},
    )
    assert result["retained_binary_context"]["pass"] is False
    assert result["d4_authorized"] is False
