from pathlib import Path

from scripts.run_g3_obs_conversion import F2_ORDER, f2_order_passes


RUNNER = Path(__file__).resolve().parents[1] / "scripts/run_g3_obs_conversion.py"


def test_f2_requires_complete_order_including_delayed_arm():
    good = {name: {"mean": value} for name, value in zip(F2_ORDER, (0.20, 0.10, 0.0, -0.2))}
    bad = {name: {"mean": value} for name, value in zip(F2_ORDER, (0.20, 0.0, 0.10, -0.2))}
    assert f2_order_passes(good)
    assert not f2_order_passes(bad)


def test_f2_order_names_are_contract_order():
    assert F2_ORDER == (
        "threshold_windowed",
        "threshold_delayed",
        "uninformed_placebo",
        "wrong_claimant",
    )


def test_runner_requires_explicit_contract_and_labels_replays():
    source = RUNNER.read_text()
    assert 'add_argument("--contract", type=Path, required=True' in source
    assert 'choices=("DEVELOPMENT", "REPLAY", "CONFIRMATORY")' in source
    assert 'G3_OBS_CONTRACT_REPLAY_NO_NEW_CONFIRMATION' in source
