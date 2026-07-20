from scripts.run_program_u1_native_risk_stage_a import ALLOWED_GROUPS, FRESH_SEEDS, tasks


def test_u1_stage_a_uses_fresh_three_tape_block_and_certified_masks_only() -> None:
    rows = tasks()
    assert ALLOWED_GROUPS == (1, 2)
    assert FRESH_SEEDS == (7_540_001, 7_540_002, 7_540_003)
    assert len(rows) == len(set(rows)) == 900
    assert {row[0] for row in rows} == {1, 2}
    assert {row[-1] for row in rows} == set(FRESH_SEEDS)


def test_u1_does_not_use_the_mathematically_degenerate_one_tape_gate() -> None:
    assert len(FRESH_SEEDS) == 3
