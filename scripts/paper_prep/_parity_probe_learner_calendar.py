"""Parity probe 2: LearnerPolicy.calendar vs the Q evaluator's loop, verbatim.

Re-implements evaluate_program_q_replication.model_calendar line-by-line
(ProgramORetOnlyEnv + model.predict) and compares calendars against this
arm's LearnerPolicy on two smoke tapes per cell.
"""
import importlib.util
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from sb3_contrib import RecurrentPPO  # noqa: E402

from supply_chain.program_o_full_des_transducer import extract_full_des_skeleton  # noqa: E402
from supply_chain.program_o_ret_env import CONFIRMED_RET_CELLS, ProgramORetOnlyEnv  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "dc", "scripts/paper_prep/deployable_comparator_v1.py"
)
dc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dc)
SCHED = dc.canonical_scheduler()
dc.SCHED = SCHED


def q_evaluator_calendar(model, skeleton, cell_index):
    """Verbatim body of evaluate_program_q_replication.model_calendar."""
    env = ProgramORetOnlyEnv(
        scheduler=SCHED,
        tape_seed_start=int(skeleton.seed),
        tape_seed_end=int(skeleton.seed),
    )
    observation, _ = env.reset(
        options={
            "skeleton": skeleton,
            "tape_seed": int(skeleton.seed),
            "cell_index": cell_index,
        }
    )
    state = None
    episode_start = np.ones((1,), dtype=bool)
    actions = []
    terminated = False
    while not terminated:
        action, state = model.predict(
            observation, state=state, episode_start=episode_start, deterministic=True
        )
        actions.append(int(np.asarray(action).item()))
        observation, _, terminated, _, _ = env.step(actions[-1])
        episode_start[:] = terminated
    return tuple(actions)


freeze = json.loads(dc.FREEZE_JSON.read_text())
seed_for_seed = sorted(freeze["checkpoints_sha256"], key=int)
model_path = dc.MODELS_DIR / f"recurrent_ppo_seed_{seed_for_seed[0]}.zip"
model = RecurrentPPO.load(model_path, device="cpu")
mine = dc.LearnerPolicy(model_path)

probes = {7550385: 0, 7550449: 1, 7550513 - 64 + 62: 2}  # one fallback tape per cell
for seed, cell_index in probes.items():
    cell = CONFIRMED_RET_CELLS[cell_index]
    sk, _ = extract_full_des_skeleton(
        seed=seed,
        scheduler=SCHED,
        regime_persistence=cell.regime_persistence,
        dominant_share=cell.dominant_share,
        downstream_freight_physics_mode="fixed_clock_physical_v1",
    )
    a = q_evaluator_calendar(model, sk, cell_index)
    b = mine.calendar(sk)
    status = "MATCH" if a == b else f"MISMATCH {a} vs {b}"
    print(f"seed {seed} cell {cell.cell_id}: learner calendar {status}")
    assert a == b
print("LEARNER_CALENDAR_PARITY_OK")
