"""Parity probe: deployable_comparator_v1 replay vs ProgramORetOnlyEnv.

Throws if the runner's observation replay is not byte-identical to the frozen
learner environment used by evaluate_program_q_replication.model_calendar.
"""
import importlib.util
import sys

import numpy as np

sys.path.insert(0, ".")
from supply_chain.program_o_full_des_transducer import extract_full_des_skeleton  # noqa: E402
from supply_chain.program_o_ret_env import CONFIRMED_RET_CELLS, ProgramORetOnlyEnv  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "dc", "scripts/paper_prep/deployable_comparator_v1.py"
)
dc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dc)
SCHED = dc.canonical_scheduler()
dc.SCHED = SCHED

seed = 7550385  # fallback-block tape; parity only, no science read
cell = CONFIRMED_RET_CELLS[0]
sk, _ = extract_full_des_skeleton(
    seed=seed,
    scheduler=SCHED,
    regime_persistence=cell.regime_persistence,
    dominant_share=cell.dominant_share,
    downstream_freight_physics_mode="fixed_clock_physical_v1",
)

actions = [1, 3, 0, 2, 1, 1, 3]  # arbitrary probe calendar (7 steps)

env = ProgramORetOnlyEnv(scheduler=SCHED, tape_seed_start=seed, tape_seed_end=seed)
obs_env, info = env.reset(options={"skeleton": sk, "tape_seed": seed, "cell_index": 0})
digests_env = [info["observation_sha256"]]
vecs_env = [np.asarray(obs_env, dtype=np.float32)]
for a in actions:
    obs_env, _reward, term, _trunc, info = env.step(a)
    digests_env.append(info["observation_sha256"])
    vecs_env.append(np.asarray(obs_env, dtype=np.float32))
    assert not term

digests_mine: list[str] = []
vecs_mine: list[np.ndarray] = []
for step in range(len(actions) + 1):
    decisions = dc.replay_decisions(sk, actions[:step])
    decision = decisions[step]
    digests_mine.append(decision.observation.observation_sha256)
    vecs_mine.append(dc.learner_observation_vector(decision))

assert digests_env == digests_mine, f"MISMATCH\n{digests_env}\n{digests_mine}"
max_diff = max(
    float(np.max(np.abs(v - w))) for v, w in zip(vecs_env, vecs_mine)
)
assert max_diff == 0.0, f"vector drift {max_diff}"
print(f"PARITY_OK over {len(digests_env)} decision points; max vector diff {max_diff}")
