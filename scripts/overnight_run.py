#!/usr/bin/env python3
"""Overnight autonomous Track-A LEARNING lane (Contract v2). Robust, gated, self-logging.

Division of labour (2026-06-26 night):
  * Codex owns the DOMINANCE / Excel-ReT lane (PPO/recurrent on the two frozen headroom profiles;
    Kaggle confirmatory `scresia-garrido-envb-confirmatory` already executed). DO NOT duplicate.
  * THIS run owns the additive LEARNING lane: DQN retained-vs-reset cold transfer (the memory /
    decision-frontier claim) + the Ed.2 operational-inertia moderator. This is what Codex has NOT
    done and what the reviews call the strongest paper claim.

Aligns to the ALREADY-FROZEN env profiles in `supply_chain/data/headroom_env_contract_v2_2026-06-26.json`
(does NOT re-freeze the env). Reward fixed to control_v1 (Contract v2 primary; ReT outcome-only) so
the two lanes are complementary, not redundant. Mask = direct_disruption_blind (regime hidden, so
memory has something to do).

Stages (each wrapped; failure logs + continues):
  F  retained-vs-reset on {faithful, cons, aggr} x {no-inertia, inertia} x tapes
  D  reward gate (control_v1 vs control_v2) on the static surface (Codex preflight requirement)
  H  morning doc integrating BOTH lanes (this retained-reset + Codex confirmatory_summary.csv)
"""

from __future__ import annotations

import csv
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv/bin/python")
STAMP = "2026-06-26"
BENCH = ROOT / "outputs/benchmarks/retention_transfer"
LOG = ROOT / f"outputs/experiments/overnight_learning_{STAMP}.log"

LEARNER_SEEDS = "11,12,13,14,15,16,17,18,19,20"   # 10 learner seeds -> seed-clustered CI per tape
TAPES = [909, 707, 505]                            # 3 exogenous regime tapes
N_BLOCKS = "12"
TRAIN_PER_BLOCK = "150"
PRETRAIN = "5000"
RHO = "0.85"
REWARD = "control_v1"
MASK = "direct_disruption_blind"
MAX_STEPS = "52"

# (label, phi, psi, inertia) -- phi/psi taken from the frozen headroom contract profiles.
ENVS = [
    ("faithful",       1.0, 1.0,  False),   # 1:1 fair fight
    ("cons",           1.0, 1.25, False),   # envb_cons profile (Partial Win B territory)
    ("aggr",           2.0, 1.5,  False),   # envb_aggr profile (Partial Win A territory)
    ("cons_inertia",   1.0, 1.25, True),    # Ed.2 moderator on cons
    ("aggr_inertia",   2.0, 1.5,  True),    # Ed.2 moderator on aggr
]


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as fh:
        fh.write(line + "\n")


def run(cmd: list[str], label: str, timeout: int) -> int:
    log(f"START {label}")
    t = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(ROOT))
        log(f"END   {label} rc={r.returncode} {time.time()-t:.0f}s\n{(r.stdout or '')[-900:]}")
        if r.returncode != 0:
            log(f"STDERR {label}:\n{(r.stderr or '')[-1200:]}")
        return r.returncode
    except subprocess.TimeoutExpired:
        log(f"TIMEOUT {label} after {timeout}s")
        return 124
    except Exception as exc:  # noqa: BLE001
        log(f"EXC {label}: {exc!r}")
        return 1


def confirmatory() -> list[dict]:
    rows = []
    for name, phi, psi, inertia in ENVS:
        for rs in TAPES:
            label = f"learn_{name}_rs{rs}"
            cmd = [PY, "scripts/retention_transfer.py", "--label", label,
                   "--reward-mode", REWARD, "--seeds", LEARNER_SEEDS, "--n-blocks", N_BLOCKS,
                   "--train-per-block", TRAIN_PER_BLOCK, "--pretrain-timesteps", PRETRAIN,
                   "--rho-disruption", RHO, "--regime-seed", str(rs), "--mask-preset", MASK,
                   "--max-steps", MAX_STEPS, "--outcome", "excel_ret",
                   "--risk-frequency-multiplier", str(phi), "--risk-impact-multiplier", str(psi)]
            if inertia:
                cmd += ["--surge-inertia"]
            run(cmd, f"F {label}", timeout=6000)
            j = BENCH / label / "transfer.json"
            if j.exists():
                d = json.loads(j.read_text())
                mem = d.get("memory_retained_minus_reset", {})
                tot = d.get("total_retained_minus_frozen", {})
                rows.append({
                    "env": name, "phi": phi, "psi": psi, "inertia": inertia, "tape": rs,
                    "mem_overall": (mem.get("overall") or {}).get("mean"),
                    "mem_ci_lo": (mem.get("overall") or {}).get("ci95_lo"),
                    "mem_ci_hi": (mem.get("overall") or {}).get("ci95_hi"),
                    "mem_late": (mem.get("late_half") or {}).get("mean"),
                    "mem_early": (mem.get("early_half") or {}).get("mean"),
                    "tot_overall": (tot.get("overall") or {}).get("mean"),
                })
    (ROOT / f"outputs/experiments/learning_lane_results_{STAMP}.json").write_text(json.dumps(rows, indent=2))
    return rows


def stage_D_reward_gate() -> None:
    run([PY, "scripts/audit_thesis_reward_surface.py", "--label", f"reward_gate_{STAMP}",
         "--reward-modes", "control_v1", "control_v2", "control_v1_pbrs", "ReT_garrido2024",
         "--risk-levels", "current", "increased", "severe",
         "--downstream-q-source", "figure_6_2", "--replications", "3", "--max-steps", "52"],
        "D reward gate", timeout=10800)


def read_codex_dominance() -> list[dict]:
    p = ROOT / "outputs/kaggle/garrido_envb_confirmatory/confirmatory_summary.csv"
    if not p.exists():
        return []
    with p.open() as fh:
        return list(csv.DictReader(fh))


def fnum(x):
    try:
        v = float(x)
        return None if v != v else round(v, 5)  # drop NaN
    except (TypeError, ValueError):
        return None


def stage_H_morning(rows: list[dict]) -> None:
    # aggregate my retained-reset by env (mean over tapes)
    by_env = {}
    for r in rows:
        by_env.setdefault(r["env"], []).append(r)
    def agg(name):
        rs = by_env.get(name, [])
        ms = [r["mem_overall"] for r in rs if isinstance(r["mem_overall"], (int, float))]
        ls = [r["mem_late"] for r in rs if isinstance(r["mem_late"], (int, float))]
        es = [r["mem_early"] for r in rs if isinstance(r["mem_early"], (int, float))]
        return (round(sum(ms)/len(ms), 6) if ms else None,
                round(sum(es)/len(es), 6) if es else None,
                round(sum(ls)/len(ls), 6) if ls else None, len(rs))
    lines = []
    for name, _phi, _psi, _i in ENVS:
        mean_delta, early_delta, late_delta, tape_count = agg(name)
        lines.append(
            f"| {name} | {mean_delta} | {early_delta} | {late_delta} | {tape_count} |"
        )
    table = "\n".join(lines)

    cod = read_codex_dominance()
    cod_lines = []
    for r in cod:
        cod_lines.append(
            f"| {r.get('label')} | {r.get('claim_path')} | {r.get('static_policy')} | "
            f"{r.get('strict_win')} | {r.get('pareto_win')} | {r.get('excel_noninferior')} | "
            f"{fnum(r.get('excel_delta_mean'))} | {fnum(r.get('resource_delta_mean'))} |")
    cod_table = "\n".join(cod_lines) if cod_lines else "| (confirmatory_summary.csv not found) |"

    doc = f"""# Track A Results — overnight ({STAMP})

Auto-generated by `scripts/overnight_run.py`. Two complementary lanes for the "best of all worlds".
Inspect raw `transfer.json` (this lane) and `outputs/kaggle/garrido_envb_confirmatory/` (Codex lane)
before trusting any single number. Pilot scale — CIs are per-tape seed-clustered, not the full 10x10.

## Lane 1 (this run) — DQN retained-vs-reset cold transfer (the LEARNING / memory claim)
Reward control_v1 (ReT outcome-only); mask `direct_disruption_blind`; {len(TAPES)} tapes x 10 learner
seeds; rho={RHO}; n_blocks={N_BLOCKS}. **memory Δ = retained − reset** (cross-block memory L_(k-1)).

| env | memory Δ (mean over tapes) | early-half | late-half | n tapes |
| --- | ---: | ---: | ---: | ---: |
{table}

Reads:
- memory Δ > 0 ⇒ accumulated cross-block learning gives a cold-start edge on new shocks.
- **Ed.2 moderator hypothesis:** memory Δ(*_inertia) > memory Δ(no-inertia), and headroom > faithful.
  If late-half > early-half, the head-start grows with exposure (learning curve).
- Per-tape CIs are in each `outputs/benchmarks/retention_transfer/learn_*/transfer.json`; the
  hierarchical (seed×tape) bootstrap is the remaining morning analysis step.

## Lane 2 (Codex) — dominance / Excel-ReT confirmatory (beat the statics)
From `outputs/kaggle/garrido_envb_confirmatory/confirmatory_summary.csv` (paired_n small → CIs nan):

| label | claim | vs static | strict | pareto | excel_noninf | excel Δ | resource Δ |
| --- | --- | --- | --- | --- | --- | ---: | ---: |
{cod_table}

## Honest synthesis (most-promising track)
- The Excel-ReT "beat Garrido" margins are tiny (~2e-4) and fragile at n=1 → **Partial Win A is weak**.
- The **conservative Pareto/resource win** (control_v2 vs the expensive S3_I1344: excel non-inferior,
  large resource saving) is the **robust dominance result** → lead with **Partial Win B**.
- The **strongest, most novel** claim is Lane 1's frontier+inertia story IF memory Δ is positive and
  grows with inertia. That is the paper's spine; dominance (Lane 2 Partial Win B) is the supporting
  efficiency result. If memory Δ ≈ 0 everywhere, report the honest null per Contract v2 §10 and lead
  with Partial Win B.

## Caveats
- Pilot scale; not the full 10×10 held-out design. Endogenous-R2 tail remains an open fidelity
  limitation (lane stays on R1-dominant regimes). Reward gate output under
  `outputs/.../reward_gate_{STAMP}` informs whether control_v2 should replace control_v1.
"""
    (ROOT / f"docs/TRACK_A_RESULTS_overnight_{STAMP}.md").write_text(doc)
    log("WROTE docs/TRACK_A_RESULTS_overnight_" + STAMP + ".md")


def main() -> int:
    log("=== OVERNIGHT LEARNING RUN START ===")
    rows = []
    try:
        rows = confirmatory()
    except Exception as exc:  # noqa: BLE001
        log(f"F failed: {exc!r}")
    try:
        stage_D_reward_gate()
    except Exception as exc:  # noqa: BLE001
        log(f"D failed: {exc!r}")
    try:
        stage_H_morning(rows)
    except Exception as exc:  # noqa: BLE001
        log(f"H failed: {exc!r}")
    log("=== OVERNIGHT LEARNING RUN DONE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
