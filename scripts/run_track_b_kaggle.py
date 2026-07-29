#!/usr/bin/env python3
"""Track B campaign runner — Kaggle-optimized with 9 improvements.

   1. Reward: ReT_excel_plus_cvar (winning +0.023267, full verdicts)
   2. Timesteps: 60k (retrain 100k improved gap +16%)
   3. Max steps: 104 (h104 = 4 months, confirmed win horizon)
   4. LR: cosine decay (Track A lesson)
   5. Norm reward: True (SB3 recommended for delta rewards)
   6. BC warm-start: 200 epochs, best_static teacher per regime
   7. Checkpoint: held-out eval every 5000 steps, keep best
   8. Seeds: 5 (bootstrap CI + Cohen's d)
   9. H2/H3 panel: learning curve CSV + per-cell CV

   Output:
     - episode_metrics.csv     every eval (static + PPO)
     - policy_summary.csv      aggregated per policy
     - learning_curve.csv      ret_excel vs training step
     - summary.json            full verdict incl. Cohen's d, CV
"""
from __future__ import annotations

import argparse, csv, json, statistics, sys, time, traceback
from pathlib import Path
from typing import Any

import numpy as np, torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_track_b_campaign import (
    CampaignCell, RISK_FAMILIES, TrackBCampaignEnv,
    finalize_row, aggregate, env_kwargs_for_cell,
    extract_downstream_multipliers,
)
from scripts.run_track_b_independent_doe import (
    Policy, action_for, parse_float_list, parse_int_list,
)
from supply_chain.external_env_interface import make_track_b_env
from supply_chain.episode_metrics import compute_episode_metrics


# ═══════════════════════════════════════════════════════════════════════
#  helpers
# ═══════════════════════════════════════════════════════════════════════
def mean(vs): return float(statistics.fmean(vs)) if vs else 0.0

def cohens_d(a, b):
    a, b = np.atleast_1d(a).astype(np.float64), np.atleast_1d(b).astype(np.float64)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2: return 0.0
    s = np.sqrt(((na-1)*np.var(a, ddof=1) + (nb-1)*np.var(b, ddof=1)) / (na+nb-2))
    return float((np.mean(a) - np.mean(b)) / s) if s > 1e-12 else 0.0


def paired_deltas(
    rows: list[dict],
    *,
    learned_policy: str,
    baseline_policy: str,
    metric: str,
) -> list[float]:
    """Return learned-minus-baseline deltas paired by eval seed and cell.

    The campaign runner evaluates static and PPO policies under matching CRN
    eval seeds.  CIs for a policy difference must be computed on those paired
    deltas, not by subtracting marginal policy confidence intervals.
    """
    baseline: dict[tuple[str, str], float] = {}
    for row in rows:
        if row.get("policy") == baseline_policy:
            baseline[(str(row.get("eval_seed")), str(row.get("cell")))] = float(row[metric])

    deltas: list[float] = []
    for row in rows:
        if row.get("policy") != learned_policy:
            continue
        key = (str(row.get("eval_seed")), str(row.get("cell")))
        if key in baseline:
            deltas.append(float(row[metric]) - baseline[key])
    return deltas


def bootstrap_ci95(values: list[float], *, n: int = 10_000, seed: int = 12345) -> tuple[float, float]:
    """Bootstrap CI for the mean of paired deltas."""
    if len(values) < 2:
        value = float(values[0]) if values else float("nan")
        return value, value
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sample_means = rng.choice(arr, size=(n, len(arr)), replace=True).mean(axis=1)
    lo, hi = np.quantile(sample_means, [0.025, 0.975])
    return float(lo), float(hi)


def write_status(output_dir: Path, phase: str, **extra: Any) -> None:
    """Write a small heartbeat/status file for long local/Kaggle jobs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": phase,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        **extra,
    }
    (output_dir / "status.json").write_text(json.dumps(payload, indent=2))


# ═══════════════════════════════════════════════════════════════════════
#  BC warm-start — thesis scale to [-1, 1] action-space conversion
# ═══════════════════════════════════════════════════════════════════════
def thesis_to_normalized(shift: int, op9_mult: float,
                         op10_mult: float, op12_mult: float) -> np.ndarray:
    """Convert thesis-scale static policy to [-1,1] 8D action array.

    The env maps [-1,1] → thesis multipliers:
      Dims 0-3, 6-7:  multiplier = 1.25 + 0.75 × x  →  x = (m - 1.25)/0.75
      Dim 4 (Op5):     multiplier = 1.00 + 0.50 × x  →  x = (m - 1.00)/0.50
      Dim 5 (shift):   -1.0→1, 0.0→2, 1.0→3
    """
    action = np.zeros(8, dtype=np.float32)
    action[0] = (1.0       - 1.25) / 0.75   # Op3 Q = base×1.0
    action[1] = (op9_mult  - 1.25) / 0.75   # Op9 Q = base×op9_mult
    action[2] = (1.0       - 1.25) / 0.75   # Op3 ROP = base×1.0
    action[3] = (1.0       - 1.25) / 0.75   # Op9 ROP = base×1.0
    action[4] = 0.0                          # Op5 = 1.0 (neutral)
    action[5] = {1: -1.0, 2: 0.0, 3: 1.0}[shift]
    action[6] = (op10_mult - 1.25) / 0.75   # Op10 dispatch
    action[7] = (op12_mult - 1.25) / 0.75   # Op12 dispatch
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def load_best_actions(gate_dir: Path,
                      cells: list[CampaignCell],
                      *,
                      allow_fallback: bool = False) -> dict[str, np.ndarray]:
    """Return best static action per regime label, in [-1,1] space."""
    rows = list(csv.DictReader((gate_dir / "seed_metrics.csv").open()))
    best: dict[str, np.ndarray] = {}
    for cell in cells:
        candidates = [r for r in rows
                      if r.get("risk_level") == cell.risk_level
                      and r.get("family") == cell.family
                      and r.get("phi", "1.0") == str(cell.phi)
                      and r.get("psi", "1.0") == str(cell.psi)
                      and r.get("demand_mult", "1.0") == str(cell.demand_mult)]
        if not candidates and allow_fallback:
            candidates = [r for r in rows
                          if r.get("risk_level") == cell.risk_level
                          and r.get("family") == cell.family]
        if not candidates:
            raise ValueError(
                "No exact BC teacher action for "
                f"{cell.label} ({cell.risk_level}/{cell.family}, "
                f"phi={cell.phi}, psi={cell.psi}, demand={cell.demand_mult}). "
                "This usually means the campaign cells do not match the "
                "headroom gate. Re-run with matching filters or pass "
                "--allow-bc-fallback intentionally."
            )
        best_row = max(candidates, key=lambda r: float(r.get("ret_excel", 0)))
        act = thesis_to_normalized(
            shift    = int(best_row.get("shift", 2)),
            op9_mult = float(best_row.get("op9_mult", 1.0)),
            op10_mult= float(best_row.get("op10_mult", 1.0)),
            op12_mult= float(best_row.get("op12_mult", 1.0)),
        )
        best[cell.label] = act
        print(f"  BC teacher {cell.label}: S{best_row.get('shift')} "
              f"op10={best_row.get('op10_mult')} op12={best_row.get('op12_mult')} "
              f"ret_excel={float(best_row['ret_excel']):.4f}", flush=True)
    return best


def collect_bc_trajectories(args, cells, best_actions,
                            collect_seed=7000):
    """Run each regime with teacher action (array step), collect (obs,act)."""
    obs_list, act_list = [], []
    for i, cell in enumerate(cells):
        action_arr = best_actions[cell.label]   # 8D [-1,1]
        env = make_track_b_env(**env_kwargs_for_cell(args, cell))
        obs, _ = env.reset(seed=collect_seed + i)
        done = False
        while not done:
            obs_list.append(np.asarray(obs, dtype=np.float32).copy())
            act_list.append(action_arr.copy())
            obs, _, terminated, truncated, _ = env.step(action_arr)
            done = bool(terminated or truncated)
        env.close()
    return (np.vstack(obs_list).astype(np.float32),
            np.vstack(act_list).astype(np.float32))


def bc_pretrain(model, obs_np, act_np, epochs=200, batch_size=128,
                seed=42):
    """MSE supervised learning on collected teacher trajectories."""
    device = model.policy.device
    obs_t = torch.as_tensor(obs_np, device=device)
    act_t = torch.as_tensor(act_np, device=device)
    rng = np.random.default_rng(seed)
    losses = []
    for epoch in range(epochs):
        order = rng.permutation(len(obs_np))
        epoch_loss = 0.0
        for start in range(0, len(obs_np), batch_size):
            idx = torch.as_tensor(order[start:start+batch_size],
                                  dtype=torch.long, device=device)
            pred = model.policy.get_distribution(obs_t[idx]).mode()
            loss = torch.nn.functional.mse_loss(pred, act_t[idx])
            model.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
            model.policy.optimizer.step()
            epoch_loss += float(loss.detach())
        losses.append(epoch_loss)
        if (epoch + 1) % 50 == 0:
            print(f"  BC epoch {epoch+1}/{epochs} loss={epoch_loss:.4f}", flush=True)
    print(f"  BC final loss={losses[-1]:.4f}", flush=True)
    return losses


# ═══════════════════════════════════════════════════════════════════════
#  evaluation
# ═══════════════════════════════════════════════════════════════════════
def evaluate_model(model, vec_norm, args, cells, eval_seed) -> list:
    """Evaluate PPO on all cells with CRN eval seed (no training rng)."""
    vec_norm.training = False
    rows = []
    for cell_idx, cell in enumerate(cells):
        eval_seed_cell = eval_seed + cell_idx * 1000
        env = make_track_b_env(**env_kwargs_for_cell(args, cell))
        obs, _ = env.reset(seed=eval_seed_cell)
        done = False
        reward_total, steps = 0.0, 0
        shift_counts = {1: 0, 2: 0, 3: 0}
        op10_vals: list[float] = []
        op12_vals: list[float] = []
        while not done:
            obs_n = vec_norm.normalize_obs(
                np.asarray(obs, dtype=np.float32)[None, :])
            a, _ = model.predict(obs_n, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(
                np.asarray(a[0], dtype=np.float32))
            reward_total += float(reward)
            shift = int(info.get("shifts_active", 1))
            shift_counts[shift] = shift_counts.get(shift, 0) + 1
            o10, o12 = extract_downstream_multipliers(info)
            op10_vals.append(o10); op12_vals.append(o12)
            steps += 1
            done = bool(terminated or truncated)
        panel = compute_episode_metrics(env.unwrapped.sim)
        rows.append(finalize_row(
            policy="ppo", seed=0, episode=1, eval_seed=eval_seed_cell,
            cell=cell, reward_total=reward_total, panel=panel, steps=steps,
            shift_counts=shift_counts, op10_values=op10_vals,
            op12_values=op12_vals, args=args))
        env.close()
    return rows


def evaluate_static_policies(args, policies, cells, seeds) -> list:
    """Evaluate static policies on all cells with CRN seed."""
    rows = []
    total = len(policies) * len(seeds) * len(cells)
    done_count = 0
    for policy_idx, policy in enumerate(policies, start=1):
        action = action_for(policy)          # dict, env accepts both
        for eval_seed_s in seeds:
            for cell_idx, cell in enumerate(cells):
                eval_seed = int(args.eval_seed0 +
                                eval_seed_s * 100_000 +
                                cell_idx * 1000)
                env = make_track_b_env(**env_kwargs_for_cell(args, cell))
                env.reset(seed=eval_seed)
                done = False
                reward_total, steps = 0.0, 0
                while not done:
                    _, r, terminated, truncated, _ = env.step(action)
                    reward_total += float(r)
                    done = bool(terminated or truncated)
                    steps += 1
                panel = compute_episode_metrics(env.unwrapped.sim)
                rows.append(finalize_row(
                    policy=policy.label, seed=eval_seed_s, episode=1,
                    eval_seed=eval_seed, cell=cell,
                    reward_total=reward_total, panel=panel, steps=steps,
                    shift_counts={policy.shift: steps},
                    op10_values=[policy.op10_mult] * steps,
                    op12_values=[policy.op12_mult] * steps, args=args))
                env.close()
                done_count += 1
        if policy_idx == 1 or policy_idx % 8 == 0 or policy_idx == len(policies):
            write_status(args.output_dir, "static_frontier",
                         n_evals=total, n_done=done_count,
                         current_policy=policy.label)
            print(f"  static {policy_idx}/{len(policies)} "
                  f"({done_count}/{total} evals)", flush=True)
    return rows


# ═══════════════════════════════════════════════════════════════════════
#  training
# ═══════════════════════════════════════════════════════════════════════
def train_with_bc_and_checkpoints(args, cells, best_actions, seed,
                                  run_dir):
    print(f"\n--- Seed {seed} ---", flush=True)

    def _init():
        return Monitor(TrackBCampaignEnv(args=args, cells=cells,
                                          seed=seed))
    vec = DummyVecEnv([_init for _ in range(max(1, args.n_envs))])
    vec_norm = VecNormalize(vec, norm_obs=True,
                            norm_reward=args.norm_reward,
                            clip_obs=10.0, clip_reward=10.0)

    # cosine LR: progress_remaining p ∈ [1.0 → 0.0]
    def cosine_lr(p):
        return float(args.learning_rate) * (
            args.lr_end_ratio + (1.0 - args.lr_end_ratio) * p)

    model = PPO("MlpPolicy", vec_norm, seed=seed, verbose=0,
                learning_rate=cosine_lr,
                n_steps=args.n_steps, batch_size=args.batch_size,
                n_epochs=args.n_epochs, gamma=args.gamma,
                gae_lambda=args.gae_lambda, ent_coef=args.ent_coef)

    # ── BC warm-start ──
    if args.bc_epochs > 0:
        print(f"  BC warm-start: {args.bc_epochs} epochs", flush=True)
        t_bc = time.time()
        bc_obs, bc_acts = collect_bc_trajectories(
            args, cells, best_actions, collect_seed=7000 + seed)
        print(f"  BC data: {len(bc_obs)} samples "
              f"({time.time()-t_bc:.0f}s)", flush=True)
        bc_pretrain(model, bc_obs, bc_acts,
                    epochs=args.bc_epochs, batch_size=128,
                    seed=seed + 1000)

    # ── PPO with checkpoint selection ──
    model_dir = run_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    best_ret = -1.0
    best_step = 0
    learning_curve: list[dict] = []
    steps_done = 0
    eval_seed_base = args.eval_seed0 + seed * 100_000

    while steps_done < args.timesteps:
        chunk = min(args.checkpoint_every,
                    args.timesteps - steps_done)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False,
                    progress_bar=False)
        steps_done += chunk

        ckpt_rows = evaluate_model(model, vec_norm, args, cells,
                                   eval_seed_base + steps_done)
        ckpt_sum = aggregate(ckpt_rows)
        ppo_r = next(r for r in ckpt_sum if r["policy"] == "ppo")
        ret = float(ppo_r["ret_excel_mean"])
        learning_curve.append({"step": steps_done,
                               "ret_excel_mean": ret})

        if ret > best_ret:
            best_ret = ret
            best_step = steps_done
            model.save(model_dir / f"ppo_seed{seed}_best.zip")
            vec_norm.save(model_dir / f"vecnorm_seed{seed}_best.pkl")

        print(f"  [36mstep={steps_done}/{args.timesteps} "
              f"ret={ret:.4f} best={best_ret:.4f} @{best_step}[0m",
              flush=True)

    # Load best checkpoint
    best_path = model_dir / f"ppo_seed{seed}_best.zip"
    if best_path.exists():
        model = PPO.load(best_path, vec_norm)
        print(f"  loaded best: step={best_step} ret={best_ret:.4f}",
              flush=True)

    return model, vec_norm, learning_curve


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gate-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--families", default="R2,R24")
    ap.add_argument("--risk-levels", default="current,increased,severe")
    ap.add_argument("--cells-per-group", type=int, default=1)
    ap.add_argument("--max-cells", type=int, default=8)
    ap.add_argument(
        "--target-psi",
        type=float,
        default=1.0,
        help=(
            "Psi filter for campaign-cell selection. Default 1.0 keeps the "
            "campaign matched to the conservative headroom gate."
        ),
    )
    ap.add_argument(
        "--no-psi-filter",
        action="store_true",
        help="Disable the psi filter intentionally; useful only for exploratory mixed-psi runs.",
    )

    # ── optimised defaults ──
    ap.add_argument("--reward-mode", default="ReT_excel_plus_cvar")
    ap.add_argument("--ret-excel-cvar-alpha", type=float, default=0.1)
    ap.add_argument("--ret-excel-cvar-tail-level", type=float, default=0.95)
    ap.add_argument("--ret-excel-cvar-window", type=int, default=8)
    ap.add_argument("--observation-version", default="v7")
    ap.add_argument("--timesteps", type=int, default=60000)
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--max-steps", type=int, default=104)
    ap.add_argument("--step-size-hours", type=float, default=168.0)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--lr-end-ratio", type=float, default=0.1)
    ap.add_argument("--n-steps", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-epochs", type=int, default=10)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--norm-reward", action="store_true", default=True)
    ap.add_argument("--no-norm-reward", action="store_false",
                    dest="norm_reward")

    ap.add_argument("--shifts", default="1,2,3")
    ap.add_argument("--op9-mults", default="1.0")
    ap.add_argument("--op10-mults", default="0.5,1.0,1.5,2.0")
    ap.add_argument("--op12-mults", default="0.5,1.0,1.5,2.0")

    # ── BC + checkpoint ──
    ap.add_argument("--bc-epochs", type=int, default=200)
    ap.add_argument(
        "--allow-bc-fallback",
        action="store_true",
        help="Allow family-level BC teacher fallback when no exact gate cell exists.",
    )
    ap.add_argument("--checkpoint-every", type=int, default=5000)
    ap.add_argument("--eval-seed0", type=int, default=9000)
    return ap


# ═══════════════════════════════════════════════════════════════════════
#  campaign cell loading (same logic as base runner)
# ═══════════════════════════════════════════════════════════════════════
def load_campaign_cells(args) -> list[CampaignCell]:
    rows = list(csv.DictReader(
        (args.gate_dir / "cell_policy_summary.csv").open()))
    families = {f.strip() for f in args.families.split(",") if f.strip()}
    risk_levels = {r.strip()
                   for r in args.risk_levels.split(",") if r.strip()}

    target_psi = None if getattr(args, "no_psi_filter", False) else args.target_psi

    candidates: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["family"] not in families: continue
        if row["risk_level"] not in risk_levels: continue
        if target_psi is not None and abs(float(row.get("psi", 1.0)) - target_psi) > 0.01:
            continue
        cell_key = row["cell"]
        if (cell_key not in candidates
                or float(row["ret_excel_mean"])
                > float(candidates[cell_key]["ret_excel_mean"])):
            candidates[cell_key] = row

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in candidates.values():
        grouped.setdefault(
            (row["risk_level"], row["family"]), []).append(row)

    selected = []
    for key in sorted(grouped):
        ranked = sorted(grouped[key],
                        key=lambda r: float(r["ret_excel_mean"]),
                        reverse=True)
        for row in ranked[:args.cells_per_group]:
            selected.append(CampaignCell(
                label=str(row["cell"]),
                risk_level=str(row["risk_level"]),
                family=str(row["family"]),
                phi=float(row["phi"]),
                psi=float(row["psi"]),
                demand_mult=float(row["demand_mult"])))
    if not selected:
        raise ValueError(
            f"No campaign cells selected from {args.gate_dir} "
            f"(psi filter={target_psi}). Check --target-psi or gate CSV.")
    # Fail early if selected cells do not have exact BC teacher rows.  A silent
    # family-level fallback previously let campaigns train on mismatched regimes.
    seed_rows = list(csv.DictReader((args.gate_dir / "seed_metrics.csv").open()))
    missing = []
    for cell in selected:
        found = any(
            r.get("risk_level") == cell.risk_level
            and r.get("family") == cell.family
            and r.get("phi", "1.0") == str(cell.phi)
            and r.get("psi", "1.0") == str(cell.psi)
            and r.get("demand_mult", "1.0") == str(cell.demand_mult)
            for r in seed_rows
        )
        if not found:
            missing.append(cell.label)
    if missing and not getattr(args, "allow_bc_fallback", False):
        raise ValueError(
            "Selected cells lack exact BC teacher rows: "
            + ", ".join(missing)
            + ". Refusing to run without --allow-bc-fallback."
        )
    return selected[:args.max_cells]


# ═══════════════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════════════
def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_status(args.output_dir, "starting", config={
        k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
    })

    cells = load_campaign_cells(args)
    seeds = list(parse_int_list(args.seeds))
    policies = [
        Policy(shift=s, op9_mult=o9, op10_mult=o10, op12_mult=o12)
        for s in parse_int_list(args.shifts)
        for o9 in parse_float_list(args.op9_mults)
        for o10 in parse_float_list(args.op10_mults)
        for o12 in parse_float_list(args.op12_mults)
    ]
    write_status(args.output_dir, "selected_cells",
                 n_cells=len(cells), n_static_policies=len(policies),
                 cells=[c.label for c in cells])

    print(f"Track B Kaggle: {len(cells)} cells × {len(policies)} statics"
          f" × {len(seeds)} seeds", flush=True)
    print(f"  reward={args.reward_mode}  max_steps={args.max_steps}"
          f"  timesteps={args.timesteps}  bc={args.bc_epochs}ep"
          f"  n_envs={args.n_envs}  cvar_α={args.ret_excel_cvar_alpha}",
          flush=True)
    for c in cells:
        print(f"  {c.label:25s}  {c.risk_level}/{c.family}  "
              f"φ={c.phi:g} ψ={c.psi:g} d={c.demand_mult:g}",
              flush=True)

    # ── 1. Static frontier (ALL statics, all regimes, all seeds) ──
    print("\n" + "─" * 60 + "\n Static frontier\n" + "─" * 60, flush=True)
    write_status(args.output_dir, "static_frontier",
                 n_evals=len(policies) * len(cells) * len(seeds))
    t0 = time.time()
    static_rows = evaluate_static_policies(args, policies, cells, seeds)
    print(f"  {len(policies)} policies × {len(cells)} cells "
          f"× {len(seeds)} seeds = {len(static_rows)} evals "
          f"({time.time()-t0:.0f}s)", flush=True)

    # ── 2. BC teacher actions ──
    print("\n" + "─" * 60 + "\n BC teachers\n" + "─" * 60, flush=True)
    write_status(args.output_dir, "bc_teachers")
    best_actions = load_best_actions(
        args.gate_dir, cells, allow_fallback=args.allow_bc_fallback)

    # ── 3. Train PPO ──
    print("\n" + "─" * 60 + "\n PPO training\n" + "─" * 60, flush=True)
    write_status(args.output_dir, "ppo_training", n_seeds=len(seeds))
    all_ppo_rows: list[dict] = []
    all_curves: list[list[dict]] = []
    t0 = time.time()
    for seed in seeds:
        write_status(args.output_dir, "ppo_training",
                     current_seed=seed, n_seeds=len(seeds))
        model, vec_norm, curve = train_with_bc_and_checkpoints(
            args, cells, best_actions, seed, args.output_dir)
        all_curves.append(curve)
        ppo_rows = evaluate_model(model, vec_norm, args, cells,
                                  args.eval_seed0 + seed * 100_000)
        all_ppo_rows.extend(ppo_rows)
        vec_norm.close()
    print(f"\n  Training done ({time.time()-t0:.0f}s)", flush=True)
    write_status(args.output_dir, "summarising")

    # ── 4. Summarise ──
    all_rows = static_rows + all_ppo_rows
    write_csv(args.output_dir / "episode_metrics.csv", all_rows)
    policy_summary = aggregate(all_rows)

    best_static = max(
        [r for r in policy_summary if r["policy"] != "ppo"],
        key=lambda r: float(r["ret_excel_mean"]))
    best_reward_static = max(
        [r for r in policy_summary if r["policy"] != "ppo"],
        key=lambda r: float(r["reward_total_mean"]))
    ppo = next(r for r in policy_summary if r["policy"] == "ppo")

    best_name = best_static["policy"]

    # ── H3: per-cell CV (best_static vs PPO, not all statics) ──
    static_cell_means: list[float] = []
    ppo_cell_means: list[float] = []
    for cell in cells:
        sv = [float(r["ret_excel"]) for r in all_rows
              if r["cell"] == cell.label and r["policy"] == best_name]
        pv = [float(r["ret_excel"]) for r in all_rows
              if r["cell"] == cell.label and r["policy"] == "ppo"]
        if sv: static_cell_means.append(mean(sv))
        if pv: ppo_cell_means.append(mean(pv))

    h3_static_cv = (float(np.std(static_cell_means) /
                          max(1e-9, np.mean(static_cell_means)))
                    if len(static_cell_means) >= 2 else float("nan"))
    h3_ppo_cv = (float(np.std(ppo_cell_means) /
                       max(1e-9, np.mean(ppo_cell_means)))
                 if len(ppo_cell_means) >= 2 else float("nan"))

    # ── Cohen's d ──
    ppo_vals = [float(r["ret_excel"]) for r in all_rows
                if r["policy"] == "ppo"]
    best_vals = [float(r["ret_excel"]) for r in all_rows
                 if r["policy"] == best_name]
    d = cohens_d(ppo_vals, best_vals)

    # ── Pareto domination ──
    dominated = [
        r["policy"] for r in policy_summary
        if r["policy"] != "ppo"
        and float(r["ret_excel_mean"]) >= float(ppo["ret_excel_mean"])
        and float(r["assembly_cost_index_mean"])
        <= float(ppo["assembly_cost_index_mean"])
    ]

    best_name = best_static["policy"]
    ret_delta_pairs = paired_deltas(
        all_rows, learned_policy="ppo", baseline_policy=best_name, metric="ret_excel"
    )
    ret_delta = (
        mean(ret_delta_pairs)
        if ret_delta_pairs
        else float(ppo["ret_excel_mean"]) - float(best_static["ret_excel_mean"])
    )
    delta_lo, delta_hi = (
        bootstrap_ci95(ret_delta_pairs)
        if ret_delta_pairs
        else (ret_delta, ret_delta)
    )

    # Per-cell paired breakdown against the selected best static.
    cell_paired_means: dict[str, float] = {}
    cell_delta_means: list[float] = []
    cell_ci: dict[str, tuple[float, float]] = {}
    cell_unpaired: dict[str, tuple[float, float]] = {}  # (ppo_mean, static_mean)
    for cell in cells:
        cell_pairs = paired_deltas(
            [r for r in all_rows if r["cell"] == cell.label],
            learned_policy="ppo",
            baseline_policy=best_name,
            metric="ret_excel",
        )
        sv = [float(r["ret_excel"]) for r in all_rows
              if r["cell"] == cell.label and r["policy"] == best_name]
        pv = [float(r["ret_excel"]) for r in all_rows
              if r["cell"] == cell.label and r["policy"] == "ppo"]
        cell_unpaired[cell.label] = (mean(pv) if pv else 0.0, mean(sv) if sv else 0.0)

        if not cell_pairs:
            continue
        dm = mean(cell_pairs)
        cell_delta_means.append(dm)
        cell_paired_means[cell.label] = dm
        cell_ci[cell.label] = bootstrap_ci95(cell_pairs)

    n_cells = len(cell_delta_means)
    n_cells_won = sum(1 for d in cell_delta_means if d > 0)

    # Per-cell breakdown for summary (use paired delta mean when available)
    cell_breakdown: dict[str, dict[str, float]] = {}
    for cell in cells:
        pm, sm = cell_unpaired.get(cell.label, (0.0, 0.0))
        if cell.label in cell_paired_means:
            dm = cell_paired_means[cell.label]
        else:
            dm = pm - sm
        clo, chi = cell_ci.get(cell.label, (dm, dm))
        cell_breakdown[cell.label] = {
            "delta_mean": dm, "delta_ci95_low": clo,
            "delta_ci95_high": chi, "static_mean": sm, "ppo_mean": pm}

    verdict = {
        "raw_ret_win": ret_delta > 0,
        "raw_ret_ci_win": ret_delta > 0 and delta_lo > 0,
        "ret_delta": ret_delta,
        "ret_delta_paired_ci95_low": delta_lo,
        "ret_delta_paired_ci95_high": delta_hi,
        "n_cells_won": n_cells_won,
        "n_cells_total": n_cells,
        "same_reward_win": (
            float(ppo["reward_total_mean"])
            > float(best_reward_static["reward_total_mean"])),
        "pareto_ret_cost": len(dominated) == 0,
        "resource_efficient_win": (
            float(ppo["ret_excel_mean"])
            >= float(best_static["ret_excel_mean"])
            and float(ppo["assembly_cost_index_mean"])
            <= float(best_static["assembly_cost_index_mean"])),
        "tail_service_win": (
            float(ppo["service_loss_auc_per_order_mean"])
            < float(best_static["service_loss_auc_per_order_mean"])),
        "cohens_d": d,
        "h3_static_cv": h3_static_cv,
        "h3_ppo_cv": h3_ppo_cv,
    }

    payload = {
        "config": {k: str(v) if isinstance(v, Path) else v
                   for k, v in vars(args).items()},
        "cells": [{"label": c.label, "risk_level": c.risk_level,
                   "family": c.family, "phi": c.phi, "psi": c.psi,
                   "demand_mult": c.demand_mult} for c in cells],
        "best_static": best_static,
        "best_reward_static": best_reward_static,
        "ppo": ppo,
        "deltas": {
            "ret_excel": ret_delta,
            "ret_excel_paired_ci95_low": delta_lo,
            "ret_excel_paired_ci95_high": delta_hi,
            "flow_fill_rate": (
                float(ppo["flow_fill_rate_mean"])
                - float(best_static["flow_fill_rate_mean"])),
            "service_loss_auc_per_order": (
                float(best_static["service_loss_auc_per_order_mean"])
                - float(ppo["service_loss_auc_per_order_mean"])),
            "assembly_cost_index": (
                float(ppo["assembly_cost_index_mean"])
                - float(best_static["assembly_cost_index_mean"])),
            "ctj_p99": (
                float(best_static["ctj_p99_mean"])
                - float(ppo["ctj_p99_mean"])),
        },
        "verdict": verdict,
        "cell_breakdown": cell_breakdown,
        "seeds_trained": len(seeds),
        "n_cells": len(cells),
        "n_static_policies": len(policies),
    }

    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2))
    write_csv(args.output_dir / "policy_summary.csv", policy_summary)
    write_status(args.output_dir, "done",
                 raw_ret_win=verdict["raw_ret_win"],
                 raw_ret_ci_win=verdict["raw_ret_ci_win"],
                 ret_delta=ret_delta,
                 ret_delta_ci95=[delta_lo, delta_hi])

    # ── H2: learning curve ──
    if all_curves:
        lc_rows = []
        for si, curve in enumerate(all_curves):
            for pt in curve:
                lc_rows.append({"seed": seeds[si], **pt})
        write_csv(args.output_dir / "learning_curve.csv", lc_rows)

    # ── Report ──
    print(f"\n{'='*60}", flush=True)
    print(f"  PPO ReT:  {ppo['ret_excel_mean']:.6f}", flush=True)
    print(f"  Best static ReT: {best_static['ret_excel_mean']:.6f}"
          f" ({best_static['policy']})", flush=True)
    print(f"  Δ ReT:    {ret_delta:+.6f}  "
          f"(paired CI95 [{delta_lo:+.4f}, {delta_hi:+.4f}])", flush=True)
    print(f"  Cohen's d: {d:+.4f}", flush=True)
    print(f"  CV (static): {h3_static_cv:.4f}  "
          f"CV (PPO): {h3_ppo_cv:.4f}", flush=True)
    print(f"  raw_ret_win={verdict['raw_ret_win']}  "
          f"same_reward={verdict['same_reward_win']}  "
          f"pareto={verdict['pareto_ret_cost']}", flush=True)
    print(f"\n  Per-regime win/loss:")
    for cell_label, bd in cell_breakdown.items():
        win = "+" if bd["delta_mean"] > 0 else "-"
        print(f"    {win} {cell_label}: "
              f"Δ={bd['delta_mean']:+.4f} "
              f"PPO={bd['ppo_mean']:.4f} "
              f"static={bd['static_mean']:.4f}", flush=True)
    print(f"\nWROTE {args.output_dir}", flush=True)
    return 0


# ═══════════════════════════════════════════════════════════════════════
#  utilities
# ═══════════════════════════════════════════════════════════════════════
def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows: return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def ci95(vals: list[float]) -> tuple[float, float]:
    """95% CI via normal approximation."""
    if len(vals) < 2:
        v = float(vals[0]) if vals else float("nan")
        return v, v
    a = np.asarray(vals, dtype=np.float64)
    h = 1.96 * a.std(ddof=1) / np.sqrt(len(a))
    return float(a.mean() - h), float(a.mean() + h)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        try:
            parser = build_parser()
            parsed, _ = parser.parse_known_args()
            if getattr(parsed, "output_dir", None):
                write_status(
                    parsed.output_dir,
                    "error",
                    error=str(exc),
                    traceback=traceback.format_exc(),
                )
        finally:
            raise
