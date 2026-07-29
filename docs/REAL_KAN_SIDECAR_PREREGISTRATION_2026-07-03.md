# Real-KAN (pykan) Track B sidecar — preregistration, 2026-07-03

## Why this is different from every prior "KAN" result

Two things already exist and are NOT this:

1. `scripts/kan_extractor.py` (`RBFKANFeaturesExtractor`) — an RBF/KAN-inspired
   layer with a full linear skip. Positive on Track B but explicitly not a KAN
   (`docs/KAN_REAL_DEMO_2026-07-02.md`).
2. `scripts/run_kan_scres_demo.py` — the official `pykan` `KAN` class, but
   fit supervised (decision-vars -> Excel ReT on the 147-cell static
   frontier). R²=0.998. Not an RL policy; does not act online.

`docs/ARCHITECTURE_REVIEW_2026-07-02.md` flagged the missing piece explicitly:
*"A real KAN as an actual PPO policy/value network... nontrivial: pykan's
`KAN` class does not natively slot into SB3's `BaseFeaturesExtractor`
pattern... would need a custom wrapper."* This preregistration is that
wrapper, built and smoke-verified today.

## What was built

`scripts/real_kan_extractor.py::RealKANFeaturesExtractor` wraps the official
`kan.KAN` class (pip package `pykan`, Liu et al. 2024, learnable B-spline edge
functions — the actual KAN mechanism) as an SB3 `BaseFeaturesExtractor`.
`scripts/run_track_b_real_kan_sidecar.py` plugs it into the same Track B smoke
pipeline used for the DMLPA and RBF-KAN sidecars (`run_track_b_smoke.py`),
same monkeypatch pattern, so the comparator table (static grid + heuristics)
and evaluation protocol are identical.

**Feasibility finding, not obvious in advance:** pykan's `KAN.forward` is
usable with standard PyTorch autograd + `torch.optim.Adam` (confirmed:
forward+backward+step succeeds, no LBFGS required), but its default
`save_act=True`/`symbolic_enabled=True` bookkeeping (activation caching for
spline plots, symbolic-regression tracking) makes a single-sample forward
pass ~160x slower than needed for online RL (measured 0.046s vs 0.00028s per
call on this machine, at width `[52,32,16]`, batch=1). Disabling both flags
makes an online PPO loop of a normal Track B scale (tens of thousands of env
steps) computationally feasible: projected ~17s of pure extractor forward
cost for 60k rollout-collection steps, ~0.011s per gradient step at
batch=256. This is the concrete technical answer to "can real KAN work here
at all" — yes, with those two flags off during training; they can be
re-enabled afterward on a frozen copy to produce a spline plot for Garrido if
wanted.

Plumbing smoke (`outputs/experiments/track_b_real_kan_sidecar_2026-07-03/plumbing_smoke/`,
1 seed, 2000 timesteps, 1 eval episode) ran end-to-end with zero shape/dtype
errors and produced a full comparator table (statics + heuristics + real-KAN
policy). This confirms the wrapper works; it says nothing yet about whether
real KAN beats PPO+MLP — that requires the confirmatory run below.

## Canonical evaluation contract (unchanged from the DMLPA/RBF-KAN sidecars)

- `make_track_b_env`, `adaptive_benchmark_v2`, `v7` observation, `track_b_v1`
  (full 8D) action contract, `control_v1` training reward, `max_steps=104`.
- Primary metric: `order_level_ret_mean` (Garrido Excel/order-level ReT).
- Comparator: same dense static grid + 6 heuristics as every other sidecar.
- CRN rule: identical `(seed, episode, eval_seed)` keys.

## Architecture specifics for this sidecar

- `KAN(width=[52, hidden_width, features_dim], grid=3, k=3, grid_range=[-6,6])`.
- Defaults: `hidden_width=32`, `features_dim=32` — kept small deliberately
  because this is a feasibility/small-scale test, not a scaled architecture
  search; grid/k kept at pykan's fast defaults for the same reason.
- `net_arch={"pi": [], "vf": []}` (pure feature extractor, linear heads),
  matching the DMLPA/RBF-KAN sidecar convention.
- No frame-stacking (unlike DMLPA) — real KAN is tested on the same
  single-frame v7 observation as canonical PPO+MLP, isolating the
  architecture question from the separate history question.

## Decision rule (same bar as the rest of the architecture bakeoff)

Promote to the manuscript spine only if real-KAN:

1. Beats canonical PPO+MLP mean `order_level_ret_mean` on the same protocol.
2. >=4/5 seeds positive.
3. Does not collapse to a near-constant policy (action-variability check).

Otherwise: joins the architecture-robustness appendix alongside RecurrentPPO
(lost), DMLPA (lost, -0.000393), RBF-KAN (tied/marginal). This is the direct,
literal answer to Garrido's repeated KAN question — not a demo on a static
147-cell table, an actual trained-and-evaluated online policy.

## Scale of this run

Given real KAN's extra per-step cost (still small in absolute terms after
disabling `save_act`/`symbolic_enabled`, but larger than MLP/RBF), and this
being explicitly framed as a small feasibility test rather than the full
5-seed/60k architecture-bakeoff scale: run on the local Mac (per standing
instruction to use local compute, not only the VPS, and to avoid contending
with the DMLPA VPS corroboration job already in flight), 3 seeds x 30,000
timesteps x 12 eval episodes, same h104/control_v1/adaptive_benchmark_v2/v7
protocol. If positive or borderline, scale to the full 5-seed/60k protocol as
a follow-up before citing it as a confirmatory result.
