#!/usr/bin/env python3
"""What does a KAN actually learn on the design surface, and is it readable?

GARRIDO'S ARGUMENT HAS ONE LEG LEFT. He sells the KAN on two things: parameter economy and
interpretability. The first is measured and false -- at 200k matched parameters KAN minus MLP is
-0.475 [-1.548, +0.598] and KAN costs 4.1x per decision. The second we had never touched, and it is
exactly where the KANbeFair preprint does NOT contradict him, because that paper compares accuracy
and not auditability.

WHAT THIS IS. A supervised surrogate of the sealed 288 design surface: four design coordinates in,
resilience out. The KAN's first layer is a sum of learned UNIVARIATE functions of each coordinate,
so those functions can be sampled and read directly -- "what did it learn about buffer hours" is a
curve, not a weight matrix. That is the interpretability claim, made concrete.

WHAT THIS IS NOT. It is not a policy, it is not evidence of a neural premium, and it does not
reopen the architecture comparison. Accuracy is reported next to a parameter-matched MLP precisely
so the readability claim cannot be smuggled into an accuracy claim.

THE FALSIFIER THAT MATTERS IS f2. Smooth univariate curves are what a spline basis produces whether
or not the data has structure. So the whole thing is refit on a SHUFFLED surface: if the shapes
survive shuffling, they are artefacts of the basis and there is nothing to interpret. That control
is the difference between an interpretability result and a Rorschach test.

Development on the burned block. Descriptive; adjudicates nothing.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supply_chain.arm_runner import seal_and_write  # noqa: E402
from supply_chain.seed_custody import custody_falsifier, module_manifest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_search_comparator_ladder_v2 import (  # noqa: E402
    COORDS, FACTOR_NAMES, CONTEXT_ORDER, load_cache,
)

MODULES = ("supply_chain/arm_runner.py", "supply_chain/seed_custody.py")
SAMPLES = 41                 # puntos por curva univariada
EPOCHS = 400
SEED = 20260806


def fit_kan(x: np.ndarray, y: np.ndarray, width: int, steps: int = EPOCHS):
    import torch
    from kan import KAN
    torch.manual_seed(SEED)
    model = KAN(width=[x.shape[1], width, 1], grid=5, k=3,
                auto_save=False, save_act=False, symbolic_enabled=False)
    xt = torch.tensor(x, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(steps):
        opt.zero_grad()
        loss = ((model(xt) - yt) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = model(xt).numpy().ravel()
    return model, pred, sum(p.numel() for p in model.parameters())


def mlp_width_for(target: int, n_in: int) -> int:
    """Width whose parameter count lands closest to the KAN's.

    The first version passed the KAN's WIDTH to the MLP, which gave 532 against 31 parameters and
    called it matched. That is the exact comparison David objected to, and shipping it would have
    repeated his complaint with our name on it."""
    return max(1, min(range(1, 4096), key=lambda w: abs((n_in + 2) * w + 1 - target)))


def fit_mlp_full(x: np.ndarray, y: np.ndarray, width: int, steps: int = EPOCHS):
    import torch
    import torch.nn as nn
    torch.manual_seed(SEED)
    model = nn.Sequential(nn.Linear(x.shape[1], width), nn.Tanh(), nn.Linear(width, 1))
    xt = torch.tensor(x, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(steps):
        opt.zero_grad()
        loss = ((model(xt) - yt) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = model(xt).numpy().ravel()
    return model, pred, sum(p.numel() for p in model.parameters())


def univariate_curves(model, x: np.ndarray) -> dict:
    """Sample the network's response to each coordinate with the others held at their median.

    This is the readable object: for a KAN the first layer IS a sum of univariate functions, so
    this sampling recovers them rather than approximating something that was never separable."""
    import torch
    base = np.median(x, axis=0)
    out = {}
    for j, name in enumerate(FACTOR_NAMES):
        grid = np.linspace(x[:, j].min(), x[:, j].max(), SAMPLES)
        probe = np.tile(base, (SAMPLES, 1))
        probe[:, j] = grid
        with torch.no_grad():
            resp = model(torch.tensor(probe, dtype=torch.float32)).numpy().ravel()
        out[name] = {"grid": grid.tolist(), "response": resp.tolist(),
                     "monotone_increasing": bool(np.all(np.diff(resp) >= -1e-9)),
                     "monotone_decreasing": bool(np.all(np.diff(resp) <= 1e-9)),
                     "range": float(resp.max() - resp.min())}
    return out


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    ss = float(((y - y.mean()) ** 2).sum())
    return float(1.0 - ((y - pred) ** 2).sum() / ss) if ss else 0.0


def curve_distance(a: dict, b: dict) -> float:
    """Mean absolute difference between two curve sets, each normalised by its own range."""
    ds = []
    for name in FACTOR_NAMES:
        ra = np.asarray(a[name]["response"]); rb = np.asarray(b[name]["response"])
        na = (ra - ra.min()) / (ra.max() - ra.min() or 1.0)
        nb = (rb - rb.min()) / (rb.max() - rb.min() or 1.0)
        ds.append(float(np.mean(np.abs(na - nb))))
    return float(np.mean(ds))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=Path("results/surface_cache/wrap288_v1"))
    ap.add_argument("--width", type=int, default=5)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--replay-of", required=True)
    ap.add_argument("--output", type=Path,
                    default=Path("results/kan_interpretability/result.json"))
    args = ap.parse_args()
    surface, contexts, seeds = load_cache(args.cache)
    rng = np.random.default_rng(SEED)

    report = {}
    for ctx in [c for c in CONTEXT_ORDER if c in contexts]:
        y = np.mean([surface[(ctx, s)] for s in seeds], axis=0)
        y = (y - y.mean()) / (y.std() or 1.0)
        # Held-out split: 532 parameters on 288 points can memorise, so an in-sample R2 would be
        # reporting capacity rather than fit. The split is fixed by SEED and shared by both models.
        perm = np.random.default_rng(SEED).permutation(len(y))
        n_tr = int(0.75 * len(y))
        tr, te = perm[:n_tr], perm[n_tr:]

        model, _, n_kan = fit_kan(COORDS[tr], y[tr], args.width)
        import torch as _t
        with _t.no_grad():
            kan_te = model(_t.tensor(COORDS[te], dtype=_t.float32)).numpy().ravel()
        curves = univariate_curves(model, COORDS)

        w_mlp = mlp_width_for(n_kan, COORDS.shape[1])
        mlp_model, _, n_mlp = fit_mlp_full(COORDS[tr], y[tr], w_mlp)
        with _t.no_grad():
            mlp_te = mlp_model(_t.tensor(COORDS[te], dtype=_t.float32)).numpy().ravel()

        # f2: the control. Same fit on a shuffled surface -- if the curves survive, they are the
        # basis talking, not the chain.
        y_shuf = y[rng.permutation(len(y))]
        model_s, _, _ = fit_kan(COORDS[tr], y_shuf[tr], args.width)
        with _t.no_grad():
            shuf_te = model_s(_t.tensor(COORDS[te], dtype=_t.float32)).numpy().ravel()
        curves_s = univariate_curves(model_s, COORDS)

        report[ctx] = {
            "kan": {"r2_heldout": r2(y[te], kan_te), "parameters": int(n_kan)},
            "mlp_matched": {"r2_heldout": r2(y[te], mlp_te), "parameters": int(n_mlp),
                            "width": int(w_mlp)},
            "shuffled_control": {"r2_heldout": r2(y_shuf[te], shuf_te),
                                 "curve_distance_vs_real": curve_distance(curves, curves_s)},
            "curves": curves,
        }
        c = report[ctx]
        shapes = "  ".join(
            f"{n}:{'↑' if v['monotone_increasing'] else '↓' if v['monotone_decreasing'] else '~'}"
            f"{v['range']:.2f}" for n, v in curves.items())
        print(f"    {ctx:<14} KAN R2_out {c['kan']['r2_heldout']:+.4f} ({n_kan} par.) · "
              f"MLP R2_out {c['mlp_matched']['r2_heldout']:+.4f} ({n_mlp} par.) · "
              f"barajado {c['shuffled_control']['r2_heldout']:+.4f} "
              f"(dist. curvas {c['shuffled_control']['curve_distance_vs_real']:.3f})")
        print(f"                   formas: {shapes}")

    kan_r2 = [b["kan"]["r2_heldout"] for b in report.values()]
    shuf_r2 = [b["shuffled_control"]["r2_heldout"] for b in report.values()]
    dists = [b["shuffled_control"]["curve_distance_vs_real"] for b in report.values()]

    falsifiers = {
        "f1_there_is_something_to_interpret": {
            "passed": bool(np.median(kan_r2) > 0.5),
            "evidence": {"why_it_can_fail": "reading the curves of a model that does not fit is "
                                            "reading noise. If the median R2 is poor, the "
                                            "interpretability claim has no object",
                         "kan_r2": kan_r2}},
        "f2_the_curves_are_not_basis_artefacts": {
            "passed": bool(np.median(dists) > 0.05 and np.median(shuf_r2) < np.median(kan_r2)),
            "evidence": {"why_it_can_fail": "THE control. A spline basis produces smooth univariate "
                                            "curves whether or not the data has structure. If the "
                                            "shuffled surface yields the same shapes, the shapes "
                                            "are the basis talking and there is nothing to read",
                         "curve_distance_real_vs_shuffled": dists,
                         "shuffled_r2": shuf_r2}},
        "f3_the_mlp_is_actually_parameter_matched": {
            "passed": all(abs(b["mlp_matched"]["parameters"] - b["kan"]["parameters"])
                          <= 0.10 * b["kan"]["parameters"] for b in report.values()),
            "evidence": {"why_it_can_fail": "the first version passed the KAN's WIDTH to the "
                                            "MLP and got 532 parameters against 31 while calling "
                                            "it matched -- David's exact objection, with our name "
                                            "on it. This checks the counts, not the label",
                         "kan_vs_mlp": {c: {"kan_params": b["kan"]["parameters"],
                                            "mlp_params": b["mlp_matched"]["parameters"],
                                            "kan_r2_heldout": b["kan"]["r2_heldout"],
                                            "mlp_r2_heldout": b["mlp_matched"]["r2_heldout"]}
                                        for c, b in report.items()}}},
        "f4_no_fresh_seeds": custody_falsifier(seeds, replay_of=args.replay_of,
                                               exclude=args.output),
    }
    falsifiers["all_passed"] = all(
        v["passed"] for k, v in falsifiers.items()
        if k != "all_passed" and isinstance(v, dict) and not v.get("not_applicable"))

    print()
    for name, f in falsifiers.items():
        if name == "all_passed" or not isinstance(f, dict):
            continue
        lab = "NO APLICA" if f.get("not_applicable") else ("PASA" if f["passed"] else "FALLA")
        print(f"    {name:<48} {lab}")

    payload = {
        "schema_version": "kan_interpretability_v1",
        "claim_status": "DESCRIPTIVE_INTERPRETABILITY_DEMO_NO_ADJUDICATION",
        "scope": "DEVELOPMENT_ON_BURNED_TAPES_SUPERVISED_SURROGATE_NOT_A_POLICY",
        "run_role": "CACHE_ANALYSIS", "replay_of": args.replay_of,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "module_manifest": module_manifest(MODULES, script=__file__),
        "what_this_is_not": ("Not a policy, not evidence of a neural premium, and not a reopening "
                             "of the architecture comparison. Accuracy is reported beside a "
                             "parameter-matched MLP so readability cannot be read as accuracy."),
        "design_coordinates": list(FACTOR_NAMES), "width": args.width, "epochs": EPOCHS,
        "contexts": contexts, "seeds": seeds, "by_context": report,
        "falsifiers": falsifiers,
    }
    digest = seal_and_write(payload, args.output, contract=args.contract,
                            reference=Path("results/architecture_bakeoff/result.json"))
    print(f"\n  -> {args.output} (sello {digest[:16]}…)")
    return 0 if falsifiers["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
