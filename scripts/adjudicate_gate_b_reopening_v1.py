"""Adjudicate the Gate B reopening against its own preregistration.

The runner emits its predecessor's falsifiers; the preregistered estimand is
different and lives in per_seed:

  Delta_k = R2(neural k) - R2(best non-neural, RESELECTED inside each bootstrap
            resample), paired by seed, 10,000 resamples over seeds.

Decision rule fixed before running (preregistration section 4):
  PREMIUM_k      if LCB95(Delta_k) > SESOI
  EQUIVALENT_k   if UCB95 < +SESOI and LCB95 > -SESOI   (TOST at +/- SESOI)
  UNDETERMINED_k otherwise
"""
import json
import math

import numpy as np

SESOI = 0.05
B = 10_000
RNG = np.random.default_rng(20260827)

d = json.load(open("/tmp/kfinal/gate_b_reopening_power_v1.json"))
per_seed = d["per_seed"]

NEURAL = ["mlp_tuned", "kan_tuned", "recurrent"]
classical = [a for a in per_seed if a not in NEURAL]

# seeds scored by every arm that produced predictions
usable = [a for a in per_seed if len(per_seed[a]) > 0]
seeds = sorted(set.intersection(*[set(per_seed[a]) for a in usable]), key=int)
n = len(seeds)
print(f"brazos con puntuacion: {len(usable)} de {len(per_seed)}")
print(f"sin puntuacion (instrumento): {[a for a in per_seed if not per_seed[a]]}")
print(f"n semillas pareadas: {n}\n")

M = {a: np.array([per_seed[a][s] for s in seeds]) for a in usable}
cls = [a for a in classical if a in M]

print("=== F1: MDE80 medido desde la sd POR SEMILLA ===")
Z = 1.959963985 + 0.8416212336
rows = []
for k in NEURAL:
    if k not in M:
        print(f"  {k:12s} SIN DATOS")
        continue
    best = max(cls, key=lambda a: M[a].mean())
    diff = M[k] - M[best]
    sd = diff.std(ddof=1)
    mde = Z * sd / math.sqrt(n)
    rows.append((k, best, diff.mean(), sd, mde))
    print(f"  {k:12s} vs {best:24s} sd={sd:.4f}  MDE80={mde:.4f}  "
          f"{'<= SESOI OK' if mde <= SESOI else '> SESOI FALLA'}")

print("\n=== ESTIMANDO PREREGISTRADO (comparador reseleccionado en cada remuestreo) ===")
verdicts = {}
for k in NEURAL:
    if k not in M:
        verdicts[k] = "UNDETERMINED_NO_INSTRUMENT"
        print(f"  {k:12s} UNDETERMINED_NO_INSTRUMENT (el brazo no produjo prediccion)")
        continue
    boot = np.empty(B)
    reselected = {}
    for b in range(B):
        idx = RNG.integers(0, n, n)
        means = {a: M[a][idx].mean() for a in cls}
        best = max(means, key=means.get)
        reselected[best] = reselected.get(best, 0) + 1
        boot[b] = M[k][idx].mean() - means[best]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    point = M[k].mean() - max(M[a].mean() for a in cls)
    if lo > SESOI:
        v = "PREMIUM"
    elif hi < SESOI and lo > -SESOI:
        v = "EQUIVALENT"
    else:
        v = "UNDETERMINED"
    verdicts[k] = v
    top = sorted(reselected.items(), key=lambda x: -x[1])[:3]
    print(f"  {k:12s} Delta={point:+.4f}  CI95=[{lo:+.4f}, {hi:+.4f}]  -> {v}")
    print(f"               comparador reseleccionado: "
          f"{', '.join(f'{a} {c*100//B}%' for a, c in top)}")

print("\n=== VEREDICTO GLOBAL ===")
vals = set(verdicts.values())
if "PREMIUM" in vals:
    overall = "PREMIUM"
elif vals == {"EQUIVALENT"}:
    overall = "EQUIVALENT"
else:
    overall = "UNDETERMINED"
print(f"  {overall}   {verdicts}")

print("\n=== contexto: R2 medio por brazo ===")
for a in sorted(M, key=lambda x: -M[x].mean())[:8]:
    tag = " [NEURAL]" if a in NEURAL else ""
    print(f"  {a:28s} {M[a].mean():.4f}{tag}")
