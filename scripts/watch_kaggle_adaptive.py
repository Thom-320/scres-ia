#!/usr/bin/env python3
"""Poll Kaggle adaptive sweep kernel, download results when done."""
import json, os, sys, time, csv, statistics
from pathlib import Path
from subprocess import run, PIPE

KERNEL = "thomaschisica/scresia-track-b-adaptive-sweep"
OUT = Path(os.environ.get("WATCHER_OUT", 
    "outputs/experiments/track_b_adaptive_sweep_kaggle_2026-07-01_v5"))
LOG = OUT / "watcher_live.log"
FETCH = OUT / "fetched"
INTERVAL = 300  # 5 min
MAX_WAIT = 6 * 3600  # 6h

KAGGLE = os.path.expanduser("~/.local/bin/kaggle")

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

def status():
    r = run([KAGGLE, "kernels", "status", KERNEL], capture_output=True, text=True, timeout=30)
    return r.stdout.strip()

def fetch():
    log("downloading output...")
    FETCH.mkdir(parents=True, exist_ok=True)
    r = run([KAGGLE, "kernels", "output", KERNEL, "-p", str(FETCH)],
            capture_output=True, text=True, timeout=300)
    log(r.stdout[-200:] if len(r.stdout) > 200 else r.stdout)

def analyze():
    summaries = sorted(FETCH.rglob("summary.json"))
    log(f"found {len(summaries)} summary files")
    results = []
    for sf in summaries:
        try:
            s = json.loads(sf.read_text())
            cfg = s.get("config", {})
            rm = cfg.get("reward_mode", "?")
            ov = cfg.get("observation_version", "?")
            alpha = cfg.get("ret_excel_cvar_alpha", "")
            name = sf.parent.name if sf.parent != FETCH else sf.stem
            ppo = s.get("ppo", {})
            bs = s.get("best_static", {})
            if ppo:
                ret = float(ppo.get("order_level_ret_mean_mean", 
                           ppo.get("order_ret_excel_mean", 0)))
                bsret = float(bs.get("order_level_ret_mean_mean",
                              bs.get("order_ret_excel_mean", 0)))
            else:
                ret = float(s.get("ppo_order_level_ret_mean", 0))
                bsret = float(s.get("best_static_order_level_ret_mean", 0))
            delta = ret - bsret
            win = "WIN" if delta > 0 else "LOSS"
            results.append((delta, name, ret, bsret, win, rm, ov, alpha))
        except Exception as e:
            log(f"error reading {sf}: {e}")
    
    results.sort(key=lambda x: -x[0])
    log("\n" + "="*70)
    log("SWEEP RESULTS (ranked by Excel ReT delta)")
    log("="*70)
    for delta, name, ret, bsret, win, rm, ov, alpha in results:
        tag = f"alpha={alpha}" if alpha else ""
        log(f"  {win:5s} {name:<50s} PPO={ret:.6f} static={bsret:.6f} Δ={delta:+.6f}  {rm} {ov} {tag}")
    
    # Save results
    json.dump([
        {"name": n, "delta": d, "ppo_ret": r, "static_ret": b, "win": w, 
         "reward": rm, "obs": ov, "alpha": a}
        for d, n, r, b, w, rm, ov, a in results
    ], open(OUT / "sweep_results.json", "w"), indent=2)
    
    # macOS notification
    run(["osascript", "-e", 
         f'display notification "Kaggle sweep done ({len(results)} configs)" with title "SCRES-IA"'],
        timeout=5)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"watcher start kernel={KERNEL}")
    start = time.time()
    
    while True:
        elapsed = time.time() - start
        if elapsed > MAX_WAIT:
            log("TIMEOUT")
            break
        
        s = status()
        log(f"status: {s}")
        
        if "COMPLETE" in s or "complete" in s:
            fetch()
            analyze()
            log("DONE")
            return 0
        
        if "ERROR" in s or "FAILED" in s or "error" in s or "CANCELLED" in s:
            log("KERNEL ERROR - fetching logs")
            try:
                fetch()
            except:
                pass
            return 1
        
        time.sleep(INTERVAL)

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log("interrupted")
        sys.exit(130)
