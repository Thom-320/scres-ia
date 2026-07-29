#!/usr/bin/env python3
"""Monitor fetched directory for Kaggle sweep results. Alert when data arrives."""
import os, sys, time, json
from pathlib import Path

OUT = Path(os.environ.get("WATCHER_OUT",
    "outputs/experiments/track_b_adaptive_sweep_kaggle_2026-07-01_v5"))
FETCH = OUT / "fetched"
LOG = OUT / "fetch_monitor.log"

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

def alert(title, body=""):
    """macOS notification."""
    try:
        import subprocess
        subprocess.run(["osascript", "-e",
            f'display notification "{body}" with title "{title}" sound name "Glass"'],
            timeout=5)
    except:
        pass

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log("fetch monitor start")
    
    seen_summaries = set()
    prev_count = 0
    
    while True:
        summaries = set()
        if FETCH.exists():
            summaries = set(str(p) for p in FETCH.rglob("summary.json"))
        
        count = len(summaries)
        new = summaries - seen_summaries
        
        if count > 0 and count > prev_count:
            log(f"found {count} summaries ({len(new)} new)")
            prev_count = count
            seen_summaries = summaries
            
            # Try to check if any summary has COMPLETE results
            for sp in sorted(summaries):
                try:
                    s = json.loads(open(sp).read())
                    phase = s.get("phase", "")
                    if "done" in str(s).lower() or "verdict" in s:
                        log(f"  potential complete result: {sp}")
                except:
                    pass
        
        # Check if sweep workbook exists (indicates postprocessing done)
        workbook = OUT / "fetched" / "track_b_adaptive_sweep_audit.xlsx"
        sweep_results = OUT / "sweep_results.json"
        
        if workbook.exists() or sweep_results.exists():
            log("SWEEP COMPLETE - workbook or results found")
            alert("Kaggle sweep DONE", f"{count} configs complete")
            log("DONE")
            return 0
        
        time.sleep(120)  # 2 min

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log("interrupted")
        sys.exit(130)
