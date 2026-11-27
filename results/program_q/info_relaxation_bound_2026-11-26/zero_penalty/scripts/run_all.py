"""Driver: 3 cells x 48 burned calibration tapes 7480001..7480048, N parallel subprocesses.
Usage: python run_all.py PYTHON REPO OUTDIR [WORKERS]"""
import subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
py, repo, out = sys.argv[1:4]; w = int(sys.argv[4]) if len(sys.argv) > 4 else 8
script = str(Path(__file__).with_name("frontier_tape.py"))
tasks = [(c, s) for c in range(3) for s in range(7480001, 7480049)]
def go(t):
    r = subprocess.run([py, "-I", script, repo, out, str(t[0]), str(t[1])], capture_output=True, text=True)
    return t, r.returncode, r.stderr[-300:]
with ThreadPoolExecutor(w) as ex:
    fails = [x for x in ex.map(go, tasks) if x[1] != 0]
print("tasks", len(tasks), "failures", len(fails)); [print(f) for f in fails[:5]]
