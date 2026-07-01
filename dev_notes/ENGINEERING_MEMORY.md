# SCRES-IA Engineering Memory — Lessons Learned

## CRITICAL: Watchers on EVERY job
- **Never launch a job without a watcher.** The watcher must:
  1. Log the PID and start time
  2. Poll every 60s for process existence (kill -0 PID)
  3. Check for output files (summary.json, report.md, transfer.json)
  4. Alert if process dies unexpectedly (exit code != 0)
  5. Print results automatically when done
  6. Log any errors to a separate error log
- **For Kaggle kernels:** The watcher must also check kernel status via Kaggle API and report "running", "error", or "complete"

## CRITICAL: Verify Kaggle kernels actually started
- After pushing a Kaggle kernel, WAIT and verify:
  1. Kernel status is "running" (not "error" or "queued")
  2. Output files are being produced
  3. No import errors or missing files
- Common Kaggle failures:
  - Missing payload files (tar.gz not found)
  - Import errors (wrong package versions)
  - GPU/CPU quota exhausted
  - Disk space issues

## CRITICAL: BC target for Track A
- **Use `--teacher best_static` NOT oracle.** The oracle from the gate is `op30_op50_op90_S2` (action [0,0,0,0]) which gives excel=0.154676 at the robust constant.
- The BEST STATIC from the full frontier re-evaluation is `op30_op50.1_op90_S2` (action [0,0.1,0,0]) with excel=0.155254.
- Teaching the oracle means PPO starts at 0.154676 and needs +0.000578 to beat the best static.
- Teaching the best static means PPO starts at 0.155254 and needs -0.000000 to tie or +0.000001 to win.

## CRITICAL: Python output buffering with nohup
- `nohup python script.py > log &` does NOT flush output. Log files stay at 0 bytes until the process exits.
- **Fix 1:** Use `python -u` flag (unbuffered)
- **Fix 2:** Use `PYTHONUNBUFFERED=1` env var
- **Fix 3:** Use `script -q /dev/null python script.py` (force pseudo-TTY)
- **Fix 4:** Add `sys.stdout.reconfigure(line_buffering=True)` at top of script
- **Fix 5:** Use `flush=True` on all print statements
- **Best practice:** Always use `python -u` + `PYTHONUNBUFFERED=1` + explicit flushes

## CRITICAL: H4 retention_transfer with continuous PPO is too slow for local
- 10 seeds × 30 blocks × 200 PPO updates × 52 max_steps = ~8+ hours local
- H4 died at seed 1/10 after 8.5h. It's too slow for local overnight.
- Use Kaggle for H4 or reduce to max_steps=12 as a probe only.

## KNOWN: Track A is on the edge
- Median ties best static (0.155247 vs 0.155254)
- Top-5 mean barely above (+0.000011 to +0.000277)
- Best seeds beat by +0.0010
- BUT: CI95 still crosses zero, mean still below static
- The win requires: correct BC target + multidiscrete + checkpoint selection + more seeds

## KNOWN: Track B variables
- Op12 dispatch is the bottleneck
- 5 optimal Track B variables: Op12 rate, Op10 rate, Op9 rate, backorder depth, Op9 buffer

## File locations
- Engineering runner: `scripts/run_track_a_engineered.py`
- V2 runner: `scripts/run_track_a_engineered_v2.py`
- Training repair runner: `scripts/run_track_a_training_repair.py`
- Retention transfer: `scripts/retention_transfer.py`
- Conflict campaign: `scripts/run_per_op_conflict_campaign.py`
- Amplified gate: `scripts/run_track_a_headroom_search.py`
