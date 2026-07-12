# Track B RecurrentPPO/LSTM History Sidecar Watch — 2026-07-03

## Why This Exists

David's DKANA proposal argues that the model may need history: a sequence of
DCA/DES states, local attention over row-wise state representations, and global
attention over the state sequence. The fair DMLPA bakeoff tests one explicit
windowed-attention implementation. This sidecar tests the smaller question:
whether recurrent memory alone helps under the corrected Track B environment.

## Active Run

- Output directory:
  `outputs/experiments/track_b_recurrent_ppo_2026-07-03/confirm_3seed_30k_h104/`
- Observed local PID: `30118`
- Command observed:

```bash
python3 -W ignore scripts/run_track_b_smoke.py \
  --output-dir outputs/experiments/track_b_recurrent_ppo_2026-07-03/confirm_3seed_30k_h104 \
  --seeds 1 2 3 \
  --train-timesteps 30000 \
  --eval-episodes 12 \
  --reward-mode control_v1 \
  --risk-level adaptive_benchmark_v2 \
  --observation-version v7 \
  --max-steps 104 \
  --n-steps 1024 \
  --batch-size 256 \
  --algo recurrent_ppo
```

## Interpretation Rule

This is a sidecar, not a Paper 1 promotion gate.

- If it beats the best static comparator but remains below PPO+MLP, it supports
  the current paper's claim that action-space alignment matters more than adding
  memory.
- If it beats PPO+MLP under the same metric, it becomes a candidate architecture
  for a confirmatory 5-seed/60k rerun before any manuscript claim changes.
- If it fails to beat static, it weakens the "history alone solves it" argument.

DKANA is still a larger architecture: this LSTM run only tests recurrent memory,
not symbolic/matricial preprocessing or local/global attention.
