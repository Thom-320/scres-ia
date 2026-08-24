# IDEAS OPENCODE — 10 low-cost activables con flags CLI existentes

**Fecha:** 2026-08-24 · **Rol:** implementador pragmático · **Restricción:** solo flags ya existentes (--gamma, control_v1_pbrs/--pbrs-*, --norm-reward en run_track_b_*, --critic-pretrain-epochs en run_track_a_v2_conservation_ppo.py). **No duplica** `PROMISING_LANES_REGISTRY.md:1` (continuous_its×v6×ReT_excel_delta/per_op_buffer ya falsificados).
**Verificado:** `FIXPACK_FEASIBILITY_OPENCODE.md:20-24` + `env_experimental_shifts.py:211,603-606,2469` + `benchmark_control_reward.py:513,553,696-718,724`.

---

### 1. PBRS bias shift para hacer shaping explotable con Q-init≈0 (Müller)
- **Paper:** `/home/ubuntu/scres-sources/texts/a3-mueller2025-arxiv-pbrs-effectiveness.txt`
- **Comando:** `python scripts/benchmark_control_reward.py --gamma 0.99 --reward-mode control_v1_pbrs --pbrs-variant cumulative --pbrs-alpha 1.0 --pbrs-beta 0.5 --pbrs-gamma 0.99` (`env_experimental_shifts.py:322` `pbrs_gamma` debe = `train_agent.py:121` gamma)
- **Claim grande:** Sin shift, `F=γΦ'-Φ` es inerte si `Φ` no alineada a `Q_init` (demuestra 30-50% ganancia sample-efficiency sin tocar red). Resuelve supresor reward terminal esparsa sin romper optimalidad Ng99.
- **Costo:** ~3 CPU-h (3 seeds×50k, `benchmark_control_reward.py:1133` PPO).

### 2. γ efectivo 0.95-0.97 / freezing slow-state para horizonte 260 semanas (Wang)
- **Paper:** `/home/ubuntu/scres-sources/texts/a4-wang2023-arxiv-freezing-slow.txt`
- **Comando:** `python scripts/run_track_b_gamma_rewardnorm_grid.py --gammas 0.95,0.97,0.99 --norm-reward-options 0,1 --train-timesteps 20000 --max-steps 104` + smoke `benchmark_control_reward.py --gamma 0.95 --reward-mode control_v1_pbrs`
- **Claim:** γ=0.99 ⇒ horizonte efectivo 100 pasos diverge con 200k; Wang prueba freezing `T=4` (γ^T=0.96) mantiene calidad con 5× menos cómputo. Gate inmediato para decidir artefacto-vs-física (SINTESIS:39).
- **Costo:** ~6 CPU-h (grid 3×2×3 seeds).

### 3. PBRS subgoal segmentado para densificar reward terminal (Okudo)
- **Paper:** `/home/ubuntu/scres-sources/texts/a1-okudo2021-ieee-access-subgoal.txt`
- **Comando:** `python scripts/benchmark_control_reward.py --reward-mode control_v1_pbrs --pbrs-variant step_level --observation-version v4 --pbrs-alpha 0.8 --pbrs-gamma 0.99` (requiere v2+ por `env_experimental_shifts.py:509` Φ usa `obs[16]` backorder_rate)
- **Claim:** Φ(s)=η·c(h) con c=subgoals alcanzados acelera 2-5× en sparse sin violar invarianza; mapea a SCRES: survive→restore fill>95%→min cost.
- **Costo:** ~3 CPU-h.

### 4. Recurrent MFRL bien configurado (separate LSTM, n_steps grande) (Ni)
- **Paper:** `/home/ubuntu/scres-sources/texts/a6-ni2021-arxiv-recurrent-pomdp.txt`
- **Comando:** `python scripts/benchmark_control_reward.py --algo recurrent_ppo --observation-version v4 --gamma 0.99 --n-steps 2048 --batch-size 512 --gae-lambda 0.95` (kwargs `benchmark_control_reward.py:513` `lstm_hidden_size:128, shared_lstm:False, enable_critic_lstm:True`)
- **Parche mínimo opcional (1 línea):** subir a 256 `RECURRENT_PPO_POLICY_KWARGS:515` si 128 no aprende.
- **Claim:** Ni demuestra recurrent MFRL gana sólo con arquitectura/hyper cuidados y off-policy; nuestro 128×1+200k es peor caso y explica Δ_N≈0.
- **Costo:** ~8 CPU-h (recurrent 1.3× slower).

### 5. Comparador justo no-invencible (Gijsbrechts)
- **Paper:** `/home/ubuntu/scres-sources/texts/a7-gijsbrechts2022-msom-can-deep-rl.txt`
- **Comando:** `python scripts/benchmark_control_reward.py --reward-mode control_v1 --risk-level increased --seeds 11 22 33 --eval-episodes 10` y comparar `comparison_table.csv` vs `heuristic_tuned`/`garrido_cf_s2` (mismo presupuesto 65k calendarios), reportar gap con 95%CI.
- **Claim:** DRL empata base-stock cuando modelo es exacto/penalización alta; nuestro empate belief-MPC es esperado, no muerte del proyecto. Publicable sólo vs MPC degradado o heurística calibrada.
- **Costo:** ~4 CPU-h.

### 6. Diseño R&S secuencial KN/FHN con IZ δ=0.01 para potencia N (Hong/Cheng)
- **Paper:** `/home/ubuntu/scres-sources/texts/a10-hong2021-fem-review-rs.txt` (+ Cheng 2023 EJOR)
- **Comando:** `python scripts/run_track_b_gamma_rewardnorm_grid.py --seeds 1,2,3,4,5 --train-timesteps 20000` + loop adaptativo: tras n0=10 seeds asignar réplicas al par con mayor var hasta KN elimine (implementar en `arm_runner.py`, 5 líneas, bound Paulson finite-sample).
- **Claim:** Varianza está en semillas (σ_seed≈0.032), no tapes. Secuencial minimiza expected sample size con garantía PCS≥1-α, evita LCB95 que cruza 0 con N fijo.
- **Costo:** 2-6 CPU-h adaptativo (ahorra vs N fijo grande).

### 7. Protocolo DT-MARL reproducible: matched seeds + 95%CI + Glass Δ (Guzmán cierra Alzheimer)
- **Paper:** `/home/ubuntu/scres-sources/texts/b4-guzman2026-cie-circular.txt` (+ `/home/ubuntu/scres-sources/texts/garrido2024_scres+AI.txt` Fig2 nodos ③↔⑧)
- **Comando:** `python scripts/benchmark_control_reward.py --seeds 11 22 33 --eval-episodes 10 --max-steps 260 --reward-mode control_v1_pbrs --pbrs-variant cumulative` con seeds vírgenes disjuntos (`EVAL_EPISODE_SEED_OFFSET=80000:307`) y export `manifest.json:1002`.
- **Claim:** 5-agent DT-MARL demuestra que disciplina de evaluación (no algoritmo nuevo) es contribución metodológica publicable; evita p-hacking post-hoc.
- **Costo:** ~4 CPU-h.

### 8. Smoke topológico 8-13 nodos no fungible filling/repairing/recruiting (Ding)
- **Paper:** `/home/ubuntu/scres-sources/texts/1-s2.0-S0925527326000861-main.txt`
- **Parche mínimo (1 línea):** `benchmark_control_reward.py:553` choices `v1-v6`→`v1-v10` (env ya 101 dims `env_experimental_shifts.py:221`)
- **Comando:** `python scripts/benchmark_control_reward.py --observation-version v10 --reward-mode control_v1_pbrs --gamma 0.97 --max-steps 52 --risk-level adaptive_benchmark_v2` vs placebo fungible (H_PI debe→0).
- **Claim:** Único headroom medido causal H_PI=0.1515 viene de recurso compartido no fungible Op5-Op7; Ding da blueprint MAPPO CTDE para escalarlo.
- **Costo:** ~6 CPU-h smoke; 20 CPU-h si escala 500k-1M (Kim IISE baseline).

### 9. APP pure strategies S11-S32 con cv/α/γ demand variability (Factory)
- **Paper:** `/home/ubuntu/scres-sources/texts/garrido2024_factory_resilience.txt` (+ `/home/ubuntu/scres-sources/texts/WRAP_Theses_Garrido_Rios_2017.txt` §6.5/§8.6)
- **Parche mínimo (2 líneas):** exponer `demand_mean_multiplier`/`initial_buffers` en `run_track_a_v2_conservation_ppo.py:286 make_train_env`
- **Comando:** `python scripts/run_track_a_v2_conservation_ppo.py --gate-dir outputs/experiments/track_a_v2_conservation_5d_gate_2026-07-03 --critic-pretrain-epochs 50 --gamma 0.97 --seeds 1,2,3 --timesteps 40000 --max-steps 36 --bc-epochs 150`
- **Claim:** Garrido IJPR prueba zero-inventory chase con smoothing cv=σGR/GR (Eq1 α/γ) supera level/hybrid en R Cobb-Douglas; test directo del tradeoff coste vs resiliencia MTS.
- **Costo:** ~5 CPU-h (hereda VecNormalize + BC, `run_track_a_v2_conservation_ppo.py:397`).

### 10. Estabilización VecNormalize + FrameStack/Transformer + critic-pretrain (Boute+Kong)
- **Paper:** `/home/ubuntu/scres-sources/texts/a8-boute2022-ejor-roadmap.txt` + `/home/ubuntu/scres-sources/texts/b10-kong2026-eai-transformer.txt`
- **Comando:** `python scripts/run_track_b_gamma_rewardnorm_grid.py --norm-reward-options 0,1 --gammas 0.99` (flag `run_track_b_gamma_rewardnorm_grid.py:40` → `VecNormalize(norm_reward=True):114`) + `benchmark_control_reward.py --frame-stack 4` + `run_track_a_v2_conservation_ppo.py --critic-pretrain-epochs 50` (`:329` fit `mlp_extractor.value_net:223` sin tocar actor).
- **Claim:** Roadmap fija que DRL sólo supera heurísticas con alta variabilidad/multi-echelon; norm-reward + stacking + critic-pretrain corrigen critic-lag (PPO erosiona BC 0/5 seeds, `-0.006` delta) sin nuevo env.
- **Costo:** ~4 CPU-h (norm grid) + ~5 CPU-h (critic-pretrain 50 epochs, `critic_pretrain:206` loss 95%↓).
