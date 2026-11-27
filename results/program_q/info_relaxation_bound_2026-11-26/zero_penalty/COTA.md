# Cota de información perfecta (relajación de información, penalización cero) — Program Q

Fecha: 2026-11-26. Agente: Claude Opus 5.5, esfuerzo medium. Diagnóstico de solo lectura; no re-adjudica Q.
Semillas usadas: solo las **calibración quemadas 7480001–7480048** (las mismas 48/celda del caché clásico de la potencia v5).
No se abrieron 950100001–950100096, nada SEALED, ni las de confirmación 7490001–7490256.

## Resultado (48 cintas quemadas por celda; ReT = `ret_visible` del transductor full-DES = endpoint de Q)

| celda | U0 = E[max_c R] (SE) | V_OL (mejor calendario fijo) | V_CL (mejor clásico) | U0−V_CL [IC95 bootstrap pareado] | U0−V_OL [IC95] | aprendiz−V_CL |
|---|---|---|---|---|---|---|
| rho75_share90 | 0.9105 (0.0024) | 0.7523 (idx 27290) | 0.8297 (max_pressure__0) | **0.0809** [0.0733, 0.0853]; LCB95 unilateral 0.0744 | 0.1583 [0.1340, 0.1722] | −0.0016 |
| rho90_share75 | 0.9146 (0.0024) | 0.8003 (idx 27311) | 0.8656 (min_cost_flow__2) | **0.0490** [0.0420, 0.0540]; LCB95 0.0431 | 0.1143 [0.0947, 0.1255] | −0.0027 |
| rho90_share90 | 0.9125 (0.0024) | 0.7327 (idx 31743) | 0.8388 (max_pressure__0) | **0.0737** [0.0651, 0.0794]; LCB95 0.0664 | 0.1798 [0.1453, 0.1952] | −0.0015 |

Brecha por cinta U0−V_CL: mínimo 0.034 / 0.006 / 0.018; mediana 0.081 / 0.050 / 0.070. En **ninguna** de las 144 cintas el mejor clásico alcanza el máximo de la cinta. El argmax por cinta es distinto en 48/48/43 cintas.

**Veredicto (regla de decisión):** U0−V_CL ≫ 0.01 en las tres celdas (LCB95 ≥ 0.043). **Ninguna celda queda certificada cerrada** por la cota de penalización cero. Esto *no* demuestra que exista una política mejor: la cota de información perfecta es floja por construcción (el oráculo ve el futuro de pedidos/llegadas/riesgos). Lo que sí queda probado: ninguna política de ningún tipo (RL, MPC, LLM, incluso anticipativa) puede superar V_CL en más de ~0.081 / 0.049 / 0.074 en estas cintas.

Comparación: el H_PI de Program O (0.1515, otro endpoint/cintas) es del mismo orden que U0−V_OL aquí (0.11–0.18).

## Validez de la cota — verificado vs supuesto

Verificado (con salida):
1. **Determinismo dada la cinta.** El contrato evalúa cada política como `frontier[cinta, calendario]`: `scripts/evaluate_program_o_ret_learner.py` (l. 465–500) extrae un esqueleto exógeno por semilla (`extract_full_des_skeleton`) y evalúa aprendiz, clásicos y los 65,536 calendarios con `simulate_full_des_frontier` sobre el mismo esqueleto; `scripts/power_program_q_replication.py::load_panels` hace lo mismo (`frontier[tape, idx]`). Recalcular la frontera dos veces sobre la cinta 7480001 dio igualdad bit a bit (`determinism_bitwise_equal: true`), y un calendario evaluado solo coincide con su entrada de la frontera (diferencia 0.0). Como la acción es Discrete(4)×8 semanas, toda política (adaptativa o no) realiza en cada cinta uno de los 65,536 calendarios ⇒ R(π, t) ≤ max_c R(c, t) cinta a cinta ⇒ U0 es cota superior para cualquier política, sin supuesto de Markov ni de no anticipación.
2. **La recomputación reproduce los artefactos congelados:** índices clásicos recalculados = caché `classical_10_cache_v1.npz` (480/480 coincidencias, 0 discrepancias por celda); mejor calendario abierto = `best_open_loop_index` de la calibración (27290/27311/31743); mejor clásico = el de la calibración (max_pressure__0 / min_cost_flow__2 / max_pressure__0); aprendiz−V_OL = 0.075758 / 0.062612 / 0.104550 y aprendiz−V_CL = −0.001646 / −0.002732 / −0.001497, idénticos a `H_learned`/`H_neural` del `result.json` de calibración.
3. Transductor vs DES directo: la auditoría de confirmación (`.../confirmation/direct_audit/independent_full_des_audit.json`) repitió 21,696 calendarios únicos en el DES directo con 0 fallos a atol 1e-8 (verificado leyendo el JSON; no lo re-ejecuté).

Supuesto / no verificado:
- Que el `ret_visible` del transductor (`program_o_full_des_transducer.py` l. 694) implementa `ret_excel_request_snapshot_v2`: lo afirma la preregistración O-R y el contrato Q ("reward = ... ret_excel_request_snapshot_v2 terminal"); no lo re-derivé.
- Que el esqueleto exógeno es invariante a la acción para **todos** los 65,536 calendarios en el DES directo (la auditoría directa cubre solo los calendarios del aprendiz). Para la cota *del valor medido por el contrato* esto no importa: el contrato mide con el transductor.
- U0 de 48 cintas es estimador insesgado de E[max_c R] (la cota); se reporta su SE. V_OL y V_CL, en cambio, están sesgados hacia arriba por selección (max sobre 65,536 o 10), lo cual hace las brechas U0−V conservadoras (más pequeñas). El bootstrap re-selecciona el máximo dentro de cada remuestreo (20,000 remuestreos para V_CL; 2,000 para V_OL).

## Cota penalizada Brown–Smith–Sun (2010): factibilidad (no se corrió piloto)

Diseño: penalización z_t = V̂_{t+1}(s_{t+1}) − E[V̂_{t+1}(s_{t+1}) | F_t, a_t] con V̂ = valor de continuación del mejor clásico; cota = E_t[max_c (R(c,t) − Σ_t z_t(c,t))]. Depende solo del prefijo de acciones: Σ_{k=0..7} 4^k = 21,845 prefijos ×4 acciones por cinta.
- **Cuello de botella (ingeniería, no cómputo):** E[·|F_t] exige muestrear el futuro exógeno *condicionado a la historia observada por la política* (que excluye `latent_regime`, `tape_id`, `seed`). El DES actual genera la cinta entera desde una semilla; no hay API para "continuar desde la semana t con la historia fija". Hace falta (a) separar el RNG por semana/flujo, (b) un filtro (partículas o exacto) sobre el régimen latente con ρ y share conocidos, (c) reanudar el transductor en mitad del episodio. Estimo 1–3 días de implementación + pruebas de que la penalización tiene media cero (prueba ejecutable: media empírica de z_t ≈ 0 bajo cualquier política).
- **Cómputo:** frontera completa ≈ 3 s/cinta en el Mac (10 núcleos; 144 cintas en 4 min con 8 procesos; VPS 6 núcleos). Con M = 32 continuaciones internas por (cinta, semana) y ~4 s por continuación (frontera + calendarios clásicos de continuación por prefijo): 144 × 8 × 32 × 4 s ≈ 41 h-CPU ≈ 5 h en 8 núcleos. Cabe en una noche; el ruido interno solo afloja la cota (sigue siendo válida si el estimador interno es insesgado e independiente).
- Recomendación: solo vale la pena si se quiere certificar el cierre; la brecha a cerrar (0.05–0.08) es grande, y una penalización buena tendría que absorber casi todo el valor de la clarividencia.

## Archivos y comandos

Entradas (clon superficial de `Thom-320/scres-ia` main a10e99a en un clon local), sha256:
- `contracts/program_q_frozen_policy_replication_v1.json` 8b25e494…2b74
- `results/program_q/power_preopen_v5_20260718/artifacts/classical_10_cache_v1.npz` a7ef6607…3405 (= `cache_sha256` del contrato)
- `results/program_o/ret_only_learner_v1/calibration_run/result.json` dc55880f…54fd
- `supply_chain/program_o_full_des_transducer.py` 84158cd4…aa67; `program_o_full_des.py` 200d748e…e685; `program_o_state_rich.py` f00677c8…9aa3; `program_o_ret_env.py` 456adfd0…3fbc953bce; `contracts/program_o_full_des_hpi_translation_v1.json` b62943da…c2b1

Búsqueda de matrices crudas `raw_calendar_matrix` de la calibración (`program-o-ret-calibration-v12-20260717`): no están en el repo (gitignored), ni en `~/Projects/research/scres-ia*`, ni en el espejo de Drive (solo corridas O de 2026-07-15), ni en el VPS (`~/program_o_runs` ya no existe; `~/program_o_runs_archive` solo tiene `compresion.log`). Por eso se recalcularon.

Scripts (en `scripts/`): `frontier_tape.py` (1942f75b…9508), `run_all.py` (45c049c2…c2e), `analyze_bound.py` (cb946c3e…e2ec2e). Entorno: venv uv Python 3.12 con numpy 2.4.6, simpy 4.1.2, gymnasium 1.3.0, pandas 3.0.3, scipy 1.17.1 (versiones de `requirements.txt`).

```
python -I scripts/frontier_tape.py <repo> <scratch>/frontier 0 7480001 --check-determinism   # 13.4 s, determinismo
python3 -I scripts/run_all.py <venv>/bin/python <repo> cota/frontier 8                       # 01:17:28→01:21:32, 144 tareas, 0 fallos
python -I scripts/analyze_bound.py <repo> cota/frontier cota/bound_result.json               # 10 s
```
Salidas: `frontier/` (144 npz, 46 MB, manifiesto `frontier_manifest.sha256` 871772f0…c13), `bound_result.json` (c6571d29…d9a5) con todos los números de la tabla, medias de los 10 clásicos y cuantiles por cinta.
- El hueco crudo contra V_CL ya incluye el 0 en las tres celdas: con 48 cintas no se distingue el techo de la política clásica.
- La extrapolación es diagnóstico (3 puntos M=8/16/32), no cota. La cota válida es la de M finito.
- Pendiente: reps 4–7 de Kaggle → M=64.

## Cifra final M=64 (reps 0–3 VPS + 4–7 Kaggle; `penalizada/full_M64.json`, `extrap_M64.json`), verificada por Opus
| celda | U0 − V_CL (clarividencia) | U_pen − J(κ) (control variate) | IC95 | U_pen − V_CL crudo [IC95] | hueco eliminado |
|---|---|---|---|---|---|
| rho75_share90 | 0,081 | **0,0045** | [0,0035, 0,0056] | 0,0053 [−0,0041, 0,0149] | 93 % |
| rho90_share75 | 0,049 | **0,0030** | [0,0022, 0,0038] | 0,0027 [−0,0058, 0,0115] | 94 % |
| rho90_share90 | 0,074 | **0,0068** | [0,0045, 0,0094] | 0,0045 [−0,0053, 0,0147] | 94 % |

**Veredicto:** se cumple la regla de parada (UCB95 del hueco < 0,01) en las tres celdas. Ninguna política no anticipativa (RL incluido) puede superar al controlador clásico κ en más de 0,0056–0,0094 de valor esperado (escala 0–1, UCB95) sobre estas cintas; el hueco crudo contra V_CL no se distingue de 0.

- La cota es válida a M finito y solo puede bajar con más M: la secuencia M=8/16/32/64 es monótona (p. ej. 0,0160 → 0,0095 → 0,0065 → 0,0053).
- Extrapolación M→∞ (diagnóstico, no cota): 0,002–0,004 con 1/M; el ajuste 1/√M da valores negativos, señal de que no es la forma correcta.
- Media cero: máx |z| = 1,80 en 51 pruebas. U_pen ≥ J(κ) en todas las cintas. Réplicas 4–7 distintas de 0–3 (comprobado en las mismas cintas).
- Salvedades vigentes: cintas de calibración quemadas (7480001–048), 48 cintas, κ elegido entre 10 clásicos. Es un resultado de desarrollo; una confirmación pide cintas nuevas preregistradas.
