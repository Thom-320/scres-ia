# Auditoría integral de prima neural, RNN y reaperturas

**Fecha:** 2026-08-09

**Rama de auditoría:** `agent/neural-premium-audit`

**Decisión prospectiva de sucesor:** Program X / O-Scale (`bbb35be`)

**Alcance:** árbol `HEAD`, refs históricas/off-HEAD localizadas, borrador v.0, tesis/artículos de
Garrido y workbooks ReT. No es un censo criptográficamente completo de todos los objetos Git.

El registro regenerado sobre `HEAD` leyó 266 artefactos, colapsó cinco re-reportes y dejó 261
entradas distintas: 3 confirmatorias, 11 diagnósticas, 25 negativas/halted, 60 replay, 103 de
desarrollo y 59 sin contrato. Es un **índice parcial**, no el universo probatorio: su builder recorre
globs del checkout y sólo incorpora dos familias off-HEAD de forma explícita. Omite, entre otras,
la confirmación terminal de Q y varias ramas Q2/T/U. Por eso estos conteos describen el snapshot
indexado; las adjudicaciones de abajo citan además la autoridad concreta `ruta@SHA`. Los grados
provienen del registro de evidencia cuando éste cubre el artefacto, no del `claim_status` escrito por
cada runner. Autoridad del índice: `scripts/build_evidence_registry_v1.py@8ddf6f7` y recibo
regenerado `results/evidence_registry/result.json@bbb35be`.

## Adjudicación ejecutiva

Sí existe un resultado neuronal real, pero es más estrecho que «prima neural»:

> **Q mostró ventaja ReT dependiente del estado frente al frontier open-loop, equivalencia frente
> al mejor clásico y un veredicto compuesto `STOP` porque falló el guardarraíl de peor-producto.
> No demostró que una red supere al mejor controlador estructurado.**

En tres celdas intactas, neural−mejor clásico fue −0,00159, −0,00072 y −0,00041; los tres
contrastes quedaron dentro de equivalencia ±0,01. La red también falló el margen de peor-producto
por una distancia estrecha: sus LCB fueron −0,02266, −0,02566 y −0,02632 contra −0,02.

Por tanto:

- no es correcto decir «RL no funciona»;
- sí es correcto decir «no hay prima neuronal de **calidad** sobre el mejor clásico»;
- relajar fairness no crea esa prima, porque el delta neural−clásico ya es aproximadamente cero;
- RNN no es el ingrediente ausente: ya fue probada y no dejó residual;
- la mejor oportunidad plausible es una prima de **amortización/generalización** al escalar el
  mecanismo mejor respaldado: productos no fungibles compartiendo un cuello.

## Cuatro afirmaciones que no deben volver a mezclarse

| afirmación | estimando correcto | estado actual |
|---|---|---|
| el feedback vale | adaptativo−open-loop | sí, O/Q |
| retener historia vale | retained−reset/delayed/shuffled | sí en V abstracto y outer-loop |
| una red aporta calidad | neural−mejor estructurado | no detectado |
| una red amortiza cómputo | no inferioridad + SLA/p95 material | no probado todavía |

La frase «prima neural» queda reservada a las dos últimas y siempre debe decir cuál.

## Los intentos que más cerca estuvieron

### 1. Program Q — el candidato legítimo y terminal

En la confirmación de Q, RecurrentPPO venció el frontier open-loop completo con puntos
+0,07952, +0,07255 y +0,11724 y LCB simultáneos +0,06608, +0,06233 y +0,10614. Eso demuestra
política dependiente del estado en tapes vírgenes. Frente al mejor clásico state-rich quedó:

| celda | neural−mejor clásico | IC95 |
|---|---:|---:|
| 1 | −0,00159 | [−0,00627, +0,00310] |
| 2 | −0,00072 | [−0,00552, +0,00408] |
| 3 | −0,00041 | [−0,00268, +0,00186] |

Es el proyecto más cerca de ganar porque pasó equivalencia, mostró ventaja ReT dependiente del
estado frente a open-loop en tapes vírgenes y usó el frontier abierto completo. Perdió
superioridad porque el belief/state-rich clásico captura
la estadística suficiente. Su peor-producto también falló el margen −0,02: LCB −0,02266,
−0,02566 y −0,02632. Q es **inmutable y terminal** en la autoridad off-HEAD
`docs/PROGRAM_Q_TERMINAL_VERDICT_2026-07-18.md@f2dfe356`; X o una prueba fairness-aware exigirían
nuevo contrato, nuevas tapes y nuevo nombre; no son una reapertura de Q.

O-R es el antecedente histórico, no el resultado confirmatorio de Q: contra open-loop obtuvo
+0,06261…+0,10455 con LCB +0,03659…+0,06630, pero quedó −0,00150…−0,00273 frente al clásico.
No se deben atribuir sus LCB a Q.

### 2. K3 — el mayor win numérico, invalidado

PPO−MPC apareció +0,017708, IC95 [+0,010417,+0,026042]. La auditoría encontró que PPO emitía una
secuencia fija de período ocho que el frontier estático no había enumerado; una política fija la
reproduce exactamente. Es un confound de comparador, no aprendizaje. No se reabre dentro de K3.

### 3. Program V abstracto — memoria causal sin residual neuronal

Bayes retained−reset fue +0,041320 [+0,026572,+0,056068], y retained venció delayed/shuffled. Pero
privileged−Bayes fue sólo +0,000764 [−0,000798,+0,002326]. La memoria importa; el filtro Bayesiano
la agota. En el DES completo, todos los contrastes fueron exactamente cero porque la acción de
proveedor no llega al endpoint.

### 4. Transferencia outer-loop — el carrier ganador fue clásico

La neurona con memoria venció cold start por +0,05439, pero perdió contra su propio marginal replay
por −0,01178 [−0,01849,−0,00484]. UCB1 sí venció marginal replay por +0,03073
[+0,01990,+0,04256]. Hay valor de experiencia, no prima de la neurona.

### 5. Track B-P y G3a — mecanismos, no premium

Track B-P ganó +0,02849 al añadir postura, pero una postura fija absorbió todo: fijo−dinámico
+0,00044, IC cruzando cero. Para G3a deben separarse tres objetos: (i) el headline histórico
reportado, H_obs≈+0,0963 [+0,0682,+0,1245], cuyo runner/contrato original no se recuperó; (ii) la
reconstrucción forense de 18.360 filas/34 políticas, que no es original ni confirmación; y (iii) el
falsador v2, H_obs +0,002789 [−0,007635,+0,012373], `G3A_DID_NOT_REPRODUCE`. Ninguno autoriza
learner nuevo en la misma física.

## Inventario terminal del repo

Las letras A/B/C son tracks; no existen Programs A/B/C canónicos independientes. Tampoco hay
Programs N, P o R independientes; P aparece como anexo y R como sufijo O-R.

`Histórico` significa resultado preservado de su campaña; `confirmatorio`, tape virgen bajo contrato;
`desarrollo/diagnóstico`, evidencia que no adjudica un claim confirmatorio; `reconstrucción`, objeto
forense que no sustituye al original; y `prospectivo`, diseño sin resultados científicos. Un SHA
off-HEAD sigue siendo autoridad Git, pero no lo cubre el registro de evidencia del checkout.

| lane | comparador, métrica y estimación verificable | clase y causa de pérdida/retractación | reapertura limpia | autoridad `ruta@SHA` |
|---|---|---|---|---|
| Track A | oracle +0,0041566 ReT; PPO−static −0,006316, 0/5 | Histórico/desarrollo; headroom menor que ruido y learner no lo convierte | sólo mecanismo nuevo | `docs/TRACK_A_V2_CONSERVATION_PPO_VERDICT_2026-07-03.md@67040f7` |
| Track B, reparación | PPO−constante full-8D −0,000018049, IC [−0,000028615,−0,000008087] | Histórico/adjudicado; el win previo usaba frontier restringido | no en mismas tapes | `docs/TRACK_B_SAME_CONTRACT_CHALLENGE_VERDICT_2026-07-10.md@441d8d5` |
| Track B, réplica limpia | PPO−full static +0,0000062, IC [−0,0000066,+0,0000184] | Réplica histórica, FAIL; segunda estimación compatible con cero, no con prima material | sólo contrato nuevo | `docs/TRACK_B_CLEAN_REPLICATION_PROTOCOL_2026-07-10.md@a02a5c0` |
| Track B recurrente | RecurrentPPO−MLP −0,0000588, 0/3; sí venció al static débil +0,000406 | Sidecar histórico; historia/recurrencia sin señal residual frente a MLP canónico | sólo si un gate nuevo prueba aliasing | `docs/TRACK_B_RECURRENT_PPO_HISTORY_SIDECAR_VERDICT_2026-07-03.md@67040f7` |
| Track B-P | 11D−8D +0,028488 [0,015813,0,041163]; fijo−dinámico +0,000440 [−0,000799,+0,001680] | Histórico/desarrollo; postura fija explica la ganancia | completar frontier; no prometer premium | `docs/TRACK_BP_GATE2_SCREEN_VERDICT_2026-07-09.md@ab3c8ed` |
| Track C | switcher−constante máximo +6,47e−5 vs barra material ≈1,5e−4 | Histórico/negativo; compromiso de 168 h domina la información | sólo nueva época/física | `docs/TRACK_C_ORACLE_PHASE_VERDICT_2026-07-10.md@311d375` |
| D1 | age-threshold−SPT +0,0387 [0,0366,0,0407]; branching +1,41e−5 | Histórico; mejora estática, casi cero valor de branching | no dentro de D1 | `docs/PROGRAM_D_D1_V2_BRANCHING_VERDICT_2026-07-11.md@368f381` |
| DRA1 | oracle dinámico +0,000087895 [0,000027576,0,000165914] | Histórico/negativo; magnitud no material | no | `docs/PROGRAM_D_DRA1_FINAL_VERDICT_2026-07-11.md@9ad0ac7` |
| DRA2/2b | ReT +0,02662 [0,01650,0,03732] / +0,02212 [0,01725,0,02734]; servicio +3,075 % [1,987,4,033] | Histórico/desarrollo; DRA2 inestable y 2b no alcanza gate de servicio 5 % | nueva familia, no tuning interno | `docs/PROGRAM_D_DRA2B_LONG_HORIZON_FINAL_VERDICT_2026-07-12.md@3bcf6e9` |
| E | MaskablePPO−envelope −0,000050621 [−0,000320016,+0,000200923], 0/10 | Histórico/terminal; árbol/PPO no convierten OOS | no retuning interno | `docs/PROGRAM_E_FINAL_VERDICT_2026-07-12.md@59bfd21` |
| E* | 192 llamadas DES; p95 0,155257 s | **Engineering-only**; pasó el gate firmado por llamadas (>60), no por latencia (presupuesto 60.480 s); no hubo seeds frescas, learner ni ejecución científica. La regla de llamadas sola no prueba cuello operacional | sólo con gate de deadline absoluto o break-even repetido predeclarado | `results/estar_hcompute_preflight_v1/result.json@50833e3` |
| F | H_PI hasta +0,022584; observable 0/24 | Histórico/terminal; información privilegiada no desplegable | sólo señal física nueva | `docs/PROGRAM_F_FINAL_VERDICT_2026-07-12.md@e830116` |
| G | cover−ABAB ReT −0,02317 [−0,02816,−0,01893]; cantidad −0,00695 | Histórico/terminal; proxy de service-loss invertía ReT/fairness | no cambiar métrica post hoc | `docs/PROGRAM_G_TERMINAL_METRIC_AUDIT_VERDICT_2026-07-12.md@006b41c` |
| GSA corregido | H_obs +0,0122868 [0,0086777,0,0160432], n=120; peor CSSU −0,129167 | Reanálisis correctivo de confirmación: sólo elige AAAA (31) o ABAB (89), ambas ya en el comparador; es selector de calendario de un bit, no adaptación secuencial/neural | sólo contrato fairness nuevo; reporte no vinculante bajo decisión PI | `results/gsa_confirmation_corrective/result.json@d3f8de5` |
| G3a, headline | H_obs≈+0,0963 [0,0682,0,1245] reportado | Histórico no recuperado: faltan runner/contrato original | recuperar objetos originales; no reejecutar como si fueran intactos | citado y delimitado en `results/g3a_forensic_reconstruction_v1/result.json@a4f1a6f` |
| G3a, forense | 18.360 filas, 34 políticas reconstruidas | Reconstrucción; no es original ni confirmatoria | sólo para auditoría | `results/g3a_forensic_reconstruction_v1/result.json@a4f1a6f` |
| G3a, falsador v2 | H_obs +0,002789 [−0,007635,+0,012373] | Desarrollo/falsador: `G3A_DID_NOT_REPRODUCE` | no rescata headline; mecanismo nuevo | `results/g3a_boundary_v2/result.json@c18c027` |
| H | fitted-Q +0,00225 [−0,00021,+0,00460]; PI +0,01641 [0,01397,0,01886] | Histórico/terminal; residual observable insuficiente | sólo señal predecisional nueva | `docs/PROGRAM_H_FINAL_VERDICT_2026-07-13.md@017b19a` |
| I branching | production +0,0000395; dispatch +0,0000113; transport +0,0000105 | Histórico/terminal; reversals reales, magnitud ínfima | no | `docs/PROGRAM_I_BRANCHING_STOP_2026-07-12.md@dd3b921` |
| J | oracle service-loss +419,1 [275,3,578,1]; PPO 0/6, deltas −4,4…−105,3 | Histórico/terminal; residual requiere futuro y PPO aprende calendario | sólo señal física nueva | `docs/PROGRAM_J_PAPER2_RL_VERDICT_2026-07-12.md@9a586cf` |
| K | +161…+212 en reward compuesto | Histórico **retractado**; shelf life 2 semanas no fuente (~3 años), ganancia por waste/servicio plano-peor y fuga de futuro; no ReT | no rescatar | `docs/PROGRAM_K_PERISHABLE_RL_WIN_2026-07-12.md@53350f4` más auditoría posterior |
| K2 | constante≈MPC | Histórico/terminal; comparador fuerte cierra la puerta | no | `docs/PROGRAM_K2_STRONG_COMPARATOR_VERDICT_2026-07-12.md@407b5ea` |
| K3 | PPO−MPC aparente +0,017708 [0,010417,0,026042] | Histórico **retractado**; PPO es una secuencia fija de ocho períodos omitida del frontier | no dentro de K3 | `results/k3/open_loop_confound_audit.json@ef6b53b` |
| L | pico H_obs +0,005411; LCB −0,007353; 0/18 LCB>0; H_PI no medido | Desarrollo/negativo; valor de alternar, no de saber; columna anterior era `heuristic_true_state_delta`, no H_PI | sólo uno de cuatro sucesores: L-1 full-ledger/Cobb; L-2 contention/fungible-fleet null; nueva física R03; o dispatch trigger/timing como decisión | `results/paper2_search/program_l_l0_adjudication.json@f34db8d` y `docs/DECISION_PI_ENDPOINT_Y_APERTURA_PROGRAM_L_2026-08-07_ENMIENDA_1.md@f34db8d` |
| M | H_PI +0,0411478, LCB +0,022779 en dos celdas | Histórico/validation stop; punto aislado, sin región conectada | no añadir celdas post hoc | `results/program_m/hpi_validation_v1/result.json@67a5640` |
| O | belief-MPC−open-loop +0,09852/+0,07347/+0,09974; LCB +0,06595/+0,04303/+0,05860 | Confirmación histórica clásica; falla CVaR, no es premium neural | O cerrado | `docs/PROGRAM_O_CORRECTIVE_HOBS_VALIDATION_VERDICT_2026-07-15.md@adbfb8f` |
| O-R | RecurrentPPO−open-loop +0,06261…+0,10455; LCB +0,03659…+0,06630; −0,00150…−0,00273 vs clásico | Histórico/calibración; feedback sí, residual neuronal no | sucesor Q ya adjudicado; O-R no reabre | `research/paper2_exhaustive_search/program_o_ret_calibration_v12_terminal_audit_20260717.json@821c8d8` |
| Q | vs open-loop +0,07952/+0,07255/+0,11724; LCB simultáneos +0,06608/+0,06233/+0,10614; vs mejor clásico −0,00159/−0,00072/−0,00041, todos equivalentes ±0,01 | **Confirmatorio off-HEAD e inmutable**; no alcanza superioridad +0,01 y peor-producto LCB −0,02266/−0,02566/−0,02632 < −0,02 | no; X/fairness serían sucesores con contrato/tapes nuevos | `results/program_q/confirmation_v1_20260718/artifacts/confirmation/adjudication.json@f2dfe356` |
| Q-R1 | retained−reset +0,0656285 [0,0524386,0,0788550]; peor producto −0,0216115 [−0,0364514,−0,0077288] | Confirmatorio prospectivamente detenido, off-HEAD; mecanismo medio con coste de fairness; sin learner | contrato sucesor nuevo, no reapertura | `results/q_r1/successor_confirmation_v1/adjudication.json@7bccf86` |
| Q2/QRDQN | +0,04266 vs estático; −0,054694 [−0,07010,−0,04038] vs estructurado; peor producto −0,1185 | Probe dinámico off-HEAD/negativo; baseline débil fabrica el win | no | `results/program_q2/qrdqn_dynamic_probe_v1/adjudication.json@def601c` |
| S | sólo preflights | Prospectivo antiguo/HOLD; transductor invalidado, reward terminal ReT y RecurrentPPO fijado; además esperaba Q, ya terminado off-HEAD | rediseñar, no ejecutar tal cual | `docs/PROGRAM_S_IMPLEMENTATION_STATUS_2026-07-17.md@1f1b2b6` |
| T | ceiling +0,0222566, LCB +0,01810; observable +0,002344, LCB +0,001253 | Desarrollo off-HEAD; casi todo el residual no es observable | sólo mecanismo nuevo | `results/program_t/causal_residual_campaign_v1/verdict.json@94b8dfe` |
| U/U1 | una celda H_obs +0,118513, LCB +0,027286; adyacentes negativas | Desarrollo off-HEAD; punto aislado y exactness falla | nueva campaña preregistrada, no promover el punto | `results/program_u1/direct_connected_region_v1/result.json@94b8dfe` |
| V abstracto | Bayes retained−constant +0,179366 [0,162989,0,195743]; retained−reset +0,041320 [0,026572,0,056068]; privileged−Bayes +0,000764 [−0,000798,0,002326] | Desarrollo/abstracto; memoria causal, pero Bayes absorbe el residual neuronal | sólo amortización/escalado | `results/program_v/prelearner_gate_v1/result.json@a4f1a6f` |
| V full DES | seis contrastes exactamente 0 | Replay de desarrollo; acción de proveedor no llega al cuello y colisiona con custodia previa | sólo física declarada nueva | `results/program_v/des_port_v1/result.json@5a2922f` |
| V storage | servicio 0,7415→0,5523; todos los headrooms siguen 0 | Desarrollo; estantería ata servicio, no material | cerrar | `results/raw_scarcity_boundary/result.json@fab2566` |
| W | objeto/artefacto original no localizado en el clon y refs remotas inventariadas; `git cat-file e761ef4` falla | Incidente sin autoridad publicada dentro de ese alcance | recuperar objeto original o reconstrucción explícitamente forense | `docs/INCIDENTE_PROGRAM_W_NO_PUBLICADO_2026-08-09.md@bbb35be` |
| X / O-Scale | preflight: 4/20/120 acciones y 65.536/25.600.000.000/42.998.169.600.000.000 calendarios; headroom/historia/premium aún no medidos | **Prospectivo, diseño-only**; `DESIGN_CONSISTENCY_PASS__NO_SCIENTIFIC_GATE_OPENED`; sin rangos, seeds, learner ni autorización científica | no “reabrir”: primero diagnóstico causal y perfil real del planner | `contracts/program_x_o_scale_amortized_control_v1.json@bbb35be`; `results/program_x/o_scale_design_preflight_v1/result.json@bbb35be` |

### Inventario de arquitecturas neuronales

Estos sidecars no son Programs nuevos y no deben mezclarse con DKANA por semejanza de nombre.

| familia | comparador/métrica y resultado | clase y lectura | reapertura | autoridad `ruta@SHA` |
|---|---|---|---|---|
| DMLPA posicional | Excel ReT pareado: MLP 0,005920 > DMLPA 0,005871 > history-MLP 0,005832; DMLPA−MLP en `order_ret_excel` −0,000049, IC cruza cero | Histórico/fair bakeoff; sin ventaja arquitectónica | sólo si cambia el estimando/entorno | `docs/TRACK_B_ARCHITECTURE_FAIR_BAKEOFF_VERDICT_2026-07-03.md@67040f7` |
| DMLPA variants | ReT medio del episodio: nhead4−base +1,588729, LCB +0,639081; 1layer−base +1,210492, LCB +0,609177 | Desarrollo sobre seeds ya abiertas; selección/tuning, no confirmación | requiere contrato y bloque nuevo para claim | `results/dmlpa_variants/result.json@2d52d4c` |
| DMLPA–KAN latent | `ret_mean_track_b_v1`: KAN−MLP −0,862251 [−1,605044,−0,119458] | Desarrollo/negativo; KAN latent peor | no en el mismo contrato | `results/dmlpa_kan_latent/result.json@e9642a9` |
| KAN surrogate | regret AUC KAN−MLP +0,010369 [0,003018,0,018926], donde menor es mejor | Desarrollo/negativo; KAN peor que MLP pareado | sólo hipótesis nueva | `results/surrogate_architecture_bakeoff/result.json@ecd4daa` |
| DKANA | existe pipeline/env/dataset/train/eval, pero no checkpoint entrenado y evaluado con resultado autoritativo en `results/` | Ingeniería/prospectivo; **no es** DMLPA–KAN latent y no hay estimación científica | puede completarse sólo tras gate causal, contrato y seeds autorizadas | `supply_chain/dkana.py@02508b1`; `docs/DKANA_CONTRIBUTOR_HANDOFF.md@3a65c50` |

## Por qué RL puede perder en un problema difícil

La dificultad del problema es una condición insuficiente. Para que RL tenga prima deben coincidir
cinco cosas:

1. **Acción causal:** la decisión debe mover la restricción que determina el endpoint. V full DES
   falla aquí.
2. **Información utilizable:** antes de comprometerse debe existir una señal que cambie la acción
   óptima. J/F/T fallan total o parcialmente aquí.
3. **Residual sobre estructura:** la política clásica no debe contener ya la estadística suficiente.
   Q y V abstracto fallan aquí.
4. **Escala amortizable:** resolver exactamente debe ser caro en la cadencia real. E* aprobó
   formalmente su gate firmado porque 192 llamadas DES excedían el presupuesto de 60; no aprobó un
   gate de latencia, pues p95 fue 0,155257 s frente a un presupuesto de 60.480 s. Era un preflight
   `engineering_only`, sin seeds frescas, learner ni ejecución científica. El gate por número de
   llamadas fue demasiado débil para establecer un cuello operacional; X debe exigir deadline
   absoluto o break-even de consultas repetidas, además de la meta 10× en p95 de latencia. Las
   llamadas DES quedan como diagnóstico y no sustituyen ese gate.
5. **Objetivo válido y comparador completo:** K3, G y el ReT terminal muestran que un reward o
   frontier incompleto puede fabricar un win.

Un problema puede ser combinatoriamente difícil y aun así tener una regla de umbral o belief de
baja dimensión. RL también paga exploración, varianza y aproximación; OR usa estructura y garantías.
Por eso la literatura prometedora es híbrida. Böttcher, Asikis y Fragkos muestran RNN de inventario
útiles cuando hay dinámica de inventario y demanda empírica no estacionaria
([INFORMS JOC, 2023](https://pubsonline.informs.org/doi/10.1287/ijoc.2022.0136)). Harsha et al.
combinan una red de valor con programación entera para espacios combinatorios, en vez de pedirle a
PPO que redescubra factibilidad
([M&SOM, 2025](https://pubsonline.informs.org/doi/10.1287/msom.2022.0617)).

El artículo IJPE 2026 adjunto sobre MARL tampoco contradice esta auditoría: compara principalmente
MAPPO con QMIX/MADDPG y otros learners, no contra el belief-MPC/DP/OR completo que aquí elimina la
prima. Es evidencia de que MARL puede ordenar una familia neural, no de superioridad sobre el mejor
controlador operacional.

## ¿Debemos usar RNN?

**No como arquitectura primaria; sí como candidato condicionado.**

RNN es memoria; RL es un criterio de entrenamiento. Son ortogonales. `L_(t-1)` puede ser un
posterior Bayesiano, estadísticas acumuladas, un case-base, parámetros actualizados o un hidden
state. Garrido no exige RNN: su figura es un neurón feed-forward que compara configuraciones y su
discusión incluye backprop, KAN y simulation-optimization/RL.

El gate RNN será:

- history-retained vence reset;
- vence delayed y shuffled;
- vence un MLP con belief explícito y mismo presupuesto;
- la acción ocurre antes de que el estado se revele;
- el efecto sobrevive fuera de muestra.

Si pasa, usar GRU pequeño primero y LSTM como sensibilidad. Si no pasa, DeepSets/MLP es más
parsimonioso. GNN sólo con topología/BOM real variable.

## ReT y Cobb-Douglas

Ambos se conservan y se reportan por separado.

### ReT Excel

La fórmula observada es, en esencia:

`IF(risk, IF(AP>0, AP/LT, 0.5/RP), 1-(B+U)/j)`.

La rama `0.5/RP` no está acotada. En los dos workbooks hay 38 celdas ReT>1; el máximo auditado es
160,2564 (`Raw_data2`, `CF12!AA2086`, RP≈0,00312 h). Por ello es fidelidad a la fuente y endpoint
secundario obligatorio, no reward denso único. Nunca se corrige silenciosamente.

### Cobb-Douglas

Es un índice APP de fábrica, no el ReT de la tesis. La fuente deja dos definiciones no equivalentes
de coste normalizado: el texto usa `κ̇=7κ/Σκ` —relativo al conjunto comparado—, mientras los
Algorithms 1–2 usan `κ̇=ΣC_t/T`. La implementación debe escoger y declarar una, o reportar ambas;
no se llamará sin matiz «el índice fuente». La ecuación publicada da signo positivo a
inventario `ζ` y capacidad ociosa `φ`, y negativo a backorders `ε`, tiempo `τ` y coste normalizado
`κ̇`; no incluye fill-rate. Sus exponentes fueron calibrados en otra simulación y
`κ̇ = |S|κ(s)/Σκ(s)` depende del conjunto comparado.

El port MFSC tiene dos cautelas adicionales verificadas. Primero, con `c=1`, en 10.368 celdas
`κ` es 85,650 % inventario y 13,734 % backorders, con `corr(κ,ζ+ε)=0,999993`: el coste vuelve a
contar `ζ/ε`. Bajo los exponentes O-Scale derivados, el coeficiente efectivo de `ln ζ` es
`+0,014274 − 0,446334×0,857 = −0,368`, inversión de signo y magnitud 26×. Ningún vector ensayado
hizo `κ̇` independiente; los cruces obtenidos rompiendo escala/independencia quedaron retirados.
Segundo, el port usa el mismo `U_t` como capacidad ociosa positiva en `φ` y como «capacidad
marginal» costosa `c_u U_t` dentro de `κ`. Puede ser el trade-off deliberado de la fuente, pero en
MFSC deben congelarse identidad física, unidades y signo; no se mapeará silenciosamente un campo a
ambos papeles sin reportar componentes crudos. Además, `U_t` no tiene la misma semántica entre
constructos: en Cobb-Douglas es capacidad ociosa y en ReT representa pedido no atendido.

Para Program X:

- se reporta el índice fuente como sensibilidad;
- el conjunto de referencia del coste se congela para que añadir una política no cambie a las demás;
- una recalibración O-Scale se etiqueta como métrica nueva y se separa del índice fuente;
- no se fusiona con ReT usando pesos elegidos después del resultado.

Autoridades: `docs/COBB_DOUGLAS_FACTORY_METRIC_AUDIT_2026-07-13.md@ff5e4a8`,
`docs/PREREGISTRO_COBB_DOUGLAS_REPARACION_ESCALA_2026-08-08.md@0191410` y
`results/cobb_douglas_scale_repair/result.json@71de444`.

La métrica física primaria será `service_resilience_auc_normalized = 1 − déficit medio de servicio`
en las épocas nativas, acotada en [0,1] y con mayor=mejor; peor-producto y pedidos perdidos serán
guardarraíles. ReT y Cobb-Douglas responden constructos de resiliencia complementarios.

## Corrección del 190.000

La tesis completa dice `190,000 units of each rm` tres veces en la Tabla 6.20 y enumera rm1…rm12
en Op2. No queda ambigüedad textual. Op2 aporta 570.000 componentes/semana frente a 180.000 de
demanda; Op3 S=1 aporta 186.000. Además, Op2 equivale a 47.500/rm/semana, casi exactamente Op3 S=3
de 47.000.

Bajar 190.000 para hacer vinculante al proveedor es una extensión válida si se declara y se mide
el precio de fidelidad. No es una reparación del port.

## Respuesta a Garrido

Las preguntas originales de Garrido son más amplias que H1–H4. Su análisis exploratorio sitúa la
propuesta en reconocimiento de patrones/nivel 3 y su Figura 5 usa un neurón feed-forward entre
corridas; no prescribe RNN ni demuestra control online:

1. ¿Qué categoría de AI puede imitar supply-chain learning?
2. ¿Cómo integrarla dentro del DES?

La respuesta empírica del repo es:

- **Q1:** una familia closed-loop con estado retenido puede imitar aprendizaje. En el inner-loop,
  RecurrentPPO mostró ventaja ReT dependiente del estado frente a open-loop; en el outer-loop,
  UCB1 confirmó transferencia. La red no es una condición necesaria y todavía no aporta calidad
  incremental sobre belief/control estructurado.
- **Q2:** el DES debe exponer un ledger causal, producir una observación sin futuro, actualizar
  `L_(t-1)`, decidir antes del shock relevante y devolver outcome/estado al siguiente ciclo. La
  identificación exige retained vs reset/delayed/shuffled y un comparador estructurado con la misma
  información.

## Qué pasa con H1–H4 del borrador

Esas cuatro hipótesis pertenecen al borrador nuevo; no son hipótesis probadas por el artículo
exploratorio de Garrido. La matriz autoritativa concluye
`3_OF_4_SUPPORTED_NONE_AS_WRITTEN...`: tres resultados apoyan objetos redefinidos o de búsqueda,
ninguno la redacción causal original. Todos son desarrollo/reanálisis, no confirmación neural.

| hipótesis | estimando exacto y resultado | clase epistemológica | qué no responde |
|---|---|---|---|
| H1 menor recovery time | `restricted_ttr=min(TTR,τ)`, τ=1344 h; horas ahorradas híbrido−static +125,9854167 [98,3467969,154,5447135], n=960 | Desarrollo sobre bloque ya abierto; endpoint redefinido | compara posturas/selector; no identifica causalidad neural ni control online. La H1 original no queda soportada *as written* |
| H2 mejora sucesiva | pendiente OLS de `(reset AUC − memory AUC)` sobre ordinal de seis contextos: +0,04220148 [0,03466394,0,04992206], n=120 | Reanálisis de desarrollo de artefactos sellados, sin seeds nuevas | el índice es un outer-loop entre DES reiniciados, no disrupciones sucesivas dentro de una corrida |
| H3 menor varianza de performance entre intensidades | contraste H3 almacenado −1,1094468323605106×10^15 [−3,648590718211967×10^15,+1,2929658238053142×10^15], n=360, p unilateral 0,8208 | Desarrollo; estimando vivo, signo contrario y escala extrema inducida por ReT | **no soportada**. H3′ mide varianza del coste de búsqueda y no la rescata |
| H4 path dependency en `R_t` | en V abstracto, Bayes retained−reset +0,041320 [0,026572,0,056068]; además, el outer-loop retained reduce AUC-regret en 6/6 familias, pero sólo 1/6 conserva LCB simultáneo >0 en simple regret final para la neurona | Desarrollo/reanálisis, no confirmación | V es un modelo abstracto de proveedor y Bayes, no red ni DES MFSC; en outer-loop `L_k` es estado de búsqueda, no resiliencia entregada en tiempo físico `t` |

Autoridad común: `results/v0_adjudication_matrix/result.json@023d053`; H4 outer-loop:
`results/retention_simultaneous/result.json@023d053`; H4 abstracta:
`results/program_v/prelearner_gate_v1/result.json@a4f1a6f`. La segunda mitad de la pregunta de
investigación —mejora de **precisión predictiva** en held-out— sigue `NOT_ANSWERED`.

H3′ está sostenida en el merge n=120 para su propio estimando: memoria−reset reduce la varianza de
coste de búsqueda en +9,3144 [+2,3491,+16,3474]. Pero la réplica local n=90 es +10,2659
[+2,3609,+18,1927] y la tajada VPS n=30 es por sí sola inconclusa: +6,4600
[−7,3088,+20,3281]. Las 90+30 seeds son disjuntas y el contrato/manifiesto coincide, pero ambas
fuentes llevan `DECLARED_REPLAY`: el bloque original se abrió una sola vez y luego se reejecutó
para corregir la ruta del contrato. No es una segunda confirmación independiente; la identidad del
snapshot VPS original no pudo reconstruirse, sólo la igualdad entre reejecuciones. No debe
renombrarse «volatilidad de servicio». Autoridad:
`results/garrido_h3_merge_adjudication/result.json@b8b34ea`.

Las reformulaciones causales recomendadas son:

- **H1:** bajo misma información y recursos, feedback observable reduce TTR restringido frente a
  open-loop.
- **H2:** con estado físico inicial emparejado, retained mejora la pendiente de desempeño entre
  ciclos frente a reset.
- **H3:** retained reduce varianza de servicio/coste sin empeorar media ni peor producto.
- **H4:** después de condicionar el estado físico, `L_(t-1)` cambia acción y outcome frente a
  reset/delayed/shuffled.
- **H5:** neural supera o amortiza al mejor controlador estructurado bajo contrato igualado.

H5 es indispensable: sin ella, H1–H4 pueden ser satisfechas completamente por Bayes, MPC o UCB1.

## Dónde ser un poco más laxos

La laxitud válida es prospectiva y semántica, no estadística retroactiva.

| cambio | permitido | razón |
|---|---|---|
| separar «ventaja dependiente del estado vs open-loop» de «desplegable» | sí | Q mostró ese contraste, pero el veredicto compuesto fue `STOP` por guardarraíl |
| no inferioridad + 10× cómputo para prima de amortización | sí, congelado antes | responde otro valor económico real |
| margen de guardarraíl con tolerancia operacional | sí, con dominio/power previos | margen cero murió por un pedido |
| física nueva y demanda estacional/stochastic PT | sí, como extensión y OOD | la fuente reconoce límites, pero no debe confundir el gate primario |
| seeds nuevas | sólo con autorización escrita del PI que declare rango nuevo, preregistro y runner commiteados antes de una única apertura | la custodia vigente registra **cero bloques disponibles** |
| bajar SESOI o mover IC después de mirar | no | redefine el win |
| eliminar classical/MPC | no | fabrica prima |
| escoger ReT o Cobb-Douglas ganador | no | endpoint shopping |
| bajar 190.000 sin etiqueta | no | contradice parámetro fuente explícito |

La autoridad de custodia más reciente supersede cualquier lenguaje anterior de “bloque reservado”:
`docs/GARRIDO_Q1_Q2_CLAIM_FREEZE_2026-08-07_ENMIENDA_4.md@bc2624f`. Program X no tiene rangos
asignados ni autorización científica; su preflight no abre una excepción.

## Dónde está la prima plausible

No está respaldada por más entrenamiento sobre Track B ni por poner LSTM sobre Program V. Program
X formula la hipótesis sucesora mejor apoyada, todavía no un resultado:

- N=2→4→8 productos no fungibles;
- tres lotes compartidos, total de recursos fijo;
- mezcla persistente parcialmente observada;
- belief-MPC completo como teacher y adversario;
- DeepSets destilada/DAgger; GRU sólo tras gate de historia;
- ReT y Cobb-Douglas obligatorios pero separados;
- servicio/fairness como física primaria;
- calidad, amortización y generalización como claims distintos.

La aritmética de diseño pasa de 65.536 a 25.600 millones y 4,30×10^16 calendarios. Eso sólo prueba
cardinalidad combinatoria: no demuestra que el planner implementado sea difícil/lento, que exista
headroom observable, que la historia añada información ni que una red conserve calidad. Program X
debe medir esos cuatro hechos antes de presentar la escala como oportunidad de amortización. Si el
planner sí incumple una cadencia predeclarada o un break-even repetido, una red podría monetizar
aproximación/generalización sin exigir superioridad de calidad sobre su teacher.

## Próximo punto de decisión

El contrato y su preflight determinista están versionados en `bbb35be`; no se abrió ninguna seed y
no existe autorización científica. El siguiente incremento debe implementar sólo:

1. kernel N-producto con paridad N=2;
2. conservación y nulos fungible/IID/señal-azar;
3. H_PI/H_obs/H_ret sin learner;
4. perfil operacional del planner.

Sólo si esos cuatro pasos pasan se congela una campaña de destilación. Es la ruta más corta hacia
una prima defendible y también la que puede falsarla antes de gastar cómputo.
