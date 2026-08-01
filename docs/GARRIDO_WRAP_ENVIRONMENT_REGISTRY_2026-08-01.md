# Registro de entornos Garrido–WRAP — 2026-08-01

Este registro evita transferir resultados entre físicas, métricas o contratos que no son el
mismo experimento. La prima neural sólo es interpretable dentro del entorno que la produjo.

| entorno | qué pregunta | estado actual | claim permitido |
|---|---|---|---|
| **E0 / thesis-native WRAP** | ¿Una familia neural supera al lineal en el panel de Garrido? | Q1 `NO_GO_NEURAL_PREMIUM_IN_WRAP_PANEL`; fidelidad conductual global aún `HOLD` | el lineal es suficiente en el panel; backprop/KAN no alcanzan el SESOI |
| **Figura 5 literal** | ¿Los drivers post-episodio pueden ser entradas para elegir una configuración no corrida? | invalidada como tarea de planificación; `drivers → ReT` es identidad/leakage | los drivers actualizan el estado después de la corrida; no rankean candidatos futuros |
| **Replay thesis90** | ¿La interfaz de memoria/orden puede reproducirse sobre Cf1–Cf90? | `SURFACE_REPLAY_ONLY`; falsadores corregidos pasan | replay algorítmico, no réplica DES independiente |
| **DES-288 Q2** | ¿`rho` retenido entre contextos mejora la búsqueda bajo endpoint service-first? | artefacto DES presente; `ARTIFACT_PRESENT_CANONICAL_CUSTODY_PENDING` | sólo el estimando `retained − reset` tras reconciliación completa |
| **E1 / CSSU split** | ¿La contención no fungible mueve el valor de la acción? | Gate A `PASS`; Gate B físico de Op11 `HOLD`; headroom observado `0`, pero placebo neural no abierto | interfaz CSSU viva; no Op11 físico validado ni prima neural |
| **Expedición** | ¿Las palancas de tiempo abren headroom? | `NO_TIMING_HEADROOM_UNDER_SERVICE_FIRST` bajo ese contrato | negativo acotado al contrato de expedición ejecutado |
| **Program O** | ¿Un recurso compartido no fungible puede crear decisión? | headroom de desarrollo alto, pero física y métrica propias | evidencia motivadora separada; no se transfiere al WRAP |
| **Program Q / Track A/B/PPO** | ¿Control adaptativo o RL supera comparadores en otros sustratos? | carriles separados; claims históricos/pre-auditoría no se transfieren | no aporta claims al artículo Garrido–WRAP/v0 |
| **G3a prospectivo** | ¿Asimetría de demanda y riesgo localizado crean valor observable de asignación? | diseño-only; `NOT_AUTHORIZED_UNTIL_SUBMISSION_A_RECEIPT`; dos reclamantes | sólo el contrato G3a; no cierra N=3 ni autoriza learners |
| **E2 prospectivo** | ¿La memoria/recurrente ayuda con estado latente aliasado? | no abierto; sólo podrá entrar tras un gate físico/observable y un contrato nuevo | sólo un nuevo preregistro puede abrirlo |

## Cambios de física y precio de fidelidad

- `thesis_1to1` no se modifica.
- La API CSSU se hizo accionable en `split_v1`, con latencia de 24 horas y conservación de masa.
  Eso cambia la interfaz, no autoriza afirmar manejo físico finito.
- `op11_handling_hours` sigue sin conectar el camino `op9_linked`; elegir una distribución o
  capacidad por orden/lote a partir de “less than one hour” sería una decisión nuestra no
  identificada por la tesis.
- El endpoint service-first v2 se estipuló para bloquear abandono; es un endpoint normativo,
  no una validación externa de la tesis.
- La expedición y E1 son extensiones: sus negativos o positivos no reescriben ReT histórico.

## Dónde puede existir una prima neural

La hipótesis que permanece abierta es condicional, no arquitectónica:

1. debe existir variación de valor entre estados y regímenes;
2. la acción debe estar físicamente viva;
3. el mejor control constante, umbral, finito, lineal y MPC/DP limitado no debe alcanzar el
   oráculo;
4. la observación feed-forward debe dejar estados relevantes aliasados;
5. la memoria retenida debe mejorar sobre un reset con la misma información y presupuesto;
6. la mejora debe sobrevivir a placebo, tapes vírgenes y guardarraíles de servicio.

E2 es el candidato natural porque introduce sólo después de un nuevo contrato un régimen latente
persistente, observación retrasada/incompleta y decisiones diarias con efecto sobre la cola. Si
una regla finita o MPC resuelve ese entorno, la conclusión será “no hacía falta una red”. Si MLP
resuelve y PPO no mejora, la conclusión será “no-linealidad útil, RL innecesario”. Sólo si PPO
recurrente supera al mejor control clásico en tapes vírgenes existirá una prima neural publicable.
