# Program B learner runtime observation — 2026-08-07

Estado: operativo, no es un resultado científico ni autorización para reiniciar jobs.

## Hechos verificados

- Los cinco jobs faltantes de `scripts/train_program_b_service_safe_learner.py` siguen vivos bajo las semillas/arquitecturas del contrato.
- A las `2026-08-07T08:00:52Z`, los cinco hijos Python consumían aproximadamente `22,8–23,0%` de CPU cada uno y ~`259 MB` RSS cada uno; todavía no existían checkpoints `.zip` en `results/program_b_learner_v1/`.
- Dos hijos habían acumulado aproximadamente `49,5 GB` de `read_bytes` en `/proc/<pid>/io`; varios estaban en estado `D` y en espera relacionada con `mem_cgroup_handle_over_high`/folio.
- El entorno `ProgramORetOnlyEnv._decisions()` reconstruye `state_rich_calendar(...)` cada vez que necesita decisiones. Durante un episodio, `step()` vuelve a solicitar la observación en cada acción no terminal. Esto está en `supply_chain/program_o_ret_env.py`, líneas 144–157 y 191–203.

## Inferencia operacional

El diseño actual puede ser muy intensivo en CPU/memoria/lectura cuando se ejecutan cinco entrenamientos en paralelo. La ausencia de checkpoints intermedios también impide distinguir progreso de learner de un job atascado sin intervenir en él.

## Decisión de esta fase

No se mata, reinicia ni duplica ningún proceso vivo: la política de custodia de la campaña prohíbe sustituir una corrida en curso por una variante no equivalente. Se espera a los cinco procesos. Sólo después de que existan checkpoints finales se ejecutan las evaluaciones de validation y se registra el resultado.

## Propuesta posterior — no aplicada

Para futuras campañas, congelar y probar una variante de entorno que cachee las decisiones por prefijo de acciones y que escriba checkpoints de progreso explícitos. No debe aplicarse retrospectivamente a estos jobs ni usarse para cambiar el estimando de la campaña actual.
