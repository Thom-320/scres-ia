# Resultado — ambas rebanadas H3′ re-selladas: los tres falsadores de fusión ya se pueden decidir

**Artefactos**
`results/garrido_meta_learner_h3power_h3_contract_local_v2/result.json` (sello `e768e0f69f187272…`, 90 réplicas)
`results/garrido_meta_learner_h3power_h3_contract_vps_v2/result.json` (30 réplicas)
Ambos bajo el contrato **H3′** `docs/PREREGISTRO_H3_POTENCIA_2026-08-01.md` (`576d02b5…`),
`schema_version = garrido_h3_source_audit_v2`, **ninguna semilla nueva**.

**Supersede** `docs/RESULTADO_AUDITORIA_FUENTE_H3_2026-08-01.md`, que quedó atado al artefacto
replay antiguo y a un solo lado del experimento.

## 1. Cada rebanada reproduce su original, exactamente

| rebanada | vs su artefacto original |
|---|---|
| local, 90 réplicas | **14 de 14 cantidades idénticas, 0 difieren** |
| VPS, 30 réplicas | **14 de 14 cantidades idénticas, 0 difieren** |

## 2. Los tres falsadores de fusión del contrato H3′, comprobados

| falsador | resultado |
|---|---|
| `f_merge_seeds_are_disjoint` | **cumple** — `6.000.001–90` y `6.000.091–120`, sin solape (90 + 30) |
| `f_merge_contexts_and_budget_match` | **cumple** — `budget`, `factors`, `contexts`, `metric` y `n_configurations` idénticos |
| `f_merge_source_is_identical` | **cumple** — manifiesto `v2`: los **siete hashes de módulo** y el hash del entry script coinciden, y el contrato es el mismo |

**Es la primera vez que este falsador puede evaluarse en vez de declararse pendiente.** Antes no
existía manifiesto y la comparación era indecidible.

## 3. Y exactamente qué establece eso, que es menos de lo que parece

Las dos rebanadas v2 se ejecutaron sobre **el mismo checkout**, así que la coincidencia de sus
manifiestos es **casi tautológica**. Lo que NO establece:

> **La identidad de fuente entre el snapshot ORIGINAL del VPS y el local sigue sin demostrarse.**
> Ese snapshot tenía un `supply_chain.py` distinto y carecía de `service_first_metric.py`, y
> ningún manifiesto posterior puede reconstruirlo.

Lo que **sí** establece, y es útil: **ambas rebanadas, bajo una fuente verificada e idéntica y un
solo contrato, reproducen sus artefactos originales al dígito y son mutuamente fusionables** como
dos bloques de semillas disjuntas.

**Y el límite que no se mueve:** `f6 = DECLARED_REPLAY / NO APLICA` en las dos. **Son semillas
quemadas.** Por tanto esto es **evidencia de desarrollo con 120 réplicas**, nunca una
**confirmación H3′ virgen**, que exigiría un bloque independiente sin abrir — y abrir raíces está
prohibido por `authority_ladder_v1` hasta el recibo de Submission A.

## 4. El efecto Alzheimer, ahora sellado en ambos lados

| rebanada | `reset − memoria` | IC95 |
|---|---:|---|
| local, n = 90 | **+7,2704** | [6,7519 · 7,7760] |
| VPS, n = 30 | **+7,6111** | [6,6110 · 8,6556] |

Los intervalos se solapan ampliamente; las dos rebanadas son consistentes. Ambas sellan además
`memory_vs_ofat` (+5,04 y +4,92) y `memory_vs_random` (+12,19 y +11,69) — **estimandos distintos
del efecto Alzheimer**, y conviene no confundirlos: una revisión externa los tomó por la misma
cifra mal escrita, y comprobarlo fue lo que destapó que el estimando titular **no se estaba
sellando**.

## 5. Lo que NO se hace aquí, deliberadamente

**No se calcula la estimación fusionada de 120.** Verificar que los tres falsadores se cumplen es
requisito de la fusión, no la fusión. Combinar los dos vectores con un script ad hoc sería
exactamente lo que la disciplina del proyecto prohíbe —*medir por la tubería, nunca con un script
improvisado*—, y ya nos costó un defecto fabricado una vez. La adjudicación de la fusión es un
paso contratado aparte.

## 6. Estado canónico

```
reproducción conductual, ambas rebanadas : PASS (14/14 cada una)
falsadores de fusión (3/3)               : COMPROBADOS sobre las re-ejecuciones v2
identidad de fuente del snapshot VPS     : NO DEMOSTRADA, y ya no es demostrable
fusión de 120 calculada                  : NO
H3' como confirmación virgen             : NO — semillas quemadas
precio del efecto Alzheimer al manuscrito: DESARROLLO, pendiente de adjudicación de fusión
```
