# Resultado — `v2` sellada, y el negativo **se extiende al endpoint sano**

**Artefacto:** `results/metric_audit/contention_service_first_v2/result.json` (sello
`01b755bad9bd405e…`, `NEGATIVE_EXTENDS_TO_THE_SOUND_ENDPOINT`) · **los seis falsadores PASAN** ·
preregistro `docs/PREREGISTRO_METRICA_SERVICE_FIRST_V2_2026-08-01.md`, commiteado antes de correr.

**Con esto `service_first_resilience_v2` deja de ser prospectiva:** tiene contrato propio y una
corrida sellada que la usa como endpoint.

## 1. El `argmax`, por régimen

| régimen | **v2** | `ret_excel` | fill rate |
|---|---:|---:|---:|
| `R2r` base | **0,5** | 0,9 | 0,5 |
| `R2r` freq ×3 | **0,5** | 0,9 | 0,5 |
| `R2r` freq ×3 imp ×2 | **0,5** | 0,9 | 0,5 |
| `R1r+R2r` base | **0,5** | 0,1 | 0,5 |
| `R1r+R2r` freq ×3 | **0,5** | 0,1 | 0,5 |
| `R1r+R2r` freq ×3 imp ×2 | **0,5** | 0,1 | 0,5 |

**`v2` coincide con el servicio en las seis celdas y discrepa de `ret_excel` en las seis.** El
endpoint hace lo que se contrató que hiciera.

**Y aparece algo que no buscaba:** `ret_excel` no sólo elige el reparto equivocado — elige
**repartos equivocados DISTINTOS según la familia de riesgo**: `0,9` bajo `R2r`, `0,1` bajo
`R1r+R2r`. Los dos son extremos, los dos abandonan una CSSU, pero **por lados opuestos**. Una
métrica que recomienda estrangular al reclamante A o al B según qué riesgos estén activos no está
midiendo resiliencia.

## 2. El headroom

    H_regime sobre `worst_claimant_fill` = 0,000000   IC95 [0,000000, 0,000000]

**Exactamente cero, y el `argmax` no se mueve en ningún régimen.**

## 3. Por qué éste es el cierre y no una derrota más

La objeción evidente contra toda la campaña era: **«mediste con una métrica rota»**. Y era una
objeción legítima — el 31 de julio quedó medido que `ret_excel` prefiere el reparto que entrega el
**50 %** de las raciones sobre el que entrega el **80 %**.

Ahora hay **tres endpoints independientes** sobre el **mismo** barrido:

| endpoint | `argmax` | `H_regime` |
|---|---|---:|
| `ret_excel` (censurada, explotable) | 0,9 / 0,1 según familia | 1,5e-04 |
| **Cobb-Douglas** (de su IJPR 2024) | **0,5 en las seis** | **0,000000** |
| **`service_first_v2`** (construida para no premiar el abandono) | **0,5 en las seis** | **0,000000** |

> **Los dos endpoints sanos coinciden entre sí y discrepan del roto. Y los dos dan exactamente
> cero.** El reparto equilibrado gana en todos los regímenes, y escalar R23 ×3 en frecuencia y ×2
> en impacto no lo mueve.

**No hay nada que decidir.** Eso ya no se puede atribuir al instrumento: uno de los dos endpoints
sanos viene del propio Garrido y el otro se construyó explícitamente contra el defecto que
encontramos.

## 4. Lo que sigue sin autorizar

* **Entrenar.** `v2` es un **endpoint normativo estipulado** —una decisión de dominio— y **no es
  evidencia** de que abandonar sea malo. Usarla para «redescubrir» el defecto de `ret_excel` sería
  circular, y sigue escrito así en la auditoría.
* **`H_regime` acota CONSTANTES.** Una regla que conmuta dentro del episodio es otra clase, y esa
  corrección está en `4d7a173`.
* **Fases 1B (presupuesto de expedición) y 1C (autotomía)** atacan otras causas del mapa, no ésta.

## 5. Limitación declarada antes de correr, y que se mantiene

Una clave lexicográfica **no admite media**, así que `H_regime` no está definida sobre `v2`
completa. Se reportan dos estimandos separados —`argmax` bajo la clave completa, y `H_regime`
sobre el componente **líder** solo— en vez de inventar una agregación. `f6` verifica que las
semillas se promedian **componente a componente** y que las tuplas resultantes se comparan con
orden de tupla, **nunca colapsadas en un escalar**.
