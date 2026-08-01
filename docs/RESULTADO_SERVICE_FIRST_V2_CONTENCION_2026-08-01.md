# Resultado — `v2` sellada, y el negativo **se extiende al endpoint sano**

**Artefacto:** `results/metric_audit/contention_service_first_v2/result.json` (sello
`9e4b7bceaac1c5a0…`, `NEGATIVE_EXTENDS_TO_THE_SOUND_ENDPOINT`) · **los seis falsadores PASAN** ·
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

**Corregido tras revisión externa — mi lectura de la orientación era un artefacto del bloque de
semillas.** Escribí que `ret_excel` elige «`0,9` bajo `R2r` y `0,1` bajo `R1r+R2r`». Comparando
los tres bloques:

| bloque | `R2r` | `R1r+R2r` |
|---|---:|---:|
| histórico `5.200.001+` | 0,1 | 0,9 |
| Cobb-Douglas `5.600.001+` | 0,1 | 0,1 |
| esta corrida `6.400.001+` | 0,9 | 0,1 |

**La orientación NO es estable.** Lo que sí lo es, y en los tres bloques y las dieciocho celdas:

> **`ret_excel` elige SIEMPRE un extremo —`0,1` o `0,9`, nunca `0,5`— y los endpoints sanos eligen
> SIEMPRE `0,5`.** De qué lado de la U cae el extremo es ruido; que caiga en un extremo, no.

La afirmación estructural —la métrica puede premiar el abandono— se conserva y sale reforzada.
La atribución por familia se retira.

## 2. El headroom

    H_regime sobre `worst_claimant_fill` = 0,000000   IC95 [0,000000, 0,000000]

**Exactamente cero, y el `argmax` no se mueve en ningún régimen.**

## 3. Por qué éste es el cierre y no una derrota más

La objeción evidente contra toda la campaña era: **«mediste con una métrica rota»**. Y era una
objeción legítima — el 31 de julio quedó medido que `ret_excel` prefiere el reparto que entrega el
**50 %** de las raciones sobre el que entrega el **80 %**.

Ahora hay **tres endpoints independientes** sobre el **mismo** barrido:

| endpoint | bloque de semillas | `argmax` | `H_regime` |
|---|---|---|---:|
| `ret_excel_risk_conditional` (censurada) | `5.200.001+` | un **extremo** en las seis | 1,527e-04 |
| **Cobb-Douglas** (su IJPR 2024) | `5.600.001+` | **0,5 en las seis** | **0,000000** |
| **`service_first_v2`** | `6.400.001+` | **0,5 en las seis** | **0,000000** |

**Esta tabla es CONVERGENCIA DESCRIPTIVA, no una comparación pareada.** Los tres endpoints
comparten diseño, regímenes y cadencia pero **corren sobre bloques de semillas distintos**, así
que son realizaciones aleatorias distintas. El `1,527e-04` es de
`ret_excel_risk_conditional` en la celda `FIFO_PARTIAL|fungible=False` del barrido histórico —
la etiqueta genérica «1,5e-4» que puse antes era imprecisa.

> **Los dos endpoints sanos coinciden entre sí y discrepan del roto. Y los dos dan exactamente
> cero.** El reparto equilibrado gana en todos los regímenes, y escalar R23 ×3 en frecuencia y ×2
> en impacto no lo mueve.

**No hay nada que decidir — y el alcance exacto de esa frase es:** repartos **constantes** de
`0,1` a `0,9`, bajo estos **seis** regímenes, con capacidad no fungible. **No evalúa políticas que
conmutan dentro del episodio**, que son otra clase que `H_regime` no acota. Dentro de ese alcance,
ya no se puede atribuir al instrumento: uno de los dos endpoints
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
semillas se promedian **componente a componente** y que las tuplas se comparan con orden de
tupla, **nunca colapsadas en un escalar**.

**`f6` corregido tras la revisión.** La primera versión comprobaba `len(COMPONENTS) == 4` — una
constante, que habría pasado igual si el ranking fuese una suma escalar. **Quinto falsador en tres
días que probaba un correlato.** Ahora hace dos cosas reales:

* **control positivo**: una reimplementación **independiente** de la comparación lexicográfica
  reproduce el `argmax` de producción en las seis celdas;
* **defecto inyectado**: rankear por la **suma escalar** de los cuatro componentes — que es en lo
  que consistiría colapsar la tupla — **cambia la respuesta** (`0,6` en vez de `0,5` en
  `R1r+R2r|base`). Es decir, el orden lexicográfico **está haciendo trabajo** y un colapso
  accidental sería detectable.

Y `f1` pasó de `any(...)` a exigir **las seis** celdas: discrepan **6/6**.
