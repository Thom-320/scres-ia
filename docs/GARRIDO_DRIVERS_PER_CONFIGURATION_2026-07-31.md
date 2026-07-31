# Los cuatro drivers de Garrido, configuración por configuración

**Artefacto:** `results/garrido_drivers_per_configuration/result.json`
(sello `6256f5d701fb745f…`) + `drivers.csv` · **Runner:**
`scripts/emit_garrido_drivers_per_configuration.py` · **Estado:**
`DEVELOPMENT_DRIVER_TABLE`, cinco falsadores pasan. **No es una afirmación sobre resiliencia**:
es la tabla de entrada para la Fig. 5 de Garrido, Pongutá & Adarme (2024).

## Qué se emitió

Las 90 configuraciones publicadas (Tablas 6.11–6.23), con sus `ρ` (`buffer_hours`, `shifts`) y,
por configuración, los cuatro drivers de su Fig. 4 como **descomposición aditiva exacta** de
`ret_excel`:

| driver | ecuación | rama del ledger |
|---|---|---|
| `Re(APj)` | Eq. 5.1 `Re^max × APj/LT`, `Re^max = 1` | `excel_autotomy` |
| `Re(RPj)` | Eq. 5.2 `Re × 1/RPj`, `Re = 0.5` | `excel_recovery` |
| `Re(DPj,RPj)` | Eq. 5.3 `Re^min × (DPj−RPj)/CTj`, `Re^min = 0` | `excel_risk_no_recovery` |
| `Re(FRt)` | Eq. 5.4 `1 − (Bt+Ut)/j` | `excel_fill_rate` |

Cada uno se reporta como `share × mean` = su **contribución** a la ReT media. Los cuatro más un
quinto término nuestro suman la ReT de la celda **exactamente** — `f2` lo verifica a 1e-12, no
lo asume.

## El hallazgo: sus hipótesis operan por drivers distintos según la familia

| familia | `Re(APj)` | `Re(RPj)` | `Re(DPj)` | `Re(FRt)` | ReT |
|---|---:|---:|---:|---:|---:|
| **R1r** (n=30) | 0 | **0,008624** | 0 | **0,000000** | 0,008624 |
| **R2r** (n=30) | 0 | 0,004318 | 0 | **0,563353** | 0,567671 |
| **R3** (n=30) | 0 | 0,000086 | 0 | **0,979881** | 0,979967 |

**Bajo R1r, el 100% de los pedidos cae en la rama de recuperación** y la rama sin riesgo **no
dispara nunca** (`Re(FRt)` es exactamente 0 en las 30 celdas). Es decir: para la familia R1r,
**su ReT ES el periodo de recuperación**, nada más. En R2r la mezcla es 60/40 y `Re(FRt)`
aporta el 99,2% del valor; en R3, el 99,99%.

Eso reencuadra sus H2/H3. Ambas reproducen en dirección —los buffers y los turnos suben ReT— y
ahora podemos decir **por qué canal**:

| | R1r | R2r | R3 |
|---|---|---|---|
| buffers (H2) | ReT 0,00789 → ~0,00924, **íntegramente vía `Re(RPj)`** | 0,488 → 0,575–0,741, **vía `Re(FRt)`** | — |
| turnos (H3) | 0,00789 → 0,00911, **vía `Re(RPj)`** | 0,488 → 0,681, **vía `Re(FRt)`** | 0,969 → 0,984, **vía `Re(FRt)`** |

Un escalar no dice esto. Es exactamente la clase de estructura que su neurona de la Fig. 5
tendría que aprender, y sale de la descomposición, no de un ajuste.

## Tres cosas que la tabla deja a la vista

**`Re(APj)` está muerto en las 90.** Nuestra constante de cumplimiento de 54 h contra `LT = 48`
hace inalcanzable la rama de autotomía. `f4` es un **trip-wire invertido**: falla el día en que
la autotomía empiece a dispararse —que es justo lo que el brazo de olas de flete busca— y obliga
a re-emitir los drivers antes de entrenar nada sobre ellos.

**`Re(DPj)` es cero medido, no ausente.** El caso `excel_risk_no_recovery` **sí ocurre**, en 4
configuraciones (share máx. 0,0004), y contribuye 0 — que es lo que su Eq. 5.3 exige con
`Re^min = 0`. `f3` exige que el caso ocurra precisamente para que el cero no sea silencio.

**El quinto término es nuestro, y vale 0.** Los pedidos que nuestro DES descarta (`unfulfilled`)
no existen en su cadena; se reportan aparte y **nunca** se pliegan en sus cuatro. Share máximo
sobre las 90: **0,0000**.

## La reproducción del 2026-07-29 quedó obsoleta, y por una sola razón

Ninguna de las 90 celdas reproduce hoy el valor sellado el 29 de julio. **Bisecado**: el
endpoint se mueve en **un único commit**, `1e4a69d` —la migración
`RET_RECOVERY_PERIOD_MODE` de `disruption` a `elapsed`, es decir la adopción del Algoritmo 2— y
es **bit a bit idéntico en todos los commits posteriores**. Cf1: `0,004070242240102` en
`435d6ed`, `0,007837224876422` desde `1e4a69d` hasta HEAD.

Eso confirma dos cosas: la reproducción es **fiel a su propio árbol**, y las ~25 opciones de
brazo añadidas el 30 y 31 de julio **no movieron el default** — como se afirmó entonces.

`f1` comprueba esa atribución de forma mecanicista: la migración cambia `RPj`, así que **solo**
puede moverse una celda cuyos pedidos lleguen a la rama de recuperación. **Con una limitación
que el artefacto declara**: las 90 tienen pedidos en esa rama, así que el test solo podría
cazar una celda que se moviera *sin* ellos, nunca una que no se moviera teniéndolos.

## Lo que sigue

1. **Cerrar `Re(APj)`** con el brazo de olas de flete (`min CTj = 48,0`). Es el único driver
   suyo que no existe en nuestra tabla, y `f4` ya está armado para exigir la re-emisión.
2. **Fig. 5** sobre `drivers.csv`: `(ρ, drivers) → ReT` y la pregunta de activación
   *«¿ReT en `x` > ReT en `x−1`?»*, backprop contra KAN.
3. Recordar al leerla: con `Re(APj) ≡ 0` y `Re(DPj) ≡ 0`, hoy la neurona tendría **dos entradas
   vivas**, no cuatro — y en R1r, **una sola**.
