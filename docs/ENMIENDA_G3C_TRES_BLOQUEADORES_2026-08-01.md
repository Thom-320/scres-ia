# Enmienda 2 a G3c — tres bloqueadores antes de que pueda ejecutarse

**Enmienda prospectiva** a `docs/PREREGISTRO_G3C_ACOPLAMIENTO_TEMPORAL_2026-08-01.md`. Ningún dato
se ha visto contra ese contrato: sigue `DESIGN_ONLY_NOT_AUTHORIZED`, y ahora además
**`BLOCKED_PENDING_THREE_AMENDMENTS`**.

## Bloqueador 1 — el contrato se contradice: dice «una sola física» y define dos

Escribí *«se añade **una** familia de acoplamiento»* y acto seguido tabulé **dos mecanismos
distintos**: permanencia mínima y coste de cambio. Son cosas diferentes —uno prohíbe cambiar,
el otro lo cobra— y un brazo que los mezcle no permite atribuir el resultado.

**Enmienda:** factorial explícito, con el nulo dentro.

| | `switch_cost = 0` | `switch_cost > 0` |
|---|---|---|
| **`min_dwell = 1`** | **nulo** (modelo actual) | sólo coste |
| **`min_dwell > 1`** | sólo permanencia | ambos |

Y **«dos niveles calibrados» no es una especificación**. Deben fijarse aquí, con su unidad y su
procedencia, antes de correr — o el contrato deja la puerta abierta a elegir el nivel que
funcione.

## Bloqueador 2 — la justificación de `δ = 0,25` es un error de razonamiento, y es mío

Escribí que el margen de `lost_orders` es defendible por ser **«4× la granularidad Monte Carlo»**
que produjo el halt espurio (0,0625 = 1/16).

**Eso está mal, y el error es de tipo, no de número.** La resolución Monte Carlo informa la
**potencia** —qué diferencias puedo *detectar*— y no dice absolutamente nada sobre qué daño es
**operacionalmente aceptable**. Justificar un margen de no inferioridad por el tamaño de la
muestra es dejar que el instrumento defina el criterio: con 64 semillas el «margen justificado»
se encogería cuatro veces sin que la operación haya cambiado en nada.

**Enmienda:** `δ` para cada guardarraíl debe tener **justificación operacional independiente**,
declarada en sus propias unidades y sin referencia al `n`. La granularidad Monte Carlo se reporta
**por separado**, como límite de detección, nunca como fundamento del margen. Mientras `δ` no
tenga esa justificación, **G3c no corre** — y esto vale también para el `δ` que G3-obs ya heredó,
cuyos guardarraíles pasaron pero **con un margen mal fundamentado**.

## Bloqueador 3 — la identidad del brazo nulo es una afirmación, no un hecho

El contrato *afirma* que `min_dwell = 1`, `switch_cost = 0` reproducen el modelo actual. **Nadie
lo ha comprobado.**

**Enmienda:** `f1` debe ejecutarse como **verificación**, comparando un **payload científico
canónico** —eventos, órdenes, acciones, ledgers, métricas— y **no** el `self_sha256`, que incluye
`created_at` y `calibration_provenance` (`supply_chain/arm_runner.py:158`) y por tanto **debe**
cambiar cuando cambia el código. Sin ese hash canónico implementado, el falsador no es ejecutable
y el contrato no puede abrirse.

## Lo que esto deja en pie

Los márgenes de G3c **siguen siendo la plantilla correcta** por compartir endpoint y guardarraíles
con G3-obs — pero **no son universales ni están «firmados» por aparecer en Markdown**. Un margen
está firmado cuando tiene justificación operacional, potencia calculada y un contrato que lo
congele antes de ver datos. Hoy tiene lo segundo y le falta lo primero.

## Estado canónico tras esta enmienda

```
G3-obs : STOP_G3_OBS_UNDERPOWERED        (instrumento limpio, primario no interpretable)
G3c    : DESIGN_ONLY + BLOCKED_PENDING_THREE_AMENDMENTS
G3a    : diseño válido, prospectivo, NO cancelado
H3'    : ARTIFACTS_PRESENT_MERGE_PENDING
```

**Orden de trabajo real, sin ciencia nueva:** corregir portabilidad, reconciliar H3′, y **someter
Submission A** — que es el único desbloqueo que existe para el resto.

## Nota de portabilidad, porque afirmé haberlo arreglado y no era cierto

Dije que el defecto de portabilidad quedaba reparado. **Reparé un archivo; quedaban decenas.**
Ahora también `docs/PREREGISTRO_G3_ASIMETRIA_V2_2026-08-01.md`, que había commiteado con rutas
absolutas.

Los ofensores restantes son de **dos clases distintas** y no admiten la misma solución:

* **artefactos fechados** (`results/**`, `research/paper2_exhaustive_search/**`,
  `docs/track_b_q1_stats_*`) — **no se editan en sitio**; el test debe excluirlos o el defecto se
  declara residual;
* **scripts** (`scripts/audit_garrido_wrap_sources.py`, ~20 `watch_*.sh`) — aquí sí se corrige,
  salvo un caso que **construye rutas relativas a `Path.home()`** apuntando a una cuenta de
  nube personal: no es literalmente una ruta de usuario, y se resuelve sacando la cuenta del
  código a la variable de entorno `SCRES_IA_DRIVE_ROOT`.
