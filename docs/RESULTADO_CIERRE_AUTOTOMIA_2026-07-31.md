# Resultado — `Re(APj)` se cierra, y cuesta más de lo que el contrato admite

**Artefacto:** `results/metric_audit/autotomy_closure_arms_v1/result.json` (sello
`3b9e1c5453813c24…`) · **Contrato:** `docs/PREREGISTRO_CIERRE_AUTOTOMIA_2026-07-31.md` ·
**Referencia:** `fidelity_reference_v4` · 12 raíces vírgenes, ambas familias, **los cinco
falsadores pasan**.

## La respuesta corta

**El brazo de olas de flete NO cierra la autotomía. Olas + `δ` + predicado de banda sí la
cierra —`d_k` de 12,40 a 1,26— y aun así NO se adopta**, porque `ret_mean` se degrada 0,95
errores estándar combinados, por encima del `EPSILON = 0,5` que el contrato declaró antes de
correr. Predije la no-adopción en §3; que se cumpla la hace informativa, no cómoda.

| brazo | `autotomy_share` R1r | `d_k` | R2r | `d_k` | `ret_mean` R1r `d_k` |
|---|---:|---:|---:|---:|---:|
| **A** constante 54, `le` (statu quo) | 0,000000 | 12,40 | 0,000000 | 4,56 | 2,18 |
| **F** olas, `le` | 0,655341 | **62,61** | 0,141279 | 19,41 | **50,70** |
| **FD** olas + `δ`, `le` | 0,000000 | 12,40 | 0,000000 | 4,56 | 2,89 |
| **FDB** olas + `δ`, banda 0,05 h | **0,003122** | **1,26** | 0,000000 | 4,56 | 3,13 |
| **Garrido (v4)** | 0,004334 | — | 0,000637 | — | — |

## El mecanismo, medido en ambos lados con la misma regla

`f2` era la afirmación de mecanismo, y se sostiene:

| | filas | en la banda `CTj ∈ [48,00 · 48,05]` | share | `min CTj` |
|---|---:|---:|---:|---:|
| **Garrido R1r** | 26.087 | 114 (las 114 con autotomía) | **0,437%** | 48,00744 |
| **Garrido R2r** | 21.693 | 36 (14 con autotomía) | 0,166% | 48,00744 |
| **nuestro A** | 5.087 | 0 | 0,000% | 54,000 |
| **nuestro F** | 5.087 | 3.181 | **62,53%** | 48,000 |
| **nuestro FDB** | 5.087 | 16 | **0,3145%** | 48,0103 |

El diagnóstico previo queda confirmado: **el problema nunca fue el desfase del suelo, sino su
incidencia**. La rejilla de olas deja el 62,5% de las órdenes exactamente en `CTj = 48,0` —
**143× su 0,437%**— y con el predicado `CTj ≤ LTj` todas se vuelven autotomía. Añadir
`δ ~ U(0,8)` vuelve el suelo raro (0,3145% contra su 0,437%) y el predicado de banda deja que
dispare.

## Las tres predicciones se cumplieron

1. **`F` sobredispara.** 0,655 contra 0,004334 — **151×** — y arrastra `ret_mean` a `d_k`
   50,70. No es un cierre.
2. **`FD` vuelve a cero, exactamente.** Con `δ > 0` casi seguro, `CTj = 48 + δ > LT`, y `le` no
   puede disparar. Añadir `δ` sin tocar el predicado **no cierra nada** — declarado en §3.3.
3. **`FDB` se acerca.** Predije ≈0,625%; salió **0,3122%**, la mitad — y la mitad es la
   explicación: solo las órdenes **no demoradas** (~50%) están en `48 + δ`; las demoradas viven
   en la rejilla de 24 h, lejos de la banda. La predicción erró por un factor que el propio
   mecanismo explica.

## Lo que la regla de aceptación dice, literalmente

Ningún brazo califica. `FDB` falla por dos vías, y conviene separarlas:

* **`ret_mean` R1r empeora +0,95** (`d_k` 2,18 → 3,13), por encima de `EPSILON`. Es el veto del
  momento protegido. **Ese es el hallazgo real:** reproducir su autotomía **degrada su propia
  métrica**.
* **«mejora `autotomy_share`» exige AMBAS familias** y R2r no se mueve (4,56 → 4,56). Mi regla
  pedía `all(...)`; R1r mejora **10×** y R2r queda igual, así que el criterio dice «no mejora».
  Lo aplico como lo escribí, y lo digo con precisión en vez de reinterpretarlo ahora.

`FDB` está en el **conjunto no dominado de las dos familias** (R1r `{A, FDB}`, R2r
`{FDB, FD}`): gana en autotomía, pierde en `ret_mean`. El contrato dice que **el conjunto no
dominado ES la salida** — no se elige aquí.

## Un efecto que no predije

En **R2r**, `δ` baja `ret_above_one_share` de `d_k` **3,99 a 0,30**. No lo anticipé en §3 y por
eso lo marco: es una mejora sustancial en el momento de la cola que ni `δ` ni la banda estaban
puestos a arreglar. Merece su propia lectura antes de que nadie la use como argumento.

## Qué significa para la Fig. 5 de Garrido

La tabla de drivers embarcada **sigue con `Re(APj) ≡ 0`**, porque `FDB` no se adopta. Pero ahora
se sabe el precio exacto de encenderlo: **1 parámetro** (la banda de 0,05 h, leída de **sus**
filas, nunca ajustada a las nuestras) y **0,95 SE combinados de `ret_mean`**.

Eso convierte una ausencia en una **frontera medida**, que es lo que se puede llevar al paper:
la autotomía de su modelo es reproducible, y reproducirla empeora la métrica que la contiene.

## Estado

`DEVELOPMENT_PREREGISTERED_AUTOTOMY_CLOSURE`. Nada adoptado, nada cambiado en los defaults. El
trip-wire `f4` de la tabla de drivers **sigue armado**: si algún día se adopta `FDB`, esa tabla
falla y obliga a re-emitirse antes de que se entrene nada encima.
