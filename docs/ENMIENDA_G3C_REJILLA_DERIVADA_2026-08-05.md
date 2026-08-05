# Enmienda G3c — la rejilla de `min_dwell`, ahora derivada de una medición

**Estado:** `LEVELS_REDERIVED_BURNED_PREFLIGHT_ONLY`
**Sustituye:** la rejilla `{1, 3, 7}` de `contracts/g3c_temporal_coupling_v2.json` §mechanism.levels
y de `docs/ENMIENDA_G3C_PREFLIGHT_BURNED_2026-08-05.md` §2.
**Evidencia:** `results/headroom/g3c_dwell_inertia/result.json`
(`scripts/measure_g3c_dwell_inertia.py`, bloque quemado `5.200.001–08`, réplica declarada).

## 1. Por qué la rejilla anterior no servía

Se eligió antes de medir nada, y el preflight quemado la rompió: `f2` falló porque
`cssu_min_dwell_days = 3` **no retiene ni una sola acción**. La caracterización posterior lo
confirma en las dos presiones y los dos regímenes, y añade lo que faltaba —**cuánto** ata cada
nivel:

| dwell | tapes que retienen (miope) | conmutaciones realizadas | supresión vs nulo |
|---:|---:|---:|---:|
| 1 (nulo) | 0/8 | 43,8 | 0,0 % |
| 3 | **0/8** | 43,8 | **0,0 %** |
| 4 | 8/8 | 42,5 | 2,9 % |
| 5 | 8/8 | 40,6 | 7,1 % |
| **6** | 8/8 | 39,1 | **10,6 %** |
| **7** | 8/8 | 37,8 | **13,7 %** |
| **14** | 8/8 | 24,0 | **45,1 %** |
| 21 | 8/8 | 16,6 | 62,0 % |

Dos umbrales distintos, y confundirlos es lo que produjo la rejilla muerta:

* **atar** empieza en **4 días** bajo la política miope (5 bajo presión máxima);
* **restringir** empieza en **6 días** — antes de eso el dwell *registra* la restricción sin
  *imponerla*, y un nivel así añade una fila a la rejilla sin añadir un problema de decisión.

## 2. La regla, fijada de antemano para la próxima vez

> Un **nivel de tratamiento** de `min_dwell` es admisible sólo si (a) retiene acciones en **todas**
> las cintas de **todos** los regímenes y (b) **suprime al menos un 10 % de las conmutaciones
> realizadas** frente al nulo. Los niveles deben estar separados por un factor ≥ 2.

La condición (b) es la que no existía, y es la que descarta 4 y 5.

## 3. La rejilla nueva

```text
cssu_min_dwell_days ∈ {1, 7, 14}
1  = nulo de regresión legacy  (0 retenciones, verificado)
7  = tratamiento débil          (13,7 % de supresión)
14 = tratamiento fuerte         (45,1 % de supresión)
```

**Por qué 7 y no 6**, que es el umbral material medido: 6 apenas cruza el criterio (10,6 %), y **7
ya se corrió** en el preflight detenido, de modo que la celda es directamente comparable con lo ya
medido en vez de empezar de cero. 14 dobla 7 y triplica la separación. Es una decisión nuestra, y
queda declarada como tal.

## 4. Lo que esta enmienda NO cambia

Primario `worst_claimant_fill`, SESOI `+0,010`, márgenes firmados, unidad de resampling = la
semilla, corrección simultánea Bonferroni sobre las **6** celdas (siguen siendo 3 niveles × 2
regímenes), presupuesto **96** semillas frescas y la regla `STOP_G3C_UNDERPOWERED`. **Ninguna
semilla nueva**, ningún learner, ninguna adjudicación.

## 5. Runner canónico

Se congela **uno solo**: `scripts/run_g3c_temporal_coupling.py`, con el rol `BURNED_PREFLIGHT` por
defecto. El runner paralelo del mismo día queda descartado; su artefacto
`results/headroom/g3c_preflight_burned/result.json` conserva su hallazgo —`f2` falla, la rejilla
tiene un nivel muerto— pero **su tabla de contrastes (histéresis vs miope) no es reproducible con
el runner canónico**, que implementa `myopic / placebo / wrong_claimant`, y por tanto **no se cita
como resultado de G3c**.

Corregido de paso un defecto real del canónico: el brazo de reclamante equivocado calculaba su
objetivo como `1.0 - 0.9 = 0.09999999999999998`, que **no es un nivel registrado** y, contra una
comparación exacta a `1e-9`, habría pedido un reparto no declarado en cada paso.
