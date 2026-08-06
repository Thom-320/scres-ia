# Preregistro — ¿alguna actitud ante el riesgo citable cae en el conjunto que califica?

**Escrito y commiteado ANTES de correr.** Runner: `scripts/run_citable_risk_attitudes_v1.py`.
Predecesor: `results/monotone_transform_family_v4/result.json`
(`A_MONOTONE_RESCALING_SURVIVES_ALL_THREE`, 9/9).
Semillas: bloque quemado `garrido_q2_des288`, réplica declarada. **Ninguna nueva.**

## 1. La pregunta, y por qué se puede contestar casi sin citas

v4 dejó abierto: la curvatura **es** una actitud ante el riesgo, así que si alguna **citable** cae
dentro del conjunto que califica, la curvatura la declara la literatura y no nosotros.

Y hay un hecho estructural que hace la prueba casi independiente de qué números exactos citemos:

> **`power(γ)` con `γ > 1` es convexa. Una utilidad convexa sobre resultados es amante del riesgo.**
> Las actitudes de la literatura de gestión de riesgo en cadenas de suministro son **aversas** —
> cóncavas — y toda utilidad cóncava cae en `γ ≤ 1`.

El conjunto que calificó en v4 vive en `γ ≈ 3–32`. Si `H(γ)` es **monótona creciente**, entonces
todo el lado cóncavo está **por debajo** de la identidad, y la identidad ya está bajo el umbral. La
vía se cerraría **por estructura**, no por una tabla de coeficientes discutible.

**Eso es exactamente lo que se mide aquí, y puede salir al revés:** si `H(γ)` no fuera monótona, o
si alguna cóncava superara a la identidad, el argumento se cae.

## 2. Qué se mide

**Brazo A — utilidades puntuales, que sí mapean exactamente a la familia.**

| familia | forma | parámetros | curvatura |
|---|---|---|---|
| neutral al riesgo | `u(x) = x` | — | `γ = 1` (identidad = **la de Garrido**) |
| CRRA / isoelástica | `u(x) = x^(1−η)/(1−η)`, `ln x` si `η=1` | `η ∈ {0,25 · 0,5 · 1 · 2 · 3 · 5 · 10}` | **cóncava** |
| CARA / exponencial | `u(x) = (1 − e^{−a x})/a` | `a ∈ {0,5 · 1 · 2 · 5 · 10}` | **cóncava** |
| amante del riesgo (control) | `u(x) = x^γ` | `γ ∈ {2 · 5 · 10 · 20 · 30}` | **convexa** |

El brazo amante del riesgo **no es una propuesta**: es el control que demuestra que esta prueba
**podría** haber devuelto «sí».

**Brazo B — CVaR sobre regímenes, declarado como lo que es.** `CVaR_α` **no** es una
transformación puntual: sustituye `mean_r` por la media de la cola inferior. Cambia el
**estimador**, no la métrica, y por eso se reporta **aparte y etiquetado**, nunca mezclado con el
brazo A. `α ∈ {0,90 · 0,95 · 0,99}`.

## 3. Reglas de lectura, fijadas antes de mirar

`GATE = 0,05`. Califica igual que en v4: `LCB95 ≥ 0,05` **y** Holm sobre la familia declarada
**y** `resolubles ≥ 0,90 ×` la identidad.

* alguna **aversa citable** califica → **`A_CITABLE_RISK_ATTITUDE_REACHES_THE_BAR`**. La curvatura
  quedaría declarada por la literatura y la adopción sería posible **con esa cita**.
* ninguna aversa califica **y** el control amante sí →
  **`ONLY_RISK_SEEKING_CURVATURE_REACHES_THE_BAR`**. La vía se cierra **por número**: el headroom
  exige una actitud que nadie defiende en esta literatura.
* ninguna califica, control incluido → **`INSTRUMENT_DETECTS_NOTHING`**, y la corrida no dice nada
  sobre las actitudes porque no se demostró que pudiera detectar ninguna.

## 4. Falsadores

| falsador | por qué puede fallar |
|---|---|
| `f1_risk_neutral_reproduces_the_sealed_scalar` | `u(x)=x` debe dar el `scalar_h_regime` sellado a 1e-9. **Ancla externa** |
| `f2_the_control_qualifies` | al menos una convexa del control debe calificar. **Si ninguna lo hace, la prueba no puede detectar nada y no hay veredicto sobre las citables** |
| `f3_concave_attitudes_land_below_the_identity` | toda cóncava debe dar `H ≤ H(identidad)`. **Falla si alguna la supera**, y entonces el argumento estructural del §1 es falso y hay que retirarlo |
| `f4_H_is_monotone_in_curvature` | `H(γ)` debe crecer con `γ` sobre toda la rejilla `0,001–1000`. **Falla si no es monótona**, y entonces no se puede razonar por lados |
| `f5_cvar_is_reported_separately` | ninguna fila de CVaR puede entrar en el veredicto del brazo A |
| `f6_no_fresh_seeds` | custodia central, réplica declarada |

## 5. Lo que esta corrida NO puede establecer, dicho antes

**Los coeficientes son rangos estándar, no citas verificadas.** No adopto ninguno como «el valor de
la literatura». El peso del resultado recae en `f3` y `f4` —que son hechos medidos sobre nuestra
superficie— y no en qué `η` exacto use tal artículo. **Antes del manuscrito, cada coeficiente que se
cite debe verificarse contra su fuente**, y esta enmienda lo deja escrito para que nadie lo dé por
hecho.

**Alcance:** desarrollo sobre tapes quemados. No abre semillas, no adjudica, **no adopta ninguna
transformación** y no cambia la primaria del contrato.
