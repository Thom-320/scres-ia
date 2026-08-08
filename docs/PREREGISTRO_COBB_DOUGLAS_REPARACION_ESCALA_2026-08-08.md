# Preregistro — reparar Cobb-Douglas por sus dos defectos de escala

**Escrito y commiteado ANTES de medir ninguna variante.** Runner:
`scripts/run_cobb_douglas_scale_repair_v1.py`. Caché: `results/cobb_douglas_component_headroom*/
aggregates.json`, bloque quemado `5.300.001+`, réplica declarada. **Ninguna semilla nueva.**

Sucede a `docs/PREREGISTRO_FAMILIA_DERIVACIONES_METRICA_2026-08-06.md` (K = 162, de las cuales 158
quedaron agrupadas). **No lo reemplaza: añade dos ejes que aquella familia no barrió**, y paga la
multiplicidad sobre el conjunto agrupado.

## 0. Qué NO es esto

No es «probar hasta que salga». La familia anterior ya midió 158 variantes y su veredicto —
`ONLY_MISSCALED_VARIANTS_REACH_THE_BAR`, cuatro cruzan y **las cuatro violan su propia cota de
share** — sigue en pie y entra al manuscrito con o sin este preregistro.

Lo que se hace aquí es distinto y más estrecho: **se han medido dos defectos concretos del índice
publicado, ambos consecuencia de aplicar sus reglas a un sistema de otra escala, y se repara cada
uno con la mínima intervención que lo corrige.** Los dos son verificables antes de correr nada, y
los dos están abajo con su número.

## 1. Defecto A — bajo `c = 1`, κ̇ es un duplicado de ζ con el signo cambiado

Su supuesto (6) de §3.1 fija los siete parámetros de coste a 1 «para mantener κ̇ aislado de la
influencia de los parámetros de coste». Eso es inofensivo cuando las variables de decisión están
en escalas comparables. En su modelo APP lo están: ζ^max ≈ 3.612 contra una producción de ~800
por periodo, un factor de ~4,5.

**En nuestro entorno el factor es 181.** Medido sobre las 10.368 celdas de la caché
(`results/cobb_douglas_component_headroom/aggregates.json`):

| componente de κ bajo c = 1 | cuota media | rango |
|---|---|---|
| inventario `c_i·I` | **85,650 %** | 71,42 – 99,28 % |
| backorders `c_b·B` | 13,734 % | 0,02 – 28,05 % |
| producción `c_p·P` | 0,440 % | 0,25 – 0,66 % |
| capacidad ociosa `c_u·U` | 0,176 % | 0,08 – 0,35 % |
| contrataciones / despidos / horas extra | 0,000 % | estructuralmente ausentes |

y `corr(κ, ζ + ε) = 0,999993`. **El término de coste no es un término de coste: es ζ + ε otra vez.**

Con los exponentes derivados de nuestros máximos (`a_ζ = 0,014274`, `n_κ = 0,446334`) y una cuota
de ζ en κ de 0,857, el coeficiente efectivo sobre `ln ζ` es

```
+0,014274  −  0,446334 × 0,857  =  −0,368
```

**El inventario entra en el índice de resiliencia con peso efectivo −0,368 cuando su construcción
lo pone en +0,014**: signo invertido y magnitud 26 veces mayor. El índice, en nuestra escala, mide
aproximadamente «menos inventario».

**Por qué repararlo es legítimo y no es ajustar a la conclusión.** El supuesto (6) es *suyo* y su
propia §5 lo relaja — «if the assumption (6) were relaxed [...] what would happen to the overall
measure of resilience?» — y encuentra `c_i`, `c_p` y `c_b` como los sensibles. Nosotros no estamos
inventando un grado de libertad: estamos usando el que él abrió.

**Por qué la rejilla económica que ya tenemos no lo cubre.**
`results/cobb_douglas/economic_sensitivity_v2` barre ×0,5, ×2 y ×5 sobre un parámetro a la vez.
Con `c_i = 0,5` la cuota del inventario baja de 85,7 % a 62,9 %: **sigue dominando**. Romper la
dominación exige ~×1/181, dos órdenes de magnitud fuera de esa rejilla. Aquel artefacto responde
«¿cambia el ranking con los precios relativos?» y su respuesta —no— sigue siendo válida; **no
responde a si κ̇ es un término independiente**, que es lo de aquí.

## 2. Defecto B — la regla `0,20/ln(x_max)` da más peso a la variable con menos rango

Su regla, citada: *«each function argument was equated to 1/5. For example, in the case of ζ,
ζ^max ≈ 3.612, from which a·Ln3.612 = 0,20, resulting in a = 0,024»*. Invirtiéndola,
`x^max = exp(0,20/aₓ)`:

| variable | exponente publicado | x^max implícito | peso relativo a ζ |
|---|---|---|---|
| ζ inventario | 0,0240 | 4.160 | 1,00× |
| ε backorders | 0,0260 | 2.191 | 1,08× |
| φ capacidad ociosa | 0,0400 | 148 | 1,67× |
| τ tiempo de cumplimiento | 0,0600 | 28,0 | 2,50× |
| κ̇ coste | **0,1771** | **3,1** | **7,38×** |

**El exponente es inverso al rango dinámico**, y κ̇ domina el índice publicado 7,38 veces a ζ por
construcción de la normalización — no por una decisión de modelado. En su paper el efecto es
moderado. **En el nuestro es catastrófico**, porque τ y κ̇ son razones acotadas cerca de 1:

| variable | x^max nuestro | exponente derivado | peso relativo a ζ |
|---|---|---|---|
| ζ | 1.216.292 | 0,014274 | 1,00× |
| ε | 136.716 | 0,016912 | 1,18× |
| φ | 2.779 | 0,025221 | 1,77× |
| τ | 1,343 | **0,678102** | **47,5×** |
| κ̇ | 1,565 | **0,446334** | **31,3×** |

Las dos razones se llevan **1,124 de los 1,181** de masa exponencial total: las tres cantidades
físicas —inventario, backorders, capacidad ociosa— se reparten el **4,7 %** del peso. Y τ está
**muerta en el 17,5 %** de las celdas y su exponente está mal condicionado por el propio criterio
del módulo (`WELL_CONDITIONED_LOG_MAX`, amplificación 3,39).

**La reparación, y por qué es su regla y no otra métrica.** Su intención declarada es que cada
argumento aporte 1/5. Pero lo que un término aporta a *discriminar entre configuraciones* no es su
valor en el máximo: es su **recorrido** sobre el conjunto, `aₓ·(ln x_max − ln x_min)`. Su regla
iguala en el máximo contra un suelo `x = 1`, lo que coincide con el recorrido **sólo si el mínimo
de cada variable es 1**. En su modelo APP inventario y backorders sí bajan a 0 (suelo 1), así que
su regla ≈ igualar recorridos. En el nuestro τ ∈ [1,00, 1,34] y κ̇ ∈ [~0,6, 1,57] nunca se acercan
al suelo, y por eso quedan sobrepesadas.

```
regla de recorrido:   aₓ = 0,20 / (ln x_max − ln x_min)
```

**Es una generalización estricta, no una métrica nueva: con `x_min = 1` se reduce exactamente a la
suya.** Eso es verificable y es el falsador `f3`.

## 3. El diseño

**Eje D — vector de costes** (3 niveles). Ningún nivel tiene parámetros libres: los tres quedan
determinados por la caché de calibración.

| nivel | definición | fuente |
|---|---|---|
| `garrido_c1` | los siete = 1 | su supuesto (6), §3.1 — la referencia |
| `holding_decoupled` | `c_i = mean(P)/mean(I)`, el resto = 1 | reparación mínima de un solo parámetro: quita **sólo** la dominación y deja su base intacta |
| `scale_neutral` | `c_x = 1/mean(componente_x)` en los cuatro activos | ningún término domina κ por magnitud de unidad; es «aislar κ̇ de la influencia de los parámetros» hecho de forma que se cumpla |

**Eje E — normalización de exponentes** (2 niveles): `at_max` (`0,20/ln x_max`, la suya) ·
`over_range` (`0,20/(ln x_max − ln x_min)`).

Cruzados con los dos ejes que la familia anterior ya declara —`variables` ∈ {`his_five`, `no_tau`,
`plus_service`} y `kappa_set` ∈ {`within`, `global`}— dan `3 × 2 × 3 × 2 = 36` celdas, de las
cuales **6 duplican** exactamente variantes ya medidas (`garrido_c1 × at_max`). **30 variantes
nuevas.**

## 4. Multiplicidad — el precio, fijado aquí

**`K = 188`** = las 158 agrupadas + estas 30. **Holm-Bonferroni sobre las 188.** Umbral por
variante: `LCB95_corregido ≥ 0,05`, el de todos los gates del proyecto. **La cota de share se
aplica igual que antes**: una variante que cruza violándola no cuenta, exactamente como las cuatro
que cruzaron en la familia anterior.

## 5. Falsadores — cada uno con por qué puede fallar

| falsador | qué exige | por qué puede fallar |
|---|---|---|
| `f1_premise_A_holds` | la cuota media del inventario en κ bajo c=1 ≥ 0,50 | si κ no estuviera dominado por el inventario, el defecto A no existiría y toda la mitad D del diseño sería una reparación de nada |
| `f2_the_repair_decouples` | bajo `scale_neutral`, `corr(ln κ̇, ln ζ) < 0,90` | **riesgo real**: ε recorre 6,40 en ln contra 1,28 de ζ, así que re-pesar hacia ε puede dejar κ siguiendo a ε en vez de desacoplarlo. Si falla, el eje D no repara lo que dice reparar y se reporta así |
| `f3_over_range_reduces_to_his_rule` | con `x_min = 1` y **sus** máximos (3.612 / 2.191 / 148 / 28,0 / 3,1) la regla de recorrido debe devolver 0,024 / 0,026 / 0,040 / 0,060 / 0,1771 a 3 cifras | falla si mi lectura de su regla es incorrecta. **Es el falsador que decide si esto es su índice reparado o una métrica nuestra disfrazada** |
| `f4_direction_was_predicted_first` | se registra **antes de correr**: `over_range` debe **bajar** `H_regime` respecto de `at_max` | de-pesa κ̇ —el único componente con `H > 0`, 0,00187— a favor de ζ y φ, cuyo `H` medido es exactamente 0. **La predicción va en contra de lo que nos conviene**; si sube, se reporta como violación de la predicción y no como confirmación de nada |
| `f5_negative_control_stays_at_zero` | en la rejilla de 288 donde una configuración es óptima en las seis regiones, **toda** variante nueva debe dar `H = 0` | si alguna reparación fabrica headroom donde no puede haberlo, es un artefacto de la métrica y **mata el eje entero** |
| `f6_share_bound_respected` | toda variante que cruce declara si respeta su cota | la misma regla que descalificó a las cuatro anteriores |
| `f7_multiplicity_applied` | ningún `LCB95` crudo se compara contra 0,05 | — |
| `f8_no_fresh_seeds` | réplica declarada del bloque quemado | custodia central |

## 6. Reglas de lectura, fijadas de antemano — las tres direcciones

* **Ninguna cruza** → `SCALE_REPAIR_DOES_NOT_CREATE_HEADROOM`. Se han reparado **dos defectos
  independientes y medidos** del índice publicado y sigue sin haber headroom. La pregunta métrica
  queda cerrada sobre 188 variantes, y el manuscrito lo dice con ese número. **Los dos defectos se
  reportan igual**, porque son hallazgos sobre el índice de Garrido con independencia del headroom.
* **Alguna cruza y respeta su cota** → `SCALE_REPAIR_REACHES_THE_BAR`. Hallazgo real. Se reporta
  con su posición entre las 188, su corrección de Holm, y **la familia entera al lado**. Se declara
  explícitamente que el headroom aparece tras reparar la métrica, con el mecanismo medido.
* **Alguna cruza violando su cota** → `ONLY_MISSCALED_VARIANTS_REACH_THE_BAR_AGAIN`. Idéntico
  tratamiento que las cuatro anteriores: propiedad de la familia métrica, no de la cadena.

**Alcance:** desarrollo sobre tapes quemados. No abre semillas, no adjudica, no autoriza aprendices.
Los dos defectos de §1 y §2 son **hallazgos sobre el índice publicado** y son reportables tanto si
el headroom aparece como si no; ésa es precisamente la razón de que medirlos no sea buscar señal.
