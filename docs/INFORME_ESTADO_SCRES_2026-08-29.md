# SCRES — Informe de estado de la evidencia

**Fecha:** 2026-08-29 · **Alcance:** todo lo medido hasta hoy sobre headroom físico,
valor de la realimentación observable, valor capturado por control estructurado, y
la ausencia de prima neural.

> **Este informe NO cierra el programa.** Documenta un estado de evidencia y fija
> las condiciones bajo las cuales una campaña nueva estaría científicamente
> autorizada. La razón de no cerrar está en §6 y es decisiva: el cambio de física
> que el asesor identificó como la causa del nulo **sigue sin implementarse**.

---

## 1. Headroom físico: negativo fuera de muestra, replicado dos veces

La pregunta es si existe una configuración mejor que encontrar. La respuesta,
con separación selección/evaluación, es que **no, y buscarla cuesta**.

### Gate-0 split-tape (bloque 7550001–7550512, 64+64 tapas/celda)

| celda | G_PI_naive | **G_PI_split** | CI95(split) | Δ_bias |
|---|---|---|---|---|
| rho75_share90 | +0,05830 | **−0,11722** | [−0,13761, −0,09803] | +0,17552 |
| rho90_share75 | +0,03283 | **−0,08571** | [−0,10070, −0,07121] | +0,11855 |
| rho90_share90 | +0,05483 | **−0,11560** | [−0,14146, −0,09077] | +0,17043 |

`lane_closed: true` · `selection_bias_material: true` · `CLOSED_LANE` en las tres.
Custodia: ganador congelado 15:33:12 (sha256 `d9b28684…`), tapes B evaluadas
16:07:16 — **34 minutos después**. F3 pasa; F4 pasa con 116 réplicas
(77 del *instrument pool* + 39 de las tapas de evaluación), `max_abs_error`
3,33e-16 contra tolerancia 1e-12.

### Replicación independiente sobre los paneles de Program Q (7490xxx, 128+128)

| celda | G_naive | **G_split** | CI95 | crossfit K=8 | Δ_bias |
|---|---|---|---|---|---|
| rho75_share90 | +0,076674 | **−0,083546** | [−0,103212, −0,074488] | −0,087434 | +0,160220 |
| rho90_share75 | +0,051982 | **−0,079412** | [−0,096452, −0,068097] | −0,075875 | +0,131394 |
| rho90_share90 | +0,076570 | **−0,140307** | [−0,163172, −0,110273] | −0,126463 | +0,216877 |

`CEILING_IS_BIAS`. Semillas distintas, el doble de tapas, mismo signo y magnitud.

### El techo clarividente era sesgo, íntegro

`control_ceiling_v1` midió +0,079088 / +0,052858 / +0,076584 sobre el mejor
clásico. El `Δ_bias` de la replicación **supera ese techo en las tres celdas**. La
firma estaba a la vista: **244, 250 y 204 argmax distintos sobre 256 tapas** —
casi cada cinta quiere su propio calendario entre 65.536.

**Y el mejor calendario fijo pierde contra el mejor clásico**: `brecha_fijo`
−0,081109 / −0,073272 / −0,117650. Ninguno de los 65.536 calendarios iguala a un
controlador clásico.

---

## 2. Valor de la realimentación observable: real, grande y conservador

Lo que sí paga es **cerrar el lazo**. El aprendiz contra el mejor calendario fijo:

| celda | H_OL | LCB95 | UCB95 |
|---|---|---|---|
| rho75_share90 | **+0,0795238** | +0,0655536 | +0,0934940 |
| rho90_share75 | **+0,0725474** | +0,0619547 | +0,0831400 |
| rho90_share90 | **+0,1172400** | +0,1055039 | +0,1289761 |

**La cifra está subestimada, no inflada.** El comparador —`max_k mean_t`— es un
máximo dentro de muestra sobre 65.536 calendarios, así que lleva winner's curse a
su favor. H_OL real es mayor.

Es el resultado positivo del programa: condicionar en observaciones dentro del
episodio vale entre +0,07 y +0,12 de ReT, y **ninguna selección de calendario lo
alcanza** (§1).

---

## 3. Valor capturado por control estructurado: todo, dentro de ±0,01

RecurrentPPO contra el mejor comparador clásico, 256 tapas vírgenes por celda,
inferencia simultánea max-t:

| celda | Δ_N | LCB95 | UCB95 | TOST ±0,01 |
|---|---|---|---|---|
| rho75_share90 | −0,0015852 | −0,0064056 | +0,0032351 | **pasa** |
| rho90_share75 | −0,0007246 | −0,0056099 | +0,0041607 | **pasa** |
| rho90_share90 | −0,0004104 | −0,0027470 | +0,0019263 | **pasa** |

Los tres intervalos cruzan cero **y** caben en ±SESOI. No es un fallo en
rechazar: es **equivalencia certificada**, y con intervalos simultáneos, que la
hacen conservadora. El rango alcanzable por encima del clásico (~0,17) es
diecisiete veces el SESOI, así que la equivalencia es aritméticamente
significativa.

La descomposición Kitagawa/Oaxaca lo confirma mecánicamente: composición
**exactamente 0,000e+00** en los seis pares, `shares_equal_across_arms: true`, y
toda la Δ en el componente intra-régimen — **calculado, no asumido**.

---

## 4. Ausencia de prima neural: demostrada en cinco peldaños

| peldaño | contraste | resultado |
|---|---|---|
| **Control** | mlp − linear_feedback | **−0,5594** [−0,7476, −0,3856], 7/48 tapas |
| **Predicción** (112 semillas) | recurrent − random_forest_lagged | **−0,0298** [−0,0356, −0,0240] |
| | mlp_tuned − random_forest_lagged | **−0,1503** [−0,1657, −0,1362] |
| **Decisión** | neural − mejor clásico | **−8,45e-06** [−3,68e-05, +1,99e-05] |
| **Bucle externo** | neuron_memory − ucb1_transfer | **−0,0070** [−0,0244, +0,0140] |
| **Arquitectura** | KAN − MLP con parámetros igualados | **+0,010369** [+0,003018, +0,018926], **p=0,0012** |

**Con potencia demostrada.** La puerta que había cerrado la prima decidía con
MDE80 de 0,091–0,292 contra un SESOI de 0,05 — no podía ver el efecto. La
reapertura con 112 semillas vírgenes y la semilla como unidad de análisis da
**MDE80 de 0,0084 a 0,0205**. El nulo dejó de ser subpotenciado.

R² retenido, 112 semillas: `random_forest_lagged` 0,9509 · `gaussian_process_lagged`
0,9505 · `gbdt_lagged` 0,9500 · `kernel_ridge_lagged` 0,9482 · **`recurrent`
0,9211** · … · **`mlp_tuned` 0,7624**. Los cuatro clásicos con retardos por
delante del mejor brazo neural.

**Y la auditoría de la propuesta.** La Fig. 5 de Garrido, implementada
literalmente, es una **identidad algebraica**: R² = 1,0, `max_abs_identity_error`
3,2e-15, y 3 de 5 columnas de drivers idénticamente cero. Su activación propuesta
`1[ReT(x) > ReT(x−1)]` es **no diferenciable**, incompatible con el
backpropagation que el mismo paper recomienda.

---

## 5. Dos defectos propios que acotan lo anterior

Se declaran aquí porque un informe que solo lista aciertos no es un informe.

**El SESOI de la Puerta B era inalcanzable.** El mejor clásico está en R² 0,9509,
así que la prima máxima concebible —con un modelo **perfecto**— es **+0,0491**,
contra un SESOI de **0,0500**. `PREMIUM` era imposible por aritmética para
cualquier arquitectura. El SESOI se heredó «verbatim para comparabilidad» sin
contrastarlo contra el techo del endpoint. En consecuencia, **el veredicto
`EQUIVALENT` del brazo `recurrent` está sobrevendido**: el TOST pasó, pero la
mitad superior de la banda no era alcanzable, así que el test fue de hecho
unilateral. El contenido honesto es que el brazo está 0,0298 por debajo, de forma
fiable, y esa brecha es el 61 % del SESOI.

**La regla de decisión no particiona.** `mlp_tuned` tiene UCB95 = −0,1362, tres
veces el SESOI por debajo, y la regla lo etiqueta `UNDETERMINED` por no ser ni
`PREMIUM` ni equivalente. La etiqueta honesta es **`INFERIOR`**; un contrato
sucesor necesita esa cuarta rama.

Contraste que también se declara: la equivalencia de §3 **no** sufre este
problema. Su rango alcanzable es ~0,17 contra un SESOI de 0,01.

---

## 6. Condiciones mínimas para autorizar una campaña nueva

### 6.1 La condición que ya está sobre la mesa y no se ha cumplido

En la reunión del **7 de agosto de 2026**, el asesor identificó la causa y
prescribió el remedio:

> «La demanda actual es uniforme discreta (entre 2.400 y 2.600): variación mínima.
> Esto hace que el modelo aprenda fácilmente, **reduciendo la diferencia entre
> arquitecturas**.»

Su instrucción, en tres partes, **ninguna implementada**:

1. **Demanda no estacionaria.** Sustituir U(2400,2600) por suavización exponencial
   con **α y γ variados por Montecarlo**, según Garrido 2024. Verificado de forma
   independiente: es su Ec. 1, `GR_{t+v} = αD_t + (1−α)(F_t+δ_t) + δ_{t+1} +
   γ(F_{t+1}−F_t) + (1−γ)δ_t`, con α, γ ~ U[0,1) **una vez por corrida** sobre una
   semilla estacional de 36 valores. Es incertidumbre de parámetro con **régimen
   latente por episodio**, que es justo lo que falta.
2. **Riesgos R2 aleatorizados** (combinaciones de frecuencia e impacto), R1 fijos.
   Con la advertencia explícita del asesor: *un patrón determinístico lo aprende
   la red rápidamente* — debe ser genuinamente aleatorio para que ninguna política
   estática sea óptima.
3. **Pruebas de linealidad**: criterio AIC, prueba de Ramsey, y graficar variable
   contra tiempo a lo largo del episodio. Su hipótesis es que las variables son
   lineales y de ahí la poca diferencia entre modelos.

**Consecuencia para este informe:** los resultados de §1–§4 son válidos y
replicados, pero **describen un sobre que el propio asesor declaró demasiado
fácil para diferenciar arquitecturas**. Publicar «no hay prima neural» sin haber
corrido su física es un resultado sobre nuestra implementación, no sobre la
pregunta.

### 6.2 Condiciones metodológicas, derivadas de lo que costó aprenderlas

Cualquier campaña nueva debe cumplir las seis, y cada una tiene su factura:

1. **Separación selección/evaluación desde el primer día.** No negociable: el
   sesgo medido es +0,119 a +0,217, de 2 a 5 veces el estimador que corrige.
2. **SESOI contrastado contra el techo alcanzable del endpoint** antes de
   firmarlo. El de la Puerta B hacía imposible su propio veredicto positivo.
3. **Potencia calibrada con señal inyectada.** Un efecto sintético del tamaño del
   SESOI que el pipeline **debe** detectar. Si no lo detecta, ningún negativo
   posterior vale — y 17 de 23 contrastes históricos habrían quedado bloqueados
   por este criterio.
4. **El comparador no recibe el estadístico suficiente del aprendiz.** En la
   Puerta B entregamos los retardos a la clase clásica y `random_forest_lagged`
   ganó con ellos. Le regalamos su arma al rival.
5. **Placebo común y fijo, nunca autonormalizado por brazo.** El de
   `grid_transfer` se construía con las visitas del propio brazo evaluado, de modo
   que la dureza del control dependía del brazo.
6. **Cada falsador debe poder pasar y poder fallar.** Cuatro defectos de esta
   clase aparecieron en una sola sesión: F3 comprobaba presencia de claves y no
   valores; F4 declaraba `passed:true` con 0 de 77 réplicas; la descomposición
   fijaba la composición en 0,0 en vez de calcularla; y F2 del split exigía
   estabilidad de un argmax que el propio diseño reselecciona.

### 6.3 Regla de autorización

Una campaña de entrenamiento queda autorizada **si y solo si**, sobre la física
nueva de §6.1 y con el protocolo de §6.2:

- el **techo clarividente con separación selección/evaluación** supera el SESOI en
  al menos una celda —no el techo ingenuo, que aquí resultó ser sesgo íntegro—, y
- una **sonda de aliasing** muestra que responder distinto a historias
  indistinguibles en el estado instantáneo tiene valor no nulo, y
- el **placebo desinformado pierde**.

Si el techo con separación queda bajo el SESOI, la lane cierra por aritmética y
ningún hiperparámetro la abre. Ese es el criterio que separa «no hay prima» de
«no la buscamos donde se podía».

---

## 7. Qué se puede publicar hoy, y qué no

**Se puede sostener**, con intervalos y replicación:

- Cerrar el lazo vale +0,073 a +0,117 de ReT, y la selección de configuración no
  lo alcanza (pierde 0,073 a 0,140).
- La familia que cierre el lazo es indiferente en esta física: equivalencia
  certificada a ±0,01 en control, y en predicción los clásicos con retardos ganan
  con potencia demostrada.
- El headroom aparente en estudios sin separación selección/evaluación es
  sesgo: +0,119 a +0,217, medido dos veces en bloques independientes. **Ningún
  paper del corpus separa selección de evaluación** — ni Ding 2026 (IJPE), ni
  Guzmán 2026 (C&IE), ni Kong 2026.
- La Fig. 5 de Garrido, implementada literalmente, es una identidad.

**No se puede sostener** hoy:

- «No hay prima neural en cadenas de suministro resilientes.» El alcance real es
  esta física, con esta demanda, que el asesor ya calificó de demasiado fácil.
- Que tunear no ayudaría bajo la demanda de §6.1: no se ha medido.

---

## Procedencia

Gate-0 `verdict.json` sha256 `fff33501…` · `q_split_bias_v1` contrato sha256
`1821af8b…` · `control_ceiling_v1` contrato sha256 `28aab3df…` · reapertura de la
Puerta B contrato sha256 `af195737…` con enmienda K=4 · reunión con el asesor
2026-08-07. Commits `2740e6f`, `a558768`, `cf62e3ca`, `9efc5ae1`, `dff90c21`,
`44436950`.
