# Lectura para el Prof. Garrido — estado al 30 de julio de 2026

**Borrador para revisión y envío por el PI. No enviado.** Redactado por el agente; el
envío requiere decisión y acción humanas.

---

Profesor Garrido:

Cerramos el paso 3 de su diseño del 28 de julio — «corres el MPC con más variables» — con
confirmación prospectiva sobre raíces nuevas. Le escribo el resultado con su discrepancia
declarada, porque es lo que hace la lectura interpretable.

## 1. Lo que cerró

Contrato congelado antes de abrir raíces, 216 posturas estáticas enumeradas por completo,
cinco futuros por candidato, replay del prefijo con verificación de hash de estado en cada
ramificación. **32 tapes vírgenes**, 16 por familia de riesgo.

| familia | veredicto | Δ MPC − mejor postura fija | CI95 | tapes |
|---|---|---:|---|---:|
| **R2r** (riesgos raros y severos) | **confirmado, material** | **+0,01247** | [+0,00911, +0,01591] | **15/16** |
| **R1r** (riesgos frecuentes, bajo impacto) | no confirmado | −0,0000195 | cruza cero | 5/16 |

En R2r el control receding-horizon **sí convierte** el valor que crean las variables
adicionales, y lo hace consumiendo **99.072 unidades menos** de material estratégico que
la postura fija. En R1r no se separa del incumbente: la respuesta a «¿sirve el MPC?»
resulta **dependiente del régimen de riesgo**, no universal.

Vale precisar cuál es el comparador: no es una postura cómoda, es el **mejor de los 216
vectores heterogéneos** enumerados sobre los mismos tapes. Los incumbentes reales
resultaron poco intuitivos —R1r (0, 0, 336) sin buffer de materia prima en ningún nodo, y
R2r (336, 0, 168)— y ninguno de los dos estaba en el conjunto que habríamos elegido a
mano.

## 2. La discrepancia, declarada

**El mismo resultado se invierte según el endpoint.** Sobre los mismos 16 tapes de R2r:

| endpoint | Δ MPC − estático | tapes a favor |
|---|---:|---:|
| ReT visible (su fórmula del Excel) | **+0,01252** | 15/16 |
| ReT visible acotada a [0,1] | **+0,01247** | 15/16 |
| **ReT sobre el ledger completo, sin censura** | **−0,00448** | **2/16** |
| tasa de cumplimiento (fill) | +0,00234 | 11/16 |
| raciones entregadas | **−25.399** | 0/16 |
| material estratégico inyectado | **−99.072** | 0/16 |

El MPC entrega **menos volumen** con **mejor tasa** y **mucho menos recurso**. Es un
intercambio, no dominancia. Y la métrica sin censura favorece al estático.

No presentamos esto como una victoria limpia. La lectura defendible es: *bajo el endpoint
tal como está especificado en su tesis, el MPC separa de forma material y consistente en
R2r; bajo la variante sin censura del mismo endpoint, no.* Ambas cifras están en el mismo
artefacto y se reportan siempre juntas.

## 3. Tres hallazgos sobre la métrica, que creemos son la contribución más fuerte

Al construir el instrumento encontramos tres propiedades de ReT. Ninguna es un error de
implementación nuestra: reproducimos su fórmula con **0 discrepancias en 47.546 filas**
contra los Excel reales.

**a) Dependencia de la frecuencia de observación — corregida.** El ReT de una trayectoria
físicamente idéntica variaba **37%** según cada cuánto se consultaba el simulador. La causa
era que `RPj` leía un cursor que el paso de simulación reiniciaba. Lo separamos: el inicio
del intervalo de caída ahora se registra una vez y no se toca. Post-corrección el ReT es
idéntico a nueve decimales entre observación horaria, diaria, semanal, mensual y de un solo
paso. Queda como prueba de regresión.

**b) La rama de autotomía es inalcanzable en nuestro modelo.** Requiere `CTj ≤ LTj`, y
nuestro ciclo mínimo es de **54 h contra una promesa de 48 h**, invariante a turnos,
buffers y riesgos. Así que `excel_case_pct_autotomy = 0` siempre y toda orden puntuada cae
en la rama de recuperación. Su propio Cf1 reporta `Media APj = 0,4486`, positiva — su
modelo sí alcanza la rama. **Esta es la brecha de fidelidad que más nos preocupa**, porque
la absorción es el mecanismo central de resiliencia de la tesis. El 54 se calibró para
reproducir el orden de magnitud de su ReF; nos gustaría entender si en su modelo hay
órdenes que se cumplen dentro de las 48 h.

**c) La rama `0,5/RPj` no está acotada.** En un tape encontramos una orden entregada 192 h
tarde con ReT = **73,91**, porque combinaba un indicador de cantidad (R24) con solo 0,0068 h
atribuibles a riesgos temporales. Siete órdenes de 3.108 inflaban la media familiar un 6%.
No afirmamos que R24 *causara* ese retraso — la atribución es retrospectiva por eventos y
la ruta causal no está identificada. Lo que sí está medido es que **ReT no es monótona en
el atraso**: en ambas familias la orden de mayor puntaje está entre las *menos* tardías,
mientras la más tardía puntúa cerca de cero.

En raíces vírgenes esa cola **no apareció**, así que la reparación resultó innecesaria para
la conclusión de §1 — el `no separado` que veíamos antes en R2r era contaminación de esos
tapes concretos, no una propiedad del controlador.

## 4. Su índice Cobb-Douglas, portado

Implementamos el índice de su artículo de 2024 (IJPR, Eq. 3–6). Los exponentes están
**re-derivados con su propia regla** `0,20/ln(x_máx)` a partir de nuestros máximos, no
copiados: los suyos codifican inventarios de miles y los nuestros corren en millones.
Costos `c = 1` en los siete coeficientes, que es su asunción (6) del §3.1.

Repara lo que buscábamos: en ReT visible el turno 1 y el turno 3 empatan con fill idéntico,
así que nada restringe elegir tres turnos; su índice los separa por el cargo `c_u·U_t` a la
capacidad ociosa. Pero también encontramos su límite: en R2r coloca segunda una postura con
**76% de fill y 16 órdenes perdidas**, robustamente en su propio barrido de sensibilidad
`c ∈ [1,2]`, porque una orden perdida sale de la cola de backorders y deja de costar.

De ahí nuestra conclusión metodológica: **ninguna de las tres métricas premia servicio**, por
tres mecanismos distintos. Lo tratamos como restricción declarada aparte, no como término
dentro del objetivo.

## 5. Lo que esto **no** autoriza

- **No autoriza entrenar una red, ni KAN ni MLP.** El residuo observable frente al mejor
  controlador estructurado no está establecido: en R1r no hay separación y en R2r la señal
  es dependiente del endpoint. La adjudicación lo marca explícitamente
  (`neural_authorization: false`).
- **No reemplaza retroactivamente ninguna cifra publicada ni congelada.** La métrica
  histórica queda intacta; lo acotado se reporta como columna adicional.
- **No adjudica el contrato conjunto buffers × turnos.** Esta confirmación fija un turno.

## 6. Dos preguntas concretas

1. **¿En sus corridas hay órdenes cumplidas dentro de las 48 h de promesa?** Es lo que
   decide si nuestro piso de 54 h es artefacto de modelado o propiedad del caso, y de ello
   depende si la rama de autotomía debe reactivarse en todo nuestro cuerpo de resultados.
2. **¿Nos daría un vector de costos con unidades, año y fuente** para los siete
   coeficientes de κ? El `c = 1` de su artículo nos sirve como réplica, pero el término de
   costo lleva el exponente más grande del índice y no queremos que la mayor ponderación
   descanse en la cantidad menos fundamentada de nuestro modelo.

Y una nota que quizá le interese: el §6.2 de su artículo de 2024 nombra como trabajo futuro
justamente lo que tenemos —fallas de máquina y faltantes de material como riesgos, «purchasing
and material requirements planning» como proceso táctico, y «discrete simulation techniques
and robust learning-based algorithms» como método.

---

## Custodia (para el expediente, no para el correo)

- Contrato `contracts/ret_metric_repair_confirmation_v1.json`, sha `c1efdc20…`, congelado
  antes de abrir raíces, `confirmation_roots_opened: false` en el momento de la firma.
- Raíces R1r 1710001–1710016 (local), R2r 1810001–1810016 (VPS ovh-agent-lab), verificadas
  una por una contra el contrato.
- R2r recuperado por `rsync --checksum` a ruta nueva y sellado read-only.
- Adjudicación `results/metric_audit/ret_metric_repair_confirmation_v1/`, sha
  `bde02309f72c9ee1…`. Commits `864472b`, `c59b3b5`.
- Todo condicionado a `GARRIDO_FULFILLMENT_DELAY_HOURS = 54` y a cadencia de decisión de
  cuatro semanas.
