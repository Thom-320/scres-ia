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
| **R1r** (riesgos frecuentes, bajo impacto) | no confirmado; efecto minúsculo negativo | −0,0000195 | [−0,0000494, −0,0000002] | 5/16 |

En R2r el control receding-horizon **sí convierte bajo el endpoint preregistrado** el valor que crean las variables
adicionales, y lo hace consumiendo **99.072 unidades menos** de material estratégico que
la postura fija. En R1r el contraste es ligeramente negativo pero dos órdenes de magnitud
menor que el SESOI: no confirma una mejora material. La respuesta a «¿sirve el MPC?»
resulta **dependiente del régimen y del endpoint**, no universal.

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

En raíces vírgenes el clipping cambió muy poco el contraste y no cambió su signo
(`+0,01252` canónico frente a `+0,01247` acotado). Por tanto, la cola extrema observada
en desarrollo no gobierna la conclusión prospectiva, aunque tampoco afirmamos que la
población nueva esté literalmente libre de valores fuera de rango.

## 4. Su índice Cobb-Douglas, portado

Implementamos el índice de su artículo de 2024 (IJPR, Eq. 3–6). Los exponentes están
**re-derivados con su propia regla** `0,20/ln(x_máx)` a partir de nuestros máximos, no
copiados: los suyos codifican inventarios de miles y los nuestros corren en millones.
Costos `c = 1` en los siete coeficientes, que es su asunción (6) del §3.1.

Hace explícito un intercambio que ReT no cobra: cuando dos niveles de turno tienen fill
idéntico, el índice puede separarlos por el cargo `c_u·U_t` a la capacidad ociosa. Pero
esa lectura es condicional a `c = 1`, no una validación económica. También encontramos
su límite: en R2r coloca segunda una postura con
**76% de fill y 16 órdenes perdidas**, robustamente en su propio barrido de sensibilidad
`c ∈ [1,2]`, porque una orden perdida sale de la cola de backorders y deja de costar.
En nuestra grilla independiente más amplia (`0,5×/1×/2×/5×`) el ganador de R1r es
estable, pero el de R2r cambia entre brazos al variar precios relativos de inventario y
backorder. Por eso el índice no selecciona políticas.

De ahí nuestra conclusión metodológica: **ninguna de las tres métricas puede garantizar
servicio por sí sola**, por mecanismos distintos. Lo tratamos como restricción declarada
aparte y como eje del frente de Pareto, no como una propiedad que se presume a partir del
objetivo escalar.

## 5. Lo que esto **no** autoriza

- **No autoriza entrenar una red, ni KAN ni MLP.** El residuo observable frente al mejor
  controlador estructurado no está establecido: en R1r no hay separación y en R2r la señal
  es dependiente del endpoint. La adjudicación lo marca explícitamente
  (`neural_authorization: false`).
- **No reemplaza retroactivamente ninguna cifra publicada ni congelada.** La métrica
  histórica queda intacta; lo acotado se reporta como columna adicional.
- **La confirmación adaptativa no adjudica el contrato conjunto buffers × turnos.**
  Después se enumeró el dominio estático completo de 648 posturas; eso cierra cobertura
  estática, no el controlador adaptativo conjunto.

## 6. Decisiones adoptadas sin convertir su respuesta en bloqueo

La tesis fija `LT=48` y define autotomía cuando `CTj=LTj`, pero no especifica un delay
fijo de 54 h. Por ello conservamos 54 h solo como ancla histórica de reproducción y
preregistramos una sensibilidad no selectiva en 42/47/48/49/54/60 h. Ningún valor podrá
elegirse por producir más headroom. También separamos `CTj=LTj` (autotomía de tesis) de
`CTj<=LTj` (servicio puntual).

Para κ no inventamos precios ni detenemos el estudio: `c=1` queda como réplica publicada,
Cobb-Douglas como sensibilidad, y la comparación primaria de recursos se hace por frente
de Pareto físico sin escalarización. Un vector monetario con unidades, año y fuente solo
será necesario para una afirmación monetaria o una recomendación económica de despliegue.

Su respuesta a ambas cuestiones sigue siendo bienvenida como validación de dominio, pero
no podrá seleccionar retrospectivamente delay, costos, endpoint o controlador. La decisión
completa está congelada en `contracts/paper_b_independent_calibration_v1.json`.

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
