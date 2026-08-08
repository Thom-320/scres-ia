# Preguntas al Prof. Garrido — 2026-08-08

**Qué es este documento.** El banco completo vive en
`research/paper2_exhaustive_search/garrido_face_validation_questions.md` (M1, M2, Q1–Q14). Esto es
el subconjunto que **hoy** decide algo, redactado para enviarse tal cual. Las demás quedaron
respondidas por el modelo o cerradas por medición.

**Una regla, y es la que hace que esto sirva.** Debajo de cada pregunta está escrito **qué implica
cada respuesta posible, antes de conocerla**. Ninguna respuesta es «la buena»: un «no» cierra una
familia con la misma limpieza con la que un «sí» la abre, y eso está fijado ahora para que no se
pueda releer después. Le pedimos respuesta **escrita y fechada** por la misma razón.

**Estado desde el que preguntamos, sin adornos.** La búsqueda interna está agotada y certificada:
no hay ninguna instancia positiva de control adaptativo en el sobre thesis-native, y las
extensiones que abrimos cerraron en negativo tras validación prospectiva. Ningún experimento
nuestro puede resolver lo que sigue: son **hechos del dominio**, no preguntas de modelado. Por eso
el cuello de botella del proyecto ya no es cómputo.

---

## 1. La pregunta que puede reabrir el problema — caducidad de misión y triaje (Q11 / R09)

**Contexto, y con una corrección nuestra por delante.** Durante un tiempo escribimos que la tesis
asume que los pedidos tardíos se vuelven backorders y nunca se abandonan. **Era falso.** La tesis
nombra un *tiempo de cancelación de pedido* tras el cual los backorders se recategorizan como `Ut`
«sin desaparecer del cálculo de resiliencia» (p. 75), y lo implementa como una **lista de backorders
con tope 60 y desalojo por desbordamiento** (p. 97, §6.5.4), con secuenciación SPT y prioridad
contingente R24. En los libros Excel reales la cola llega al tope 60 en las 20 configuraciones Cf y
se reportan entre 148 y 993 pedidos `Ut`.

Esa variante —la disparada por **capacidad**— ya la medimos y está cerrada: existe autoridad real de
racionamiento **constante** (≈ +0,0105 sobre el default de la tesis), y la regla preferida sí varía
con el estado (≈75/16/7 % de reparto), pero esa dependencia del estado aporta sólo ≈ +0,0011 y **no
es observablemente convertible**. Lo que R09 necesita es lo que el modelo **no** contiene: un plazo
**temporal** real más ajustado que las recuperaciones, y autoridad de admisión más rica que
«desalojar el último de la lista».

**Preguntas.** Para los requerimientos de teatro, en especial los pedidos de demanda contingente
R24:

* **(a)** ¿Un requerimiento no cubierto tiene un **plazo duro** tras el cual queda **permanentemente
  abandonado** —la misión se mueve o termina y las raciones ya no se necesitan—, en vez de quedar en
  backorder indefinidamente?
* **(b)** Si lo tiene, ¿cuál es la distribución del plazo, y es **más ajustada** que las escalas de
  recuperación de 24–120 h de R21/R23/R22?
* **(c)** ¿Tiene la agencia logística **autoridad doctrinal** para triar o rechazar qué pedidos
  entran al pipeline de cumplimiento (control de admisión)?
* **(d)** ¿Los pedidos R24 llevan **clases de criticidad de misión** más allá de la única bandera de
  prioridad contingente?
* **(e)** El «tiempo de cancelación de pedido» de la p. 75, ¿es un **plazo físico real** —y de cuánto—
  o es la descripción conceptual de la implementación de lista con tope 60 de la p. 97?
* **(f)** ¿El tope de 60 refleja capacidad operativa real o una conveniencia de modelado en Simulink?
  ¿Desalojar el **último** de la lista ordenada por SPT es una regla doctrinal intencional?

**Qué implica cada respuesta.** *Reabre* R09 si existen **a la vez** plazos duros más ajustados que
las recuperaciones **y** autoridad de triaje: eso crea un problema de admisión *no
work-conserving*, materialmente distinto del que ya cerramos. *Cierra* R09 si los pedidos siempre
entran en backorder, o los plazos son más laxos que la recuperación, o no hay autoridad de triaje —
en ese caso colapsa a lo ya medido. Bajo esos hechos queda **prohibido** tratar un pedido caducado
como si nunca hubiera existido: seguiría contando en el denominador.

**Campo ligado — carga de misión.** Antes del despliegue, ¿puede logística repartir una dotación
total fija de raciones entre cohortes usando duración de misión observada y planes de resuministro?
Límites de masa y volumen, autoridad de decisión, marcas de tiempo y exactitud de la señal, y reglas
de devolución o transferencia.

---

## 2. Dónde vive el headroom que sí medimos — recurso compartido escaso (Q6 / Q7)

**Por qué preguntamos exactamente esto.** El único headroom material que hemos encontrado en todo el
proyecto aparece bajo **contención por un recurso escaso y no fungible**: medimos `H_PI = 0,1515`
(LCB95 0,116), y —el control decisivo— **al hacer el recurso plenamente fungible el headroom es
exactamente 0**. Es un mecanismo causal, no una correlación. La pregunta es si esa contención
**existe en la MFSC real** o si es una construcción nuestra.

**Preguntas.**

* **(a)** ¿Existe **un** recurso nombrado —un equipo, vehículo, cuadrilla o bolsa de habilidades
  concreta— asignado de forma **mutuamente excluyente** entre (i) recuperación de planta y ensamble,
  (ii) reparación de línea de comunicaciones y (iii) respuesta en teatro, de modo que comprometerlo
  en uno **impide** los otros durante un tiempo real de activación y permanencia?
* **(b)** ¿Dónde está basado, con qué habilidades, y qué tiempos de viaje, activación y permanencia
  mínima aplican? ¿Cómo cambia causalmente una unidad el tiempo de recuperación en cada destino?
* **(c)** Tras un evento R21 o R3, ¿tiene el Batallón de Mantenimiento **menos equipos
  multi-habilitados que sitios inhabilitados**, forzando a serializar las reparaciones? ¿Cuántos
  equipos hay y dónde?
* **(d)** ¿Son las reparaciones **interrumpibles**? ¿Qué evaluación de daño existe **antes** de
  despachar, con qué demora y qué error?
* **(e)** Los valores de 120 h y 672 h de la tesis, ¿representan **trabajo de reparación de un
  equipo** o **inactividad autónoma** del sitio?

**Qué implica cada respuesta.** *Reabre* si hay un recurso genuinamente escaso y mutuamente
excluyente, con tiempo de viaje y reparación no nulo, y con evaluación de daño **no privilegiada**
antes del compromiso. *Cierra* si los presupuestos son separados o paralelos, o si hay al menos
tantos equipos como sitios caídos. La parte **(d)** importa más de lo que parece: si el evaluador
conoce el daño sin error, el problema deja de tener valor de información y vuelve a ser estático.

---

## 3. Validación externa del constructo de dos clases (Q13)

**Contexto.** La tesis comprime 21 tipos reales de ración en un producto homogéneo, y lo declara
como limitación en el Cap. 8. Nuestro Program O restituye ese rasgo con **dos clases no
sustituibles** que comparten la línea Op5–Op7 con capacidad limitada.

**Sea explícito el estatus: esto no es una petición de permiso.** La extensión está declarada y se
motiva sola desde la limitación de la propia tesis, y la investigación interna procede sin esperar.
Lo que su respuesta decide es si el resultado puede presentarse como **representativo de la MFSC
real** o sólo como un resultado de frontera.

**Preguntas.** Entre los 21 tipos reales:

* **(a)** ¿Existen **dos o más clases tales que una requisición de una NO puede cubrirse con otra**
  —halal/kosher, médica/terapéutica, específica de clima—, es decir no sustituibilidad genuina y no
  simplemente SKUs distintos?
* **(b)** ¿Comparten esas clases el **mismo recurso Op5–Op7 con capacidad limitada**, de modo que
  producir una consume capacidad que la otra necesita en el mismo periodo; o cada clase tiene
  capacidad dedicada sin contención?
* **(c)** ¿Es la **mezcla de demanda entre clases incierta y variable con persistencia**? Valores
  realistas de cuota de la clase dominante y persistencia del régimen (nosotros barremos 0,75 y
  0,90 en ambos).
* **(d)** ¿Es la mezcla **parcialmente observable por adelantado** del compromiso semanal de
  producción —con qué demora y qué error—, o se conoce exactamente, o no se conoce en absoluto?
* **(e)** ¿Hay **setup o cambio de formato** al conmutar el ensamble entre clases? (Hoy asumimos
  setup cero y BOM, masa y tasa idénticos.)
* **(f)** ¿Es realista una **asignación semanal de pocos lotes** —usamos tres sobre ocho semanas— o
  la decisión real es continua, diaria o anual?
* **(g)** ¿Es la reducción a **dos clases** una abstracción defendible, o la estructura real son
  muchas clases pequeñas?

**Qué implica cada respuesta.** *Valida* y licencia la lectura MFSC-representativa si se cumplen a
la vez (a), (b), (c) y (d). Un setup no nulo en (e) **no** colapsa el constructo: se añade a la
física y se vuelve a correr. *Colapsa* al nulo fungible exacto si todos los tipos son sustituibles
al cumplir, o cada clase tiene capacidad dedicada, o la mezcla es determinista o conocida de
antemano.

**Un matiz que le debemos.** Aunque la respuesta sea «valida», eso restaura la representatividad
**del techo**, no produce un positivo: la conversión observable de Program O ya falló fuera de
muestra en validación prospectiva.

---

## 4. Economía del transporte aguas abajo (Q14) — calibra, no rescata

**Contexto.** Nuestros controladores usan ≈ **2.280 de 5.376** horas-vehículo cargadas aguas abajo
(**42 % de utilización**) y entregan raciones que de otro modo quedarían varadas, llenando capacidad
ociosa. Que ese uso sea **gratis** o **comprado** decide qué afirmación de recurso es defendible.

**Preguntas.** Para Op10 (Batallón de Abastecimiento → CSSU) y Op12 (CSSU → Teatro):

* **(a)** ¿La capacidad de flete está **reservada y pagada con horario fijo** —vehículos y horas
  contratados y cobrados vayan cargados o vacíos—, o el coste se incurre **por viaje o por unidad
  transportada**?
* **(b)** Si es horario fijo: ¿hay rutinariamente **capacidad reservada ociosa** dentro del sobre
  contratado que entregas adicionales podrían llenar sin coste incremental?
* **(c)** Si es pago por uso: ¿cuál es la estructura de coste marginal —por viaje, por hora-vehículo,
  por ración-km?
* **(d)** ¿Son las 112 franjas diarias de despacho y las 5.376 horas-vehículo un **derecho reservado
  duro**, un objetivo blando, o un recurso medido?

**Qué implica cada respuesta, y sea claro: esta pregunta no puede rescatar nada.** La validación
fuera de muestra bajo reloj fijo **ya falló** en consistencia prospectiva. Un «reloj fijo con
capacidad ociosa» sólo hace **honesto en recursos** un hallazgo de desarrollo ya retirado y define
el alcance de una eventual sucesora preregistrada aparte. Un «pago por uso» limita la afirmación a
la rama de frontera y prohíbe cualquier positivo bajo esa economía.

---

## 5. El criterio de aceptación, que hoy bloquea el manuscrito (M2)

**Contexto, dicho contra nuestro propio interés.** La validación correctiva de Program O **pasó**
su gate canónico de ReT medio en las tres celdas —LCB95 entre 0,043 y 0,066, con 27 de 27 placebos
batidos— y **falló** un guardrail conjunto de no-inferioridad en la cola (`ret_visible_cvar10`) en
dos de las tres. Ese guardrail es un estándar de desplegabilidad **que añadimos nosotros**, no
parte del constructo de la tesis. Si pertenece o no a la regla de aceptación es una pregunta del
dueño del dominio, y **debe responderse prospectivamente y por escrito**, nunca inferirse de qué
respuesta nos conviene.

**Preguntas, deliberadamente de dos caras y respondidas por separado.**

* **(a)** Para **su constructo de tesis**: ¿el criterio de aceptación previsto era el ReT medio
  canónico **solo**, o la definición prevista de resiliencia incluía además un requisito sobre los
  **peores** resultados?
* **(b)** Para la **doctrina militar real**: ¿existen requisitos **vinculantes** sobre el peor
  servicio de teatro o la peor campaña —pisos de servicio, riesgo de cola—, o la resiliencia media es
  la medida operativa para decidir aceptación?

**Regla de interpretación, congelada ahora.** Una respuesta escrita y fechada de que el criterio es
el ReT medio canónico **(a)** *y* que la doctrina no impone requisito vinculante de cola **(b)**
autoriza —tras auditoría independiente del instrumento— un contrato de aprendiz nuevo con CVaR como
**reporte secundario, nunca como gate**. Cualquier otra respuesta mantiene la no-inferioridad de
cola dentro de la regla de aceptación. **Ninguna de las dos reescribe el STOP ya emitido**: esto
gobierna lo que venga después, no lo que ya se decidió.

---

## 6. Aclaración de métrica (M1)

En la barrera de peticiones original de Simulink, ¿los backorders acumulados `sumBt` y los pedidos
no atendidos `sumUt` se **congelaban en el instante en que se generaba** el pedido `j`, en `OPTj`?
Y si otra terminación, salida de cola o pérdida ocurría en la **misma marca temporal** de
simulación, ¿ese evento actualizaba `sumBt/sumUt` **antes o después** del snapshot adjunto a la
petición `j`?

**Qué implica.** Nada de headroom adaptativo. Valida o corrige únicamente la convención
determinista de orden de eventos de nuestro contrato de métrica. La ambigüedad mantiene bloqueada
la confirmación sobre semillas vírgenes, y cualquier corrección obliga a re-puntuar todas las cintas
de desarrollo y todos los comparadores.

---

## 7. Lo que le reportamos, no lo que le preguntamos

Tres resultados que le debemos porque van **contra** lo que esperábamos:

1. **El KAN busca peor que un MLP con parámetros igualados** (`KAN_SEARCHES_WORSE_THAN_A_MATCHED_MLP`).
   Con la arquitectura más gruesa fijando la rejilla para que la comparación sea justa. Sabemos que
   el KAN era la apuesta arquitectónica del artículo de 2024; la medición no la sostiene en este
   dominio.
2. **La superficie no premia expresividad.** Curvatura 0,076 frente a ruido 0,317: el MLP es **peor**
   que el ajuste lineal. Eso responde su Q1 con un número, y la respuesta es que la familia adecuada
   aquí no es la más expresiva.
3. **El índice Cobb–Douglas tiene dos defectos de escala medidos.** κ es 86–88 % inventario, y el peso
   efectivo sobre ζ resulta −0,368 frente al +0,014 nominal, porque la regla de exponentes es inversa
   al rango dinámico. Ningún vector de costes desacopla κ̇.

Y un positivo, que es el que sostiene el trabajo: **el efecto Alzheimer tiene precio medido**. La
neurona de su Fig. 5 arrastrando memoria alcanza el óptimo en **7,24** corridas, frente a **13,54**
reseteada y **12,42** para el OFAT de la propia tesis, con la ventaja creciendo a lo largo de seis
contextos. El bucle cerrado que usted describe **funciona** — en el bucle **externo** de búsqueda de
configuración, que es exactamente donde su Fig. 2 lo coloca, entre los nodos ③ y ⑧.
