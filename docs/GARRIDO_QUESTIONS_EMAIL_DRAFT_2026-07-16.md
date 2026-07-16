# Borrador de correo a Garrido — preguntas de validación (2026-07-16)

**Estado: BORRADOR — no enviado.** El usuario revisa, ajusta el tono/relación y envía.
Fuente canónica de las preguntas (redacción completa y reglas de interpretación):
`research/paper2_exhaustive_search/garrido_face_validation_questions.md` (batch Q1–Q14).
Este correo prioriza las 4 decisivas + la aclaración métrica M1; el batch completo puede ir
como adjunto.

---

**Asunto:** Consulta sobre supuestos operacionales de la MFSC — 4 preguntas puntuales

Estimado profesor Garrido:

Seguimos trabajando sobre el gemelo digital de la cadena MFSC de su tesis. Tras un programa
extenso de experimentos, tenemos un resultado sólido pero condicional: bajo los supuestos del
modelo, las políticas estáticas bien elegidas resultan casi óptimas y el valor adicional de
políticas adaptativas es nulo o no convertible. Antes de publicar, necesitamos validar con usted
cuatro supuestos operacionales que son exactamente los que podrían revertir esa conclusión. Le
agradeceríamos respuestas aunque sean cualitativas:

1. **Caducidad de misión y triage (la más importante).** Su modelo abandona pedidos por desborde
   de la lista de 60 backorders (§6.5.4: el último de la lista se etiqueta *lost/unattended*), y en
   sus datos ese mecanismo es el modo de falla dominante (12–31% de los pedidos terminan en Ut, con
   la lista saturada cerca de 60 en todas las configuraciones con riesgo). Nuestra pregunta es sobre
   la realidad detrás de esa simplificación: ¿existe además una caducidad TEMPORAL dura — plazos de
   misión tras los cuales el requerimiento se abandona — más corta que los tiempos de recuperación
   típicos (24–120 h)? Su tesis menciona un "order cancellation time" (p. 75): ¿es un plazo físico
   real — y de cuánto — o una descripción conceptual del límite de 60? ¿El tope de 60 backorders
   refleja capacidad operacional real o fue una conveniencia de modelado en Simulink? ¿Eliminar el
   último pedido de la lista es una regla doctrinal intencional? ¿Y la autoridad real permite solo
   priorizar pedidos contingentes (R24), o también admitir/rechazar/abandonar pedidos — con clases
   de criticidad dentro de los contingentes?

2. **Recurso de restauración compartido.** Su Figura 6.1 muestra UN solo Maintenance Battalion
   para todo el sistema, pero el modelo asume tiempos de recuperación INDEPENDIENTES por operación
   (Tablas 6.6b/6.7b). En la realidad, tras disrupciones simultáneas (planta + LOC + unidad de
   avanzada), ¿ese batallón (u otro recurso único: cuadrilla, equipo especializado) debe
   SECUENCIAR las recuperaciones — menos equipos que sitios afectados — o son efectivamente
   paralelas e independientes como asume el modelo?

3. **Clases de ración no sustituibles.** Su tesis documenta que el MFSC real ensambla 21 tipos de
   ración según requisitos nutricionales Y condiciones climáticas (§6.3.1), simplificados a uno
   solo en el modelo (§6.5.3). ¿Hay al menos dos clases mutuamente NO sustituibles (p. ej.
   clima frío vs. selva, o religiosa/médica) que compartan la misma línea de ensamblaje, con mezcla
   de demanda incierta, persistente y parcialmente anticipable? ¿Magnitudes aproximadas
   (participación de cada clase, persistencia)?

4. **Modelo económico de la flota descendente.** Los convoyes de distribución (Op10/Op12),
   ¿operan con programación fija reservada (salen según calendario y consumen horas-vehículo vayan
   cargados o vacíos) o bajo pago-por-uso (cada viaje cargado consume recurso marginal)? En su
   experiencia, ¿qué utilización típica tiene esa flota?

5. **Competencia y reparto entre CSSUs (Op11).** Su topología incluye dos CSSUs, pero el modelo
   no describe ninguna regla de reparto entre ellas y los registros agregan los pedidos sin
   destino. En la realidad: ¿varias CSSUs compiten simultáneamente por una misma entrega escasa?
   ¿Quién decide el reparto y con qué regla (¿equidad entre teatros, prioridad de misión?)? ¿El
   decisor observa demanda/backlog/nivel de servicio POR CSSU al momento de repartir?

Además, una aclaración métrica breve (M1 del documento adjunto): en su cálculo de ReT en Excel,
¿el estado de cada pedido se evalúa en el instante de la solicitud? Queremos confirmar que nuestra
réplica formal coincide con su intención.

Con gusto le compartimos el borrador del artículo; su tesis es la base empírica y nos importa que
el modelo quede fielmente representado.

Un cordial saludo,
Thomas Chisica

---

**Nota interna (no enviar):** mapeo a los reopeners del certificado — (1)=Q11/R09 *strongest*
(refinada 2026-07-16: la variante cap-60 ya fue agotada por D1; solo el deadline temporal real +
autoridad de admisión/evicción reabren); (2)=Q6/Q7; (3)=Q13 (solo restaura representatividad del
techo H_PI 0.152, no un positivo); (4)=Q14 (solo delimita el hallazgo retirado de desarrollo; NO
decisivo tras el STOP OOS 26/48); (5)=hecho de dominio que decide el estatus del probe op11
(HOLD_PENDING_DOMAIN_FACT — sin confirmación de competencia multi-CSSU con autoridad de reparto
observable, op11/max_min_fill es una extensión sintética, no una puerta tesis-nativa).
Reglas de uso: una respuesta "reopens" NO autoriza entrenamiento — autoriza preregistrar el
contrato correspondiente con oracle-first. Ver sección "What these questions do NOT authorize".
