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

1. **Caducidad de misión y triage (la más importante).** Cuando un requerimiento de raciones no
   puede atenderse a tiempo, ¿existe un plazo duro tras el cual se abandona definitivamente (no se
   acumula como backorder)? ¿Ese plazo es más corto que los tiempos típicos de recuperación de las
   disrupciones recurrentes (días)? ¿Y la agencia logística tiene autoridad doctrinal para hacer
   triage/admisión de pedidos?

2. **Recurso de restauración compartido.** Tras disrupciones simultáneas (planta, línea de
   comunicación, teatro), ¿existe UN recurso real —equipo, cuadrilla, vehículo especializado— que
   se asigna de forma mutuamente excluyente entre esos frentes, con menos equipos que sitios
   afectados (obligando a secuenciar)? ¿O los presupuestos de recuperación son separados y paralelos?

3. **Clases de ración no sustituibles.** De los 21 tipos reales de ración, ¿hay al menos dos clases
   mutuamente NO sustituibles (religiosa/médica/climática) que compartan la misma capacidad de
   ensamblaje restringida, con una mezcla de demanda incierta, persistente y parcialmente
   anticipable? ¿Podría darnos magnitudes aproximadas (participación, persistencia)?

4. **Modelo económico de la flota descendente.** Los convoyes de distribución (Op10/Op12),
   ¿operan con programación fija reservada (salen según calendario y consumen horas-vehículo vayan
   cargados o vacíos) o bajo pago-por-uso (cada viaje cargado consume recurso marginal)? En su
   experiencia, ¿qué utilización típica tiene esa flota?

Además, una aclaración métrica breve (M1 del documento adjunto): en su cálculo de ReT en Excel,
¿el estado de cada pedido se evalúa en el instante de la solicitud? Queremos confirmar que nuestra
réplica formal coincide con su intención.

Con gusto le compartimos el borrador del artículo; su tesis es la base empírica y nos importa que
el modelo quede fielmente representado.

Un cordial saludo,
Thomas Chisica

---

**Nota interna (no enviar):** mapeo a los reopeners del certificado — (1)=Q11/R09 *strongest*;
(2)=Q6/Q7; (3)=Q13 (solo restaura representatividad del techo H_PI 0.152, no un positivo);
(4)=Q14 (solo delimita el hallazgo retirado de desarrollo; NO decisivo tras el STOP OOS 26/48).
Reglas de uso: una respuesta "reopens" NO autoriza entrenamiento — autoriza preregistrar el
contrato correspondiente con oracle-first. Ver sección "What these questions do NOT authorize".
