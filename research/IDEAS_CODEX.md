# Diez ideas experimentales rankeadas para SCRES-IA

Criterio de criba: sólo carriles aditivos y prospectivos; se excluyen el fix-pack, la lane topológica, los experimentos registrados en las primeras 120 líneas de `PROMISING_LANES_REGISTRY.md` y duplicados posteriores detectados en el repositorio. Ninguna idea autoriza reinterpretar O/O-R/Q ni reutilizar sus semillas adjudicadas.

## 1. Cota óptima exacta en un micro-MFSC
- **Idea (una línea):** Construir una instancia finita reducida del MFSC —dos productos, inventario/backlog/belief discretizados y las acciones canónicas— y resolverla por iteración de valor para medir la brecha óptima de RL, belief-MPC y heurísticas.
- **Paper que la motiva:** `/home/ubuntu/scres-sources/texts/a7-gijsbrechts2022-msom-can-deep-rl.txt`
- **Por qué podría dar un claim grande:** Distingue por primera vez si `Delta_N≈0` significa que ambos controladores son casi óptimos o que ambos están lejos del óptimo; además convierte “RL empata a MPC” en brechas certificadas, no sólo rankings.
- **Costo estimado CPU:** 50–120 CPU-h (barrido de discretización, solución exacta y evaluación CRN; la ingeniería no está incluida).
- **Riesgo de invalidar contratos históricos:** Nulo formal; es un modelo reducido nuevo con gate de fidelidad y semillas vírgenes, por lo que sólo acota mecanismos y no readjudica el DES completo.

## 2. Selección contextual y restringida de controlador
- **Idea (una línea):** Aplicar R&S contextual-restringido para aprender un mapa prescrito `x→{RL, belief-MPC, APP}` según riesgo, horizonte y entropía posterior, maximizando ReT sujeto a fill mínimo, edad de backlog y costo.
- **Paper que la motiva:** `/home/ubuntu/scres-sources/texts/a10-hong2021-fem-review-rs.txt`
- **Por qué podría dar un claim grande:** Sustituye el veredicto global casi nulo por una frontera estadísticamente controlada de “qué controlador es seguro y mejor, dónde”, con PCS/PGS condicional mínima y no sólo tres celdas agregadas.
- **Costo estimado CPU:** 35–70 CPU-h (políticas congeladas, tapes CRN nuevos y asignación secuencial de réplicas).
- **Riesgo de invalidar contratos históricos:** Bajo; debe preregistrarse como estimando nuevo y no puede reclasificar celdas O/O-R/Q ya adjudicadas, aunque sí podría estrechar claims globales futuros.

## 3. Comparadores APP dinámicos, no sólo posturas estáticas
- **Idea (una línea):** Portar como controladores online las cinco reglas APP compatibles con el contrato sin overtime —S11/S12/S13 chase y S21/S22 workforce constante— usando el mismo forecast, estado observable y conjunto de turnos que RL y belief-MPC.
- **Paper que la motiva:** `/home/ubuntu/scres-sources/texts/garrido2024_factory_resilience.txt`
- **Por qué podría dar un claim grande:** Prueba si la aparente necesidad de aprendizaje depende de haber comparado contra configuraciones estáticas; una victoria sobre match-chase dinámico sería mucho más creíble y una derrota identificaría una regla estructural transferible.
- **Costo estimado CPU:** 20–45 CPU-h (implementación determinista, tests de causalidad y evaluación; S31/S32 quedan explícitamente fuera por requerir overtime inexistente).
- **Riesgo de invalidar contratos históricos:** Nulo formal y medio narrativo; agrega una familia comparadora, pero podría retirar cualquier lenguaje amplio que aún sugiera ventaja frente a “lo clásico”.

## 4. Recurrent SAC/TD3 bien especificado para el POMDP
- **Idea (una línea):** Comparar Recurrent-SAC y Recurrent-TD3 con memorias separadas para actor y crítico, entradas previas `(o,a,r)` y contextos 5/64/completo frente a RecurrentPPO, PPO, belief-MPC y un oracle de estado.
- **Paper que la motiva:** `/home/ubuntu/scres-sources/texts/a6-ni2021-arxiv-recurrent-pomdp.txt`
- **Por qué podría dar un claim grande:** El negativo histórico sólo cierra una implementación on-policy de RecurrentPPO; este factorial puede demostrar que la memoria sí importa cuando algoritmo, gradientes y contexto son adecuados, o cerrar la hipótesis POMDP con mucha más fuerza.
- **Costo estimado CPU:** 180–320 CPU-h (dos algoritmos, tres contextos, checkpoints y confirmación multisemilla).
- **Riesgo de invalidar contratos históricos:** Nulo formal y medio narrativo; requiere preregistro y semillas vírgenes, y sólo podría revisar la generalización “memory is not a lever”, no el resultado de RecurrentPPO observado.

## 5. PBRS con subobjetivos de recuperación ordenados
- **Idea (una línea):** Entrenar con PBRS dependiente de historia cuyo potencial cuenta en orden “detener crecimiento de backlog viejo → recuperar fill del producto débil → liquidar backlog causado por el shock”, contra Q21, orden permutado, subobjetivos aleatorios y bonus ingenuo.
- **Paper que la motiva:** `/home/ubuntu/scres-sources/texts/a1-okudo2021-ieee-access-subgoal.txt`
- **Por qué podría dar un claim grande:** Ataca exactamente el tail guardrail que cerró Q sin cambiar el endpoint primario y permite atribuir una mejora a estructura operacional correcta, no a más recompensa; el control permutado prueba causalmente el orden.
- **Costo estimado CPU:** 90–170 CPU-h (screen de escala `eta`, 4–5 brazos y confirmación; potencial terminal fijado en cero).
- **Riesgo de invalidar contratos históricos:** Medio si se mezclara con Q; mitigación obligatoria: brazo Q-nuevo, invariancia verificada, mismo evaluador y semillas vírgenes, sin reabrir el veredicto Q histórico.

## 6. Frontera dosis-respuesta de calidad del forecast
- **Idea (una línea):** Dar a RL y belief-MPC el mismo forecast cross-fitted y barrer horizonte, sesgo, ruido y retraso entre anclas oracle/no-forecast para estimar `Delta_N` y seguridad de cola como función continua de WMAPE/Brier/calibración.
- **Paper que la motiva:** `/home/ubuntu/scres-sources/texts/b10-kong2026-eai-transformer.txt`
- **Por qué podría dar un claim grande:** Va más allá de quitar una columna: identifica el umbral de calidad informativa donde cada controlador gana o se vuelve inseguro y separa valor de predicción de ventaja arquitectónica.
- **Costo estimado CPU:** 60–120 CPU-h (generación de forecasts congelados, 12–20 celdas de corrupción y CRN; entrenamiento sólo si la política no tolera el protocolo compartido).
- **Riesgo de invalidar contratos históricos:** Bajo; es sensibilidad OOD aditiva y no debe reinterpretarse como fallo del H_OL canónico ni permitir información distinta entre brazos.

## 7. Shocks hard/soft/complex con daño físico igualado
- **Idea (una línea):** Crear tríadas CRN de shocks hard, soft y complex con igual pérdida integrada capacidad×tiempo y demanda expuesta, manteniendo fija la red y las acciones, para comparar `Delta_N`, nadir, tiempo de respuesta y costo de recuperación.
- **Paper que la motiva:** `/home/ubuntu/scres-sources/texts/1-s2.0-S0925527326000861-main.txt`
- **Por qué podría dar un claim grande:** Puede mostrar que la ventaja relativa depende de irreversibilidad y acoplamiento, no de “más severidad”; sería una taxonomía mecanística de cuándo belief-MPC o RL es apropiado sin entrar en reconfiguración topológica.
- **Costo estimado CPU:** 45–90 CPU-h (calibración del matching físico, tres clases, varios niveles y evaluación de políticas congeladas).
- **Riesgo de invalidar contratos históricos:** Bajo-medio; son escenarios sintéticos no-thesis que deben etiquetarse OOD y no pueden sustituir resultados canónicos, aunque podrían limitar su extrapolación.

## 8. Incertidumbre de inputs en dos niveles
- **Idea (una línea):** Propagar incertidumbre epistemológica con un loop externo que remuestrea tasas/severidades/demanda calibradas y un loop interno CRN que estima RL−belief-MPC, reportando probabilidad posterior del signo y descomposición input-versus-simulation.
- **Paper que la motiva:** `/home/ubuntu/scres-sources/texts/a10-hong2021-fem-review-rs.txt`
- **Por qué podría dar un claim grande:** Revela si `Delta_N≈0` es robusto a parámetros del mundo o sólo a semillas con inputs fijados; podría demostrar que los intervalos Monte Carlo actuales omiten la fuente dominante de incertidumbre.
- **Costo estimado CPU:** 40–100 CPU-h con diseño anidado secuencial o metamodelo validado (más si se usa fuerza bruta).
- **Riesgo de invalidar contratos históricos:** Nulo formal y medio interpretativo; los resultados previos siguen siendo condicionales a sus inputs, mientras el nuevo claim debe declarar la fuente y soporte del posterior.

## 9. Robustez minimax con adversario de riesgos calibrado
- **Idea (una línea):** Entrenar RL y robustificar belief-MPC contra el mismo adversario que elige frecuencia, severidad, demanda y lead time dentro de una ambiguity set calibrada, comparándolos además con domain randomization nominal.
- **Paper que la motiva:** `/home/ubuntu/scres-sources/texts/a8-boute2022-ejor-roadmap.txt`
- **Por qué podría dar un claim grande:** Somete a ambos paradigmas a model misspecification justa y puede descubrir una inversión del empate nominal en CVaR/worst-decile, produciendo un resultado de robustez y no otra carrera de medias IID.
- **Costo estimado CPU:** 160–280 CPU-h (juego alternado adversario-controlador, robust-MPC y confirmación fuera de la ambiguity set de entrenamiento).
- **Riesgo de invalidar contratos históricos:** Bajo-medio; nueva lane OOD con set congelado y simétrico, incapaz de readjudicar el mundo nominal pero capaz de acotar su validez externa.

## 10. Escalera epistemológica R1/R2/R3 con severidad emparejada
- **Idea (una línea):** Evaluar known-known, known-unknown y unknown-unknown dando respectivamente calendario, distribución o ningún modelo del shock, con daño esperado emparejado, y medir transferencia zero-shot de RL frente a belief-MPC.
- **Paper que la motiva:** `/home/ubuntu/scres-sources/texts/WRAP_Theses_Garrido_Rios_2017.txt`; `/home/ubuntu/scres-sources/texts/b4-guzman2026-cie-circular.txt`
- **Por qué podría dar un claim grande:** Convierte la taxonomía militar de riesgos en un experimento de información limpio: establece si el valor de RL emerge por adaptación a desconocidos o desaparece cuando el modelo probabilístico es correcto.
- **Costo estimado CPU:** 70–140 CPU-h (tres niveles de conocimiento, matching de daño, una política bloqueada sin retuning y tapes vírgenes).
- **Riesgo de invalidar contratos históricos:** Bajo; R3 y zero-shot deben permanecer OOD y los resultados sólo restringen generalización, nunca cambian los contratos históricos de R1/R2.
