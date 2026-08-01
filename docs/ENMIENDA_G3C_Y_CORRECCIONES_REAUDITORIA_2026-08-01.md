# Enmienda — el sucesor inmediato es la conversión observable, no física nueva

**Enmienda prospectiva a `docs/PREREGISTRO_G3C_ACOPLAMIENTO_TEMPORAL_2026-08-01.md`**, escrita
**antes de ejecutar nada** contra él y sin tocar el documento fechado. Ningún dato se ha visto
contra el contrato de G3c: sigue `DESIGN_ONLY_NOT_AUTHORIZED`.

## 1. La corrección que cambia el orden de trabajo

Escribí que la reauditoría demuestra que **«la restricción vinculante era la clase de política,
no la simetría»**, y lo usé para degradar G3a y adelantar G3c. **Es demasiado fuerte, y lo
retiro.**

El experimento **no manipuló** ni demanda asimétrica ni etiquetas físicas. Midió el valor de una
política adaptativa **en la física existente**. De ahí no se sigue que la asimetría sea
innecesaria para G3a. Lo defendible es más estrecho:

> En este contrato legacy, una política diaria con acceso al **estado verdadero** mejora la
> distribución del servicio entre reclamantes frente a constantes y frente a una variación **no
> informada**. No es una prima neural, no es una ventaja desplegable, y no dice nada sobre si la
> asimetría hace falta.

**Y el sucesor inmediato no es física nueva.** Lo que tengo es un **techo clarividente**: la
política lee el estado verdadero. Mi propia escalera exige `G2 — conversión observable` antes de
añadir mecanismo, y me la salté. El paso correcto, más barato y sobre **tapes quemados**, es:

> **Sustituir el clarividente por observaciones causales previas a la acción** —backlog por
> reclamante, inventario, tránsito, edad de pedidos, rutas arriba/abajo— y comparar contra
> constante, umbral, árbol, política tabular y DP/MPC.

**Si un umbral de dos ramas sobre observables captura el valor, no hay residual y el carril se
cierra** sin construir permanencia mínima ni coste de cambio.

## 2. El orden enmendado

| # | paso | por qué |
|---|---|---|
| **1** | **G3-obs — conversión observable sobre la física actual** | convierte un techo clarividente en `H_obs`, o lo mata. Tapes quemados, sin semillas nuevas |
| 2 | **G3c — acoplamiento temporal** | sólo si G3-obs deja residual sobre el mejor control estructurado. Su contrato y **sus márgenes firmados siguen vigentes tal cual** |
| 3 | **G3a — asimetría física** | **no degradado**: es el único contrato que puede expresar una equivarianza A↔B real, hoy `NOT_EXPRESSIBLE_IN_split_v1`. Su prioridad relativa la decide G3-obs, no la reauditoría |

**Los márgenes de no inferioridad del preregistro de G3c no cambian** y se aplican también a
G3-obs: `flow_fill_rate` δ = 0,005 · `lost_orders` δ = 0,25 · `backorder_qty_final` δ = 1,0 %
relativo · identidades algebraicas δ = 0 exacto · SESOI primario +0,010 · potencia ≥ 0,90 fijada
por preflight sobre tapes quemados.

## 3. Cinco correcciones de lectura sobre la reauditoría

Todas ciertas, todas mías:

1. **«Los ocho falsadores pasan» es un sobrealcance.** La lectura correcta:
   **seis operativos pasan, `f3b` NO ES EVALUABLE, y `f7` FALLA.** Un falsador que no puede
   fallar no se cuenta entre los que pasaron — que es exactamente la regla que yo mismo escribí
   tras el incidente del `passed: True` cableado.
2. **`f4` no reproduce el artefacto sellado.** Comprueba shares, reglas y prefijo de semillas: es
   **compatibilidad de contrato**, no reproducción de trayectorias. El nombre en el preregistro
   (`..._reproduces_sealed_artifact`) promete más de lo que el runner hace; la evidencia dentro
   del artefacto ya lo decía, el nombre no.
3. **`f5` prueba el rango usado, no la custodia completa.** Demuestra que *esta* corrida usó
   `5.200.001–16`. No sustituye al registro central — que ahora existe y **se declara a sí mismo
   `BASELINE_INVENTORY_INCOMPLETE`**.
4. **El fallo de `f7` no puede llamarse «ruido» todavía.** Con 16 semillas, `+0,0625` pedidos es
   **compatible con variación muestral, no demostrado como tal**. La acción tomada —no corregir,
   no re-correr, preregistrar márgenes con potencia— era la correcta; la palabra no lo era.
5. **El hash no es «imposible», es el instrumento inadecuado.** `self_sha256` es un sello interno
   válido; lo que no es, por incluir `created_at` y `calibration_provenance`
   (`supply_chain/arm_runner.py:158`), es un hash estable de payload científico.

Y una sexta, ya corregida en el estado canónico: **H3′ no ha aterrizado**. Hay dos rebanadas con
semillas disjuntas y **no existe** un artefacto merged; el estado es
`ARTIFACTS_PRESENT_MERGE_PENDING` con la rebanada VPS en `HOLD_SOURCE_AUDIT`.

## 4. Etiqueta canónica de la reauditoría

El artefacto `results/headroom/contention_policy_class/result.json` conserva
`HALTED_FALSIFIER_FAILED` y se lee como:

> `DEVELOPMENT_POLICY_CLASS_SIGNAL_BUT_HALTED_BY_SERVICE_GUARDRAIL`

Sin promoción, sin `H_obs` demostrado, sin autorización de entrenamiento.
