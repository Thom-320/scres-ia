# Frontera de claims tras H3′, y la autorización de semillas — cinco correcciones

Corrige `docs/RESULTADO_H3_ADJUDICACION_FUSION_2026-08-02.md` §4 y
`docs/DONDE_PODEMOS_SER_LAXOS_2026-08-02.md` §1. **No se reescribe el cuerpo de ninguno**; las
cifras siguen siendo válidas, lo que se estrecha es su alcance.

## 1. La autorización, y cómo se resuelve la gobernanza

Argumenté que `authority_ladder_v1` no puede bloquear porque su propio estado dice
`DRAFT_PROSPECTIVE_UNOPENED_NOT_AUTHORITY`. **La revisión externa corrige bien la mitad que me
faltaba:** que un documento no sea autoridad **no convierte la ausencia de autoridad en permiso**.
Ambas cosas son ciertas a la vez, y la resolución es una tercera:

> **El borrador ni prohíbe ni autoriza. La autoridad es el PI.** El PI autorizó explícitamente en
> sesión el 2026-08-02, y esa autorización queda **registrada en el propio bloque de semillas**
> (`research/seed_custody_registry.json`, bloque `g3_obs_v2_powered`, campo `authorization`).

La revisión admite exactamente esta vía —*«o una supersession explícita y registrada»*—, y esto lo
es. **El bloque se abre sobre la autorización del PI, no sobre mi lectura del borrador**, y así
está escrito en el artefacto.

## 2. «G3-obs es indecidible para siempre» — retirado

Escribí eso. **Es falso.** G3-obs está **subpotenciado ahora**: MDE 0,0256/0,0286 contra un SESOI
de 0,010, resoluble con ~106–132 réplicas. Es una limitación de **potencia**, no de ciencia, y una
autorización la levanta. La formulación correcta era *«indecidible sin más semillas»*.

**Y la regla que sigue intacta:** un resultado subpotenciado **nunca** se convierte en un nulo.

## 3. H3′ no es la H3 del borrador v.0

| | |
|---|---|
| **H3 del borrador v.0** | *«learning-enabled models reduce performance **variance across heterogeneous disruption intensities**»* — varianza del **desempeño/resiliencia** |
| **H3′, lo que está sostenido** | varianza del **COSTE DE BÚSQUEDA entre contextos** |

**Son constructos distintos.** H3′ es una reformulación **más estrecha**, adoptada porque la H3
original no era testable en este entorno. En el manuscrito debe escribirse **H3′ sostenida**,
nunca «H3 probada», y con su definición al lado.

## 4. H4 — apoyo estrecho, no demostración

Escribí que el efecto Alzheimer «cubre H4 (Path Dependency)». **Sobreafirmado.** Lo medido es que
**el estado retenido `ρ` reduce el coste de búsqueda en este contrato**. Eso **no** demuestra
dependencia histórica de la resiliencia de la cadena en general. Redacción defendible:

> **Apoyo estrecho a H4:** la memoria entre campañas reduce de forma medible el coste de encontrar
> una buena configuración. La resiliencia entregada no se midió como función de la historia.

## 5. H2 es exploratoria, no adjudicable hoy

Dije que H2 era «adjudicable sin abrir semillas» porque las curvas de regret ya están en
`per_context`. **Tener los datos no es tener una hipótesis.** Falta el estimando y la regla
inferencial **preregistrados**. Hasta entonces, H2 se reporta como **análisis descriptivo
exploratorio**, con esa etiqueta.

## 6. Y G3c tiene DOS bloqueadores, no tres

El 2 quedó reparado por `docs/ENMIENDA_G3C_MARGENES_OPERACIONALES_2026-08-02.md`. Siguen abiertos
el **1** (el contrato mezcla `min_dwell` y `switch_cost`; hace falta factorial con niveles
especificados) y el **3** (identidad del brazo nulo sin verificar con payload canónico).

## 7. La frontera de claims para C&IE

| afirmación | estado defendible |
|---|---|
| **Q1 — qué IA imita el SCL** | **condicional**: en cuatro contratos separados los controles estructurados capturaron el valor que las redes no capturaron. No universal |
| **Q2 — cómo integrarla en el DES** | interfaz DES → política → estado retenido **especificada y parcialmente probada**; WRAP-288 y E\* no son validación completa |
| **H1** | **no evaluable**: TTR censurado. H1′ es otro constructo |
| **H2** | curvas de aprendizaje **descriptivas**, sin adjudicación |
| **H3′** | **SOSTENIDA, n = 120**, LCB95 +2,3491 — como reducción de varianza del **coste de búsqueda** |
| **H4** | **apoyo estrecho** vía memoria; no path-dependency general |
| **G3-obs** | **subpotenciado, NO negativo** — ahora autorizado a repetirse con potencia |
| **G3c** | bloqueado, 2 bloqueadores |
| **E\*** | sólo diseño |

**La contribución fuerte para C&IE no es «la red ganó».** Es una integración DES–aprendizaje con
memoria, trazabilidad completa y un resultado positivo estrecho sobre estabilidad de búsqueda,
mientras la superioridad neural sobre controles estructurados **no queda establecida** — y eso
responde las dos preguntas de Garrido con evidencia que su artículo exploratorio no tiene.
