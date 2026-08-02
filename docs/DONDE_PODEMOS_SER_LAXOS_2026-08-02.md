# Dónde podemos ser laxos para avanzar — y dónde no, con la razón de cada caso

El PI fijó la regla en `CLAUDE.md`: *«No artificial or useless blockers. Guardrails must earn their
place; several of our own past constraints —not Garrido's— are what killed the only headroom we
ever found.»* Este documento aplica esa regla a los bloqueos vigentes.

## 1. El bloqueo principal se declara a sí mismo NO vinculante

`contracts/authority_ladder_v1.json`, la autoridad que he estado invocando para congelar toda
ejecución científica, tiene en su propio campo de estado:

```
"status": "DRAFT_PROSPECTIVE_UNOPENED_NOT_AUTHORITY"
```

**Un documento que declara literalmente `NOT_AUTHORITY` no puede ser el bloqueo duro que yo —y
cuatro revisiones externas— hemos tratado como vinculante.** Sus banderas
(`fresh_roots_opened: false`, `neural_training_authorized: false`) son el estado de **su propio
programa**, no una prohibición sobre todo el repositorio.

**Consecuencia:** abrir semillas nuevas **es una decisión del PI**, no una violación de contrato.
Lo que sí exige disciplina es *cómo* se abren: bloque declarado en el registro, contrato antes de
correr, y regla de lectura fijada. **Eso vale la pena; la congelación no.**

Aquí me corrijo de forma explícita: he invocado ese contrato como impedimento absoluto en varios
mensajes. Era una lectura excesiva de un borrador.

## 2. Ser laxo con «desarrollo vs confirmación» cuando el contrato dice otra cosa

Llamé a las 120 réplicas de H3′ «evidencia de desarrollo sobre semillas quemadas». **Falso, y lo
desmiente el contrato**: el bloque `6.000.001–120` se abrió **una vez y para H3′**, virgen. El
defecto fue de **etiqueta** —el runner selló contra su contrato por defecto—, no de ciencia, y las
re-ejecuciones reproducen los originales al último decimal.

> **Regla que adopto:** un error de etiquetado no degrada un resultado a desarrollo si el bloque
> era virgen para su contrato, la regla de lectura no se movió y la reproducción es exacta. Lo que
> sí lo degradaría es mirar y luego extender, mover el umbral, o re-correr hasta que pase.

## 3. Dónde SÍ aflojar, con criterio

| bloqueo | veredicto | por qué |
|---|---|---|
| **congelación de semillas por `authority_ladder_v1`** | **aflojar** — decisión del PI | el documento no es autoridad; y sin semillas G3-obs es indecidible para siempre |
| **G3-obs sin potencia** (MDE 0,026 vs SESOI 0,010) | **aflojar el bloqueo, no el SESOI** | hacen falta ~106–132 semillas. **El SESOI no se toca**: aflojarlo sería elegir el resultado |
| **`ret_excel` como métrica** | **no aflojar** | está **medido** premiando el abandono de un reclamante |
| **«un margen por guardarraíl»** | **aflojar la exigencia de margen cero** | ya causó un halt por **un** pedido en 16 semillas |
| **prohibir entrenar** | **aflojar sólo tras el gate de residual** | Program Q ya midió que la familia estructurada iguala a la red |
| **esperar la firma de Garrido para la física E\*** | **aflojar parcialmente** | `CLAUDE.md`: *«no dependemos de Garrido… lo decidimos, lo declaramos como nuestra asunción y lo tasamos»*. Se puede construir E\* con asunciones declaradas y precio medido |

## 4. Dónde NO aflojar — cada una tiene un incidente detrás

* **Preregistrar antes de correr.** Sin esto no hay forma de distinguir hipótesis de post hoc.
* **Un falsador debe poder fallar, y verse fallar.** Un `passed: True` cableado dejó pasar una
  fuga real. Un falsador no evaluable se cuenta `NO APLICA`, en **ninguna** columna.
* **No re-ajustar un margen después de ver su resultado.** Es superioridad disfrazada.
* **No editar artefactos fechados en sitio.** Se supersede con banner.
* **Medir por la tubería, nunca con un script ad hoc.** Ya fabricó un defecto falso.
* **No confundir estimandos.** Hoy estuve a punto de adjudicar H3′ sobre la media en vez de sobre
  la varianza: habría resuelto otra hipótesis.

## 5. Lo que esto desbloquea, en orden

1. **Ya hecho:** H3′ adjudicada, `SUSTAINED` a n = 120 → cubre **H3 (Volatility Reduction)** del
   borrador v.0, y el efecto Alzheimer cubre **H4 (Path Dependency)**.
2. **Sin semillas:** reconciliar WRAP-288; reparar la justificación de `δ`; adjudicar **H2**
   (curva de aprendizaje) desde `per_context.regret_curve`, que **ya está en los artefactos**.
3. **Con semillas, si el PI las autoriza:** G3-obs con ~130 réplicas — el único experimento cuyo
   resultado hoy es literalmente indecidible por potencia, no por ciencia.

**Y el punto de calendario:** el bloqueo real de Submission A es **editorial y humano**, no
experimental. Aflojar la congelación de semillas no acelera la sumisión; acelera **el paper
siguiente**, que es donde H3′ y el efecto Alzheimer viven.
