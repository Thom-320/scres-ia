# Rescate — la confirmación H2/H3 que llevaba nueve días fuera de la rama científica

**Esto es un rescate, no una re-ejecución.** Los tres ficheros se traen **byte a byte** desde
`codex/paper-b-retained-v5` con `git checkout <rama> -- <ruta>`. Ningún número se recalcula,
ningún artefacto fechado se edita.

| fichero | bytes | sha256 |
|---|---:|---|
| `results/garrido_h2_h3_confirmation_v1/result.json` | 28.037 | `bc375d3021b64d10…` |
| `results/garrido_h2_h3_confirmation_v1/completion_receipt.json` | 859 | `d4305bcf6bf5209d…` |
| `results/garrido_h2_h3_confirmation_v1/tape_level_deltas.json` | 42.637 | `e12f3cf944c7ac0f…` |

Origen: `codex/paper-b-retained-v5`, creado `2026-07-29T16:53:36Z`,
`code_commit 9829084de18d0e1bf57d0da31d54beca56dd6997`, recibo de congelación `352a4dcaa4635c4a…`.

## Por qué importa

El barrido completo del repositorio encontró que **en toda la rama científica existe una sola
confirmación de grado-confirmación** (`grid_transfer_confirmation_v2`). Ésta es la segunda, y
nunca aterrizó aquí:

* `status: CONFIRM_H2_H3_ALL_SIX_PANELS`
* `global_confirmation_pass: true`
* `confirmation_roots_opened: true` — **doce raíces vírgenes**, `96111336 … 97836128`
* **los seis paneles pasan Holm**: p entre `4,76e−17` y `1,90e−15`
* en los seis: `delivered_lcb_positive`, `fill_lcb_positive`, `full_ledger_lcb_positive`,
  `unresolved_ucb_negative`, `generated_orders_exact_zero`

Es decir: el DES reconstruido reproduce **prospectivamente y sobre raíces vírgenes** las
hipótesis de moderación H2 (buffer) y H3 (turno) de Garrido en R1r, R2r y R3. **Eso es la sección
de validación del manuscrito**, y es lo único de esa sección que tiene grado de confirmación.

## La frontera, literal, que va con ella siempre

> *"Confirmation applies only to H2/H3 resource interventions in the frozen thesis-grounded
> reconstructed DES; it does not establish learner, feedback, or architectural value."*

No se cita para nada más. No dice que un aprendiz sirva, no dice que haya retroalimentación, y no
dice nada sobre arquitecturas.

## Lo que NO se rescata, y por qué

`codex/paper-b-cf1-cf20-replication:results/q_r1/successor_confirmation_v1/` lleva
`claim_status: PROSPECTIVE_CONFIRMATION` y un contraste positivo
(`early_ret_complete_cohort` IC95 `[+0,0875, +0,1047]`, 32 historias). **No se trae como claim:**
es el componente E1 del Programa Q, que está adjudicado con **STOP compuesto** sobre el
guardarraíl `worst_product_fill`. Se cita como negativo, con su veredicto terminal, o no se cita.

## Nota de procedencia

`main` está en `89acc81` (28-jul) y no contiene ninguna de las dos confirmaciones. La rama
científica es `codex/expanded-contract-comparators-v2`. Este rescate no reconcilia `main`; sólo
pone la evidencia donde el manuscrito la va a leer.
