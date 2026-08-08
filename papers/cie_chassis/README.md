# Chasis de sumisión a C&IE — maquinaria, sin ciencia

Estos cinco ficheros vienen **byte a byte** de `origin/codex/submission-a-program-q`,
`papers/submission_a_program_q/`, por `git show`. Nada se recalculó.

| fichero | qué es | por qué sobrevive al cambio de paper |
|---|---|---|
| `CIE_GUIDE_AUDIT_2026-07-29.md` | 17 requisitos de la guía viva de C&IE contra evidencia | la guía no cambió de paper |
| `RELEASE_AND_SUBMISSION_CHECKLIST.md` | 8 hechos, 11 abiertos | los abiertos son humanos y siguen abiertos |
| `TITLE_PAGE.tex` | plantilla `elsarticle` separada, para el doble ciego | todos los campos humanos siguen en `PENDING` |
| `HIGHLIGHTS.txt` | el formato: 3–5 líneas, ≤85 caracteres | el contenido se reescribe entero |
| `GENERATIVE_AI_DISCLOSURE_DRAFT.md` | la redacción de la declaración de IA | exigida por la guía, independiente del contenido |

## Lo que NO se trajo, y es deliberado

**Ningún texto científico.** El paquete de origen sostiene *"When Feedback Beats an Exhaustive
Open-Loop Frontier but Not Structured Control"*, un paper de RecurrentPPO cuya familia de claims
la tabla canónica del 2026-08-07 §7 **prohíbe**, y cuyo `C1_VIRGIN_BLOCK` está en `NO-GO`. Traer
su `main.tex`, sus tablas o sus figuras sería reintroducir una generación de claims que ya se
retiró.

Se quedan fuera a propósito: `main.tex`, `references.bib`, las seis tablas generadas, las cuatro
figuras PNG, `COVER_LETTER_DRAFT.md` (habla del paper viejo), `CLAIMS_TABLE_2026-07-28.md` y el
PDF compilado.

## Lo que falta traer

`scripts/build_submission_a_cie_review_bundle.py` — el constructor de paquete anónimo con guarda
de identidad. No está en la rama bajo ese nombre; hay que localizarlo antes de poder producir un
envío doble-ciego, y cuando aparezca hay que **parametrizar `PAPER_ROOT`** (hoy cableado a
`papers/submission_a_program_q`) y **añadir los tokens de identidad de cualquier autor nuevo**,
porque sin eso la guarda no guarda nada.

## Regla

Cada cifra que entre a un manuscrito construido sobre este chasis debe existir en
`docs/TABLA_CANONICA_DE_CLAIMS_2026-08-07.md` o en su enmienda. Si no está ahí, no se cita.
