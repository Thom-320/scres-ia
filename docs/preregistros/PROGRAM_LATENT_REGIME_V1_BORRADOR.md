# CONTRATO SIDE QUEST — program_latent_regime_v1 (borrador para firma de gates)

**Fecha:** 2026-08-25 · **Estado:** BORRADOR — los gates de semana 1 se ejecutan
bajo el preregistro GATE0 ya firmado; este contrato solo se activa si pasan.

## Decisión del PI (2026-08-25, en sesión)

«Side quest: elegir camino (régimen latente según 2/3 planes vs safety-first HPRS
según OpenCode) y firmar su contrato.»

**Camino elegido: régimen latente no estacionario** (convergencia Codex E2 +
subagente plan A). Razones: (i) es el único cuyo régimen favorable está declarado
textualmente por los papers benchmark (Gijsbrechts `a7:1605-1608`, Kaynov
`b12:97-103`); (ii) cadena de falsificadores más barata y temprana (~16–22 CPU-h
decide); (iii) mantiene el comparador fuerte con simetría legítima de información.
El camino safety-first HPRS queda como fallback pre-autorizado si el gate de
aliasing mata este.

## Física nueva (contrato nuevo prospectivo — nada toca O/O-R/Q)

Demanda Markov-modulada con **2–3 regímenes latentes** que modulan cv, tendencia
y mezcla de productos P_C/P_H. El régimen NO es observable: ni learner ni clásicos
reciben el campo. Calibración de parámetros de régimen a variabilidad plausible
(ancla narrativa: Garrido 2024 §3).

Simetría de información (lo que legitima la caza): min_cost_flow / max_pressure /
belief-MPC pierden su modelo exacto y deben re-estimar el régimen online desde la
misma historia observable que el learner.

## Gates previos (antes de entrenar NADA) — ~16–22 CPU-h

| Gate | Diseño | Regla de cierre |
|---|---|---|
| **G0-split** | Ya firmado y EN EJECUCIÓN bajo `gate0_split_tape_v1` (mismas 3 celdas, física base). Su resultado calibra el evaluador. | UCB95(G_PI_split)<0.01 → cerrar lane |
| **G0-LATENTE** (nuevo, específico) | Mismo diseño split-tape pero sobre la física con régimen latente: tapes A/B del bloque nuevo, k* sobre 65.536 calendarios, oráculo por-cinta vs mejor fijo y mejor clásico con re-estimación online. | UCB95(G_PI_split_latente)<0.01 → cerrar sin entrenar |
| **Sonda de aliasing** | Construir pares de historias indistinguibles en estado instantáneo (21 campos) pero con regímenes distintos; medir con política-oráculo cuánto vale responder distinto. | gap-oráculo-por-historia <0.01 → memoria no es mecanismo aquí, cerrar |
| **Placebo fungible** | Obligatorio: =0 exacto o la lane se invalida. | ≠0 → invalidar |

## Semillas

Bloque nuevo a solicitar tras scan de colisiones (candidato: siguiente rango libre
post-755xxxx; verificación obligatoria contra `program_q_s_seed_registry_v1.json`
y scan repository-wide como se hizo para 7550001–7550512).

## Brazos si TODOS los gates pasan (smoke primero)

1. Frontera open-loop completa reseleccionada intra-bootstrap.
2. min_cost_flow__2 y max_pressure__0 con re-estimación online.
3. belief-MPC con modelo estimado.
4. RecurrentPPO LSTM128 + PBRS-Q21 + γ corregido + 400k condicional (fix-pack b–e).
5. PPO feed-forward sobre posterior expuesto (ablación memoria-vs-observación).
6. Placebo desinformado.

Presupuesto completo ≈160–200 CPU-h (gates ~20 + smoke 42 + campaña 100–140).
Reserva final SIEMPRE: 20–30 CPU-h para confirmatorio virgen.

## Estimandos separados desde el preregistro

- Eficacia: Δ_N sobre ReT snapshot v2 (comparable con Q), δ=0.01.
- Seguridad: worst_product_fill no-inferioridad δ=0.01 prospectivo (la lección de Q:
  nunca optimizar media y descubrir cola después).

## Falsadores

1. G_PI_split_latente < 0.01 → física sin headroom, cerrar.
2. Sonda de aliasing nula → memoria no es mecanismo, cerrar.
3. Clásico con re-estimación empareja al learner → equivalencia otra vez
   (publicable como frontera, no prima).
4. Learner gana media y pierde cola → jerarquía del objetivo era el supresor
   dominante → derivar al fallback HPRS.

## Compromiso

Se publica gane o pierda. Un negativo aquí refuerza el manuscrito CIE («la
no estacionariedad sola no crea prima neural en esta clase de DES»). Nunca se
presenta como réplica o extensión de Q.
