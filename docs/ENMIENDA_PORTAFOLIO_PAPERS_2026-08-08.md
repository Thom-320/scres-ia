# Enmienda — lock del portafolio publicable SCRES-IA

Esta enmienda fija la numeración y la propiedad de claims para el release actual sin reescribir
los documentos históricos. La autoridad operativa está en
`papers/PORTFOLIO_CLAIM_LOCK.json`; el lock particular del manuscrito gobierna el wording que entra
al paper.

## Decisiones

- P2 es la única sumisión inmediata a *Computers & Industrial Engineering*.
- P1 queda reducido a nota conceptual; no se somete en paralelo.
- P3 permanece como dossier `HOLD_DOSSIER` y no autoriza nuevas ejecuciones.
- La única ejecución científica nueva autorizada es la sensibilidad de demanda estacional,
  `DEVELOPMENT_SENSITIVITY`, con límite de 72 horas, sin semillas vírgenes, entrenamiento ni
  retuning.
- El registro de lanes prometedoras conserva su inventario para trazabilidad, pero queda retirado
  como autoridad de lanes abiertas para este portafolio: no hay lanes genuinamente abiertas.

## Correcciones de autoridad

- El censo de confirmaciones se deduplica en **tres confirmaciones únicas**; dos son utilizables por
  P2.
- La custodia viva contiene **35 bloques registrados**, **cero bloques vírgenes disponibles** y
  `new_seed_opening=false`.
- `repro_probe/A` y `repro_probe/B` son pruebas de no determinismo de la misma DMLPA y semilla; no
  son reproducción entre arquitecturas.
- La demanda se comparte como declaración de alcance, no como contribución duplicada.
- `GSA` se limita a censo o suplemento y no puede ser la espina de otro paper.

Los papers antiguos mantienen valor histórico y trazabilidad, pero no pueden reabrir un lane,
semilla o claim cuyo propietario actual está fijado en el lock.
