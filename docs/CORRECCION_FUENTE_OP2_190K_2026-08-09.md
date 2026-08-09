# Auditoría complementaria de fuente — Op2 son 190.000 unidades de **cada** materia prima

**Fecha:** 2026-08-09

**Estado:** `SOURCE_EXPLICIT_PER_RAW_MATERIAL__INDEPENDENTLY_CONFIRMED`

**Artefacto:** `results/procurement_overorder_source_v2/result.json`

**Productor:** `scripts/verify_procurement_overorder_source_v2.py`

## Relación con la corrección canónica

La base remota canónica `c09cd2d` ya corrigió D5 a `MATCHES-PUBLISHED`; el artefacto v1 también
incorporó la Tabla 6.20 y el control S=3. Este artefacto v2 no supersede esa corrección ni reclama
haberla originado. Es una auditoría complementaria e independiente que acota las búsquedas a las
secciones Op2→Op3 y Tabla 6.20→Tabla 6.21, y separa el estatus textual de la aritmética derivada.

La comprobación independiente encuentra dos evidencias directas:

1. La descripción de Op2, página impresa 84, enumera explícitamente
   `Q = {190,000 rm1, ..., 190,000 rm12}`.
2. La Tabla 6.20, página impresa 108, repite para S=1, S=2 y S=3
   **«190,000 units of each rm»**.
3. La misma tabla hace variar Op3 con la capacidad: 15.500, 31.000 y 47.000 unidades de cada
   materia prima por semana.

Por tanto, la lectura por materia prima queda sustentada directamente por el texto de la fuente. La
aritmética y la interpretación de diseño que siguen son comprobaciones separadas.

## Aritmética publicada

| cantidad | componentes/semana | razón sobre demanda S=1 |
|---|---:|---:|
| demanda media: 2.500 raciones/día × 6 días × 12 rm | 180.000 | 1,0000× |
| Op2: 190.000/rm ÷ (672/168 = 4 semanas) × 12 rm | 570.000 | 3,1667× |
| Op3 S=1: 15.500/rm/semana × 12 rm | 186.000 | 1,0333× |
| Op3 S=3: 47.000/rm/semana × 12 rm | 564.000 | 3,1333× |

Op2 equivale a **47.500 unidades/rm/semana**, sólo 1,06 % por encima del flujo S=3 de Op3. Esto es
compatible con un aprovisionamiento dimensionado para el techo del experimento de capacidad y
mantenido constante entre S=1, S=2 y S=3. La explicación de dimensionamiento es una inferencia de
diseño; no se presenta como cita ni intención declarada del autor.

## Alcance frente al artefacto v1

Esta auditoría no vuelve a ejecutar ni reinterpreta la simulación de v1. Conserva como referencias
separadas de aquella cinta:

- el contrafactual que divide 190.000 entre doce;
- el resultado del contrafactual en esa cinta (servicio 0,4015 y stock final cero);
- el resultado de la cinta auditada de v1: bajo ese escenario, el stock final es positivo y
  Op2 no ata;

La auditoría v2 sí recalcula de forma independiente la aritmética
570.000/180.000 = 3,17× y mantiene la misma frontera metodológica: bajar el volumen contratado es
una extensión declarada, no una reparación de la fuente.

## Consecuencia de diseño, fuera del claim textual

No se tocarán silenciosamente los 190.000 para «hacer que RL gane». Son válidos dos brazos
distintos, siempre etiquetados:

- **brazo fuente:** Op2 permanece en 190.000 por rm cada 672 h;
- **brazo extensión:** capacidad de aprovisionamiento próxima al consumo, con el precio de
  fidelidad medido y sin atribuir el parámetro a Garrido.

Esta corrección sólo adjudica el texto y la aritmética de fuente. No valida headroom, Program V ni
ningún sucesor neural; esas afirmaciones dependen de sus propios contratos y artefactos.
