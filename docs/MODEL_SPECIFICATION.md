# Model Specification: SCRES+IA vs Garrido (2017)

## Overview
This document specifies the formal Discrete Event Simulation (DES) model of a military supply chain (MFSC) based on Garrido-Rios (2017), Section 6.3. The model is single-product ("Cold weather combat ration # 1", Section 6.5.3).

## 13 Operations Architecture (Reference: Section 6.3, Figure 6.2)

| Op | Name | Processing Time (PT) | Batch Quantity (Q) | Reorder Point (ROP) |
|----|------|----------------------|--------------------|---------------------|
| 1 | Military Logistics Agency | 672 hrs | 12 contracts | 4,032 hrs |
| 2 | Suppliers (12 suppliers) | 24 hrs | 190,000 rm × 12 | 672 hrs |
| 3 | Warehouse & Distribution Centre | 24 hrs | 15,500 rm × 12 | 168 hrs |
| 4 | LOC WDC→AL | 24 hrs | 15,500 rm × 12 | 168 hrs |
| 5 | Assembly Line - Pre-assembly | 1/λ = 0.003125 hrs | 1 | Immediate |
| 6 | Assembly Line - Assembly | 1/λ = 0.003125 hrs | 1 | Immediate |
| 7 | Assembly Line - QC & Shipping | 1/λ = 0.003125 hrs | 5,000 rations | 48 hrs |
| 8 | LOC AL→SB | 24 hrs | 5,000 rations | 48 hrs |
| 9 | Supply Battalion | 24 hrs | U(2400,2600) | 24 hrs |
| 10 | LOC SB→CSSUs | 24 hrs | U(2400,2600) | 24 hrs |
| 11 | Combat Service Support Units (2) | 0 hrs | U(2400,2600) | 24 hrs |
| 12 | LOC CSSUs→Theatre | 24 hrs | U(2400,2600) | 24 hrs |
| 13 | Theatre of Operations (Demand) | — | Drg ~ U(2400, 2600) | — |

*Note: λ = 320.5 rations/hour (Section 6.3.3).*
*Note on Op9-12 Q:* Table 6.20 lists U(2000,2500) but text (Section 6.3.3) and Figure 6.2 define it as U(2400,2600). The code implements U(2400,2600) following the text.

## Material Flow Topology
Raw Materials (RM) → Rations → Theatre
1. Op1 (Contracts) → Op2 (Suppliers) provide raw materials.
2. RM flows through Op3 (WDC) via Op4 (LOC) to Op5 (Assembly Line).
3. Op5-Op7 assemble RM into Rations at rate λ.
4. Rations flow from Op7 via Op8 (LOC) to Op9 (Supply Battalion).
5. Op9 distributes via Op10 to Op11 (CSSUs), then Op12 to Op13 (Theatre).

## State Variables
- `raw_material_wdc`: RM at Warehouse (Op3)
- `raw_material_al`: RM at Assembly Line (Op5)
- `rations_sb`: Rations at Supply Battalion (Op9)
- `rations_cssu`: Rations at CSSUs (Op11)
- `total_backorders`: Scalar tracking unfulfilled ration demand.
