# Paper 2 Maintenance Intervention Ledger

| Element | Thesis anchor | Extension | Falsifier |
|---|---|---|---|
| Op5-Op7 serial line | Operations 5-7 and 320.5 rations/hour | finite intermediate WIP | mass or serial-flow preflight fails |
| Weekly maintenance | 24 hours/week deducted from capacity | allocate slot to one station | action does not change future state |
| R11 | Op5/Op6 breakdown, inadequate maintenance listed as cause | condition-dependent realization | hazard does not increase with condition |
| R14 | defects at Op7, rework through Op6 | condition-dependent defect vulnerability | identical defect burden across condition |
| Shared crew | maintenance/repair capability exists organizationally | one finite crew | overlapping crew work or unequal hours |
| Sensor | none | imperfect condition signal | privileged-schema or calibration test fails |

The exogenous ledger records wear innovations, R11 candidate onsets/targets/base
repair draws, R14 innovations, demand, and sensor errors. The endogenous ledger
records realized failures, defects, maintenance, repair queues, WIP blocking,
starvation and resource hours. Both ledgers are hashed separately.
