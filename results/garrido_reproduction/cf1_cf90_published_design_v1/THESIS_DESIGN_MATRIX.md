# Thesis Design Matrix Report

Created UTC: `2026-07-29T05:01:09.751073+00:00`
Overall status: `PASS`

## Summary

| metric | value |
|---|---:|
| row_count | 90 |
| mismatch_count | 0 |
| family_counts | `{"capacity": 30, "inventory": 30, "risk_r1": 10, "risk_r2": 10, "risk_r3": 10}` |
| horizon_year_counts | `{"10": 54, "20": 36}` |
| inventory_period_counts | `{"1344": 6, "168": 6, "336": 6, "504": 6, "672": 6}` |
| shift_counts | `{"1": 69, "2": 11, "3": 10}` |

## Design Rows

| Cfi | status | family | source_cfi | risks | risk_overrides | shifts | I period | horizon_hours | failed_fields |
|---:|---|---|---:|---|---|---:|---:|---:|---|
| 1 | MATCH | risk_r1 | 1 | R11,R12,R13,R14 | `{"R11": "current", "R12": "current", "R13": "increased", "R14": "increased"}` | 1 |  | 161280 |  |
| 2 | MATCH | risk_r1 | 2 | R11,R12,R13,R14 | `{"R11": "current", "R12": "increased", "R13": "current", "R14": "current"}` | 1 |  | 161280 |  |
| 3 | MATCH | risk_r1 | 3 | R11,R12,R13,R14 | `{"R11": "increased", "R12": "current", "R13": "increased", "R14": "increased"}` | 1 |  | 80640 |  |
| 4 | MATCH | risk_r1 | 4 | R11,R12,R13,R14 | `{"R11": "increased", "R12": "increased", "R13": "increased", "R14": "current"}` | 1 |  | 80640 |  |
| 5 | MATCH | risk_r1 | 5 | R11,R12,R13,R14 | `{"R11": "current", "R12": "current", "R13": "increased", "R14": "current"}` | 1 |  | 80640 |  |
| 6 | MATCH | risk_r1 | 6 | R11,R12,R13,R14 | `{"R11": "increased", "R12": "increased", "R13": "current", "R14": "increased"}` | 1 |  | 80640 |  |
| 7 | MATCH | risk_r1 | 7 | R11,R12,R13,R14 | `{"R11": "increased", "R12": "current", "R13": "current", "R14": "increased"}` | 1 |  | 80640 |  |
| 8 | MATCH | risk_r1 | 8 | R11,R12,R13,R14 | `{"R11": "increased", "R12": "current", "R13": "current", "R14": "current"}` | 1 |  | 80640 |  |
| 9 | MATCH | risk_r1 | 9 | R11,R12,R13,R14 | `{"R11": "current", "R12": "increased", "R13": "increased", "R14": "increased"}` | 1 |  | 80640 |  |
| 10 | MATCH | risk_r1 | 10 | R11,R12,R13,R14 | `{"R11": "current", "R12": "increased", "R13": "current", "R14": "increased"}` | 1 |  | 80640 |  |
| 11 | MATCH | risk_r2 | 11 | R21,R22,R23,R24 | `{"R21": "increased", "R22": "current", "R23": "increased", "R24": "increased"}` | 1 |  | 80640 |  |
| 12 | MATCH | risk_r2 | 12 | R21,R22,R23,R24 | `{"R21": "increased", "R22": "current", "R23": "current", "R24": "current"}` | 1 |  | 80640 |  |
| 13 | MATCH | risk_r2 | 13 | R21,R22,R23,R24 | `{"R21": "increased", "R22": "increased", "R23": "current", "R24": "increased"}` | 1 |  | 80640 |  |
| 14 | MATCH | risk_r2 | 14 | R21,R22,R23,R24 | `{"R21": "increased", "R22": "increased", "R23": "increased", "R24": "current"}` | 1 |  | 80640 |  |
| 15 | MATCH | risk_r2 | 15 | R21,R22,R23,R24 | `{"R21": "current", "R22": "current", "R23": "increased", "R24": "increased"}` | 1 |  | 80640 |  |
| 16 | MATCH | risk_r2 | 16 | R21,R22,R23,R24 | `{"R21": "current", "R22": "increased", "R23": "current", "R24": "current"}` | 1 |  | 80640 |  |
| 17 | MATCH | risk_r2 | 17 | R21,R22,R23,R24 | `{"R21": "current", "R22": "increased", "R23": "increased", "R24": "current"}` | 1 |  | 80640 |  |
| 18 | MATCH | risk_r2 | 18 | R21,R22,R23,R24 | `{"R21": "current", "R22": "current", "R23": "increased", "R24": "current"}` | 1 |  | 80640 |  |
| 19 | MATCH | risk_r2 | 19 | R21,R22,R23,R24 | `{"R21": "current", "R22": "increased", "R23": "current", "R24": "increased"}` | 1 |  | 80640 |  |
| 20 | MATCH | risk_r2 | 20 | R21,R22,R23,R24 | `{"R21": "increased", "R22": "increased", "R23": "increased", "R24": "increased"}` | 1 |  | 80640 |  |
| 21 | MATCH | risk_r3 | 21 | R3 | `{"R3": "current"}` | 1 |  | 161280 |  |
| 22 | MATCH | risk_r3 | 22 | R3 | `{"R3": "increased"}` | 1 |  | 161280 |  |
| 23 | MATCH | risk_r3 | 23 | R3 | `{"R3": "increased"}` | 1 |  | 161280 |  |
| 24 | MATCH | risk_r3 | 24 | R3 | `{"R3": "increased"}` | 1 |  | 161280 |  |
| 25 | MATCH | risk_r3 | 25 | R3 | `{"R3": "increased"}` | 1 |  | 161280 |  |
| 26 | MATCH | risk_r3 | 26 | R3 | `{"R3": "current"}` | 1 |  | 161280 |  |
| 27 | MATCH | risk_r3 | 27 | R3 | `{"R3": "current"}` | 1 |  | 161280 |  |
| 28 | MATCH | risk_r3 | 28 | R3 | `{"R3": "current"}` | 1 |  | 161280 |  |
| 29 | MATCH | risk_r3 | 29 | R3 | `{"R3": "increased"}` | 1 |  | 161280 |  |
| 30 | MATCH | risk_r3 | 30 | R3 | `{"R3": "current"}` | 1 |  | 161280 |  |
| 31 | MATCH | inventory | 1 | R11,R12,R13,R14 | `{"R11": "current", "R12": "current", "R13": "increased", "R14": "increased"}` | 1 | 504.0 | 161280 |  |
| 32 | MATCH | inventory | 2 | R11,R12,R13,R14 | `{"R11": "current", "R12": "increased", "R13": "current", "R14": "current"}` | 1 | 336.0 | 161280 |  |
| 33 | MATCH | inventory | 3 | R11,R12,R13,R14 | `{"R11": "increased", "R12": "current", "R13": "increased", "R14": "increased"}` | 1 | 168.0 | 80640 |  |
| 34 | MATCH | inventory | 4 | R11,R12,R13,R14 | `{"R11": "increased", "R12": "increased", "R13": "increased", "R14": "current"}` | 1 | 1344.0 | 80640 |  |
| 35 | MATCH | inventory | 5 | R11,R12,R13,R14 | `{"R11": "current", "R12": "current", "R13": "increased", "R14": "current"}` | 1 | 336.0 | 80640 |  |
| 36 | MATCH | inventory | 6 | R11,R12,R13,R14 | `{"R11": "increased", "R12": "increased", "R13": "current", "R14": "increased"}` | 1 | 1344.0 | 80640 |  |
| 37 | MATCH | inventory | 7 | R11,R12,R13,R14 | `{"R11": "increased", "R12": "current", "R13": "current", "R14": "increased"}` | 1 | 672.0 | 80640 |  |
| 38 | MATCH | inventory | 8 | R11,R12,R13,R14 | `{"R11": "increased", "R12": "current", "R13": "current", "R14": "current"}` | 1 | 672.0 | 80640 |  |
| 39 | MATCH | inventory | 9 | R11,R12,R13,R14 | `{"R11": "current", "R12": "increased", "R13": "increased", "R14": "increased"}` | 1 | 168.0 | 80640 |  |
| 40 | MATCH | inventory | 10 | R11,R12,R13,R14 | `{"R11": "current", "R12": "increased", "R13": "current", "R14": "increased"}` | 1 | 504.0 | 80640 |  |
| 41 | MATCH | inventory | 11 | R21,R22,R23,R24 | `{"R21": "increased", "R22": "current", "R23": "increased", "R24": "increased"}` | 1 | 1344.0 | 80640 |  |
| 42 | MATCH | inventory | 12 | R21,R22,R23,R24 | `{"R21": "increased", "R22": "current", "R23": "current", "R24": "current"}` | 1 | 336.0 | 80640 |  |
| 43 | MATCH | inventory | 13 | R21,R22,R23,R24 | `{"R21": "increased", "R22": "increased", "R23": "current", "R24": "increased"}` | 1 | 504.0 | 80640 |  |
| 44 | MATCH | inventory | 14 | R21,R22,R23,R24 | `{"R21": "increased", "R22": "increased", "R23": "increased", "R24": "current"}` | 1 | 168.0 | 80640 |  |
| 45 | MATCH | inventory | 15 | R21,R22,R23,R24 | `{"R21": "current", "R22": "current", "R23": "increased", "R24": "increased"}` | 1 | 504.0 | 80640 |  |
| 46 | MATCH | inventory | 16 | R21,R22,R23,R24 | `{"R21": "current", "R22": "increased", "R23": "current", "R24": "current"}` | 1 | 1344.0 | 80640 |  |
| 47 | MATCH | inventory | 17 | R21,R22,R23,R24 | `{"R21": "current", "R22": "increased", "R23": "increased", "R24": "current"}` | 1 | 168.0 | 80640 |  |
| 48 | MATCH | inventory | 18 | R21,R22,R23,R24 | `{"R21": "current", "R22": "current", "R23": "increased", "R24": "current"}` | 1 | 336.0 | 80640 |  |
| 49 | MATCH | inventory | 19 | R21,R22,R23,R24 | `{"R21": "current", "R22": "increased", "R23": "current", "R24": "increased"}` | 1 | 672.0 | 80640 |  |
| 50 | MATCH | inventory | 20 | R21,R22,R23,R24 | `{"R21": "increased", "R22": "increased", "R23": "increased", "R24": "increased"}` | 1 | 672.0 | 80640 |  |
| 51 | MATCH | inventory | 21 | R3 | `{"R3": "current"}` | 1 | 672.0 | 161280 |  |
| 52 | MATCH | inventory | 22 | R3 | `{"R3": "increased"}` | 1 | 1344.0 | 161280 |  |
| 53 | MATCH | inventory | 23 | R3 | `{"R3": "increased"}` | 1 | 672.0 | 161280 |  |
| 54 | MATCH | inventory | 24 | R3 | `{"R3": "increased"}` | 1 | 1344.0 | 161280 |  |
| 55 | MATCH | inventory | 25 | R3 | `{"R3": "increased"}` | 1 | 504.0 | 161280 |  |
| 56 | MATCH | inventory | 26 | R3 | `{"R3": "current"}` | 1 | 504.0 | 161280 |  |
| 57 | MATCH | inventory | 27 | R3 | `{"R3": "current"}` | 1 | 336.0 | 161280 |  |
| 58 | MATCH | inventory | 28 | R3 | `{"R3": "current"}` | 1 | 336.0 | 161280 |  |
| 59 | MATCH | inventory | 29 | R3 | `{"R3": "increased"}` | 1 | 168.0 | 161280 |  |
| 60 | MATCH | inventory | 30 | R3 | `{"R3": "current"}` | 1 | 168.0 | 161280 |  |
| 61 | MATCH | capacity | 1 | R11,R12,R13,R14 | `{"R11": "current", "R12": "current", "R13": "increased", "R14": "increased"}` | 2 |  | 161280 |  |
| 62 | MATCH | capacity | 2 | R11,R12,R13,R14 | `{"R11": "current", "R12": "increased", "R13": "current", "R14": "current"}` | 1 |  | 161280 |  |
| 63 | MATCH | capacity | 3 | R11,R12,R13,R14 | `{"R11": "increased", "R12": "current", "R13": "increased", "R14": "increased"}` | 3 |  | 80640 |  |
| 64 | MATCH | capacity | 4 | R11,R12,R13,R14 | `{"R11": "increased", "R12": "increased", "R13": "increased", "R14": "current"}` | 3 |  | 80640 |  |
| 65 | MATCH | capacity | 5 | R11,R12,R13,R14 | `{"R11": "current", "R12": "current", "R13": "increased", "R14": "current"}` | 1 |  | 80640 |  |
| 66 | MATCH | capacity | 6 | R11,R12,R13,R14 | `{"R11": "increased", "R12": "increased", "R13": "current", "R14": "increased"}` | 2 |  | 80640 |  |
| 67 | MATCH | capacity | 7 | R11,R12,R13,R14 | `{"R11": "increased", "R12": "current", "R13": "current", "R14": "increased"}` | 1 |  | 80640 |  |
| 68 | MATCH | capacity | 8 | R11,R12,R13,R14 | `{"R11": "increased", "R12": "current", "R13": "current", "R14": "current"}` | 2 |  | 80640 |  |
| 69 | MATCH | capacity | 9 | R11,R12,R13,R14 | `{"R11": "current", "R12": "increased", "R13": "increased", "R14": "increased"}` | 3 |  | 80640 |  |
| 70 | MATCH | capacity | 10 | R11,R12,R13,R14 | `{"R11": "current", "R12": "increased", "R13": "current", "R14": "increased"}` | 3 |  | 80640 |  |
| 71 | MATCH | capacity | 11 | R21,R22,R23,R24 | `{"R21": "increased", "R22": "current", "R23": "increased", "R24": "increased"}` | 1 |  | 80640 |  |
| 72 | MATCH | capacity | 12 | R21,R22,R23,R24 | `{"R21": "increased", "R22": "current", "R23": "current", "R24": "current"}` | 3 |  | 80640 |  |
| 73 | MATCH | capacity | 13 | R21,R22,R23,R24 | `{"R21": "increased", "R22": "increased", "R23": "current", "R24": "increased"}` | 2 |  | 80640 |  |
| 74 | MATCH | capacity | 14 | R21,R22,R23,R24 | `{"R21": "increased", "R22": "increased", "R23": "increased", "R24": "current"}` | 3 |  | 80640 |  |
| 75 | MATCH | capacity | 15 | R21,R22,R23,R24 | `{"R21": "current", "R22": "current", "R23": "increased", "R24": "increased"}` | 2 |  | 80640 |  |
| 76 | MATCH | capacity | 16 | R21,R22,R23,R24 | `{"R21": "current", "R22": "increased", "R23": "current", "R24": "current"}` | 3 |  | 80640 |  |
| 77 | MATCH | capacity | 17 | R21,R22,R23,R24 | `{"R21": "current", "R22": "increased", "R23": "increased", "R24": "current"}` | 2 |  | 80640 |  |
| 78 | MATCH | capacity | 18 | R21,R22,R23,R24 | `{"R21": "current", "R22": "current", "R23": "increased", "R24": "current"}` | 1 |  | 80640 |  |
| 79 | MATCH | capacity | 19 | R21,R22,R23,R24 | `{"R21": "current", "R22": "increased", "R23": "current", "R24": "increased"}` | 2 |  | 80640 |  |
| 80 | MATCH | capacity | 20 | R21,R22,R23,R24 | `{"R21": "increased", "R22": "increased", "R23": "increased", "R24": "increased"}` | 1 |  | 80640 |  |
| 81 | MATCH | capacity | 21 | R3 | `{"R3": "current"}` | 1 |  | 161280 |  |
| 82 | MATCH | capacity | 22 | R3 | `{"R3": "increased"}` | 3 |  | 161280 |  |
| 83 | MATCH | capacity | 23 | R3 | `{"R3": "increased"}` | 2 |  | 161280 |  |
| 84 | MATCH | capacity | 24 | R3 | `{"R3": "increased"}` | 3 |  | 161280 |  |
| 85 | MATCH | capacity | 25 | R3 | `{"R3": "increased"}` | 2 |  | 161280 |  |
| 86 | MATCH | capacity | 26 | R3 | `{"R3": "current"}` | 3 |  | 161280 |  |
| 87 | MATCH | capacity | 27 | R3 | `{"R3": "current"}` | 2 |  | 161280 |  |
| 88 | MATCH | capacity | 28 | R3 | `{"R3": "current"}` | 1 |  | 161280 |  |
| 89 | MATCH | capacity | 29 | R3 | `{"R3": "increased"}` | 2 |  | 161280 |  |
| 90 | MATCH | capacity | 30 | R3 | `{"R3": "current"}` | 1 |  | 161280 |  |
