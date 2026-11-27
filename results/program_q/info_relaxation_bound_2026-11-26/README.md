# Information-relaxation bound for Program Q (2026-11-26)

Upper bound on the value of *any* non-anticipative policy (RL included) in Program Q
(Discrete(4) actions, 8 weeks; cells rho75_share90, rho90_share75, rho90_share90), computed on
the burned calibration tapes 7480001–7480048. Development result, not a confirmation.

## Result (M = 64 inner samples; `penalized/full_M64.json`)
| cell | perfect-information gap U0 − V_CL | penalized gap U_pen − J(κ) | 95 % CI |
|---|---|---|---|
| rho75_share90 | 0.081 | 0.0045 | [0.0035, 0.0056] |
| rho90_share75 | 0.049 | 0.0030 | [0.0022, 0.0038] |
| rho90_share90 | 0.074 | 0.0068 | [0.0045, 0.0094] |

The stop rule (UCB95 of the penalized gap < 0.01) holds in all three cells: no non-anticipative
policy can beat the classical controller κ by more than 0.0094 in expected value on these tapes.

## Method
- `zero_penalty/`: perfect-information (clairvoyant) bound U0, exhaustive 4^8 tree per tape
  (`scripts/run_all.py`, `frontier_tape.py`, `analyze_bound.py`; summary in `bound_result.json`, notes in `COTA.md`).
- `penalized/`: Brown–Smith–Sun penalty built from the classical controller κ's continuation value,
  estimated by nested Monte Carlo from the exact Bayesian filter (`scripts/bss_inner.py`), then
  J = Σ_t [Q_t − V_t] by exact DP over the tree (`scripts/bss_bound.py`). The finite-M bound is valid
  and biased upwards; `bss_extrap.py` extrapolates M → ∞ as a diagnostic only. Validity checks in `VALIDEZ.md`.
- Inner samples: 8 replicates × M=8, reps 0–3 on the VPS (`scripts/run_reps.sh`), 4–7 on Kaggle.

## Reproduce
Inputs: this repo at a10e99a (`results/program_q/power_preopen_v5_20260718/artifacts/classical_10_cache_v1.npz`,
`results/program_o/ret_only_learner_v1/calibration_run/result.json`). The npz intermediates (46 MB
frontier, 550 MB inner) are not committed; their SHA-256 are in `zero_penalty/frontier_manifest.sha256`
and `penalized/inner_manifest.sha256`.

    SCRES_COTA_DIR=<dir with frontier/> python penalized/scripts/bss_bound.py <inner dir> 8 0-7 out.json \
        0:7480001-7480048 1:7480001-7480048 2:7480001-7480048

## SHA-256 of the committed files
```
f4c218995169ecc3aee1e5592fd4d1eb4d483a2249afbca2ee357ba299df20eb  ./penalized/VALIDEZ.md
584763bfac299e00185d1708fced9a65829fc1a89dd368d3cfcefbf4f8c3d080  ./penalized/extrap_M32.json
a4fde02794bc95f16f88e9eaca6b61a9fe15f662e7867be640aefe5acc63d889  ./penalized/extrap_M64.json
f873ff30228714bc6c6920fe48843a6d6a285b124d9bd6300df3f7f89f267825  ./penalized/full_M32.json
db061bc71df08b048fdc93aabd858e05dc0c307521c5adfdca1d740e2e28d51b  ./penalized/full_M64.json
c35c81d8dec6448980e0a707ed18e69dc54ad5ce9a070adf2b7a83a72d31eeec  ./penalized/full_M8_rep0.json
608d49acff8fd32101d2ec19dfa90b598affc49e26edd678c122479cd548d108  ./penalized/inner_manifest.sha256
d3b6acbdcd46e55d399244a406d902cb6c9f25c16d1f012fc913ee4689d44453  ./penalized/pilot_M8.json
fba0a0ccee34c00d3f17c475614a984eeab31cb0e432f4a843e618b218030a1e  ./penalized/scripts/bss_bound.py
1b3e930fe145e3bac196c4265e44e4d78287f457208d9e8772d92cc3dd6674cc  ./penalized/scripts/bss_convergence.py
a797d3be165f5dfbd64675782be994d7d81e5e12b3aadc4a56a2fd117a486d61  ./penalized/scripts/bss_extrap.py
30989a57a7cdcca1240705d40a9b892408828af5f0e1d33cbffdfe50c8727ea5  ./penalized/scripts/bss_inner.py
6dc3885a420101ad93367b6647bae7bf39efca686787836c60c7d55fb1b3453e  ./penalized/scripts/bss_lib.py
c6d30da20f3d0faab30865c9e05b6a76e0a367b7bdc085408dd304646b942c9f  ./penalized/scripts/run_reps.sh
fdb09928643db07f2f18b007705dca7aa699eac8ef7dabf3440fad925fe34582  ./penalized/scripts/step0_direct_audit.py
fae7515ac374e51493e4d20d20290618b58f2792f3cf003bb6ece00c1307ff63  ./penalized/scripts/step1_validate.py
6568a39cc78b3af7c5753db359a9ef8e0977bd617f3c1a8f29092fe0cf236423  ./penalized/scripts/step1b_filter_exact.py
a4ccc4ad72958b59a4adaf07aef7fa770820167c45660c0f788c5dc00890ec95  ./penalized/step0_direct_audit.json
6471504d487bd8b27e0269efa659494c5f372a3b59ccff134d518cf430efe56c  ./penalized/step1_validate.json
5ea7bdd5215c11942fc2a5acb074f90edb01f75afc1d14d690d46c725f0cdf63  ./penalized/step1b_filter_exact.json
cbcd71d6468f43f2205701f2c097af525791978743ab098f56a2d048039041ef  ./zero_penalty/COTA.md
c6571d293c8426f6f59b0ec8fed216bc06915676e53402bb6f82c5ebd122d9a5  ./zero_penalty/bound_result.json
871772f05bfce09a826779eac027c2b4408eb9fc88c7a56f478aa08133436c13  ./zero_penalty/frontier_manifest.sha256
cb946c3e5ab35d79585cd227ae03f98e1573e1baf05f02175f62675829ddec2e  ./zero_penalty/scripts/analyze_bound.py
1942f75b86b5b3bf678eca0bb2f97ff97853b273dc3990323ad0ebb56eef9508  ./zero_penalty/scripts/frontier_tape.py
45c049c299add5ed9094c4ab74ee733b8d90da45e18672d63a45f00863bb39ef  ./zero_penalty/scripts/run_all.py
```
