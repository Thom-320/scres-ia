# Kaggle Cloud Execution Setup - 2026-06-24

## Local Kaggle Setup

Installed Kaggle CLI from the official GitHub repo because PyPI only exposed
`kaggle==1.7.4.5` while the modern token/OAuth flow requires the newer CLI.

```text
kaggle --version
Kaggle CLI 2.2.2
```

Authentication completed via OAuth:

```text
kaggle auth login
logged in as thomaschisica
```

Verification:

```text
kaggle kernels list --mine --page-size 5
```

This succeeded and listed existing kernels.

Note: `kaggle quota` returned a CLI parsing error on this installation, but
kernel API calls are authenticated and working.

Credentials are stored locally under:

```text
~/.kaggle/credentials.json
```

The file has `0600` permissions and should never be committed.

## Next Logical Cloud Run

Do not launch the full confirmatory retained-vs-reset run yet. The local
`sweep_retcd_v1` pilot showed:

```text
rho=0.3334  ret-reset=+0.03394
rho=0.6000  ret-reset=+0.01008
rho=0.9000  ret-reset=+0.00754
```

Retained learning was positive against reset in all three rho values, but the
H2 dose-response was not present. The next cloud run should therefore be a
diagnostic budget sweep, not a powered confirmatory claim.

Purpose:

```text
Does the retained-minus-reset rho gradient appear as the online learning budget grows?
```

Prepared Kaggle kernel:

```text
kaggle/retcd_rho_budget_sweep
```

It runs `ReT_cd` with Figure 6.2 as the paper-facing downstream-Q source, using
three diagnostic budgets:

```text
budget100: cycles=4, pretrain=300,  online=100
budget300: cycles=6, pretrain=600,  online=300
budget600: cycles=8, pretrain=1000, online=600
```

This is CPU-heavy rather than GPU-heavy because the current environment is a
SimPy DES wrapped by SB3. The GPU will not help much unless the environment is
vectorized or the neural model becomes substantially larger.

## Launch Command

From the repo root:

```text
kaggle kernels push \
  --path kaggle/retcd_rho_budget_sweep \
  --timeout 21600
```

The kernel is private and CPU-only.

## Monitor

```text
kaggle kernels status thomaschisica/scresia-retcd-rho-budget-sweep
kaggle kernels logs thomaschisica/scresia-retcd-rho-budget-sweep
```

## Download Outputs

```text
kaggle kernels output thomaschisica/scresia-retcd-rho-budget-sweep \
  -p outputs/kaggle_retcd_rho_budget_sweep_latest
```

Expected output root inside the kernel:

```text
/kaggle/working/scresia_retcd_rho_budget_sweep_outputs
```

