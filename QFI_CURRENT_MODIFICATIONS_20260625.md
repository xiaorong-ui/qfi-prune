# QFi Current Modifications Summary (2026-06-25)

This document summarizes the current QFi-related code and script changes in the `CDPruner` workspace. It is intended as a compact engineering record for the current state of the project, separate from the long-form experiment context document.

## Scope

This summary covers the current QFi / EA-QFi / CG-EAQF implementation and the two later pilot branches:

- Budget-Calibrated QFi
- Spectral-Filtered QFi

It does not redefine the final method. The current final method remains the confidence-gated QFi configuration:

- `qfid_density_cg_eaqf_final`

## Main Code Changes

### 1. `llava/model/pruners/ec_pruner.py`

Current additions in the pruner include:

- QFi probability-source handling for semantic / CLS-mix / gated CLS-mix paths
- confidence-gated CLS prior fusion used by the current final method
- budget calibration support, default-off
- spectral filter support, default-off
- richer debug bookkeeping through `last_qfid_info`

Relevant QFi extensions now present in the file:

- `EC_QFID_BUDGET_CALIB`
- `EC_QFID_BUDGET_ALPHA`
- `EC_QFID_SPECTRAL_FILTER`
- `EC_QFID_SPECTRAL_GAMMA`
- `EC_QFID_SPECTRAL_EPS`
- `EC_QFID_SPECTRAL_TRACE_NORM`

Important implementation note:

- Budget calibration and spectral filtering are both optional branches and are not part of the default final profile.
- The current final CG-EAQF / QFi path keeps:
  - density kernel
  - fixed `K`
  - fixed depolarization
  - residual QF selection

### 2. `run_gqa_profile.sh`

The profile runner now includes:

- final non-gated EA-QF profile:
  - `qfid_density_eaqf_final`
- final gated QFi profile:
  - `qfid_density_cg_eaqf_final`
- budget pilot profiles:
  - `qfid_density_cg_eaqf_budget_alpha2`
  - `qfid_density_cg_eaqf_budget_alpha3`
  - `qfid_density_cg_eaqf_budget_alpha4`
- spectral pilot profiles:
  - `qfid_density_cg_eaqf_spectral_g05`
  - `qfid_density_cg_eaqf_spectral_g075`
  - `qfid_density_cg_eaqf_spectral_g125`
  - `qfid_density_cg_eaqf_spectral_g150`

### 3. `scripts/v1_5/eval/gqa.sh`

The GQA evaluation script now supports profile-aware output naming for QFi variants.

Current behaviors include:

- baseline CG-EAQF final outputs:
  - `ec_pruner_qfid_cg_eaqf_gate_agreement_beta_base0105_layer_m2_density_depol015_K${TOKEN}`
- budget pilot outputs:
  - `ec_pruner_qfid_cg_eaqf_budget_${EC_QFID_BUDGET_PROFILE}_K${TOKEN}`
- spectral pilot outputs:
  - `ec_pruner_qfid_cg_eaqf_spectral_${EC_QFID_SPECTRAL_PROFILE}_K${TOKEN}`

This avoids collisions between:

- final baseline QFi runs
- budget calibration pilot runs
- spectral pilot runs

### 4. `scripts/summarize_qfi_budget_gqa.sh`

This summarization helper was added for budget pilot runs.

Purpose:

- extract metrics from budget pilot logs
- compare pilot runs against QFi baseline
- produce compact TSV / markdown summary files

### 5. `scripts/summarize_qfi_budget_calib_gqa.sh`

This helper summarizes the later budget calibration experiments, especially the negative pilot branch.

Purpose:

- generate result tables for alpha-based calibration runs
- separate valid full results from dry-run / incomplete entries

### 6. `scripts/summarize_qfi_spectral_gqa.sh`

This helper summarizes spectral pilot results.

Purpose:

- collect spectral run logs from `logs/qfi_spectral_gqa`
- compare each gamma against the QFi baseline
- recompute metrics and completion status from the underlying log files instead of trusting stale TSV rows
- generate:
  - `qfi_spectral_results.tsv`
  - `qfi_spectral_results.md`

## Experiment State Reflected by the Current Code

### Final method

The current final method remains:

- `qfid_density_cg_eaqf_final`

This corresponds to:

- CLS-mixed probability
- agreement-based gate
- density-state QFi selection
- fixed `K`

### Budget calibration branch

Status:

- implemented as a pilot branch
- default-off
- not part of the final method

Interpretation:

- this branch exists for experiment reproducibility and negative-result tracking
- it should not be treated as enabled by default in final QFi runs

### Spectral branch

Status:

- implemented as a pilot branch
- default-off
- currently only three named spectral profiles are exposed in the profile runner:
  - `g05`
  - `g075`
  - `g125`

Interpretation:

- spectral filtering is available for controlled pilot studies
- it is not currently promoted to the final QFi configuration

## Files Most Directly Affected by QFi Work

Core implementation:

- `llava/model/pruners/ec_pruner.py`

Profile / runner layer:

- `run_gqa_profile.sh`
- `scripts/v1_5/eval/gqa.sh`

QFi analysis helpers:

- `scripts/summarize_qfi_budget_gqa.sh`
- `scripts/summarize_qfi_budget_calib_gqa.sh`
- `scripts/summarize_qfi_spectral_gqa.sh`

Long-form experiment record:

- `EC_PRUNER_EXPERIMENT_CONTEXT.md`

## Known Operational Notes

### 1. `g150` is not a profile

This note is obsolete after the current script refresh.

The profile runner now exposes `qfid_density_cg_eaqf_spectral_g150`.

As a result:

- `gamma=1.5` can now be launched through the normal profile runner
- older `Unknown profile` logs for `g150` should be treated as stale pre-fix artifacts

### 2. Final QFi profiles still disable pilot branches by default

The stable final profiles keep pilot branches off:

- `qfid_density_depol`
- `qfid_density_eaqf_final`
- `qfid_density_cg_eaqf_final`

In practice this means:

- budget calibration is off in final profiles
- spectral filtering is off in final profiles

### 3. Spectral and budget pilots should be interpreted separately

Both branches are present in code, but they should be understood as:

- exploratory branches
- not part of the main final QFi claim unless explicitly re-selected later

## Recommended Usage

For final reproducible QFi runs, use:

- `qfid_density_cg_eaqf_final`

For spectral pilot runs already supported by the profile runner, use:

- `qfid_density_cg_eaqf_spectral_g05`
- `qfid_density_cg_eaqf_spectral_g075`
- `qfid_density_cg_eaqf_spectral_g125`
- `qfid_density_cg_eaqf_spectral_g150`

## Summary

The current workspace contains:

- a stable final QFi / CG-EAQF implementation
- a budget calibration pilot branch, kept for negative-result reproducibility
- a spectral pilot branch, currently exposed for `gamma = 0.5 / 0.75 / 1.25 / 1.5`

The key practical point is:

- final QFi logic remains intact and default pilot branches remain disabled unless explicitly selected.
