# PolarsFE Model Prep Inventory

## Current Surface

PolarsFE has legacy dataset helpers in `PolarsFE/datasets.py`, including random, time, timeseries, and stratified sampling utilities.

Those functions remain available. The vNext model-prep layer adds a scoring-safe plan/spec/artifact contract without replacing the legacy helpers.

## vNext Scope

The vNext model-prep layer adds:

- `polars_partition_plan()`
- `polars_fit_partition_plan()`
- `polars_apply_partition_plan()`
- `polars_create_folds()`
- `generate_polars_model_prep_artifacts()`

Supported vNext partition modes:

- random train/test or train/validation/test
- stratified train/test or train/validation/test
- grouped partitions that keep groups together
- time partitions that keep earlier rows in earlier partitions
- random, stratified, and grouped k-fold assignments

## Contract

The fitted plan stores:

- original plan
- row-level partition assignments
- row-level fold assignments
- partition manifest
- fold manifest
- diagnostics
- warnings

The artifact generator returns a lightweight, app-agnostic result that can be consumed by benchmark harnesses or future reporting systems.

## Leakage Safety

Grouped partitions preserve group integrity. Time partitions use sorted date order. Stratified partitions use the target only to balance assignment, not to create features.

## Out Of Scope

- model training
- model-based features
- target encoding
- WOE / credibility encoding
- broad recipe frameworks
- large-benchmark optimization work
