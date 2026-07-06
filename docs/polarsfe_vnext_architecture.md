# PolarsFE vNext Architecture

PolarsFE vNext adds a single plan/spec layer without removing existing public functions. The design mirrors Rodeo vNext where useful so both ecosystems can be benchmarked under comparable workloads.

## Public API

- `polars_feature_plan()`
- `polars_fit_feature_plan()`
- `polars_transform_feature_plan()`
- `polars_fit_transform_feature_plan()`
- `generate_polars_feature_engineering_artifacts()`

## Supported Families

| Family | vNext status | Notes |
|---|---|---|
| Numeric | Implemented | `log1p`, `sqrt`, `standardize`, `winsorize`. |
| Categorical | Implemented | Top-N one-hot encoding with rare and unseen level handling. |
| Calendar | Implemented | Year, month, day, weekday, week, quarter, weekend flag. |
| Text | Implemented | Lightweight counts and ratios only. |
| Missingness | Implemented | Binary missingness indicators. |
| Interactions | Implemented | Numeric x numeric, categorical x numeric, categorical x categorical with feature caps. |
| Cross-row | Deferred | Existing lag/rolling/diff functions remain benchmarks until sort/group contracts are finalized. |
| Model prep | Deferred | Existing dataset helpers remain available, but are not merged into the first plan layer. |
| Model-Based Features | Deferred | Requires a separate leakage-safe redesign and modern dependency review. |

## Fitted Spec

The fitted spec stores numeric parameters, categorical levels, generated feature manifest, diagnostics, warnings, interaction definitions, and fit metadata. It is intended to be reused on scoring data without recomputing training-only statistics.

## Artifact Generator

`generate_polars_feature_engineering_artifacts()` returns app-agnostic dictionaries containing:

- Overview text.
- Config table.
- Feature manifest.
- Diagnostics.
- Engineered data summary.
- Optional benchmark summary.
- Engineered data and fitted plan in `value`.

## Benchmark Alignment

Full performance comparisons live in the Benchmarks repo. vNext defaults should stay conservative until benchmark evidence supports workload-specific implementation choices.
