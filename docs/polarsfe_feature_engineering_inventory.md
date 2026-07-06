# PolarsFE Feature Engineering Inventory

PolarsFE is a compact Python package built around Polars feature engineering helpers. Existing modules cover numeric transforms, categorical encodings, calendar/holiday features, dataset partitioning/imputation, and window/cross-row operations.

Model-based features are deferred from this modernization pass.

| Public function | File | Family | Purpose | Eager/lazy behavior | Train/scoring safety | Dependencies | Disposition |
|---|---|---|---|---|---|---|---|
| `standardize` | `PolarsFE/numeric.py` | Numeric | Standardize numeric columns globally or by groups; can return score table. | Eager Polars | Yes when score table is saved. | polars | Keep legacy; vNext wraps concept in plan/spec. |
| `percent_rank` | `PolarsFE/numeric.py` | Numeric | Percent-rank transform with apply/backtransform support. | Eager Polars with asof joins | Yes when score table is saved. | polars, math | Keep legacy; benchmark. |
| `numeric_transform` | `PolarsFE/numeric.py` | Numeric | General numeric transforms. | Eager Polars | Mixed by transform/settings. | polars, numpy | Keep legacy; benchmark/wrap later. |
| `dummy_variables` | `PolarsFE/character.py` | Categorical | One-hot/dummy variables. | Eager Polars | Needs saved levels for scoring safety. | polars | Keep legacy; vNext adds explicit top-N/rare/unseen spec. |
| `categorical_encoding` | `PolarsFE/character.py` | Categorical | Categorical encodings. | Eager Polars | Mixed. | polars | Keep legacy; benchmark. |
| `calendar_features` | `PolarsFE/calendar.py` | Calendar | Calendar unit extraction. | Eager Polars | Yes for deterministic date units. | polars | Keep legacy; vNext wraps core units. |
| `cyclic_features` | `PolarsFE/calendar.py` | Calendar | Cyclical encodings for calendar fields. | Eager Polars | Yes. | polars, math/numpy | Keep legacy; vNext candidate. |
| `holiday_features` | `PolarsFE/calendar.py` | Calendar | Holiday/pre/post holiday features. | Eager Polars | Yes with fixed country/calendar configuration. | polars, holidays | Keep legacy; benchmark/wrap later. |
| `partition_random`, `partition_time`, `partition_timeseries`, `stratified_sample`, `impute_missing` | `PolarsFE/datasets.py` | Model prep | Partitioning, sampling, imputation helpers. | Eager Polars | Mixed; partition specs should be saved for reproducibility. | polars, numpy | Keep legacy; separate model-prep design later. |
| `lags`, `rolling_features`, `differences`, `adstock` | `PolarsFE/window.py` | Cross-row / time | Lag, rolling, diff, and adstock transforms. | Eager Polars | Requires explicit sort/group contracts. | polars, numpy, scipy | Benchmark-first; vNext architecture placeholder. |
| `polars_feature_plan`, `polars_fit_feature_plan`, `polars_transform_feature_plan`, `polars_fit_transform_feature_plan`, `generate_polars_feature_engineering_artifacts` | `PolarsFE/vnext.py` | vNext plan/spec | Scoring-safe plan layer across numeric, categorical, calendar, text, missingness, and interactions. | Eager Polars | Yes. | polars | New vNext entry points. |

## Benchmark Targets

Benchmarks should compare existing PolarsFE functions, PolarsFE vNext, direct Polars eager, direct Polars lazy, pandas, and DuckDB where appropriate. Spark is out of scope for this phase.
