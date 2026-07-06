"""PolarsFE vNext plan/spec feature engineering helpers.

The vNext layer is intentionally additive. Existing PolarsFE modules remain
legacy/public APIs, while these functions provide a single scoring-safe
fit/transform contract for numeric, categorical, calendar, text, missingness,
and interaction features.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import random
from typing import Any, Dict, Iterable, List, Optional

import polars as pl


def _as_list(value: Optional[Iterable[str]]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(x) for x in value]


def polars_feature_plan(
    numeric: Optional[Dict[str, Any]] = None,
    categorical: Optional[Dict[str, Any]] = None,
    calendar: Optional[Dict[str, Any]] = None,
    text: Optional[Dict[str, Any]] = None,
    missingness: Optional[Dict[str, Any]] = None,
    interactions: Optional[Dict[str, Any]] = None,
    cross_row: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a PolarsFE vNext feature plan."""
    return {
        "numeric": numeric or {
            "columns": [],
            "transforms": ["log1p", "sqrt", "standardize", "winsorize"],
            "winsorize_probs": [0.01, 0.99],
        },
        "categorical": categorical or {
            "columns": [],
            "top_n": 10,
            "rare_level": "__RARE__",
            "unseen_level": "__UNSEEN__",
            "one_hot": True,
            "keep_original": True,
        },
        "calendar": calendar or {
            "columns": [],
            "features": ["year", "month", "day", "wday", "week", "quarter", "is_weekend"],
        },
        "text": text or {
            "columns": [],
            "features": ["char_count", "word_count", "digit_count", "punct_count", "upper_ratio", "blank"],
        },
        "missingness": missingness or {"columns": [], "suffix": "_is_missing"},
        "interactions": interactions or {
            "numeric_pairs": [],
            "categorical_numeric": [],
            "categorical_pairs": [],
            "max_features": 50,
        },
        "cross_row": cross_row or {"enabled": False},
        "metadata": metadata or {},
        "created_at": datetime.utcnow().isoformat(timespec="seconds"),
    }


def _existing_columns(data: pl.DataFrame, columns: Iterable[str]) -> List[str]:
    return [col for col in _as_list(columns) if col in data.columns]


def _manifest_row(feature: str, source_column: str, family: str, transform: str) -> Dict[str, Any]:
    return {
        "feature": feature,
        "source_column": source_column,
        "family": family,
        "transform": transform,
        "scoring_safe": True,
    }


def _make_feature_name(*parts: str) -> str:
    return "_".join(str(part).replace(" ", "_").replace("/", "_") for part in parts if part is not None)


def _numeric_fit_stats(data: pl.DataFrame, columns: List[str], probs: Iterable[float]) -> Dict[str, Dict[str, Any]]:
    """Collect numeric fit statistics for all columns in one Polars projection."""
    if not columns:
        return {}
    lower_prob, upper_prob = [float(x) for x in probs]
    exprs: List[pl.Expr] = []
    for col in columns:
        exprs.extend(
            [
                pl.col(col).mean().alias(f"{col}__mean"),
                pl.col(col).std().alias(f"{col}__sd"),
                pl.col(col).quantile(lower_prob).alias(f"{col}__lower"),
                pl.col(col).quantile(upper_prob).alias(f"{col}__upper"),
            ]
        )
    row = data.select(exprs).to_dicts()[0]
    return {
        col: {
            "mean": row.get(f"{col}__mean"),
            "sd": row.get(f"{col}__sd"),
            "lower": row.get(f"{col}__lower"),
            "upper": row.get(f"{col}__upper"),
        }
        for col in columns
    }


def polars_fit_feature_plan(data: pl.DataFrame, plan: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Fit a PolarsFE vNext feature plan and return a reusable fitted spec."""
    plan = plan or polars_feature_plan()
    warnings: List[str] = []
    manifest: List[Dict[str, Any]] = []

    requested = set()
    for family in ("numeric", "categorical", "calendar", "text", "missingness"):
        requested.update(_as_list(plan.get(family, {}).get("columns")))
    missing_requested = sorted(requested.difference(data.columns))
    if missing_requested:
        warnings.append("Requested columns not found: " + ", ".join(missing_requested))

    numeric_specs: Dict[str, Dict[str, Any]] = {}
    numeric_columns = _existing_columns(data, plan["numeric"].get("columns"))
    transforms = _as_list(plan["numeric"].get("transforms"))
    probs = plan["numeric"].get("winsorize_probs", [0.01, 0.99])
    numeric_stats = _numeric_fit_stats(data, numeric_columns, probs)
    for col in numeric_columns:
        numeric_specs[col] = {**numeric_stats[col], "transforms": transforms}
        for transform in transforms:
            manifest.append(_manifest_row(f"{col}_{transform}", col, "numeric", transform))

    categorical_specs: Dict[str, Dict[str, Any]] = {}
    top_n = int(plan["categorical"].get("top_n", 10))
    rare_level = str(plan["categorical"].get("rare_level", "__RARE__"))
    unseen_level = str(plan["categorical"].get("unseen_level", "__UNSEEN__"))
    for col in _existing_columns(data, plan["categorical"].get("columns")):
        counts = (
            data.select(pl.col(col).cast(pl.Utf8).alias(col))
            .group_by(col)
            .len()
            .sort("len", descending=True)
        )
        levels = [x for x in counts[col].head(top_n).to_list() if x is not None]
        levels = list(dict.fromkeys(levels + [rare_level, unseen_level]))
        categorical_specs[col] = {"levels": levels, "rare_level": rare_level, "unseen_level": unseen_level}
        for level in levels:
            manifest.append(_manifest_row(_make_feature_name(col, str(level)), col, "categorical", "one_hot"))

    calendar_columns = _existing_columns(data, plan["calendar"].get("columns"))
    for col in calendar_columns:
        for feature in _as_list(plan["calendar"].get("features")):
            manifest.append(_manifest_row(f"{col}_{feature}", col, "calendar", feature))

    text_columns = _existing_columns(data, plan["text"].get("columns"))
    for col in text_columns:
        for feature in _as_list(plan["text"].get("features")):
            manifest.append(_manifest_row(f"{col}_{feature}", col, "text", feature))

    missingness_columns = _existing_columns(data, plan["missingness"].get("columns"))
    suffix = str(plan["missingness"].get("suffix", "_is_missing"))
    for col in missingness_columns:
        manifest.append(_manifest_row(f"{col}{suffix}", col, "missingness", "is_missing"))

    interaction_specs, interaction_manifest, interaction_warnings = _fit_interactions(data, plan.get("interactions", {}))
    manifest.extend(interaction_manifest)
    warnings.extend(interaction_warnings)

    diagnostics = [
        {"check": "input_rows", "status": "ok", "detail": str(data.height)},
        {"check": "generated_features", "status": "ok", "detail": str(len(manifest))},
        {
            "check": "cross_row",
            "status": "deferred" if plan.get("cross_row", {}).get("enabled") else "skipped",
            "detail": "Cross-row vNext wrappers are deferred until benchmark contracts are settled.",
        },
    ]

    return {
        "plan": plan,
        "numeric_specs": numeric_specs,
        "categorical_specs": categorical_specs,
        "calendar_columns": calendar_columns,
        "text_columns": text_columns,
        "missingness_columns": missingness_columns,
        "interaction_specs": interaction_specs,
        "feature_manifest": pl.DataFrame(manifest) if manifest else pl.DataFrame(schema={
            "feature": pl.Utf8,
            "source_column": pl.Utf8,
            "family": pl.Utf8,
            "transform": pl.Utf8,
            "scoring_safe": pl.Boolean,
        }),
        "diagnostics": pl.DataFrame(diagnostics),
        "warnings": sorted(set(warnings)),
        "fitted_at": datetime.utcnow().isoformat(timespec="seconds"),
    }


def _fit_interactions(data: pl.DataFrame, interactions: Dict[str, Any]):
    max_features = int(interactions.get("max_features", 50))
    specs = {"numeric_pairs": [], "categorical_numeric": [], "categorical_pairs": []}
    manifest: List[Dict[str, Any]] = []
    warnings: List[str] = []
    n_features = 0

    for pair in interactions.get("numeric_pairs", []):
        pair = _as_list(pair)
        if len(pair) == 2 and all(col in data.columns for col in pair) and n_features < max_features:
            specs["numeric_pairs"].append(pair)
            manifest.append(_manifest_row(f"{pair[0]}_x_{pair[1]}", ",".join(pair), "interaction", "numeric_x_numeric"))
            n_features += 1

    for item in interactions.get("categorical_numeric", []):
        cat_col = item.get("categorical") if isinstance(item, dict) else item[0]
        num_col = item.get("numeric") if isinstance(item, dict) else item[1]
        if cat_col in data.columns and num_col in data.columns:
            remaining = max_features - n_features
            levels = (
                data.select(pl.col(cat_col).cast(pl.Utf8).alias(cat_col))
                .unique()
                .head(remaining)
                .get_column(cat_col)
                .to_list()
            )
            for level in levels:
                if n_features >= max_features:
                    break
                specs["categorical_numeric"].append({"categorical": cat_col, "numeric": num_col, "level": level})
                manifest.append(_manifest_row(_make_feature_name(cat_col, str(level), "x", num_col), f"{cat_col},{num_col}", "interaction", "categorical_x_numeric"))
                n_features += 1

    for pair in interactions.get("categorical_pairs", []):
        pair = _as_list(pair)
        if len(pair) == 2 and all(col in data.columns for col in pair) and n_features < max_features:
            specs["categorical_pairs"].append(pair)
            manifest.append(_manifest_row(f"{pair[0]}_x_{pair[1]}", ",".join(pair), "interaction", "categorical_x_categorical"))
            n_features += 1

    if n_features >= max_features:
        warnings.append(f"Interaction feature cap reached: {max_features}")
    return specs, manifest, warnings


def polars_transform_feature_plan(data: pl.DataFrame, fitted_plan: Dict[str, Any]) -> pl.DataFrame:
    """Transform data with a fitted PolarsFE vNext feature plan."""
    column_set = set(data.columns)
    plan = fitted_plan["plan"]
    exprs: List[pl.Expr] = []

    for col, spec in fitted_plan["numeric_specs"].items():
        if col not in column_set:
            continue
        if "log1p" in spec["transforms"]:
            exprs.append(pl.when(pl.col(col) > -1).then(pl.col(col).log1p()).otherwise(None).alias(f"{col}_log1p"))
        if "sqrt" in spec["transforms"]:
            exprs.append(pl.when(pl.col(col) >= 0).then(pl.col(col).sqrt()).otherwise(None).alias(f"{col}_sqrt"))
        if "standardize" in spec["transforms"]:
            sd = spec["sd"] or 1.0
            exprs.append(((pl.col(col) - spec["mean"]) / sd).alias(f"{col}_standardize"))
        if "winsorize" in spec["transforms"]:
            exprs.append(pl.col(col).clip(spec["lower"], spec["upper"]).alias(f"{col}_winsorize"))

    for col, spec in fitted_plan["categorical_specs"].items():
        if col not in column_set:
            continue
        mapped = (
            pl.when(pl.col(col).cast(pl.Utf8).is_in(spec["levels"]))
            .then(pl.col(col).cast(pl.Utf8))
            .otherwise(pl.lit(spec["unseen_level"]))
        )
        for level in spec["levels"]:
            exprs.append((mapped == pl.lit(level)).cast(pl.Int8).alias(_make_feature_name(col, str(level))))

    for col in fitted_plan["calendar_columns"]:
        if col not in column_set:
            continue
        d = pl.col(col).cast(pl.Date)
        features = _as_list(plan["calendar"].get("features"))
        if "year" in features:
            exprs.append(d.dt.year().alias(f"{col}_year"))
        if "month" in features:
            exprs.append(d.dt.month().alias(f"{col}_month"))
        if "day" in features:
            exprs.append(d.dt.day().alias(f"{col}_day"))
        if "wday" in features:
            exprs.append(d.dt.weekday().alias(f"{col}_wday"))
        if "week" in features:
            exprs.append(d.dt.week().alias(f"{col}_week"))
        if "quarter" in features:
            exprs.append(d.dt.quarter().alias(f"{col}_quarter"))
        if "is_weekend" in features:
            exprs.append(d.dt.weekday().is_in([6, 7]).cast(pl.Int8).alias(f"{col}_is_weekend"))

    for col in fitted_plan["text_columns"]:
        if col not in column_set:
            continue
        x = pl.col(col).cast(pl.Utf8).fill_null("")
        features = _as_list(plan["text"].get("features"))
        if "char_count" in features:
            exprs.append(x.str.len_chars().alias(f"{col}_char_count"))
        if "word_count" in features:
            exprs.append(x.str.count_matches(r"\S+").alias(f"{col}_word_count"))
        if "digit_count" in features:
            exprs.append(x.str.count_matches(r"\d").alias(f"{col}_digit_count"))
        if "punct_count" in features:
            exprs.append(x.str.count_matches(r"[[:punct:]]").alias(f"{col}_punct_count"))
        if "upper_ratio" in features:
            exprs.append(
                pl.when(x.str.len_chars() > 0)
                .then(x.str.count_matches(r"[A-Z]") / x.str.len_chars())
                .otherwise(0)
                .alias(f"{col}_upper_ratio")
            )
        if "blank" in features:
            exprs.append((x.str.strip_chars().str.len_chars() == 0).cast(pl.Int8).alias(f"{col}_blank"))

    suffix = str(plan["missingness"].get("suffix", "_is_missing"))
    for col in fitted_plan["missingness_columns"]:
        if col in column_set:
            exprs.append(pl.col(col).is_null().cast(pl.Int8).alias(f"{col}{suffix}"))

    for pair in fitted_plan["interaction_specs"]["numeric_pairs"]:
        exprs.append((pl.col(pair[0]) * pl.col(pair[1])).alias(f"{pair[0]}_x_{pair[1]}"))
    for item in fitted_plan["interaction_specs"]["categorical_numeric"]:
        exprs.append(
            (pl.col(item["categorical"]).cast(pl.Utf8) == pl.lit(str(item["level"])))
            .cast(pl.Int8)
            .mul(pl.col(item["numeric"]))
            .alias(_make_feature_name(item["categorical"], str(item["level"]), "x", item["numeric"]))
        )
    for pair in fitted_plan["interaction_specs"]["categorical_pairs"]:
        exprs.append((pl.col(pair[0]).cast(pl.Utf8) + pl.lit("__") + pl.col(pair[1]).cast(pl.Utf8)).alias(f"{pair[0]}_x_{pair[1]}"))

    if exprs:
        return data.with_columns(exprs)
    return data


def polars_fit_transform_feature_plan(data: pl.DataFrame, plan: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Fit a vNext plan and transform the same data."""
    fitted_plan = polars_fit_feature_plan(data, plan or polars_feature_plan())
    return {"engineered_data": polars_transform_feature_plan(data, fitted_plan), "fitted_plan": fitted_plan}


def generate_polars_feature_engineering_artifacts(
    data: pl.DataFrame,
    plan: Optional[Dict[str, Any]] = None,
    fitted_plan: Optional[Dict[str, Any]] = None,
    benchmark_summary: Optional[pl.DataFrame] = None,
) -> Dict[str, Any]:
    """Generate app-agnostic PolarsFE feature engineering artifacts."""
    if fitted_plan is None:
        fit = polars_fit_transform_feature_plan(data, plan or polars_feature_plan())
        fitted_plan = fit["fitted_plan"]
        engineered_data = fit["engineered_data"]
    else:
        engineered_data = polars_transform_feature_plan(data, fitted_plan)

    engineered_summary = pl.DataFrame(
        {
            "metric": ["input_rows", "input_columns", "engineered_columns", "generated_features"],
            "value": [data.height, len(data.columns), len(engineered_data.columns), fitted_plan["feature_manifest"].height],
        }
    )
    return {
        "artifacts": {
            "overview_text": "PolarsFE vNext feature engineering run completed.",
            "config_table": pl.DataFrame(
                {
                    "family": ["numeric", "categorical", "calendar", "text", "missingness", "interactions", "cross_row"],
                    "enabled": [
                        bool(fitted_plan["numeric_specs"]),
                        bool(fitted_plan["categorical_specs"]),
                        bool(fitted_plan["calendar_columns"]),
                        bool(fitted_plan["text_columns"]),
                        bool(fitted_plan["missingness_columns"]),
                        bool(sum(len(v) for v in fitted_plan["interaction_specs"].values())),
                        bool(fitted_plan["plan"].get("cross_row", {}).get("enabled")),
                    ],
                }
            ),
            "feature_manifest": fitted_plan["feature_manifest"],
            "diagnostics": fitted_plan["diagnostics"],
            "engineered_data_summary": engineered_summary,
            "benchmark_summary": benchmark_summary,
        },
        "metadata": {"generator": "generate_polars_feature_engineering_artifacts", "generated_at": datetime.utcnow().isoformat(timespec="seconds")},
        "warnings": fitted_plan["warnings"],
        "diagnostics": fitted_plan["diagnostics"],
        "value": {
            "engineered_data": engineered_data,
            "fitted_plan": fitted_plan,
            "feature_manifest": fitted_plan["feature_manifest"],
            "diagnostics": fitted_plan["diagnostics"],
            "warnings": fitted_plan["warnings"],
        },
    }


def polars_partition_plan(
    method: str = "random",
    fractions: Optional[Dict[str, float]] = None,
    target_col: Optional[str] = None,
    group_col: Optional[str] = None,
    date_col: Optional[str] = None,
    seed: int = 123,
    row_id_col: str = ".row_id",
    partition_col: str = ".partition",
    fold_col: str = ".fold_id",
    k: int = 5,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a PolarsFE vNext model-prep partition plan."""
    method = str(method)
    if method not in {"random", "stratified", "grouped", "time"}:
        raise ValueError("method must be one of random, stratified, grouped, or time")
    normalized = _normalize_fractions(fractions or {"train": 0.8, "test": 0.2})
    return {
        "method": method,
        "fractions": normalized,
        "target_col": target_col,
        "group_col": group_col,
        "date_col": date_col,
        "seed": int(seed),
        "row_id_col": str(row_id_col),
        "partition_col": str(partition_col),
        "fold_col": str(fold_col),
        "k": int(k),
        "metadata": metadata or {},
        "created_at": datetime.utcnow().isoformat(timespec="seconds"),
    }


def _normalize_fractions(fractions: Dict[str, float]) -> Dict[str, float]:
    values = {str(k): float(v) for k, v in fractions.items()}
    if not values or any(v <= 0 for v in values.values()):
        raise ValueError("fractions must contain positive numeric values")
    total = sum(values.values())
    return {k: v / total for k, v in values.items()}


def _with_row_id(data: pl.DataFrame, row_id_col: str) -> pl.DataFrame:
    if row_id_col in data.columns:
        return data
    return data.with_row_index(row_id_col, offset=1)


def _cut_labels(ids: List[int], fractions: Dict[str, float]) -> List[str]:
    n = len(ids)
    sizes = {name: int(frac * n) for name, frac in fractions.items()}
    remainder = n - sum(sizes.values())
    keys = list(fractions.keys())
    for key in keys[:remainder]:
        sizes[key] += 1
    labels: List[str] = []
    for key in keys:
        labels.extend([key] * sizes[key])
    return labels[:n]


def _random_assignments(ids: List[int], fractions: Dict[str, float], seed: int, row_id_col: str, partition_col: str) -> pl.DataFrame:
    shuffled = list(ids)
    random.Random(seed).shuffle(shuffled)
    return pl.DataFrame({row_id_col: shuffled, partition_col: _cut_labels(shuffled, fractions)})


def polars_fit_partition_plan(data: pl.DataFrame, plan: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Fit a PolarsFE vNext partition plan and return reusable assignments."""
    plan = plan or polars_partition_plan()
    data_with_id = _with_row_id(data, plan["row_id_col"])
    warnings: List[str] = []
    method = plan["method"]
    row_id_col = plan["row_id_col"]
    partition_col = plan["partition_col"]

    if method == "stratified" and (not plan.get("target_col") or plan["target_col"] not in data_with_id.columns):
        warnings.append("Stratified partition requested without a valid target_col; falling back to random partitioning.")
        method = "random"
    if method == "grouped" and (not plan.get("group_col") or plan["group_col"] not in data_with_id.columns):
        warnings.append("Grouped partition requested without a valid group_col; falling back to random partitioning.")
        method = "random"
    if method == "time" and (not plan.get("date_col") or plan["date_col"] not in data_with_id.columns):
        warnings.append("Time partition requested without a valid date_col; falling back to random partitioning.")
        method = "random"

    if method == "stratified":
        frames: List[pl.DataFrame] = []
        for idx, key in enumerate(data_with_id.select(plan["target_col"]).unique().to_series().to_list()):
            ids = data_with_id.filter(pl.col(plan["target_col"]) == key)[row_id_col].to_list()
            frames.append(_random_assignments(ids, plan["fractions"], plan["seed"] + idx + 1, row_id_col, partition_col))
        assignments = pl.concat(frames)
    elif method == "grouped":
        groups = data_with_id.select(plan["group_col"]).unique().to_series().to_list()
        shuffled = list(groups)
        random.Random(plan["seed"]).shuffle(shuffled)
        labels = _cut_labels(list(range(len(shuffled))), plan["fractions"])
        group_map = {group: labels[idx] for idx, group in enumerate(shuffled)}
        assignments = data_with_id.select(
            [
                pl.col(row_id_col),
                pl.col(plan["group_col"]).replace(group_map).alias(partition_col),
            ]
        )
    elif method == "time":
        ordered_ids = data_with_id.sort([plan["date_col"], row_id_col])[row_id_col].to_list()
        assignments = pl.DataFrame({row_id_col: ordered_ids, partition_col: _cut_labels(ordered_ids, plan["fractions"])})
    else:
        assignments = _random_assignments(data_with_id[row_id_col].to_list(), plan["fractions"], plan["seed"], row_id_col, partition_col)

    folds = polars_create_folds(
        data_with_id,
        k=plan["k"],
        target_col=plan.get("target_col") if method == "stratified" else None,
        group_col=plan.get("group_col") if method == "grouped" else None,
        seed=plan["seed"],
        row_id_col=row_id_col,
        fold_col=plan["fold_col"],
    )
    assignments = assignments.join(folds, on=row_id_col, how="left").sort(row_id_col)
    partition_manifest = assignments.group_by(partition_col).len().rename({"len": "rows"}).sort(partition_col)
    fold_manifest = assignments.group_by(plan["fold_col"]).len().rename({"len": "rows"}).sort(plan["fold_col"])
    diagnostics = pl.DataFrame(
        {
            "check": ["input_rows", "partition_method", "partition_count", "fold_count"],
            "status": ["ok", "ok", "ok", "ok"],
            "detail": [str(data.height), method, str(partition_manifest.height), str(fold_manifest.height)],
        }
    )
    fitted = dict(plan)
    fitted["method"] = method
    return {
        "plan": fitted,
        "assignments": assignments,
        "partition_manifest": partition_manifest,
        "fold_manifest": fold_manifest,
        "diagnostics": diagnostics,
        "warnings": sorted(set(warnings)),
        "fitted_at": datetime.utcnow().isoformat(timespec="seconds"),
    }


def polars_apply_partition_plan(data: pl.DataFrame, fitted_plan: Dict[str, Any]) -> pl.DataFrame:
    """Apply fitted partition and fold assignments to a Polars DataFrame."""
    data_with_id = _with_row_id(data, fitted_plan["plan"]["row_id_col"])
    return data_with_id.join(fitted_plan["assignments"], on=fitted_plan["plan"]["row_id_col"], how="left")


def polars_create_folds(
    data: pl.DataFrame,
    k: int = 5,
    target_col: Optional[str] = None,
    group_col: Optional[str] = None,
    seed: int = 123,
    row_id_col: str = ".row_id",
    fold_col: str = ".fold_id",
) -> pl.DataFrame:
    """Create reproducible fold assignments."""
    data_with_id = _with_row_id(data, row_id_col)
    k = max(2, int(k))
    rng = random.Random(int(seed))

    if group_col and group_col in data_with_id.columns:
        groups = data_with_id.select(group_col).unique().to_series().to_list()
        rng.shuffle(groups)
        group_map = {group: (idx % k) + 1 for idx, group in enumerate(groups)}
        return data_with_id.select([pl.col(row_id_col), pl.col(group_col).replace(group_map).cast(pl.Int64).alias(fold_col)]).sort(row_id_col)

    if target_col and target_col in data_with_id.columns:
        frames: List[pl.DataFrame] = []
        for key in data_with_id.select(target_col).unique().to_series().to_list():
            ids = data_with_id.filter(pl.col(target_col) == key)[row_id_col].to_list()
            rng.shuffle(ids)
            frames.append(pl.DataFrame({row_id_col: ids, fold_col: [(idx % k) + 1 for idx in range(len(ids))]}))
        return pl.concat(frames).sort(row_id_col)

    ids = data_with_id[row_id_col].to_list()
    rng.shuffle(ids)
    return pl.DataFrame({row_id_col: ids, fold_col: [(idx % k) + 1 for idx in range(len(ids))]}).sort(row_id_col)


def generate_polars_model_prep_artifacts(
    data: pl.DataFrame,
    plan: Optional[Dict[str, Any]] = None,
    fitted_plan: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate app-agnostic PolarsFE model-prep artifacts."""
    fitted_plan = fitted_plan or polars_fit_partition_plan(data, plan or polars_partition_plan())
    prepared_data = polars_apply_partition_plan(data, fitted_plan)
    return {
        "artifacts": {
            "overview_text": "PolarsFE vNext model-prep partition run completed.",
            "partition_manifest": fitted_plan["partition_manifest"],
            "fold_manifest": fitted_plan["fold_manifest"],
            "diagnostics": fitted_plan["diagnostics"],
            "assignment_manifest": fitted_plan["assignments"],
        },
        "metadata": {
            "generator": "generate_polars_model_prep_artifacts",
            "generated_at": datetime.utcnow().isoformat(timespec="seconds"),
            "method": fitted_plan["plan"]["method"],
            "seed": fitted_plan["plan"]["seed"],
            "leakage_safe": True,
        },
        "warnings": fitted_plan["warnings"],
        "diagnostics": fitted_plan["diagnostics"],
        "value": {
            "prepared_data": prepared_data,
            "fitted_plan": fitted_plan,
            "partition_manifest": fitted_plan["partition_manifest"],
            "fold_manifest": fitted_plan["fold_manifest"],
        },
    }


def _qa_fixture() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": list(range(1, 9)),
            "x": [1.0, 2.0, 3.0, None, 5.0, 100.0, 7.0, 8.0],
            "y": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
            "cat": ["A", "A", "B", "B", "C", "C", "D", None],
            "cat2": ["K", "L", "K", "L", "K", "L", "K", "L"],
            "date": pl.date_range(start=pl.date(2024, 1, 1), end=pl.date(2024, 1, 8), interval="1d", eager=True),
            "text": ["Hello WORLD", "two words", "", None, "ABC123!", "small", "More text.", "LAST"],
        }
    )


def _qa_plan() -> Dict[str, Any]:
    return polars_feature_plan(
        numeric={"columns": ["x", "y"], "transforms": ["log1p", "sqrt", "standardize", "winsorize"], "winsorize_probs": [0.1, 0.9]},
        categorical={"columns": ["cat"], "top_n": 2, "rare_level": "__RARE__", "unseen_level": "__UNSEEN__", "one_hot": True, "keep_original": True},
        calendar={"columns": ["date"], "features": ["year", "month", "day", "wday", "week", "quarter", "is_weekend"]},
        text={"columns": ["text"], "features": ["char_count", "word_count", "digit_count", "punct_count", "upper_ratio", "blank"]},
        missingness={"columns": ["x", "cat", "text"], "suffix": "_is_missing"},
        interactions={
            "numeric_pairs": [["x", "y"]],
            "categorical_numeric": [{"categorical": "cat", "numeric": "y"}],
            "categorical_pairs": [["cat", "cat2"]],
            "max_features": 20,
        },
    )


def _qa_result(test: str, passed: bool, detail: str) -> Dict[str, Any]:
    return {"test": test, "passed": bool(passed), "detail": detail}


def qa_polarsfe_vnext_numeric() -> Dict[str, Any]:
    out = polars_fit_transform_feature_plan(_qa_fixture(), _qa_plan())["engineered_data"]
    cols = ["x_log1p", "x_sqrt", "x_standardize", "x_winsorize"]
    return _qa_result("numeric", all(col in out.columns for col in cols), ", ".join(cols))


def qa_polarsfe_vnext_categorical() -> Dict[str, Any]:
    fit = polars_fit_transform_feature_plan(_qa_fixture(), _qa_plan())
    score = _qa_fixture().with_columns(pl.when(pl.col("id") == 1).then(pl.lit("NEW")).otherwise(pl.col("cat")).alias("cat"))
    scored = polars_transform_feature_plan(score, fit["fitted_plan"])
    unseen_col = _make_feature_name("cat", "__UNSEEN__")
    return _qa_result("categorical", unseen_col in scored.columns and scored[unseen_col][0] == 1, "unseen category handled")


def qa_polarsfe_vnext_calendar() -> Dict[str, Any]:
    out = polars_fit_transform_feature_plan(_qa_fixture(), _qa_plan())["engineered_data"]
    return _qa_result("calendar", all(col in out.columns for col in ["date_year", "date_month", "date_is_weekend"]), "calendar columns exist")


def qa_polarsfe_vnext_text() -> Dict[str, Any]:
    out = polars_fit_transform_feature_plan(_qa_fixture(), _qa_plan())["engineered_data"]
    return _qa_result("text", all(col in out.columns for col in ["text_char_count", "text_word_count", "text_blank"]), "text columns exist")


def qa_polarsfe_vnext_interactions() -> Dict[str, Any]:
    out = polars_fit_transform_feature_plan(_qa_fixture(), _qa_plan())["engineered_data"]
    return _qa_result("interactions", all(col in out.columns for col in ["x_x_y", "cat_x_cat2"]), "interaction columns exist")


def qa_polarsfe_vnext_fit_transform() -> Dict[str, Any]:
    fit = polars_fit_transform_feature_plan(_qa_fixture(), _qa_plan())
    return _qa_result("fit_transform", "fitted_plan" in fit and fit["engineered_data"].height == _qa_fixture().height, "fit/transform reusable")


def qa_generate_polars_feature_engineering_artifacts() -> Dict[str, Any]:
    out = generate_polars_feature_engineering_artifacts(_qa_fixture(), _qa_plan())
    return _qa_result("artifact_generator", all(key in out for key in ["artifacts", "metadata", "warnings", "diagnostics", "value"]), "structured output")


def _model_prep_fixture() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": list(range(1, 121)),
            "target": ["no", "yes"] * 60,
            "group": [f"g{(i // 5) + 1}" for i in range(120)],
            "event_date": pl.date_range(start=pl.date(2024, 1, 1), end=pl.date(2024, 4, 29), interval="1d", eager=True),
            "x": list(range(1, 121)),
        }
    )


def qa_polarsfe_vnext_model_prep() -> pl.DataFrame:
    data = _model_prep_fixture()
    random_fit = polars_fit_partition_plan(data, polars_partition_plan(method="random", fractions={"train": 0.7, "test": 0.3}, seed=1))
    strat_fit = polars_fit_partition_plan(data, polars_partition_plan(method="stratified", fractions={"train": 0.7, "test": 0.3}, target_col="target", seed=1))
    group_fit = polars_fit_partition_plan(data, polars_partition_plan(method="grouped", fractions={"train": 0.7, "test": 0.3}, group_col="group", seed=1))
    time_fit = polars_fit_partition_plan(data, polars_partition_plan(method="time", fractions={"train": 0.7, "test": 0.3}, date_col="event_date", seed=1))

    group_applied = polars_apply_partition_plan(data, group_fit)
    time_applied = polars_apply_partition_plan(data, time_fit)
    folds = polars_create_folds(data, k=5, target_col="target", seed=2)

    group_check = (
        group_applied.group_by("group")
        .agg(pl.col(".partition").n_unique().alias("n_partitions"))
        .select((pl.col("n_partitions") == 1).all())
        .item()
    )
    train_max = time_applied.filter(pl.col(".partition") == "train").select(pl.col("event_date").max()).item()
    test_min = time_applied.filter(pl.col(".partition") == "test").select(pl.col("event_date").min()).item()

    return pl.DataFrame(
        [
            _qa_result("random_partition", set(random_fit["partition_manifest"][".partition"].to_list()) == {"train", "test"}, "train/test assigned"),
            _qa_result("stratified_partition", set(polars_apply_partition_plan(data, strat_fit).filter(pl.col(".partition") == "train")["target"].to_list()) == {"no", "yes"}, "target classes preserved in train"),
            _qa_result("grouped_partition", bool(group_check), "groups do not cross partitions"),
            _qa_result("time_partition", train_max <= test_min, "training dates precede test dates"),
            _qa_result("folds", set(folds[".fold_id"].unique().to_list()) == {1, 2, 3, 4, 5}, "fold ids assigned"),
            _qa_result("manifest", all(key in group_fit for key in ["partition_manifest", "fold_manifest", "assignments"]), "structured fitted plan"),
        ]
    )


def qa_generate_polars_model_prep_artifacts() -> Dict[str, Any]:
    out = generate_polars_model_prep_artifacts(
        _model_prep_fixture(),
        polars_partition_plan(method="stratified", fractions={"train": 0.7, "validation": 0.1, "test": 0.2}, target_col="target"),
    )
    return _qa_result("model_prep_artifact_generator", all(key in out for key in ["artifacts", "metadata", "warnings", "diagnostics", "value"]), "structured output")


def qa_polarsfe_vnext() -> pl.DataFrame:
    return pl.DataFrame(
        [
            qa_polarsfe_vnext_numeric(),
            qa_polarsfe_vnext_categorical(),
            qa_polarsfe_vnext_calendar(),
            qa_polarsfe_vnext_text(),
            qa_polarsfe_vnext_interactions(),
            qa_polarsfe_vnext_fit_transform(),
            qa_generate_polars_feature_engineering_artifacts(),
            *qa_polarsfe_vnext_model_prep().to_dicts(),
            qa_generate_polars_model_prep_artifacts(),
        ]
    )
