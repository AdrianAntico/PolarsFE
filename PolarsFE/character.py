import polars as pl
from typing import List, Optional, Union, Dict, Any


def dummy_variables(
    df: pl.DataFrame, 
    columns: list[str], 
    levels: dict[str, list] = None, 
    top_n: dict[str, int] = None, 
    keep_original: bool = False, 
    return_levels: bool = False,
):
    """
    Create dummy variables for specified columns in a Polars DataFrame with custom level selection.
    
    Parameters:
      df (pl.DataFrame): The original dataframe.
      columns (list[str]): The list of categorical columns to encode.
      levels (dict[str, list], optional): Specific category levels to include per column.
      top_n (dict[str, int], optional): The number of top frequent categories to encode.
      keep_original (bool, optional): Whether to keep the original categorical columns. Default is False.
      return_levels (bool, optional): Whether to return the selected category levels for reproducibility.
    
    Returns:
      pl.DataFrame: The modified DataFrame with dummy variables.
      dict[str, list]: (Optional) Dictionary with selected category levels per column.
    
    Example:
      from polars_feature_engineering import character
      import polars as pl
      
      df = pl.DataFrame({
          "Category": ["A", "B", "A", "C", "B", "C", "A", "B", "D"],
          "Color": ["Red", "Blue", "Green", "Red", "Green", "Blue", "Red", "Red", "Green"],
          "Value": [10, 20, 30, 40, 50, 60, 70, 80, 90]  # Numeric column (won't be affected)
      })
      
      # Create dummies for 'Category' and 'Color' and keep the original columns
      df_dummies, levels_used = character.dummy_variables(
          df,
          columns=["Category", "Color"],
          levels=None,  # {"Category": ["A","B","G"], "Color": ["Red","Blue"]},
          top_n=None,  # {"Category": 2, "Color": 3},
          keep_original=True,
          return_levels=True)
      
      print(df_dummies)
      print(levels_used)
    """
    df = df.clone()  # Avoid modifying the original DataFrame
    selected_levels = {}  # Store selected levels for reproducibility
    
    for col in columns:  # col = columns[0]
        unique_counts = df[col].value_counts().sort("count", descending=True)
        
        # Determine which levels to use
        if levels and col in levels:
            selected = levels[col]
        elif top_n and col in top_n:
            selected = unique_counts.head(top_n[col])[col].to_list()
        else:
            selected = unique_counts[col].to_list()  # Default to all unique levels
        
        selected_levels[col] = selected  # Store for reproducibility

        # Create dummy variables for selected levels
        for level in selected:
            df = df.with_columns((df[col] == level).cast(pl.Int8()).alias(f"{col}_{level}"))

    # Drop original columns if requested
    if not keep_original:
        df = df.drop(columns)

    if return_levels:
        return df, selected_levels
    return df


def categorical_encoding(
    data: pl.DataFrame,
    ML_Type: str = "regression",
    group_variables: Optional[List[str]] = None,
    target_variable: Optional[str] = None,
    method: str = "target_encoding",
    scoring: bool = False,
    impute_value_scoring: Optional[Union[int, float]] = None,
    return_factor_level_list: bool = True,
    supply_factor_level_list: Optional[Dict[str, pl.DataFrame]] = None,
    keep_original_factors: bool = True,
    debug: bool = False,
) -> Union[pl.DataFrame, Dict[str, Union[pl.DataFrame, Dict[str, pl.DataFrame]]]]:
    """
    Categorical encoding using either target encoding or James–Stein encoding.

    This version has **no file I/O**. All mappings are kept in memory and
    passed via `supply_factor_level_list` / returned via `factor_components`.

    Parameters
    ----------
    data : pl.DataFrame
        Input data.
    ML_Type : str
        "classification", "multiclass", or "regression".
    group_variables : list[str] or None
        Categorical columns to encode.
    target_variable : str or None
        The target column.
    method : str
        Either "target_encoding" or "james-stein".
    scoring : bool
        If True, apply existing mappings (scoring mode).
    impute_value_scoring : int or float or None
        Value to impute if mapping is missing (unseen levels → this value).
    return_factor_level_list : bool
        If True (and not scoring) returns mapping components.
    supply_factor_level_list : dict or None
        Mapping components to use in scoring mode.
    keep_original_factors : bool
        If False, original factor columns are dropped.
    debug : bool
        If True, prints debug messages.

    Returns
    -------
    Either a Polars DataFrame or a dict with keys:
      - "data"             : encoded DataFrame
      - "factor_components": dict of mapping DataFrames (training only)
    """

    # Only allow supported methods.
    if method not in ["james-stein", "target_encoding"]:
        if debug:
            print("Method not recognized. Returning original data.")
        return data

    # Ensure data is a Polars DataFrame.
    if not isinstance(data, pl.DataFrame):
        data = pl.DataFrame(data)

    # Use only valid group variables.
    group_variables = [gv for gv in (group_variables or []) if gv in data.columns]

    # For holding mapping tables (if in training mode)
    factor_components: Optional[Dict[str, pl.DataFrame]] = {} if not scoring else None

    ML_Type = ML_Type.lower()
    method_lower = method.lower()

    # ----- TARGET ENCODING -----
    if method_lower == "target_encoding":
        for group in group_variables:
            if debug:
                print(f"Target encoding on '{group}'")

            if not scoring:
                # TRAINING MODE
                if ML_Type == "multiclass":
                    # Count rows per (group, target) pair.
                    df_counts = data.group_by([group, target_variable]).agg(
                        pl.len().alias("N")
                    )
                    # Sum counts over each target level.
                    target_totals = df_counts.group_by(target_variable).agg(
                        pl.col("N").sum().alias("total")
                    )
                    df_counts = df_counts.join(target_totals, on=target_variable, how="left")
                    df_counts = df_counts.with_columns(
                        (pl.col("N") / pl.col("total")).alias(f"{group}_TargetEncode")
                    )
                    # Pivot so that each target level becomes a separate column.
                    mapping_df = df_counts.pivot(
                        values=f"{group}_TargetEncode",
                        index=group,
                        on=target_variable,
                        aggregate_function="first",
                    )
                    # Rename columns to include the target level.
                    rename_dict = {
                        col: f"{group}_TargetEncode_TargetLevel_{col}"
                        for col in mapping_df.columns
                        if col != group
                    }
                    mapping_df = mapping_df.rename(rename_dict)
                else:
                    mapping_df = data.group_by(group).agg(
                        pl.col(target_variable).mean().alias(f"{group}_TargetEncode")
                    )

            else:
                # SCORING MODE
                if supply_factor_level_list is not None and group in supply_factor_level_list:
                    mapping_df = supply_factor_level_list[group]
                    if not isinstance(mapping_df, pl.DataFrame):
                        mapping_df = pl.from_pandas(mapping_df)
                else:
                    raise ValueError(
                        "In scoring mode you must supply 'supply_factor_level_list' "
                        "containing a mapping DataFrame for each group variable."
                    )

            # Join encoded column(s) back to data
            data = data.join(mapping_df, on=group, how="left")

            if not keep_original_factors:
                data = data.drop(group)

            # Fill unseen levels with impute_value_scoring
            if scoring and impute_value_scoring is not None:
                if ML_Type == "multiclass":
                    for col in mapping_df.columns:
                        if col != group:
                            data = data.with_columns(
                                pl.col(col).fill_null(impute_value_scoring)
                            )
                else:
                    new_col = f"{group}_TargetEncode"
                    data = data.with_columns(
                        pl.col(new_col).fill_null(impute_value_scoring)
                    )

            if not scoring and factor_components is not None:
                factor_components[group] = mapping_df

        if not scoring and return_factor_level_list:
            return {"data": data, "factor_components": factor_components}
        else:
            return data

    # ----- JAMES–STEIN ENCODING -----
    elif method_lower == "james-stein":
        for group in group_variables:
            if debug:
                print(f"James–Stein encoding on '{group}'")

            if not scoring:
                # TRAINING MODE
                if ML_Type == "multiclass":
                    df_counts = data.group_by([group, target_variable]).agg(
                        pl.len().alias("N")
                    )
                    grand_sum = df_counts.select(pl.col("N").sum()).item()
                    target_totals = df_counts.group_by(target_variable).agg(
                        pl.col("N").sum().alias("TargetSum")
                    )
                    df_counts = df_counts.join(target_totals, on=target_variable, how="left")
                    df_counts = df_counts.with_columns([
                        pl.lit(grand_sum).alias("GrandSum"),
                        (pl.col("TargetSum") / pl.lit(grand_sum)).alias("TargetMean"),
                        (pl.col("N") / pl.col("TargetSum")).alias("TargetGroupMean"),
                    ])
                    df_counts = df_counts.with_columns([
                        ((pl.col("TargetMean") * (1 - pl.col("TargetMean"))) /
                         pl.col("TargetSum")).alias("TargetVariance"),
                        ((pl.col("TargetGroupMean") * (1 - pl.col("TargetGroupMean"))) /
                         pl.col("N")).alias("TargetGroupVariance"),
                    ])
                    df_counts = df_counts.with_columns(
                        (pl.col("TargetGroupVariance") /
                         (pl.col("TargetGroupVariance") + pl.col("TargetVariance"))
                         ).alias("Z")
                    )
                    df_counts = df_counts.with_columns(
                        ((1 - pl.col("Z")) * pl.col("TargetGroupMean") +
                         pl.col("Z") * pl.col("TargetMean")).alias(f"{group}_JamesStein")
                    )
                    df_counts = df_counts.select([group, target_variable, f"{group}_JamesStein"])
                    mapping_df = df_counts.pivot(
                        values=f"{group}_JamesStein",
                        index=group,
                        on=target_variable,
                        aggregate_function="first",
                    )
                    rename_dict = {
                        col: f"{group}_JamesStein_TargetLevel_{col}"
                        for col in mapping_df.columns
                        if col != group
                    }
                    mapping_df = mapping_df.rename(rename_dict)

                else:
                    grand_mean = data[target_variable].mean()
                    if ML_Type in ["classification", "classifier"]:
                        mapping_df = data.group_by(group).agg([
                            pl.col(target_variable).mean().alias("Mean"),
                            pl.len().alias("N")
                        ])
                        mapping_df = mapping_df.with_columns(
                            (pl.col("Mean") * (1 - pl.col("Mean")) / pl.col("N")).alias("Var_Group")
                        )
                        total_count = data.height
                        pop_var = (grand_mean * (1 - grand_mean)) / total_count
                        mapping_df = mapping_df.with_columns(
                            (pl.lit(pop_var) / (pl.col("Var_Group") + pl.lit(pop_var))).alias("Z")
                        )
                        mapping_df = mapping_df.with_columns(
                            (pl.col("Z") * pl.col("Mean") +
                             (1 - pl.col("Z")) * pl.lit(grand_mean)
                             ).alias(f"{group}_JamesStein")
                        )
                        mapping_df = mapping_df.select([group, f"{group}_JamesStein"])

                    elif ML_Type == "regression":
                        mapping_df = data.group_by(group).agg([
                            pl.col(target_variable).mean().alias("Mean"),
                            pl.col(target_variable).var().alias("EPV"),
                            pl.len().alias("N")
                        ])
                        overall_epv = mapping_df["EPV"].mean()
                        mapping_df = mapping_df.with_columns(pl.lit(overall_epv).alias("EPV"))
                        group_means = mapping_df["Mean"].to_list()
                        num_groups = len(group_means)
                        V = (
                            sum((m - grand_mean) ** 2 for m in group_means) /
                            (num_groups - 1)
                        ) if num_groups > 1 else 0
                        mapping_df = mapping_df.with_columns(
                            (pl.lit(V) - pl.col("EPV") / pl.col("N")).alias("VHM")
                        )
                        first_VHM = mapping_df.select("VHM").to_series()[0]
                        K = overall_epv / first_VHM if first_VHM != 0 else 0
                        mapping_df = mapping_df.with_columns(
                            (pl.col("N") / (pl.col("N") + pl.lit(K))).alias("Z")
                        )
                        mapping_df = mapping_df.with_columns(
                            (pl.col("Z") * pl.col("Mean") +
                             (1 - pl.col("Z")) * pl.lit(grand_mean)
                             ).alias(f"{group}_JamesStein")
                        )
                        mapping_df = mapping_df.select([group, f"{group}_JamesStein"])

            else:
                # SCORING MODE
                if supply_factor_level_list is not None and group in supply_factor_level_list:
                    mapping_df = supply_factor_level_list[group]
                    if not isinstance(mapping_df, pl.DataFrame):
                        mapping_df = pl.from_pandas(mapping_df)
                else:
                    raise ValueError(
                        "In scoring mode you must supply 'supply_factor_level_list' "
                        "containing a mapping DataFrame for each group variable."
                    )

            # Join encoded column(s) back to data
            data = data.join(mapping_df, on=group, how="left")

            if not keep_original_factors:
                data = data.drop(group)

            # Fill unseen levels with impute_value_scoring
            if scoring and impute_value_scoring is not None:
                if ML_Type == "multiclass":
                    for col in mapping_df.columns:
                        if col != group:
                            data = data.with_columns(
                                pl.col(col).fill_null(impute_value_scoring)
                            )
                else:
                    new_col = f"{group}_JamesStein"
                    data = data.with_columns(
                        pl.col(new_col).fill_null(impute_value_scoring)
                    )

            if not scoring and factor_components is not None:
                factor_components[group] = mapping_df

        if not scoring and return_factor_level_list:
            return {"data": data, "factor_components": factor_components}
        else:
            return data
