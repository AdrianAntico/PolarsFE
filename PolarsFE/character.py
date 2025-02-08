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
    ML_Type: str = "classification",
    group_variables: Optional[List[str]] = None,
    target_variable: Optional[str] = None,
    method: str = "target_encoding",
    save_path: Optional[str] = None,
    scoring: bool = False,
    impute_value_scoring: Optional[Union[int, float]] = None,
    return_factor_level_list: bool = True,
    supply_factor_level_list: Optional[Dict[str, pl.DataFrame]] = None,
    keep_original_factors: bool = True,
    debug: bool = False,
) -> Union[pl.DataFrame, Dict[str, Union[pl.DataFrame, Dict[str, pl.DataFrame]]]]:
    """
    Categorical encoding using either target encoding or James–Stein encoding.
    (Polars-based implementation using group_by.)
    
    Parameters:
      - data (pl.DataFrame): Input data.
      - ML_Type (str): "classification", "multiclass", or "regression".
      - group_variables (Optional[List[str]]): Categorical columns to encode.
      - target_variable (Optional[str]): The target column.
      - method (str): Either "target_encoding" or "james-stein".
      - save_path (Optional[str]): Directory to save/read mapping files.
      - scoring (bool): If True, then we are in scoring mode.
      - impute_value_scoring (Optional[Union[int, float]]): Value to impute if mapping is missing.
      - return_factor_level_list (bool): If True (and not scoring) returns mapping components.
      - supply_factor_level_list (Optional[Dict[str, pl.DataFrame]]): Mapping components to use in scoring mode.
      - keep_original_factors (bool): If False, original factor columns are dropped.
      - debug (bool): If True, prints debug messages.
      
    Returns:
      Either a Polars DataFrame or a dict with keys "data" and "factor_components".

    Examples:
      # ----------------------------------------------------------------------------------
      # Classification target version
      # ----------------------------------------------------------------------------------
      
      import os
      import numpy as np
      import polars as pl
      from PolarsFE import character
      
      # Set a seed for reproducibility
      np.random.seed(42)
      
      # Define parameters for fake data
      num_rows = 1000
      num_factors = 10
      categories = ["A", "B", "C", "D", "E"]
      
      # Build fake data for factors
      fake_data = {f"Factor_{i}": np.random.choice(categories, size=num_rows)
                   for i in range(1, num_factors + 1)}
      
      # Create a binary target column ("Adrian")
      fake_data["Adrian"] = np.random.binomial(1, 0.5, size=num_rows)
      
      # Convert to a Polars DataFrame
      df = pl.DataFrame(fake_data)
      
      print("=== Fake Data Sample ===")
      print(df.head())
      
      # List of factor columns to encode
      factor_columns = [f"Factor_{i}" for i in range(1, num_factors + 1)]
      
      # --- Test Target Encoding ---
      print("\n=== Testing Target Encoding ===")
      result_target = character.categorical_encoding(
          data=df,
          ML_Type="classification",
          group_variables=factor_columns,
          target_variable="Adrian",
          method="target_encoding",
          save_path=None,          # No file saving for this test
          scoring=False,           # Training mode
          keep_original_factors=False,
          debug=True,              # Enable debug prints
      )
      
      if isinstance(result_target, dict):
          encoded_df_target = result_target["data"]
      else:
          encoded_df_target = result_target
      
      print("\n--- Target Encoding Result Sample ---")
      print(encoded_df_target.head())
      
      # --- Test James–Stein Encoding ---
      print("\n=== Testing James–Stein Encoding ===")
      result_js = character.categorical_encoding(
          data=df,
          ML_Type="classification",
          group_variables=factor_columns,
          target_variable="Adrian",
          method="james-stein",
          save_path=None,          # No file saving for this test
          scoring=False,           # Training mode
          keep_original_factors=True,
          debug=True,              # Enable debug prints
      )
      
      if isinstance(result_js, dict):
          encoded_df_js = result_js["data"]
      else:
          encoded_df_js = result_js
      
      print("\n--- James–Stein Encoding Result Sample ---")
      print(encoded_df_js.head())
      
      
      import os
      import numpy as np
      import polars as pl
      
      # ------------------------------------------------------------------------------
      # Regression target version
      # ------------------------------------------------------------------------------
      
      # Create a fake regression dataset
      np.random.seed(42)
      
      num_rows = 1000
      num_factors = 5
      categories = ["A", "B", "C", "D", "E"]
      
      # Build fake data for categorical factors
      data_dict = {f"Factor_{i}": np.random.choice(categories, size=num_rows)
                   for i in range(1, num_factors + 1)}
      
      # Create a continuous target variable (e.g., normally distributed)
      data_dict["target"] = np.random.normal(loc=50, scale=10, size=num_rows)
      
      # Convert the dictionary into a Polars DataFrame
      df_reg = pl.DataFrame(data_dict)
      
      print("=== Regression Data Sample ===")
      print(df_reg.head())
      
      # List of factor columns to encode
      factor_columns = [f"Factor_{i}" for i in range(1, num_factors + 1)]
      
      # --- Test Target Encoding for Regression ---
      print("\n=== Testing Target Encoding for Regression ===")
      result_target_reg = character.categorical_encoding(
          data=df_reg,
          ML_Type="regression",
          group_variables=factor_columns,
          target_variable="target",
          method="target_encoding",
          save_path=None,          # Not saving to disk in this test
          scoring=False,           # Training mode
          keep_original_factors=True,
          debug=True,              # Enable debug prints
      )
      
      # If the function returns a dict (with mapping components), extract the data
      if isinstance(result_target_reg, dict):
          encoded_df_target_reg = result_target_reg["data"]
      else:
          encoded_df_target_reg = result_target_reg
      
      print("\n--- Target Encoding (Regression) Result Sample ---")
      print(encoded_df_target_reg.head())
      
      # --- Test James–Stein Encoding for Regression ---
      print("\n=== Testing James–Stein Encoding for Regression ===")
      result_js_reg = character.categorical_encoding(
          data=df_reg,
          ML_Type="regression",
          group_variables=factor_columns,
          target_variable="target",
          method="james-stein",
          save_path=None,          # Not saving to disk in this test
          scoring=False,           # Training mode
          keep_original_factors=False,
          debug=True,              # Enable debug prints
      )
      
      if isinstance(result_js_reg, dict):
          encoded_df_js_reg = result_js_reg["data"]
      else:
          encoded_df_js_reg = result_js_reg
      
      print("\n--- James–Stein Encoding (Regression) Result Sample ---")
      print(encoded_df_js_reg.head())
      
      
      import os
      import numpy as np
      import polars as pl
      
      # ------------------------------------------------------------------------------
      # MultiClass target version
      # ------------------------------------------------------------------------------
      
      # Create a fake multiclass dataset
      np.random.seed(42)
      
      num_rows = 1000
      num_factors = 5
      # For our categorical factors, use 5 possible levels.
      factor_categories = ["A", "B", "C", "D", "E"]
      
      # Build fake data for factors
      data_dict = {f"Factor_{i}": np.random.choice(factor_categories, size=num_rows)
                   for i in range(1, num_factors + 1)}
      
      # Create a categorical target variable with more than 2 levels.
      target_categories = ["class1", "class2", "class3"]
      # Optionally, you can set probabilities for each class.
      data_dict["target_class"] = np.random.choice(target_categories, size=num_rows, p=[0.3, 0.4, 0.3])
      
      # Convert the dictionary into a Polars DataFrame
      df_multi = pl.DataFrame(data_dict)
      
      print("=== Multiclass Data Sample ===")
      print(df_multi.head())
      
      # List of factor columns to encode
      factor_columns = [f"Factor_{i}" for i in range(1, num_factors + 1)]
      
      # --- Test Target Encoding for Multiclass ---
      print("\n=== Testing Target Encoding for Multiclass ===")
      result_target_multi = character.categorical_encoding(
          data=df_multi,
          ML_Type="multiclass",
          group_variables=factor_columns,
          target_variable="target_class",
          method="target_encoding",
          save_path=None,          # Not saving to disk in this test
          scoring=False,           # Training mode
          keep_original_factors=False,
          debug=True,              # Enable debug prints
      )
      
      # If the function returns a dict (with mapping components), extract the data.
      if isinstance(result_target_multi, dict):
          encoded_df_target_multi = result_target_multi["data"]
      else:
          encoded_df_target_multi = result_target_multi
      
      print("\n--- Target Encoding (Multiclass) Result Sample ---")
      print(encoded_df_target_multi.head())
      
      # --- Test James–Stein Encoding for Multiclass ---
      print("\n=== Testing James–Stein Encoding for Multiclass ===")
      result_js_multi = character.categorical_encoding(
          data=df_multi,
          ML_Type="multiclass",
          group_variables=factor_columns,
          target_variable="target_class",
          method="james-stein",
          save_path=None,          # Not saving to disk in this test
          scoring=False,           # Training mode
          keep_original_factors=False,
          debug=True,              # Enable debug prints
      )
      
      if isinstance(result_js_multi, dict):
          encoded_df_js_multi = result_js_multi["data"]
      else:
          encoded_df_js_multi = result_js_multi
      
      print("\n--- James–Stein Encoding (Multiclass) Result Sample ---")
      print(encoded_df_js_multi.head())
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
    factor_components = {} if not scoring else None

    ML_Type = ML_Type.lower()

    # ----- TARGET ENCODING -----
    if method.lower() == "target_encoding":
        for group in group_variables:  # group = group_variables[0]
            if debug:
                print(f"Target encoding on '{group}'")

            if not scoring:
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

                if save_path:
                    os.makedirs(save_path, exist_ok=True)
                    csv_file = os.path.join(save_path, f"{group}_TargetEncode.csv")
                    mapping_df.write_csv(csv_file)
            else:
                if supply_factor_level_list is not None and group in supply_factor_level_list:
                    mapping_df = supply_factor_level_list[group]
                    if not isinstance(mapping_df, pl.DataFrame):
                        mapping_df = pl.from_pandas(mapping_df)
                elif save_path:
                    csv_file = os.path.join(save_path, f"{group}_TargetEncode.csv")
                    mapping_df = pl.read_csv(csv_file)
                else:
                    raise ValueError(
                        "In scoring mode you must supply either a 'supply_factor_level_list' or a valid 'save_path'."
                    )

            data = data.join(mapping_df, on=group, how="left")
            if not keep_original_factors:
                data = data.drop(group)

            if scoring and impute_value_scoring is not None:
                if ML_Type == "multiclass":
                    for col in mapping_df.columns:
                        if col != group:
                            data = data.with_columns(pl.col(col).fill_null(impute_value_scoring))
                else:
                    new_col = f"{group}_TargetEncode"
                    data = data.with_columns(pl.col(new_col).fill_null(impute_value_scoring))
            if not scoring:
                factor_components[group] = mapping_df

        if not scoring and return_factor_level_list:
            return {"data": data, "factor_components": factor_components}
        else:
            return data

    # ----- JAMES–STEIN ENCODING -----
    elif method.lower() == "james-stein":
        for group in group_variables:  # group = group_variables[0]
            if debug:
                print(f"James–Stein encoding on '{group}'")

            if not scoring:
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
                        ((pl.col("TargetMean") * (1 - pl.col("TargetMean"))) / pl.col("TargetSum")).alias("TargetVariance"),
                        ((pl.col("TargetGroupMean") * (1 - pl.col("TargetGroupMean"))) / pl.col("N")).alias("TargetGroupVariance"),
                    ])
                    df_counts = df_counts.with_columns(
                        (pl.col("TargetGroupVariance") /
                         (pl.col("TargetGroupVariance") + pl.col("TargetVariance"))
                         ).alias("Z")
                    )
                    df_counts = df_counts.with_columns(
                        ((1 - pl.col("Z")) * pl.col("TargetGroupMean") + pl.col("Z") * pl.col("TargetMean")).alias(f"{group}_JamesStein")
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
                            (pl.col("Z") * pl.col("Mean") + (1 - pl.col("Z")) * pl.lit(grand_mean)
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
                        V = sum((m - grand_mean) ** 2 for m in group_means) / (num_groups - 1) if num_groups > 1 else 0
                        mapping_df = mapping_df.with_columns(
                            (pl.lit(V) - pl.col("EPV") / pl.col("N")).alias("VHM")
                        )
                        first_VHM = mapping_df.select("VHM").to_series()[0]
                        K = overall_epv / first_VHM if first_VHM != 0 else 0
                        mapping_df = mapping_df.with_columns(
                            (pl.col("N") / (pl.col("N") + pl.lit(K))).alias("Z")
                        )
                        mapping_df = mapping_df.with_columns(
                            (pl.col("Z") * pl.col("Mean") + (1 - pl.col("Z")) * pl.lit(grand_mean)
                             ).alias(f"{group}_JamesStein")
                        )
                        mapping_df = mapping_df.select([group, f"{group}_JamesStein"])
            else:
                if supply_factor_level_list is not None and group in supply_factor_level_list:
                    mapping_df = supply_factor_level_list[group]
                    if not isinstance(mapping_df, pl.DataFrame):
                        mapping_df = pl.from_pandas(mapping_df)
                elif save_path:
                    csv_file = os.path.join(save_path, f"{group}_JamesStein.csv")
                    mapping_df = pl.read_csv(csv_file)
                else:
                    raise ValueError(
                        "In scoring mode you must supply either a 'supply_factor_level_list' or a valid 'save_path'."
                    )

            if not scoring and save_path and method.lower() == "james-stein":
                csv_file = os.path.join(save_path, f"{group}_JamesStein.csv")
                mapping_df.write_csv(csv_file)

            data = data.join(mapping_df, on=group, how="left")
            if not keep_original_factors:
                data = data.drop(group)

            if scoring and impute_value_scoring is not None:
                if ML_Type == "multiclass":
                    for col in mapping_df.columns:
                        if col != group:
                            data = data.with_columns(pl.col(col).fill_null(impute_value_scoring))
                else:
                    new_col = f"{group}_JamesStein"
                    data = data.with_columns(pl.col(new_col).fill_null(impute_value_scoring))
            if not scoring:
                factor_components[group] = mapping_df

        if not scoring and return_factor_level_list:
            return {"data": data, "factor_components": factor_components}
        else:
            return data
