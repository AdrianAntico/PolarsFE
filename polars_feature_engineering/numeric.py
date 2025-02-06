import math
from typing import List, Optional, Union, Tuple, Dict
import polars as pl


def standardize(
    data: pl.DataFrame,
    col_names: List[str],
    group_vars: Optional[List[str]] = None,
    center: bool = True,
    scale: bool = True,
    score_table: bool = False,
    mode: str = "train",  # "train", "apply", or "backtransform"
    score_table_data: Optional[pl.DataFrame] = None,
    debug: bool = False,
) -> Union[pl.DataFrame, Tuple[pl.DataFrame, pl.DataFrame]]:
    """
    Standardize numerical columns with optional grouping, and optionally provide a score table
    for later backtransformation or application on new data.
    
    In training mode (mode == "train"), the function computes, for each column in col_names,
    the mean and standard deviation (globally if group_vars is None, or per group otherwise)
    and then creates new standardized columns (named "<col>_Standardized").
    
    Optionally, if score_table is True, a score table (a DataFrame of group-wise means and sds)
    is returned along with the transformed data. This score table can then be used in scoring mode.
    
    In scoring mode:
      - If mode == "apply", the function applies the standardization transformation to new data,
        using the provided score_table_data.
      - If mode == "backtransform", the function reverses the transformation (backtransforms)
        the standardized values into their original scale using the provided score_table_data.
    
    Parameters:
      data (pl.DataFrame): Input data.
      col_names (List[str]): List of numerical column names to standardize.
      group_vars (Optional[List[str]]): Columns to group by. If None, standardization is computed globally.
      center (bool): Whether to subtract the mean (default True).
      scale (bool): Whether to divide by the standard deviation (default True).
      score_table (bool): In training mode, if True, also return a score table with group means and sds.
      mode (str): "train" to compute and apply standardization on training data,
                  "apply" to standardize new data using the provided score_table_data,
                  or "backtransform" to reverse the standardization.
      score_table_data (Optional[pl.DataFrame]): In scoring mode, the score table (with columns "<col>_mean" and "<col>_sd")
                                                 to use for transforming the new data.
      debug (bool): If True, print debug messages.
    
    Returns:
      If mode=="train" and score_table is True, returns a tuple (transformed_data, score_table).
      Otherwise, returns the transformed DataFrame.
    
    Examples:
      import numpy as np
      import polars as pl
      from polars_feature_engineering import standardize
      
      # Set seed for reproducibility
      np.random.seed(42)
      n = 100
      
      # Create a fake dataset with a grouping variable "Group" and two numeric variables.
      groups = np.random.choice(["A", "B", "C"], size=n)
      # Generate Value1 with different means per group.
      value1 = np.where(groups == "A", np.random.normal(50, 5, size=n),
                        np.where(groups == "B", np.random.normal(60, 5, size=n),
                                 np.random.normal(70, 5, size=n)))
      # Generate Value2 as a normally distributed variable.
      value2 = np.random.normal(100, 10, size=n)
      
      df = pl.DataFrame({
          "Group": groups,
          "Value1": value1,
          "Value2": value2
      })
      
      print("=== Original Dataset ===")
      print(df.head())
      
      # -------------------------------
      # TRAINING MODE: Compute standardization parameters by Group
      # -------------------------------
      # This call computes group-wise means and standard deviations for Value1 and Value2,
      # creates standardized columns, and returns a score table.
      transformed_train, score_tbl = standardize(
          data=df,
          col_names=["Value1", "Value2"],
          group_vars=["Group"],
          center=True,
          scale=True,
          score_table=True,
          mode="train",
          debug=True
      )
      
      print("\n=== Transformed Training Data ===")
      print(transformed_train.head())
      
      print("\n=== Score Table (Group-wise Means and SDs) ===")
      print(score_tbl)
      
      # -------------------------------
      # APPLICATION MODE: Apply standardization to new data using the score table
      # -------------------------------
      # Here we simulate new data by cloning the original dataset.
      # The new data does not have the standardized columns.
      new_data = df.clone()
      
      transformed_apply = standardize(
          data=new_data,
          col_names=["Value1", "Value2"],
          group_vars=["Group"],
          center=True,
          scale=True,
          mode="apply",
          score_table_data=score_tbl,
          debug=True
      )
      
      print("\n=== Transformed New Data (Standardized) ===")
      print(transformed_apply.head())
      
      # -------------------------------
      # BACKTRANSFORMATION MODE: Reverse the standardization on the new data
      # -------------------------------
      # This reverses the standardized values back to their original scale.
      backtransformed = standardize(
          data=transformed_apply,
          col_names=["Value1", "Value2"],
          group_vars=["Group"],
          center=True,
          scale=True,
          mode="backtransform",
          score_table_data=score_tbl,
          debug=True
      )
      
      print("\n=== Backtransformed Data (Reversed Standardization) ===")
      print(backtransformed.head())
    """
    
    if mode not in {"train", "apply", "backtransform"}:
        raise ValueError("mode must be one of 'train', 'apply', or 'backtransform'")
    
    # ----- TRAINING MODE -----
    if mode == "train":
        # No grouping: compute global means and sds.
        if not group_vars:
            means: Dict[str, float] = {col: data[col].mean() for col in col_names}
            sds: Dict[str, float] = {col: data[col].std() for col in col_names}
            
            # For each column, create a new standardized column.
            for col in col_names:
                expr = pl.col(col)
                if center:
                    expr = expr - means[col]
                if scale:
                    # Protect against division by zero.
                    expr = expr / sds[col] if sds[col] != 0 else expr
                data = data.with_columns(expr.alias(f"{col}_Standardized"))
            
            if score_table:
                # Create a one-row score table with the means and sds.
                mean_df = pl.DataFrame({f"{col}_mean": [means[col]] for col in col_names})
                sd_df = pl.DataFrame({f"{col}_sd": [sds[col]] for col in col_names})
                score_tbl = mean_df.hstack(sd_df)
                if debug:
                    print("Global score table created:")
                    print(score_tbl)
                return data, score_tbl
            else:
                return data
        else:
            # Grouped standardization.
            # Compute group-wise means.
            agg_exprs_mean = [pl.col(col).mean().alias(f"{col}_mean") for col in col_names]
            agg_exprs_sd = [pl.col(col).std().alias(f"{col}_sd") for col in col_names]
            score_tbl_mean = data.group_by(group_vars).agg(agg_exprs_mean)
            score_tbl_sd = data.group_by(group_vars).agg(agg_exprs_sd)
            # Join the means and sds.
            score_tbl = score_tbl_mean.join(score_tbl_sd, on=group_vars)
            if debug:
                print("Grouped score table:")
                print(score_tbl)
            # Join the score table back to the original data.
            data = data.join(score_tbl, on=group_vars, how="left")
            
            for col in col_names:
                mean_col = f"{col}_mean"
                sd_col = f"{col}_sd"
                expr = pl.col(col)
                if center:
                    expr = expr - pl.col(mean_col)
                if scale:
                    expr = expr / pl.col(sd_col)
                data = data.with_columns(expr.alias(f"{col}_Standardized"))
            
            if score_table:
                return data, score_tbl
            else:
                return data

    # ----- SCORING MODE -----
    else:
        if score_table_data is None:
            raise ValueError("score_table_data must be provided in scoring mode")
        
        if not group_vars:
            # Global standardization.
            for col in col_names:
                mean_val = score_table_data[f"{col}_mean"][0]
                sd_val = score_table_data[f"{col}_sd"][0]
                expr = pl.col(col)
                if center:
                    expr = expr - mean_val
                if scale:
                    expr = expr / sd_val if sd_val != 0 else expr
                data = data.with_columns(expr.alias(f"{col}_Standardized"))
            return data
        else:
            # For grouped data, join the score table with new data.
            data = data.join(score_table_data, on=group_vars, how="left")
            if mode == "apply":
                for col in col_names:
                    mean_col = f"{col}_mean"
                    sd_col = f"{col}_sd"
                    expr = pl.col(col)
                    if center:
                        expr = expr - pl.col(mean_col)
                    if scale:
                        expr = expr / pl.col(sd_col)
                    data = data.with_columns(expr.alias(f"{col}_Standardized"))
                # Optionally, drop the joined score table columns.
                drop_cols = [f"{col}_mean" for col in col_names] + [f"{col}_sd" for col in col_names]
                data = data.drop(drop_cols)
                return data
            elif mode == "backtransform":
                # Reverse transformation: new value = standardized_value * sd + mean.
                for col in col_names:  # col = col_names[0]
                    mean_col = f"{col}_mean"
                    sd_col = f"{col}_sd"
                    expr = pl.col(f"{col}_Standardized")
                    if scale:
                        expr = expr * pl.col(sd_col)
                    if center:
                        expr = expr + pl.col(mean_col)
                    data = data.with_columns(expr.alias(col))
                drop_cols = [f"{col}_mean" for col in col_names] + [f"{col}_sd" for col in col_names]
                data = data.drop(drop_cols)
                return data


def percent_rank(
    data: pl.DataFrame,
    col_names: List[str],
    group_vars: Optional[List[str]] = None,
    granularity: float = 0.001,
    mode: str = "train",  # "train", "apply", or "backtransform"
    score_table_data: Optional[pl.DataFrame] = None,
    roll_direction: str = "forward",
    score_table: bool = False,
    debug: bool = False,
) -> Union[pl.DataFrame, Tuple[pl.DataFrame, pl.DataFrame]]:
    """
    Generate percent ranks for specified columns in a Polars DataFrame.
    
    In **training mode** (mode=="train"), for each column in `col_names` the function computes
    the percent rank as:
    
        PercRank = (rank(x) / N)
    
    where N is the number of rows (or the group size, if grouping is applied). The result is then
    rounded to a number of decimals determined by the `granularity` parameter (e.g. granularity=0.001
    yields 3 decimals). The new column is named "<col>_PercRank". If `score_table` is True, the function
    also returns a score table containing the unique mappings (and, if grouping is used, the group columns)
    of the original values and their computed percent ranks.
    
    In **apply mode** (mode=="apply"), the function takes new data along with a previously generated
    score table (passed via `score_table_data`) and uses an asof join (with a specified roll direction)
    to assign the appropriate percent rank to each row.
    
    In **backtransform mode** (mode=="backtransform"), the function uses the provided score table
    to reverse the transformation. Given a percent rank (in a column named "<col>_PercRank"), an asof join
    is performed to look up the corresponding original value.
    
    Parameters:
      data (pl.DataFrame): Input DataFrame.
      col_names (List[str]): List of numeric column names for which to compute percent ranks.
      group_vars (Optional[List[str]]): Optional grouping columns. If provided, percent ranks are computed
                                          within each group.
      granularity (float): Granularity for rounding the percent rank (e.g. 0.001 rounds to 3 decimals).
      mode (str): One of "train", "apply", or "backtransform".
      score_table_data (Optional[pl.DataFrame]): In "apply" or "backtransform" mode, the score table generated during training.
      roll_direction (str): For "apply" or "backtransform" modes, the rolling join strategy ("forward", "backward", or "nearest").
      score_table (bool): In training mode, if True the function returns a tuple (transformed_data, score_table).
      debug (bool): If True, print debug information.
    
    Returns:
      - In training mode with score_table=True: Tuple[pl.DataFrame, pl.DataFrame] where the first element is the
        DataFrame with new percent rank columns and the second is the score table.
      - Otherwise, returns a single pl.DataFrame.
    
    Examples:
      import numpy as np
      import polars as pl
      from polars_feature_engineering import percent_rank
      
      # Set seed for reproducibility
      np.random.seed(42)
      n = 100
      
      # Create a fake dataset with:
      # - A grouping variable "Group" (levels: "A", "B", "C")
      # - Two numeric columns "Value1" and "Value2"
      groups = np.random.choice(["A", "B", "C"], size=n)
      value1 = np.random.normal(50, 10, size=n)
      value2 = np.random.normal(100, 20, size=n)
      
      df = pl.DataFrame({
          "Group": groups,
          "Value1": value1,
          "Value2": value2
      })
      
      print("=== Original Training Data ===")
      print(df.head())
      
      # --------------
      # TRAINING MODE: Compute percent ranks by Group for Value1 and Value2.
      # --------------
      transformed_train, score_tbl = percent_rank(
          data=df,
          col_names=["Value1", "Value2"],
          group_vars=["Group"],
          granularity=0.001,
          mode="train",
          score_table=True,
          debug=True
      )
      
      print("\n=== Transformed Training Data with Percent Ranks ===")
      print(transformed_train.head())
      
      print("\n=== Score Table ===")
      print(score_tbl)
      
      # --------------
      # APPLY MODE: Use the score table to assign percent ranks to new data.
      # --------------
      # Simulate new data.
      new_groups = np.random.choice(["A", "B", "C"], size=n)
      new_value1 = np.random.normal(50, 10, size=n)
      new_value2 = np.random.normal(100, 20, size=n)
      new_df = pl.DataFrame({
          "Group": new_groups,
          "Value1": new_value1,
          "Value2": new_value2
      })
      
      print("\n=== Original New Data ===")
      print(new_df.head())
      
      transformed_new = percent_rank(
          data=new_df,
          col_names=["Value1", "Value2"],
          group_vars=["Group"],
          granularity=0.001,
          mode="apply",
          score_table_data=score_tbl,
          roll_direction="nearest",
          debug=True
      )
      
      print("\n=== Transformed New Data with Percent Ranks (Applied) ===")
      print(transformed_new.head())
      
      # --------------
      # BACKTRANSFORM MODE: Reverse the percent rank transformation to recover original values.
      # --------------
      # For demonstration, use the new data with percent rank columns (from the apply mode).
      backtransformed = percent_rank(
          data=transformed_new,
          col_names=["Value1", "Value2"],
          group_vars=["Group"],
          granularity=0.001,
          mode="backtransform",
          score_table_data=score_tbl,
          roll_direction="nearest",
          debug=True
      )
      
      print("\n=== Backtransformed Data (Recovered Original Values) ===")
      print(backtransformed.head())
    """

    # Determine the number of decimals to round to based on granularity.
    decimals = int(-math.log10(granularity)) if granularity < 1 else 0

    if mode not in {"train", "apply", "backtransform"}:
        raise ValueError("mode must be one of 'train', 'apply', or 'backtransform'")

    # --------------------------
    # TRAINING MODE: Compute percent ranks.
    # --------------------------
    if mode == "train":
        if not group_vars or len(group_vars) == 0:
            N = data.height
            for col in col_names:
                data = data.with_columns(
                    ((pl.col(col).rank("ordinal") / N).round(decimals)).alias(f"{col}_PercRank")
                )
        else:
            for col in col_names:
                data = data.with_columns(
                    ((pl.col(col).rank("ordinal").over(group_vars) / pl.len().over(group_vars)).round(decimals))
                    .alias(f"{col}_PercRank")
                )
        if score_table:
            if not group_vars or len(group_vars) == 0:
                score_tbl = data.select(col_names + [f"{col}_PercRank" for col in col_names]).unique()
            else:
                score_tbl = data.select(group_vars + col_names + [f"{col}_PercRank" for col in col_names]).unique()
            if debug:
                print("Score Table:")
                print(score_tbl)
            return data, score_tbl
        else:
            return data

    # --------------------------
    # APPLY MODE: Use score table to assign percent ranks.
    # --------------------------
    elif mode == "apply":
        if score_table_data is None:
            raise ValueError("In apply mode, score_table_data must be provided.")
        new_data = data
        for col in col_names:
            key = col  # join key is the original column value
            perc_col = f"{col}_PercRank"
            if not group_vars or len(group_vars) == 0:
                new_data = new_data.sort(key)
                score_tbl_col = score_table_data.select([key, perc_col]).sort(key)
                new_data = new_data.join_asof(
                    score_tbl_col,
                    left_on=key,
                    right_on=key,
                    strategy=roll_direction.lower(),
                )
            else:
                sort_keys = group_vars + [key]
                new_data = new_data.sort(sort_keys)
                score_tbl_col = score_table_data.select(group_vars + [key, perc_col]).sort(sort_keys)
                new_data = new_data.join_asof(
                    score_tbl_col,
                    left_on=key,
                    right_on=key,
                    by=group_vars,
                    strategy=roll_direction.lower(),
                )
        return new_data

    # --------------------------
    # BACKTRANSFORM MODE: Reverse the percent rank transformation.
    # --------------------------
    elif mode == "backtransform":
        if score_table_data is None:
            raise ValueError("In backtransform mode, score_table_data must be provided.")
        new_data = data
        for col in col_names:  # col = col_names[0]
            key = f"{col}_PercRank"  # now we join on the percent rank column
            # The score table is assumed to have both the original column (col) and the corresponding percent rank.
            if not group_vars or len(group_vars) == 0:
                new_data = new_data.sort(key)
                score_tbl_col = score_table_data.select([col, key]).sort(key)
                new_data = new_data.join_asof(
                    score_tbl_col,
                    left_on=key,
                    right_on=key,
                    strategy=roll_direction.lower(),
                )
            else:
                sort_keys = group_vars + [key]
                new_data = new_data.sort(sort_keys)
                score_tbl_col = score_table_data.select(group_vars + [col, key]).sort(sort_keys)
                new_data = new_data.join_asof(
                    score_tbl_col,
                    left_on=key,
                    right_on=key,
                    by=group_vars,
                    strategy=roll_direction.lower(),
                )
        # Optionally, drop the percent rank columns.
        new_data = new_data.drop([f"{col}_PercRank" for col in col_names])
        return new_data


def numeric_transform(
    data: pl.DataFrame,
    col_names: List[str],
    transformation: str,
    mode: str = "apply",  # "apply" for forward transformation, "backtransform" for inverse
    A: Optional[float] = None,
    debug: bool = False,
    modify_original: bool = False,
) -> pl.DataFrame:
    """
    Apply or backtransform numeric transformations on specified columns of a Polars DataFrame.
    
    Supported transformations (the transformation parameter is case-insensitive):
    
      - "Asinh":  
          Forward:  y = asinh(x) = log(x + sqrt(x² + 1))
          Inverse:  x = sinh(y)
      
      - "Log":  
          Forward:  y = log(x)
          Inverse:  x = exp(y)
      
      - "LogPlusA":  
          Forward:  y = log(x + A)  
          A is chosen so that (x + A) ≥ 1. In apply mode, if A is None, then for each column 
                   A is computed as A = max(0, 1 - min(x)).
          Inverse:  x = exp(y) - A
      
      - "Sqrt":  
          Forward:  y = sqrt(x)
          Inverse:  x = y²
      
      - "Asin":  
          Forward:  y = arcsin(x)  (x must be in [-1,1])
          Inverse:  x = sin(y)
      
      - "Logit":  
          Forward:  y = log( x / (1-x) )  (x must be in (0,1))
          Inverse:  x = exp(y) / (1 + exp(y))
    
    Parameters:
      data (pl.DataFrame): Input DataFrame.
      col_names (List[str]): List of column names to transform.
         • In "apply" mode these are the original columns.
         • In "backtransform" mode, these should be the names of the forward‑transformed columns.
      transformation (str): The transformation to apply (one of the supported names).
      mode (str): "apply" to perform the forward transformation,
                  "backtransform" to compute the inverse transformation.
      A (Optional[float]): Constant to add for LogPlusA. If None (in apply mode),
                           A is computed for each column as max(0, 1 - min(x)).
      debug (bool): If True, print debug information.
      modify_original (bool): If True, the transformation is applied in place (i.e., the original column is modified).
                                If False, a new column is created for the transformation.
    
    Returns:
      A Polars DataFrame with the transformed columns.
      
      • In "apply" mode:
         - If modify_original is False, for each original column `col` a new column named 
           `"{col}_{transformation_lowercase}"` is created.
         - If modify_original is True, the original column is replaced with its transformed values.
      • In "backtransform" mode:
         - If modify_original is False, for each column in `col_names` (which should be the transformed columns)
           a new column named `"{original}_{transformation_lowercase}_back"` is created, where `original`
           is derived by removing the transformation suffix from the supplied name (if present).
         - If modify_original is True, the original transformed column is replaced with the backtransformed values.
    
    Examples:
      import numpy as np
      import polars as pl
      from polars_feature_engineering import numeric_transform
      
      # Create a fake dataset.
      np.random.seed(42)
      n = 10
      df = pl.DataFrame({
          "Positive": np.random.uniform(5, 100, size=n),    # for BoxCox, Log, LogPlusA, Sqrt
          "AnyValue": np.random.uniform(-50, 50, size=n),     # for YeoJohnson
          "Angle": np.random.uniform(-1, 1, size=n),          # for Asin (input should be in [-1,1])
          "Probability": np.random.uniform(0.01, 0.99, size=n)  # for Logit (values in (0,1))
      })
      
      print("=== Original Data ===")
      print(df)
      
      # --------------------------
      # Log Transformation
      # --------------------------
      df_log = numeric_transform(df, col_names=["Positive"], transformation="Log", mode="apply", debug=True)
      print("\n=== Log Applied ===")
      print(df_log.select(["Positive", "Positive_log"]))
      
      df_log_back = numeric_transform(df_log, col_names=["Positive_log"], transformation="Log", mode="backtransform", debug=True)
      print("\n=== Log Backtransformed ===")
      print(df_log_back.select(["Positive_log", "Positive_log_back"]))
      
      # --------------------------
      # LogPlusA Transformation
      # --------------------------
      df_logplusa = numeric_transform(df, col_names=["Positive"], transformation="LogPlusA", mode="apply", A=None, debug=True)
      print("\n=== LogPlusA Applied ===")
      print(df_logplusa.select(["Positive", "Positive_logplusa"]))
      
      # For backtransformation, you must supply the same A. Compute it from the original column.
      min_val = df.select(pl.col("Positive")).min().item()
      A_val = max(1, 1 - min_val)
      df_logplusa_back = numeric_transform(df_logplusa, col_names=["Positive_logplusa"], transformation="LogPlusA", mode="backtransform", A=A_val, debug=True)
      print("\n=== LogPlusA Backtransformed ===")
      print(df_logplusa_back.select(["Positive_logplusa", "Positive_logplusa_back"]))
      
      # --------------------------
      # Sqrt Transformation
      # --------------------------
      df_sqrt = numeric_transform(df, col_names=["Positive"], transformation="Sqrt", mode="apply", debug=True)
      print("\n=== Sqrt Applied ===")
      print(df_sqrt.select(["Positive", "Positive_sqrt"]))
      
      df_sqrt_back = numeric_transform(df_sqrt, col_names=["Positive_sqrt"], transformation="Sqrt", mode="backtransform", debug=True)
      print("\n=== Sqrt Backtransformed ===")
      print(df_sqrt_back.select(["Positive_sqrt", "Positive_sqrt_back"]))
      
      # --------------------------
      # Asin Transformation
      # --------------------------
      df_asin = numeric_transform(df, col_names=["Angle"], transformation="Asin", mode="apply", debug=True)
      print("\n=== Asin Applied ===")
      print(df_asin.select(["Angle", "Angle_asin"]))
      
      df_asin_back = numeric_transform(df_asin, col_names=["Angle"], transformation="Asin", mode="backtransform", debug=True)
      print("\n=== Asin Backtransformed ===")
      print(df_asin_back.select(["Angle_asin", "Angle_asin_back"]))
      
      # --------------------------
      # Logit Transformation
      # --------------------------
      df_logit = numeric_transform(df, col_names=["Probability"], transformation="Logit", mode="apply", debug=True)
      print("\n=== Logit Applied ===")
      print(df_logit.select(["Probability", "Probability_logit"]))
      
      df_logit_back = numeric_transform(df_logit, col_names=["Probability"], transformation="Logit", mode="backtransform", debug=True)
      print("\n=== Logit Backtransformed ===")
      print(df_logit_back.select(["Probability_logit", "Probability_logit_back"]))

    """

    # Work on a clone so the original data remains unchanged.
    data = data.clone()
    trans = transformation.lower()

    # Define helper function
    def _back_col_name(col: str, trans: str) -> str:
        """
        If the given column name ends with '_{trans}', remove that suffix;
        then return f"{base}_{trans}_back".
        """
        suffix = f"_{trans}"
        if col.lower().endswith(suffix):
            base = col[:-len(suffix)]
        else:
            base = col
        return f"{base}_{trans}_back"

    # Create transformation
    for col in col_names:
        if trans == "log":
            if mode == "apply":
                new_expr = pl.col(col).log()
                new_col_name = f"{col}_log"
            elif mode == "backtransform":
                new_expr = pl.col(col).exp()
                new_col_name = _back_col_name(col, "log")
            else:
                raise ValueError("mode must be 'apply' or 'backtransform'")
        
        elif trans == "logplusa":
            if mode == "apply":
                if A is None:
                    min_val = data.select(pl.col(col)).min().item()
                    a_val = max(1, 1 - min_val)
                else:
                    a_val = A
                if debug:
                    print(f"For column {col}, LogPlusA constant A = {a_val}")
                # Only add a_val if it is negative; otherwise, leave the value unchanged.
                if a_val < 0:
                    new_expr = (pl.col(col) + a_val).log()
                else:
                    new_expr = pl.col(col).log()
                new_col_name = f"{col}_logplusa"
                
            elif mode == "backtransform":
                if A is None:
                    raise ValueError("A must be provided for backtransformation of LogPlusA")
                else:
                    a_val = A
                # Only remove a_val (i.e., subtract it) if it is negative; otherwise, just take the exponential.
                if a_val < 0:
                    new_expr = pl.col(col).exp() - a_val
                else:
                    new_expr = pl.col(col).exp()
                new_col_name = _back_col_name(col, "logplusa")
                
            else:
                raise ValueError("mode must be 'apply' or 'backtransform'")
        
        elif trans == "sqrt":
            if mode == "apply":
                new_expr = pl.col(col).sqrt()
                new_col_name = f"{col}_sqrt"
            elif mode == "backtransform":
                new_expr = pl.col(col) ** 2
                new_col_name = _back_col_name(col, "sqrt")
            else:
                raise ValueError("mode must be 'apply' or 'backtransform'")
        
        elif trans == "asin":
            if mode == "apply":
                new_expr = pl.col(col).asin()
                new_col_name = f"{col}_asin"
            elif mode == "backtransform":
                new_expr = pl.col(col).sin()
                new_col_name = _back_col_name(col, "asin")
            else:
                raise ValueError("mode must be 'apply' or 'backtransform'")
        
        elif trans == "logit":
            if mode == "apply":
                new_expr = (pl.col(col) / (1 - pl.col(col))).log()
                new_col_name = f"{col}_logit"
            elif mode == "backtransform":
                new_expr = pl.col(col).exp() / (1 + pl.col(col).exp())
                new_col_name = _back_col_name(col, "logit")
            else:
                raise ValueError("mode must be 'apply' or 'backtransform'")
        
        else:
            raise ValueError(f"Transformation '{transformation}' is not supported.")
        
        # Decide whether to overwrite the original column or create a new one.
        target_col = col if modify_original else new_col_name
        data = data.with_columns(new_expr.alias(target_col))
    
    return data
