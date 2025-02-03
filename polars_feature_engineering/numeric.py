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


