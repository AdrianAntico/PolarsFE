from typing import List, Union, Optional, Any
import polars as pl

def lags(
    data: pl.DataFrame,
    date_col: str,
    columns: List[str],
    lags: Union[int, List[int]] = 1,
    group_vars: Optional[List[str]] = None,
    fill_value: Optional[Any] = None,
    is_sorted: bool = False
) -> pl.DataFrame:
    """
    Create lag features for specified columns in a Polars DataFrame.

    The function first sorts the DataFrame by the date column (and, if provided, by the grouping variables)
    and then computes lag features for each column in `columns` for each lag in `lags`.

    Parameters:
      data (pl.DataFrame): The input DataFrame.
      date_col (str): The name of the datetime column to sort by.
      columns (List[str]): A list of column names for which to create lag features.
      lags (Union[int, List[int]]): A single lag or a list of lags to compute (default is 1).
      group_vars (Optional[List[str]]): A list of grouping columns. If provided, lag features are computed within each group.
      fill_value (Optional[Any]): An optional value to fill missing values generated by the lag operation.

    Returns:
      pl.DataFrame: A DataFrame with additional lag feature columns appended. Each new column is named
                    "{column}_lag_{lag}".

    Example:
      from PolarsFE import window

      # Create a sample DataFrame.
      df = pl.DataFrame({
          "date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06"],
          "sales": [100, 150, 200, 250, 300, 350],
          "store": ["A", "A", "B", "B", "A", "B"]
      })
      
      print("=== Original DataFrame ===")
      print(df)

      # Create lag features for "sales" with lags 1 and 2.
      # Compute lags within each store.
      df_lags = window.lags(df, date_col="date", columns=["sales"], lags=[1,2], group_vars=["store"], fill_value=0)

      print("\n=== DataFrame with Lag Features ===")
      print(df_lags)
    """
    # Ensure that lags is a list.
    if isinstance(lags, int):
        lags = [lags]
    
    # Sort the data: if grouping variables are provided, sort by group_vars and then by the date column.
    if not is_sorted:
        if group_vars is not None:
            sort_cols = group_vars + [date_col]
        else:
            sort_cols = [date_col]
        data_sorted = data.sort(sort_cols)
    
    # Build a list of lag expressions.
    lag_exprs = []
    for col in columns:
        for lag in lags:
            expr = pl.col(col).shift(lag)
            if group_vars is not None:
                # Compute lag values over each group.
                expr = expr.over(group_vars)
            if fill_value is not None:
                expr = expr.fill_null(fill_value)
            lag_exprs.append(expr.alias(f"{col}_lag_{lag}"))
    
    # Append the lag columns to the sorted DataFrame.
    data_out = data_sorted.with_columns(lag_exprs)
    return data_out


def rolling_features(
    data: pl.DataFrame,
    date_col: str,
    columns: List[str],
    window: Union[int, List[int]] = 3,
    agg: str = "mean",
    group_vars: Optional[List[str]] = None,
    fill_value: Optional[Any] = None,
    is_sorted: bool = False,
    min_samples: Optional[int] = 1,
    center: bool = False
) -> pl.DataFrame:
    """
    Create rolling window (moving aggregate) features for specified columns in a Polars DataFrame.
    
    The function optionally sorts the DataFrame by the date column (and group variables, if provided)
    and then computes the specified rolling aggregate for each column over the provided window size(s).
    
    Supported aggregation functions (case-insensitive) include:
      - "mean": Rolling mean.
      - "min": Rolling minimum.
      - "max": Rolling maximum.
      - "std": Rolling standard deviation.
      - "sum": Rolling sum.
    
    Parameters:
      data (pl.DataFrame): The input DataFrame.
      date_col (str): The name of the datetime column to sort by.
      columns (List[str]): A list of numeric column names for which to compute rolling features.
      window (Union[int, List[int]]): A single window size or a list of window sizes (in number of rows). Default is 3.
      agg (str): The aggregation function to use ("mean", "min", "max", "std", "sum"). Default is "mean".
      group_vars (Optional[List[str]]): If provided, the DataFrame is grouped by these columns and the rolling
                                        calculation is done within each group.
      fill_value (Optional[Any]): If provided, missing values produced by the rolling window are filled with this value.
      is_sorted (bool): If True, the function assumes the DataFrame is already sorted by the date column (and group_vars).
                        If False, the function sorts the DataFrame.
      min_samples (Optional[int]): Minimum number of observations required in the window to have a value. Default is 1.
      center (bool): If True, the window is centered on each row. Default is False.
    
    Returns:
      pl.DataFrame: The input DataFrame with new rolling feature columns appended. Each new column is named
                    "{column}_{agg}_rolling_{window}".
    
    Example:
      from PolarsFE import window
      
      # Create a sample DataFrame.
      df = pl.DataFrame({
          "date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06"],
          "sales": [100, 150, 200, 250, 300, 350],
          "store": ["A", "A", "B", "B", "A", "B"]
      })
      
      print("=== Original DataFrame ===")
      print(df)
      
      # Example 1: Compute rolling mean for "sales" with window sizes 2 and 3, grouped by "store".
      df_roll_mean = window.rolling_features(
          data=df,
          date_col="date",
          columns=["sales"],
          window=[2, 3],
          agg="mean",
          group_vars=["store"],
          fill_value=0,
          is_sorted=False,
          min_samples=1,
          center=False
      )
      print("\n=== DataFrame with Rolling Mean Features ===")
      print(df_roll_mean)
      
      # Example 2: Compute rolling standard deviation for "sales" with window size 3, grouped by "store".
      df_roll_std = window.rolling_features(
          data=df,
          date_col="date",
          columns=["sales"],
          window=3,
          agg="std",
          group_vars=["store"],
          fill_value=0,
          is_sorted=False,
          min_samples=1,
          center=False
      )
      print("\n=== DataFrame with Rolling Standard Deviation Features ===")
      print(df_roll_std)
    """
    # Ensure window is a list.
    if isinstance(window, int):
        window = [window]
    
    # Sort the DataFrame if not already sorted.
    if not is_sorted:
        if group_vars is not None:
            sort_cols = group_vars + [date_col]
        else:
            sort_cols = [date_col]
        data = data.sort(sort_cols)
    
    # Define a helper to get the rolling expression.
    def get_rolling_expr(col: str, w: int) -> pl.Expr:
        agg_lower = agg.lower()
        if agg_lower == "mean":
            expr = pl.col(col).rolling_mean(window_size=w, min_samples=min_samples, center=center)
        elif agg_lower == "min":
            expr = pl.col(col).rolling_min(window_size=w, min_samples=min_samples, center=center)
        elif agg_lower == "max":
            expr = pl.col(col).rolling_max(window_size=w, min_samples=min_samples, center=center)
        elif agg_lower == "std":
            expr = pl.col(col).rolling_std(window_size=w, min_samples=min_samples, center=center)
        elif agg_lower == "sum":
            expr = pl.col(col).rolling_sum(window_size=w, min_samples=min_samples, center=center)
        else:
            raise ValueError(f"Aggregation method '{agg}' is not supported.")
        return expr

    rolling_exprs = []
    for col in columns:
        for w in window:
            expr = get_rolling_expr(col, w)
            if group_vars is not None:
                expr = expr.over(group_vars)
            if fill_value is not None:
                expr = expr.fill_null(fill_value)
            rolling_exprs.append(expr.alias(f"{col}_{agg.lower()}_rolling_{w}"))
    
    return data.with_columns(rolling_exprs)


from typing import List, Union, Optional, Any
import polars as pl

def differences(
    data: pl.DataFrame,
    date_col: str,
    columns: List[str],
    diffs: Union[int, tuple[int, int], List[Union[int, tuple[int, int]]]] = 1,
    group_vars: Optional[List[str]] = None,
    fill_value: Optional[Any] = None,
    is_sorted: bool = False
) -> pl.DataFrame:
    """
    Compute differences for specified columns in a Polars DataFrame based on a given difference period.

    The function supports two types of difference specifications:
      - A single integer k: computes the difference between the current value and the value shifted by k.
        That is: diff = value - value.shift(k)
      - A tuple (k1, k2): computes the difference between the value shifted by k1 and the value shifted by k2.
        That is: diff = value.shift(k1) - value.shift(k2)
    
    If grouping variables are provided, the function computes differences within each group.
    If the data is not already sorted, the function will sort by the grouping variables (if provided) 
    and then by the date column.

    Parameters:
      data (pl.DataFrame): The input DataFrame.
      date_col (str): The name of the datetime column to sort by.
      columns (List[str]): The list of column names for which to compute differences.
      diffs (Union[int, tuple[int, int], List[Union[int, tuple[int, int]]]]): 
             A single difference specification or a list of them.
             For an integer k, the difference is computed as: current value - value.shift(k).
             For a tuple (k1, k2), the difference is computed as: value.shift(k1) - value.shift(k2).
             Default is 1.
      group_vars (Optional[List[str]]): A list of grouping columns. If provided, differences are computed within each group.
      fill_value (Optional[Any]): An optional value to fill missing values generated by the differencing.
      is_sorted (bool): If True, the function assumes data is already sorted (by group_vars and date_col) and skips sorting.

    Returns:
      pl.DataFrame: The input DataFrame with additional difference columns appended. New columns are named using the pattern:
                    - For a single integer k: "{column}_diff_{k}"
                    - For a tuple (k1, k2): "{column}_diff_{k1}_{k2}"

    Examples:
      from PolarsFE import window
      
      # Create a sample DataFrame.
      df = pl.DataFrame({
          "date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"],
          "sales": [100, 150, 200, 250, 300],
          "store": ["A", "A", "B", "B", "A"]
      })
      
      print("=== Original DataFrame ===")
      print(df)
      
      # Example 1: Compute difference with a single integer (lag=1) for "sales" within each store.
      df_diff1 = window.differences(
          data=df,
          date_col="date",
          columns=["sales"],
          diffs=1,
          group_vars=["store"],
          fill_value=0,
          is_sorted=False
      )
      print("\n=== DataFrame with Sales Difference (lag=1) ===")
      print(df_diff1)
      
      # Example 2: Compute difference using a tuple (e.g., difference between lag 1 and lag 2) for "sales" within each store.
      df_diff2 = window.differences(
          data=df,
          date_col="date",
          columns=["sales"],
          diffs=(1, 2),
          group_vars=["store"],
          fill_value=0,
          is_sorted=False
      )
      print("\n=== DataFrame with Sales Difference (lag 1 - lag 2) ===")
      print(df_diff2)
      
      # Example 3: Multiple difference specifications (lag 1 and tuple (1,2)) for "sales".
      df_diff3 = window.differences(
          data=df,
          date_col="date",
          columns=["sales"],
          diffs=[1, (1, 2)],
          group_vars=["store"],
          fill_value=0,
          is_sorted=False
      )
      print("\n=== DataFrame with Multiple Sales Differences ===")
      print(df_diff3)
    """
    # Ensure diffs is a list.
    if isinstance(diffs, (int, tuple)):
        diffs = [diffs]
    
    # Sort the data if not already sorted.
    if not is_sorted:
        if group_vars is not None:
            sort_cols = group_vars + [date_col]
        else:
            sort_cols = [date_col]
        data = data.sort(sort_cols)
    
    diff_exprs = []
    for col in columns:
        for diff_spec in diffs:
            if isinstance(diff_spec, int):
                expr = pl.col(col) - pl.col(col).shift(diff_spec)
                new_col_name = f"{col}_diff_{diff_spec}"
            elif isinstance(diff_spec, tuple) and len(diff_spec) == 2:
                k1, k2 = diff_spec
                expr = pl.col(col).shift(k1) - pl.col(col).shift(k2)
                new_col_name = f"{col}_diff_{k1}_{k2}"
            else:
                raise ValueError("Each difference specification must be either an integer or a tuple of two integers.")
            
            if group_vars is not None:
                expr = expr.over(group_vars)
            if fill_value is not None:
                expr = expr.fill_null(fill_value)
            diff_exprs.append(expr.alias(new_col_name))
    
    return data.with_columns(diff_exprs)
