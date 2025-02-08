import polars as pl
import numpy as np
from typing import List, Optional, Any, Union
import math
from collections import Counter


def partition_random(
    data: pl.DataFrame,
    num_partitions: int,
    seed: Optional[int] = None,
    percentages: Optional[List[float]] = None
) -> List[pl.DataFrame]:
    """
    Randomly partition a Polars DataFrame into a specified number of subsets.
    
    The function shuffles the rows of the input DataFrame and then splits the shuffled
    DataFrame into `num_partitions` smaller DataFrames. By default, the data is split into equally sized
    partitions. However, if a list of percentages is supplied (one percentage per partition), the data
    will be split according to those percentages.
    
    Parameters:
      data (pl.DataFrame): The input DataFrame to partition.
      num_partitions (int): The number of partitions to create (must be >= 1).
      seed (Optional[int]): An optional random seed to ensure reproducibility.
      percentages (Optional[List[float]]): A list of fractions (e.g. [0.3, 0.3, 0.4]) specifying the fraction
                                             of rows to allocate to each partition. If supplied, its length must
                                             equal num_partitions. If not supplied, the DataFrame is split equally.
    
    Returns:
      List[pl.DataFrame]: A list of DataFrames, each being one of the partitions.
    
    Examples:
      from PolarsFE import datasets

      df = pl.DataFrame({
         "id": np.arange(1, 101),
         "value": np.random.rand(100) * 100,
         "category": np.random.choice(["A", "B", "C"], size=100)
      })
      print("=== Original Dataset ===")
      print(df.head(10))
      print(f"Total rows: {df.height}\\n")
      # Partition into 3 equally sized parts with seed=42.
      parts_equal = partition_random(data=df, num_partitions=3, seed=42)
      for idx, part in enumerate(parts_equal, start=1):
         print(f"--- Equal Partition {idx} (rows: {part.height}) ---")
         print(part)
         print()
      # Partition into 3 parts using percentages (30%, 30%, 40%).
      parts_pct = partition_random(data=df, num_partitions=3, seed=42, percentages=[0.3, 0.3, 0.4])
      for idx, part in enumerate(parts_pct, start=1):
         print(f"--- Percentage Partition {idx} (rows: {part.height}) ---")
         print(part)
         print()
    """

    if num_partitions < 1:
        raise ValueError("num_partitions must be at least 1")
    
    # Validate and normalize percentages if provided.
    if percentages is not None:
        if len(percentages) != num_partitions:
            raise ValueError("Length of percentages list must equal num_partitions")
        total = sum(percentages)
        # Normalize so that they sum to 1.
        percentages = [p / total for p in percentages]
    
    # Shuffle the DataFrame.
    shuffled = data.sample(fraction=1.0, with_replacement=False, shuffle=True, seed=seed)
    n = shuffled.height

    if percentages is not None:
        # Compute partition sizes based on percentages.
        sizes = [int(n * p) for p in percentages]
        total_assigned = sum(sizes)
        remainder = n - total_assigned
        # Distribute any remaining rows one per partition until exhausted.
        for i in range(remainder):
            sizes[i] += 1
    else:
        # Equally sized partitions.
        base_size = n // num_partitions
        remainder = n % num_partitions
        sizes = [base_size + 1 if i < remainder else base_size for i in range(num_partitions)]
    
    partitions = []
    start = 0
    for size in sizes:
        partitions.append(shuffled.slice(start, size))
        start += size

    return partitions


def partition_time(
    data: pl.DataFrame,
    time_col: str,
    num_partitions: int,
    percentages: Optional[List[float]] = None
) -> List[pl.DataFrame]:
    """
    Partition a Polars DataFrame based on a time column.

    This function supports two modes:

    1. **Equal‑time partitions (default):**  
       The overall time range (from the minimum to maximum value in the time column) is split
       into `num_partitions` equal intervals (in nanoseconds), and each returned partition contains
       the rows whose time value (cast to Int64) falls into that interval.  
       
    2. **Percentage‑based partitions:**  
       If a list of percentages is provided, then the DataFrame is first sorted by the time column.
       The DataFrame is then split into partitions so that the number of rows in each partition is determined 
       by the supplied percentages. (The percentages will be normalized if they do not sum to 1.)
    
    Parameters:
      data (pl.DataFrame): The input DataFrame.
      time_col (str): The name of the time column (should be of Datetime type).
      num_partitions (int): The number of partitions to create (must be >= 1).
      percentages (Optional[List[float]]): A list of fractions specifying the row proportions for each partition.
                                           Must have length equal to num_partitions if provided.
    
    Returns:
      List[pl.DataFrame]: A list of DataFrames, each corresponding to one partition.
    
    Examples:
      from PolarsFE import datasets
      import datetime

      # Create a DataFrame with dates spanning 100 days.
      dates = [datetime.datetime(2023, 1, 1) + datetime.timedelta(days=i) for i in range(100)]
      df = pl.DataFrame({
          "date": dates,
          "value": np.random.rand(100)
      })
      # Partition into 4 equal time intervals.
      parts_equal = partition_time(df, time_col="date", num_partitions=4)
      for idx, part in enumerate(parts_equal, start=1):
          print(f"--- Equal Partition {idx} (rows: {part.height}) ---")
          print(part)
      
      # Partition into 4 parts using percentages: 10%, 20%, 30%, 40%.
      parts_pct = partition_time(df, time_col="date", num_partitions=4, percentages=[0.1, 0.2, 0.3, 0.4])
      for idx, part in enumerate(parts_pct, start=1):
          print(f"--- Percentage Partition {idx} (rows: {part.height}) ---")
          print(part)
    """

    if num_partitions < 1:
        raise ValueError("num_partitions must be at least 1")

    # First, sort the data by the time column.
    sorted_data = data.sort(time_col)

    if percentages is not None:
        # Validate and normalize percentages.
        if len(percentages) != num_partitions:
            raise ValueError("Length of percentages list must equal num_partitions")
        total = sum(percentages)
        percentages = [p / total for p in percentages]
        
        n = sorted_data.height
        sizes = [int(n * p) for p in percentages]
        total_assigned = sum(sizes)
        remainder = n - total_assigned
        for i in range(remainder):
            sizes[i] += 1

        partitions = []
        start = 0
        for size in sizes:
            partitions.append(sorted_data.slice(start, size))
            start += size
        return partitions
    else:
        # Equal-time partitions based on the overall time range.
        # Polars stores datetime values as integers (nanoseconds).
        min_ns = sorted_data.select(pl.col(time_col).cast(pl.Int64).min()).item()
        max_ns = sorted_data.select(pl.col(time_col).cast(pl.Int64).max()).item()

        # Create equally spaced boundaries.
        boundaries = np.linspace(min_ns, max_ns, num=num_partitions+1)

        partitions = []
        for i in range(num_partitions):
            lower = int(boundaries[i])
            upper = int(boundaries[i+1])
            if i < num_partitions - 1:
                part = sorted_data.filter(
                    (pl.col(time_col).cast(pl.Int64) >= lower) &
                    (pl.col(time_col).cast(pl.Int64) < upper)
                )
            else:
                # Include the final boundary.
                part = sorted_data.filter(
                    (pl.col(time_col).cast(pl.Int64) >= lower) &
                    (pl.col(time_col).cast(pl.Int64) <= upper)
                )
            partitions.append(part)
        return partitions


def partition_timeseries(
    data: pl.DataFrame,
    time_col: str,
    panel_vars: List[str],
    num_partitions: int,
    percentages: Optional[List[float]] = None
) -> List[pl.DataFrame]:
    """
    Partition a time‑series DataFrame into subsets based on a datetime column while ensuring 
    that every panel (as defined by the categorical columns in `panel_vars`) is included in each partition.
    
    The function supports two partitioning strategies:
    
    1. **Equal‑Time Intervals (default):**  
       The overall time range (from the minimum to maximum value in the time column) is determined
       (with datetime values cast to nanoseconds). The range is then split into `num_partitions` equal intervals.
       Rows are assigned to a partition if their time value falls into the corresponding interval.
       
    2. **Percentage‑Based Partitioning:**  
       If a list of percentages is provided (with length equal to `num_partitions`), the DataFrame (after sorting by time)
       is split into partitions so that the number of rows in each partition is determined by the supplied percentages.
       The percentages will be normalized if they do not sum to 1.
       
    **After time-based partitioning, the function ensures that every panel is included in each partition.**
    It does so by performing a left join with the unique panels (as defined by `panel_vars`). Any missing values 
    in the non‑panel columns are then filled with 0 (instead of leaving them as null).
    
    Parameters:
      data (pl.DataFrame): The input DataFrame.
      time_col (str): The name of the datetime column.
      panel_vars (List[str]): List of column names that define the panels (e.g. region, store, etc.).
      num_partitions (int): The number of partitions to create (must be >= 1).
      percentages (Optional[List[float]]): A list of fractions (e.g. [0.1, 0.2, 0.3, 0.4]) specifying the row proportions 
                                           for each partition. If provided, its length must equal num_partitions.
    
    Returns:
      List[pl.DataFrame]: A list of DataFrames, each corresponding to one partition. Every partition will contain 
      every unique panel (with missing values replaced by 0 if that panel had no rows in that time interval).
    
    Examples:
      from PolarsFE import datasets
      import datetime
      
      # Create a fake dataset with dates spanning 100 days and a panel column.
      dates = [datetime.datetime(2023, 1, 1) + datetime.timedelta(days=i) for i in range(100)]
      df = pl.DataFrame({
          "date": dates,
          "value": np.random.rand(100) * 100,
          "panel": np.random.choice(["A", "B", "C"], size=100)
      })
      
      print("=== Original Dataset (first 10 rows) ===")
      print(df.head(10))
      print(f"Total rows: {df.height}\n")
      
      # --- Test 1: Equal-Time Partitions ---
      print("=== Equal-Time Partitions ===")
      parts_equal = partition_timeseries(df, time_col="date", panel_vars=["panel"], num_partitions=4)
      for idx, part in enumerate(parts_equal, start=1):
          print(f"--- Equal Partition {idx} (rows: {part.height}) ---")
          print(part)
          print()
      
      # --- Test 2: Percentage-Based Partitions ---
      # For example, partition into 4 parts using percentages [0.1, 0.2, 0.3, 0.4].
      print("=== Percentage-Based Partitions ===")
      parts_pct = partition_timeseries(df, time_col="date", panel_vars=["panel"], num_partitions=4, percentages=[0.1, 0.2, 0.3, 0.4])
      for idx, part in enumerate(parts_pct, start=1):
          print(f"--- Percentage Partition {idx} (rows: {part.height}) ---")
          print(part)
          print()
    """

    if num_partitions < 1:
        raise ValueError("num_partitions must be at least 1")

    # Get the unique panels.
    unique_panels = data.select(panel_vars).unique()

    # Sort data by the time column.
    sorted_data = data.sort(time_col)

    if percentages is not None:
        # Validate and normalize percentages.
        if len(percentages) != num_partitions:
            raise ValueError("Length of percentages list must equal num_partitions")
        total = sum(percentages)
        percentages = [p / total for p in percentages]

        n = sorted_data.height
        sizes = [int(n * p) for p in percentages]
        total_assigned = sum(sizes)
        remainder = n - total_assigned
        for i in range(remainder):
            sizes[i] += 1

        partitions = []
        start = 0
        for size in sizes:
            part = sorted_data.slice(start, size)
            # Left join unique panels and fill missing values with 0.
            part_complete = unique_panels.join(part, on=panel_vars, how="left").fill_null(0)
            partitions.append(part_complete)
            start += size
        return partitions
    else:
        # Equal time intervals based on the overall time range.
        min_ns = sorted_data.select(pl.col(time_col).cast(pl.Int64).min()).item()
        max_ns = sorted_data.select(pl.col(time_col).cast(pl.Int64).max()).item()

        # Create equally spaced boundaries (num_partitions+1 boundaries).
        boundaries = np.linspace(min_ns, max_ns, num=num_partitions+1)

        partitions = []
        for i in range(num_partitions):
            lower = int(boundaries[i])
            upper = int(boundaries[i+1])
            # For all but the last partition, include rows with time < upper; last partition includes time == upper.
            if i < num_partitions - 1:
                part = sorted_data.filter(
                    (pl.col(time_col).cast(pl.Int64) >= lower) &
                    (pl.col(time_col).cast(pl.Int64) < upper)
                )
            else:
                part = sorted_data.filter(
                    (pl.col(time_col).cast(pl.Int64) >= lower) &
                    (pl.col(time_col).cast(pl.Int64) <= upper)
                )
            # Left join with unique panels and fill nulls with 0.
            part_complete = unique_panels.join(part, on=panel_vars, how="left").fill_null(0)
            partitions.append(part_complete)
        return partitions


def stratified_sample(
    data: pl.DataFrame,
    stratify_by: Union[str, List[str]],
    frac: float = 0.1
) -> pl.DataFrame:
    """
    Return a stratified sample of the DataFrame using a window function approach.
    
    The function creates a new column with a shuffled index for each row within its group 
    (as defined by the `stratify_by` column(s)) and a column with the count of rows per group.
    Then, it filters rows where the shuffled index is less than the product of the group count and `frac`.
    
    This ensures that approximately `frac` of the rows are sampled from each group.
    
    Parameters:
      data (pl.DataFrame): The input DataFrame.
      stratify_by (str or List[str]): The column name or list of column names used for stratification.
      frac (float): The fraction of rows to sample from each stratum (must be between 0 and 1).
    
    Returns:
      pl.DataFrame: A new DataFrame containing the stratified sample.
    
    Examples:
      from PolarsFE import datasets
      import datetime

      # Create a fake dataset with a datetime column and a stratification (panel) column.
      dates = [datetime.datetime(2023, 1, 1) + datetime.timedelta(days=i) for i in range(100)]
      df = pl.DataFrame({
          "date": dates,
          "value": np.random.rand(100) * 100,
          "panel": np.random.choice(["A", "B", "C"], size=100)
      })
      
      print("=== Original Dataset (first 10 rows) ===")
      print(df.head(10))
      print(f"Total rows: {df.height}\n")
      
      # Test 1: Stratified sampling on a single column ("panel").
      sample_df = stratified_sample(df, stratify_by="panel", frac=0.2)
      print("=== Stratified Sample (20% from each panel) ===")
      print(sample_df)
      print(f"Sample rows: {sample_df.height}\n")
      
      # Test 2: Stratified sampling on multiple columns.
      # Create a dataset with two stratification variables.
      df2 = pl.DataFrame({
          "id": np.arange(1, 201),
          "group": np.random.choice(["A", "B"], size=200),
          "region": np.random.choice(["North", "South"], size=200),
          "value": np.random.rand(200) * 100
      })
      print("=== Original Dataset with Multiple Stratification Columns ===")
      print(df2.head(10))
      print(f"Total rows: {df2.height}\n")
      
      sample_df2 = stratified_sample(df2, stratify_by=["group", "region"], frac=0.15)
      print("=== Stratified Sample with 'group' and 'region' (15% from each stratum) ===")
      print(sample_df2)
      print(f"Sample rows: {sample_df2.height}")
    """

    if not (0 <= frac <= 1):
        raise ValueError("frac must be between 0 and 1")
    
    # Ensure stratify_by is a list.
    if isinstance(stratify_by, str):
        stratify_by = [stratify_by]
    
    # Create two helper columns:
    # 1. "sample_idx": A shuffled index for each row within each group.
    # 2. "group_count": The number of rows in the group.
    # pl.int_range(0, pl.len()) creates a range of row indices.
    df_with_sample = data.with_columns([
        pl.int_range(0, pl.len()).shuffle().over(stratify_by).alias("sample_idx"),
        pl.len().over(stratify_by).alias("group_count")
    ])

    # For each row, if its sample index is less than group_count * frac, we include it.
    sampled = df_with_sample.filter(pl.col("sample_idx") < (pl.col("group_count") * frac))
    
    # Drop the helper columns.
    sampled = sampled.drop(["sample_idx", "group_count"])
    
    return sampled


def impute_missing(
    data: pl.DataFrame,
    method: str = "constant",
    value: Optional[Any] = None,
    columns: Optional[List[str]] = None,
    group_vars: Optional[List[str]] = None,
) -> pl.DataFrame:
    """
    Impute missing values in a Polars DataFrame using various methods, optionally within groups.

    If `columns` is provided, only those columns are processed; otherwise, all columns are processed.
    If `group_vars` is provided, imputation is done within each group defined by these columns; otherwise, imputation is global.

    Supported methods:
      - "constant": Replace missing values with a provided constant (the `value` parameter must be supplied).
      - "mean": Replace missing values with the mean of the column. If `group_vars` is provided, use the mean within each group.
      - "median": Replace missing values with the median of the column. If `group_vars` is provided, use the median within each group.
      - "mode": Replace missing values with the mode (most common value) of the column. For groups, the mode is computed per group.
      - "ffill" or "forward_fill": Forward-fill missing values (using the last non-null value). If grouping, fill forward within each group.
      - "bfill" or "backward_fill": Backward-fill missing values (using the next non-null value). If grouping, fill backward within each group.

    Parameters:
      data (pl.DataFrame): The input DataFrame.
      method (str): The imputation method to use (case-insensitive).
      value (Optional[Any]): The constant value for "constant" imputation.
      columns (Optional[List[str]]): The list of column names to impute. If None, all columns are processed.
      group_vars (Optional[List[str]]): A list of column names to define groups. If provided, imputation is computed per group.

    Returns:
      pl.DataFrame: A new DataFrame with missing values imputed according to the specified method.

    Examples:
      from PolarsFE import datasets

      df = pl.DataFrame({
          "A": [1, None, 3, None],
          "B": [None, 2, None, 4],
          "C": ["x", None, "y", "z"],
          "group": ["G1", "G1", "G2", "G2"]
      })

      # Constant imputation for columns A and B with 0.
      imputed_const = impute_missing(df, method="constant", value=0, columns=["A", "B"])

      # Global mean imputation for numeric columns A and B.
      imputed_mean = impute_missing(df, method="mean", columns=["A", "B"])

      # Group-based median imputation for columns A and B.
      imputed_median_group = impute_missing(df, method="median", columns=["A", "B"], group_vars=["group"])

      # Forward-fill imputation globally.
      imputed_ffill = impute_missing(df, method="ffill")

      # Group-based median imputation for columns A and B.
      imputed_median_group = impute_missing(df, method="median", columns=["A", "B"], group_vars=["group"])
      print("\n=== Group-Based Median Imputation for A and B ===")
      print(imputed_median_group)

      # Group-based forward fill imputation for all columns.
      imputed_ffill = impute_missing(df, method="ffill", group_vars=["group"])
      print("\n=== Group-Based Forward Fill Imputation ===")
      print(imputed_ffill)
    """

    # If no columns are provided, process all columns.
    if columns is None:
        columns = data.columns

    method = method.lower()

    if method == "constant":
        if value is None:
            raise ValueError("For constant imputation, a value must be provided.")
        # Constant imputation is independent of groups.
        return data.with_columns([pl.col(col).fill_null(value) for col in columns])

    elif method == "mean":
        if group_vars is None:
            return data.with_columns([pl.col(col).fill_null(pl.col(col).mean()) for col in columns])
        else:
            return data.with_columns([pl.col(col).fill_null(pl.col(col).mean().over(group_vars)) for col in columns])

    elif method == "median":
        if group_vars is None:
            return data.with_columns([pl.col(col).fill_null(pl.col(col).median()) for col in columns])
        else:
            return data.with_columns([pl.col(col).fill_null(pl.col(col).median().over(group_vars)) for col in columns])

    elif method == "mode":
        # Define a helper to compute the mode of a list of values.
        def mode_func(lst: List[Any]) -> Any:
            filtered = [x for x in lst if x is not None]
            if not filtered:
                return None
            # Use Counter to get the most common element.
            counter = Counter(filtered)
            return counter.most_common(1)[0][0]

        if group_vars is None:
            # Global mode for each column.
            return data.with_columns([pl.col(col).fill_null(pl.lit(mode_func(data[col].to_list()))) for col in columns])
        else:
            # For each column, compute mode per group using group_by, then join back.
            df_out = data
            for col in columns:
                mode_df = data.group_by(group_vars).agg(
                    pl.col(col).apply(lambda s: mode_func(s) if s is not None else None).alias(f"{col}_mode")
                )
                # Join the mode per group back to the original data.
                df_out = df_out.join(mode_df, on=group_vars, how="left").with_column(
                    pl.col(col).fill_null(pl.col(f"{col}_mode"))
                ).drop(f"{col}_mode")
            return df_out

    elif method in {"ffill", "forward_fill"}:
        if group_vars is None:
            return data.with_columns([pl.col(col).forward_fill() for col in columns])
        else:
            return data.with_columns([pl.col(col).forward_fill().over(group_vars) for col in columns])

    elif method in {"bfill", "backward_fill"}:
        if group_vars is None:
            return data.with_columns([pl.col(col).backward_fill() for col in columns])
        else:
            return data.with_columns([pl.col(col).backward_fill().over(group_vars) for col in columns])

    else:
        raise ValueError(f"Imputation method '{method}' is not recognized")
