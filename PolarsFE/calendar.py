import polars as pl
from typing import List, Dict, Optional
import math


def calendar_features(
    data: pl.DataFrame,
    date_col: str,
    features: List[str] = None
) -> pl.DataFrame:
    """
    Add calendar-based features to a Polars DataFrame based on a datetime column.
    
    The user can supply a list of desired features to create. Supported features include:
      - "millisecond": The millisecond component
      - "second": The second component
      - "minute": The minute component
      - "hour": The hour component
      - "year": The year component.
      - "month": The month component (1-12).
      - "day": The day of the month.
      - "day_of_week": The day of the week (Monday = 0, Sunday = 6).
      - "week": The ISO week number.
      - "quarter": The quarter of the year (1-4).
      - "day_of_year": The ordinal day of the year (1 through 365/366).
      - "week_of_month": The week of the month computed as ((day - 1) // 7) + 1.
      - "is_weekend": A boolean flag indicating if the date falls on a weekend (Saturday or Sunday).
    
    If `features` is not provided, the default set is:
        ["year", "month", "day", "day_of_week", "week", "quarter", "day_of_year", "week_of_month", "is_weekend"]
    
    Each new column is created with the name "{date_col}_{feature}".
    
    Parameters:
      data (pl.DataFrame): The input DataFrame.
      date_col (str): The name of the column containing datetime values.
      features (List[str]): The list of features to create.
      
    Returns:
      pl.DataFrame: The DataFrame with the additional calendar feature columns appended.
    
    Example:
      from PolarsFE import calendar
      import datetime
      
      # Create a fake dataset with a datetime column.
      dates = [datetime.datetime(2023, 1, 1) + datetime.timedelta(days=i) for i in range(10)]
      df = pl.DataFrame({
          "date": dates,
          "value": [i * 10 for i in range(10)]
      })
      
      print("=== Original DataFrame ===")
      print(df)

      # Test 1: Only extract 'year', 'month', and 'day'.
      df_partial = calendar.calendar_features(df, "date", features=["year", "month", "day"])
      print("\n=== DataFrame with 'year', 'month', and 'day' Only ===")
      print(df_partial)
    """

    if features is None:
        raise Error("User needs to supply values for features parameter")

    # Define a mapping from feature name to the corresponding Polars expression.
    feature_map = {
        "millisecond": pl.col(date_col).dt.millisecond(),
        "second": pl.col(date_col).dt.second(),
        "minute": pl.col(date_col).dt.minute(),
        "hour": pl.col(date_col).dt.hour(),
        "day": pl.col(date_col).dt.day(),
        "day_of_week": pl.col(date_col).dt.weekday(),
        "is_weekend": (pl.col(date_col).dt.weekday() >= 5)
        "day_of_year": pl.col(date_col).dt.ordinal_day(),
        "week": pl.col(date_col).dt.week(),
        "week_of_month": ((pl.col(date_col).dt.day() - 1) // 7 + 1).cast(pl.Int64),
        "month": pl.col(date_col).dt.month(),
        "quarter": pl.col(date_col).dt.quarter(),
        "year": pl.col(date_col).dt.year(),
    }
    
    unsupported = [feat for feat in features if feat not in feature_map]
    if unsupported:
        raise ValueError(f"Unsupported features requested: {unsupported}")

    exprs = [feature_map[feat].alias(f"{date_col}_{feat}") for feat in features]
    return data.with_columns(exprs)


def cyclic_features(
    data: pl.DataFrame,
    date_col: str,
    columns: Optional[List[str]] = None,
    drop_original: bool = False,
    prefix: Optional[str] = None
) -> pl.DataFrame:
    """
    Transform cyclic calendar features into sine and cosine components.
    
    The function assumes that calendar feature columns were created using a naming convention
    in which the base date column name is prepended, followed by an underscore, and then the 
    feature name. For example, if the date column is "date", then a calendar feature might be 
    named "date_day_of_week". This function uses that prefix to extract the cyclic feature name 
    (e.g. "day_of_week") and then uses a default mapping of feature names to periods to compute an angle:
    
        angle = 2π * value / period
    
    It then creates two new columns for each input column:
      - "{original}_sin": the sine of the angle.
      - "{original}_cos": the cosine of the angle.
    
    Parameters:
      data (pl.DataFrame): The input DataFrame.
      date_col (str): The base date column name used to create calendar features.
      columns (Optional[List[str]]): List of column names to transform. If None, all columns starting with "{prefix}" are used.
      drop_original (bool): If True, drop the original cyclic feature columns after transformation.
      prefix (Optional[str]): The prefix used in the calendar feature column names. Defaults to "{date_col}_".
    
    Returns:
      pl.DataFrame: The DataFrame with new sine and cosine columns appended.
    
    Example:
      from PolarsFE import calendar

      # Create a sample DataFrame with calendar features.
      df = pl.DataFrame({
          "date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"],
          "date_day_of_week": [6, 0, 1, 2],  # e.g., Saturday=6, Sunday=0, Monday=1, Tuesday=2
          "date_month": [1, 1, 1, 1]
      })
      
      print("=== Original DataFrame ===")
      print(df)
      
      # Transform the cyclic features "date_day_of_week" and "date_month".
      df_transformed = calendar.cyclic_features(
          data=df,
          date_col="date",
          columns=["date_day_of_week", "date_month"],
          drop_original=False
      )
      
      print("\n=== DataFrame with Transformed Cyclic Features ===")
      print(df_transformed)
      
      # Optionally, drop the original columns.
      df_transformed_drop = calendar.cyclic_features(
          data=df,
          date_col="date",
          columns=["date_day_of_week", "date_month"],
          drop_original=True
      )
      print("\n=== DataFrame with Transformed Cyclic Features (Originals Dropped) ===")
      print(df_transformed_drop)
    """

    if prefix is None:
        prefix = f"{date_col}_"
    
    # If no columns are provided, select all columns that start with the prefix.
    if columns is None:
        columns = [col for col in data.columns if col.startswith(prefix)]
    
    default_periods = {
        "millisecond": 1000,
        "second": 60,
        "minute": 60,
        "hour": 24,
        "day_of_week": 7,
        "day_of_year": 365,
        "week_of_month": 5,
        "week_of_year": 52,
        "month": 12,
        "quarter": 4,
    }

    new_exprs = []
    for col in columns:
        if not col.startswith(prefix):
            raise ValueError(f"Column '{col}' does not start with the expected prefix '{prefix}'.")
        # Extract the cyclic feature from the column name by removing the prefix.
        feature = col[len(prefix):]  # e.g. "day_of_week" from "date_day_of_week"
        if feature not in default_periods:
            raise ValueError(f"Feature '{feature}' (from column '{col}') is not supported. Supported features: {list(default_periods.keys())}")
        period = default_periods[feature]
        # Compute the angle: 2π * value / period.
        angle_expr = (2 * math.pi * pl.col(col) / period)
        sin_expr = angle_expr.sin().alias(f"{col}_sin")
        cos_expr = angle_expr.cos().alias(f"{col}_cos")
        new_exprs.extend([sin_expr, cos_expr])
    
    df_out = data.with_columns(new_exprs)
    if drop_original:
        df_out = df_out.drop(columns)
    return df_out




