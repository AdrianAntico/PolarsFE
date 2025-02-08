import polars as pl
from typing import List, Dict, Optional, Any
import math
import datetime
import holidays


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
        raise ValueError("User needs to supply values for features parameter")

    # Define a mapping from feature name to the corresponding Polars expression.
    feature_map = {
        "millisecond": pl.col(date_col).dt.millisecond(),
        "second": pl.col(date_col).dt.second(),
        "minute": pl.col(date_col).dt.minute(),
        "hour": pl.col(date_col).dt.hour(),
        "day": pl.col(date_col).dt.day(),
        "day_of_week": pl.col(date_col).dt.weekday(),
        "is_weekend": (pl.col(date_col).dt.weekday() >= 5),
        "day_of_year": pl.col(date_col).dt.ordinal_day(),
        "week": pl.col(date_col).dt.week(),
        "week_of_month": ((pl.col(date_col).dt.day() - 1) // 7 + 1).cast(pl.Int64),
        "month": pl.col(date_col).dt.month(),
        "quarter": pl.col(date_col).dt.quarter(),
        "year": pl.col(date_col).dt.year(),
    }

    # Compute the features on a DataFrame of unique dates.
    unique_dates = data.select(date_col).unique()
    exprs = [feature_map[feat].alias(f"{date_col}_{feat}") for feat in features]
    unique_dates = unique_dates.with_columns(exprs)
    
    # Join the computed features back to the original DataFrame.
    data_out = data.join(unique_dates, on=date_col, how="left")
    return data_out


def cyclic_features(
    data: pl.DataFrame,
    date_col: str,
    columns: Optional[List[str]] = None,
    drop_original: bool = False,
    prefix: Optional[str] = None
) -> pl.DataFrame:
    """
    Transform cyclic calendar feature columns into sine and cosine components by computing the 
    transformation on the unique values and then joining back to the original DataFrame.

    The function assumes that each cyclic feature column follows a naming convention where the 
    column name is prefixed with a base name (e.g., "date_") followed by the feature name (e.g., 
    "day_of_week", "month", etc.). For each column, the function:
      1. Extracts its unique values.
      2. Infers the cyclic feature type by removing the prefix (if provided).
      3. Looks up the default period for that feature.
      4. Computes the angle as: angle = 2π * value / period.
      5. Computes sine and cosine components from the angle.
      6. Joins the computed sine and cosine columns back to the original DataFrame on the cyclic column.

    If `drop_original` is True, the original cyclic feature column is removed.

    Parameters:
      data (pl.DataFrame): The input DataFrame.
      columns (List[str]): A list of column names containing cyclic features (e.g., ["date_day_of_week", "date_month"]).
      prefix (Optional[str]): The prefix to remove from the column name to infer the cyclic feature type.
                              For example, if prefix is "date_", then "date_day_of_week" becomes "day_of_week".
                              If None, the entire column name is used as the feature key.
      drop_original (bool): Whether to drop the original cyclic feature columns after transformation.
    
    Returns:
      pl.DataFrame: The DataFrame with new sine and cosine columns appended for each cyclic feature.
    
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

    df_out = data
    for col in columns:
        # Extract unique values for the column.
        unique_vals_df = data.select(col).unique()
        # Infer the cyclic feature by removing the prefix if provided.
        if prefix:
            if not col.startswith(prefix):
                raise ValueError(f"Column '{col}' does not start with the expected prefix '{prefix}'.")
            feature_type = col[len(prefix):]
        else:
            feature_type = col
        if feature_type not in default_periods:
            raise ValueError(f"Feature type '{feature_type}' (from column '{col}') is not supported. "
                             f"Supported features: {list(default_periods.keys())}")
        period = default_periods[feature_type]
        # Compute the angle and then its sine and cosine.
        # angle = (2 * π * value) / period
        unique_vals_df = unique_vals_df.with_columns([
            ((2 * math.pi * pl.col(col)) / period).sin().alias(f"{col}_sin"),
            ((2 * math.pi * pl.col(col)) / period).cos().alias(f"{col}_cos")
        ])
        # Join the computed columns back to the original DataFrame on the given column.
        df_out = df_out.join(unique_vals_df, on=col, how="left")
        if drop_original:
            df_out = df_out.drop(col)

    return df_out


def holiday_features(
    data: pl.DataFrame,
    date_col: str,
    country: str = "US",
    years: Optional[List[int]] = None,
    pre_window: int = 0,
    post_window: int = 0,
    add_holiday_name: bool = False
) -> pl.DataFrame:
    """
    Add holiday-based features to a Polars DataFrame using the holidays library.
    
    For each row in the DataFrame (based on the date in `date_col`), this function computes:
      - is_holiday: 1 if the date exactly matches a holiday, else 0.
      - pre_holiday: 1 if the date is within `pre_window` days before a holiday, else 0.
      - post_holiday: 1 if the date is within `post_window` days after a holiday, else 0.
      - holiday_effect: 1 if any of the above flags is 1, else 0.
    
    Additionally, if `add_holiday_name` is True, the function adds a column `holiday_name` that
    contains the holiday name for dates that are holidays, and None otherwise.
    
    The function uses the holidays library to generate holiday dates dynamically for the specified
    country and for the years covering the date range in the data (if `years` is not supplied, the function
    determines the years from the date column). Since many holidays (e.g. Martin Luther King Jr. Day,
    Thanksgiving) do not fall on the same date each year, using the holidays library ensures that the holiday
    dates are correct for the specified period.
    
    Parameters:
      data (pl.DataFrame): The input DataFrame.
      date_col (str): The name of the column containing date or datetime values.
      country (str): The country for which to generate holidays (default "US").
      years (Optional[List[int]]): A list of years for which to generate holidays. If None, the function
                                   uses the year range from the data.
      pre_window (int): Number of days before a holiday to flag as pre_holiday.
      post_window (int): Number of days after a holiday to flag as post_holiday.
      add_holiday_name (bool): If True, add a column "holiday_name" with the holiday name for dates that are holidays.
    
    Returns:
      pl.DataFrame: The original DataFrame with additional columns:
          - is_holiday
          - pre_holiday
          - post_holiday
          - holiday_effect
          - (optionally) holiday_name
    
    Examples:
      from PolarsFE import calendar
      import datetime
      
      # Option 1: Use dynamic holiday generation for the US.
      df = pl.DataFrame({
          "date": [datetime.date(2023, 1, 1) + datetime.timedelta(days=i) for i in range(-3, 5)]
      })
      df_holidays = calendar.holiday_features(df, date_col="date", pre_window=2, post_window=2, add_holiday_name=True)
      print("=== DataFrame with Holiday Features (Dynamic Holidays) ===")
      print(df_holidays)
      
      # Option 2: Supply specific years.
      df2 = pl.DataFrame({
          "date": [datetime.date(2022, 12, 30) + datetime.timedelta(days=i) for i in range(10)]
      })
      df2_holidays = calendar.holiday_features(df2, date_col="date", country="US", years=[2022, 2023], pre_window=1, post_window=1)
      print("\n=== DataFrame with Holiday Features (Supplied Years) ===")
      print(df2_holidays)
    """

    # Determine the years if not supplied.
    # Extract the date column as a list of date objects.
    date_series = data.select(pl.col(date_col)).to_series().to_list()
    # Ensure they are datetime.date objects.
    def to_date(x: Any) -> datetime.date:
        if isinstance(x, datetime.datetime):
            return x.date()
        return x
    date_list = [to_date(x) for x in date_series]
    
    if years is None:
        min_year = min(d.year for d in date_list)
        max_year = max(d.year for d in date_list)
        years = list(range(min_year, max_year + 1))
    
    # Generate holidays using the holidays library.
    holiday_obj = holidays.CountryHoliday(country, years=years)
    # Build a dictionary mapping holiday dates to holiday names.
    # The holidays library returns a dictionary-like object.
    holiday_dict: Dict[datetime.date, str] = {date: name for date, name in sorted(holiday_obj.items())}
    holiday_set = set(holiday_dict.keys())
    
    def holiday_flags(date_val: datetime.date) -> Dict[str, Any]:
        # Compute differences in days between each holiday in the holiday_set and the date.
        # (We iterate over holiday_set, which contains all holiday dates for the specified years.)
        differences = [(h - date_val).days for h in holiday_set]
        is_holiday = 1 if 0 in differences else 0
        pre = 1 if any(1 <= d <= pre_window for d in differences) else 0
        post = 1 if any(-post_window <= d <= -1 for d in differences) else 0
        effect = max(is_holiday, pre, post)
        # Optionally include holiday_name if this date is a holiday.
        holiday_name = holiday_dict.get(date_val, None)
        return {
            "is_holiday": is_holiday,
            "pre_holiday": pre,
            "post_holiday": post,
            "holiday_effect": effect,
            "holiday_name": holiday_name
        }
    
    # Create a DataFrame with unique dates.
    unique_dates = data.select(date_col).unique()
    unique_date_list = [to_date(x) for x in unique_dates[date_col].to_list()]
    # Compute holiday flags for each unique date.
    flags_list = [holiday_flags(d) for d in unique_date_list]
    flags_df = pl.DataFrame(flags_list)
    # Combine the unique dates and flags.
    unique_dates = unique_dates.hstack(flags_df)
    
    # Join the holiday features back to the original DataFrame.
    data_out = data.join(unique_dates, on=date_col, how="left")
    return data_out
