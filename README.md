![PyPI](https://img.shields.io/pypi/v/PolarsFE?label=pypi%20package)
![PyPI - Downloads](https://img.shields.io/pypi/dm/PolarsFE)
![Build + Test](https://github.com/AdrianAntico/PolarsFE/actions/workflows/python-package.yml/badge.svg)

![](https://github.com/AdrianAntico/PolarsFE/raw/main/PolarsFE/Images/Logo.PNG)

PolarsFE is a Python package 

# Installation
```bash
pip install PolarsFE

or 

pip install git+https://github.com/AdrianAntico/PolarsFE.git#egg=PolarsFE
```

<br>


# Feature Engineering Code Examples


<br>


## Categorical

### Create Dummy Variables

<details><summary>Click for code example</summary>

```python
from PolarsFE import character
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
```

</details>


### Categorical Encoding

<details><summary>Click for code example</summary>

```python
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
```

</details>


<br>


## Numeric

### Standardization

<details><summary>Click for code example</summary>

```python
import numpy as np
import polars as pl
from PolarsFE import numeric

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
transformed_train, score_tbl = numeric.standardize(
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

transformed_apply = numeric.standardize(
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
backtransformed = numeric.standardize(
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
```

</details>


### Percent Rank

<details><summary>Click for code example</summary>

```python
import numpy as np
import polars as pl
from PolarsFE import numeric

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
transformed_train, score_tbl = numeric.percent_rank(
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

transformed_new = numeric.percent_rank(
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
backtransformed = numeric.percent_rank(
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
```

</details>


### Numeric Transformations

<details><summary>Click for code example</summary>

```python
import numpy as np
import polars as pl
from PolarsFE import numeric

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
df_log = numeric.numeric_transform(
    df,
    col_names=["Positive"],
    transformation="Log",
    mode="apply", debug=True
)
print("\n=== Log Applied ===")
print(df_log.select(["Positive", "Positive_log"]))

df_log_back = numeric.numeric_transform(
    df_log,
    col_names=["Positive_log"],
    transformation="Log",
    mode="backtransform", debug=True
)
print("\n=== Log Backtransformed ===")
print(df_log_back.select(["Positive_log", "Positive_log_back"]))

# --------------------------
# LogPlusA Transformation
# --------------------------
df_logplusa = numeric.numeric_transform(
    df,
    col_names=["Positive"],
    transformation="LogPlusA",
    mode="apply",
    A=None,
    debug=True
)
print("\n=== LogPlusA Applied ===")
print(df_logplusa.select(["Positive", "Positive_logplusa"]))

# For backtransformation, you must supply the same A. Compute it from the original column.
min_val = df.select(pl.col("Positive")).min().item()
A_val = max(1, 1 - min_val)
df_logplusa_back = numeric.numeric_transform(
    df_logplusa,
    col_names=["Positive_logplusa"],
    transformation="LogPlusA",
    mode="backtransform",
    A=A_val,
    debug=True
)
print("\n=== LogPlusA Backtransformed ===")
print(df_logplusa_back.select(["Positive_logplusa", "Positive_logplusa_back"]))

# --------------------------
# Sqrt Transformation
# --------------------------
df_sqrt = numeric.numeric_transform(
    df,
    col_names=["Positive"],
    transformation="Sqrt",
    mode="apply",
    debug=True
)
print("\n=== Sqrt Applied ===")
print(df_sqrt.select(["Positive", "Positive_sqrt"]))

df_sqrt_back = numeric.numeric_transform(
    df_sqrt,
    col_names=["Positive_sqrt"],
    transformation="Sqrt",
    mode="backtransform",
    debug=True
)
print("\n=== Sqrt Backtransformed ===")
print(df_sqrt_back.select(["Positive_sqrt", "Positive_sqrt_back"]))

# --------------------------
# Asin Transformation
# --------------------------
df_asin = numeric.numeric_transform(
    df,
    col_names=["Angle"],
    transformation="Asin",
    mode="apply",
    debug=True
)
print("\n=== Asin Applied ===")
print(df_asin.select(["Angle", "Angle_asin"]))

df_asin_back = numeric.numeric_transform(
    df_asin,
    col_names=["Angle"],
    transformation="Asin",
    mode="backtransform",
    debug=True
)
print("\n=== Asin Backtransformed ===")
print(df_asin_back.select(["Angle_asin", "Angle_asin_back"]))

# --------------------------
# Logit Transformation
# --------------------------
df_logit = numeric.numeric_transform(
    df,
    col_names=["Probability"],
    transformation="Logit",
    mode="apply",
    debug=True
)
print("\n=== Logit Applied ===")
print(df_logit.select(["Probability", "Probability_logit"]))

df_logit_back = numeric.numeric_transform(
    df_logit,
    col_names=["Probability"],
    transformation="Logit",
    mode="backtransform",
    debug=True
)
print("\n=== Logit Backtransformed ===")
print(df_logit_back.select(["Probability_logit", "Probability_logit_back"]))
```

</details>


<br>


## Datasets

### Partition Random

<details><summary>Click for code example</summary>

```python
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
parts_equal = datasets.partition_random(
    data=df,
    num_partitions=3,
    seed=42
)
for idx, part in enumerate(parts_equal, start=1):
    print(f"--- Equal Partition {idx} (rows: {part.height}) ---")
    print(part)
    print()

# Partition into 3 parts using percentages (30%, 30%, 40%).
parts_pct = datasets.partition_random(
    data=df,
    num_partitions=3,
    seed=42,
    percentages=[0.3, 0.3, 0.4]
)
for idx, part in enumerate(parts_pct, start=1):
    print(f"--- Percentage Partition {idx} (rows: {part.height}) ---")
    print(part)
print()
```

</details>


### Partition Time

<details><summary>Click for code example</summary>

```python
from PolarsFE import datasets
import datetime

# Create a DataFrame with dates spanning 100 days.
dates = [datetime.datetime(2023, 1, 1) + datetime.timedelta(days=i) for i in range(100)]
df = pl.DataFrame({
    "date": dates,
    "value": np.random.rand(100)
})

# Partition into 4 equal time intervals.
parts_equal = datasets.partition_time(
    df,
    time_col="date",
    num_partitions=4
)
for idx, part in enumerate(parts_equal, start=1):
    print(f"--- Equal Partition {idx} (rows: {part.height}) ---")
    print(part)

# Partition into 4 parts using percentages: 10%, 20%, 30%, 40%.
parts_pct = datasets.partition_time(
    df,
    time_col="date",
    num_partitions=4,
    percentages=[0.1, 0.2, 0.3, 0.4]
)
for idx, part in enumerate(parts_pct, start=1):
    print(f"--- Percentage Partition {idx} (rows: {part.height}) ---")
    print(part)
```

</details>


### Partition Timeseries

<details><summary>Click for code example</summary>

```python
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
parts_equal = datasets.partition_timeseries(
    df,
    time_col="date",
    panel_vars=["panel"],
    num_partitions=4
)
for idx, part in enumerate(parts_equal, start=1):
    print(f"--- Equal Partition {idx} (rows: {part.height}) ---")
    print(part)
    print()

# --- Test 2: Percentage-Based Partitions ---
# For example, partition into 4 parts using percentages [0.1, 0.2, 0.3, 0.4].
print("=== Percentage-Based Partitions ===")
parts_pct = datasets.partition_timeseries(
    df,
    time_col="date",
    panel_vars=["panel"],
    num_partitions=4,
    percentages=[0.1, 0.2, 0.3, 0.4]
)
for idx, part in enumerate(parts_pct, start=1):
    print(f"--- Percentage Partition {idx} (rows: {part.height}) ---")
    print(part)
    print()
```

</details>


### Stratified Sample

<details><summary>Click for code example</summary>

```python
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
sample_df = datasets.stratified_sample(df, stratify_by="panel", frac=0.2)
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

sample_df2 = datasets.stratified_sample(df2, stratify_by=["group", "region"], frac=0.15)
print("=== Stratified Sample with 'group' and 'region' (15% from each stratum) ===")
print(sample_df2)
print(f"Sample rows: {sample_df2.height}")
```

</details>


### Impute Missing Values

<details><summary>Click for code example</summary>

```python
from PolarsFE import datasets

df = pl.DataFrame({
    "A": [1, None, 3, None],
    "B": [None, 2, None, 4],
    "C": ["x", None, "y", "z"],
    "group": ["G1", "G1", "G2", "G2"]
})

# Constant imputation for columns A and B with 0.
imputed_const = datasets.impute_missing(
    df,
    method="constant",
    value=0,
    columns=["A", "B"]
)

# Global mean imputation for numeric columns A and B.
imputed_mean = datasets.impute_missing(
    df,
    method="mean",
    columns=["A", "B"]
)

# Group-based median imputation for columns A and B.
imputed_median_group = datasets.impute_missing(
    df,
    method="median",
    columns=["A", "B"],
    group_vars=["group"]
)

# Forward-fill imputation globally.
imputed_ffill = datasets.impute_missing(
    df,
    method="ffill"
)

# Group-based median imputation for columns A and B.
imputed_median_group = datasets.impute_missing(
    df,
    method="median",
    columns=["A", "B"],
    group_vars=["group"]
)

# Group-based forward fill imputation for all columns.
imputed_ffill = datasets.impute_missing(
    df,
    method="ffill",
    group_vars=["group"]
)
```

</details>


<br>


## Calendar

### Calendar Features

<details><summary>Click for code example</summary>

```python
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
```

</details>


### Cyclic Features

<details><summary>Click for code example</summary>

```python
from PolarsFE import calendar  # Assumes both functions are in the "calendar" module
import datetime

# Create a sample DataFrame with a date column.
df = pl.DataFrame({
    "date": [
        datetime.date(2023, 1, 1),
        datetime.date(2023, 1, 2),
        datetime.date(2023, 1, 3),
        datetime.date(2023, 1, 4)
    ],
    "value": [10, 20, 30, 40]
})

print("=== Original DataFrame ===")
print(df)

# Step 1: Compute calendar features.
# For example, here we extract "day_of_week" and "month". The resulting columns will be named
# "date_day_of_week" and "date_month".
df_cal = calendar.calendar_features(data=df, date_col="date", features=["day_of_week", "month"])
print("\n=== DataFrame with Calendar Features ===")
print(df_cal)

# Step 2: Transform the cyclic features.
# Now, use the cyclic_features function to transform "date_day_of_week" and "date_month" into sine and cosine components.
df_cyclic = calendar.cyclic_features(
    data=df_cal,
    date_col="date",
    columns=["date_day_of_week", "date_month"],
    drop_original=False
)
print("\n=== DataFrame with Transformed Cyclic Features ===")
print(df_cyclic)

# Optionally, if you wish to drop the original cyclic feature columns after transformation:
df_cyclic_drop = calendar.cyclic_features(
    data=df_cal,
    date_col="date",
    columns=["date_day_of_week", "date_month"],
    drop_original=True
)
print("\n=== DataFrame with Transformed Cyclic Features (Originals Dropped) ===")
print(df_cyclic_drop)
```

</details>


### Holiday Features

<details><summary>Click for code example</summary>

```python
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
```

</details>


<br>


## Window

### Lags

<details><summary>Click for code example</summary>

```python
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
```

</details>
