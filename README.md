![PyPI](https://img.shields.io/pypi/v/PolarsFE?label=pypi%20package)
![PyPI - Downloads](https://img.shields.io/pypi/dm/PolarsFE)
![Build + Test](https://github.com/AdrianAntico/PolarsFE/actions/workflows/python-package.yml/badge.svg)

# PolarsFE

![](https://github.com/AdrianAntico/PolarsFE/raw/main/PolarsFE/Images/Logo.PNG)

PolarsFE is a Python package that enables one to plot Echarts quickly. It piggybacks off of the pyecharts package that pipes into Apache Echarts. Pyecharts is a great package for fully customizing plots but is quite a challenge to make use of quickly. PolarsFE solves this with a simple API for defining plotting elements and data, along with automatic data wrangling operations, using polars, to correctly structure data fast.

For the Code Examples below, there is a dataset in the PolarsFE/datasets folder named FakeBevData.csv that you can download for replication purposes.

# Installation
```bash
pip install PolarsFE

or 

pip install git+https://github.com/AdrianAntico/PolarsFE.git#egg=PolarsFE
```

<br>


# Code Examples


<br>


## Character-Based Feature Engineering

### Create Dummy Variables

<details><summary>Click for code example</summary>

```python
from PolarsFE import dummy_variables
import polars as pl

df = pl.DataFrame({
    "Category": ["A", "B", "A", "C", "B", "C", "A", "B", "D"],
    "Color": ["Red", "Blue", "Green", "Red", "Green", "Blue", "Red", "Red", "Green"],
    "Value": [10, 20, 30, 40, 50, 60, 70, 80, 90]  # Numeric column (won't be affected)
})
        
# Create dummies for 'Category' and 'Color' and keep the original columns
df_dummies, levels_used = dummy_variables(
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
from PolarsFE import categorical_encoding

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
result_target = categorical_encoding(
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
result_js = categorical_encoding(
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
result_target_reg = categorical_encoding(
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
result_js_reg = categorical_encoding(
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
result_target_multi = categorical_encoding(
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
result_js_multi = categorical_encoding(
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


## Numeric-Based Feature Engineering

### Standardization

<details><summary>Click for code example</summary>

```python
import numpy as np
import polars as pl
from PolarsFE import standardize

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
```

</details>


### Percent Rank

<details><summary>Click for code example</summary>

```python
import numpy as np
import polars as pl
from PolarsFE import percent_rank

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

```

</details>


### Numeric Transformations

<details><summary>Click for code example</summary>

```python
import numpy as np
import polars as pl
from PolarsFE import numeric_transform

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
```

</details>
