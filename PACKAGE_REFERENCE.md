# Package Structure and API Reference

This document describes the current structure and callable API of
`my_krml_24999690`. Most supported functions and classes can be imported
directly from the package root:

```python
from my_krml_24999690 import comprehensive_report, split_data
```

## Package Structure

```text
src/my_krml_24999690/
|-- __init__.py
|-- data/
|   |-- loaders.py
|   |-- exploration.py
|   `-- canvas_downloader.py
|-- features/
|   |-- engineering.py
|   `-- preprocessing.py
|-- models/
|   |-- training.py
|   |-- performance.py
|   `-- tracking.py
`-- visualization/
    `-- eda.py
```

| Module | Responsibility |
|---|---|
| `data.loaders` | Load assignment datasets and summarize dataframes. |
| `data.exploration` | Inspect shapes, duplicates, missing values, and data quality. |
| `data.canvas_downloader` | Download Canvas course files and page attachments. |
| `features.engineering` | Create time features, lag features, and dataset splits. |
| `features.preprocessing` | Encode, scale, clean, and balance tabular data. |
| `models.training` | Train, cross-validate, and tune estimators. |
| `models.performance` | Evaluate models and visualize model performance. |
| `models.tracking` | Store and compare experiment results. |
| `visualization.eda` | Create common exploratory data analysis plots. |

## Installation

Install the core package:

```bash
pip install my_krml_24999690
```

Canvas downloading and class balancing use optional dependencies:

```bash
pip install "my_krml_24999690[canvas]"
pip install "my_krml_24999690[balance]"
```

## Recommended Workflow

Split data before fitting an encoder, scaler, sampler, or model. This prevents
validation and test data from influencing training-time preprocessing.

```python
from my_krml_24999690 import (
    DFDummyEncoder,
    DFStandardScaler,
    evaluate_classification,
    split_data,
    train_classifier,
)
from sklearn.linear_model import LogisticRegression

X_train, y_train, X_val, y_val, X_test, y_test = split_data(
    df,
    target_col="target",
    test_size=0.2,
    val_size=0.1,
    stratify_col="target",
)

encoder = DFDummyEncoder(columns=["city"], drop_first=False)
X_train = encoder.fit_transform(X_train)
X_val = encoder.transform(X_val)
X_test = encoder.transform(X_test)

scaler = DFStandardScaler(columns=["age", "income"])
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

result = train_classifier(
    LogisticRegression(max_iter=1000),
    X_train,
    y_train,
    X_val,
    y_val,
)

model = result["model"]
metrics = evaluate_classification(
    y_test,
    model.predict(X_test),
    model.predict_proba(X_test),
    labels=model.classes_,
)
```

## Data Loading

### `load_data_at2`

```python
load_data_at2(file_path, skiprows, sep, verbose=False)
```

Loads a CSV file, requires a `time` column, and parses that column as datetime.
Returns a pandas `DataFrame`.

```python
from my_krml_24999690 import load_data_at2

df = load_data_at2("data/measurements.csv", skiprows=0, sep=",")
```

### `get_experiment_files`

```python
get_experiment_files(experiment_number, base_path, verbose=False)
```

Loads the 12 expected classification and regression train, validation, and
test CSV files from `base_path/experiment_number`.

The return order is:

```text
X_class_train, y_class_train, X_class_val, y_class_val,
X_class_test, y_class_test, X_reg_train, y_reg_train,
X_reg_val, y_reg_val, X_reg_test, y_reg_test
```

```python
from my_krml_24999690 import get_experiment_files

datasets = get_experiment_files("experiment_01", "data/experiments")
X_class_train, y_class_train = datasets[0], datasets[1]
```

### `load_data_at3`

```python
load_data_at3(dataset_dir, pattern="*.csv", sep=";", verbose=False)
```

Loads matching CSV files, requires `timeOpen` in every file, sorts each file by
that column, and concatenates the results. Returns `(combined_dataframe,
file_names)`.

```python
from my_krml_24999690 import load_data_at3

df, source_files = load_data_at3("data/crypto", pattern="*.csv")
```

### `summarize_dataframe`

```python
summarize_dataframe(df, include_all=False)
```

Returns a dictionary containing `head`, `tail`, `describe`, and `column_info`.
Set `include_all=True` to include non-numeric columns in `describe`.

```python
from my_krml_24999690 import summarize_dataframe

summary = summarize_dataframe(df, include_all=True)
print(summary["column_info"])
```

## Data Exploration

### `get_shapes`

```python
get_shapes(**datasets)
```

Prints the shape of each named dataset. It does not return a value.

```python
from my_krml_24999690 import get_shapes

get_shapes(train=X_train, validation=X_val, test=X_test)
```

### `check_duplicates`

```python
check_duplicates(**datasets)
```

Prints duplicate counts and returns `{dataset_name: duplicate_count}`.

### `check_nulls`

```python
check_nulls(**datasets)
```

Prints total missing-value counts and returns
`{dataset_name: total_null_count}`.

### `check_null_columns`

```python
check_null_columns(**datasets)
```

Prints and returns missing-value counts by column for each named dataset.

```python
from my_krml_24999690 import check_duplicates, check_null_columns, check_nulls

duplicate_counts = check_duplicates(train=X_train, test=X_test)
null_totals = check_nulls(train=X_train, test=X_test)
nulls_by_column = check_null_columns(train=X_train, test=X_test)
```

### `check_duplicates_df`

```python
check_duplicates_df(df)
```

Prints and returns the duplicate-row count for one dataframe.

### `comprehensive_report`

```python
comprehensive_report(df, top_n=10, sample_rows=5, display=True)
```

Builds a reusable data-quality report. It returns:

| Key | Content |
|---|---|
| `overview` | Dataset dimensions, missingness, duplicate count, and memory use. |
| `column_summary` | Per-column type, missingness, and uniqueness. |
| `missing_summary` | Missing counts and percentages for affected columns. |
| `numeric_summary` | Numeric descriptive statistics. |
| `categorical_summary` | Frequent values for categorical columns. |
| `datetime_summary` | Datetime ranges and missing-value information. |
| `sample` | A sample of the input rows. |
| `quality_warnings` | Detected quality concerns requiring review. |

Use `display=False` in scripts and tests when only the returned data is needed.

```python
from my_krml_24999690 import comprehensive_report

report = comprehensive_report(df, top_n=5, sample_rows=3, display=False)
print(report["missing_summary"])
print(report["quality_warnings"])
```

## Canvas Downloading

### `download_canvas_courses`

```python
download_canvas_courses(
    api_url,
    api_key,
    course_ids=None,
    output_dir="CanvasDownloads",
    logger=None,
    progress_cb=None,
    allowed_exts=None,
    verbose=False,
)
```

Downloads files, pages, and linked documents from accessible Canvas courses.
Install the `canvas` extra before using it. Returns a list of per-course result
dictionaries.

- `course_ids=None` processes all accessible courses.
- `allowed_exts` limits downloaded file extensions.
- `logger(message)` receives log messages when provided.
- `progress_cb(done, total, message)` receives progress updates.

```python
from my_krml_24999690 import download_canvas_courses

results = download_canvas_courses(
    api_url="https://canvas.example.edu",
    api_key="your-api-token",
    course_ids=[12345],
    output_dir="downloads",
    allowed_exts={".pdf", ".csv"},
)
```

Do not commit Canvas API tokens to source control.

## Feature Engineering

### `add_cyclical_time_features`

```python
add_cyclical_time_features(df, date_col)
```

Returns a copy with sine and cosine features for day of year, day of week,
hour, and month. The source date column must be parseable as datetime.

### `add_lags_stats_and_marketcap_changes`

```python
add_lags_stats_and_marketcap_changes(
    df,
    date_col,
    value_cols=None,
    value_lags=(1, 3, 5),
    market_cap_col="marketCap",
    marketcap_lags=(1, 7, 30),
)
```

Returns a date-sorted copy containing lag and rolling-statistic features for
the selected numeric columns, plus percentage-change features for market
capitalization when that column is available.

```python
from my_krml_24999690 import (
    add_cyclical_time_features,
    add_lags_stats_and_marketcap_changes,
)

featured = add_cyclical_time_features(df, "time")
featured = add_lags_stats_and_marketcap_changes(
    featured,
    date_col="time",
    value_cols=["price", "volume"],
)
```

Lag and rolling features create missing values near the beginning of the
series. Handle those rows before model training.

### `split_time_series`

```python
split_time_series(
    df,
    date_col="",
    target_col="",
    drop_cols=(),
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    verbose=False,
)
```

Sorts chronologically and returns:

```text
X_train, y_train, X_val, y_val, X_test, y_test
```

The three ratios must be positive and sum to `1.0`. Use this function for
ordered time-series data where random shuffling would leak future information.

### `pop_target`

```python
pop_target(df, target)
```

Returns `(X, y)` without modifying the input dataframe.

### `split_data`

```python
split_data(
    df,
    target_col,
    test_size=0.2,
    val_size=0,
    stratify_col=None,
    random_state=42,
    verbose=False,
)
```

Creates random train, optional validation, and test splits. Returns:

```text
X_train, y_train, X_val, y_val, X_test, y_test
```

`X_val` and `y_val` are `None` when `val_size=0`. Set `stratify_col` to a
categorical target or grouping column when class proportions should be
preserved. Continuous numeric columns with many unique values are not
stratified.

```python
from my_krml_24999690 import pop_target, split_data, split_time_series

X, y = pop_target(df, "target")

random_splits = split_data(
    df,
    target_col="target",
    test_size=0.2,
    val_size=0.1,
    stratify_col="target",
)

time_splits = split_time_series(
    df,
    date_col="time",
    target_col="target",
)
```

## Preprocessing

### `DFStandardScaler`

```python
DFStandardScaler(columns=None)
```

A scikit-learn-compatible transformer that mean-imputes and standardizes
selected dataframe columns. When `columns=None`, it uses numeric columns.
Fitted state is available in `scaler_` and `cols_used_`.

```python
scaler.fit(X, y=None)
scaler.transform(X)
```

```python
from my_krml_24999690 import DFStandardScaler

scaler = DFStandardScaler(columns=["age", "income"])
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Fit only on training data. `transform` preserves the dataframe structure and
uses the imputation and scaling values learned during `fit`.

### `DFDummyEncoder`

```python
DFDummyEncoder(columns=None, drop_first=True)
```

A scikit-learn-compatible one-hot encoder for pandas dataframes. It stores the
training output columns in `dummy_columns_` and aligns later data to that
schema.

```python
encoder.fit(X, y=None)
encoder.transform(X)
```

```python
from my_krml_24999690 import DFDummyEncoder

encoder = DFDummyEncoder(columns=["city"], drop_first=False)
X_train_encoded = encoder.fit_transform(X_train)
X_test_encoded = encoder.transform(X_test)
```

Unseen test categories do not create new columns, and categories seen only
outside training cannot influence the training feature schema.

### `standardize_train` and `standardize_test`

```python
standardize_train(df, columns)
standardize_test(df, cols_used, scaler)
```

`standardize_train` fits mean imputation and standardization, then returns
`(transformed_dataframe, scaler, columns_used)`. `standardize_test` reuses that
training state.

```python
from my_krml_24999690 import standardize_test, standardize_train

X_train, scaler, scaled_columns = standardize_train(
    X_train,
    columns=["age", "income"],
)
X_test = standardize_test(X_test, scaled_columns, scaler)
```

### `drop_columns`

```python
drop_columns(df, columns, verbose=False)
```

Returns a copy without the requested columns. Missing column names are ignored.

### `remove_duplicates`

```python
remove_duplicates(df, subset=None, keep="first", verbose=False)
```

Returns a deduplicated copy. Its arguments follow
`pandas.DataFrame.drop_duplicates`.

### `create_dummies_train` and `create_dummies_test`

```python
create_dummies_train(df, columns=None, drop_first=True)
create_dummies_test(df, dummy_columns, columns=None, drop_first=True)
```

The training function returns `(encoded_dataframe, dummy_columns)`. Pass those
columns to the test function to produce the same feature schema.

```python
from my_krml_24999690 import create_dummies_test, create_dummies_train

X_train, dummy_columns = create_dummies_train(
    X_train,
    columns=["city"],
    drop_first=False,
)
X_test = create_dummies_test(
    X_test,
    dummy_columns,
    columns=["city"],
    drop_first=False,
)
```

### `create_dummies`

```python
create_dummies(df, columns=None, drop_first=True)
```

One-hot encodes one dataframe. Use it for exploration or standalone data only.
For model evaluation, prefer `DFDummyEncoder` or the train/test function pair
so the feature schema is learned from training data.

### `balance_classes`

```python
balance_classes(
    X_train,
    y_train,
    method="smote",
    random_state=42,
    verbose=False,
)
```

Balances classification data using `"smote"` or `"undersample"` and returns
`(X_resampled, y_resampled)`. Install the `balance` extra first.

```python
from my_krml_24999690 import balance_classes

X_balanced, y_balanced = balance_classes(
    X_train,
    y_train,
    method="smote",
    random_state=42,
)
```

Balance only the training set. Resampling validation or test data makes model
evaluation unrepresentative.

## Model Training

### `train_classifier`

```python
train_classifier(
    model,
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    average="macro",
)
```

Fits a classifier and returns a dictionary containing the fitted `model`,
`train_metrics`, and optional `val_metrics`.

### `train_regressor`

```python
train_regressor(model, X_train, y_train, X_val=None, y_val=None)
```

Fits a regressor and returns a dictionary containing the fitted `model`,
`train_metrics`, and optional `val_metrics`.

```python
from my_krml_24999690 import train_classifier, train_regressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

classification_result = train_classifier(
    RandomForestClassifier(random_state=42),
    X_train,
    y_train,
    X_val,
    y_val,
)

regression_result = train_regressor(
    RandomForestRegressor(random_state=42),
    X_train,
    y_train,
    X_val,
    y_val,
)
```

### `cross_validate_model`

```python
cross_validate_model(
    model,
    X,
    y,
    task_type="classification",
    cv_folds=5,
    is_time_series=False,
    **kwargs,
)
```

Runs cross-validation and returns mean and standard deviation metrics. Set
`task_type` to `"classification"` or `"regression"`. Set
`is_time_series=True` to use ordered time-series folds.

```python
from my_krml_24999690 import cross_validate_model

scores = cross_validate_model(
    model,
    X_train,
    y_train,
    task_type="classification",
    cv_folds=5,
)
```

### `tune_hyperparameters`

```python
tune_hyperparameters(
    model,
    param_grid,
    X_train,
    y_train,
    search_type="random",
    cv=5,
    n_iter=10,
    scoring="accuracy",
    random_state=42,
    verbose=False,
)
```

Runs randomized or grid hyperparameter search and returns the best fitted
estimator. Use `search_type="random"` or `search_type="grid"`.

```python
from my_krml_24999690 import tune_hyperparameters
from sklearn.ensemble import RandomForestClassifier

best_model = tune_hyperparameters(
    RandomForestClassifier(random_state=42),
    {"n_estimators": [100, 200], "max_depth": [None, 5, 10]},
    X_train,
    y_train,
    search_type="grid",
    scoring="f1_macro",
)
```

## Model Evaluation

### `evaluate_classification`

```python
evaluate_classification(
    y_true,
    y_pred,
    y_pred_proba=None,
    average="macro",
    labels=None,
    pos_label=None,
    multi_class="ovr",
)
```

Returns classification metrics in one dictionary:

- `accuracy`
- `precision`
- `recall`
- `f1_score`
- `confusion_matrix`
- `labels`
- `roc_auc`
- `brier_score`

Pass probabilities to calculate AUC. For binary classification, probabilities
may be the positive-class vector or a two-column matrix. For multiclass
classification, pass one probability column per class. When using a
two-dimensional probability matrix, pass `labels=model.classes_` so the
probability-column order is explicit. For binary metrics, `pos_label` defaults
to the last resolved class label. `multi_class` accepts `"ovr"` or `"ovo"`.

```python
from my_krml_24999690 import evaluate_classification

metrics = evaluate_classification(
    y_test,
    model.predict(X_test),
    model.predict_proba(X_test),
    average="macro",
    labels=model.classes_,
)
print(metrics["confusion_matrix"])
print(metrics["roc_auc"])
```

`roc_auc` is `None` when probabilities are omitted or AUC is undefined, such
as when `y_true` contains only one class.

### `evaluate_regression`

```python
evaluate_regression(y_true, y_pred)
```

Returns `mae`, `mse`, `rmse`, `r2_score`, and `mape`.

### `print_metrics`

```python
print_metrics(metrics_dict, title="Metrics Summary")
```

Prints a readable representation of a metrics dictionary.

### `compare_metrics`

```python
compare_metrics(**metrics_dicts)
```

Returns a dataframe comparing scalar metrics from multiple named result
dictionaries.

```python
from my_krml_24999690 import compare_metrics, print_metrics

print_metrics(metrics, title="Test metrics")
comparison = compare_metrics(baseline=baseline_metrics, tuned=metrics)
```

### `kaggle_submission`

```python
kaggle_submission(
    model,
    X_test,
    sample_path,
    output_path,
    target_col="target",
    predict_proba=True,
    verbose=False,
)
```

Loads a sample submission CSV, replaces `target_col` with predictions, saves
the result to `output_path`, and returns the submission dataframe. With
`predict_proba=True` and a compatible classifier, it uses the second
probability column; otherwise it calls `predict`. The sample submission and
`X_test` must have the same number of rows.

```python
from my_krml_24999690 import kaggle_submission

submission = kaggle_submission(
    model,
    X_test,
    sample_path="data/sample_submission.csv",
    output_path="outputs/submission.csv",
    target_col="target",
)
```

### `plot_feature_importance`

```python
plot_feature_importance(
    model,
    feature_names,
    top_n=20,
    title="Feature Importances",
)
```

Plots the most important tree-based feature importances or absolute linear
model coefficients and returns the Matplotlib axes. Pipelines are supported
when the final step exposes `feature_importances_` or `coef_`.

### `plot_confusion_matrix`

```python
plot_confusion_matrix(
    y_true,
    y_pred,
    class_names=None,
    title="Confusion Matrix",
)
```

Plots a count-based confusion matrix and returns the Matplotlib axes. Optional
`class_names` must match the matrix dimensions.

### `plot_roc_curve`

```python
plot_roc_curve(y_true, y_pred_proba, title="ROC Curve")
```

Plots a binary ROC curve and returns the Matplotlib axes.

```python
from my_krml_24999690 import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curve,
)

plot_confusion_matrix(y_test, y_pred)
plot_roc_curve(y_test, y_pred_proba[:, 1])
plot_feature_importance(model, X_test.columns)
```

## Experiment Tracking

### `ExperimentTracker`

```python
ExperimentTracker(filepath="experiments_log.csv")
```

A lightweight CSV-backed experiment log. Existing records are loaded at
initialization, and updated records are written back to the configured file.

#### `log_experiment`

```python
tracker.log_experiment(
    experiment_name,
    model_name,
    parameters,
    metrics,
    cv_score=None,
    notes="",
)
```

Adds one experiment and returns the updated log dataframe. `model_name` may be
a string or estimator. `parameters` may be a dictionary or estimator; for an
estimator, non-default constructor parameters are extracted automatically.
Nested parameter, metric, and CV dictionaries are flattened into CSV-friendly
columns.

#### `get_log`

```python
tracker.get_log(sort_by=None, ascending=False)
```

Returns a copy of the experiment log dataframe, optionally sorted by one
column.

#### `plot_comparison`

```python
tracker.plot_comparison(metric, title=None)
```

Displays a metric comparison across logged experiments. The method accepts
either the full stored name, such as `metric_accuracy`, or the short name
`accuracy`.

#### `clear_log`

```python
tracker.clear_log(confirm=False)
```

With `confirm=True`, clears the in-memory records and deletes the CSV file.
Without confirmation it makes no changes.

```python
from my_krml_24999690 import ExperimentTracker

tracker = ExperimentTracker("outputs/experiments.csv")
tracker.log_experiment(
    experiment_name="random_forest_v1",
    model_name=model,
    parameters=model,
    metrics={"accuracy": 0.91, "f1_score": 0.89},
    notes="Baseline with encoded city feature",
)

log = tracker.get_log(sort_by="metric_accuracy")
tracker.plot_comparison("accuracy")
```

## EDA Visualization

All EDA plotting helpers display the plot. Most return Matplotlib axes;
`plot_pairplot` returns a seaborn `PairGrid`.

### `plot_distribution`

```python
plot_distribution(df, col, hue=None, title=None, figsize=(8, 5))
```

Plots a numeric histogram with a kernel density estimate and optional color
grouping.

### `plot_barchart`

```python
plot_barchart(df, col, hue=None, title=None, figsize=(8, 5))
```

Plots categorical value counts and labels each bar.

### `plot_categorical`

`plot_categorical` is a public alias of `plot_barchart` and accepts the same
arguments.

### `plot_boxplot`

```python
plot_boxplot(
    df,
    cat_col,
    num_col,
    hue=None,
    title=None,
    figsize=(8, 5),
)
```

Plots a numeric variable grouped by a categorical variable.

### `plot_scatter`

```python
plot_scatter(
    df,
    x_col,
    y_col,
    hue=None,
    trendline=False,
    title=None,
    figsize=(8, 5),
)
```

Plots the relationship between two variables with optional color grouping.
Set `trendline=True` to draw a linear regression line when `hue` is not used.

### `plot_correlation_heatmap`

```python
plot_correlation_heatmap(
    df,
    columns=None,
    title=None,
    figsize=(10, 8),
)
```

Plots correlations for selected or automatically detected numeric columns.

### `plot_pairplot`

```python
plot_pairplot(df, columns=None, hue=None, title=None)
```

Creates pairwise plots for selected columns.

```python
from my_krml_24999690 import (
    plot_barchart,
    plot_boxplot,
    plot_correlation_heatmap,
    plot_distribution,
    plot_pairplot,
    plot_scatter,
)

plot_distribution(df, "income", hue="target")
plot_barchart(df, "city", hue="target")
plot_boxplot(df, cat_col="city", num_col="income")
plot_scatter(df, x_col="age", y_col="income", trendline=True)
plot_correlation_heatmap(df)
plot_pairplot(df, columns=["age", "income"], hue="target")
```

## Deprecated Functions

The following functions remain in `models.training` for compatibility but are
deprecated and are not exported from the package root:

### `at2_create_time_series_splits`

```python
from my_krml_24999690.models.training import at2_create_time_series_splits
```

Use `split_time_series` for new code.

### `at2_prepare_training`

```python
from my_krml_24999690.models.training import at2_prepare_training
```

Use the current feature engineering, preprocessing, and target-splitting
helpers for new code.

## Internal Helpers

Functions beginning with `_` are implementation details, may change without
notice, and should not be imported by package users:

| Function | Module | Role |
|---|---|---|
| `_safe_nunique` | `data.exploration` | Calculates uniqueness without failing on unusual values. |
| `_safe_value_counts` | `data.exploration` | Produces defensive value counts for reports. |
| `_print_comprehensive_report` | `data.exploration` | Renders the structured comprehensive report. |
| `_resolve_class_labels` | `models.performance` | Validates and resolves classification label order. |
| `_flatten_dict` | `models.tracking` | Flattens nested experiment data for CSV storage. |

Prefer the documented public functions and classes instead of these helpers.
