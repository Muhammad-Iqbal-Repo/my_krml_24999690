# my_krml_24999690

Package for Assignment 1

## Installation

```bash
pip install my_krml_24999690
```

Install optional features only when needed:

```bash
pip install "my_krml_24999690[canvas]"
pip install "my_krml_24999690[balance]"
```

## Usage

`my_krml_24999690` provides focused helpers for data loading, exploration,
feature engineering, preprocessing, model training, and evaluation.

See the [Package Structure and API Reference](PACKAGE_REFERENCE.md) for the
complete module structure, function and class reference, and usage examples.

### Loading and Exploring Data
```python
from my_krml_24999690 import load_data_at2, check_nulls, get_shapes

# Load dataset
df = load_data_at2("path/to/your/data.csv", skiprows=0, sep=",")

# Quick diagnostics
get_shapes(my_data=df)
check_nulls(my_data=df)
```

Create a reusable data-quality report:

```python
from my_krml_24999690 import comprehensive_report

report = comprehensive_report(df)

# Every section is also available programmatically.
print(report["overview"])
print(report["missing_summary"])
print(report["quality_warnings"])
```

### Feature Engineering & Preprocessing
```python
from my_krml_24999690 import add_cyclical_time_features, standardize_train

# Automatically add sin/cos temporal features
df = add_cyclical_time_features(df, date_col="time")

# Standardize numerical columns
df_scaled, scaler, cols = standardize_train(df, columns=["feature1", "feature2"])
```

### Modeling
```python
from my_krml_24999690 import split_time_series, train_regressor
from sklearn.ensemble import RandomForestRegressor

# Split data chronologically
X_train, y_train, X_val, y_val, X_test, y_test = split_time_series(
    df_scaled, date_col="time", target_col="target"
)

model = RandomForestRegressor()
results = train_regressor(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
)

print(results["val_metrics"])
```

### Classification Metrics
```python
from my_krml_24999690 import evaluate_classification

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

metrics = evaluate_classification(
    y_test,
    y_pred,
    y_pred_proba,
    average="macro",
    labels=model.classes_,
)

print(metrics["accuracy"])
print(metrics["precision"])
print(metrics["recall"])
print(metrics["f1_score"])
print(metrics["roc_auc"])
print(metrics["confusion_matrix"])
```

Pass `labels=model.classes_` whenever probabilities are two-dimensional so
their column order is explicit. ROC AUC is `None` when it is not requested or
cannot be defined because the evaluated data contains only one class.

### Safe Preprocessing Order

Split data before fitting preprocessing:

```python
from my_krml_24999690 import DFDummyEncoder

encoder = DFDummyEncoder(columns=["city"], drop_first=False)
X_train_encoded = encoder.fit_transform(X_train)
X_val_encoded = encoder.transform(X_val)
X_test_encoded = encoder.transform(X_test)
```

This keeps validation and test data from influencing training feature
construction. Missing files, invalid split settings, and unavailable optional
features raise clear exceptions instead of silently returning fallback data.

### Additional Helpers

The package root also exports:

- `tune_hyperparameters`
- `balance_classes`
- `plot_feature_importance`
- `plot_confusion_matrix`
- `plot_roc_curve`
- `summarize_classification_result`
- `download_canvas_courses`

Most reusable helpers are quiet by default. Use `verbose=True` on functions
that support progress output.

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`my_krml_24999690` was created by Muhammad Iqbal. Muhammad Iqbal retains all rights to the source and it may not be reproduced, distributed, or used to create derivative works.

## Credits

`my_krml_24999690` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
