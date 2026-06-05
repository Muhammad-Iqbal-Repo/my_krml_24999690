# my_krml_24999690

Package for Assignment 1

## Installation

```bash
$ pip install my_krml_24999690
```

## Usage

`my_krml_24999690` provides a comprehensive suite of tools for data loading, exploration, feature engineering, and modeling. Here is a quick example of how to use it:

### Loading and Exploring Data
```python
from my_krml_24999690 import load_data_at2, check_nulls, get_shapes

# Load dataset
df = load_data_at2("path/to/your/data.csv", skiprows=0, sep=",")

# Quick diagnostics
get_shapes(my_data=df)
check_nulls(my_data=df)
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
from my_krml_24999690 import split_time_series, train_evaluate_regression_model
from sklearn.ensemble import RandomForestRegressor

# Split data chronologically
X_train, y_train, X_val, y_val, X_test, y_test = split_time_series(
    df_scaled, date_col="time", target_col="target"
)

# Train and evaluate model
results = {}
model = RandomForestRegressor()
train_evaluate_regression_model(
    model, X_train, y_train, X_val, y_val, X_test, y_test, 
    experiment_name="baseline_rf", dict_results=results
)
```
## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`my_krml_24999690` was created by Muhammad Iqbal. Muhammad Iqbal retains all rights to the source and it may not be reproduced, distributed, or used to create derivative works.

## Credits

`my_krml_24999690` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
