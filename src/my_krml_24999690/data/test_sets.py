
import re
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

from sets import *

# fixtures
@pytest.fixture
def df_basic():
    # create dataframe
    return pd.DataFrame({
        "A": [1, 2, 3, 3],
        "B": [1.0, np.nan, 3.0, 3.0],
        "C": ["x", "y", "z", "z"],
    })

@pytest.fixture
def df_numeric():
    return pd.DataFrame({
        "x1": [10.0, 12.0, 14.0, 16.0],
        "x2": [1.0, 2.0, 3.0, 4.0],
        "y":  [0, 1, 0, 1],
    })

# test
# the print function will not be tested

# this will test the drop_columns function
def test_drop_columns_removes_columns(df_basic):
    out = drop_columns(df_basic.copy(), ["B"])
    assert "B" not in out.columns
    assert list(out.columns) == ["A", "C"]

# this will test the standardize_train function
def test_standardize_train_returns_scaled_df_and_scaler(df_numeric):
    cols = ["x1", "x2"]
    df_scaled, scaler = standardize_train(df_numeric.copy(), cols)
    for c in cols:
        mu = float(np.mean(df_scaled[c]))
        sigma = float(np.std(df_scaled[c]))
        assert abs(mu) < 1e-7
        assert abs(sigma - 1.0) < 1e-6
    assert hasattr(scaler, "mean_") and hasattr(scaler, "scale_")

# this will test the standardize_test function
def test_standardize_test_uses_provided_scaler(df_numeric):
    train_cols = ["x1", "x2"]
    train_df = df_numeric[train_cols].iloc[:3].copy()
    test_df = df_numeric[train_cols].iloc[3:].copy()

    df_train_scaled, scaler = standardize_train(train_df.copy(), train_cols)
    df_test_scaled = standardize_test(test_df.copy(), train_cols, scaler)

    expected = (test_df.values - scaler.mean_) / scaler.scale_
    np.testing.assert_allclose(df_test_scaled.values, expected, atol=1e-8)

# this will test the pop_target function
def test_pop_target_removes_and_returns(df_numeric):
    df = df_numeric.copy()
    X, y = pop_target(df, "y")
    assert "y" not in X.columns
    assert y.name == "y"
    assert "y" not in df.columns

# this will test the evaluate_model function
def test_evaluate_model_trains_and_returns_probas(capsys):
    X, y = make_classification(n_samples=120, n_features=5, n_informative=3, random_state=0)
    X_train, X_val = X[:80], X[80:]
    y_train, y_val = y[:80], y[80:]
    model = LogisticRegression()

    y_train_p, y_val_p = evaluate_model(model, X_train, y_train, X_val, y_val)
    check = capsys.readouterr().out

    assert y_train_p.shape[0] == len(X_train)
    assert y_val_p.shape[0] == len(X_val)