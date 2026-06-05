import pandas as pd
import numpy as np
import pytest
from my_krml_24999690.features.engineering import add_cyclical_time_features, pop_target, split_time_series, add_lags_stats_and_marketcap_changes

def test_add_cyclical_time_features():
    # Create dummy data
    dates = pd.date_range(start="2023-01-01", periods=3, freq="D")
    df = pd.DataFrame({"date": dates, "value": [10, 20, 30]})
    
    # Apply function
    df_out = add_cyclical_time_features(df.copy(), "date")
    
    # Check if new columns are created
    expected_cols = [
        "doy_sin", "doy_cos", 
        "dow_sin", "dow_cos", 
        "hour_sin", "hour_cos", 
        "month_sin", "month_cos"
    ]
    for col in expected_cols:
        assert col in df_out.columns
        
    # Values should be between -1 and 1
    for col in expected_cols:
        assert df_out[col].min() >= -1.0
        assert df_out[col].max() <= 1.0

def test_pop_target():
    df = pd.DataFrame({
        "feat1": [1, 2, 3],
        "feat2": [4, 5, 6],
        "target": [0, 1, 0]
    })
    
    X, y = pop_target(df.copy(), "target")
    
    assert "target" not in X.columns
    assert list(X.columns) == ["feat1", "feat2"]
    assert list(y) == [0, 1, 0]

def test_split_time_series():
    # Create 100 rows
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    df = pd.DataFrame({"date": dates, "feat": np.random.rand(100), "target": np.random.randint(0, 2, 100)})
    
    X_train, y_train, X_val, y_val, X_test, y_test = split_time_series(
        df=df.copy(), 
        date_col="date", 
        target_col="target", 
        train_ratio=0.7, 
        val_ratio=0.15, 
        test_ratio=0.15
    )
    
    assert len(X_train) == 70
    assert len(X_val) == 15
    assert len(X_test) == 15
    
    assert len(y_train) == 70
    
def test_split_time_series_invalid_ratios():
    df = pd.DataFrame({"date": ["2023-01-01"], "target": [1]})
    with pytest.raises(ValueError):
        split_time_series(df, date_col="date", target_col="target", train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)
