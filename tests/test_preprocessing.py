import pandas as pd
import numpy as np
from numpy.testing import assert_almost_equal
from my_krml_24999690.features.preprocessing import (
    standardize_train, standardize_test, DFStandardScaler, DFDummyEncoder
)
from sklearn.pipeline import Pipeline

def test_standardize_train():
    # Create a DataFrame with known values
    df = pd.DataFrame({
        "A": [10, 20, 30, 40, 50],
        "B": [1, 2, 3, 4, 5],
        "C": ["cat", "dog", "cat", "dog", "cat"] # Should be ignored if not specified
    })
    
    df_out, scaler, cols_used = standardize_train(df.copy(), columns=["A", "B"])
    
    assert cols_used == ["A", "B"]
    
    # After scaling, mean should be ~0 and std should be ~1 (with population std logic)
    assert_almost_equal(df_out["A"].mean(), 0.0, decimal=5)
    assert_almost_equal(df_out["B"].mean(), 0.0, decimal=5)
    assert_almost_equal(df_out["A"].std(ddof=0), 1.0, decimal=5)
    
    # String column should remain unchanged
    assert list(df_out["C"]) == list(df["C"])

def test_standardize_test():
    # Create a training set to fit the scaler
    df_train = pd.DataFrame({"A": [10, 20, 30, 40, 50]})
    _, scaler, cols_used = standardize_train(df_train, columns=["A"])
    
    # Create a test set
    df_test = pd.DataFrame({"A": [30, 60], "B": [1, 2]})
    
    df_test_out = standardize_test(df_test.copy(), cols_used, scaler)
    
    # The mean of train data was 30, std was ~14.14
    # For value 30, it should be 0.0
    assert_almost_equal(df_test_out["A"].iloc[0], 0.0, decimal=5)
    
    # Column B should remain unchanged
    assert list(df_test_out["B"]) == [1, 2]

def test_df_standard_scaler():
    df_train = pd.DataFrame({"A": [10, 20, 30, 40, 50], "B": ["cat", "dog", "cat", "dog", "cat"]})
    df_test = pd.DataFrame({"A": [30, 60], "B": ["dog", "cat"]})
    
    scaler = DFStandardScaler(columns=["A"])
    scaler.fit(df_train)
    
    df_train_scaled = scaler.transform(df_train)
    df_test_scaled = scaler.transform(df_test)
    
    assert_almost_equal(df_train_scaled["A"].mean(), 0.0, decimal=5)
    assert_almost_equal(df_test_scaled["A"].iloc[0], 0.0, decimal=5)
    assert list(df_test_scaled["B"]) == ["dog", "cat"]

def test_df_dummy_encoder():
    df_train = pd.DataFrame({"cat_col": ["A", "B", "A"]})
    df_test = pd.DataFrame({"cat_col": ["B", "C"]}) # "C" is new, "A" is missing
    
    encoder = DFDummyEncoder(columns=["cat_col"], drop_first=False)
    encoder.fit(df_train)
    
    df_train_enc = encoder.transform(df_train)
    df_test_enc = encoder.transform(df_test)
    
    # Train dummy columns should be 'cat_col_A' and 'cat_col_B'
    assert list(df_train_enc.columns) == ["cat_col_A", "cat_col_B"]
    assert list(df_test_enc.columns) == ["cat_col_A", "cat_col_B"]
    
    # For test data:
    # Row 0 ('B'): cat_col_A=0, cat_col_B=1
    # Row 1 ('C'): cat_col_A=0, cat_col_B=0 (unseen class ignored/encoded as 0)
    assert df_test_enc.loc[0, "cat_col_B"] == 1
    assert df_test_enc.loc[0, "cat_col_A"] == 0
    assert df_test_enc.loc[1, "cat_col_A"] == 0
    assert df_test_enc.loc[1, "cat_col_B"] == 0

def test_pipeline_integration():
    df_train = pd.DataFrame({
        "num": [10, 20, 30, 40, 50],
        "cat": ["A", "B", "A", "B", "A"]
    })
    
    pipe = Pipeline([
        ("dummy", DFDummyEncoder(columns=["cat"], drop_first=False)),
        ("scale", DFStandardScaler(columns=["num"]))
    ])
    
    # Fit and transform
    df_transformed = pipe.fit_transform(df_train)
    
    # Columns should be num, cat_A, cat_B
    assert set(df_transformed.columns) == {"num", "cat_A", "cat_B"}
    assert_almost_equal(df_transformed["num"].mean(), 0.0, decimal=5)
