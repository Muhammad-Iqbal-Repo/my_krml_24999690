import pandas as pd
import numpy as np
from numpy.testing import assert_almost_equal
from my_krml_24999690.features.preprocessing import standardize_train, standardize_test

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
