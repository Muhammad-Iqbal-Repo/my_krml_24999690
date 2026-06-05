import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def standardize_train(df, columns):
    """Standardizes the specified columns in a training dataframe."""
    """
        Example:
        standardized_train_df = standardize_train(iris_df, ['sepal_length', 'sepal_width'])
        It will return the iris_df dataframe with standardized sepal_length and sepal_width columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # store the columns used for scaling
    cols_used = list(columns)

    # avoiding NaN issues by imputing first
    imputer = SimpleImputer(strategy="mean")
    X_num = imputer.fit_transform(df[cols_used])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num)

    df_out = df.copy()
    df_out[cols_used] = X_scaled

    # return the scaled dataframe and the scaler object
    scaler._imputer = imputer
    return df_out, scaler, cols_used

def standardize_test(df, cols_used, scaler):
    """Standardizes the specified columns in a testing dataframe."""
    """
        Example:
        standardized_test_df = standardize_test(iris_df, ['sepal_length', 'sepal_width'], scaler)
        It will return the iris_df dataframe with standardized sepal_length and sepal_width columns
    """
    if scaler is None:
        raise ValueError("Scaler must be provided for test/val data")
    if any(col not in df.columns for col in cols_used):
        missing = [c for c in cols_used if c not in df.columns]
        raise ValueError(f"Columns missing in DataFrame: {missing}")

    # impute with the training statistics, then scale
    X_num = scaler._imputer.transform(df[cols_used])
    X_scaled = scaler.transform(X_num)

    df_out = df.copy()
    df_out[cols_used] = X_scaled
    return df_out
