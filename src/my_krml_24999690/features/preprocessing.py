import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

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

def drop_columns(df, columns, verbose=False):
    """
    Safely drops specified columns from a dataframe.
    Ignores columns that do not exist instead of throwing an error.
    """
    if isinstance(columns, str):
        columns = [columns]
    
    # Only drop columns that actually exist in the dataframe
    cols_to_drop = [c for c in columns if c in df.columns]
    
    if cols_to_drop:
        if verbose:
            print(f"Dropped {len(cols_to_drop)} columns.")
        return df.drop(columns=cols_to_drop)
    
    return df.copy()

def remove_duplicates(df, subset=None, keep='first', verbose=False):
    """
    Removes duplicated rows from a dataframe and resets the index.
    """
    initial_len = len(df)
    df_clean = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
    dropped = initial_len - len(df_clean)
    
    if verbose and dropped > 0:
        print(f"Removed {dropped} duplicate rows.")
    
    return df_clean

def create_dummies_train(df, columns=None, drop_first=True):
    """
    Creates dummy variables for training data and records the expected columns.
    
    Returns:
        tuple: (df_dummies, dummy_columns_list)
    """
    df_dummies = pd.get_dummies(df, columns=columns, drop_first=drop_first, dtype=int)
    dummy_columns = df_dummies.columns.tolist()
    return df_dummies, dummy_columns

def create_dummies_test(df, dummy_columns, columns=None, drop_first=True):
    """
    Creates dummy variables for test data, perfectly aligning columns with the training set.
    """
    df_dummies = pd.get_dummies(df, columns=columns, drop_first=drop_first, dtype=int)
    
    # 1. Add missing columns (categories that were in train but not in test)
    missing_cols = set(dummy_columns) - set(df_dummies.columns)
    for c in missing_cols:
        df_dummies[c] = 0
        
    # 2. Reorder columns and safely ignore unexpected columns (categories in test but not train)
    # By strictly selecting `dummy_columns`, we guarantee alignment with the model's expectations
    df_dummies = df_dummies[dummy_columns]
    
    return df_dummies

def create_dummies(df, columns=None, drop_first=True):
    """
    Creates dummy variables for a dataframe in one simple step.

    Prefer fitting DFDummyEncoder on training data and transforming validation
    and test data with the fitted encoder.
    """
    return pd.get_dummies(df, columns=columns, drop_first=drop_first, dtype=int)

class DFStandardScaler(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer that standardizes numeric columns while
    maintaining the pandas DataFrame structure (instead of converting to a numpy array).
    """
    def __init__(self, columns=None):
        self.columns = columns
        self.scaler_ = None
        self.cols_used_ = None

    def fit(self, X, y=None):
        _, self.scaler_, self.cols_used_ = standardize_train(X, columns=self.columns)
        return self

    def transform(self, X):
        if self.scaler_ is None:
            raise ValueError("Transformer has not been fitted yet.")
        return standardize_test(X, self.cols_used_, self.scaler_)

class DFDummyEncoder(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer that applies dummy/one-hot encoding to pandas DataFrames.
    It guarantees that the test set columns perfectly align with the training set columns.
    """
    def __init__(self, columns=None, drop_first=True):
        self.columns = columns
        self.drop_first = drop_first
        self.dummy_columns_ = None

    def fit(self, X, y=None):
        _, self.dummy_columns_ = create_dummies_train(X, columns=self.columns, drop_first=self.drop_first)
        return self

    def transform(self, X):
        if self.dummy_columns_ is None:
            raise ValueError("Transformer has not been fitted yet.")
        return create_dummies_test(X, self.dummy_columns_, columns=self.columns, drop_first=self.drop_first)

def balance_classes(
    X_train,
    y_train,
    method="smote",
    random_state=42,
    verbose=False,
):
    """
    Balances the classes in the training data using SMOTE or Random Undersampling.
    Requires the 'imbalanced-learn' package (pip install imbalanced-learn).

    Args:
        X_train: Training features.
        y_train: Training target.
        method: 'smote' (oversampling) or 'undersample' (undersampling).
        random_state: Seed for reproducibility.

    Returns:
        X_train_resampled, y_train_resampled
    """
    try:
        if method.lower() == "smote":
            from imblearn.over_sampling import SMOTE
            sampler = SMOTE(random_state=random_state)
        elif method.lower() == "undersample":
            from imblearn.under_sampling import RandomUnderSampler
            sampler = RandomUnderSampler(random_state=random_state)
        else:
            raise ValueError("method must be 'smote' or 'undersample'")

        if verbose:
            print(
                f"Original dataset shape: "
                f"{pd.Series(y_train).value_counts().to_dict()}"
            )
        X_res, y_res = sampler.fit_resample(X_train, y_train)
        if verbose:
            print(
                f"Resampled dataset shape: "
                f"{pd.Series(y_res).value_counts().to_dict()}"
            )

        return X_res, y_res
    except ImportError as exc:
        raise ImportError(
            "balance_classes requires the optional 'balance' dependencies. "
            "Install them with: pip install my_krml_24999690[balance]"
        ) from exc
