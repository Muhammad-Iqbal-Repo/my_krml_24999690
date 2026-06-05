import pandas as pd
import numpy as np
from typing import Sequence, Iterable, Optional
from sklearn.model_selection import train_test_split

def add_cyclical_time_features(df, date_col):
    """
    Add cyclical time features (sin/cos) for day-of-year, day-of-week, and hour-of-day.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe with a datetime column.
    date_col : str, default="date"
        Name of the datetime column.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with new cyclical columns.
    """
    
    # make sure the date_col is in datetime format
    df[date_col] = pd.to_datetime(df[date_col])
    
    # day of year (1–365)
    df["doy"] = df[date_col].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365)

    # day of week (0–6, Monday=0)
    df["dow"] = df[date_col].dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)

    # hour of day (0–23)
    df["hour"] = df[date_col].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    
    # month
    df["month"] = df[date_col].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    # drop the intermediate columns
    df = df.drop(columns=["doy", "dow", "hour", "month"])

    return df


def split_time_series(
    df: pd.DataFrame,
    date_col: str = "",
    target_col: str = "",
    drop_cols: Sequence[str] = (),
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
):
    """
    Split a time-ordered dataset into train, validation, and test sets.

    Args:
        df (pd.DataFrame): Full dataset.
        date_col (str): Column name used for sorting by time.
        target_col (str): Name of target column (y).
        drop_cols (Sequence[str]): Columns to drop (e.g., time-like columns).
        train_ratio (float): Fraction of data for training.
        val_ratio (float): Fraction for validation.
        test_ratio (float): Fraction for testing.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    # ensure sorted by time
    df = df.sort_values(date_col).reset_index(drop=True)

    # check ratio sum
    if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    # compute split sizes
    n = len(df)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)

    # slice by index (time-ordered)
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size + val_size]
    test_df = df.iloc[train_size + val_size:]

    # drop columns
    train_df = train_df.drop(columns=drop_cols, errors="ignore")
    val_df = val_df.drop(columns=drop_cols, errors="ignore")
    test_df = test_df.drop(columns=drop_cols, errors="ignore")

    # split X and y
    X_train, y_train = train_df.drop(columns=[target_col]), train_df[target_col]
    X_val, y_val = val_df.drop(columns=[target_col]), val_df[target_col]
    X_test, y_test = test_df.drop(columns=[target_col]), test_df[target_col]

    print(f"Train set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def add_lags_stats_and_marketcap_changes(
    df: pd.DataFrame,
    date_col: str,
    value_cols: Optional[Iterable[str]] = None,
    value_lags: Iterable[int] = (1, 3, 5),
    market_cap_col: str = "marketCap",
    marketcap_lags: Iterable[int] = (1, 7, 30),
) -> pd.DataFrame:
    """
    Add lag features, rolling statistics, and market cap change metrics.

    - Sorts by `date_col` ascending.
    - For each value_col:
        Adds lag, mean, max, min, median, std over each window in value_lags.
    - For market cap:
        Adds difference and percentage change for each n in marketcap_lags.

    Returns:
        pd.DataFrame with new columns added.
    """
    df = df.copy()

    # Ensure proper datetime order
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col], utc=True, errors="coerce")
    df = df.sort_values(date_col).reset_index(drop=True)

    # Default value columns if not given
    if value_cols is None:
        value_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]

    # Add lag and rolling stats for value columns
    for col in value_cols:
        for n in value_lags:
            df[f"{col}_lag{n}"] = df[col].shift(n)
            df[f"{col}_mean{n}"] = df[col].rolling(window=n).mean()
            df[f"{col}_max{n}"] = df[col].rolling(window=n).max()
            df[f"{col}_min{n}"] = df[col].rolling(window=n).min()
            df[f"{col}_median{n}"] = df[col].rolling(window=n).median()
            df[f"{col}_std{n}"] = df[col].rolling(window=n).std()

    # Add market cap diff and pct change for separate lags
    if market_cap_col in df.columns:
        for n in marketcap_lags:
            lagged = df[market_cap_col].shift(n)
            df[f"{market_cap_col}_diff_{n}"] = df[market_cap_col] - lagged
            df[f"{market_cap_col}_pct_{n}"] = df[market_cap_col].div(lagged).sub(1.0)

    return df

def pop_target(df, target):
    """Separates the target column from the features in a dataframe."""
    """
        Example:
        X, y = pop_target(iris_df, 'species')
        It will return the features dataframe X and the target series y
    """
    y = df.pop(target)
    X = df
    return X, y

def split_data(df: pd.DataFrame, target_col: str, test_size: float = 0.2, val_size: float = 0.0, stratify_col: Optional[str] = None, random_state: int = 42):
    """
    Robust function to split data randomly into train, validation, and test sets.
    
    Args:
        df: Input dataframe.
        target_col: The name of the target column.
        test_size: Proportion of the dataset to include in the test split.
        val_size: Proportion of the dataset to include in the validation split (taken from the remaining train set).
        stratify_col: Column name to use for stratified splitting (usually the target column for classification).
        random_state: Seed for reproducibility.
        
    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
               If val_size is 0, X_val and y_val will be None.
    """
    stratify_series = df[stratify_col] if stratify_col else None
    
    # First split into train/val and test
    df_train_val, df_test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=stratify_series
    )
    
    # Second split into train and val if requested
    if val_size > 0:
        # Calculate the adjusted validation proportion from the remaining training data
        adj_val_size = val_size / (1.0 - test_size)
        
        stratify_val_series = df_train_val[stratify_col] if stratify_col else None
        
        df_train, df_val = train_test_split(
            df_train_val, test_size=adj_val_size, random_state=random_state, stratify=stratify_val_series
        )
    else:
        df_train = df_train_val
        df_val = None

    # Separate X and y
    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]
    
    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]
    
    if df_val is not None:
        X_val = df_val.drop(columns=[target_col])
        y_val = df_val[target_col]
        print(f"Splits -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
        return X_train, y_train, X_val, y_val, X_test, y_test
    else:
        print(f"Splits -> Train: {len(X_train)} | Test: {len(X_test)}")
        return X_train, y_train, None, None, X_test, y_test
