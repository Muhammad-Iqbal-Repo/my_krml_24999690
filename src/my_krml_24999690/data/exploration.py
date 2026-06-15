import numpy as np
import pandas as pd


def _safe_nunique(series: pd.Series) -> int:
    """Count unique values, including for columns containing lists or dicts."""
    try:
        return int(series.nunique(dropna=True))
    except TypeError:
        return int(series.dropna().astype(str).nunique())


def _safe_value_counts(series: pd.Series) -> pd.Series:
    """Count values, including for columns containing unhashable objects."""
    non_null = series.dropna()
    try:
        return non_null.value_counts()
    except TypeError:
        return non_null.astype(str).value_counts()


def get_shapes(**datasets):
    """Prints the shape of multiple datasets passed as keyword arguments."""
    """
        Example:
        get_shapes(iris=iris_df, titanic=titanic_df)
        iris and titanic are the names of the datasets
        iris_df and titanic_df are the actual dataframes
        
        It will print:
        The shape of the iris dataset: 150 rows and 5 columns
        The shape of the titanic dataset: 891 rows and 12 columns
    """
    
    for name, df in datasets.items():
        print(f"The shape of the {name} dataset: {df.shape[0]} rows and {df.shape[1]} columns")
        
def check_duplicates(**datasets):
    """Prints and returns the number of duplicated rows for multiple datasets passed as keyword arguments."""
    """
        Example:
        check_duplicates(iris=iris_df, titanic=titanic_df)
        iris and titanic are the names of the datasets
        iris_df and titanic_df are the actual dataframes
        
        It will print:
        Number of duplicated rows in the iris data: 0
        Number of duplicated rows in the titanic data: 0
    """
    results = {}
    for name, df in datasets.items():
        count = int(df.duplicated().sum())
        print(f"Number of duplicated rows in the {name} data: {count}")
        results[name] = count
    return results
        

def check_nulls(**datasets):
    """Prints and returns the number of null values for multiple datasets passed as keyword arguments."""
    """
        Example:
        check_nulls(iris=iris_df, titanic=titanic_df)
        iris and titanic are the names of the datasets
        iris_df and titanic_df are the actual dataframes
        It will print:
        Null values in the iris dataset: 0
        Null values in the titanic dataset: 177
    """
    results = {}
    for name, df in datasets.items():
        count = int(df.isnull().sum().sum())
        print(f"Null values in the {name} dataset: {count}")
        results[name] = count
    return results
        
def check_null_columns(**datasets):
    """Prints and returns columns with null values for multiple datasets passed as keyword arguments."""
    """
        Example:
        check_null_columns(iris=iris_df, titanic=titanic_df)
        iris and titanic are the names of the datasets
        iris_df and titanic_df are the actual dataframes
        
        It will print:
        Number of columns with null values in the iris dataset:
        No columns with null values
        Number of columns with null values in the titanic dataset:
        Age          1
    """
    results = {}
    for name, df in datasets.items():
        null_columns = df.isnull().sum()
        null_columns = null_columns[null_columns > 0]
        print(f"Number of columns with null values in the {name} dataset:")
        if null_columns.empty:
            print("No columns with null values")
        else:
            print(null_columns)
        results[name] = null_columns.to_dict()
    return results
            
def check_duplicates_df(df):
    """Returns the number of duplicated rows in a dataframe."""
    """
        Example:
        num_duplicates = check_duplicates_df(iris_df)
        It will return the number of duplicated rows in the iris_df dataframe
    """
    count = int(df.duplicated().sum())
    print(f"Number of duplicated rows in the dataframe: {count}")
    return count

def comprehensive_report(
    df: pd.DataFrame,
    top_n: int = 10,
    sample_rows: int = 5,
    display: bool = True,
) -> dict:
    """
    Build a practical data-quality and descriptive report for a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to inspect.
    top_n : int, default=10
        Maximum number of rows shown in ranked report sections.
    sample_rows : int, default=5
        Number of initial rows included in the sample.
    display : bool, default=True
        Print a readable report when True.

    Returns
    -------
    dict
        Structured report containing overview, column summary, missing-value
        details, descriptive statistics, sample rows, and quality warnings.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if not isinstance(top_n, int) or top_n < 1:
        raise ValueError("top_n must be a positive integer.")
    if not isinstance(sample_rows, int) or sample_rows < 0:
        raise ValueError("sample_rows must be a non-negative integer.")

    row_count, column_count = df.shape
    total_cells = row_count * column_count
    missing_counts = df.isna().sum()
    missing_percentages = (
        missing_counts.div(row_count).mul(100)
        if row_count
        else pd.Series(0.0, index=df.columns)
    )
    unique_counts = pd.Series(
        {
            column: _safe_nunique(df[column])
            for column in df.columns
        },
        dtype="int64",
    )
    unique_percentages = (
        unique_counts.div(row_count).mul(100)
        if row_count
        else pd.Series(0.0, index=df.columns)
    )
    try:
        duplicate_rows = int(df.duplicated().sum())
    except TypeError:
        duplicate_rows = int(df.astype(str).duplicated().sum())
    memory_bytes = int(df.memory_usage(index=True, deep=True).sum())

    column_summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "non_null": df.notna().sum(),
        "missing": missing_counts,
        "missing_pct": missing_percentages.round(2),
        "unique": unique_counts,
        "unique_pct": unique_percentages.round(2),
    })
    column_summary.index.name = "column"

    missing_summary = (
        column_summary.loc[column_summary["missing"] > 0, ["missing", "missing_pct"]]
        .sort_values(["missing", "missing_pct"], ascending=False)
        .head(top_n)
    )

    numeric_data = df.select_dtypes(include=[np.number])
    numeric_summary = (
        numeric_data.describe().T.round(4)
        if not numeric_data.empty and row_count
        else pd.DataFrame()
    )

    datetime_data = df.select_dtypes(include=["datetime", "datetimetz"])
    if not datetime_data.empty and row_count:
        datetime_summary = pd.DataFrame({
            "non_null": datetime_data.notna().sum(),
            "missing": datetime_data.isna().sum(),
            "min": datetime_data.min(),
            "max": datetime_data.max(),
        })
    else:
        datetime_summary = pd.DataFrame()

    categorical_columns = [
        column
        for column in df.columns
        if column not in numeric_data.columns
        and column not in datetime_data.columns
    ]
    categorical_rows = []
    for column in categorical_columns:
        series = df[column]
        non_null = series.dropna()
        value_counts = _safe_value_counts(series)
        categorical_rows.append({
            "column": column,
            "non_null": int(non_null.size),
            "missing": int(series.isna().sum()),
            "unique": int(unique_counts[column]),
            "top": value_counts.index[0] if not value_counts.empty else None,
            "top_frequency": (
                int(value_counts.iloc[0]) if not value_counts.empty else 0
            ),
        })
    categorical_summary = (
        pd.DataFrame(categorical_rows).set_index("column")
        if categorical_rows
        else pd.DataFrame()
    )

    constant_columns = [
        column for column in df.columns
        if row_count > 0 and unique_counts[column] <= 1
    ]
    all_missing_columns = [
        column for column in df.columns
        if missing_counts[column] == row_count and row_count > 0
    ]
    high_cardinality_columns = [
        column for column in categorical_columns
        if row_count > 0
        and unique_counts[column] > 20
        and unique_percentages[column] >= 50
    ]

    quality_warnings = []
    if duplicate_rows:
        quality_warnings.append(
            f"{duplicate_rows} duplicated row(s) detected."
        )
    if missing_counts.sum():
        quality_warnings.append(
            f"{int(missing_counts.sum())} missing value(s) detected."
        )
    if constant_columns:
        quality_warnings.append(
            f"Constant columns: {', '.join(map(str, constant_columns))}."
        )
    if all_missing_columns:
        quality_warnings.append(
            f"All-missing columns: {', '.join(map(str, all_missing_columns))}."
        )
    if high_cardinality_columns:
        quality_warnings.append(
            "High-cardinality categorical columns: "
            f"{', '.join(map(str, high_cardinality_columns))}."
        )
    if not quality_warnings:
        quality_warnings.append("No obvious data-quality issues detected.")

    overview = {
        "rows": row_count,
        "columns": column_count,
        "total_cells": total_cells,
        "missing_values": int(missing_counts.sum()),
        "missing_pct": (
            round(float(missing_counts.sum() / total_cells * 100), 2)
            if total_cells
            else 0.0
        ),
        "duplicate_rows": duplicate_rows,
        "duplicate_pct": (
            round(duplicate_rows / row_count * 100, 2)
            if row_count
            else 0.0
        ),
        "memory_bytes": memory_bytes,
        "memory_mb": round(memory_bytes / (1024 ** 2), 4),
    }

    report = {
        "overview": overview,
        "column_summary": column_summary,
        "missing_summary": missing_summary,
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
        "datetime_summary": datetime_summary,
        "sample": df.head(sample_rows).copy(),
        "quality_warnings": quality_warnings,
    }

    if display:
        _print_comprehensive_report(report, top_n=top_n)

    return report


def _print_comprehensive_report(report: dict, top_n: int) -> None:
    """Print a concise console representation of a comprehensive report."""
    overview = report["overview"]
    separator = "=" * 72

    print(separator)
    print("DATAFRAME COMPREHENSIVE REPORT")
    print(separator)
    print(
        f"Rows: {overview['rows']:,} | Columns: {overview['columns']:,} | "
        f"Memory: {overview['memory_mb']:.4f} MB"
    )
    print(
        f"Missing: {overview['missing_values']:,} "
        f"({overview['missing_pct']:.2f}%) | "
        f"Duplicates: {overview['duplicate_rows']:,} "
        f"({overview['duplicate_pct']:.2f}%)"
    )

    print("\nCOLUMN SUMMARY")
    print("-" * 72)
    column_summary = report["column_summary"]
    print(
        column_summary.to_string()
        if not column_summary.empty
        else "No columns."
    )

    print(f"\nMISSING VALUES - TOP {top_n}")
    print("-" * 72)
    missing_summary = report["missing_summary"]
    print(
        missing_summary.to_string()
        if not missing_summary.empty
        else "No missing values."
    )

    for heading, key, empty_message in (
        ("NUMERIC SUMMARY", "numeric_summary", "No numeric columns."),
        (
            "CATEGORICAL SUMMARY",
            "categorical_summary",
            "No categorical columns.",
        ),
        ("DATETIME SUMMARY", "datetime_summary", "No datetime columns."),
    ):
        print(f"\n{heading}")
        print("-" * 72)
        section = report[key]
        print(section.to_string() if not section.empty else empty_message)

    print("\nQUALITY CHECKS")
    print("-" * 72)
    for warning in report["quality_warnings"]:
        print(f"- {warning}")

    print("\nSAMPLE")
    print("-" * 72)
    sample = report["sample"]
    print(sample.to_string(index=False) if not sample.empty else "No rows.")
    print(separator)
