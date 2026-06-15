import pandas as pd
from pathlib import Path


def load_data_at2(
    file_path: str | Path,
    skiprows: int,
    sep: str,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Load the dataset from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The loaded dataset.
    """
    data = pd.read_csv(file_path, skiprows=skiprows, sep=sep)
    if "time" not in data.columns:
        raise ValueError("Loaded data must contain a 'time' column.")
    data["time"] = pd.to_datetime(data["time"])

    if verbose:
        print(f"Data loaded successfully from {file_path}")
        print(f"Data shape: {data.shape}")
        print(f"Data columns: {data.columns.tolist()}")

    return data


def get_experiment_files(
    experiment_number: str,
    base_path: str | Path,
    verbose: bool = False,
):
    """
    Get the file path for the specified experiment number, data type, and dataset type.

    Parameters:
    - experiment_number (int): The experiment number (e.g., exp1)
    - base_path (Path): The base path where the experiment folders are located.
    """
    
    csv_path = Path(base_path) / experiment_number
    if not csv_path.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {csv_path}")

    csv_files = list(csv_path.glob("*.csv"))
    data_dict = {
        csv_file.stem.replace(f"{experiment_number}_", ""): pd.read_csv(csv_file)
        for csv_file in csv_files
    }

    required_names = (
        "X_class_train",
        "y_class_train",
        "X_class_val",
        "y_class_val",
        "X_class_test",
        "y_class_test",
        "X_reg_train",
        "y_reg_train",
        "X_reg_val",
        "y_reg_val",
        "X_reg_test",
        "y_reg_test",
    )
    missing = [name for name in required_names if name not in data_dict]
    if missing:
        raise FileNotFoundError(
            f"Missing experiment CSV files in {csv_path}: {', '.join(missing)}"
        )

    if verbose:
        for name in required_names:
            print(f"{name}: {data_dict[name].shape}")

    return tuple(data_dict[name] for name in required_names)


def load_data_at3(
    dataset_dir: str | Path,
    pattern: str = "*.csv",
    sep: str = ";",
    verbose: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Load all CSVs in `dataset_dir` (semicolon-separated), sort each by 'timeOpen',
    and concatenate into a single DataFrame.

    Returns:
        df_all (pd.DataFrame): concatenated DataFrame
        file_names (list[str]): list of CSV file names found
    """
    dataset_path = Path(dataset_dir)

    if not dataset_path.exists() or not dataset_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {dataset_path}")

    files = sorted(dataset_path.glob(pattern))
    file_names = [f.name for f in files]

    if not files:
        raise FileNotFoundError(
            f"No files matching {pattern!r} found in {dataset_path}."
        )

    dataframes = []
    for f in files:
        df = pd.read_csv(f, sep=sep)
        if "timeOpen" not in df.columns:
            raise ValueError(f"{f} must contain a 'timeOpen' column.")
        dataframes.append(df.sort_values("timeOpen").reset_index(drop=True))
        if verbose:
            print(f"Loaded {f.name} with shape {df.shape}")

    return pd.concat(dataframes, ignore_index=True), file_names

def summarize_dataframe(df: pd.DataFrame, include_all: bool = False) -> dict:
    """
    Return key summaries for a given DataFrame:
    - head (first 5 rows)
    - tail (last 5 rows)
    - descriptive statistics
    - column info (name and dtype)
    
    Args:
        df (pd.DataFrame): the dataframe to summarize
    
    Returns:
        dict: containing head, tail, describe, and column_info
    """
    describe_df = df.describe(include='all') if include_all else df.describe()

    summary = {
        "head": df.head(),
        "tail": df.tail(),
        "describe": describe_df,
        "column_info": pd.DataFrame({
            "column": df.columns,
            "dtype": df.dtypes.astype(str)
        }).reset_index(drop=True)
    }
    return summary
