import pandas as pd
from pathlib import Path

def load_data_at2(file_path: str, skiprows: int, sep: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The loaded dataset.
    """
    try:
        data = pd.read_csv(file_path, skiprows=skiprows, sep=sep)  # Skip the specified number of rows
        data['time'] = pd.to_datetime(data['time'])  # Ensure 'time' column is in datetime format
        print(f"Data loaded successfully from {file_path}")
        print(f"Data shape: {data.shape}")
        print(f"Data columns: {data.columns.tolist()}")
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}. Please ensure the data file is in the correct directory.")
        return pd.DataFrame()  # Return an empty DataFrame if file not found
    
def get_experiment_files(experiment_number:str, base_path):
    """
    Get the file path for the specified experiment number, data type, and dataset type.

    Parameters:
    - experiment_number (int): The experiment number (e.g., exp1)
    - base_path (Path): The base path where the experiment folders are located.
    """
    
    # assign the base path
    csv_path = base_path / experiment_number
    
    # list all the csv files in the folder
    csv_files = list(csv_path.glob("*.csv"))
   
    # read all the csv files and store it in a dictionary
    data_dict = {}
    
    for csv_file in csv_files:
        var_name = csv_file.stem.replace(f"{experiment_number}_", "")
        data_dict[var_name] = pd.read_csv(csv_file)
        
    # assign each variable to its respective name
    
    X_class_train = data_dict["X_class_train"]
    y_class_train = data_dict["y_class_train"]
    X_class_val = data_dict["X_class_val"]
    y_class_val = data_dict["y_class_val"]
    X_class_test = data_dict["X_class_test"]
    y_class_test = data_dict["y_class_test"]
    X_reg_train = data_dict["X_reg_train"]
    y_reg_train = data_dict["y_reg_train"]
    X_reg_val = data_dict["X_reg_val"]
    y_reg_val = data_dict["y_reg_val"]
    X_reg_test = data_dict["X_reg_test"]
    y_reg_test = data_dict["y_reg_test"]
    
    # print the shape of each variable
    for var_name, df in data_dict.items():
        print(f"{var_name}: {df.shape}")
        
    # return all the variables
    return (X_class_train, y_class_train, X_class_val, y_class_val, X_class_test, y_class_test,
            X_reg_train, y_reg_train, X_reg_val, y_reg_val, X_reg_test, y_reg_test)

def load_data_at3(dataset_dir: str | Path, pattern: str = "*.csv", sep=";") -> tuple[pd.DataFrame, list[str]]:
    """
    Load all CSVs in `dataset_dir` (semicolon-separated), sort each by 'timeOpen',
    and concatenate into a single DataFrame.

    Returns:
        df_all (pd.DataFrame): concatenated DataFrame
        file_names (list[str]): list of CSV file names found
    """
    dataset_path = Path(dataset_dir)
    print("Current working directory:", Path.cwd())
    print("Dataset directory:", dataset_path.resolve())

    if not dataset_path.exists() or not dataset_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {dataset_path}")

    files = list(dataset_path.glob(pattern))
    file_names = [f.name for f in files]

    if not files:
        print("No CSV files found.")
        return pd.DataFrame(), file_names

    df_all = pd.DataFrame()
    for f in files:
        df = pd.read_csv(f, sep=sep)
        df = df.sort_values(by='timeOpen', ascending=True).reset_index(drop=True)
        print(f"Loaded {f.name} with shape {df.shape}")
        df_all = pd.concat([df_all, df], ignore_index=True)

    return df_all, file_names

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
