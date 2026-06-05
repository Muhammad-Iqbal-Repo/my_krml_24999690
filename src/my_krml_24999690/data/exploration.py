import pandas as pd
import numpy as np
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
        

def drop_columns(df, columns):
    """Drops one or multiple columns from a dataframe and returns the modified dataframe."""
    """
        Example:
        modified_df = drop_columns(iris_df, ['species', 'petal_width'])
        It will return the iris_df dataframe without the 'species' and 'petal_width' columns
    """
    try:
        return df.drop(columns=columns)
    except Exception as e:
        print(f"Error occurred while dropping columns: {e}")
        return df

def check_duplicates(**datasets):
    """Prints the number of duplicated rows for multiple datasets passed as keyword arguments."""
    """
        Example:
        check_duplicates(iris=iris_df, titanic=titanic_df)
        iris and titanic are the names of the datasets
        iris_df and titanic_df are the actual dataframes
        
        It will print:
        Number of duplicated rows in the iris data: 0
        Number of duplicated rows in the titanic data: 0
    """
    for name, df in datasets.items():
        print(f"Number of duplicated rows in the {name} data: {df.duplicated().sum()}")
        

def check_nulls(**datasets):
    """Prints the number of null values for multiple datasets passed as keyword arguments."""
    """
        Example:
        check_nulls(iris=iris_df, titanic=titanic_df)
        iris and titanic are the names of the datasets
        iris_df and titanic_df are the actual dataframes
        It will print:
        Null values in the iris dataset: 0
        Null values in the titanic dataset: 177
    """
    for name, df in datasets.items():
        print(f"Null values in the {name} dataset: {df.isnull().sum().sum()}")
        
def check_null_columns(**datasets):
    """Prints the number of columns with null values for multiple datasets passed as keyword arguments."""
    """
        Example:
        check_null_columns(iris=iris_df, titanic=titanic_df
        iris and titanic are the names of the datasets
        iris_df and titanic_df are the actual dataframes
        
        It will print:
        Number of columns with null values in the iris dataset:
        No columns with null values
        Number of columns with null values in the titanic dataset:
        Age          1
    """
    for name, df in datasets.items():
        null_columns = df.isnull().sum()
        null_columns = null_columns[null_columns > 0]
        print(f"Number of columns with null values in the {name} dataset:")
        if null_columns.empty:
            print("No columns with null values")
        else:
            print(null_columns)
            
def check_duplicates_df(df):
    """Returns the number of duplicated rows in a dataframe."""
    """
        Example:
        num_duplicates = check_duplicates_df(iris_df)
        It will return the number of duplicated rows in the iris_df dataframe
    """
    print(f"Number of duplicated rows in the dataframe: {df.duplicated().sum()}")

def comprehensive_report(df):
    """
    Prints a comprehensive report of a pandas DataFrame.
    Includes shape, data types, descriptive statistics, missing values, and duplicate rows.
    """
    print("=" * 50)
    print("COMPREHENSIVE DATAFRAME REPORT")
    print("=" * 50)
    print(f"\n1. Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    print("\n2. Data Types:")
    print(df.dtypes)
    
    print("\n3. Missing Values (Top 10 columns with most missing):")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False).head(10)
    if missing.empty:
        print("No missing values found.")
    else:
        print(missing)
        
    print("\n4. Duplicated Rows:")
    print(f"{df.duplicated().sum()} duplicate rows")
    
    print("\n5. Descriptive Statistics (Numerical):")
    # Using .T for transpose so it's easier to read with many columns
    numerical_desc = df.select_dtypes(include=[np.number])
    if not numerical_desc.empty:
        print(numerical_desc.describe().T)
    else:
        print("No numerical columns found.")
        
    print("\n6. Descriptive Statistics (Categorical/Non-Numerical):")
    categorical_desc = df.select_dtypes(exclude=[np.number])
    if not categorical_desc.empty:
        print(categorical_desc.describe().T)
    else:
        print("No categorical columns found.")
    
    print("=" * 50)
