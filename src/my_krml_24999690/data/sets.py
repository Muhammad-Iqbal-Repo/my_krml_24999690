import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# build a function to load the data

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
        print(f"Data loaded successfully from {file_path}")
        print(f"Data shape: {data.shape}")
        print(f"Data columns: {data.columns.tolist()}")
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}. Please ensure the data file is in the correct directory.")
        return pd.DataFrame()  # Return an empty DataFrame if file not found

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

def standardize_train(df, columns):
    """Standardizes the specified columns in a training dataframe."""
    """
        Example:
        standardized_train_df = standardize_train(iris_df, ['sepal_length', 'sepal_width'])
        It will return the iris_df dataframe with standardized sepal_length and sepal_width columns
    """
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df, scaler

def plot_class_distribution(target_column, figsize, title, xlabel, ylabel):
    
    # count the unique values in the target column to serve as the basis for the coloring
    # use random colors based on the number of unique values
    """
        Example:
        plot_class_distribution(df['species'], (10, 5), 'Iris Species Distribution', 'Species', 'Count')
        It will plot a bar chart showing the distribution of the species in the iris dataset
    """
    
    value_counts = target_column.value_counts()
    plt.figure(figsize=figsize)
    value_counts.plot(kind='bar', color=plt.cm.Paired.colors[:len(value_counts)])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=0)
    plt.show()
    
    print(f"Class distribution:\n{value_counts}")
    
    
def boxplot_regression(df, col, target_col, figsize):
    """Plots a boxplot of a categorical column relative to a regression target variable."""
    """
        Example:
        boxplot_regression(iris_df, 'species', 'sepal_length', (10, 5))
        It will plot a boxplot showing the distribution of sepal_length for each species in the iris dataset
    """
    plt.figure(figsize=figsize)
    sns.boxplot(data=df, x=col, y=target_col, palette='Set2', hue=col)
    plt.title(f"Boxplot of '{target_col}' by '{col}'")
    plt.xlabel(col)
    plt.ylabel(target_col)
    plt.show()
    
def plot_categorical_distribution_with_target(df, col, target_col, figsize):
    """Plots the distribution of a categorical column relative to a target variable."""
    """
        Example:
        plot_categorical_distribution_with_target(iris_df, 'species', 'is_setosa', (10, 5))
        It will plot a bar chart showing the distribution of the species in the iris dataset relative to the target variable is_setosa
    """
    plt.figure(figsize=figsize)
    sns.countplot(data=df, x=col, hue=target_col, palette='Set2')
    plt.title(f"Distribution of '{col}' by '{target_col}'")
    plt.xlabel(target_col)
    plt.ylabel(col)
    plt.show()

def plot_numerical_with_target(df, col, target_col, figsize, type):
    """Plot a histogram of a numerical column colored by a target variable.
    Args:
        df (_type_): dataframe source
        col (_type_): numerical column to plot
        target_col (_type_): target column for coloring
        figsize (_type_): figure size
    """
    if type == "classification":
        plt.figure(figsize=figsize)
        sns.histplot(data=df, x=col, hue=target_col, bins=20, kde=True, palette='Set2')
        plt.title(f"Distribution of '{col}' by '{target_col}'")
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()
    else:
        plt.figure(figsize=figsize)
        sns.scatterplot(data=df, x=col, y=target_col)
        plt.title(f"Scatter plot of '{col}' vs '{target_col}'")
        plt.xlabel(col)
        plt.ylabel(target_col)
        plt.show()

def standardize_test(df, columns, scaler):
    """Standardizes the specified columns in a testing dataframe."""
    """
        Example:
        standardized_test_df = standardize_test(iris_df, ['sepal_length', 'sepal_width'], scaler)
        It will return the iris_df dataframe with standardized sepal_length and sepal_width columns
    """
    if scaler is None:
        raise ValueError("Scaler must be provided for test data")
    df[columns] = scaler.transform(df[columns])
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


def evaluate_model(model, X_train, y_train, X_val, y_val):
    """Evaluates a model's performance on training and testing sets."""
    """
        Example:
        evaluate_model(model, X_train, y_train, X_test, y_test)
        It will print the accuracy of the model on training and testing sets
    """
    
    model.fit(X_train, y_train)
    
    # Predict probabilities
    y_train_pred_proba = model.predict_proba(X_train)[:, 1]
    y_val_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Compute AUROC
    train_score = roc_auc_score(y_train, y_train_pred_proba)
    val_score = roc_auc_score(y_val, y_val_pred_proba)
    
    # Print results
    model_name = model.__class__.__name__
    print(f"{model_name} AUROC on training data: {train_score:.4f}")
    print(f"{model_name} AUROC on validation data: {val_score:.4f}")
    
    return y_train_pred_proba, y_val_pred_proba
    

def print_auroc(y_true, y_pred_proba, type='train'):
    """Prints the AUROC score given true labels and predicted probabilities."""
    """
        Example:
        print_auroc(y_test, y_test_pred_proba)
        It will print the AUROC
    """
    score = roc_auc_score(y_true, y_pred_proba)
    if type == 'train':
        print(f"AUROC Score on the training data: {score:.4f}")
    else:
        print(f"AUROC Score on the validation data: {score:.4f}")

def kaggle_submission(model, X_test, sample_path, output_path, target_col=''):
    """Submit to Kaggle competition."""
    """
        Example:
        kaggle_submission(model, X_test, 'sample_submission.csv', 'submission.csv', target_col='target')
        It will create a submission file 'submission.csv' ready for Kaggle submission
    """
    
    # Predict probabilities
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Load sample submission
    submission_df = pd.read_csv(sample_path)
    
    # Assign predictions to the target column
    submission_df[target_col] = y_test_pred_proba
    
    # Save submission file
    submission_df.to_csv(output_path, index=False)
    
    print(f"Submission file saved to {output_path}")
    
    return submission_df

def at2_create_time_series_splits(df_classification, df_regression, validation_threshold='2024-07-01'):
    """
    Splits classification and regression dataframes into train, validation, and test sets.

    This function is designed for time-series data and splits it as follows:
    1.  **Test Set**: Data from the year 2025 onwards.
    2.  **Validation Set**: Data from the last 6 months of 2024 (July 1st onwards).
    3.  **Training Set**: All data before July 1st, 2024.

    Args:
        df_classification (pd.DataFrame): The complete dataframe for the classification task.
                                          It must contain a 'time' column.
        df_regression (pd.DataFrame): The complete dataframe for the regression task.
                                      It must contain a 'time' column.

    Returns:
        tuple: A tuple containing six dataframes in the order:
               (class_train, class_val, class_test,
                reg_train, reg_val, reg_test)
    """
    # ensure the 'time' column is in datetime format
    df_classification['time'] = pd.to_datetime(df_classification['time'])
    
    # split the classification dataframe
    # Isolate data before 2025 for training and validatio
    pre_2025_class_data = df_classification[df_classification['time'].dt.year < 2025].copy()
    
    # the test set is all data from 2025 onwards
    class_test = df_classification[df_classification['time'].dt.year >= 2025].copy()
    
    # split the pre-2025 data into training and validation sets
    class_val = pre_2025_class_data[pre_2025_class_data['time'] >= '2024-07-01'].copy()
    class_train = pre_2025_class_data[pre_2025_class_data['time'] < '2024-07-01'].copy()

    # regression dataframe
    # ensure the 'time' column is in datetime format
    df_regression['time'] = pd.to_datetime(df_regression['time'])

    # split the data prior to 2025 for training and validation
    pre_2025_reg_data = df_regression[df_regression['time'].dt.year < 2025].copy()

    # the test set is all data from 2025 onwards
    reg_test = df_regression[df_regression['time'].dt.year >= 2025].copy()

    # split the pre-2025 data into training and validation sets
    reg_val = pre_2025_reg_data[pre_2025_reg_data['time'] >= '2024-07-01'].copy()
    reg_train = pre_2025_reg_data[pre_2025_reg_data['time'] < '2024-07-01'].copy()

    # print summary of the splits
    print("Data Split Summary:")
    print("Classification Task:")
    print(f"Training set shape:{class_train.shape}")
    print(f"Validation set shape:{class_val.shape}")
    print(f"Test set shape:{class_test.shape}")

    print("\nRegression Task:")
    print(f"Training set shape:{reg_train.shape}")
    print(f"Validation set shape:{reg_val.shape}")
    print(f"Test set shape:{reg_test.shape}")
    
    return class_train, class_val, class_test, reg_train, reg_val, reg_test

