import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import gaussian_kde
from pathlib import Path
from sklearn.impute import SimpleImputer
import os
from typing import Sequence
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error, root_mean_squared_error, roc_auc_score

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

def at2_create_time_series_splits(df, validation_threshold, type):
    # create docstring
    """
    Splits a time series dataframe into training, validation, and test sets based on a specified validation threshold date.
    Args:
        df (pd.DataFrame): The input dataframe containing a 'time' column.
        validation_threshold (str): The date string (YYYY-MM-DD) to split the training and validation sets.
        type (str): The type of task, either "classification" or "regression".
    Returns:
        tuple: A tuple containing the training, validation, and test sets as dataframes.
    """ 
    
    if type == "classification":
        # classification dataframe
        # ensure the 'time' column is in datetime format
        df['time'] = pd.to_datetime(df['time'])
        
        # split the classification dataframe
        # Isolate data before 2025 for training and validation
        pre_2025_class_data = df[df['time'].dt.year < 2025].copy()
        
        # the test set is all data from 2025 onwards
        class_test = df[df['time'].dt.year >= 2025].copy()
        
        # split the pre-2025 data into training and validation sets
        class_val = pre_2025_class_data[pre_2025_class_data['time'] >= validation_threshold].copy()
        class_train = pre_2025_class_data[pre_2025_class_data['time'] < validation_threshold].copy()

        # print the summary of the splits
        
        print("Data Split Summary:")
        print("Classification Task:")
        print(f"Training set shape:{class_train.shape}")
        print(f"Validation set shape:{class_val.shape}")
        print(f"Test set shape:{class_test.shape}")
        
        
        return class_train, class_val, class_test
    
    
    if type == "regression":
        # regression dataframe
        # ensure the 'time' column is in datetime format
        df['time'] = pd.to_datetime(df['time'])

        # split the data prior to 2025 for training and validation
        pre_2025_reg_data = df[df['time'].dt.year < 2025].copy()

        # the test set is all data from 2025 onwards
        reg_test = df[df['time'].dt.year >= 2025].copy()

        # split the pre-2025 data into training and validation sets
        reg_val = pre_2025_reg_data[pre_2025_reg_data['time'] >= validation_threshold].copy()
        reg_train = pre_2025_reg_data[pre_2025_reg_data['time'] < validation_threshold].copy()
        
        # print the summary of the splits
        print("Data Split Summary:")
        print("Regression Task:")
        print(f"Training set shape:{reg_train.shape}")
        print(f"Validation set shape:{reg_val.shape}")
        print(f"Test set shape:{reg_test.shape}")        
        
        return reg_train, reg_val, reg_test
    
def at2_train_evaluate_classification_model(model, X_train, y_train, X_val, y_val, X_test, y_test, experiment_name:str, dict_results):
    """
    Trains and evaluates a classification model.

    Args:
        model: The classification model to train and evaluate.
        X_train: The training features.
        y_train: The training labels.
        X_val: The validation features.
        y_val: The validation labels.
        X_test: The test features.
        y_test: The test labels.
        experiment_name: The name of the experiment (for logging purposes).
        dict_results: A dictionary to store the results of the experiment.
    """
    # get model name
    model_name = model.__class__.__name__
    
    # fit the model
    
    model.fit(X_train, y_train)
    
    # predict the results
    
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    # calculate the performance metrics
    
    dict_results[experiment_name] = {
        "model_name": model_name,
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "val_accuracy": accuracy_score(y_val, y_val_pred),
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "train_precision": precision_score(y_train, y_train_pred),
        "val_precision": precision_score(y_val, y_val_pred),
        "test_precision": precision_score(y_test, y_test_pred),
        "train_recall": recall_score(y_train, y_train_pred),
        "val_recall": recall_score(y_val, y_val_pred),
        "test_recall": recall_score(y_test, y_test_pred),
        "train_f1": f1_score(y_train, y_train_pred),
        "val_f1": f1_score(y_val, y_val_pred),
        "test_f1": f1_score(y_test, y_test_pred),
        "train_roc_auc": roc_auc_score(y_train, y_train_pred),
        "val_roc_auc": roc_auc_score(y_val, y_val_pred),
        "test_roc_auc": roc_auc_score(y_test, y_test_pred)
    }
    
    # print the results
    
    print(f"Experiment: {experiment_name}")
    print(f"Model: {model_name}")
    print(f"Train Accuracy: {dict_results[experiment_name]['train_accuracy']:.4f}")
    print(f"Validation Accuracy: {dict_results[experiment_name]['val_accuracy']:.4f}")
    print(f"Test Accuracy: {dict_results[experiment_name]['test_accuracy']:.4f}")
    print(f"Train Precision: {dict_results[experiment_name]['train_precision']:.4f}")
    print(f"Validation Precision: {dict_results[experiment_name]['val_precision']:.4f}")
    print(f"Test Precision: {dict_results[experiment_name]['test_precision']:.4f}")
    print(f"Train Recall: {dict_results[experiment_name]['train_recall']:.4f}")
    print(f"Validation Recall: {dict_results[experiment_name]['val_recall']:.4f}")
    print(f"Test Recall: {dict_results[experiment_name]['test_recall']:.4f}")
    print(f"Train F1-Score: {dict_results[experiment_name]['train_f1']:.4f}")
    print(f"Validation F1-Score: {dict_results[experiment_name]['val_f1']:.4f}")
    print(f"Test F1-Score: {dict_results[experiment_name]['test_f1']:.4f}")
    print(f"Train ROC-AUC: {dict_results[experiment_name]['train_roc_auc']:.4f}")
    print(f"Validation ROC-AUC: {dict_results[experiment_name]['val_roc_auc']:.4f}")
    print(f"Test ROC-AUC: {dict_results[experiment_name]['test_roc_auc']:.4f}")
    
    # print confusion matrix display
    
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_val = confusion_matrix(y_val, y_val_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)
    
    disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train)
    disp_val = ConfusionMatrixDisplay(confusion_matrix=cm_val)
    disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test)
    
    disp_train.plot()
    plt.title(f"Confusion Matrix - Train Set ({experiment_name})")
    plt.show()
    
    disp_val.plot()
    plt.title(f"Confusion Matrix - Validation Set ({experiment_name})")
    plt.show()
    
    disp_test.plot()
    plt.title(f"Confusion Matrix - Test Set ({experiment_name})")
    plt.show()
    
    # print a separator
    
    print("-" * 50)
    print(f"The results of the experiment '{experiment_name}' have been recorded.")
    print("-" * 50)
    
def at2_prepare_training(train, val, test, final_features, target, type):
       
    """
    Splits the data into features and target for classification and regression tasks.
    Args:
        train (pd.DataFrame): The training dataframe.
        val (pd.DataFrame): The validation dataframe.
        test (pd.DataFrame): The test dataframe.
        final_features (list): List of feature column names to include.
        target (str): The target column name.
        type (str): The type of task - "classification" or "regression".
    Returns:
        tuple: A tuple containing the features and target for training, validation, and test sets.
    """
    
    if type == "classification":
        X_class_train = train[final_features].drop(columns=['time'])
        y_class_train = train[target]
        X_class_val = val[final_features].drop(columns=['time'])
        y_class_val = val[target]
        X_class_test = test[final_features].drop(columns=['time'])
        y_class_test = test[target]
        
        # print the shapes of the datasets
        
        print("Classification Task:")
        print(f"X_class_train: {X_class_train.shape}, y_class_train: {y_class_train.shape}")
        print(f"X_class_val: {X_class_val.shape}, y_class_val: {y_class_val.shape}")
        print(f"X_class_test: {X_class_test.shape}, y_class_test: {y_class_test.shape}")
        
        return X_class_train, y_class_train, X_class_val, y_class_val, X_class_test, y_class_test
    if type == "regression":
        X_reg_train = train[final_features].drop(columns=['time'])
        y_reg_train = train[target]
        X_reg_val = val[final_features].drop(columns=['time'])
        y_reg_val = val[target]
        X_reg_test = test[final_features].drop(columns=['time'])
        y_reg_test = test[target]
        
        # print the shapes of the datasets
        print("Regression Task:")
        print(f"X_reg_train: {X_reg_train.shape}, y_reg_train: {y_reg_train.shape}")
        print(f"X_reg_val: {X_reg_val.shape}, y_reg_val: {y_reg_val.shape}")
        print(f"X_reg_test: {X_reg_test.shape}, y_reg_test: {y_reg_test.shape}")
        
        
        return X_reg_train, y_reg_train, X_reg_val, y_reg_val, X_reg_test, y_reg_test
   
def train_evaluate_regression_model(model, X_train, y_train, X_val, y_val, X_test, y_test, experiment_name:str, dict_results):
    """
    Trains and evaluates a regression model.
    Args:
        model: The regression model to train and evaluate.
        X_train: The training features.
        y_train: The training labels.
        X_val: The validation features.
        y_val: The validation labels.
        X_test: The test features.
        y_test: The test labels.
        experiment_name: The name of the experiment (for logging purposes).
        dict_results: A dictionary to store the results of the experiment.
    """
    
    # get model name
    model_name = model.__class__.__name__
    
    # fit the model
    
    model.fit(X_train, y_train)
    
    # predict the results
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    # calculate the performance metrics
    
    dict_results[experiment_name] = {
        "model_name": model_name,
        "train_mae" : mean_absolute_error(y_train, y_train_pred),
        "val_mae" : mean_absolute_error(y_val, y_val_pred),
        "test_mae" : mean_absolute_error(y_test, y_test_pred),
        "train_rmse" : root_mean_squared_error(y_train, y_train_pred),
        "val_rmse" : root_mean_squared_error(y_val, y_val_pred),
        "test_rmse" : root_mean_squared_error(y_test, y_test_pred)
    }
    
    # print the results
    
    print(f"Experiment: {experiment_name}")
    print(f"Model: {model_name}")
    print(f"Train MAE: {dict_results[experiment_name]['train_mae']:.4f}")
    print(f"Validation MAE: {dict_results[experiment_name]['val_mae']:.4f}")
    print(f"Test MAE: {dict_results[experiment_name]['test_mae']:.4f}")
    print(f"Train RMSE: {dict_results[experiment_name]['train_rmse']:.4f}")
    print(f"Validation RMSE: {dict_results[experiment_name]['val_rmse']:.4f}")
    print(f"Test RMSE: {dict_results[experiment_name]['test_rmse']:.4f}")
    
    # print a separator
    
    # convert the dict_results to a dataframe for better visualization

    df_results = pd.DataFrame.from_dict(dict_results, orient='index')
    df_results.index.name = 'experiment_name'
    df_results.reset_index(inplace=True)

    print(df_results.to_string(index=False, max_rows=250))

    print("-" * 50)
    print(f"The results of the experiment '{experiment_name}' have been recorded.")
    print("-" * 50)
    
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

import pandas as pd
from typing import Iterable, Optional

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


def plot_distribution_and_trend(df, col, date_col, figsize=(), log_scale=True):
    """
    Plots distribution (with KDE + optional log scale) and yearly trend of a continuous target.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe containing at least `col` and `date_col`.
    col : str
        Name of the continuous  column.
    date_col : str
        Name of the datetime column.
    log_scale : bool, optional (default=True)
        If True, also shows a log-scaled version of the histogram.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[col, date_col])
    df = df[df[col].notna()]

    # --- Distribution with KDE ---
    x = df[col].dropna()
    plt.figure(figsize=figsize if figsize else (6, 4))
    plt.hist(x, bins=50, edgecolor='black', alpha=0.6, density=True)
    kde = gaussian_kde(x)
    xs = np.linspace(x.min(), x.max(), 300)
    plt.plot(xs, kde(xs), color='red', lw=1.5, label='KDE')
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    # --- Optional log-scale plot ---
    if log_scale:
        plt.figure(figsize=figsize if figsize else (6, 4))
        plt.hist(np.log1p(x), bins=50, edgecolor='black', alpha=0.6, density=True)
        kde_log = gaussian_kde(np.log1p(x))
        xs_log = np.linspace(np.log1p(x).min(), np.log1p(x).max(), 300)
        plt.plot(xs_log, kde_log(xs_log), color='red', lw=1.5, label='KDE (log scale)')
        plt.title(f"Log-Scaled Distribution of {col}")
        plt.xlabel(f"log(1 + {col})")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    # --- Line chart per year ---
    df['year'] = df[date_col].dt.year
    yearly_mean = df.groupby('year')[col].mean()

    plt.figure(figsize=figsize if figsize else (6, 4))
    plt.plot(yearly_mean.index, yearly_mean.values, marker='o')
    plt.title(f"{col} Trend per Year")
    plt.xlabel("Year")
    plt.ylabel(f"Average {col}")
    plt.grid(alpha=0.3)
    plt.show()