import pandas as pd
from sklearn.prepocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

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
            
def drop_duplicates(df):
    """Drops duplicated rows from a dataframe and returns the cleaned dataframe."""
    """
        Example:
        cleaned_df = drop_duplicates(iris_df)
        It will return the iris_df dataframe without duplicated rows
    """
    return df.drop_duplicates().reset_index(drop=True)

def check_duplicates_df(df):
    """Returns the number of duplicated rows in a dataframe."""
    """
        Example:
        num_duplicates = check_duplicates_df(iris_df)
        It will return the number of duplicated rows in the iris_df dataframe
    """
    print(f"Number of duplicated rows in the dataframe: {df.duplicated().sum()}")

def standardize_data(df, columns, type='train'):
    """Standardizes the specified columns in a dataframe and returns the standardized dataframe."""
    """
        Example:
        standardized_df = standardize_data(iris_df, ['sepal_length', 'sepal_width'], type='train')
        It will return the iris_df dataframe with standardized sepal_length and sepal_width columns
    """
    
    scaler = StandardScaler()
    if type == 'train':
        df[columns] = scaler.fit_transform(df[columns])
    else:
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


def split_random(X, y, test_size=0.2, random_state=42, stratify=None):
    """Splits a dataframe into training and testing sets."""
    """
        Example:
        X_train, X_test, y_train, y_test = split_random(iris_df, 'species', test_size=0.2, random_state=42)
        It will return the training and testing sets for features and target
    """
    if stratify:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_train, y_train, X_val, y_val):
    """Evaluates a model's performance on training and testing sets."""
    """
        Example:
        evaluate_model(model, X_train, y_train, X_test, y_test)
        It will print the accuracy of the model on training and testing sets
    """
    
    model.fit(X_train, y_train)
    
    # Train the model
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
    

def print_auroc(y_true, y_pred_proba):
    """Prints the AUROC score given true labels and predicted probabilities."""
    """
        Example:
        print_auroc(y_test, y_test_pred_proba)
        It will print the AUROC score for the test set
    """
    score = roc_auc_score(y_true, y_pred_proba)
    print(f"AUROC: {score:.4f}")
    return score

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

