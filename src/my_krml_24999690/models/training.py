import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, mean_absolute_error, root_mean_squared_error

def at2_create_time_series_splits(df, validation_threshold, type):
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
    
    print("-" * 50)
    print(f"The results of the experiment '{experiment_name}' have been recorded.")
    print("-" * 50)
    
def at2_prepare_training(train, val, test, final_features, target, type):
    """
    Splits the data into features and target for classification and regression tasks.
    """
    if type == "classification":
        X_class_train = train[final_features].drop(columns=['time'])
        y_class_train = train[target]
        X_class_val = val[final_features].drop(columns=['time'])
        y_class_val = val[target]
        X_class_test = test[final_features].drop(columns=['time'])
        y_class_test = test[target]
        
        print("Classification Task:")
        print(f"X_class_train: {X_class_train.shape}, y_class_train: {y_class_train.shape}")
        
        return X_class_train, y_class_train, X_class_val, y_class_val, X_class_test, y_class_test
        
    if type == "regression":
        X_reg_train = train[final_features].drop(columns=['time'])
        y_reg_train = train[target]
        X_reg_val = val[final_features].drop(columns=['time'])
        y_reg_val = val[target]
        X_reg_test = test[final_features].drop(columns=['time'])
        y_reg_test = test[target]
        
        print("Regression Task:")
        print(f"X_reg_train: {X_reg_train.shape}, y_reg_train: {y_reg_train.shape}")
        
        return X_reg_train, y_reg_train, X_reg_val, y_reg_val, X_reg_test, y_reg_test
   
def train_evaluate_regression_model(model, X_train, y_train, X_val, y_val, X_test, y_test, experiment_name:str, dict_results):
    """
    Trains and evaluates a regression model.
    """
    model_name = model.__class__.__name__
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    dict_results[experiment_name] = {
        "model_name": model_name,
        "train_mae" : mean_absolute_error(y_train, y_train_pred),
        "val_mae" : mean_absolute_error(y_val, y_val_pred),
        "test_mae" : mean_absolute_error(y_test, y_test_pred),
        "train_rmse" : root_mean_squared_error(y_train, y_train_pred),
        "val_rmse" : root_mean_squared_error(y_val, y_val_pred),
        "test_rmse" : root_mean_squared_error(y_test, y_test_pred)
    }
    
    print(f"Experiment: {experiment_name}")
    print(f"Model: {model_name}")
    print(f"Train MAE: {dict_results[experiment_name]['train_mae']:.4f}")
    
    df_results = pd.DataFrame.from_dict(dict_results, orient='index')
    df_results.index.name = 'experiment_name'
    df_results.reset_index(inplace=True)

    print(df_results.to_string(index=False, max_rows=250))
    print("-" * 50)
