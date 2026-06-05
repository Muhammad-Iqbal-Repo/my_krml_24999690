import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, TimeSeriesSplit, KFold, StratifiedKFold
import warnings
from .performance import evaluate_classification, evaluate_regression

def train_classifier(model, X_train, y_train, X_val=None, y_val=None, average='macro'):
    """
    Fits a classification model and evaluates it safely.

    Args:
        model: Sklearn-compatible classifier.
        X_train, y_train: Training data.
        X_val, y_val: Optional validation data.
        average: Averaging strategy for metrics ('binary', 'macro', etc.).

    Returns:
        dict: Contains the trained 'model', 'train_metrics', and 'val_metrics' (if provided).
    """
    model.fit(X_train, y_train)
    
    results = {"model": model}
    
    # Get predictions
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train) if hasattr(model, "predict_proba") else None
    results["train_metrics"] = evaluate_classification(y_train, y_train_pred, y_train_proba, average=average)
    
    if X_val is not None and y_val is not None:
        y_val_pred = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val) if hasattr(model, "predict_proba") else None
        results["val_metrics"] = evaluate_classification(y_val, y_val_pred, y_val_proba, average=average)
        
    return results

def train_regressor(model, X_train, y_train, X_val=None, y_val=None):
    """
    Fits a regression model and evaluates it safely.
    """
    model.fit(X_train, y_train)
    
    results = {"model": model}
    
    y_train_pred = model.predict(X_train)
    results["train_metrics"] = evaluate_regression(y_train, y_train_pred)
    
    if X_val is not None and y_val is not None:
        y_val_pred = model.predict(X_val)
        results["val_metrics"] = evaluate_regression(y_val, y_val_pred)
        
    return results

def cross_validate_model(model, X, y, task_type='classification', cv_folds=5, is_time_series=False, **kwargs):
    """
    Robust cross-validation that automatically adapts to the data type.

    Args:
        model: Sklearn-compatible model.
        X: Feature dataframe or array.
        y: Target series or array.
        task_type: 'classification' or 'regression'.
        cv_folds: Number of folds (default: 5).
        is_time_series: If True, uses TimeSeriesSplit (respects chronological order, no data leakage from future to past).
        **kwargs: Additional arguments like 'random_state'.

    Returns:
        dict: Aggregated cross-validation metrics (mean and std across folds).
    """
    # 1. Choose the safest Splitting Strategy
    if is_time_series:
        cv_strategy = TimeSeriesSplit(n_splits=cv_folds)
    elif task_type == 'classification':
        cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=kwargs.get('random_state', 42))
    else:
        cv_strategy = KFold(n_splits=cv_folds, shuffle=True, random_state=kwargs.get('random_state', 42))

    # 2. Choose comprehensive scoring metrics based on task
    if task_type == 'classification':
        scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        # Try ROC_AUC if classifier supports proba
        if hasattr(model, "predict_proba"):
            if len(np.unique(y)) == 2:
                scoring.append('roc_auc')
            else:
                scoring.append('roc_auc_ovr_weighted')
    else:
        scoring = ['neg_mean_absolute_error', 'neg_root_mean_squared_error', 'r2']
        
    # 3. Execute Cross Validation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv_results = cross_validate(
            model, X, y, cv=cv_strategy, scoring=scoring, 
            return_train_score=True, n_jobs=-1
        )
    
    # 4. Aggregate results into a readable dictionary
    summary = {}
    for metric_name in cv_results.keys():
        if metric_name.startswith('test_') or metric_name.startswith('train_'):
            val_array = cv_results[metric_name]
            
            # Sklearn returns negative errors for loss functions, so we flip them back
            clean_metric_name = metric_name
            if 'neg_' in clean_metric_name:
                val_array = -val_array
                clean_metric_name = clean_metric_name.replace('neg_', '')
                
            summary[f"{clean_metric_name}_mean"] = float(np.mean(val_array))
            summary[f"{clean_metric_name}_std"] = float(np.std(val_array))
            
    return summary

# ---------------------------------------------------------
# DEPRECATED FUNCTIONS (Kept temporarily for backward compat)
# ---------------------------------------------------------
def at2_create_time_series_splits(df, validation_threshold, type):
    warnings.warn("at2_create_time_series_splits is deprecated. Use split_data from engineering.py instead.", DeprecationWarning)
    # Basic backwards compatibility stub
    df['time'] = pd.to_datetime(df['time'])
    pre_2025 = df[df['time'].dt.year < 2025].copy()
    test = df[df['time'].dt.year >= 2025].copy()
    val = pre_2025[pre_2025['time'] >= validation_threshold].copy()
    train = pre_2025[pre_2025['time'] < validation_threshold].copy()
    return train, val, test

def at2_prepare_training(train, val, test, final_features, target, type):
    warnings.warn("at2_prepare_training is deprecated.", DeprecationWarning)
    return (
        train[final_features].drop(columns=['time'], errors='ignore'), train[target],
        val[final_features].drop(columns=['time'], errors='ignore'), val[target],
        test[final_features].drop(columns=['time'], errors='ignore'), test[target]
    )
