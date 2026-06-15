import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, TimeSeriesSplit, KFold, StratifiedKFold, RandomizedSearchCV, GridSearchCV
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
    if (X_val is None) != (y_val is None):
        raise ValueError("X_val and y_val must be provided together.")

    model.fit(X_train, y_train)
    
    results = {"model": model}
    
    # Get predictions
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train) if hasattr(model, "predict_proba") else None
    class_labels = getattr(model, "classes_", None)
    results["train_metrics"] = evaluate_classification(
        y_train,
        y_train_pred,
        y_train_proba,
        average=average,
        labels=class_labels,
    )
    
    if X_val is not None and y_val is not None:
        y_val_pred = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val) if hasattr(model, "predict_proba") else None
        results["val_metrics"] = evaluate_classification(
            y_val,
            y_val_pred,
            y_val_proba,
            average=average,
            labels=class_labels,
        )
        
    return results

def train_regressor(model, X_train, y_train, X_val=None, y_val=None):
    """
    Fits a regression model and evaluates it safely.
    """
    if (X_val is None) != (y_val is None):
        raise ValueError("X_val and y_val must be provided together.")

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
    if task_type not in {"classification", "regression"}:
        raise ValueError("task_type must be 'classification' or 'regression'.")
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2.")

    if is_time_series:
        cv_strategy = TimeSeriesSplit(n_splits=cv_folds)
    elif task_type == 'classification':
        cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=kwargs.get('random_state', 42))
    else:
        cv_strategy = KFold(n_splits=cv_folds, shuffle=True, random_state=kwargs.get('random_state', 42))

    # 2. Choose comprehensive scoring metrics based on task
    if task_type == 'classification':
        scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        if hasattr(model, "predict_proba") or hasattr(model, "decision_function"):
            if len(np.unique(y)) == 2:
                scoring.append('roc_auc')
            else:
                scoring.append('roc_auc_ovr_weighted')
    else:
        scoring = ['neg_mean_absolute_error', 'neg_root_mean_squared_error', 'r2']
        
    cv_results = cross_validate(
        model,
        X,
        y,
        cv=cv_strategy,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1,
        error_score="raise",
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

def tune_hyperparameters(
    model,
    param_grid,
    X_train,
    y_train,
    search_type="random",
    cv=5,
    n_iter=10,
    scoring="accuracy",
    random_state=42,
    verbose=False,
):
    """
    Wrapper for hyperparameter tuning using RandomizedSearchCV or GridSearchCV.

    Args:
        model: Sklearn-compatible model.
        param_grid: Dictionary with parameters names as keys and lists of parameter settings to try as values.
        X_train, y_train: Training data.
        search_type: 'random' or 'grid'.
        cv: Number of cross-validation folds.
        n_iter: Number of parameter settings that are sampled (only for 'random').
        scoring: Scoring metric (e.g., 'accuracy', 'roc_auc', 'neg_mean_squared_error').
        random_state: Seed for reproducibility.

    Returns:
        Best fitted estimator.
    """
    if cv < 2:
        raise ValueError("cv must be at least 2.")
    if search_type == "random":
        if n_iter < 1:
            raise ValueError("n_iter must be at least 1.")
        search = RandomizedSearchCV(
            model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            random_state=random_state,
            n_jobs=-1,
            verbose=int(verbose),
            error_score="raise",
        )
    elif search_type == "grid":
        search = GridSearchCV(
            model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=int(verbose),
            error_score="raise",
        )
    else:
        raise ValueError("search_type must be 'random' or 'grid'")

    search.fit(X_train, y_train)

    if verbose:
        print(f"Best parameters found: {search.best_params_}")
        print(f"Best cross-validation {scoring}: {search.best_score_:.4f}")

    return search.best_estimator_
