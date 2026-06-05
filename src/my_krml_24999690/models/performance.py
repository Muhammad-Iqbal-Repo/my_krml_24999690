import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss,
    mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
)

def evaluate_classification(y_true, y_pred, y_pred_proba=None, average='macro'):
    """
    Calculates comprehensive classification metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted hard labels.
        y_pred_proba: Predicted probabilities (for ROC AUC and Brier Score). Default is None.
        average: The averaging strategy for multiclass targets ('binary', 'micro', 'macro', 'weighted').

    Returns:
        dict: A dictionary of classification metrics.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }

    if y_pred_proba is not None:
        try:
            # Handle binary vs multiclass ROC AUC
            if len(np.unique(y_true)) == 2:
                # If y_pred_proba is 2D (Nx2), extract the positive class probabilities
                probs = y_pred_proba[:, 1] if y_pred_proba.ndim == 2 else y_pred_proba
                metrics["roc_auc"] = float(roc_auc_score(y_true, probs))
                metrics["brier_score"] = float(brier_score_loss(y_true, probs))
            else:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average=average))
        except Exception as e:
            metrics["roc_auc"] = None
            print(f"Warning: Could not calculate ROC AUC. {e}")

    return metrics

def evaluate_regression(y_true, y_pred):
    """
    Calculates comprehensive regression metrics.

    Args:
        y_true: True continuous values.
        y_pred: Predicted continuous values.

    Returns:
        dict: A dictionary of regression metrics.
    """
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2_score": float(r2_score(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred))
    }

def print_metrics(metrics_dict, title="Metrics Summary"):
    """Pretty prints a dictionary of metrics."""
    print(f"--- {title} ---")
    for key, value in metrics_dict.items():
        if value is not None:
            print(f"{key.upper().replace('_', ' ')}: {value:.4f}")
    print("-" * 25)

def kaggle_submission(model, X_test, sample_path, output_path, target_col='target', predict_proba=True):
    """
    Generates a submission file for a Kaggle competition.
    """
    if predict_proba and hasattr(model, "predict_proba"):
        # For classification competitions that require probability of positive class
        preds = model.predict_proba(X_test)[:, 1]
    else:
        # For regression or hard classification
        preds = model.predict(X_test)
        
    # Load sample submission
    submission_df = pd.read_csv(sample_path)
    
    # Assign predictions to the target column
    submission_df[target_col] = preds
    
    # Save submission file
    submission_df.to_csv(output_path, index=False)
    
    print(f"Submission file saved to {output_path}")
    return submission_df
