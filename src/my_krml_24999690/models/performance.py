import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss,
    mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error,
    confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
)
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_consistent_length, column_or_1d
import matplotlib.pyplot as plt
import seaborn as sns

_SUPPORTED_AVERAGES = {"binary", "micro", "macro", "weighted"}


def _resolve_class_labels(y_true, y_pred, labels):
    if labels is None:
        return np.asarray(unique_labels(y_true, y_pred))

    class_labels = np.asarray(labels)
    if class_labels.ndim != 1 or class_labels.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional sequence.")
    if len(set(class_labels.tolist())) != len(class_labels):
        raise ValueError("labels must not contain duplicate values.")

    observed_labels = unique_labels(y_true, y_pred)
    missing_labels = [
        label for label in observed_labels
        if label not in class_labels
    ]
    if missing_labels:
        raise ValueError(
            f"labels is missing observed class values: {missing_labels}."
        )

    return class_labels


def evaluate_classification(
    y_true,
    y_pred,
    y_pred_proba=None,
    average="macro",
    labels=None,
    pos_label=None,
    multi_class="ovr",
):
    """
    Calculate classification metrics and a confusion matrix.

    Args:
        y_true: True labels.
        y_pred: Predicted hard labels.
        y_pred_proba: Predicted probabilities used for ROC AUC. For binary
            classification, pass a one-dimensional positive-class probability
            vector or a two-column matrix. For multiclass classification, pass
            one probability column per class.
        average: Averaging strategy for precision, recall, F1, and multiclass
            ROC AUC. Supported values are 'binary', 'micro', 'macro', and
            'weighted'.
        labels: Class labels in the same order as the columns in
            y_pred_proba. If omitted, labels are inferred from y_true and
            y_pred.
        pos_label: Positive class for binary metrics. Defaults to the last
            value in labels.
        multi_class: Multiclass ROC AUC strategy, either 'ovr' or 'ovo'.

    Returns:
        dict: Accuracy, precision, recall, F1, ROC AUC, Brier score for
        binary probabilities, class labels, and the confusion matrix.

    Raises:
        ValueError: If labels, averaging, or probability shapes are invalid.
    """
    if average not in _SUPPORTED_AVERAGES:
        raise ValueError(
            f"average must be one of {sorted(_SUPPORTED_AVERAGES)}."
        )
    if multi_class not in {"ovr", "ovo"}:
        raise ValueError("multi_class must be either 'ovr' or 'ovo'.")

    y_true = column_or_1d(y_true)
    y_pred = column_or_1d(y_pred)
    check_consistent_length(y_true, y_pred)

    class_labels = _resolve_class_labels(y_true, y_pred, labels)
    if class_labels.size > 2 and average == "binary":
        raise ValueError(
            "average='binary' is only valid for binary classification."
        )

    effective_pos_label = pos_label
    if class_labels.size <= 2:
        effective_pos_label = (
            class_labels[-1] if pos_label is None else pos_label
        )
        if effective_pos_label not in class_labels:
            raise ValueError(
                f"pos_label={effective_pos_label!r} is not present in labels."
            )

    metric_kwargs = {"average": average, "zero_division": 0}
    if average == "binary":
        metric_kwargs["pos_label"] = effective_pos_label
    else:
        metric_kwargs["labels"] = class_labels

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="A single label was found in 'y_true' and 'y_pred'.*",
            category=UserWarning,
        )
        matrix = confusion_matrix(y_true, y_pred, labels=class_labels)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, **metric_kwargs)),
        "recall": float(recall_score(y_true, y_pred, **metric_kwargs)),
        "f1_score": float(f1_score(y_true, y_pred, **metric_kwargs)),
        "roc_auc": None,
        "brier_score": None,
        "labels": class_labels.copy(),
        "confusion_matrix": matrix,
    }

    if y_pred_proba is None:
        return metrics

    probabilities = np.asarray(y_pred_proba)
    check_consistent_length(y_true, probabilities)

    true_labels = unique_labels(y_true)
    if true_labels.size < 2:
        warnings.warn(
            "ROC AUC is undefined because y_true contains only one class.",
            RuntimeWarning,
            stacklevel=2,
        )
        return metrics

    if class_labels.size <= 2:
        if probabilities.ndim == 1:
            positive_probabilities = probabilities
        elif probabilities.ndim == 2 and probabilities.shape[1] == 2:
            positive_index = np.flatnonzero(
                class_labels == effective_pos_label
            )
            positive_probabilities = probabilities[:, positive_index[0]]
        else:
            raise ValueError(
                "Binary y_pred_proba must be one-dimensional or have "
                "exactly two columns."
            )

        binary_y_true = (y_true == effective_pos_label).astype(int)
        metrics["roc_auc"] = float(
            roc_auc_score(binary_y_true, positive_probabilities)
        )
        metrics["brier_score"] = float(
            brier_score_loss(binary_y_true, positive_probabilities)
        )
        return metrics

    if probabilities.ndim != 2:
        raise ValueError(
            "Multiclass y_pred_proba must be a two-dimensional array."
        )
    if probabilities.shape[1] != class_labels.size:
        raise ValueError(
            "The number of y_pred_proba columns must match the number "
            "of labels."
        )
    if true_labels.size != class_labels.size:
        warnings.warn(
            "ROC AUC is undefined because y_true does not contain every "
            "class listed in labels.",
            RuntimeWarning,
            stacklevel=2,
        )
        return metrics

    metrics["roc_auc"] = float(
        roc_auc_score(
            y_true,
            probabilities,
            labels=class_labels,
            multi_class=multi_class,
            average=average,
        )
    )

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
        if value is None:
            continue
        if np.isscalar(value) and not isinstance(value, (str, bytes)):
            print(f"{key.upper().replace('_', ' ')}: {value:.4f}")
        else:
            print(f"{key.upper().replace('_', ' ')}:")
            print(value)
    print("-" * 25)

def compare_metrics(**metrics_dicts):
    """
    Combines multiple metrics dictionaries into a single pandas DataFrame for easy comparison.
    
    Args:
        **metrics_dicts: Keyword arguments mapping a split name (e.g., 'Train') to its metrics dictionary.
        
    Returns:
        pd.DataFrame: A beautifully formatted dataframe comparing the metrics.
    """
    valid_metrics = {}
    for name, metric_values in metrics_dicts.items():
        if metric_values is None:
            continue
        valid_metrics[name] = {
            key: value
            for key, value in metric_values.items()
            if np.isscalar(value)
            and not isinstance(value, (str, bytes))
        }
    
    df = pd.DataFrame(valid_metrics)
    
    # Clean up index names for better readability
    df.index = [str(idx).upper().replace('_', ' ') for idx in df.index]
    
    return df


def summarize_classification_result(
    result,
    splits=("train", "val", "test"),
    show_cm=False,
):
    """
    Summarize classification metrics stored by train_classifier.

    Binary confusion matrices include TN, FP, FN, and TP counts. Multiclass
    results include the shared metrics without binary-only columns.
    """
    if not isinstance(result, dict):
        raise TypeError("result must be a dictionary.")
    if "model" not in result:
        raise ValueError("result must contain a 'model' entry.")
    if isinstance(splits, str):
        splits = (splits,)

    rows = []
    model_name = type(result["model"]).__name__

    for split in splits:
        metrics = result.get(f"{split}_metrics")
        if metrics is None:
            continue
        if not isinstance(metrics, dict):
            raise TypeError(f"{split}_metrics must be a dictionary.")
        if "confusion_matrix" not in metrics:
            raise ValueError(
                f"{split}_metrics must contain a 'confusion_matrix' entry."
            )

        matrix = np.asarray(metrics["confusion_matrix"])
        if (
            matrix.ndim != 2
            or matrix.shape[0] != matrix.shape[1]
            or matrix.shape[0] == 0
        ):
            raise ValueError(
                f"{split} confusion matrix must be a non-empty square matrix."
            )

        labels = np.asarray(
            metrics.get("labels", np.arange(matrix.shape[0]))
        )
        if labels.ndim != 1 or len(labels) != matrix.shape[0]:
            raise ValueError(
                f"{split} labels must match the confusion matrix size."
            )

        row = {
            "model": model_name,
            "split": split,
            "accuracy": metrics.get("accuracy"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1_score": metrics.get("f1_score"),
            "roc_auc": metrics.get("roc_auc"),
            "brier_score": metrics.get("brier_score"),
        }

        if matrix.shape == (2, 2):
            tn, fp, fn, tp = matrix.ravel()
            row.update({
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "actual_negative": tn + fp,
                "actual_positive": fn + tp,
                "predicted_negative": tn + fn,
                "predicted_positive": fp + tp,
            })

        rows.append(row)

        if show_cm:
            display = ConfusionMatrixDisplay(
                confusion_matrix=matrix,
                display_labels=labels,
            )
            display.plot(values_format="d")
            display.ax_.set_title(
                f"{model_name} - {split} Confusion Matrix"
            )
            plt.show()

    return pd.DataFrame(rows).round(4)

def kaggle_submission(
    model,
    X_test,
    sample_path,
    output_path,
    target_col="target",
    predict_proba=True,
    verbose=False,
):
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

    if len(preds) != len(submission_df):
        raise ValueError(
            f"Predictions length ({len(preds)}) does not match sample submission "
            f"length ({len(submission_df)}). Please check that X_test corresponds to the sample submission."
        )

    # Assign predictions to the target column
    submission_df[target_col] = preds

    # Save submission file
    submission_df.to_csv(output_path, index=False)

    if verbose:
        print(f"Submission file saved to {output_path}")
    return submission_df


def plot_feature_importance(model, feature_names, top_n=20, title="Feature Importances"):
    """
    Extracts and plots feature importances from tree-based models or coefficients from linear models.
    """
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    importances = None

    # Handle pipelines
    if hasattr(model, 'named_steps'):
        step_name = list(model.named_steps.keys())[-1]
        model = model.named_steps[step_name]

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
        importances = importances.flatten()

    if importances is None:
        raise ValueError(
            "Model must provide feature_importances_ or coef_."
        )

    if len(importances) != len(feature_names):
        raise ValueError(
            f"Received {len(feature_names)} feature names for "
            f"{len(importances)} importances."
        )

    df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    df_imp = df_imp.sort_values('Importance', ascending=False).head(top_n)

    plt.figure(figsize=(10, max(5, min(top_n * 0.4, 12))))
    ax = sns.barplot(data=df_imp, x='Importance', y='Feature', palette='viridis', hue='Feature', legend=False)
    plt.title(title)
    plt.tight_layout()
    plt.show()
    return ax


def plot_confusion_matrix(y_true, y_pred, class_names=None, title="Confusion Matrix"):
    """Plots a beautiful confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    if class_names is not None and len(class_names) != cm.shape[0]:
        raise ValueError(
            "class_names length must match the confusion matrix size."
        )
    tick_labels = class_names if class_names is not None else "auto"
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=tick_labels,
        yticklabels=tick_labels,
    )
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(title)
    plt.tight_layout()
    plt.show()
    return ax


def plot_roc_curve(y_true, y_pred_proba, title="ROC Curve"):
    """Plots the Receiver Operating Characteristic (ROC) curve."""
    if len(np.unique(y_true)) > 2:
        raise ValueError("plot_roc_curve supports binary classification only.")

    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
    return plt.gca()
