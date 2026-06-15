import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from my_krml_24999690.models.training import (
    cross_validate_model,
    train_classifier,
    train_regressor,
    tune_hyperparameters,
)


def test_train_classifier_returns_train_and_validation_metrics():
    X, y = make_classification(
        n_samples=60,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )

    result = train_classifier(
        LogisticRegression(max_iter=500, solver="liblinear"),
        X[:40],
        y[:40],
        X[40:],
        y[40:],
    )

    assert result["model"].classes_.tolist() == [0, 1]
    assert "confusion_matrix" in result["train_metrics"]
    assert "val_metrics" in result


def test_train_regressor_requires_complete_validation_pair():
    with pytest.raises(ValueError, match="provided together"):
        train_regressor(
            LinearRegression(),
            [[0], [1]],
            [0, 1],
            X_val=[[2]],
        )


def test_cross_validate_model_supports_classification_and_regression():
    X_class, y_class = make_classification(
        n_samples=60,
        n_features=4,
        random_state=42,
    )
    classification = cross_validate_model(
        LogisticRegression(max_iter=500, solver="liblinear"),
        X_class,
        y_class,
        cv_folds=3,
    )

    X_reg, y_reg = make_regression(
        n_samples=60,
        n_features=3,
        random_state=42,
    )
    regression = cross_validate_model(
        LinearRegression(),
        X_reg,
        y_reg,
        task_type="regression",
        cv_folds=3,
    )

    assert "test_accuracy_mean" in classification
    assert "test_roc_auc_mean" in classification
    assert "test_root_mean_squared_error_mean" in regression


def test_cross_validate_model_rejects_invalid_options():
    with pytest.raises(ValueError, match="task_type"):
        cross_validate_model(
            LogisticRegression(),
            [[0], [1]],
            [0, 1],
            task_type="clustering",
        )

    with pytest.raises(ValueError, match="at least 2"):
        cross_validate_model(
            LogisticRegression(),
            [[0], [1]],
            [0, 1],
            cv_folds=1,
        )


def test_tune_hyperparameters_returns_best_estimator(capsys):
    X, y = make_classification(
        n_samples=60,
        n_features=4,
        random_state=42,
    )

    result = tune_hyperparameters(
        DecisionTreeClassifier(random_state=42),
        {"max_depth": [1, 2]},
        X,
        y,
        search_type="grid",
        cv=3,
    )

    assert result.max_depth in {1, 2}
    assert capsys.readouterr().out == ""
