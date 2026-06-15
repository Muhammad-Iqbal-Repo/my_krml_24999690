import numpy as np
import pytest

from my_krml_24999690.models.performance import (
    compare_metrics,
    evaluate_classification,
    kaggle_submission,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curve,
    print_metrics,
)


def test_evaluate_binary_classification_metrics():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    y_pred_proba = [0.1, 0.7, 0.8, 0.9]

    result = evaluate_classification(
        y_true,
        y_pred,
        y_pred_proba,
        average="binary",
    )

    assert result["accuracy"] == pytest.approx(0.75)
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(1.0)
    assert result["f1_score"] == pytest.approx(0.8)
    assert result["roc_auc"] == pytest.approx(1.0)
    assert result["brier_score"] == pytest.approx(0.1375)
    np.testing.assert_array_equal(result["labels"], [0, 1])
    np.testing.assert_array_equal(
        result["confusion_matrix"],
        [[1, 1], [0, 2]],
    )


def test_evaluate_binary_classification_uses_explicit_label_order():
    y_true = ["yes", "no", "yes", "no"]
    y_pred = ["yes", "no", "no", "no"]
    y_pred_proba = [
        [0.1, 0.9],
        [0.8, 0.2],
        [0.6, 0.4],
        [0.7, 0.3],
    ]

    result = evaluate_classification(
        y_true,
        y_pred,
        y_pred_proba,
        average="binary",
        labels=["no", "yes"],
        pos_label="yes",
    )

    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(0.5)
    assert result["roc_auc"] == pytest.approx(1.0)
    np.testing.assert_array_equal(
        result["confusion_matrix"],
        [[2, 0], [1, 1]],
    )


def test_evaluate_multiclass_classification_metrics():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 2, 1]
    y_pred_proba = [
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.2, 0.7],
        [0.7, 0.2, 0.1],
        [0.1, 0.4, 0.5],
        [0.1, 0.6, 0.3],
    ]

    result = evaluate_classification(
        y_true,
        y_pred,
        y_pred_proba,
        average="macro",
        labels=[0, 1, 2],
    )

    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["roc_auc"] is not None
    assert result["brier_score"] is None
    assert result["confusion_matrix"].shape == (3, 3)


def test_evaluate_classification_warns_when_auc_is_undefined():
    with pytest.warns(RuntimeWarning, match="only one class"):
        result = evaluate_classification(
            [1, 1, 1],
            [1, 1, 1],
            [0.8, 0.9, 0.7],
            average="binary",
        )

    assert result["roc_auc"] is None
    np.testing.assert_array_equal(result["confusion_matrix"], [[3]])


def test_evaluate_classification_rejects_invalid_probability_shape():
    with pytest.raises(ValueError, match="exactly two columns"):
        evaluate_classification(
            [0, 1],
            [0, 1],
            [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]],
            average="binary",
        )


def test_metric_display_helpers_handle_confusion_matrix(capsys):
    result = evaluate_classification([0, 1], [0, 1])

    print_metrics(result)
    output = capsys.readouterr().out
    assert "CONFUSION MATRIX" in output

    comparison = compare_metrics(test=result)
    assert "CONFUSION MATRIX" not in comparison.index
    assert comparison.loc["ACCURACY", "test"] == pytest.approx(1.0)


def test_model_plot_helpers_return_axes(monkeypatch):
    import matplotlib.pyplot as plt
    from sklearn.tree import DecisionTreeClassifier

    monkeypatch.setattr(plt, "show", lambda: None)
    model = DecisionTreeClassifier(max_depth=1, random_state=42)
    model.fit([[0, 0], [1, 1]], [0, 1])

    importance_ax = plot_feature_importance(model, ["a", "b"])
    confusion_ax = plot_confusion_matrix([0, 1], [0, 1])
    roc_ax = plot_roc_curve([0, 1], [0.1, 0.9])

    assert importance_ax is not None
    assert confusion_ax is not None
    assert roc_ax is not None
    plt.close("all")


def test_plot_helpers_raise_for_invalid_inputs():
    class UnsupportedModel:
        pass

    with pytest.raises(ValueError, match="feature_importances"):
        plot_feature_importance(UnsupportedModel(), ["feature"])

    with pytest.raises(ValueError, match="class_names"):
        plot_confusion_matrix([0, 1], [0, 1], class_names=["only_one"])

    with pytest.raises(ValueError, match="binary"):
        plot_roc_curve([0, 1, 2], [0.1, 0.5, 0.9])


def test_kaggle_submission_is_quiet_by_default(tmp_path, capsys):
    class FakeModel:
        def predict(self, X):
            return np.asarray([1, 0])

    sample_path = tmp_path / "sample.csv"
    output_path = tmp_path / "submission.csv"
    np.savetxt(
        sample_path,
        np.asarray([["id", "target"], [1, 0], [2, 0]], dtype=object),
        delimiter=",",
        fmt="%s",
    )

    result = kaggle_submission(
        FakeModel(),
        [[0], [1]],
        sample_path,
        output_path,
        predict_proba=False,
    )

    assert result["target"].tolist() == [1, 0]
    assert output_path.exists()
    assert capsys.readouterr().out == ""
