import pandas as pd
import pytest

from my_krml_24999690.data.loaders import (
    get_experiment_files,
    load_data_at2,
    load_data_at3,
)


def test_load_data_at2_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data_at2(tmp_path / "missing.csv", skiprows=0, sep=",")


def test_load_data_at2_requires_time_column(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"value": [1]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="'time' column"):
        load_data_at2(path, skiprows=0, sep=",")


def test_load_data_at2_is_quiet_by_default(tmp_path, capsys):
    path = tmp_path / "data.csv"
    pd.DataFrame({"time": ["2026-01-01"], "value": [1]}).to_csv(
        path, index=False
    )

    result = load_data_at2(path, skiprows=0, sep=",")

    assert pd.api.types.is_datetime64_any_dtype(result["time"])
    assert capsys.readouterr().out == ""


def test_get_experiment_files_reports_all_missing_datasets(tmp_path):
    experiment_dir = tmp_path / "exp1"
    experiment_dir.mkdir()
    pd.DataFrame({"value": [1]}).to_csv(
        experiment_dir / "exp1_X_class_train.csv",
        index=False,
    )

    with pytest.raises(FileNotFoundError, match="y_class_train"):
        get_experiment_files("exp1", tmp_path)


def test_get_experiment_files_returns_required_order(tmp_path):
    experiment_dir = tmp_path / "exp1"
    experiment_dir.mkdir()
    names = (
        "X_class_train",
        "y_class_train",
        "X_class_val",
        "y_class_val",
        "X_class_test",
        "y_class_test",
        "X_reg_train",
        "y_reg_train",
        "X_reg_val",
        "y_reg_val",
        "X_reg_test",
        "y_reg_test",
    )
    for index, name in enumerate(names):
        pd.DataFrame({"value": [index]}).to_csv(
            experiment_dir / f"exp1_{name}.csv",
            index=False,
        )

    loaded = get_experiment_files("exp1", tmp_path)

    assert [frame.loc[0, "value"] for frame in loaded] == list(range(12))


def test_load_data_at3_sorts_files_and_rows(tmp_path):
    pd.DataFrame({
        "timeOpen": [2, 1],
        "value": ["b", "a"],
    }).to_csv(tmp_path / "b.csv", sep=";", index=False)
    pd.DataFrame({
        "timeOpen": [4, 3],
        "value": ["d", "c"],
    }).to_csv(tmp_path / "a.csv", sep=";", index=False)

    result, file_names = load_data_at3(tmp_path)

    assert file_names == ["a.csv", "b.csv"]
    assert result["value"].tolist() == ["c", "d", "a", "b"]
