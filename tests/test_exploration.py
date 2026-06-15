import numpy as np
import pandas as pd
import pytest

from my_krml_24999690 import comprehensive_report
from my_krml_24999690.features.preprocessing import drop_columns

def test_drop_columns():
    df = pd.DataFrame({
        "col1": [1, 2],
        "col2": [3, 4],
        "col3": [5, 6]
    })
    
    # Drop single column
    df_out = drop_columns(df.copy(), ["col2"])
    assert "col2" not in df_out.columns
    assert "col1" in df_out.columns
    
    # Drop missing column (should not raise exception, just return original df)
    df_out2 = drop_columns(df.copy(), ["col_missing"])
    assert list(df_out2.columns) == ["col1", "col2", "col3"]


def test_comprehensive_report_returns_structured_results():
    df = pd.DataFrame({
        "amount": [10.0, 20.0, np.nan, 10.0],
        "city": ["Jakarta", "Bandung", None, "Jakarta"],
        "constant": [1, 1, 1, 1],
        "created_at": pd.to_datetime([
            "2026-01-01",
            "2026-01-02",
            "2026-01-03",
            "2026-01-04",
        ]),
    })

    report = comprehensive_report(df, display=False)

    assert report["overview"]["rows"] == 4
    assert report["overview"]["columns"] == 4
    assert report["overview"]["missing_values"] == 2
    assert report["overview"]["duplicate_rows"] == 0
    assert report["column_summary"].loc["city", "missing"] == 1
    assert "amount" in report["numeric_summary"].index
    assert "city" in report["categorical_summary"].index
    assert "created_at" in report["datetime_summary"].index
    assert any(
        "Constant columns" in warning
        for warning in report["quality_warnings"]
    )


def test_comprehensive_report_detects_duplicates_and_limits_sample(capsys):
    df = pd.DataFrame({
        "value": [1, 1, 2],
        "label": ["A", "A", "B"],
    })

    report = comprehensive_report(df, sample_rows=1)
    output = capsys.readouterr().out

    assert report["overview"]["duplicate_rows"] == 1
    assert len(report["sample"]) == 1
    assert "DATAFRAME COMPREHENSIVE REPORT" in output
    assert "QUALITY CHECKS" in output


def test_comprehensive_report_handles_empty_dataframe():
    report = comprehensive_report(
        pd.DataFrame(columns=["value", "label"]),
        display=False,
    )

    assert report["overview"]["rows"] == 0
    assert report["overview"]["missing_pct"] == 0.0
    assert report["numeric_summary"].empty
    assert report["sample"].empty


def test_comprehensive_report_handles_unhashable_object_values():
    df = pd.DataFrame({
        "tags": [["ml", "data"], ["ml", "data"], ["python"]],
    })

    report = comprehensive_report(df, display=False)

    assert report["overview"]["duplicate_rows"] == 1
    assert report["column_summary"].loc["tags", "unique"] == 2


@pytest.mark.parametrize(
    ("kwargs", "error_type"),
    [
        ({"top_n": 0}, ValueError),
        ({"sample_rows": -1}, ValueError),
    ],
)
def test_comprehensive_report_validates_options(kwargs, error_type):
    with pytest.raises(error_type):
        comprehensive_report(pd.DataFrame(), display=False, **kwargs)


def test_comprehensive_report_requires_dataframe():
    with pytest.raises(TypeError, match="DataFrame"):
        comprehensive_report([1, 2, 3], display=False)
