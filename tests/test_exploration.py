import pandas as pd
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
