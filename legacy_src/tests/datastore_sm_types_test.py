import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date
from datastore import SheetManager

@pytest.fixture
def messy_data():
    """
    Creates a dictionary with mixed types and missing values 
    to test normalization and export logic.
    """
    return {
        "ids": [1, 2, 3, None],            # Nullable Integers
        "names": ["Alice", "Bob", None, "Dave"], # Strings with Nulls
        "dates": [
            pd.Timestamp("2023-01-01"),
            pd.Timestamp("2023-01-02"),
            pd.NaT,                       # Missing Date
            pd.Timestamp("2023-01-04")
        ],
        "amounts": [10.5, np.nan, 20.0, 30.5] # Floats with NaN
    }

def test_internal_type_normalization(messy_data):
    """
    Test that SheetManager.__init__ converts 'object' columns to 
    strict Pandas types (String, Int64, etc) where possible.
    """
    # Initialize
    sm = SheetManager("Test", messy_data)
    df = sm.get_df()

    # 1. Check ID column (Should be nullable Int64, not float or object)
    # Note: 'convert_dtypes()' usually handles this if input allows
    assert pd.api.types.is_numeric_dtype(df["ids"])

    # 2. Check String column
    # Should ideally be 'string' dtype or object, but definitely not mixed
    assert df["names"].iloc[0] == "Alice"
    assert pd.isna(df["names"].iloc[2]) # Should remain standard NaN/None

def test_export_date_conversion(messy_data):
    """
    Test that to_excel_compatible_df() converts Pandas Timestamps 
    to Python datetime objects.
    """
    sm = SheetManager("Test", messy_data)
    
    # Act: Get the export version
    export_df = sm.to_excel_compatible_df()
    
    dates = export_df["dates"]
    
    # 1. Check valid date
    first_date = dates[0]
    assert isinstance(first_date, datetime), \
        f"Expected datetime.datetime, got {type(first_date)}"
    
    # 2. Check NaT conversion
    missing_date = dates[2]
    assert missing_date is None, \
        f"Expected None for NaT, got {missing_date}"

def test_export_null_handling(messy_data):
    """
    Test that to_excel_compatible_df() converts all NaN/NaT variants 
    to simple Python 'None'.
    """
    sm = SheetManager("Test", messy_data)
    export_df = sm.to_excel_compatible_df()
    
    # Check Integer None
    assert export_df["ids"][3] is None
    
    # Check String None
    assert export_df["names"][2] is None
    
    # Check Float NaN
    assert export_df["amounts"][1] is None

def test_mixed_numeric_conversion():
    """Test that mixed string/int input is standardized."""
    data = {
        "mixed": [1, "2", 3, "4"] # Common issue in raw inputs
    }
    
    # Note: DocumentManager defaults to converting this to object.
    # If we want numeric, we usually need explicit processing, 
    # but let's check what convert_dtypes does.
    sm = SheetManager("Mixed", data)
    df = sm.get_df()
    
    # In standard pandas, this stays object/string unless explicitly cast.
    # This test confirms that our system doesn't crash on it.
    assert len(df) == 4
    
    # Check Export safety
    export_df = sm.to_excel_compatible_df()
    assert export_df["mixed"][1] == "2" # Should persist as valid data