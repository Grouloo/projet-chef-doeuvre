import pytest
import pandas as pd
from app.preprocessing import preprocess_input
from app.schemas import StudentInput


def test_preprocess_input_returns_dataframe_with_correct_columns():
    """Test that preprocess_input returns a DataFrame with the expected columns."""
    # Sample model columns (simplified)
    model_columns = [
        "traveltime", "studytime", "failures", "freetime", "absences", "G1", "G2",
        "source_por", "famsize_LE3", "reason_home", "reason_other", "reason_reputation",
        "guardian_mother", "guardian_other", "schoolsup_yes", "famsup_yes",
        "paid_yes", "activities_yes", "nursery_yes", "higher_yes", "internet_yes"
    ]
    
    input_data = StudentInput(
        source="mat",
        famsize="GT3",
        reason="course",
        guardian="father",
        schoolsup="no",
        famsup="no",
        paid="no",
        activities="yes",
        nursery="yes",
        higher="yes",
        internet="yes",
        traveltime=1,
        studytime=2,
        failures=0,
        freetime=3,
        absences=5,
        G1=12,
        G2=13
    )
    
    result = preprocess_input(input_data, model_columns)
    
    # Assert it's a DataFrame
    assert isinstance(result, pd.DataFrame)
    
    # Assert all expected columns are present
    assert list(result.columns) == model_columns
    
    # Assert encoded values are correct
    assert result["source_por"].iloc[0] == 0  # source was "mat", so source_por = 0
    assert result["activities_yes"].iloc[0] == 0  # activities was "yes"
    assert result["G1"].iloc[0] == 12
    assert result["G2"].iloc[0] == 13


def test_preprocess_input_adds_missing_columns():
    """Test that missing columns are added with value 0."""
    model_columns = ["paid_yes"]
    
    input_data = StudentInput(
        source="por",
        famsize="LE3",
        reason="home",
        guardian="mother",
        schoolsup="yes",
        famsup="yes",
        paid="yes",
        activities="no",
        nursery="no",
        higher="no",
        internet="no",
        traveltime=2,
        studytime=3,
        failures=1,
        freetime=4,
        absences=10,
        G1=8,
        G2=9
    )
    
    result = preprocess_input(input_data, model_columns)
    
    # Assert the nonexistent column was added with value 0
    assert result["paid_yes"].iloc[0] == 0
