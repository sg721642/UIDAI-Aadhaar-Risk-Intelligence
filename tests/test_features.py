import pytest
import pandas as pd
import numpy as np
from src.config import Config
from src.features.engineer import FeatureEngineer

def test_feature_engineering():
    # Setup dummy data
    data = {
        "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
        "pincode": [110001, 110001, 110001],
        "bio_age_5_17": [10, 20, 30],
        "bio_age_17_": [90, 80, 70],
        "demo_age_5_17": [5, 10, 15],
        "demo_age_17_": [95, 90, 85]
    }
    df = pd.DataFrame(data)
    
    config = Config()
    engineer = FeatureEngineer(config)
    output = engineer.engineer_features(df)
    
    # Assert columns
    assert "biometric_total" in output.columns
    assert "demographic_total" in output.columns
    assert "bio_demo_ratio" in output.columns
    assert "age_transition_skew" in output.columns
    assert "rolling_mean" in output.columns
    assert "rolling_std" in output.columns
    
    # Assert values
    # biometric_total = 10 + 90 = 100
    # demographic_total = 5 + 95 = 100
    assert output.loc[0, "biometric_total"] == 100
    assert output.loc[0, "demographic_total"] == 100
    # bio_demo_ratio = 100 / (100 + 1) = 100 / 101 approx 0.99
    assert abs(output.loc[0, "bio_demo_ratio"] - 100/101) < 1e-5
    # age_transition_skew = 10 / (90 + 1) = 10 / 91
    assert abs(output.loc[0, "age_transition_skew"] - 10/91) < 1e-5
