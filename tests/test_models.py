import pytest
import pandas as pd
from src.config import Config
from src.models.predict import RiskPredictor

def test_assign_semantic_context():
    config = Config()
    predictor = RiskPredictor(config)
    
    # Test rules
    d1 = pd.to_datetime("2025-04-15")  # April -> School admission period
    d2 = pd.to_datetime("2025-02-10")  # Feb -> Biometric refresh cycle
    d3 = pd.to_datetime("2025-07-20")  # July -> Monsoon / low mobility period
    d4 = pd.to_datetime("2025-10-05")  # Oct -> Normal operational period
    
    assert predictor.assign_semantic_context(d1) == "School admission period"
    assert predictor.assign_semantic_context(d2) == "Biometric refresh cycle"
    assert predictor.assign_semantic_context(d3) == "Monsoon / low mobility period"
    assert predictor.assign_semantic_context(d4) == "Normal operational period"

def test_generate_explanation():
    config = Config()
    predictor = RiskPredictor(config)
    
    row_normal = {
        "bio_demo_ratio": 0.5,
        "lstm_score": 0.2,
        "spatial_score": 0.1,
        "iso_score": 0.3
    }
    assert predictor.generate_explanation(row_normal) == "normal lifecycle behavior"
    
    row_anomaly = {
        "bio_demo_ratio": 4.2,  # > 3 -> high bio_demo_ratio
        "lstm_score": 0.8,      # > 0.7 -> sustained abnormal
        "spatial_score": 0.2,
        "iso_score": 0.9        # > 0.7 -> deviation from normal lifecycle
    }
    explanation = predictor.generate_explanation(row_anomaly)
    assert "unusually high biometric-to-demographic update ratio" in explanation
    assert "sustained abnormal behavior over time" in explanation
    assert "deviation from normal Aadhaar update lifecycle" in explanation
