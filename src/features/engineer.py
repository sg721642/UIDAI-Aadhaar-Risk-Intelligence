import pandas as pd
from src.config import Config
from src.utils.logger import get_logger

logger = get_logger("feature_engineering", log_file="pipeline.log")

class FeatureEngineer:
    def __init__(self, config: Config):
        self.config = config
        self.rolling_window = config.get("feature_engineering.rolling_window", 7)
        self.min_periods = config.get("feature_engineering.min_periods", 1)

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts behavioral, age transition, and temporal rolling features.
        """
        logger.info("Starting Feature Engineering...")
        
        # Ensure we don't modify the original dataframe in-place
        df = df.copy()
        
        # 1. Totals
        logger.info("Computing Biometric and Demographic totals...")
        df["biometric_total"] = df["bio_age_5_17"] + df["bio_age_17_"]
        df["demographic_total"] = df["demo_age_5_17"] + df["demo_age_17_"]
        
        # 2. Core behavioral ratio
        logger.info("Computing biometric-to-demographic update ratio...")
        df["bio_demo_ratio"] = df["biometric_total"] / (df["demographic_total"] + 1)
        
        # 3. Age transition skew
        logger.info("Computing age transition skew...")
        df["age_transition_skew"] = df["bio_age_5_17"] / (df["bio_age_17_"] + 1)
        
        # 4. Temporal rolling features per pincode
        logger.info(f"Computing {self.rolling_window}-day rolling mean and standard deviation per pincode...")
        
        # Group by pincode and sort by date to compute rolling features correctly
        # Make sure data is sorted by date before rolling
        df = df.sort_values(["pincode", "date"])
        
        df["rolling_mean"] = (
            df.groupby("pincode")["bio_demo_ratio"]
              .transform(lambda x: x.rolling(self.rolling_window, min_periods=self.min_periods).mean())
        )
        
        df["rolling_std"] = (
            df.groupby("pincode")["bio_demo_ratio"]
              .transform(lambda x: x.rolling(self.rolling_window, min_periods=self.min_periods).std().fillna(0))
        )
        
        # Fill any remaining NaNs with 0
        df["rolling_mean"] = df["rolling_mean"].fillna(0)
        df["rolling_std"] = df["rolling_std"].fillna(0)
        df["age_transition_skew"] = df["age_transition_skew"].fillna(0)
        df["bio_demo_ratio"] = df["bio_demo_ratio"].fillna(0)
        
        logger.info("Feature Engineering completed successfully.")
        return df
