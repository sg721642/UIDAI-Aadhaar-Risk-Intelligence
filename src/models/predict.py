import os
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
from src.config import Config
from src.utils.logger import get_logger

logger = get_logger("model_prediction", log_file="pipeline.log")

class RiskPredictor:
    def __init__(self, config: Config):
        self.config = config
        self.scaler_path = config.get_absolute_path("scaler_joblib")
        self.iforest_path = config.get_absolute_path("isolation_forest_joblib")
        self.autoencoder_path = config.get_absolute_path("autoencoder_keras")
        self.lstm_path = config.get_absolute_path("lstm_keras")
        
        self.feature_cols = [
            "bio_demo_ratio",
            "rolling_mean",
            "rolling_std",
            "age_transition_skew"
        ]
        
        self.weights = config.get("risk_scoring.weights")
        self.thresholds = config.get("risk_scoring.thresholds")

    def load_models(self):
        """Loads all model checkpoints from disk."""
        logger.info("Loading trained models for inference...")
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler checkpoint missing: {self.scaler_path}")
        if not os.path.exists(self.iforest_path):
            raise FileNotFoundError(f"Isolation Forest checkpoint missing: {self.iforest_path}")
        
        scaler = joblib.load(self.scaler_path)
        iforest = joblib.load(self.iforest_path)
        
        autoencoder = None
        if os.path.exists(self.autoencoder_path):
            autoencoder = tf.keras.models.load_model(self.autoencoder_path)
            logger.info("Autoencoder model loaded.")
        else:
            logger.warning("Autoencoder checkpoint missing. Autoencoder scores will default to 0.")
            
        lstm = None
        if os.path.exists(self.lstm_path):
            lstm = tf.keras.models.load_model(self.lstm_path)
            logger.info("LSTM model loaded.")
        else:
            logger.warning("LSTM checkpoint missing. LSTM scores will default to 0.")
            
        return scaler, iforest, autoencoder, lstm

    def compute_spatial_scores(self, df: pd.DataFrame) -> pd.Series:
        """Computes spatial neighborhood deviation scores grouped by date."""
        logger.info("Computing spatial neighborhood anomaly scores...")
        spatial_scores = pd.Series(0.0, index=df.index)
        n_neighbors = self.config.get("models.spatial.n_neighbors", 5)

        for date, subset in df.groupby("date"):
            if len(subset) < n_neighbors:
                continue
            
            values = subset[["bio_demo_ratio"]].values
            nn = NearestNeighbors(n_neighbors=n_neighbors).fit(values)
            indices = nn.kneighbors(return_distance=False)
            
            local_means = np.array([values[idx].mean() for idx in indices])
            spatial_scores.loc[subset.index] = np.abs(
                subset["bio_demo_ratio"].values - local_means
            )
            
        # Min-max normalize
        s_min, s_max = spatial_scores.min(), spatial_scores.max()
        if s_max - s_min > 1e-6:
            spatial_scores = (spatial_scores - s_min) / (s_max - s_min)
        else:
            spatial_scores = pd.Series(0.0, index=df.index)
            
        return spatial_scores

    def compute_lstm_scores(self, df: pd.DataFrame, lstm_model) -> pd.Series:
        """Computes sequential temporal deviation scores per pincode using LSTM."""
        logger.info("Computing temporal sequential anomaly scores using LSTM...")
        lstm_scores = pd.Series(0.0, index=df.index)
        
        if lstm_model is None:
            return lstm_scores

        seq_len = self.config.get("models.lstm.sequence_length", 14)
        
        X_seq = []
        index_map = []
        y_seq = []

        for pincode, group in df.groupby("pincode"):
            series = group["bio_demo_ratio"].values.astype("float32")
            if len(series) <= seq_len:
                continue
                
            for i in range(len(series) - seq_len):
                X_seq.append(series[i:i + seq_len])
                y_seq.append(series[i + seq_len])
                index_map.append(group.index[i + seq_len])

        if not X_seq:
            return lstm_scores

        X_seq = np.array(X_seq).reshape(-1, seq_len, 1)
        y_seq = np.array(y_seq).reshape(-1, 1)
        
        # Predict
        preds = lstm_model.predict(X_seq, verbose=0)
        errors = np.abs(preds.flatten() - y_seq.flatten())
        
        # Map back to original dataframe indices
        for idx, err in zip(index_map, errors):
            lstm_scores.loc[idx] = err
            
        # Min-max normalize
        l_min, l_max = lstm_scores.min(), lstm_scores.max()
        if l_max - l_min > 1e-6:
            lstm_scores = (lstm_scores - l_min) / (l_max - l_min)
        else:
            lstm_scores = pd.Series(0.0, index=df.index)
            
        return lstm_scores

    def assign_semantic_context(self, date):
        """Rules engine to tag operational/societal context based on date."""
        if date.month in [4, 5, 6]:
            return "School admission period"
        elif date.month in [1, 2, 3]:
            return "Biometric refresh cycle"
        elif date.month in [7, 8]:
            return "Monsoon / low mobility period"
        else:
            return "Normal operational period"

    def generate_explanation(self, row):
        """Generates natural language explanations of anomalies."""
        reasons = []
        if row["bio_demo_ratio"] > 3:
            reasons.append("unusually high biometric-to-demographic update ratio")
        if row["lstm_score"] > 0.7:
            reasons.append("sustained abnormal behavior over time")
        if row["spatial_score"] > 0.6:
            reasons.append("behavior isolated from neighboring regions")
        if row["iso_score"] > 0.7:
            reasons.append("deviation from normal Aadhaar update lifecycle")
            
        if not reasons:
            return "normal lifecycle behavior"
        return "; ".join(reasons)

    def predict_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Orchestrates prediction, risk fusion, alert generation, and enrichment.
        """
        logger.info("Executing risk prediction pipeline...")
        df = df.copy()
        
        # 1. Load models
        scaler, iforest, autoencoder, lstm = self.load_models()
        
        # 2. Scale features
        X = df[self.feature_cols].fillna(0)
        X_scaled = scaler.transform(X)
        
        # 3. Isolation Forest score
        logger.info("Computing Isolation Forest anomaly scores...")
        predictions = iforest.predict(X_scaled)
        df["iso_score"] = (1.0 - predictions) / 2.0  # map 1 (normal) -> 0, -1 (anomaly) -> 1
        
        # 4. Autoencoder score
        if autoencoder is not None:
            logger.info("Computing Autoencoder reconstruction anomaly scores...")
            reconstructed = autoencoder.predict(X_scaled)
            ae_loss = np.mean(np.square(X_scaled - reconstructed), axis=1)
            ae_min, ae_max = ae_loss.min(), ae_loss.max()
            if ae_max - ae_min > 1e-6:
                df["autoencoder_score"] = (ae_loss - ae_min) / (ae_max - ae_min)
            else:
                df["autoencoder_score"] = 0.0
        else:
            df["autoencoder_score"] = 0.0
            
        # 5. LSTM score
        df["lstm_score"] = self.compute_lstm_scores(df, lstm)
        
        # 6. Spatial score
        df["spatial_score"] = self.compute_spatial_scores(df)
        
        # 7. Risk Score Fusion
        logger.info("Fusing component anomaly scores into a final risk score...")
        w_iforest = self.weights.get("isolation_forest", 0.3)
        w_ae = self.weights.get("autoencoder", 0.3)
        w_lstm = self.weights.get("lstm", 0.2)
        w_spatial = self.weights.get("spatial", 0.2)
        
        df["final_risk_score"] = (
            w_iforest * df["iso_score"] +
            w_ae * df["autoencoder_score"] +
            w_lstm * df["lstm_score"] +
            w_spatial * df["spatial_score"]
        )
        
        # 8. Alert levels classification
        norm_max = self.thresholds.get("normal_max", 0.4)
        mon_max = self.thresholds.get("monitor_max", 0.7)
        
        def classify_risk(score):
            if score < norm_max:
                return "Normal"
            elif score < mon_max:
                return "Monitor"
            else:
                return "High Risk"
                
        df["risk_level"] = df["final_risk_score"].apply(classify_risk)
        
        # 9. Natural Language Explanations
        logger.info("Generating alert explanations...")
        df["explanation"] = df.apply(self.generate_explanation, axis=1)
        
        # 10. Semantic Context mapping
        df["semantic_context"] = df["date"].apply(self.assign_semantic_context)
        
        # 11. Alerts Persistence, Trends and Early Warnings
        logger.info("Computing risk trends, persistence and early warnings...")
        df["high_risk_flag"] = (df["risk_level"] == "High Risk").astype(int)
        
        df["risk_persistence"] = (
            df.groupby("pincode")["high_risk_flag"]
              .transform(lambda x: x.rolling(14, min_periods=1).sum())
        )
        
        df["risk_trend"] = df.groupby("pincode")["final_risk_score"].diff(7).fillna(0.0)
        
        df["early_warning"] = (
            (df["risk_trend"] > 0.15) & (df["final_risk_score"] < mon_max)
        ).astype(int)
        
        # 12. Simulation deviation (Monte Carlo similarity dev)
        # Permutes ratios per pincode to establish a baseline deviation metric
        df["simulated_ratio"] = (
            df.groupby("pincode")["bio_demo_ratio"]
              .transform(lambda x: np.random.permutation(x.values) if len(x) > 1 else x.values)
        )
        df["simulation_deviation"] = np.abs(df["bio_demo_ratio"] - df["simulated_ratio"])
        
        logger.info("Risk prediction pipeline execution completed.")
        return df
