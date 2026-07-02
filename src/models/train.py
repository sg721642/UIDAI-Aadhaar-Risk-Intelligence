import os
import random
import joblib
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, LSTM, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from src.config import Config
from src.utils.logger import get_logger

logger = get_logger("model_training", log_file="pipeline.log")

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

class ModelTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.models_dir = config.get_absolute_path("models_dir")
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Paths to save checkpoints
        self.scaler_path = config.get_absolute_path("scaler_joblib")
        self.iforest_path = config.get_absolute_path("isolation_forest_joblib")
        self.autoencoder_path = config.get_absolute_path("autoencoder_keras")
        self.lstm_path = config.get_absolute_path("lstm_keras")
        self.metadata_path = os.path.join(self.models_dir, "metadata.json")
        
        self.feature_cols = [
            "bio_demo_ratio",
            "rolling_mean",
            "rolling_std",
            "age_transition_skew"
        ]

    def prepare_data(self, df: pd.DataFrame):
        """Standardizes engineered features."""
        logger.info("Scaling features for Isolation Forest and Autoencoder...")
        X = df[self.feature_cols].fillna(0)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Save scaler
        joblib.dump(scaler, self.scaler_path)
        logger.info(f"Saved StandardScaler checkpoint to: {self.scaler_path}")
        
        return X_scaled, scaler

    def train_isolation_forest(self, X_scaled):
        """Trains Isolation Forest anomaly detector."""
        params = self.config.get("models.isolation_forest")
        logger.info(f"Training Isolation Forest model with params: {params}...")
        
        iso_model = IsolationForest(
            n_estimators=params.get("n_estimators", 200),
            contamination=params.get("contamination", 0.05),
            random_state=params.get("random_state", 42),
            n_jobs=-1
        )
        
        iso_model.fit(X_scaled)
        
        # Save model
        joblib.dump(iso_model, self.iforest_path)
        logger.info(f"Saved Isolation Forest checkpoint to: {self.iforest_path}")
        return iso_model

    def train_autoencoder(self, X_scaled):
        """Trains reconstruction autoencoder with validation split and optimization callbacks."""
        params = self.config.get("models.autoencoder")
        logger.info(f"Training Autoencoder with params: {params}...")
        
        input_dim = X_scaled.shape[1]
        input_layer = Input(shape=(input_dim,))
        
        # Encode
        encoded = input_layer
        for units in params.get("dense_units", [8]):
            encoded = Dense(units, activation=params.get("activation", "relu"))(encoded)
            
        # Decode
        decoded = Dense(input_dim, activation=params.get("output_activation", "linear"))(encoded)
        
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(
            optimizer=params.get("optimizer", "adam"),
            loss="mse"
        )
        
        # Setup production callbacks
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
            ModelCheckpoint(filepath=self.autoencoder_path, monitor="val_loss", save_best_only=True, verbose=1)
        ]
        
        history = autoencoder.fit(
            X_scaled,
            X_scaled,
            epochs=params.get("epochs", 20),
            batch_size=params.get("batch_size", 256),
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save training history
        hist_df = pd.DataFrame(history.history)
        hist_path = os.path.join(self.models_dir, "autoencoder_history.csv")
        hist_df.to_csv(hist_path, index=False)
        logger.info(f"Saved Autoencoder training history to: {hist_path}")
        
        return autoencoder

    def train_lstm(self, df: pd.DataFrame):
        """Trains sequential LSTM trend predictor with callbacks."""
        params = self.config.get("models.lstm")
        seq_len = params.get("sequence_length", 14)
        max_seqs = params.get("max_sequences")
        
        logger.info(f"Preparing sequential dataset for LSTM (seq_len={seq_len})...")
        X_seq = []
        y_seq = []
        
        for pincode, group in df.groupby("pincode"):
            series = group["bio_demo_ratio"].values.astype("float32")
            if len(series) <= seq_len:
                continue
                
            for i in range(len(series) - seq_len):
                X_seq.append(series[i:i + seq_len])
                y_seq.append(series[i + seq_len])
                
                # Check for max_sequences cap (only apply if max_seqs is specified and > 0)
                if max_seqs and max_seqs > 0 and len(X_seq) >= max_seqs:
                    break
            if max_seqs and max_seqs > 0 and len(X_seq) >= max_seqs:
                break
                
        if not X_seq:
            logger.warning("No sequences could be constructed. Skipping LSTM training.")
            return None
            
        X_seq = np.array(X_seq).reshape(-1, seq_len, 1)
        y_seq = np.array(y_seq).reshape(-1, 1)
        
        logger.info(f"LSTM training dataset size: {X_seq.shape}")
        
        lstm_model = Sequential([
            Input(shape=(seq_len, 1)),
            LSTM(params.get("units", 8)),
            Dense(1)
        ])
        
        lstm_model.compile(optimizer="adam", loss="mse")
        
        # Setup production callbacks
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
            ModelCheckpoint(filepath=self.lstm_path, monitor="val_loss", save_best_only=True, verbose=1)
        ]
        
        history = lstm_model.fit(
            X_seq,
            y_seq,
            epochs=params.get("epochs", 2),
            batch_size=params.get("batch_size", 128),
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save training history
        hist_df = pd.DataFrame(history.history)
        hist_path = os.path.join(self.models_dir, "lstm_history.csv")
        hist_df.to_csv(hist_path, index=False)
        logger.info(f"Saved LSTM training history to: {hist_path}")
        
        return lstm_model

    def save_metadata(self, dataset_size, lstm_size):
        """Saves final training configuration metadata and model summary parameters."""
        metadata = {
            "dataset_size_records": int(dataset_size),
            "lstm_dataset_sequences": int(lstm_size),
            "features_used": self.feature_cols,
            "weights": self.config.get("risk_scoring.weights"),
            "thresholds": self.config.get("risk_scoring.thresholds"),
            "tensorflow_version": tf.__version__
        }
        
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Saved model training metadata to: {self.metadata_path}")
        
        # Write duplicates to processed/ for downstream apps
        processed_dir = self.config.get_absolute_path("processed_dir")
        with open(os.path.join(processed_dir, "model_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

    def train_all(self, df: pd.DataFrame):
        """Orchestrates the entire training cycle."""
        logger.info("Executing training cycle for all model components...")
        
        # Scaling
        X_scaled, _ = self.prepare_data(df)
        
        # Component 1: Isolation Forest
        self.train_isolation_forest(X_scaled)
        
        # Component 2: Autoencoder
        self.train_autoencoder(X_scaled)
        
        # Component 3: LSTM
        lstm_model = self.train_lstm(df)
        
        # Calculate sizes and save metadata
        dataset_size = len(df)
        lstm_size = 0
        if lstm_model is not None:
            # Reconstruct sequence counts safely
            params = self.config.get("models.lstm")
            seq_len = params.get("sequence_length", 14)
            max_seqs = params.get("max_sequences")
            seqs_count = 0
            for pincode, group in df.groupby("pincode"):
                series = group["bio_demo_ratio"].values
                if len(series) > seq_len:
                    seqs_count += len(series) - seq_len
                    if max_seqs and max_seqs > 0 and seqs_count >= max_seqs:
                        seqs_count = max_seqs
                        break
            lstm_size = seqs_count
            
        self.save_metadata(dataset_size, lstm_size)
        logger.info("All model components trained and checkpointed successfully.")
