# Model Training & Evaluation Report

This report outlines the finalized training metrics, configurations, and evaluation characteristics of the machine learning models in the **UIDAI Anomaly & Risk Intelligence System**.

---

## 📊 1. Dataset Dimensions
The datasets were fully loaded, cleaned, and merged to form the base feature representation.
* **Full Processed Merged Dataset**: **805,712 records** (post-deduplication of 130,876 duplicate rows).
* **Base Feature Dimension**: 4 features (`bio_demo_ratio`, `rolling_mean`, `rolling_std`, `age_transition_skew`).
* **LSTM Sequences**: Generates up to **780,000 temporal sequence windows** (of length 14 days) when training on the full dataset in Google Colab.

---

## 🛠️ 2. Optimized Pipeline Configuration
To ensure optimal performance and prevent overfitting, the deep learning training pipeline in Keras/TensorFlow utilizes the following mechanisms:
1. **EarlyStopping**: Monitors validation loss and stops training once it ceases to improve for 3 consecutive epochs. Restores best weights automatically.
2. **ReduceLROnPlateau**: Decays learning rate dynamically by a factor of 0.5 if validation loss plateaus for 2 epochs.
3. **ModelCheckpoint**: Monitors validation loss and writes the best model state to disk automatically.
4. **Reproducibility**: Sets static random seeds (`42`) across NumPy, Random, and TensorFlow.

---

## 📈 3. Final Model Performance Metrics

### A. StandardScaler
- **Training size**: 805,712 records
- **Validation**: N/A (Standard feature scaling)
- **File size**: 999 bytes
- **Location**: `models/scaler.joblib`

### B. Isolation Forest
- **Training size**: 805,712 records
- **Parameters**: `n_estimators=200`, `contamination=0.05`, `random_state=42`
- **File size**: 1.62 MB
- **Location**: `models/isolation_forest.joblib`

### C. Dense Autoencoder
- **Training size**: 805,712 records
- **Parameters**: `epochs=20`, `batch_size=256`, `learning_rate=0.001`, `dense_units=[8]`
- **Validation Split**: 20%
- **Validation Loss (Best)**: `~0.1079`
- **File size**: 23.5 KB
- **Location**: `models/autoencoder.keras`
- **History log**: `models/autoencoder_history.csv`

### D. LSTM Network
- **Training size**: 5,000 sequence windows (capped locally) / ~780,000 sequences (Google Colab run)
- **Parameters**: `seq_len=14`, `epochs=2`, `batch_size=128`, `lstm_units=8`
- **Validation Split**: 20%
- **Validation Loss (Best)**: `~0.0049`
- **File size**: 27.6 KB
- **Location**: `models/lstm.keras`
- **History log**: `models/lstm_history.csv`

---

## ☁️ 4. Google Colab Training Integration
Due to local hardware constraints (CPU-only, low memory), full training is decoupled from local operations. Developers must run the training pipeline inside Google Colab:
1. Upload datasets and source code to Google Drive.
2. Launch and run [Colab_Training_Pipeline.ipynb](file:///c:/Users/hp/Dropbox/PC\Documents/UIDAI-PROJECT-main/UIDAI-PROJECT-main/research/Colab_Training_Pipeline.ipynb).
3. The notebook will automatically mount Google Drive, train all components on the full available dataset without limits, and save the finalized checkpoints.
4. Download the trained models and copy them to the local `models/` directory of the repository.
