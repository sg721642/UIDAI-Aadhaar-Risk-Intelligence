# Machine Learning Models

The UIDAI Risk Engine employs four machine learning and mathematical models to compute component anomaly scores, which are then fused into a single risk score.

## 1. Isolation Forest (Behavioral Engine)
- **Library**: `scikit-learn`
- **Objective**: Identifies multi-dimensional out-of-distribution updates.
- **Inputs**: Scaled features (`bio_demo_ratio`, `rolling_mean`, `rolling_std`, `age_transition_skew`).
- **Score Mapping**: `(1.0 - predictions) / 2.0` (maps anomalies to 1 and normal records to 0).

## 2. Dense Autoencoder (Behavioral Skew Engine)
- **Library**: `Keras / TensorFlow`
- **Architecture**: A fully connected feed-forward autoencoder (`Input(4) -> Dense(8, relu) -> Dense(4, linear)`).
- **Objective**: Learns dependencies between features and flags structural discrepancies (e.g. unexpected age transitions).
- **Score Mapping**: Mean squared reconstruction error, min-max normalized.

## 3. LSTM Network (Temporal Trend Engine)
- **Library**: `Keras / TensorFlow`
- **Architecture**: Sequential LSTM network (`Input(14, 1) -> LSTM(8) -> Dense(1)`).
- **Objective**: Fits on sequence histories of length 14 days for `bio_demo_ratio` per pincode. Predicts the next ratio and compares the prediction with the actual value.
- **Score Mapping**: Absolute error of prediction vs actual value, min-max normalized.

## 4. Nearest Neighbors (Spatial Clustering Engine)
- **Library**: `scikit-learn`
- **Objective**: Evaluates geographical clusters of updates.
- **Algorithm**: For each date, groups records and fits a 1D `NearestNeighbors` model on `bio_demo_ratio` (k=5). Computes the absolute deviation of a pincode's ratio from the local neighborhood mean.
- **Score Mapping**: Absolute deviation, min-max normalized.

## 5. Unified Risk Fusion
The components are aggregated using a weighted average:
$$RiskScore = w_1 \times IF + w_2 \times AE + w_3 \times LSTM + w_4 \times Spatial$$
Defaults weights are `0.3, 0.3, 0.2, 0.2` respectively.
Risk classification levels:
- **Normal**: $<0.4$
- **Monitor**: $0.4 - 0.7$
- **High Risk**: $\geq0.7$
- **Early Warning**: Assigned if risk trend rises by $>0.15$ over 7 days while the current risk remains under the High Risk threshold.
