# System Architecture Specification

The system processes aggregated update data into actionable risk intelligence through a multi-layered decoupled pipeline.

```mermaid
graph TD
    A[Demographic Dataset CSVs] -->|Load & Deduplicate| C[DataLoader]
    B[Enrolment Dataset CSVs] -->|Load & Deduplicate| C
    C -->|Canonicalize States & Align Schema| D[Merged Dataset]
    D -->|Feature Extraction| E[FeatureEngineer]
    E -->|Scale Features| F[StandardScaler]
    F -->|Anomalies fit| G[Isolation Forest]
    F -->|Reconstruction fit| H[Dense Autoencoder]
    E -->|Temporal Windows| I[LSTM Sequencer]
    E -->|Spatial Clustered Ratios| J[Nearest Neighbors]
    
    G -->|Component Scores| K[Risk Fusion Engine]
    H -->|Component Scores| K
    I -->|Component Scores| K
    J -->|Component Scores| K
    
    K -->|Weights Aggregator| L[Final Risk Scores]
    L -->|Classify Alerts| M[Actionable Anomalies CSV]
    L -->|Cache DataFrame| N[FastAPI Backend Server]
    
    N -->|REST Endpoints| O[Streamlit Dashboard Web App]
```

## Architectural Layers

1. **Ingestion & Data Prep Layer**
   - **DataLoader**: Decouples reading and concatenating directory files. Removes duplicate entries, parses dates, canonicalizes 36 states, and maps enrolment to biometric data schemas.

2. **Feature Engineering Layer**
   - **FeatureEngineer**: Computes biometric-demographic ratios, age skews, and rolling statistics.

3. **Machine Learning Layer**
   - Includes **Isolation Forest**, **Dense Autoencoder**, **LSTM network**, and **Nearest Neighbors**. Checkpoints are saved under the `models/` directory for fast startup.

4. **Risk Aggregator & Interpretation Layer**
   - Combines component anomaly scores using weighted equations.
   - Enriches records with natural language explanations, semantic context, risk persistence trackers, risk trends, and simulation deviations.

5. **REST API Service Layer**
   - FastAPI server caching risk data in-memory to expose performant JSON endpoints.

6. **Presentation Layer**
   - Streamlit dashboard loading records dynamically. Incorporates Plotly timelines, density heatmaps, and sliders for real-time recalculations.
