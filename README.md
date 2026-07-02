# 🛡️ UIDAI Anomaly & Risk Intelligence System (Risk Engine)

An AI-powered, production-grade risk intelligence system designed to analyze Aadhaar demographic and enrolment update patterns. By applying behavioral anomaly detection, temporal intelligence, and spatial clustering, the system identifies regions (pincodes) exhibiting irregularities that may require closer administrative scrutiny.

---

## 📈 Key Capabilities
- **Privacy-Preserving**: Operates strictly on aggregated, anonymized spatial-temporal count data. No Personally Identifiable Information (PII) or individual Aadhaar numbers are stored or processed.
- **Multidimensional Anomaly Detection**: Combines an unsupervised **Isolation Forest** and a **Dense Autoencoder** to flag deviations from expected demographic age skews and biometric-to-demographic update ratios.
- **Temporal Analysis**: Employs an **LSTM recurrent neural network** trained on sequential pincode timeline series to forecast trends and flag sudden, sustained anomalies.
- **Spatial clustering**: Uses **Nearest Neighbors** clustering to measure local density deviations. Surfaces pincodes whose behavior drastically diverges from adjacent geographic regions.
- **Explainable Anomaly Alerts**: Generates natural-language reasoning explanations for every flagged region, detailing why it was classified under the "High Risk" or "Monitor" status.

---

## 🏗️ System Architecture

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
    L -->|Classify Alerts| M[Actionable Alerts CSV]
    L -->|Cache DataFrame| N[FastAPI Backend Server]
    
    N -->|REST Endpoints| O[Streamlit Dashboard Web App]
```

Detailed technical specifications are available in the [docs/](file:///c:/Users/hp/Dropbox/PC/Documents/UIDAI-PROJECT-main/UIDAI-PROJECT-main/docs/) directory:
- [Architecture.md](file:///c:/Users/hp/Dropbox/PC/Documents/UIDAI-PROJECT-main/UIDAI-PROJECT-main/docs/Architecture.md) — Multi-layered workflow description.
- [Dataset.md](file:///c:/Users/hp/Dropbox/PC/Documents/UIDAI-PROJECT-main/UIDAI-PROJECT-main/docs/Dataset.md) — Raw CSV schema layouts and canonical cleaning rules.
- [Models.md](file:///c:/Users/hp/Dropbox/PC/Documents/UIDAI-PROJECT-main/UIDAI-PROJECT-main/docs/Models.md) — ML component details and equations.
- [API.md](file:///c:/Users/hp/Dropbox/PC/Documents/UIDAI-PROJECT-main/UIDAI-PROJECT-main/docs/API.md) — Backend query endpoints reference.
- [Deployment.md](file:///c:/Users/hp/Dropbox/PC/Documents/UIDAI-PROJECT-main/UIDAI-PROJECT-main/docs/Deployment.md) — Docker & Cloud setups.

---

## 🗂️ Project Repository Layout
```
UIDAI-PROJECT-main/
├── config/
│   └── config.yaml               # Central configuration (paths, weights, parameters)
├── data/
│   ├── raw/                      # Links to original raw datasets (unchanged)
│   └── processed/                # Destination for scores and validation logs
├── models/                       # Trained checkpoint models (.joblib, .keras)
├── src/
│   ├── config.py                 # Configuration loader module
│   ├── data/
│   │   └── data_loader.py        # Loading, deduplication, and cleaning pipeline
│   ├── features/
│   │   └── engineer.py           # Feature engineering and rolling statistics
│   ├── models/
│   │   ├── train.py              # Model training orchestration
│   │   └── predict.py            # Risk prediction and alerts engine
│   ├── api/
│   │   └── app.py                # FastAPI REST API server
│   └── utils/
│       └── logger.py             # Rotating file logging setup
├── app.py                        # Streamlit interactive dashboard web app
├── run_pipeline.py               # CLI runner to execute ML pipeline
├── start_services.py             # Script to start API and Dashboard concurrently
├── tests/                        # PyTest test cases
├── Dockerfile                    # Containerization script
└── docker-compose.yml            # Container orchestration config
```

---

## ⚙️ Installation & Usage

For step-by-step setup guides, refer to the [Installation Guide](file:///c:/Users/hp/Dropbox/PC/Documents/UIDAI-PROJECT-main/UIDAI-PROJECT-main/docs/Installation.md).

### 1. Build Virtual Environment & Install Dependencies
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix/macOS
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Data Cleaning & Model Training
Execute the end-to-end pipeline CLI:
```bash
python run_pipeline.py --train
```
This loads raw CSV files, canonicalizes states, extracts rolling features, trains models, saves checkpoints under `models/`, and generates default reports in `data/processed/` and `outputs/`.

### 3. Launch API Backend & Dashboard
Start the concurrent runner script:
```bash
python start_services.py
```
- **FastAPI Endpoints**: Open `http://localhost:8000/docs` to test endpoints.
- **Interactive Dashboard**: Open `http://localhost:8501` to view the UI.

---

## 📊 Risk Classification Scheme

$$FinalRiskScore = 0.3 \times IF + 0.3 \times AE + 0.2 \times LSTM + 0.2 \times Spatial$$

| Risk Level | Range | Meaning & Action |
| --- | --- | --- |
| **Normal** | $< 0.40$ | Regular lifecycle behavior; normal logs. |
| **Monitor** | $0.40 - 0.70$ | Elevated activity; requires passive tracking. |
| **High Risk** | $\geq 0.70$ | Structural discrepancy; flagged for audit inspection. |
| **Early Warning** | flag = 1 | 7-day risk score trend increases by $>0.15$ while currently under the High Risk limit. |

## 🚀 Production Deployment

For production deployments, the backend and frontend are decoupled to scale independently:

1. **Backend REST API (Render / Cloud Docker)**:
   * Deployed via **Docker** runtime.
   * Container exposes **FastAPI** on port `8000`.
   * **Health Check Endpoint**: `https://<your-api-url>/health`
   * **API URL Example**: `https://uidai-risk-api.onrender.com`

2. **Interactive Dashboard (Streamlit Community Cloud)**:
   * Deployed via GitHub connection directly pointing to `app.py`.
   * **Environment Variable**: Set `API_URL` to point to your backend Render URL.
   * **Dashboard URL Example**: `https://uidai-risk-dashboard.streamlit.app`

For detailed production instructions, configuration environment variables, and Docker Compose orchestration, see the [Deployment Guide](docs/Deployment.md).

---

## 🧪 Testing Suite
Ensure all components are validated by running:
```bash
python -m pytest tests/ --tb=short
```

---

## 📝 License
This project is distributed under the terms of the MIT License. Details can be found in the [LICENSE](file:///c:/Users/hp/Dropbox/PC/Documents/UIDAI-PROJECT-main/UIDAI-PROJECT-main/LICENSE) file.
