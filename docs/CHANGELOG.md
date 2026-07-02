# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2026-07-02
### Added
- Created professional Python package layout structure (`src/`, `config/`, `models/`, `data/`, `outputs/`, `plots/`, `logs/`, `tests/`, `docs/`, `.github/`).
- Implemented robust `DataLoader` with schema verification, state name canonicalization to 36 states, and duplicate record cleaning.
- Implemented modular `FeatureEngineer` mapping to compute biometric totals, demographic totals, update ratios, skewness, and 7-day rolling window statistics.
- Built Keras-based Autoencoder reconstruction anomaly model and sequential LSTM model.
- Integrated Isolation Forest anomaly classifier and Nearest Neighbors spatial risk calculator.
- Created FastAPI backend REST API with health check, dashboard aggregate summaries, paginated alerts list, and pincode timeline queries.
- Built premium, interactive dark-themed Streamlit dashboard with dynamically adjustable weights, alert filter views, risk timeline breakdowns, and CSV export.
- Set up unit testing using `pytest`.
- Added containerization files (`Dockerfile`, `docker-compose.yml`) and Github actions CI workflow.
