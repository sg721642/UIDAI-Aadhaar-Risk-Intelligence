# API Documentation

The FastAPI backend server provides REST endpoints to query risk stats, filter alerts, retrieve region history, and trigger asynchronous retraining pipelines.

## Server Details
- **Default Port**: 8000
- **Base URL**: `http://localhost:8000`
- **Swagger Docs**: `http://localhost:8000/docs`

## Endpoints

### 1. Health Check
`GET /api/health`
- **Response**:
  ```json
  {
    "status": "healthy",
    "version": "1.0.0",
    "database_file_exists": true,
    "cache_loaded": true,
    "total_cached_records": 861171
  }
  ```

### 2. Summary Stats
`GET /api/summary`
- **Response**: Returns overall metrics including active alert counts, risk distributions, and average risk scores.

### 3. Actionable Alerts List
`GET /api/alerts`
- **Query Params**:
  - `state` (string, optional)
  - `district` (string, optional)
  - `pincode` (integer, optional)
  - `risk_level` (string, optional - e.g. "Monitor", "High Risk")
  - `early_warning` (integer, optional - 0 or 1)
  - `page` (integer, default 1)
  - `page_size` (integer, default 50)
- **Response**: Paginated list of active anomaly records sorted by risk score descending.

### 4. Pincode Deep-Dive
`GET /api/region/{pincode}`
- **Path Params**: `pincode` (integer)
- **Response**: Detailed historical timeline for the pincode, including scores from individual models (Isolation Forest, LSTM, Spatial, Autoencoder), explanations, and risk persistence trends.

### 5. Spatial Taxonomy Filters
- `GET /api/states`: List of unique states.
- `GET /api/districts/{state}`: List of districts under a state.
- `GET /api/pincodes/{district}`: List of pincodes under a district.

### 6. Asynchronous Retraining
`POST /api/trigger-retrain`
- **Response**: Triggers the machine learning pipeline to run in the background. Reloads the cache automatically once training completes.
