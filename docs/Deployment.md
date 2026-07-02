# Production Deployment Guide

This guide details the steps to deploy the **UIDAI Anomaly & Risk Intelligence System** in a production environment using best practices. The backend API server and frontend dashboard are split to enable independent scaling and isolation.

---

## 🏗️ 1. Backend REST API Deployment (Render / Cloud Docker)

The backend is packaged inside a multi-stage Docker container running **FastAPI** with **Uvicorn**.

### Deployment Steps on Render:
1. Push the repository to GitHub.
2. Sign in to [Render](https://render.com/).
3. Click **New +** and select **Web Service**.
4. Connect your GitHub repository.
5. In the settings, configure:
   * **Runtime**: `Docker`
   * **Dockerfile Path**: `Dockerfile` (automatically uses backend-only multi-stage build)
   * **Instance Type**: Free/Paid (as appropriate)
6. Render will automatically build the Docker image and pass a dynamic `PORT` environment variable to the container.
7. Set up the **Health Check Path** in the Advanced Settings:
   * **Path**: `/health` or `/api/health`
   * **Interval**: 10-30 seconds
8. **Verify Deployed URL**: The service will expose an endpoint like:
   `https://uidai-risk-api.onrender.com`

---

## 🎨 2. Dashboard Web UI Deployment (Streamlit Community Cloud)

The interactive dashboard runs on **Streamlit Community Cloud** and communicates with the backend API over HTTP.

### Deployment Steps on Streamlit Cloud:
1. Log in to [Streamlit Share](https://share.streamlit.io/).
2. Click **New App**.
3. Choose your GitHub repository, branch (`main`), and set the main file path to:
   `app.py`
4. Expand the **Advanced Settings** before deploying.
5. Under **Secrets**, define the environment variable to point to your backend API hosted on Render:
   ```toml
   API_URL = "https://uidai-risk-api.onrender.com"
   ```
6. Click **Deploy**.
7. **Verify Deployed URL**: The service will expose an app like:
   `https://uidai-risk-dashboard.streamlit.app`

---

## 🎛️ 3. Environment Variables Summary

| Variable Name | Required By | Description | Example Value |
| --- | --- | --- | --- |
| `PORT` | FastAPI (Render) | Port for the backend API container (Render injects this dynamically). | `8000` |
| `API_URL` | Streamlit Cloud | URL of the running FastAPI server. Streamlit uses this to fetch data. If empty, falls back to local CSV. | `https://uidai-risk-api.onrender.com` |

---

## 🐋 4. Local Orchestration (Docker Compose)

For local development, you can spin up the backend API using Docker Compose:

```bash
docker-compose up --build
```
This builds and launches the API container on `http://localhost:8000`. You can run the Streamlit dashboard locally pointing to it:
```bash
$env:API_URL="http://localhost:8000"
streamlit run app.py
```
Or simply run both locally using the orchestrator script:
```bash
python start_services.py
```
