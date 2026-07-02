# Deployment Guide

The UIDAI Risk Engine is production-ready and can be deployed using Docker, docker-compose, or standard cloud hosting platforms (Render, Railway, Streamlit Cloud).

## 🐋 Docker & docker-compose (Recommended)
Containerization encapsulates the FastAPI backend and Streamlit dashboard.

### 1. Build and Run Container
```bash
docker-compose up --build
```
This launches:
- **FastAPI API Server**: Port 8000
- **Streamlit Dashboard Web App**: Port 8501

### 2. Environment Variables
Configurations can be customized by editing the `.env` file at root level:
- `PORT`: REST API server port (default 8000)
- `HOST`: REST API host interface (default 0.0.0.0)
- `DEBUG`: Enables API debugger (default False)

---

## ☁️ Cloud Platform Configurations

### 1. Streamlit Cloud
1. Push your reconstructed repository to GitHub.
2. Log in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click "New App", select your repository, branch (`main`), and set Main File Path to `app.py`.
4. Deploy!

### 2. Render / Railway (REST API & Dashboard)
You can deploy the API and Streamlit as two separate Web Services on Render or Railway.
- **API Web Service**:
  - Build Command: `pip install -r requirements.txt`
  - Start Command: `uvicorn src.api.app:app --host 0.0.0.0 --port $PORT`
- **Dashboard Web Service**:
  - Build Command: `pip install -r requirements.txt`
  - Start Command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
  - Define Environment Variables: Set `API_URL` to your deployed API server endpoint to retrieve records over REST.
