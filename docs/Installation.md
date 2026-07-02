# Installation Guide

Follow these steps to set up the UIDAI Anomaly & Risk Intelligence System on your local environment.

## Prerequisites
- Python 3.9, 3.10, or 3.11
- Git

## Step 1: Clone Repository
```bash
git clone https://github.com/<your-username>/UIDAI-PROJECT.git
cd UIDAI-PROJECT
```

## Step 2: Set Up Virtual Environment
Using Python's standard `venv`:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix/macOS
source venv/bin/activate
```

## Step 3: Install Dependencies
```bash
pip install -r requirements.txt
# For development setup (tests, formatting)
pip install -r requirements-dev.txt
```

## Step 4: Run Machine Learning Pipeline
Train models and generate initial processed risk scores:
```bash
python run_pipeline.py --train
```

## Step 5: Start Services
Start both the FastAPI backend and Streamlit dashboard together:
```bash
python start_services.py
```
Or start them individually:
```bash
# Start API Backend (Port 8000)
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

# Start Streamlit Dashboard (Port 8501)
streamlit run app.py
```
