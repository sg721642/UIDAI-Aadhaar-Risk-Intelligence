import os
import json
import threading
from fastapi import FastAPI, Query, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from src.config import Config
from src.utils.logger import get_logger
from run_pipeline import run as run_ml_pipeline

logger = get_logger("api", log_file="api.log")

app = FastAPI(
    title="UIDAI Aadhaar Anomaly & Risk Intelligence API",
    description="REST API for regional risk analysis, temporal alert monitoring, and spatial clustering anomalies.",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to cache predictions
_df_cache = None
_summary_cache = None
_cache_lock = threading.Lock()

def generate_mock_data(scores_path, summary_path):
    """Generates a synthetic/mock Aadhaar risk dataset when real pipeline files are absent."""
    logger.info("Pipeline files missing. Generating synthetic/mock data for deployment bootstrap...")
    
    import numpy as np
    from datetime import datetime, timedelta
    
    os.makedirs(os.path.dirname(scores_path), exist_ok=True)
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    
    np.random.seed(42)
    n_records = 1000
    
    states = ["Delhi", "Maharashtra", "Karnataka", "Tamil Nadu", "Uttar Pradesh", "West Bengal"]
    pincodes = [110001, 400001, 560001, 600001, 226001, 700001]
    
    dates = [datetime(2026, 6, 1) + timedelta(days=int(i)) for i in np.random.randint(0, 30, n_records)]
    selected_states = np.random.choice(states, n_records)
    selected_pincodes = [pincodes[states.index(s)] for s in selected_states]
    
    bio_demo_ratio = np.random.normal(0.4, 0.15, n_records)
    bio_demo_ratio = np.clip(bio_demo_ratio, 0.0, 1.5)
    
    anomaly_indices = np.random.choice(n_records, int(n_records * 0.05), replace=False)
    bio_demo_ratio[anomaly_indices] += np.random.uniform(0.5, 1.0, len(anomaly_indices))
    
    rolling_mean = bio_demo_ratio * 0.9
    rolling_std = np.random.uniform(0.01, 0.1, n_records)
    age_transition_skew = np.random.normal(0.0, 0.2, n_records)
    age_transition_skew[anomaly_indices] += np.random.uniform(0.3, 0.8, len(anomaly_indices))
    
    iso_score = np.random.uniform(0.0, 0.4, n_records)
    iso_score[anomaly_indices] = np.random.uniform(0.6, 0.9, len(anomaly_indices))
    
    autoencoder_score = np.random.uniform(0.0, 0.4, n_records)
    autoencoder_score[anomaly_indices] = np.random.uniform(0.6, 0.9, len(anomaly_indices))
    
    lstm_score = np.random.uniform(0.0, 0.3, n_records)
    lstm_score[anomaly_indices] = np.random.uniform(0.5, 0.8, len(anomaly_indices))
    
    spatial_score = np.random.uniform(0.0, 0.3, n_records)
    spatial_score[anomaly_indices] = np.random.uniform(0.5, 0.8, len(anomaly_indices))
    
    final_risk_score = 0.3 * iso_score + 0.3 * autoencoder_score + 0.2 * lstm_score + 0.2 * spatial_score
    
    risk_level = []
    early_warning = []
    explanations = []
    
    for score in final_risk_score:
        if score >= 0.7:
            risk_level.append("High Risk")
            early_warning.append(np.random.choice([0, 1], p=[0.7, 0.3]))
            explanations.append("Elevated biometric-to-demographic update ratio and significant age transition skew.")
        elif score >= 0.4:
            risk_level.append("Monitor")
            early_warning.append(0)
            explanations.append("Moderate variation in updates; normal baseline.")
        else:
            risk_level.append("Normal")
            early_warning.append(0)
            explanations.append("Standard lifecycle biometric activity.")
            
    semantic_contexts = []
    for d in dates:
        if d.month in [4, 5, 6]:
            semantic_contexts.append("School admission period")
        elif d.month in [1, 2, 3]:
            semantic_contexts.append("Biometric refresh cycle")
        elif d.month in [7, 8]:
            semantic_contexts.append("Monsoon / low mobility period")
        else:
            semantic_contexts.append("Normal operational period")

    df = pd.DataFrame({
        "date": dates,
        "state": selected_states,
        "district": [s + " District" for s in selected_states],
        "pincode": selected_pincodes,
        "demo_age_5_17": np.random.randint(10, 50, n_records),
        "demo_age_17_": np.random.randint(50, 200, n_records),
        "bio_age_5_17": np.random.randint(10, 50, n_records),
        "bio_age_17_": np.random.randint(50, 200, n_records),
        "bio_demo_ratio": bio_demo_ratio,
        "rolling_mean": rolling_mean,
        "rolling_std": rolling_std,
        "age_transition_skew": age_transition_skew,
        "iso_score": iso_score,
        "autoencoder_score": autoencoder_score,
        "lstm_score": lstm_score,
        "spatial_score": spatial_score,
        "final_risk_score": final_risk_score,
        "risk_level": risk_level,
        "early_warning": early_warning,
        "explanation": explanations,
        "semantic_context": semantic_contexts
    })
    
    df = df.sort_values("date").reset_index(drop=True)
    df["risk_trend"] = df.groupby("pincode")["final_risk_score"].diff(7).fillna(0.0)
    df.to_csv(scores_path, index=False)
    
    summary = {
        "total_records": int(len(df)),
        "total_alerts": int(len(df[df["risk_level"] == "High Risk"])),
        "risk_distribution": {
            "Normal": int(len(df[df["risk_level"] == "Normal"])),
            "Monitor": int(len(df[df["risk_level"] == "Monitor"])),
            "High Risk": int(len(df[df["risk_level"] == "High Risk"]))
        },
        "early_warning_alerts": int(len(df[df["early_warning"] == 1])),
        "states_monitored": int(df["state"].nunique()),
        "pincodes_monitored": int(df["pincode"].nunique()),
        "average_risk_score": float(df["final_risk_score"].mean())
    }
    
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
        
    logger.info("Synthetic bootstrap data generated successfully.")

def load_cache():
    """Loads prediction outputs into memory cache."""
    global _df_cache, _summary_cache
    config = Config()
    scores_path = config.get_absolute_path("risk_scores_csv")
    summary_path = config.get_absolute_path("summary_json")
    
    with _cache_lock:
        if not os.path.exists(scores_path) or not os.path.exists(summary_path):
            generate_mock_data(scores_path, summary_path)
            
        if os.path.exists(scores_path):
            logger.info(f"Loading risk scores cache from {scores_path}...")
            df = pd.read_csv(scores_path)
            # Ensure proper types
            df["date"] = pd.to_datetime(df["date"])
            df["pincode"] = df["pincode"].astype(int)
            _df_cache = df
            logger.info(f"Loaded {len(_df_cache)} records into memory cache.")
        else:
            logger.warning(f"Risk scores CSV not found at: {scores_path}")
            _df_cache = None
            
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                _summary_cache = json.load(f)
        else:
            logger.warning(f"Summary JSON not found at: {summary_path}")
            _summary_cache = None

# Load cache on module import
load_cache()

@app.get("/api/health")
def health():
    """Health check endpoint."""
    config = Config()
    scores_path = config.get_absolute_path("risk_scores_csv")
    db_loaded = _df_cache is not None
    
    return {
        "status": "healthy",
        "version": "1.0.0",
        "database_file_exists": os.path.exists(scores_path),
        "cache_loaded": db_loaded,
        "total_cached_records": len(_df_cache) if db_loaded else 0
    }

@app.get("/health")
def root_health():
    """Root health check for Render/orchestration systems."""
    return health()

@app.get("/api/data")
def get_full_data():
    """Returns the full cached dataset as a list of records for the dashboard."""
    if _df_cache is None:
        raise HTTPException(
            status_code=404, 
            detail="Risk data not loaded. Run model pipeline first."
        )
    records_df = _df_cache.copy()
    records_df["date"] = records_df["date"].dt.strftime("%Y-%m-%d")
    return records_df.to_dict(orient="records")

@app.get("/api/summary")
def get_summary():
    """Returns summarized system-wide risk metrics."""
    if _summary_cache is None:
        # Generate summary dynamically if file is missing but cache is loaded
        if _df_cache is not None:
            mon_max = Config().get("risk_scoring.thresholds.monitor_max", 0.7)
            actionable_data = _df_cache[
                (_df_cache["risk_level"] == "High Risk") | 
                (_df_cache["early_warning"] == 1)
            ]
            return {
                "total_records": len(_df_cache),
                "total_alerts": len(actionable_data),
                "risk_distribution": _df_cache["risk_level"].value_counts().to_dict(),
                "early_warning_alerts": int(_df_cache["early_warning"].sum()),
                "states_monitored": int(_df_cache["state"].nunique()),
                "pincodes_monitored": int(_df_cache["pincode"].nunique()),
                "average_risk_score": float(_df_cache["final_risk_score"].mean())
            }
        raise HTTPException(
            status_code=404, 
            detail="Risk scoring summary not found. Run model pipeline first."
        )
    return _summary_cache

@app.get("/api/alerts")
def get_alerts(
    state: str = None,
    district: str = None,
    pincode: int = None,
    risk_level: str = None,
    early_warning: int = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500)
):
    """Returns paginated, filterable alert records requiring action."""
    if _df_cache is None:
        raise HTTPException(
            status_code=404, 
            detail="Risk data not loaded. Run model pipeline first."
        )

    # Filter to show only actionable regions (High Risk or Early Warning) unless explicitly querying normal
    query_df = _df_cache
    if risk_level is None:
        # Default to alerts (Monitor or High Risk, or Early Warning)
        query_df = query_df[
            (query_df["risk_level"].isin(["Monitor", "High Risk"])) |
            (query_df["early_warning"] == 1)
        ]
        
    # Apply user filters
    if state:
        query_df = query_df[query_df["state"].str.lower() == state.lower()]
    if district:
        query_df = query_df[query_df["district"].str.lower() == district.lower()]
    if pincode:
        query_df = query_df[query_df["pincode"] == pincode]
    if risk_level:
        query_df = query_df[query_df["risk_level"].str.lower() == risk_level.lower()]
    if early_warning is not None:
        query_df = query_df[query_df["early_warning"] == early_warning]

    # Sort alerts by final risk score descending
    query_df = query_df.sort_values("final_risk_score", ascending=False)
    
    total_alerts = len(query_df)
    
    # Paginate
    start = (page - 1) * page_size
    end = start + page_size
    paginated_df = query_df.iloc[start:end]

    # Convert date to string for JSON serialization
    records = paginated_df.to_dict(orient="records")
    for r in records:
        r["date"] = r["date"].strftime("%Y-%m-%d")

    return {
        "total_alerts": total_alerts,
        "page": page,
        "page_size": page_size,
        "total_pages": (total_alerts + page_size - 1) // page_size,
        "alerts": records
    }

@app.get("/api/region/{pincode}")
def get_region_history(pincode: int):
    """Fetches details and timelines for a specific pincode."""
    if _df_cache is None:
        raise HTTPException(status_code=404, detail="Data not loaded.")
        
    region_df = _df_cache[_df_cache["pincode"] == pincode]
    if region_df.empty:
        raise HTTPException(status_code=404, detail=f"Pincode {pincode} not found.")
        
    # Sort chronologically
    region_df = region_df.sort_values("date")
    
    # Split into history array
    history = region_df.to_dict(orient="records")
    for h in history:
        h["date"] = h["date"].strftime("%Y-%m-%d")
        
    # Get latest status
    latest = history[-1]
    
    return {
        "pincode": pincode,
        "state": latest["state"],
        "district": latest["district"],
        "latest_risk_score": latest["final_risk_score"],
        "latest_risk_level": latest["risk_level"],
        "latest_explanation": latest["explanation"],
        "risk_persistence": latest["risk_persistence"],
        "early_warning": latest["early_warning"],
        "timeline": history
    }

@app.get("/api/states")
def get_states():
    """Gets list of unique states."""
    if _df_cache is None:
        return []
    return sorted(_df_cache["state"].unique().tolist())

@app.get("/api/districts/{state}")
def get_districts(state: str):
    """Gets list of unique districts for a state."""
    if _df_cache is None:
        return []
    subset = _df_cache[_df_cache["state"].str.lower() == state.lower()]
    return sorted(subset["district"].unique().tolist())

@app.get("/api/pincodes/{district}")
def get_pincodes(district: str):
    """Gets list of unique pincodes for a district."""
    if _df_cache is None:
        return []
    subset = _df_cache[_df_cache["district"].str.lower() == district.lower()]
    return sorted(subset["pincode"].unique().tolist())

def run_retrain_task():
    """Worker task to retrain models and reload cache."""
    try:
        logger.info("Background retraining task started...")
        run_ml_pipeline(train_flag=True, predict_flag=True)
        logger.info("Pipeline run complete. Reloading cache...")
        load_cache()
        logger.info("API cache successfully updated.")
    except Exception as e:
        logger.error(f"Error running pipeline in background: {str(e)}")

@app.post("/api/trigger-retrain")
def trigger_retrain(background_tasks: BackgroundTasks):
    """Triggers asynchronous retraining of all model components."""
    background_tasks.add_task(run_retrain_task)
    return {
        "status": "triggered",
        "message": "Model retraining initiated in the background. Cache will be updated upon completion."
    }
