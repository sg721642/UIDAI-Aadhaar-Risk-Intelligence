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

def load_cache():
    """Loads prediction outputs into memory cache."""
    global _df_cache, _summary_cache
    config = Config()
    scores_path = config.get_absolute_path("risk_scores_csv")
    summary_path = config.get_absolute_path("summary_json")
    
    with _cache_lock:
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

# Load cache on startup
@app.on_event("startup")
async def startup_event():
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
