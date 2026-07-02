import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from src.config import Config
from src.utils.logger import get_logger

logger = get_logger("dashboard", log_file="dashboard.log")

# Page Configuration
st.set_page_config(
    page_title="UIDAI Risk Intelligence & Anomaly Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Dark Mode Theme Injection
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Outfit:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .sub-title {
        font-size: 1.1rem;
        color: #94a3b8;
        margin-bottom: 2rem;
    }
    
    .kpi-card {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-5px);
        border-color: rgba(99, 102, 241, 0.4);
    }
    
    .kpi-value {
        font-family: 'Outfit', sans-serif;
        font-size: 2.4rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 5px 0;
    }
    
    .kpi-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .alert-high {
        border-left: 5px solid #ef4444 !important;
    }
    .alert-monitor {
        border-left: 5px solid #f59e0b !important;
    }
    .alert-normal {
        border-left: 5px solid #10b981 !important;
    }
    
    /* Streamlit Customizations */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0 0;
        font-weight: 600;
        font-size: 1rem;
    }
    
    .stTabs [aria-selected="true"] {
        color: #6366f1 !important;
        border-bottom-color: #6366f1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

import requests

@st.cache_data
def load_data(file_path):
    """Cached data loader with local fallback and API support."""
    # Support both Streamlit Cloud secrets and environment variables
    api_url = None
    try:
        api_url = st.secrets.get("API_URL") or os.getenv("API_URL")
    except Exception:
        api_url = os.getenv("API_URL")
    if api_url:
        api_url = api_url.rstrip("/")
        try:
            logger.info(f"Attempting to fetch dataset from remote API: {api_url}/api/data")
            response = requests.get(f"{api_url}/api/data", timeout=15)
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data)
                df["date"] = pd.to_datetime(df["date"])
                df["pincode"] = df["pincode"].astype(int)
                logger.info(f"Loaded {len(df)} records from API.")
                return df
            else:
                logger.warning(f"API returned status code {response.status_code}. Falling back to local file.")
        except Exception as e:
            logger.error(f"Failed to fetch data from API: {e}. Falling back to local file.")
            
    # Fallback to local file
    if not os.path.exists(file_path):
        return None
    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"])
    df["pincode"] = df["pincode"].astype(int)
    return df

# Initialize Configuration
config = Config()
scores_path = config.get_absolute_path("risk_scores_csv")
df_raw = load_data(scores_path)

if df_raw is None:
    st.error("⚠️ Data files or API backend not detected!")
    st.markdown(
        """
        To run the dashboard in production, set the `API_URL` environment variable pointing to the FastAPI backend.
        
        To generate processed scores locally, run the pipeline command:
        ```bash
        python run_pipeline.py --train
        ```
        """
    )
    st.stop()

# Sidebar Setup
st.sidebar.markdown(
    f"<h3 style='font-family: Outfit; font-weight:700; color:#f8fafc; margin-bottom:-10px;'>🛡️ UIDAI Risk Engine</h3>"
    f"<p style='font-size:0.8rem; color:#64748b; margin-bottom:20px;'>Version {config.version}</p>",
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

# Navigation Selector
nav_option = st.sidebar.radio(
    "Navigation Menu",
    ["Overview Dashboard", "Alert Center", "Pincode Deep-Dive", "Risk Weights Config"]
)

# -----------------
# REAL-TIME RISK RE-CALCULATION
# -----------------
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Live Threshold Adjuster")

# Allow dynamic weighting adjustments
w_iforest = st.sidebar.slider("Isolation Forest Weight", 0.0, 1.0, config.get("risk_scoring.weights.isolation_forest", 0.3), 0.05)
w_ae = st.sidebar.slider("Autoencoder Weight", 0.0, 1.0, config.get("risk_scoring.weights.autoencoder", 0.3), 0.05)
w_lstm = st.sidebar.slider("LSTM Weight", 0.0, 1.0, config.get("risk_scoring.weights.lstm", 0.2), 0.05)
w_spatial = st.sidebar.slider("Spatial Weight", 0.0, 1.0, config.get("risk_scoring.weights.spatial", 0.2), 0.05)

# Validate weights sum
total_weight = w_iforest + w_ae + w_lstm + w_spatial
if abs(total_weight - 1.0) > 1e-4:
    st.sidebar.warning(f"⚠️ Weights sum to {total_weight:.2f}. Standardizing to sum to 1.0.")
    # Standardize
    w_sum = total_weight if total_weight > 0 else 1
    w_iforest /= w_sum
    w_ae /= w_sum
    w_lstm /= w_sum
    w_spatial /= w_sum

# Adjust alert thresholds
threshold_normal = st.sidebar.slider("Normal Risk Limit", 0.1, 0.9, config.get("risk_scoring.thresholds.normal_max", 0.4), 0.05)
threshold_monitor = st.sidebar.slider("Monitor Risk Limit", 0.2, 0.95, config.get("risk_scoring.thresholds.monitor_max", 0.7), 0.05)

# Re-calculate final risk scores and categories in-memory dynamically!
df = df_raw.copy()
df["final_risk_score"] = (
    w_iforest * df["iso_score"] +
    w_ae * df["autoencoder_score"] +
    w_lstm * df["lstm_score"] +
    w_spatial * df["spatial_score"]
)

def classify_live_risk(score):
    if score < threshold_normal:
        return "Normal"
    elif score < threshold_monitor:
        return "Monitor"
    else:
        return "High Risk"

df["risk_level"] = df["final_risk_score"].apply(classify_live_risk)
df["high_risk_flag"] = (df["risk_level"] == "High Risk").astype(int)
df["early_warning"] = ((df["risk_trend"] > 0.15) & (df["final_risk_score"] < threshold_monitor)).astype(int)

# Sidebar Filters
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Filter Scope")
states = sorted(df["state"].unique())
selected_state = st.sidebar.selectbox("Select State", ["All States"] + list(states))

if selected_state != "All States":
    districts = sorted(df[df["state"] == selected_state]["district"].unique())
    selected_district = st.sidebar.selectbox("Select District", ["All Districts"] + list(districts))
else:
    selected_district = "All Districts"

# Filter df based on selection
df_filtered = df.copy()
if selected_state != "All States":
    df_filtered = df_filtered[df_filtered["state"] == selected_state]
    if selected_district != "All Districts":
        df_filtered = df_filtered[df_filtered["district"] == selected_district]

# Define metrics
total_pincodes = df_filtered["pincode"].nunique()
active_alerts = len(df_filtered[df_filtered["risk_level"] == "High Risk"])
monitored_regions = len(df_filtered[df_filtered["risk_level"] == "Monitor"])
early_warning_alerts = df_filtered["early_warning"].sum()
avg_risk_score = df_filtered["final_risk_score"].mean()

# Header layout
st.markdown("<h1 class='main-title'>Aadhaar Anomaly & Risk Intelligence Engine</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>State, District, and Pincode level spatial-temporal risk aggregated dashboard</p>", unsafe_allow_html=True)

# -----------------
# 1. OVERVIEW DASHBOARD
# -----------------
if nav_option == "Overview Dashboard":
    # Metric KPI Row
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    with kpi1:
        st.markdown(
            f"<div class='kpi-card alert-high'>"
            f"<div class='kpi-label'>🔴 High Risk Alerts</div>"
            f"<div class='kpi-value'>{active_alerts:,}</div>"
            f"<div style='font-size:0.8rem; color:#ef4444;'>Active inspections</div>"
            f"</div>",
            unsafe_allow_html=True
        )
        
    with kpi2:
        st.markdown(
            f"<div class='kpi-card alert-monitor'>"
            f"<div class='kpi-label'>🟡 Monitor Status</div>"
            f"<div class='kpi-value'>{monitored_regions:,}</div>"
            f"<div style='font-size:0.8rem; color:#f59e0b;'>Regions to watch</div>"
            f"</div>",
            unsafe_allow_html=True
        )
        
    with kpi3:
        st.markdown(
            f"<div class='kpi-card'>"
            f"<div class='kpi-label'>⚡ Early Warnings</div>"
            f"<div class='kpi-value'>{early_warning_alerts:,}</div>"
            f"<div style='font-size:0.8rem; color:#a855f7;'>Elevated trends</div>"
            f"</div>",
            unsafe_allow_html=True
        )
        
    with kpi4:
        st.markdown(
            f"<div class='kpi-card alert-normal'>"
            f"<div class='kpi-label'>🟢 Average Risk</div>"
            f"<div class='kpi-value'>{avg_risk_score:.3f}</div>"
            f"<div style='font-size:0.8rem; color:#10b981;'>Aggregated score</div>"
            f"</div>",
            unsafe_allow_html=True
        )
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Graphs Row
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Risk Score Distribution by State")
        state_risk = df_filtered.groupby("state")["final_risk_score"].mean().reset_index().sort_values("final_risk_score", ascending=False).head(10)
        fig = px.bar(
            state_risk,
            x="final_risk_score",
            y="state",
            orientation="h",
            color="final_risk_score",
            color_continuous_scale="reds",
            labels={"final_risk_score": "Mean Risk Score", "state": "State"},
            height=400
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.markdown("### 📊 Distribution of Risk Categories")
        risk_dist = df_filtered["risk_level"].value_counts().reset_index()
        risk_dist.columns = ["Category", "Count"]
        fig = px.pie(
            risk_dist,
            values="Count",
            names="Category",
            color="Category",
            color_discrete_map={"Normal": "#10b981", "Monitor": "#f59e0b", "High Risk": "#ef4444"},
            hole=0.4,
            height=400
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### ⏳ Temporal Risk Heatmap Trend")
    plot_df = df_filtered.copy()
    plot_df["date_str"] = plot_df["date"].dt.strftime("%Y-%m")
    heatmap_data = plot_df.groupby(["state", "date_str"])["final_risk_score"].mean().reset_index()
    
    fig = px.density_heatmap(
        heatmap_data,
        x="date_str",
        y="state",
        z="final_risk_score",
        histfunc="avg",
        color_continuous_scale="YlOrRd",
        labels={"date_str": "Timeline", "state": "State", "final_risk_score": "Risk Level"},
        height=500
    )
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

# -----------------
# 2. ALERT CENTER
# -----------------
elif nav_option == "Alert Center":
    st.markdown("### 🔔 Active Risk Alerts & Explanations")
    st.markdown("Use this table to audit anomalies requiring administrative attention.")
    
    # Filter only actionable rows
    alerts_df = df_filtered[
        (df_filtered["risk_level"] == "High Risk") | 
        (df_filtered["early_warning"] == 1)
    ].sort_values("final_risk_score", ascending=False)
    
    # Filter selection
    alert_lvl = st.selectbox("Show alert levels", ["All Alerts", "High Risk Only", "Early Warnings Only"])
    if alert_lvl == "High Risk Only":
        alerts_df = alerts_df[alerts_df["risk_level"] == "High Risk"]
    elif alert_lvl == "Early Warnings Only":
        alerts_df = alerts_df[alerts_df["early_warning"] == 1]
        
    st.markdown(f"**Found {len(alerts_df)} matching alerts.**")
    
    # Display table with relevant columns
    cols_to_show = [
        "date", "state", "district", "pincode", 
        "bio_demo_ratio", "final_risk_score", "risk_level", 
        "early_warning", "explanation", "semantic_context"
    ]
    
    # Format date
    display_df = alerts_df[cols_to_show].copy()
    display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
    
    def style_risk_score(val):
        try:
            score = float(val)
            alpha = min(max(score, 0.0), 1.0) * 0.7
            return f"background-color: rgba(239, 68, 68, {alpha:.2f}); color: #f8fafc"
        except (ValueError, TypeError):
            return ""
            
    styled_df = display_df.style
    if hasattr(styled_df, "map"):
        styled_df = styled_df.map(style_risk_score, subset=["final_risk_score"])
    else:
        styled_df = styled_df.applymap(style_risk_score, subset=["final_risk_score"])
        
    st.dataframe(
        styled_df,
        use_container_width=True,
        height=500
    )
    
    # Download Button
    csv = display_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Active Alerts CSV",
        data=csv,
        file_name="uidai_risk_alerts.csv",
        mime="text/csv"
    )

# -----------------
# 3. PINCODE DEEP-DIVE
# -----------------
elif nav_option == "Pincode Deep-Dive":
    st.markdown("### 🔍 Pincode Behavioral Inspection")
    
    pincodes_list = sorted(df_filtered["pincode"].unique())
    selected_pincode = st.selectbox("Select Pincode for Deep-Dive", pincodes_list)
    
    pincode_df = df[df["pincode"] == selected_pincode].sort_values("date")
    
    # Metadata info
    latest_row = pincode_df.iloc[-1]
    
    inf1, inf2, inf3, inf4 = st.columns(4)
    with inf1:
        st.metric("Region State", f"{latest_row['state']}")
    with inf2:
        st.metric("District", f"{latest_row['district']}")
    with inf3:
        st.metric("Latest Risk Score", f"{latest_row['final_risk_score']:.3f}", delta=f"{latest_row['risk_trend']:.3f} (7-day trend)")
    with inf4:
        st.metric("Risk Level Status", f"{latest_row['risk_level']}")
        
    st.markdown(f"**Natural Language Root Cause**: *{latest_row['explanation']}*")
    st.markdown(f"**Operational context**: *{latest_row['semantic_context']}*")
    
    st.markdown("---")
    
    # Risk timeline chart
    st.markdown("#### 📅 Risk Timeline & Component Breakdown")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pincode_df["date"], y=pincode_df["final_risk_score"], name="Final Combined Risk", line=dict(color="#6366f1", width=3, dash='solid')))
    fig.add_trace(go.Scatter(x=pincode_df["date"], y=pincode_df["iso_score"], name="Behavioral Score (I-Forest)", line=dict(color="#f59e0b", width=1.5, dash='dash')))
    fig.add_trace(go.Scatter(x=pincode_df["date"], y=pincode_df["autoencoder_score"], name="Reconstruction Score (AE)", line=dict(color="#ec4899", width=1.5, dash='dash')))
    fig.add_trace(go.Scatter(x=pincode_df["date"], y=pincode_df["lstm_score"], name="Temporal Anomaly (LSTM)", line=dict(color="#a855f7", width=1.5, dash='dash')))
    fig.add_trace(go.Scatter(x=pincode_df["date"], y=pincode_df["spatial_score"], name="Spatial Anomaly (NN)", line=dict(color="#10b981", width=1.5, dash='dash')))
    
    fig.update_layout(
        template="plotly_dark",
        title=f"Evolution of Risk Metrics - Pincode {selected_pincode}",
        xaxis_title="Date",
        yaxis_title="Normalized Risk (0 - 1)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Biometric vs Demographic total timeline
    st.markdown("#### 👥 Monthly Biometric vs Demographic Updates Volume")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=pincode_df["date"], y=pincode_df["biometric_total"], name="Biometric Updates", marker_color="#3b82f6"))
    fig.add_trace(go.Bar(x=pincode_df["date"], y=pincode_df["demographic_total"], name="Demographic Updates", marker_color="#93c5fd"))
    
    fig.update_layout(
        barmode='group',
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Total Update Records",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------
# 4. RISK WEIGHTS CONFIG
# -----------------
elif nav_option == "Risk Weights Config":
    st.markdown("### 🔬 Model Performance & Risk Engine Configuration")
    
    st.markdown(
        """
        The **Risk Fusion Engine** synthesizes scores from four distinct AI components.
        Adjust the weights in the sidebar to simulate different threat scenarios and balance component outputs:
        """
    )
    
    # Diagnostic component metrics
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🛡️ Active Score Configuration Weights")
        weights_data = pd.DataFrame({
            "Risk Component": ["Isolation Forest", "Autoencoder", "LSTM", "Spatial"],
            "Weight": [w_iforest, w_ae, w_lstm, w_spatial]
        })
        fig = px.bar(weights_data, x="Risk Component", y="Weight", color="Weight", color_continuous_scale="viridis", height=300)
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.markdown("#### 🚦 Reclassified Category Distribution")
        reclass_counts = df["risk_level"].value_counts().reset_index()
        reclass_counts.columns = ["Risk Level", "Pincode Count"]
        st.table(reclass_counts)
        st.info("The values above are updated dynamically in real-time as you drag the sliders in the sidebar!")
        
    # Model descriptions
    st.markdown("---")
    st.markdown("### 🤖 Mathematical Models in Play")
    
    st.markdown(
        """
        1. **Behavioral Intelligence (Isolation Forest)**
           - **Weights set**: `%.2f`
           - **Role**: Measures overall multidimensional out-of-distribution updates. Catches extreme spikes in volumes.
        
        2. **Behavioral Reconstruction (Dense Autoencoder)**
           - **Weights set**: `%.2f`
           - **Role**: Learns the normal coordinate ratios between demographic age brackets and biometric categories. Flagging anomalies where age skew is mismatched.
        
        3. **Temporal Intelligence (LSTM Network)**
           - **Weights set**: `%.2f`
           - **Role**: Learns long-term temporal trends. Separates normal cyclical spikes (like school admission updates) from sustained abnormal trends.
        
        4. **Spatial Neighborhood Aggregator (Nearest Neighbors)**
           - **Weights set**: `%.2f`
           - **Role**: Measures localized density patterns. Flags pincodes whose updates differ drastically from their immediate numeric neighbors on the same date.
        """ % (w_iforest, w_ae, w_lstm, w_spatial)
    )
