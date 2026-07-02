import os
import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.config import Config
from src.data.data_loader import DataLoader
from src.features.engineer import FeatureEngineer
from src.models.train import ModelTrainer
from src.models.predict import RiskPredictor
from src.utils.logger import get_logger

logger = get_logger("run_pipeline", log_file="pipeline.log")

def run(train_flag=False, predict_flag=False, sample_size=None):
    logger.info("Initializing UIDAI Anomaly & Risk Intelligence Pipeline...")
    
    # 1. Config Loader
    config = Config()
    
    # 2. Data Pipeline
    loader = DataLoader(config)
    merged_data = loader.run_pipeline()
    
    # 3. Feature Engineering
    engineer = FeatureEngineer(config)
    engineered_data = engineer.engineer_features(merged_data)
    
    if sample_size is not None:
        logger.info(f"Sampling dataset to {sample_size} rows for local low-resource execution.")
        engineered_data = engineered_data.head(sample_size)
    
    # Check if models are missing and we need to train them
    scaler_path = config.get_absolute_path("scaler_joblib")
    models_missing = not os.path.exists(scaler_path)
    
    # 4. Training (if flag or missing models)
    if train_flag or models_missing:
        if models_missing:
            logger.info("No model checkpoints detected. Triggering automatic training.")
        logger.info("Starting model training phase...")
        trainer = ModelTrainer(config)
        trainer.train_all(engineered_data)
        logger.info("Model training completed.")
    
    # 5. Prediction / Inference
    if predict_flag or not train_flag:
        logger.info("Starting risk scoring and prediction phase...")
        predictor = RiskPredictor(config)
        enriched_data = predictor.predict_risk(engineered_data)
        
        # Ensure processed directory exists
        processed_dir = config.get_absolute_path("processed_dir")
        os.makedirs(processed_dir, exist_ok=True)
        
        # Save enriched scores
        scores_csv_path = config.get_absolute_path("risk_scores_csv")
        enriched_data.to_csv(scores_csv_path, index=False)
        logger.info(f"Saved enriched risk scores to: {scores_csv_path}")
        
        # Save actionable regions (High Risk or Early Warning)
        actionable_path = config.get_absolute_path("actionable_regions_csv")
        mon_max = config.get("risk_scoring.thresholds.monitor_max", 0.7)
        
        actionable_data = enriched_data[
            (enriched_data["risk_level"] == "High Risk") |
            (enriched_data["early_warning"] == 1)
        ]
        actionable_data.to_csv(actionable_path, index=False)
        logger.info(f"Saved actionable alerts to: {actionable_path}")
        
        # Save outputs directory CSVs
        outputs_dir = os.path.join(os.path.dirname(processed_dir), "..", "outputs")
        os.makedirs(outputs_dir, exist_ok=True)
        
        enriched_data.to_csv(os.path.join(outputs_dir, "risk_scores.csv"), index=False)
        actionable_data.to_csv(os.path.join(outputs_dir, "alerts.csv"), index=False)
        logger.info(f"Saved results to outputs/ folder.")
        
        # 6. Generate Summary JSON
        summary = {
            "total_records": len(enriched_data),
            "total_alerts": len(actionable_data),
            "risk_distribution": enriched_data["risk_level"].value_counts().to_dict(),
            "early_warning_alerts": int(enriched_data["early_warning"].sum()),
            "states_monitored": int(enriched_data["state"].nunique()),
            "pincodes_monitored": int(enriched_data["pincode"].nunique()),
            "average_risk_score": float(enriched_data["final_risk_score"].mean())
        }
        
        summary_json_path = config.get_absolute_path("summary_json")
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)
        with open(os.path.join(outputs_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)
        logger.info(f"Saved execution summary JSON to: {summary_json_path}")
        
        # 7. Generate Heatmap Plot
        logger.info("Generating risk heatmap plot...")
        plots_dir = config.get_absolute_path("plots_dir")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Pivot table for state/date risk aggregation
        # Format date as string for plotting readability
        plot_df = enriched_data.copy()
        plot_df["date_str"] = plot_df["date"].dt.strftime("%Y-%m")
        heatmap_data = plot_df.groupby(["state", "date_str"])["final_risk_score"].mean().unstack()
        
        plt.figure(figsize=(14, 8))
        sns.heatmap(heatmap_data, cmap="YlOrRd", cbar_kws={'label': 'Mean Risk Score'})
        plt.title("Context-Aware Aadhaar Risk Heatmap (State-Time)")
        plt.xlabel("Timeline")
        plt.ylabel("States / UTs")
        plt.tight_layout()
        
        heatmap_path = config.get_absolute_path("heatmap_png")
        plt.savefig(heatmap_path, dpi=300)
        plt.close()
        
        # Also copy to outputs/plots/
        out_plots_dir = os.path.join(outputs_dir, "..", "plots")
        os.makedirs(out_plots_dir, exist_ok=True)
        shutil_path = os.path.join(out_plots_dir, "heatmap.png")
        # Save a duplicate copy
        plt.figure(figsize=(14, 8))
        sns.heatmap(heatmap_data, cmap="YlOrRd", cbar_kws={'label': 'Mean Risk Score'})
        plt.title("Context-Aware Aadhaar Risk Heatmap (State-Time)")
        plt.tight_layout()
        plt.savefig(shutil_path, dpi=300)
        plt.close()
        
        logger.info(f"Saved risk heatmap visualization to: {heatmap_path}")
        
    logger.info("Pipeline execution completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UIDAI Risk Engine - ML Pipeline CLI")
    parser.add_argument("--train", action="store_true", help="Force train the anomaly detection models")
    parser.add_argument("--predict", action="store_true", help="Force prediction phase only")
    parser.add_argument("--sample", type=int, default=None, help="Limit execution to a smaller dataset sample for low-resource environments")
    args = parser.parse_args()
    
    run(train_flag=args.train, predict_flag=args.predict, sample_size=args.sample)
