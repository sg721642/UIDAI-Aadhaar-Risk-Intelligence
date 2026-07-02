import os
import re
import json
import pandas as pd
from src.config import Config
from src.utils.logger import get_logger

logger = get_logger("data_loader", log_file="pipeline.log")

# Standard Indian states & UTs canonical mapping
INDIA_STATE_CANONICAL = {
    "andhra pradesh": "Andhra Pradesh",
    "arunachal pradesh": "Arunachal Pradesh",
    "assam": "Assam",
    "bihar": "Bihar",
    "chhattisgarh": "Chhattisgarh",
    "goa": "Goa",
    "gujarat": "Gujarat",
    "haryana": "Haryana",
    "himachal pradesh": "Himachal Pradesh",
    "jharkhand": "Jharkhand",
    "karnataka": "Karnataka",
    "kerala": "Kerala",
    "madhya pradesh": "Madhya Pradesh",
    "maharashtra": "Maharashtra",
    "manipur": "Manipur",
    "meghalaya": "Meghalaya",
    "mizoram": "Mizoram",
    "nagaland": "Nagaland",
    "odisha": "Odisha",
    "orissa": "Odisha",
    "punjab": "Punjab",
    "rajasthan": "Rajasthan",
    "sikkim": "Sikkim",
    "tamil nadu": "Tamil Nadu",
    "telangana": "Telangana",
    "tripura": "Tripura",
    "uttar pradesh": "Uttar Pradesh",
    "uttarakhand": "Uttarakhand",
    "west bengal": "West Bengal",
    "westbengal": "West Bengal",
    "west bangal": "West Bengal",
    "delhi": "Delhi",
    "nct of delhi": "Delhi",
    "chandigarh": "Chandigarh",
    "ladakh": "Ladakh",
    "lakshadweep": "Lakshadweep",
    "jammu & kashmir": "Jammu & Kashmir",
    "jammu and kashmir": "Jammu & Kashmir",
    "jammukashmir": "Jammu & Kashmir",
    "andaman & nicobar islands": "Andaman & Nicobar Islands",
    "andaman and nicobar islands": "Andaman & Nicobar Islands",
    "puducherry": "Puducherry",
    "pondicherry": "Puducherry",
    "dadra & nagar haveli": "Dadra & Nagar Haveli and Daman & Diu",
    "dadra and nagar haveli": "Dadra & Nagar Haveli and Daman & Diu",
    "daman & diu": "Dadra & Nagar Haveli and Daman & Diu",
    "daman and diu": "Dadra & Nagar Haveli and Daman & Diu",
    "dadra & nagar haveli and daman & diu": "Dadra & Nagar Haveli and Daman & Diu",
    "dadra and nagar haveli and daman and diu": "Dadra & Nagar Haveli and Daman & Diu"
}

def canonicalize_state(state):
    if pd.isna(state):
        return state
    s = str(state).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return INDIA_STATE_CANONICAL.get(s, s.title())

class DataLoader:
    def __init__(self, config: Config):
        self.config = config
        self.demo_dir = config.get_absolute_path("demographic_raw_dir")
        self.enrol_dir = config.get_absolute_path("enrolment_raw_dir")
        self.processed_dir = config.get_absolute_path("processed_dir")
        
        # Ensure processed directory exists
        os.makedirs(self.processed_dir, exist_ok=True)

    def load_prefixed_files(self, folder_path, prefix):
        """Loads and standardizes CSV files in a folder starting with a prefix."""
        logger.info(f"Scanning folder {folder_path} for files with prefix '{prefix}'")
        dfs = []
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        for file in os.listdir(folder_path):
            if not file.lower().startswith(prefix.lower()):
                continue
            if not file.endswith(".csv"):
                continue

            full_path = os.path.join(folder_path, file)
            logger.info(f"Loading file: {file}")
            df = pd.read_csv(full_path)
            
            # Standardize column names
            df.columns = df.columns.str.lower().str.strip()
            df["source_file"] = file
            dfs.append(df)

        if not dfs:
            raise ValueError(f"No CSV files found with prefix '{prefix}' in {folder_path}")

        return pd.concat(dfs, ignore_index=True)

    def validate_dataset(self, df, name):
        """Generates a structural and quality audit report for the dataset."""
        logger.info(f"Validating dataset '{name}'...")
        total_rows = len(df)
        dups = int(df.duplicated().sum())
        missing = df.isnull().sum().to_dict()
        missing = {k: int(v) for k, v in missing.items()}
        
        # Check for expected columns
        columns = df.columns.tolist()
        
        report = {
            "dataset_name": name,
            "total_rows": total_rows,
            "duplicate_rows": dups,
            "missing_values": missing,
            "columns": columns
        }
        return report

    def clean_dataset(self, df):
        """Removes duplicates and drops invalid dates."""
        initial_rows = len(df)
        # Drop strict duplicates
        df = df.drop_duplicates()
        cleaned_dups = initial_rows - len(df)
        if cleaned_dups > 0:
            logger.info(f"Removed {cleaned_dups} duplicate rows.")

        # Safe date parsing
        df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
        initial_len = len(df)
        df = df.dropna(subset=["date"])
        invalid_dates = initial_len - len(df)
        if invalid_dates > 0:
            logger.warning(f"Dropped {invalid_dates} rows due to invalid date formats.")

        return df

    def run_pipeline(self):
        """Orchestrates data loading, cleaning, validation, and merging."""
        logger.info("Starting Data Pipeline execution...")
        
        # 1. Load data
        # Demographic data directory has files starting with api_data_aadhar_demographic
        # Enrolment data directory has files starting with api_data_aadhar_enrolment
        raw_demo_df = self.load_prefixed_files(self.demo_dir, "api_data_aadhar_demographic")
        raw_enrol_df = self.load_prefixed_files(self.enrol_dir, "api_data_aadhar_enrolment")
        
        logger.info(f"Raw demographic shape: {raw_demo_df.shape}")
        logger.info(f"Raw enrolment shape: {raw_enrol_df.shape}")

        # 2. Validate raw data
        demo_validation = self.validate_dataset(raw_demo_df, "Demographic")
        enrol_validation = self.validate_dataset(raw_enrol_df, "Enrolment")

        # Save validation report
        validation_report = {
            "demographic_audit": demo_validation,
            "enrolment_audit": enrol_validation
        }
        report_path = os.path.join(self.processed_dir, "validation_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(validation_report, f, indent=4)
        logger.info(f"Saved validation report to: {report_path}")

        # 3. Clean raw datasets
        demo_cleaned = self.clean_dataset(raw_demo_df)
        enrol_cleaned = self.clean_dataset(raw_enrol_df)

        # 4. Map Enrolment Columns (to represent biometric data)
        # enrolment columns: age_5_17 -> bio_age_5_17, age_18_greater -> bio_age_17_
        logger.info("Mapping enrolment columns to act as biometric update records...")
        mapping = self.config.get("data_mapping.enrolment_to_biometric")
        enrol_cleaned = enrol_cleaned.rename(columns=mapping)

        # 5. State canonicalization
        logger.info("Standardizing state names...")
        demo_cleaned["state"] = demo_cleaned["state"].apply(canonicalize_state)
        enrol_cleaned["state"] = enrol_cleaned["state"].apply(canonicalize_state)

        # 6. Merge datasets
        logger.info("Merging demographic and enrolment datasets...")
        merged_df = pd.merge(
            enrol_cleaned,
            demo_cleaned,
            on=["date", "state", "district", "pincode"],
            suffixes=("_bio", "_demo"),
            how="inner"
        )
        
        # Sort by pincode and date
        merged_df = merged_df.sort_values(["pincode", "date"])
        merged_df.reset_index(drop=True, inplace=True)
        
        logger.info(f"Merged dataset shape: {merged_df.shape}")
        
        return merged_df
