# Dataset Specifications

The UIDAI Risk Engine processes aggregated demographic and enrolment updates at the spatial-temporal grain: `date, state, district, pincode`.

## Raw Data Inputs
1. **Demographic Dataset**
   - **Path**: `DATASET_USED-20260702T045137Z-3-001/DATASET_USED/api_data_aadhar_demographic/`
   - **Files**: 5 CSV files totaling ~91MB and ~2.07 million rows.
   - **Schema**:
     - `date`: D-M-Y format
     - `state`: Localized string
     - `district`: Localized string
     - `pincode`: 6-digit numeric integer
     - `demo_age_5_17`: Counts of demographic updates for children aged 5-17
     - `demo_age_17_`: Counts of demographic updates for adults aged 17 and above
     
2. **Enrolment Dataset**
   - **Path**: `DATASET_USED-20260702T045137Z-3-001/DATASET_USED/api_data_aadhar_enrolment/`
   - **Files**: 3 CSV files totaling ~45MB and ~1.00 million rows.
   - **Schema**:
     - `date`: D-M-Y format
     - `state`: Localized string
     - `district`: Localized string
     - `pincode`: 6-digit numeric integer
     - `age_0_5`: Enrolments for infants under 5 (excluded from biometrics)
     - `age_5_17`: Enrolments for children aged 5-17 (requires biometric)
     - `age_18_greater`: Enrolments for adults aged 18 and above (requires biometric)

## Data Cleaning & Processing
1. **Deduplication**: Strict duplicates are removed.
2. **Date Parsing**: Standardized using `dayfirst=True`; rows with corrupt dates are dropped.
3. **State Canonicalization**: Standardizes all spelling/whitespace variations to 36 canonical Indian States & UTs.
4. **Biometric Mapping**: Maps `age_5_17` and `age_18_greater` from the Enrolment dataset to represent biometric update records, as these require biometric submission.
5. **Inner Merge**: Merges both datasets on `date, state, district, pincode` to generate a unified view of regional update activities.
