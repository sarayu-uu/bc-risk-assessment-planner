"""
Check columns in NHANES data
"""

from nhanes.load import load_NHANES_data
import pandas as pd

# Load NHANES data
print("Loading NHANES data...")
nhanes_data = load_NHANES_data(year='2017-2018')

# Print column names
print("\nNHANES data columns:")
for col in nhanes_data.columns:
    print(f"  - {col}")

# Check if specific columns exist
variables = {
    'ALQ130': 'alcohol_days_per_year',
    'ALQ120Q': 'alcohol_drinks_per_day',
    'BMXBMI': 'body_weight_bmi',
    'PAQ650': 'physical_inactivity_level',
    'PAQ665': 'physical_activity_vigorous',
    'DBQ700': 'diet_quality_fruit',
    'DBQ710': 'diet_quality_vegetables',
    'DBQ720': 'diet_quality_greens',
    'SMQ020': 'smoking_status',
    'SMQ040': 'smoking_current',
    'RHQ131': 'reproductive_age_first_birth',
    'RHQ160': 'reproductive_breastfed',
    'RHQ420': 'hormone_birth_control',
    'RHQ540': 'hormone_hrt',
    'RIAGENDR': 'gender',
    'RIDAGEYR': 'age',
}

print("\nChecking for specific variables:")
for var, name in variables.items():
    exists = var in nhanes_data.columns
    print(f"  - {var} -> {name}: {'Found' if exists else 'Not found'}")

# Print data shape
print(f"\nData shape: {nhanes_data.shape}")
print(f"Number of rows: {len(nhanes_data)}")
print(f"Number of columns: {len(nhanes_data.columns)}")

# Print first few rows
print("\nFirst 5 rows:")
print(nhanes_data.head())