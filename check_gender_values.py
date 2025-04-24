"""
Check the values in the Gender column of the NHANES dataset
"""

from nhanes.load import load_NHANES_data
import pandas as pd

# Load NHANES data
print("Loading NHANES data...")
nhanes_data = load_NHANES_data(year='2017-2018')

# Check if Gender column exists
if 'Gender' in nhanes_data.columns:
    print("\nGender column exists")
    
    # Get unique values in Gender column
    gender_values = nhanes_data['Gender'].unique()
    print(f"Unique values in Gender column: {gender_values}")
    
    # Count values in Gender column
    gender_counts = nhanes_data['Gender'].value_counts()
    print("\nCounts of values in Gender column:")
    print(gender_counts)
    
    # Check for NaN values
    nan_count = nhanes_data['Gender'].isna().sum()
    print(f"\nNumber of NaN values in Gender column: {nan_count}")
    
    # Print a few rows with Gender values
    print("\nSample rows with Gender values:")
    sample_rows = nhanes_data[['Gender']].head(10)
    print(sample_rows)
else:
    print("\nGender column does not exist in the dataset")