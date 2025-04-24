"""
Test the NHANES data loader to identify the error
"""

from nhanes_data_loader_fixed import integrate_lifestyle_factors_with_breast_cancer_data

print("Testing NHANES data loader...")
try:
    df, original_feature_names = integrate_lifestyle_factors_with_breast_cancer_data(use_nhanes=True)
    print("Success! Dataset shape:", df.shape)
    print("Lifestyle columns:", [col for col in df.columns if col not in original_feature_names])
except Exception as e:
    print(f"Error: {e}")