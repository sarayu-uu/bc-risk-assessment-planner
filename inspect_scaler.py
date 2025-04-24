import joblib
import numpy as np

# Load the scaler
scaler = joblib.load('evidence_based_scaler.pkl')

# Print scaler attributes
print("Scaler type:", type(scaler))
print("Number of features:", scaler.n_features_in_)

# Check if feature_names_in_ attribute exists
if hasattr(scaler, 'feature_names_in_'):
    print("Feature names:", scaler.feature_names_in_)
else:
    print("Scaler does not have feature_names_in_ attribute")

# Print other attributes
print("Scale:", scaler.scale_)
print("Mean:", scaler.mean_)
print("Var:", scaler.var_)