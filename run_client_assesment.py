import pandas as pd
import numpy as np
import joblib
import json
import random
import warnings

# Import the necessary function and constants from the utility file
from risk_assessment_utils import assess_risk_and_plan, METADATA_FILENAME, ALL_FEATURE_NAMES

# Ignore convergence warnings if they pop up during loading/prediction
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# --- Constants (Imported or defined if not in utils) ---
# MODEL_FILENAME, SCALER_FILENAME, SELECTOR_FILENAME, METADATA_FILENAME are imported
# LIFESTYLE_FEATURES_NAMES are used internally by the planner in utils

# --- REMOVE PreventionPlanGenerator Class Definition --- 
# (It's now implicitly used within the imported assess_risk_and_plan)

# --- REMOVE assess_risk_and_plan Function Definition --- 
# (It's now imported from risk_assessment_utils)


# --- AGENT INTERACTION STARTS HERE ---

print("\n--- Client Risk Assessment & Lifestyle Planner ---")

# 1. Agent Collects Client Data
#    (In reality, this comes from forms, interviews, devices, EHRs)
#    For this example, we'll simulate it.

#    a) Simulate Medical Features (Agent might get these from EHR or tests)
#       Using average values from the dataset as a *placeholder* - replace with REAL client data
print("Simulating client medical data (Replace with actual values)...")
# Ensure ALL_FEATURE_NAMES is available from the import
original_feature_names = [f for f in ALL_FEATURE_NAMES if f not in LIFESTYLE_FEATURES_NAMES and f not in INTERACTION_FEATURES_NAMES] # Recreate list of original features based on ALL
client_medical_data = { # Using placeholders - needs 30 values
    'mean radius': 14.1, 'mean texture': 19.2, 'mean perimeter': 91.9, 'mean area': 654.8,
    'mean smoothness': 0.096, 'mean compactness': 0.104, 'mean concavity': 0.088,
    'mean concave points': 0.048, 'mean symmetry': 0.181, 'mean fractal dimension': 0.062,
    'radius error': 0.405, 'texture error': 1.216, 'perimeter error': 2.866, 'area error': 40.33,
    'smoothness error': 0.007, 'compactness error': 0.025, 'concavity error': 0.031,
    'concave points error': 0.011, 'symmetry error': 0.020, 'fractal dimension error': 0.003,
    'worst radius': 16.2, 'worst texture': 25.6, 'worst perimeter': 107.2, 'worst area': 880.5,
    'worst smoothness': 0.132, 'worst compactness': 0.254, 'worst concavity': 0.272,
    'worst concave points': 0.114, 'worst symmetry': 0.290, 'worst fractal dimension': 0.083
}
if len(client_medical_data) != len(original_feature_names):
     print(f"Warning: Mismatch in number of medical features provided ({len(client_medical_data)}) vs expected ({len(original_feature_names)}). Ensure all 30 medical features are present.")
     # Consider adding a stop/error here in a real app


#    b) Simulate Lifestyle Features (Agent asks client questions / uses survey data)
#       Scale: 0-1 (higher generally means more/better, except for stress, alcohol, smoking)
print("Simulating client lifestyle data (Replace with actual values)...")
client_lifestyle_data = {
    'physical_activity': 0.3,  # Low activity
    'diet_quality': 0.4,       # Mediocre diet
    'stress_level': 0.8,       # High stress
    'sleep_quality': 0.5,      # Okay sleep
    'alcohol_consumption': 0.7,# High consumption
    'smoking_history': 0.1     # Smoked a little in past, low current risk factor score
}

# 2. Prepare Input DataFrame for the Model
print("Preparing client data for model...")
client_data_all_features = {}
client_data_all_features.update(client_medical_data)
client_data_all_features.update(client_lifestyle_data)

# Calculate Interaction Features
try:
    client_data_all_features['activity_immune_interaction'] = client_data_all_features['physical_activity'] * client_data_all_features['worst concave points']
    client_data_all_features['diet_cell_interaction'] = client_data_all_features['diet_quality'] * client_data_all_features['mean texture']
    client_data_all_features['stress_immune_interaction'] = (1 - client_data_all_features['stress_level']) * client_data_all_features['mean smoothness']
except KeyError as e:
    print(f"Error calculating interaction features. Missing base feature: {e}. Cannot proceed.")
    exit()

# Create the DataFrame
client_input_df = pd.DataFrame([client_data_all_features])

# Note: Ordering check is now done inside assess_risk_and_plan

# 3. Run the Assessment and Planning Function (Now imported)
print("Running assessment and generating plan...")
assessment, plan = assess_risk_and_plan(client_input_df)

# 4. Agent Reviews and Presents Results to Client
if assessment and plan:
    print("\n\n--- FOR HEALTHCARE AGENT REVIEW ---")

    print("\nClient Risk Assessment:")
    print(f"  - Risk Score: {assessment['risk_score']}")
    print(f"  - Risk Category: {assessment['risk_category']}")
    print(f"  - Basis: {assessment['explanation']}")

    print("\nGenerated Personalized Prevention Plan:")
    print(f"  - Key Lifestyle Factors Identified for Improvement:")
    if plan['key_lifestyle_factors_for_improvement']:
        for factor in plan['key_lifestyle_factors_for_improvement']:
            print(f"    * {factor.replace('_', ' ').title()}")
    else:
        print("    * None - Current lifestyle factors appear within healthy ranges.")

    print("\n  - Personalized Recommendations:")
    for i, rec in enumerate(plan['personalized_recommendations']):
        print(f"    {i+1}. {rec}")

    print("\n  - Suggested Timeline:")
    for item in plan['suggested_timeline']:
        print(f"    - {item}")

    print("\n  - Monitoring Suggestions:")
    for item in plan['monitoring_suggestions']:
        print(f"    - {item}")

    print("\n--- END OF REPORT ---")

    # Agent would now discuss these results with the client.

else:
    print("\n--- Could not generate assessment and plan for the client. Check logs for errors. ---")
