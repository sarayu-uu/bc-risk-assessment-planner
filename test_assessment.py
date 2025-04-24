import pandas as pd
import numpy as np
from risk_assessment_utils import assess_risk_and_plan, LIFESTYLE_FEATURES_NAMES

# Create a test input with lifestyle features
test_input = {}

# Add lifestyle features
for feature in LIFESTYLE_FEATURES_NAMES:
    test_input[feature] = 0.5  # Default value

# Add some medical features (not all needed, our fix should handle missing ones)
medical_features = [
    "mean radius", "mean texture", "mean perimeter", "mean area", 
    "mean smoothness", "mean compactness", "mean concavity", "mean concave points",
    "mean symmetry", "mean fractal dimension", "worst radius", "worst texture",
    "worst perimeter", "worst area", "worst smoothness", "worst compactness",
    "worst concavity", "worst concave points", "worst symmetry", "worst fractal dimension"
]

for feature in medical_features:
    test_input[feature] = 0.5  # Default value

# Create DataFrame
input_df = pd.DataFrame([test_input])

print("Running assessment with test input...")
print(f"Input features: {input_df.columns.tolist()}")

# Run assessment
assessment_result, prevention_plan = assess_risk_and_plan(input_df)

if assessment_result and prevention_plan:
    print("\nAssessment successful!")
    print(f"Risk score: {assessment_result['risk_score']}")
    print(f"Risk category: {assessment_result['risk_category']}")
    print("\nPrevention plan generated with recommendations:")
    for rec in prevention_plan.get('personalized_recommendations', []):
        print(f"- {rec}")
else:
    print("\nAssessment failed!")