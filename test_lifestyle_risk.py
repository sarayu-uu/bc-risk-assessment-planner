import pandas as pd
import numpy as np
from risk_assessment_utils import calculate_lifestyle_risk_score, LIFESTYLE_FEATURES_NAMES

# Test with all lifestyle factors set to high risk (0.8)
high_risk_test = pd.DataFrame([[0.8] * len(LIFESTYLE_FEATURES_NAMES)], columns=LIFESTYLE_FEATURES_NAMES)
print("\n--- Testing with all lifestyle factors set to HIGH RISK (0.8) ---")
risk_score, risk_category, explanation = calculate_lifestyle_risk_score(high_risk_test)
print(f"Risk Score: {risk_score:.3f}")
print(f"Risk Category: {risk_category}")
print(f"Explanation: {explanation}")

# Test with all lifestyle factors set to low risk (0.2)
low_risk_test = pd.DataFrame([[0.2] * len(LIFESTYLE_FEATURES_NAMES)], columns=LIFESTYLE_FEATURES_NAMES)
print("\n--- Testing with all lifestyle factors set to LOW RISK (0.2) ---")
risk_score, risk_category, explanation = calculate_lifestyle_risk_score(low_risk_test)
print(f"Risk Score: {risk_score:.3f}")
print(f"Risk Category: {risk_category}")
print(f"Explanation: {explanation}")

# Test with mixed risk factors
mixed_risk_test = pd.DataFrame([{
    'alcohol_consumption': 0.9,
    'body_weight_bmi': 0.8,
    'physical_inactivity_level': 0.7,
    'poor_diet_quality': 0.6,
    'smoking_history': 0.9,
    'environmental_exposures': 0.3,
    'hormone_use': 0.2,
    'family_history_genetic': 0.1,
    'reproductive_history': 0.2,
    'menstrual_history': 0.3
}])
print("\n--- Testing with MIXED RISK factors ---")
risk_score, risk_category, explanation = calculate_lifestyle_risk_score(mixed_risk_test)
print(f"Risk Score: {risk_score:.3f}")
print(f"Risk Category: {risk_category}")
print(f"Explanation: {explanation}")