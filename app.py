import streamlit as st
import pandas as pd
import numpy as np
import json
import os

# Import the assessment function and constants from our utility file
from risk_assessment_utils import assess_risk_and_plan, METADATA_FILENAME, LIFESTYLE_FEATURES_NAMES

# --- Page Configuration ---
st.set_page_config(
    page_title="Holistic Breast Cancer Risk Assessment",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Holistic Breast Cancer Risk Assessment & Lifestyle Planner")
st.write("Enter patient details to assess risk and generate a personalized prevention plan.")

# --- Load Metadata --- 
# Needed to get the list of all expected features for the form
def load_metadata():
    if not os.path.exists(METADATA_FILENAME):
        st.error(f"Metadata file '{METADATA_FILENAME}' not found. Please run the training script first.")
        return None
    try:
        with open(METADATA_FILENAME, 'r') as f:
            metadata = json.load(f)
        if 'all_features' not in metadata:
             st.error(f"Metadata file '{METADATA_FILENAME}' is missing the required 'all_features' key.")
             return None
        return metadata
    except Exception as e:
        st.error(f"Error loading or parsing metadata file: {e}")
        return None

metadata = load_metadata()

# --- Input Form --- 
if metadata:
    all_feature_names = metadata['all_features']
    # Separate features for better UI organization
    original_feature_names = metadata.get('original_features', [f for f in all_feature_names if '_interaction' not in f and f not in LIFESTYLE_FEATURES_NAMES]) # Derive if needed
    lifestyle_feature_names = LIFESTYLE_FEATURES_NAMES
    interaction_feature_names = metadata.get('interaction_features', [f for f in all_feature_names if '_interaction' in f]) # Derive if needed

    st.sidebar.header("Client Input Data")
    input_data = {}

    # --- Lifestyle Inputs (Sliders are good for 0-1 scale) ---
    st.sidebar.subheader("Lifestyle Factors (0-1 Scale)")
    input_data['physical_activity'] = st.sidebar.slider("Physical Activity Score (0=Low, 1=High)", 0.0, 1.0, 0.5, 0.05)
    input_data['diet_quality'] = st.sidebar.slider("Diet Quality Score (0=Poor, 1=Excellent)", 0.0, 1.0, 0.5, 0.05)
    input_data['stress_level'] = st.sidebar.slider("Stress Level Score (0=Low, 1=High)", 0.0, 1.0, 0.5, 0.05)
    input_data['sleep_quality'] = st.sidebar.slider("Sleep Quality Score (0=Poor, 1=Good)", 0.0, 1.0, 0.5, 0.05)
    input_data['alcohol_consumption'] = st.sidebar.slider("Alcohol Consumption Score (0=None, 1=High)", 0.0, 1.0, 0.3, 0.05)
    input_data['smoking_history'] = st.sidebar.slider("Smoking History Score (0=None, 1=Heavy/Current)", 0.0, 1.0, 0.1, 0.05)

    # --- Medical Inputs (Using number_input) ---
    st.sidebar.subheader("Medical Features (Enter clinical values)")
    # Use placeholder values for now - ideally load means/medians from training data
    # Creating 30 input fields manually is verbose. Let's group them or use expanders.
    with st.sidebar.expander("Enter all 30 Medical Feature Values"):
        # Example placeholders - replace with more realistic defaults if possible
        default_medical_values = {
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
        for feature in original_feature_names:
            # Use default values if available, otherwise 0.0
            default_val = float(default_medical_values.get(feature, 0.0))
            input_data[feature] = st.number_input(f"{feature}", value=default_val, format="%.3f")

    # Calculate Interaction features based on inputs
    try:
        input_data['activity_immune_interaction'] = input_data['physical_activity'] * input_data['worst concave points']
        input_data['diet_cell_interaction'] = input_data['diet_quality'] * input_data['mean texture']
        input_data['stress_immune_interaction'] = (1 - input_data['stress_level']) * input_data['mean smoothness']
    except KeyError as e:
        st.error(f"Error: Could not calculate interaction features. Missing input for: {e}")
        # Handle missing key error appropriately, maybe disable button
        interaction_error = True
    else:
        interaction_error = False

    # --- Run Button --- 
    st.sidebar.markdown("---")
    run_button = st.sidebar.button("Assess Risk and Generate Plan", disabled=interaction_error)
    st.sidebar.markdown("---")

    # --- Display Area ---
    st.subheader("Assessment Results")
    results_placeholder = st.empty() # Placeholder to show results or messages

    if run_button:
        results_placeholder.info("Processing... Please wait.")
        # Prepare DataFrame in the correct order (handled inside assess_risk_and_plan now)
        client_input_df = pd.DataFrame([input_data])

        # Run assessment
        assessment, plan = assess_risk_and_plan(client_input_df)

        if assessment and plan:
            results_placeholder.empty() # Clear the processing message

            # Display Assessment
            st.markdown("#### Risk Assessment")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Predicted Risk Score", value=f"{assessment['risk_score']:.3f}")
            with col2:
                risk_cat = assessment['risk_category']
                if risk_cat == 'High': color = "error"
                elif risk_cat == 'Medium': color = "warning"
                else: color = "success"
                st.markdown(f"**Risk Category:** <span style='padding: 5px; border-radius: 5px; background-color:{''.join(filter(str.isalpha, color))}; color:white;' >{risk_cat}</span>", unsafe_allow_html=True)

            st.caption(assessment['explanation'])
            st.markdown("---")

            # Display Plan
            st.markdown("#### Personalized Prevention Plan")

            st.markdown("**Key Lifestyle Factors Identified for Improvement:**")
            if plan['key_lifestyle_factors_for_improvement']:
                for factor in plan['key_lifestyle_factors_for_improvement']:
                    st.markdown(f"- {factor.replace('_', ' ').title()}")
            else:
                st.success("Current lifestyle factors appear within healthy ranges based on input.")

            st.markdown("**Personalized Recommendations:**")
            for rec in plan['personalized_recommendations']:
                st.markdown(f"- {rec}") # Recommendations now include context and markdown links

            with st.expander("Suggested Timeline & Monitoring"):
                st.markdown("**Suggested Timeline:**")
                for item in plan['suggested_timeline']:
                    st.markdown(f"- {item}")
                st.markdown("**Monitoring Suggestions:**")
                for item in plan['monitoring_suggestions']:
                    st.markdown(f"- {item}")

        else:
            results_placeholder.error("Could not generate assessment and plan. Check terminal logs for details.")
    else:
         results_placeholder.info("Enter client data in the sidebar and click the button to run the assessment.")

else:
    st.warning("Could not load model metadata. Ensure the main training script has been run successfully.")

# --- Optional: Display SHAP Plot ---
st.sidebar.markdown("---")
st.sidebar.subheader("Model Interpretability")
shap_image_path = "shap_summary_plot.png"
if os.path.exists(shap_image_path):
    if st.sidebar.button("Show Feature Importance Plot (SHAP)"):
        st.subheader("SHAP Feature Importance Summary")
        st.image(shap_image_path, caption="SHAP summary plot from model training")
else:
    st.sidebar.caption("SHAP plot not found. Run training script.") 