import pandas as pd
import numpy as np
import joblib
import json
import random
import warnings
import os
import genai
import streamlit as st

# Import the necessary function and constants from the utility file
from risk_assessment_utils import assess_risk_and_plan, METADATA_FILENAME, ALL_FEATURE_NAMES, LIFESTYLE_FEATURES_NAMES, INTERACTION_FEATURES_NAMES, GEMINI_AVAILABLE, gemini_model

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

# --- Configure Gemini API Key ---
API_KEY = os.getenv("GOOGLE_API_KEY")

GEMINI_AVAILABLE = False
gemini_model = None # Initialize as None

if not API_KEY:
    print("\nWARNING: GOOGLE_API_KEY not found...")
else:
    try:
        genai.configure(api_key=API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash') # Or your chosen model
        print(f"Gemini API configured successfully using '{gemini_model.model_name}' model.")
        GEMINI_AVAILABLE = True
    except Exception as e:
        print(f"Error configuring Gemini API...")
        gemini_model = None # Ensure it's None on error

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

# ---- ADD EXPORT IF NEEDED ----
# If gemini_model wasn't automatically module-level, ensure it is:
# (Your current code seems okay, but this is a safeguard)
__all__ = [
    'assess_risk_and_plan',
    'PreventionPlanGenerator',
    'GEMINI_AVAILABLE',
    'gemini_model', # Make sure it's exported if needed by import *
    'METADATA_FILENAME',
    # ... other constants/functions you import elsewhere ...
]

# --- Streamlit App Layout ---
st.set_page_config(layout="wide")
st.title("Holistic Breast Cancer Risk Assessment & AI Planner")
st.markdown("""
Welcome! Please provide your details in the sidebar to receive a risk assessment
and a personalized lifestyle plan suggestion. An AI Chatbot will be available below
after generating a plan for general questions about breast cancer.
""")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("Input Patient Data")
    # ... (Your existing sliders and expander for medical/lifestyle inputs) ...
    assess_button = st.button("Assess Risk and Generate Plan", key="assess_button")

# --- Main Area ---
# Initialize placeholders for results if needed, or handle scope carefully
assessment_result = None
prevention_plan = None

if assess_button:
    # Prepare input_df from sidebar state
    # ... (Your existing logic to collect inputs into client_lifestyle_data, client_medical_data) ...
    # ... (Your existing logic to calculate interaction features) ...
    # ... (Your existing logic to combine data into input_features_df) ...

    try:
        assessment_result, prevention_plan = assess_risk_and_plan(input_features_df)
        # Store results in session state to persist them
        st.session_state.assessment_result = assessment_result
        st.session_state.prevention_plan = prevention_plan
        st.session_state.show_results = True # Flag to show results area
        # Clear previous chat state when new assessment is run
        if "chat_session" in st.session_state:
            del st.session_state.chat_session
        if "messages" in st.session_state:
            del st.session_state.messages

    except FileNotFoundError as e:
        st.error(f"Error loading model artifacts: {e}. Please ensure training script has been run.")
        st.session_state.show_results = False
    except KeyError as e:
        st.error(f"Error accessing expected data: {e}. Check input data or metadata file.")
        st.session_state.show_results = False
    except Exception as e:
        st.error(f"An unexpected error occurred during assessment: {e}")
        st.session_state.show_results = False

# --- Display Assessment Results and Chatbot (Only if button was clicked and results exist) ---
# Use session state to control visibility
if st.session_state.get('show_results', False):
    assessment_result = st.session_state.get('assessment_result')
    prevention_plan = st.session_state.get('prevention_plan')

    if assessment_result and prevention_plan:
        st.subheader("Risk Assessment Result")
        risk_score = assessment_result.get('risk_score', 'N/A')
        risk_cat = assessment_result.get('risk_category', 'N/A')
        st.metric(label="Predicted Risk Score", value=f"{risk_score:.3f}",
                  help="Score ranges from 0 (low risk) to 1 (high risk)")
        st.markdown(f"**Risk Category:** {risk_cat}")
        st.markdown(assessment_result.get('explanation', ''))

        st.subheader("Personalized Prevention Plan Suggestion")
        # ... (Your existing display logic for key factors, recommendations, timeline, monitoring) ...
        # Example for displaying factors:
        st.markdown("**Key Lifestyle Factors for Improvement:**")
        factors = prevention_plan.get('key_lifestyle_factors_for_improvement', [])
        if factors:
             st.markdown(f"`{', '.join(factors)}`")
        else:
             st.markdown("_None identified based on input._")
        # (Add similar display for other plan parts)


        # --- Add Separator ---
        st.divider()

        # --- AI Chatbot Section (Now conditionally displayed) ---
        st.subheader("AI Chatbot") # Keep subheader outside expander

        if not GEMINI_AVAILABLE:
            st.info("💡 AI Chatbot requires setup: Add your GOOGLE_API_KEY to Streamlit Secrets to enable it.")
        else:
            # Use an expander for the chatbot
            with st.expander("🤖 Ask the AI Chatbot a Question (Beta)"):
                st.markdown("Ask general questions about breast cancer using the context of the plan above. **Note:** Informational responses only, not medical advice.")

                # --- Initialize Chat with Context ---
                # Construct system prompt context based on current results
                risk_cat_ctx = assessment_result.get('risk_category', 'N/A')
                risk_scr_ctx = assessment_result.get('risk_score', 'N/A')
                plan_factors_ctx = prevention_plan.get('key_lifestyle_factors_for_improvement', [])
                plan_recs_ctx = prevention_plan.get('personalized_recommendations', [])
                plan_timeline_ctx = prevention_plan.get('suggested_timeline', [])
                plan_monitor_ctx = prevention_plan.get('monitoring_suggestions', [])

                system_context = f"""
Here is the latest assessment for the user you are chatting with:
- Risk Category: {risk_cat_ctx}
- Risk Score: {risk_scr_ctx:.3f}
- Key Lifestyle Factors Identified for Improvement: {', '.join(plan_factors_ctx) if plan_factors_ctx else 'None'}
- Personalized Recommendations Provided: {'; '.join(plan_recs_ctx) if plan_recs_ctx else 'None'}
- Suggested Timeline: {' '.join(plan_timeline_ctx) if plan_timeline_ctx else 'None'}
- Monitoring Suggestions: {' '.join(plan_monitor_ctx) if plan_monitor_ctx else 'None'}

You are a helpful assistant knowledgeable about breast cancer. Use the provided assessment context when relevant to the user's questions. Provide informative and general answers. Do not give medical advice. Keep responses concise.
"""
                initial_history = [
                    {'role': 'user', 'parts': [system_context]},
                    {'role': 'model', 'parts': ["Okay, I understand the user's latest assessment results and plan. I will use this context when relevant and provide general information about breast cancer, avoiding medical advice and keeping responses concise."]}
                ]

                # Initialize chat session and messages in state *if not already done for this assessment*
                # Note: Clicking the button again will reset this based on the clearing logic above
                if "chat_session" not in st.session_state:
                    st.session_state.chat_session = gemini_model.start_chat(history=initial_history)
                if "messages" not in st.session_state:
                    st.session_state.messages = initial_history # Start display history with context

                # --- Display Chat History ---
                # Skip displaying the initial system prompt/ack in the chat window for cleaner UI
                for message in st.session_state.messages:
                    if message['role'] == 'user' and message['parts'][0].startswith("Here is the latest assessment"):
                        continue # Don't show the long system context prompt
                    if message['role'] == 'model' and message['parts'][0].startswith("Okay, I understand"):
                         continue # Don't show the initial model acknowledgement
                    with st.chat_message(message["role"]):
                         st.markdown(message["parts"][0]) # Access parts correctly

                # --- Handle Chat Input ---
                if prompt := st.chat_input("Ask about the plan or breast cancer..."):
                    # Add user message to session state messages for display
                    st.session_state.messages.append({"role": "user", "parts": [prompt]})
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    # Generate and display AI response
                    with st.chat_message("assistant"):
                        message_placeholder = st.empty()
                        full_response = ""
                        try:
                            with st.spinner("Thinking..."):
                                # Send message to the existing chat session
                                response = st.session_state.chat_session.send_message(prompt)
                                full_response = response.text
                                message_placeholder.markdown(full_response)
                                # Add AI response to session state messages for display
                                st.session_state.messages.append({"role": "assistant", "parts": [full_response]}) # Store parts correctly
                        except Exception as e:
                            full_response = f"Sorry, an error occurred: {e}"
                            message_placeholder.error(full_response)
                            # Optionally add error message to history
                            # st.session_state.messages.append({"role": "assistant", "parts": [full_response]})

    else:
         # Handle case where button clicked but results failed to generate (covered by exceptions)
         st.warning("Could not display results or chatbot due to errors during assessment.")

# --- SHAP Plot Display (Placed outside the main conditional block, in the sidebar) ---
st.sidebar.markdown("---") # Add a separator in the sidebar
st.sidebar.subheader("Model Interpretability") # Add a subheader in the sidebar
shap_image_path = "shap_summary_plot.png"
if os.path.exists(shap_image_path):
    # Use a button in the sidebar to trigger the display
    # Ensure a unique key for the button
    if st.sidebar.button("SHOW FEATURE IMPORTANCE PLOT (SHAP)", key="shap_button"):
        st.subheader("SHAP Feature Importance Summary") # Display title in the main area
        st.image(
            shap_image_path,
            caption="SHAP summary plot from model training",
            use_container_width=True # UPDATED PARAMETER
        ) # Display image in the main area
else:
    # Inform the user if the plot file is missing
    st.sidebar.caption("SHAP plot image not found. Run training script or ensure file is deployed.")
