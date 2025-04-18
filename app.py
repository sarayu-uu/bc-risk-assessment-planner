import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import random # Ensure random is imported

# Import the assessment function and constants from our utility files
from risk_assessment_utils import (
    assess_risk_and_plan, 
    METADATA_FILENAME, 
    LIFESTYLE_FEATURES_NAMES,
    # INTERACTION_FEATURES_NAMES, # Not used directly in app.py it seems
    GEMINI_AVAILABLE,
    gemini_model
)

# --- Page Configuration ---
st.set_page_config(layout="wide")
st.title("Holistic Breast Cancer Risk Assessment & AI Planner")
st.markdown("""
Welcome! Please provide your details in the sidebar to receive a risk assessment
and a personalized lifestyle plan suggestion. An AI Chatbot will be available below
after generating a plan for general questions about breast cancer.
""")

# --- Load Metadata and Constants ---
# Assuming METADATA_FILENAME is defined in risk_assessment_utils or here
METADATA_PATH = os.path.join(os.path.dirname(__file__), METADATA_FILENAME)
ALL_FEATURE_NAMES = []
EXPECTED_FEATURES = [] # Define or load expected features needed for DataFrame creation
try:
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
        # Load all features seen during training for default values/input generation
        ALL_FEATURE_NAMES = metadata.get('all_features', []) 
        # Load the feature list in the exact order expected by the scaler/selector/model
        # This might be all_features before selection, or selected_features after selection
        EXPECTED_FEATURES = metadata.get('features_after_selection', metadata.get('all_features', [])) # Adjust key based on your metadata file
        if not EXPECTED_FEATURES:
             st.sidebar.error("Could not determine expected feature order from metadata. Assessment might fail.")
             EXPECTED_FEATURES = ALL_FEATURE_NAMES # Fallback, might be wrong order
        print(f"Loaded {len(ALL_FEATURE_NAMES)} total features, expecting {len(EXPECTED_FEATURES)} ordered features for model.") # Debug print
except FileNotFoundError:
    st.sidebar.error(f"Metadata file ({METADATA_FILENAME}) not found at {METADATA_PATH}. Training script needs to be run.")
    st.stop()
except json.JSONDecodeError:
    st.sidebar.error(f"Error reading metadata file ({METADATA_FILENAME}). Ensure it's valid JSON.")
    st.stop()
except Exception as e:
    st.sidebar.error(f"An unexpected error occurred loading metadata: {e}")
    st.stop()

if not ALL_FEATURE_NAMES:
     st.sidebar.warning("Could not load feature names from metadata. Inputs may be incomplete.")

# Separate original medical features (assuming they are the ones NOT lifestyle/interaction)
# Use ALL_FEATURE_NAMES for generating the input fields
ORIGINAL_MEDICAL_FEATURES = [f for f in ALL_FEATURE_NAMES if f not in LIFESTYLE_FEATURES_NAMES and not f.endswith('_interaction')]
# Define plausible defaults (replace with actual defaults if needed)
DEFAULT_MEDICAL_VALUES = {feat: 0.5 for feat in ORIGINAL_MEDICAL_FEATURES} # Example default

# --- Sidebar Inputs ---
client_lifestyle_data = {}
client_medical_data = {}
with st.sidebar:
    st.header("Input Patient Data")
    st.subheader("Lifestyle Factors")
    # Ensure LIFESTYLE_FEATURES_NAMES is defined correctly above or in utils
    for feature in LIFESTYLE_FEATURES_NAMES:
        # Use a unique key for each slider based on the feature name
        client_lifestyle_data[feature] = st.slider(f"{feature.replace('_', ' ').title()}", 0.0, 1.0, 0.5, 0.05, key=f"slider_{feature}")

    with st.expander("Medical Features (Defaults Provided)"):
        st.caption(f"Input {len(ORIGINAL_MEDICAL_FEATURES)} medical features. Using defaults initially.")
        for feature in ORIGINAL_MEDICAL_FEATURES:
            default_val = DEFAULT_MEDICAL_VALUES.get(feature, 0.0) # Use defined default
            # Use a unique key for each number input
            client_medical_data[feature] = st.number_input(f"{feature.replace('_', ' ').title()}", value=default_val, step=0.01, format="%.3f", key=f"num_{feature}")

    assess_button = st.button("Assess Risk and Generate Plan", key="assess_button") # Added key

# --- Main Area ---
# Initialize placeholders in session state if they don't exist
if 'show_results' not in st.session_state:
    st.session_state.show_results = False
if 'assessment_result' not in st.session_state:
    st.session_state.assessment_result = None
if 'prevention_plan' not in st.session_state:
    st.session_state.prevention_plan = None

if assess_button:
    print("Assess button clicked") # Debug print
    # Prepare input_df from sidebar state
    combined_input_data = {**client_lifestyle_data, **client_medical_data}

    # --- Calculate Interaction Features --- (Ensure names match utils)
    # These names MUST match those expected by the model/pipeline
    interaction_feature_names_expected = [f for f in EXPECTED_FEATURES if f.endswith('_interaction')]
    INTERACTION_FEATURES_DICT = {}
    try:
        # Example calculations - adjust based on your actual definitions in training/utils
        # Ensure the base features used here exist in combined_input_data
        INTERACTION_FEATURES_DICT['activity_immune_interaction'] = combined_input_data['physical_activity'] * combined_input_data['worst concave points']
        INTERACTION_FEATURES_DICT['diet_cell_interaction'] = combined_input_data['diet_quality'] * combined_input_data['mean texture']
        INTERACTION_FEATURES_DICT['stress_immune_interaction'] = (1 - combined_input_data['stress_level']) * combined_input_data['mean smoothness']
        
        # Verify all expected interaction features were calculated
        if set(INTERACTION_FEATURES_DICT.keys()) != set(interaction_feature_names_expected):
             print(f"Warning: Calculated interactions {list(INTERACTION_FEATURES_DICT.keys())} don't match expected {interaction_feature_names_expected}")
             # Handle mismatch if necessary, maybe raise error or use defaults

    except KeyError as e:
        st.error(f"Error calculating interaction features. Missing base feature: {e}. Cannot proceed.")
        combined_input_data = None # Signal error
    except Exception as e:
         st.error(f"Unexpected error calculating interaction features: {e}")
         combined_input_data = None # Signal error

    if combined_input_data:
        # Update combined data with calculated interactions
        combined_input_data.update(INTERACTION_FEATURES_DICT)

        # Create DataFrame in the expected order
        try:
            # Create DataFrame from the dictionary
            input_features_df = pd.DataFrame([combined_input_data])
            # Reorder columns to match the exact order expected by the pipeline
            input_features_df = input_features_df[EXPECTED_FEATURES]
            print(f"Input DataFrame created with columns: {input_features_df.columns.tolist()}") # Debug print

        except KeyError as e:
            st.error(f"Mismatch creating DataFrame: Missing expected feature {e}. Check metadata or calculation logic.")
            st.session_state.show_results = False
            input_features_df = None # Prevent assess_risk_and_plan call
        except Exception as e:
            st.error(f"Error creating input DataFrame: {e}")
            st.session_state.show_results = False
            input_features_df = None


        if input_features_df is not None:
            try:
                print("Calling assess_risk_and_plan...") # Debug print
                assessment_result, prevention_plan = assess_risk_and_plan(input_features_df)
                # Store results in session state to persist them
                st.session_state.assessment_result = assessment_result
                st.session_state.prevention_plan = prevention_plan
                st.session_state.show_results = True # Flag to show results area
                print("Assessment complete. Results stored in session state.") # Debug print
                # Clear previous chat state when new assessment is run
                if "chat_session" in st.session_state:
                    del st.session_state.chat_session
                    print("Cleared chat session state.") # Debug print
                if "messages" in st.session_state:
                    del st.session_state.messages
                    print("Cleared chat messages state.") # Debug print

            except FileNotFoundError as e:
                st.error(f"Error loading model artifacts: {e}. Please ensure training script has been run and files are deployed.")
                st.session_state.show_results = False
            except KeyError as e:
                st.error(f"Error accessing expected data during assessment: {e}. Check model/scaler/selector compatibility with input features.")
                st.session_state.show_results = False
            except ValueError as e:
                 st.error(f"Data processing error during assessment: {e}. Check data types or feature scaling.")
                 st.session_state.show_results = False
            except Exception as e:
                st.error(f"An unexpected error occurred during assessment: {e}")
                st.session_state.show_results = False
                print(f"Assessment Error: {e}") # Debug print
        else:
             print("Assessment skipped due to input DataFrame error.") # Debug print
    else:
        print("Assessment skipped due to interaction feature calculation error.") # Debug print
        st.session_state.show_results = False # Ensure results aren't shown

# --- Display Assessment Results and Chatbot (Only if flag is set in session state) ---
if st.session_state.get('show_results', False):
    print("Displaying results section.") # Debug print
    # Retrieve results from session state
    assessment_result = st.session_state.get('assessment_result')
    prevention_plan = st.session_state.get('prevention_plan')

    if assessment_result and prevention_plan:
        # --- Display Assessment Result --- 
        st.subheader("Risk Assessment Result")
        risk_score = assessment_result.get('risk_score', 'N/A')
        risk_cat = assessment_result.get('risk_category', 'N/A')
        # Ensure risk_score is a number before formatting
        if isinstance(risk_score, (int, float)):
            st.metric(label="Predicted Risk Score", value=f"{risk_score:.3f}",
                      help="Score ranges from 0 (low risk) to 1 (high risk)")
        else:
             st.metric(label="Predicted Risk Score", value="N/A")
        st.markdown(f"**Risk Category:** {risk_cat}")
        st.markdown(assessment_result.get('explanation', ''))

        # --- Display Prevention Plan --- 
        st.subheader("Personalized Prevention Plan Suggestion")
        # Display key factors
        st.markdown("**Key Lifestyle Factors for Improvement:**")
        factors = prevention_plan.get('key_lifestyle_factors_for_improvement', [])
        if factors:
             st.markdown(f"`{', '.join(factors)}`")
        else:
             st.markdown("_None identified based on input._")
        
        # Display recommendations
        st.markdown("**Recommendations:**")
        recs = prevention_plan.get('personalized_recommendations', [])
        if recs:
            for rec in recs:
                st.markdown(f"- {rec}") # Using markdown for potential links
        else:
             st.markdown("_No specific recommendations generated._")

        # Display timeline
        st.markdown("**Suggested Timeline:**")
        timeline = prevention_plan.get('suggested_timeline', [])
        if timeline:
            for item in timeline:
                 st.markdown(f"- {item}")
        else:
             st.markdown("_Timeline not generated._")

        # Display monitoring
        st.markdown("**Monitoring Suggestions:**")
        monitoring = prevention_plan.get('monitoring_suggestions', [])
        if monitoring:
             for item in monitoring:
                  st.markdown(f"- {item}")
        else:
             st.markdown("_Monitoring suggestions not generated._")

        # --- Add Separator --- 
        st.divider()

        # --- AI Chatbot Section (Conditionally displayed) ---
        st.subheader("AI Chatbot") # Keep subheader outside expander

        if not GEMINI_AVAILABLE:
            st.info("💡 AI Chatbot requires setup: Add your GOOGLE_API_KEY to Streamlit Secrets to enable it.")
        else:
            print("AI Chatbot section available.") # Debug print
            # Use an expander for the chatbot
            with st.expander("🤖 Ask the AI Chatbot a Question (Beta)"):
                st.markdown("Ask general questions about breast cancer using the context of the plan above. **Note:** Informational responses only, not medical advice.")

                # --- Initialize Chat with Context ---
                # Construct system prompt context based on current results from session state
                risk_cat_ctx = assessment_result.get('risk_category', 'N/A')
                risk_scr_ctx = assessment_result.get('risk_score', 'N/A')
                plan_factors_ctx = prevention_plan.get('key_lifestyle_factors_for_improvement', [])
                plan_recs_ctx = prevention_plan.get('personalized_recommendations', [])
                plan_timeline_ctx = prevention_plan.get('suggested_timeline', [])
                plan_monitor_ctx = prevention_plan.get('monitoring_suggestions', [])
                
                # Ensure risk score is formatted correctly for the prompt
                risk_scr_str = f"{risk_scr_ctx:.3f}" if isinstance(risk_scr_ctx, (int, float)) else "N/A"

                system_context = f"""
Here is the latest assessment for the user you are chatting with:
- Risk Category: {risk_cat_ctx}
- Risk Score: {risk_scr_str}
- Key Lifestyle Factors Identified for Improvement: {', '.join(plan_factors_ctx) if plan_factors_ctx else 'None'}
- Personalized Recommendations Provided: {'; '.join(plan_recs_ctx) if plan_recs_ctx else 'None'}
- Suggested Timeline: {' '.join(plan_timeline_ctx) if plan_timeline_ctx else 'None'}
- Monitoring Suggestions: {' '.join(plan_monitor_ctx) if plan_monitor_ctx else 'None'}

You are a helpful assistant knowledgeable about breast cancer. Use the provided assessment context when relevant to the user's questions. Provide informative, detailed, and actionable suggestions based on the user's request. If asked for plans, calendars, or diet ideas related to the assessment, generate helpful examples and suggestions based on the provided context and general health principles. Frame these as AI-generated ideas for consideration.
"""
                initial_history = [
                    {'role': 'user', 'parts': [system_context]},
                    {'role': 'model', 'parts': ["Okay, I understand the user's latest assessment results and plan. I will use this context when relevant and provide helpful, actionable suggestions and examples as requested."]}
                ]

                # Initialize chat session and messages in state *if not already done for this assessment*
                # Note: Clicking the button again will reset this based on the clearing logic above
                if "chat_session" not in st.session_state:
                    print("Initializing chat session with context.") # Debug print
                    # Ensure gemini_model is not None before calling start_chat
                    if gemini_model:
                        st.session_state.chat_session = gemini_model.start_chat(history=initial_history)
                    else:
                         st.warning("Gemini model not available for chat.")
                         # Optionally hide the input if model is None
                if "messages" not in st.session_state:
                    print("Initializing chat messages with context.") # Debug print
                    st.session_state.messages = initial_history # Start display history with context

                # --- Display Chat History --- 
                # Skip displaying the initial system prompt/ack in the chat window for cleaner UI
                # Ensure messages exist before iterating
                current_messages = st.session_state.get("messages", [])
                for message in current_messages:
                    # Safety check for message structure
                    if isinstance(message, dict) and 'role' in message and 'parts' in message and isinstance(message['parts'], list) and message['parts']:
                         display_role = message['role']
                         display_content = message['parts'][0]
                         
                         # Skip initial context messages
                         if display_role == 'user' and display_content.startswith("Here is the latest assessment"):
                             continue 
                         if display_role == 'model' and display_content.startswith("Okay, I understand"):
                              continue 
                         
                         with st.chat_message(display_role):
                              st.markdown(display_content) 
                    else:
                        print(f"Skipping invalid message format: {message}") # Debug print

                # --- Handle Chat Input --- 
                # Only show input if chat session was successfully initialized
                if "chat_session" in st.session_state and st.session_state.chat_session is not None:
                    if prompt := st.chat_input("Ask about the plan or breast cancer...", key="chat_input"):
                        print(f"User asked: {prompt}") # Debug print
                        # Add user message to session state messages for display
                        st.session_state.messages.append({"role": "user", "parts": [prompt]})
                        # Display user message immediately
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
                                    st.session_state.messages.append({"role": "assistant", "parts": [full_response]})
                                    print("Assistant responded.") # Debug print
                            except Exception as e:
                                full_response = f"Sorry, an error occurred: {e}"
                                message_placeholder.error(full_response)
                                print(f"Chatbot Error: {e}") # Debug print
                                # Add error message to history for context if needed
                                # st.session_state.messages.append({"role": "assistant", "parts": [full_response]})
                else:
                     if gemini_model:
                         st.caption("Chat session initialized. Ready for input.")
                     else:
                          st.caption("Chatbot model not available.")
    else:
         # Handle case where button clicked but results failed to generate (covered by exceptions)
         # Check if the flag was set but results are None
         if st.session_state.get('show_results', False):
            st.warning("Could not display results or chatbot due to errors during assessment.")
            print("Results/Plan object missing, skipping display.") # Debug print

# --- SHAP Plot Display (Placed outside the main conditional block, in the sidebar) ---
st.sidebar.markdown("---") # Add a separator in the sidebar
st.sidebar.subheader("Model Interpretability") # Add a subheader in the sidebar
shap_image_path = "shap_summary_plot.png"
if os.path.exists(shap_image_path):
    # Use a button in the sidebar to trigger the display
    # Ensure a unique key for the button
    if st.sidebar.button("Show Feature importance Plot (SHAP)", key="shap_button"):
        st.subheader("SHAP Feature Importance Summary") # Display title in the main area
        st.image(
            shap_image_path, 
            caption="SHAP summary plot from model training", 
            use_container_width=True
        ) # Display image in the main area
else:
    # Inform the user if the plot file is missing
    st.sidebar.caption("SHAP plot image not found. Run training script or ensure file is deployed.")