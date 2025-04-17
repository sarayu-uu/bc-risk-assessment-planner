import pandas as pd
import numpy as np
import joblib
import json
import random
import warnings
from sklearn.exceptions import NotFittedError
import google.generativeai as genai # Import Gemini library
import os # To read API key from environment
from dotenv import load_dotenv # Import dotenv

# --- Load Environment Variables --- 
load_dotenv() # Load variables from .env file into environment

# --- Constants (Centralized) ---
LIFESTYLE_FEATURES_NAMES = [
    'physical_activity', 'diet_quality', 'stress_level',
    'sleep_quality', 'alcohol_consumption', 'smoking_history'
]
INTERACTION_FEATURES_NAMES = [
    'activity_immune_interaction', 'diet_cell_interaction', 'stress_immune_interaction'
]
MODEL_FILENAME = 'breast_cancer_lifestyle_model.pkl'
SCALER_FILENAME = 'lifestyle_scaler.pkl'
SELECTOR_FILENAME = 'lifestyle_feature_selector.pkl'
METADATA_FILENAME = 'lifestyle_model_metadata.json'

# --- Configure Gemini API Key --- 
API_KEY = os.getenv("GOOGLE_API_KEY") # Read from environment

GEMINI_AVAILABLE = False # Default to False
if not API_KEY:
    print("\nWARNING: GOOGLE_API_KEY not found in environment variables or .env file. Dynamic plan generation will be disabled.\n")
else:
    try:
        genai.configure(api_key=API_KEY)
        # Initialize the Gemini model
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        print("Gemini API configured successfully using 'gemini-1.5-flash' model.")
        GEMINI_AVAILABLE = True
    except Exception as e:
        print(f"Error configuring Gemini API from environment variable: {e}. Dynamic plan generation will be disabled.")

# --- Enhanced Prevention Plan Generator with Dynamic Elements ---
class PreventionPlanGenerator:
    # Removed docstring to fix persistent SyntaxError
    def __init__(self):
        # Recommendations mapped to lifestyle features needing improvement
        # Removed unnecessary backslash escapes from quotes
        self.recommendations = {
            'physical_activity': {
                'low': [
                    "Aim for 150 mins of moderate aerobic activity weekly (e.g., brisk walking). Resource: [WHO Guidelines](https://www.who.int/news-room/fact-sheets/detail/physical-activity)",
                    "Incorporate strength training focusing on major muscle groups twice a week. Resource: [ACSM Recommendations](https://www.acsm.org/education-resources/trending-topics-resources/physical-activity-guidelines)"
                ],
                'very_low': [
                    "Start gradually: Aim for 10-15 minutes of walking daily and slowly increase duration.",
                    "Break up long periods of sitting with short walks or stretches every 30-60 minutes.",
                    "Find activities you enjoy to ensure long-term adherence (e.g., swimming, dancing, gardening)."
                ]
            },
            'diet_quality': {
                 'low': [
                    "Increase daily intake of fruits and vegetables (aim for 5-9 servings). Resource: [ChooseMyPlate](https://www.myplate.gov/)",
                    "Choose whole grains (oats, quinoa, brown rice) over refined grains.",
                    "Include lean protein sources (fish, poultry, beans, lentils)."
                 ],
                 'very_low': [
                     "Focus on adding one extra serving of vegetables to your main meals daily.",
                     "Swap sugary drinks for water or herbal tea.",
                     "Limit processed foods, sugary drinks, and red meat consumption. Resource: [Dietary Guidelines for Americans](https://www.dietaryguidelines.gov/)"
                 ]
            },
            'stress_level': { # High score = high stress (bad)
                 'high': [
                    "Practice daily mindfulness or meditation (start with 5-10 minutes). Apps like Headspace or Calm can help.",
                    "Incorporate regular relaxation techniques like deep breathing exercises or progressive muscle relaxation.",
                    "Ensure adequate social support and connection with friends or family."
                 ],
                 'very_high': [
                     "Identify major stressors and develop specific coping strategies.",
                     "Consider professional help like therapy or counseling if stress is overwhelming. Resource: [NIMH - Stress](https://www.nimh.nih.gov/health/topics/stress)",
                     "Prioritize regular sleep and physical activity, as they significantly impact stress resilience."
                 ]
            },
            'sleep_quality': {
                 'low': [
                    "Aim for 7-9 hours of quality sleep per night.",
                    "Maintain a consistent sleep schedule, going to bed and waking up around the same time daily.",
                    "Create a relaxing bedtime routine (e.g., reading, warm bath, avoiding screens)."
                 ],
                 'very_low': [
                     "Optimize your sleep environment: ensure it's dark, quiet, and cool. Use blackout curtains or earplugs if needed.",
                     "Avoid caffeine and heavy meals several hours before bedtime.",
                     "Consult a doctor if sleep problems persist. Resource: [Sleep Foundation](https://www.sleepfoundation.org/)"
                 ]
            },
            'alcohol_consumption': { # High score = high consumption (bad)
                 'high': [
                    "Limit alcohol intake according to guidelines (<=1 drink/day for women). Resource: [NIAAA Rethinking Drinking](https://www.rethinkingdrinking.niaaa.nih.gov/)",
                    "Incorporate several alcohol-free days each week.",
                    "Avoid drinking alcohol on an empty stomach."
                 ],
                 'very_high': [
                     "Track your alcohol consumption to understand patterns.",
                     "Identify triggers for drinking and find alternative coping mechanisms.",
                     "Seek support from a healthcare provider or support groups if reducing intake is difficult."
                 ]
            },
            'smoking_history': { # High score = more smoking (bad)
                 'high': [ # Assumes current or recent smoker
                    "Quitting smoking is the single best step for overall health. Resource: [Smokefree.gov](https://smokefree.gov/)",
                    "Explore smoking cessation resources (e.g., counseling, nicotine replacement therapy, medication).",
                    "Set a quit date and create a support system (friends, family, support groups)."
                 ],
                 'very_high': [ # Assumes heavier/longer smoking history
                    "Discuss cessation options with your healthcare provider.",
                    "Identify triggers and develop strong strategies to manage cravings.",
                    "Focus on the immediate and long-term health benefits of quitting."
                 ]
            }
        }
        self.lifestyle_feature_names = LIFESTYLE_FEATURES_NAMES

    def generate_plan(self, input_features_df, risk_score):
        plan = {
            'risk_score': round(risk_score, 3),
            'risk_category': 'High' if risk_score >= 0.7 else 'Medium' if risk_score >= 0.3 else 'Low',
            'key_lifestyle_factors_for_improvement': [],
            'personalized_recommendations': [],
            # Timeline and Monitoring will be generated dynamically
            'suggested_timeline': [],
            'monitoring_suggestions': []
        }
        risk_factors_to_improve = {}
        input_features = input_features_df.iloc[0]
        added_recs = set()
        improvement_details = [] # Store details for prompt

        for factor in self.lifestyle_feature_names:
            value = input_features[factor]
            level = None # e.g., 'low', 'very_low', 'high', 'very_high'
            if factor in ['stress_level', 'alcohol_consumption', 'smoking_history']:
                if value > 0.8: level = 'very_high'
                elif value > 0.6: level = 'high'
            else:
                if value < 0.2: level = 'very_low'
                elif value < 0.4: level = 'low'

            if level:
                 risk_factors_to_improve[factor] = level
                 improvement_details.append(f"{factor.replace('_', ' ').title()} ({level.replace('_', ' ').title()} level)")
                 if factor in self.recommendations and level in self.recommendations[factor]:
                     available_recs = [rec for rec in self.recommendations[factor][level] if rec not in added_recs]
                     num_to_sample = min(len(available_recs), random.randint(1, 2))
                     if num_to_sample > 0:
                         recs = random.sample(available_recs, num_to_sample)
                         plan['personalized_recommendations'].extend([f"({factor.replace('_', ' ').title()} - {level.replace('_', ' ').title()}): {rec}" for rec in recs])
                         added_recs.update(recs)

        plan['key_lifestyle_factors_for_improvement'] = list(risk_factors_to_improve.keys())

        if not plan['personalized_recommendations']:
             plan['personalized_recommendations'].append("Overall lifestyle factors appear within healthy ranges based on input. Maintain current habits and continue regular check-ups.")

        # --- Generate Dynamic Timeline & Monitoring using Gemini --- 
        if GEMINI_AVAILABLE:
            try:
                # print("Attempting to generate dynamic plan elements with Gemini...") # Removed debug print
                # Using standard triple quotes for the prompt string
                prompt = f'''
                Generate a realistic, encouraging, and step-by-step health improvement plan for a patient.
                Patient Context:
                - Overall Cancer Risk Category: {plan['risk_category']}
                - Key Lifestyle Factors Identified for Improvement: {(', '.join(improvement_details) if improvement_details else 'None - Maintain current habits')}
                - Specific Recommendations Provided: {'; '.join(plan['personalized_recommendations'])}
                
                Task:
                1. Create a 'Suggested Timeline' with 3-5 actionable steps spread over approximately 3-6 months, focusing on the key factors and recommendations.
                2. Create 'Monitoring Suggestions' with 3-5 practical ways the patient can track their progress relevant to the recommendations.
                
                Output Format:
                Provide the timeline and monitoring suggestions as separate markdown bulleted lists. Start each list immediately after the labels "Suggested Timeline:" and "Monitoring Suggestions:" respectively. Do not add any extra introductory or concluding text.
                
                Example:
                Suggested Timeline:
                * Month 1: Focus on [Action 1 related to recommendation].
                * Month 1-2: Gradually implement [Action 2 related to recommendation].
                * Month 3: Review progress on [Action 1 & 2] and start [Action 3].
                
                Monitoring Suggestions:
                * Use a journal or app to track [Metric 1 related to recommendation] daily.
                * Schedule a check-in with your healthcare provider after [Timeframe].
                * Reflect weekly on [Qualitative aspect related to recommendation].
                '''
                
                response = gemini_model.generate_content(prompt)
                generated_text = response.text
                # print(f"--- Gemini Raw Response ---") # Removed debug print
                # print(generated_text) # Removed debug print
                # print(f"--- End Gemini Raw Response ---") # Removed debug print
                
                # Parse the response (simple parsing based on expected headers)
                timeline_items = []
                monitoring_items = []
                current_section = None
                # Use splitlines() for more robust line splitting
                for line in generated_text.splitlines(): 
                    line_stripped = line.strip()
                    if line_stripped.startswith("Suggested Timeline:"):
                        current_section = 'timeline'
                        continue
                    elif line_stripped.startswith("Monitoring Suggestions:"):
                        current_section = 'monitoring'
                        continue
                    
                    if current_section == 'timeline' and line_stripped.startswith( ('* ', '- ') ):
                        timeline_items.append(line_stripped[2:].strip()) # Remove bullet
                    elif current_section == 'monitoring' and line_stripped.startswith( ('* ', '- ') ):
                        monitoring_items.append(line_stripped[2:].strip())
                
                if timeline_items:
                    # print("Successfully parsed timeline from Gemini response.") # Removed debug print
                    plan['suggested_timeline'] = timeline_items
                else:
                    # print("Could not parse timeline from Gemini response. Using fallback.") # Removed debug print
                    # Restore original fallback if parsing fails but API call succeeded
                    plan['suggested_timeline'] = [ # RESTORED HERE
                        "Month 1-2: Focus on incorporating one small change consistently.",
                        "Month 3-4: Gradually increase effort or add a second goal.",
                        "Month 5-6: Evaluate progress and adjust goals.",
                        "Ongoing: Maintain habits and regular check-ins."
                    ]

                if monitoring_items:
                    # print("Successfully parsed monitoring suggestions from Gemini response.") # Removed debug print
                    plan['monitoring_suggestions'] = monitoring_items
                else:
                    # print("Could not parse monitoring suggestions from Gemini response. Using fallback.") # Removed debug print
                    # Restore original fallback if parsing fails but API call succeeded
                    plan['monitoring_suggestions'] = [ # RESTORED HERE
                        "Keep a journal or use a habit tracker.",
                        "Check in on how you feel regularly.",
                        "Schedule necessary health screenings.",
                        "Discuss progress with healthcare provider."
                    ]
                
            except Exception as e:
                # print(f"ERROR calling Gemini API or parsing response: {e}") # Removed debug print
                print(f"Warning: Error during Gemini API call or parsing: {e}. Using static fallback plan elements.") # Keep a warning
                # print("Falling back to 'AI not working' indicator.") # Removed debug print
                # Restore original fallback in case of API error
                plan['suggested_timeline'] = [ # RESTORED HERE
                    "Month 1-2: Focus on incorporating one small change consistently.",
                    "Month 3-4: Gradually increase effort or add a second goal.",
                    "Month 5-6: Evaluate progress and adjust goals.",
                    "Ongoing: Maintain habits and regular check-ins."
                ]
                plan['monitoring_suggestions'] = [ # RESTORED HERE
                    "Keep a journal or use a habit tracker.",
                    "Check in on how you feel regularly.",
                    "Schedule necessary health screenings.",
                    "Discuss progress with healthcare provider."
                ]
        else:
             # print("Gemini API not available. Using fallback indicator.") # Removed debug print
             # Restore original fallback if API is not available
             plan['suggested_timeline'] = [ # RESTORED HERE
                "Month 1-2: Focus on incorporating one small change consistently.",
                "Month 3-4: Gradually increase effort or add a second goal.",
                "Month 5-6: Evaluate progress and adjust goals.",
                "Ongoing: Maintain habits and regular check-ins."
            ]
             plan['monitoring_suggestions'] = [ # RESTORED HERE
                "Keep a journal or use a habit tracker.",
                "Check in on how you feel regularly.",
                "Schedule necessary health screenings.",
                "Discuss progress with healthcare provider."
            ]

        # print("\n--- Prevention Plan Generated ---") # Removed debug print
        return plan

# --- Risk Assessment Function (Uses loaded components) ---
def assess_risk_and_plan(input_features_df):
    # Removed docstring to fix persistent SyntaxError
    try:
        # Load saved components
        model = joblib.load(MODEL_FILENAME)
        scaler = joblib.load(SCALER_FILENAME)
        selector = joblib.load(SELECTOR_FILENAME)
        print("Loaded model, scaler, and selector.")
    except FileNotFoundError:
        print("Error: Model/Scaler/Selector files not found. Please train the model first.")
        return None, None
    except Exception as e:
        print(f"Error loading model components: {e}")
        return None, None

    # Load metadata to get expected feature order for scaler
    try:
        with open(METADATA_FILENAME, 'r') as f:
            metadata = json.load(f)
        expected_features = metadata['all_features']
    except FileNotFoundError:
        print(f"Error: Metadata file '{METADATA_FILENAME}' not found.")
        return None, None
    except KeyError:
        print(f"Error: Metadata file '{METADATA_FILENAME}' is missing 'all_features' key.")
        return None, None
    except Exception as e:
        print(f"Error reading metadata file: {e}")
        return None, None

    # Ensure input DataFrame has the correct columns in the right order
    try:
        input_features_df_ordered = input_features_df[expected_features]
    except KeyError as e:
         print(f"Error: Input data missing features required by the scaler: {e}")
         print(f"Required features: {expected_features}")
         return None, None
    except Exception as e:
        print(f"Error ordering input features: {e}")
        return None, None

    # Preprocess features: Scale -> Select
    try:
        features_scaled = scaler.transform(input_features_df_ordered)
        features_selected = selector.transform(features_scaled)
        print("Input features preprocessed.")
    except NotFittedError as e:
         print(f"Error during transform (scaler or selector not fitted?): {e}")
         return None, None
    except ValueError as e:
         print(f"Error during transform (likely feature mismatch): {e}")
         return None, None
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None, None


    # Get prediction probability (risk score)
    try:
        risk_score = model.predict_proba(features_selected)[:, 1][0]
        print(f"Calculated risk score: {risk_score:.3f}")
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None

    # Generate prevention plan using the original input features (before scaling/selection)
    try:
        prevention_planner = PreventionPlanGenerator() # Instantiate planner
        # Pass the original unordered df is fine, planner uses specific column names
        prevention_plan = prevention_planner.generate_plan(input_features_df, risk_score)
    except Exception as e:
        print(f"Error generating prevention plan: {e}")
        # Return assessment even if plan fails
        assessment_result = {
            'risk_score': round(risk_score, 3),
            'risk_category': 'Unknown',
            'explanation': f"Risk score based on analysis of {features_selected.shape[1]} selected features. Plan generation failed."
        }
        return assessment_result, None


    assessment_result = {
        'risk_score': round(risk_score, 3),
        'risk_category': prevention_plan['risk_category'], # Get category from plan
        'explanation': f"Risk score based on analysis of {features_selected.shape[1]} selected features."
    }

    return assessment_result, prevention_plan 