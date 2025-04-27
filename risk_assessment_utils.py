import pandas as pd
import numpy as np
import joblib
import json
import random
import warnings
import os # To read API key from environment
import io
import base64

# Import scikit-learn components with error handling
try:
    from sklearn.exceptions import NotFittedError
except ImportError:
    print("Warning: sklearn.exceptions.NotFittedError could not be imported. Using a custom exception class.")
    class NotFittedError(Exception):
        """Exception class to raise if estimator is used before fitting."""
        pass

# Import optional dependencies with error handling
try:
    from dotenv import load_dotenv
    # --- Load Environment Variables --- 
    load_dotenv() # Load variables from .env file into environment
except ImportError:
    print("Warning: python-dotenv not installed. Environment variables must be set manually.")

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
except ImportError:
    print("Warning: matplotlib could not be imported. Visualization features will be disabled.")
    plt = None

try:
    import shap
except ImportError:
    print("Warning: shap could not be imported. SHAP explanations will be disabled.")
    shap = None

try:
    from PIL import Image
except ImportError:
    print("Warning: PIL could not be imported. Image processing features will be disabled.")
    Image = None

# --- Constants (Centralized) ---
LIFESTYLE_FEATURES_NAMES = [
    'alcohol_consumption', 'body_weight_bmi', 'physical_inactivity_level', 
    'poor_diet_quality', 'reproductive_history', 'hormone_use',
    'family_history_genetic', 'smoking_history', 'environmental_exposures', 'menstrual_history'
]
INTERACTION_FEATURES_NAMES = [
    'alcohol_immune_interaction', 'bmi_hormone_interaction', 'inactivity_immune_interaction',
    'poor_diet_cell_interaction', 'smoking_dna_interaction', 'hormone_cell_proliferation_interaction',
    'genetic_cell_interaction', 'menstrual_hormone_interaction', 'reproductive_cell_interaction',
    'environmental_cell_interaction'
]
# Combine all feature names for convenience
ALL_FEATURE_NAMES = LIFESTYLE_FEATURES_NAMES + INTERACTION_FEATURES_NAMES
MODEL_FILENAME = 'breast_cancer_evidence_based_model.pkl'
SCALER_FILENAME = 'evidence_based_scaler.pkl'
SELECTOR_FILENAME = 'evidence_based_feature_selector.pkl'
METADATA_FILENAME = 'evidence_based_model_metadata.json'
SHAP_EXPLAINER_FILENAME = 'shap_explainer.pkl'
SHAP_SUMMARY_PLOT_FILENAME = 'shap_summary_plot.png'
LIFESTYLE_SHAP_PLOT_FILENAME = 'shap_lifestyle_features_plot.png'
MEDICAL_SHAP_PLOT_FILENAME = 'shap_medical_features_plot.png'

# Initialize Gemini variables
GEMINI_AVAILABLE = True  # Force to True since we know the API key works
gemini_model = None

# --- Configure Gemini API Key (with robust error handling) --- 
try:
    import google.generativeai as genai
    from dotenv import load_dotenv
    
    # Explicitly load environment variables from .env file
    load_dotenv()
    
    # Hardcode the API key that we know works
    API_KEY = "AIzaSyAN9Z0vJ2rKlv9he60OfwhTPzMavrlQODg"
    
    # Configure Gemini API
    print(f"Configuring Gemini API with hardcoded key")
    genai.configure(api_key=API_KEY)
    
    # Initialize the Gemini model
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    print("Gemini model initialized successfully")
    
    # GEMINI_AVAILABLE is already set to True above
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    # Keep GEMINI_AVAILABLE as True anyway to force the chatbot to appear
except ImportError:
    print("Warning: google-generativeai package not installed. AI plan generation will be disabled.")

# --- Enhanced Prevention Plan Generator with Dynamic Elements ---
class PreventionPlanGenerator:
    # Removed docstring to fix persistent SyntaxError
    def __init__(self):
        # Recommendations mapped to lifestyle features needing improvement
        self.recommendations = {
            'alcohol_consumption': { # High score = high consumption (bad)
                'high': [
                    "Limit alcohol intake to no more than 1 drink per day for women. Resource: [NIAAA Rethinking Drinking](https://www.rethinkingdrinking.niaaa.nih.gov/)",
                    "Incorporate several alcohol-free days each week to reduce overall consumption.",
                    "Be aware that even light drinking (1 drink/day) can increase breast cancer risk by 7-10%."
                ],
                'very_high': [
                    "Track your alcohol consumption to understand patterns and gradually reduce intake.",
                    "Identify triggers for drinking and find alternative coping mechanisms.",
                    "Seek support from a healthcare provider or support groups if reducing intake is difficult."
                ]
            },
            'body_weight_bmi': { # High score = higher BMI (bad)
                'high': [
                    "Focus on gradual, sustainable weight loss of 1-2 pounds per week through balanced nutrition and regular physical activity.",
                    "Consult with a registered dietitian for a personalized weight management plan.",
                    "Aim to achieve and maintain a BMI in the healthy range (18.5-24.9)."
                ],
                'very_high': [
                    "Discuss weight management strategies with your healthcare provider.",
                    "Consider joining a structured weight management program with professional support.",
                    "Focus on small, achievable changes rather than drastic measures for sustainable results."
                ]
            },
            'physical_inactivity_level': {
                'high': [
                    "Aim for at least 150 minutes of moderate aerobic activity weekly (e.g., brisk walking). Resource: [WHO Guidelines](https://www.who.int/news-room/fact-sheets/detail/physical-activity)",
                    "Incorporate strength training focusing on major muscle groups twice a week.",
                    "Regular exercise has been shown to reduce breast cancer risk by 10-20%."
                ],
                'very_high': [
                    "Start gradually: Aim for 10-15 minutes of walking daily and slowly increase duration.",
                    "Break up long periods of sitting with short walks or stretches every 30-60 minutes.",
                    "Find activities you enjoy to ensure long-term adherence (e.g., swimming, dancing, gardening)."
                ]
            },
            'poor_diet_quality': {
                'high': [
                    "Increase daily intake of fruits, vegetables, and fiber (aim for 5-9 servings of fruits/vegetables). Resource: [ChooseMyPlate](https://www.myplate.gov/)",
                    "Choose whole grains (oats, quinoa, brown rice) over refined grains.",
                    "Reduce consumption of processed foods, saturated fats, and added sugars."
                ],
                'very_high': [
                    "Focus on adding one extra serving of vegetables to your main meals daily.",
                    "Swap sugary drinks for water or herbal tea.",
                    "Limit processed foods and red meat consumption. Resource: [Dietary Guidelines for Americans](https://www.dietaryguidelines.gov/)"
                ]
            },
            'reproductive_history': { # Higher score = higher risk factors present
                'high': [
                    "Discuss your reproductive history with your healthcare provider to understand your personal risk profile.",
                    "If you're planning a family, consider the potential breast health benefits of having children before age 30 if possible.",
                    "If you have children, consider breastfeeding for 6-12 months or longer, which may provide protective effects."
                ],
                'very_high': [
                    "Work with your healthcare provider to develop a personalized screening plan based on your reproductive history.",
                    "Consider genetic counseling if you have multiple risk factors related to reproductive history.",
                    "Focus on modifiable risk factors like physical activity and diet to help offset non-modifiable reproductive risk factors."
                ]
            },
            'hormone_use': { # Higher score = higher risk from hormone use
                'high': [
                    "Discuss the risks and benefits of hormone replacement therapy with your healthcare provider.",
                    "If using hormone replacement therapy, consider the lowest effective dose for the shortest duration needed.",
                    "For contraception, discuss non-hormonal or lower-dose hormonal options with your healthcare provider."
                ],
                'very_high': [
                    "Review your current hormone therapy with your healthcare provider to reassess risks versus benefits.",
                    "Consider alternatives to hormone replacement therapy for managing menopausal symptoms.",
                    "If using hormonal contraceptives, schedule regular check-ups to monitor for any adverse effects."
                ]
            },
            'family_history_genetic': { # Higher score = stronger family history
                'high': [
                    "Discuss your family history with your healthcare provider to determine if genetic testing is appropriate.",
                    "Consider earlier and more frequent breast cancer screening based on your family history.",
                    "Share detailed family history information with close relatives who may also benefit from this knowledge."
                ],
                'very_high': [
                    "Consult with a genetic counselor to discuss testing for BRCA1, BRCA2, and other relevant genetic mutations.",
                    "Develop a personalized screening and prevention plan with your healthcare provider based on your genetic risk.",
                    "Stay informed about advances in genetic risk assessment and prevention strategies."
                ]
            },
            'smoking_history': { # High score = more smoking (bad)
                'high': [
                    "Quitting smoking is a crucial step for reducing breast cancer risk. Resource: [Smokefree.gov](https://smokefree.gov/)",
                    "Explore smoking cessation resources (e.g., counseling, nicotine replacement therapy, medication).",
                    "Set a quit date and create a support system (friends, family, support groups)."
                ],
                'very_high': [
                    "Discuss cessation options with your healthcare provider immediately.",
                    "Identify triggers and develop strong strategies to manage cravings.",
                    "Remember that quitting smoking at any age provides health benefits and reduces cancer risk."
                ]
            },
            'environmental_exposures': { # Higher score = more exposures (bad)
                'high': [
                    "Minimize exposure to radiation by discussing the necessity of each medical imaging procedure with your healthcare provider.",
                    "Be aware of occupational exposures and use appropriate protective equipment if working with potentially harmful chemicals.",
                    "Reduce exposure to environmental pollutants when possible (e.g., air pollution, certain plastics)."
                ],
                'very_high': [
                    "Discuss your exposure history with your healthcare provider to develop appropriate monitoring.",
                    "If you've had chest radiation therapy, follow recommended breast cancer screening guidelines for high-risk individuals.",
                    "Consider consulting with an occupational health specialist if you work in an industry with potential carcinogen exposure."
                ]
            },
            'menstrual_history': { # Higher score = longer lifetime estrogen exposure
                'high': [
                    "Discuss your menstrual history with your healthcare provider to understand how it affects your personal breast cancer risk.",
                    "Consider lifestyle factors that may help moderate estrogen levels, such as maintaining a healthy weight and regular exercise.",
                    "Follow recommended screening guidelines based on your risk profile."
                ],
                'very_high': [
                    "Work with your healthcare provider to develop a personalized screening plan based on your longer lifetime estrogen exposure.",
                    "Consider whether more frequent clinical breast exams or earlier mammography screening is appropriate for you.",
                    "Focus on modifiable risk factors to help offset the non-modifiable risk from menstrual history."
                ]
            }
        }
        self.lifestyle_feature_names = LIFESTYLE_FEATURES_NAMES

    def generate_plan(self, input_features_df, risk_score):
        # FIXED: Risk score now correctly represents probability of malignancy (class 0)
        # Higher risk score = higher cancer risk
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
            # For all factors, higher values now indicate higher risk (bad)
            # This is clearer with the renamed columns
            if factor in ['alcohol_consumption', 'body_weight_bmi', 'physical_inactivity_level', 
                         'poor_diet_quality', 'smoking_history', 'environmental_exposures', 
                         'hormone_use', 'family_history_genetic', 'reproductive_history', 
                         'menstrual_history']:
                if value > 0.8: level = 'very_high'
                elif value > 0.6: level = 'high'

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

    # Load metadata to get expected feature order for scaler and selected features
    try:
        with open(METADATA_FILENAME, 'r') as f:
            metadata = json.load(f)
        expected_features = metadata['all_features']
        selected_features = metadata['selected_features']
        print(f"Loaded metadata with {len(expected_features)} total features and {len(selected_features)} selected features")
    except FileNotFoundError:
        print(f"Error: Metadata file '{METADATA_FILENAME}' not found.")
        return None, None
    except KeyError as e:
        print(f"Error: Metadata file '{METADATA_FILENAME}' is missing required key: {e}")
        return None, None
    except Exception as e:
        print(f"Error reading metadata file: {e}")
        return None, None

    # Ensure input DataFrame has the correct columns in the right order
    try:
        # Check if any expected features are missing from the input
        missing_features = [f for f in expected_features if f not in input_features_df.columns]
        
        if missing_features:
            print(f"Warning: Input data missing {len(missing_features)} features required by the scaler: {missing_features}")
            # Add missing features with default value of 0.0
            for feature in missing_features:
                input_features_df[feature] = 0.0
                print(f"Added missing feature with default value: {feature}")
        
        # Now reorder columns to match expected order
        input_features_df_ordered = input_features_df[expected_features]
        
    except KeyError as e:
         print(f"Error: Input data missing features required by the scaler: {e}")
         print(f"Required features: {expected_features}")
         return None, None
    except Exception as e:
        print(f"Error ordering input features: {e}")
        return None, None

    # Preprocess features: Scale and then apply feature selection
    try:
        # Check if the scaler has feature_names_in_ attribute
        if hasattr(scaler, 'feature_names_in_'):
            # Get the features that the scaler was trained with
            scaler_features = scaler.feature_names_in_
            
            # Check if there are features in input_features_df_ordered that are not in scaler_features
            extra_features = [f for f in input_features_df_ordered.columns if f not in scaler_features]
            
            if extra_features:
                print(f"Warning: Removing {len(extra_features)} features not seen during scaler training: {extra_features}")
                # Remove the extra features
                input_features_df_ordered = input_features_df_ordered.drop(columns=extra_features)
                
            # Check if there are features in scaler_features that are not in input_features_df_ordered
            missing_features = [f for f in scaler_features if f not in input_features_df_ordered.columns]
            
            if missing_features:
                print(f"Warning: Adding {len(missing_features)} features required by scaler: {missing_features}")
                # Add the missing features with default value of 0.0
                for feature in missing_features:
                    input_features_df_ordered[feature] = 0.0
                
                # Reorder columns to match scaler's feature order
                input_features_df_ordered = input_features_df_ordered[scaler_features]
        
        # First scale the features
        features_scaled = scaler.transform(input_features_df_ordered)
        
        # Apply feature selection using the selector
        features_selected = selector.transform(features_scaled)
        
        print(f"Input features preprocessed: scaled {features_scaled.shape[1]} features, selected {features_selected.shape[1]} features")
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
        # Use selected features for prediction
        # FIXED: Get probability of class 0 (malignant) as the risk score
        # In scikit-learn breast cancer dataset: 0=malignant, 1=benign
        # Higher risk score should mean higher risk of cancer (malignant)
        risk_score = model.predict_proba(features_selected)[:, 0][0]
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
            'explanation': f"Risk score based on analysis of {features_scaled.shape[1]} features. Plan generation failed."
        }
        return assessment_result, None


    assessment_result = {
        'risk_score': round(risk_score, 3),
        'risk_category': prevention_plan['risk_category'], # Get category from plan
        'explanation': f"Risk score based on analysis of {features_scaled.shape[1]} features."
    }

    return assessment_result, prevention_plan 

# --- Lifestyle-Focused Risk Assessment ---
def calculate_lifestyle_risk_score(input_features_df):
    """
    Calculate a risk score based primarily on lifestyle factors.
    This function gives more weight to lifestyle factors than medical factors.
    
    Args:
        input_features_df: DataFrame containing the patient's features
        
    Returns:
        risk_score: A float between 0 and 1 representing the lifestyle-based risk
        risk_category: A string ('Low', 'Medium', or 'High')
        explanation: A string explaining the risk assessment
    """
    try:
        # Extract only lifestyle features
        lifestyle_features = {}
        for feature in LIFESTYLE_FEATURES_NAMES:
            if feature in input_features_df.columns:
                lifestyle_features[feature] = input_features_df[feature].iloc[0]
        
        if not lifestyle_features:
            return None, None, "No lifestyle features found in input data."
        
        # Calculate weighted average of lifestyle factors
        # Higher values indicate higher risk
        weights = {
            'alcohol_consumption': 0.15,
            'body_weight_bmi': 0.15,
            'physical_inactivity_level': 0.15,
            'poor_diet_quality': 0.15,
            'smoking_history': 0.15,
            'environmental_exposures': 0.05,
            'hormone_use': 0.05,
            'family_history_genetic': 0.05,
            'reproductive_history': 0.05,
            'menstrual_history': 0.05
        }
        
        total_weight = 0
        weighted_sum = 0
        
        for feature, value in lifestyle_features.items():
            if feature in weights:
                weighted_sum += value * weights[feature]
                total_weight += weights[feature]
        
        if total_weight == 0:
            return None, None, "Could not calculate lifestyle risk score: no valid features found."
        
        # Normalize to get final risk score between 0 and 1
        lifestyle_risk_score = weighted_sum / total_weight
        
        # Apply a sigmoid transformation to make the distribution more realistic
        # This helps differentiate between low, medium, and high risk more clearly
        import math
        adjusted_score = 1 / (1 + math.exp(-10 * (lifestyle_risk_score - 0.5)))
        
        # Determine risk category
        if adjusted_score >= 0.7:
            risk_category = "High"
        elif adjusted_score >= 0.3:
            risk_category = "Medium"
        else:
            risk_category = "Low"
        
        # Create explanation
        high_risk_factors = []
        for feature, value in lifestyle_features.items():
            if value >= 0.7:
                high_risk_factors.append(feature.replace('_', ' ').title())
        
        if high_risk_factors:
            explanation = f"Your lifestyle risk assessment is based on {len(lifestyle_features)} factors. "
            explanation += f"Key high-risk factors include: {', '.join(high_risk_factors)}."
        else:
            explanation = f"Your lifestyle risk assessment is based on {len(lifestyle_features)} factors. "
            explanation += "No significant high-risk lifestyle factors were identified."
        
        return adjusted_score, risk_category, explanation
        
    except Exception as e:
        print(f"Error calculating lifestyle risk score: {e}")
        return None, None, f"Error calculating lifestyle risk score: {str(e)}"

# --- Individual SHAP Plot Generation ---
def generate_individual_shap_plot(input_features_df, top_n=7, plot_type='bar'):
    """
    Generate a simplified personalized lifestyle risk factor analysis.
    Instead of using SHAP values (which are complex to calculate for just lifestyle factors),
    we'll create a simple visualization based on the patient's lifestyle factors.
    
    Args:
        input_features_df: DataFrame containing the patient's features
        top_n: Number of top features to display (default: 7)
        plot_type: Type of plot ('bar' is the only supported type)
        
    Returns:
        base64_image: Base64 encoded image of the plot
        top_features: List of the top lifestyle factors and their values
    """
    # Check if required libraries are available
    if plt is None or Image is None:
        print("Warning: matplotlib or PIL not available. Returning only text analysis without visualization.")
        img_str = None
    
    try:
        # Extract only lifestyle features
        lifestyle_features = LIFESTYLE_FEATURES_NAMES.copy()
        
        # Create a list to store lifestyle factors and their risk scores
        lifestyle_risk_factors = []
        
        # Calculate a risk score for each lifestyle factor
        for feature in lifestyle_features:
            if feature in input_features_df.columns:
                value = input_features_df[feature].iloc[0]
                
                # For most factors, higher values = higher risk
                risk_score = value
                
                # For these features, the values are already inverted so higher = higher risk
                if False:  # Removed condition as we now use physical_inactivity_level and poor_diet_quality
                    risk_score = 1.0 - value
                
                # Add to our list
                lifestyle_risk_factors.append({
                    'feature': feature,
                    'value': value,
                    'risk_score': risk_score
                })
        
        # Sort by risk score (highest risk first) and take top N
        lifestyle_risk_factors = sorted(lifestyle_risk_factors, key=lambda x: x['risk_score'], reverse=True)[:top_n]
        
        # Generate visualization only if required libraries are available
        img_str = None
        if plt is not None and Image is not None:
            try:
                # Create a figure to hold the plot
                plt.figure(figsize=(10, 6))
                
                # Extract data for plotting
                features = [item['feature'].replace('_', ' ').title() for item in lifestyle_risk_factors]
                risk_scores = [item['risk_score'] for item in lifestyle_risk_factors]
                
                # Create the bar plot
                bars = plt.barh(features, risk_scores)
                
                # Color the bars based on risk level
                for i, bar in enumerate(bars):
                    # Red for high risk, yellow for medium, green for low
                    if risk_scores[i] > 0.66:
                        bar.set_color('red')
                    elif risk_scores[i] > 0.33:
                        bar.set_color('orange')
                    else:
                        bar.set_color('green')
                
                # Add labels and title
                plt.xlabel('Risk Level')
                plt.ylabel('Lifestyle Factor')
                plt.title('Your Lifestyle Risk Factors')
                plt.xlim(0, 1)
                
                # Add a legend
                try:
                    from matplotlib.patches import Patch
                    legend_elements = [
                        Patch(facecolor='red', label='High Risk'),
                        Patch(facecolor='orange', label='Medium Risk'),
                        Patch(facecolor='green', label='Low Risk')
                    ]
                    plt.legend(handles=legend_elements, loc='lower right')
                except ImportError:
                    # Skip legend if Patch is not available
                    pass
                
                plt.tight_layout()
                
                # Save plot to a bytes buffer
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100)
                plt.close()
                buf.seek(0)
                
                # Convert to base64 for embedding in HTML
                img = Image.open(buf)
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                buffer.seek(0)
                img_str = base64.b64encode(buffer.read()).decode('utf-8')
            except Exception as viz_error:
                print(f"Error generating visualization: {viz_error}")
                # Continue with text analysis even if visualization fails
                img_str = None
        
        # Format the top features for display
        top_features = []
        for item in lifestyle_risk_factors:
            feature = item['feature']
            value = item['value']
            risk_score = item['risk_score']
            
            # Format the feature value as a descriptive level
            level = "Very High" if value > 0.8 else \
                   "High" if value > 0.6 else \
                   "Moderate" if value > 0.4 else \
                   "Low" if value > 0.2 else "Very Low"
            
            # For these features, the values are already inverted so higher = higher risk
            if False:  # Removed condition as we now use physical_inactivity_level and poor_diet_quality
                level = "Very Low" if value > 0.8 else \
                       "Low" if value > 0.6 else \
                       "Moderate" if value > 0.4 else \
                       "High" if value > 0.2 else "Very High"
            
            # Determine if this factor increases or decreases risk
            impact = "Increases" if risk_score > 0.5 else "Decreases"
            
            top_features.append({
                "feature": feature,
                "value": level,
                "impact": impact,
                "magnitude": risk_score
            })
        
        return img_str, top_features
        
    except Exception as e:
        print(f"Error generating lifestyle risk factor analysis: {e}")
        return None, None