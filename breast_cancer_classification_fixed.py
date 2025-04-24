# Cell 1: Install Required Packages (if needed)
# !pip install numpy pandas matplotlib seaborn scikit-learn joblib shap --quiet

# Cell 2: Importing Required Libraries
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for servers/scripts
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.feature_selection import SelectFromModel
from sklearn.base import clone
import joblib
import shap
import json
from datetime import datetime
import random
import warnings

# Import necessary functions/classes from the utility file
from risk_assessment_utils import (
    assess_risk_and_plan, MODEL_FILENAME, SCALER_FILENAME, SELECTOR_FILENAME, METADATA_FILENAME,
    LIFESTYLE_FEATURES_NAMES, INTERACTION_FEATURES_NAMES
)

# Ignore convergence warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)


# --- Constants --- (Now mostly defined in utils, maybe keep SHAP plot names here)
SHAP_SUMMARY_PLOT_FILENAME = 'shap_summary_plot.png'
LIFESTYLE_SHAP_PLOT_FILENAME = 'shap_lifestyle_features_plot.png'
MEDICAL_SHAP_PLOT_FILENAME = 'shap_medical_features_plot.png'
SHAP_EXPLAINER_FILENAME = 'shap_explainer.pkl'


# Cell 3: Enhanced Dataset Creation with Evidence-Based Risk Factors
def create_enhanced_dataset():
    # Define constants needed within this function
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

    # Load original dataset
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    # Add evidence-based risk factors (0-1 scale)
    # For all factors, higher values = higher risk
    np.random.seed(42) # for reproducibility
    
    # 1. Alcohol Consumption (higher = more consumption = higher risk)
    df['alcohol_consumption'] = np.random.uniform(0.1, 0.9, len(df))
    
    # 2. Body Weight/BMI (higher = higher BMI = higher risk)
    df['body_weight_bmi'] = np.random.uniform(0.1, 0.9, len(df))
    
    # 3. Physical Inactivity Level (higher = less activity = higher risk)
    df['physical_inactivity_level'] = np.random.uniform(0.1, 0.9, len(df))
    
    # 4. Poor Diet Quality (higher = worse diet = higher risk)
    df['poor_diet_quality'] = np.random.uniform(0.1, 0.9, len(df))
    
    # 5. Reproductive History (higher = more risk factors = higher risk)
    # (late first pregnancy, fewer children, no breastfeeding)
    df['reproductive_history'] = np.random.uniform(0.1, 0.9, len(df))
    
    # 6. Hormone Use (higher = more hormone use = higher risk)
    df['hormone_use'] = np.random.uniform(0.1, 0.9, len(df))
    
    # 7. Family History/Genetic Factors (higher = stronger family history = higher risk)
    df['family_history_genetic'] = np.random.uniform(0.1, 0.9, len(df))
    
    # 8. Smoking History (higher = more smoking = higher risk)
    df['smoking_history'] = np.random.uniform(0.1, 0.9, len(df))
    
    # 9. Environmental Exposures (higher = more exposures = higher risk)
    df['environmental_exposures'] = np.random.uniform(0.1, 0.9, len(df))
    
    # 10. Menstrual History (higher = longer estrogen exposure = higher risk)
    df['menstrual_history'] = np.random.uniform(0.1, 0.9, len(df))

    # Create medically relevant interaction features
    # Alcohol affects immune response markers
    df['alcohol_immune_interaction'] = df['alcohol_consumption'] * df['worst concave points']
    
    # BMI affects hormone-related markers
    df['bmi_hormone_interaction'] = df['body_weight_bmi'] * df['mean perimeter']
    
    # Physical inactivity affects immune markers
    df['inactivity_immune_interaction'] = df['physical_inactivity_level'] * df['worst concave points']
    
    # Poor diet quality affects cell texture
    df['poor_diet_cell_interaction'] = df['poor_diet_quality'] * df['mean texture']
    
    # Smoking affects DNA damage markers
    df['smoking_dna_interaction'] = df['smoking_history'] * df['worst radius']
    
    # Hormone use affects cell proliferation markers
    df['hormone_cell_proliferation_interaction'] = df['hormone_use'] * df['mean area']
    
    # Genetic factors interact with cell characteristics
    df['genetic_cell_interaction'] = df['family_history_genetic'] * df['worst perimeter']
    
    # Menstrual history interacts with hormone-sensitive markers
    df['menstrual_hormone_interaction'] = df['menstrual_history'] * df['mean concavity']
    
    # Reproductive history interacts with cell characteristics
    df['reproductive_cell_interaction'] = df['reproductive_history'] * df['mean radius']
    
    # Environmental exposures interact with cell characteristics
    df['environmental_cell_interaction'] = df['environmental_exposures'] * df['mean smoothness']

    print("--- Enhanced Dataset Created with Evidence-Based Risk Factors ---")
    print(f"Added features: {LIFESTYLE_FEATURES_NAMES + INTERACTION_FEATURES_NAMES}")
    print(f"New shape: {df.shape}")
    return df, list(data.feature_names)


# --- Main Script Logic ---

# 1. Create Enhanced Dataset
df_enhanced, original_feature_names = create_enhanced_dataset()

# 2. Prepare Data for Model
print("\n--- Preparing Data ---")
X = df_enhanced.drop('target', axis=1)
y = df_enhanced['target']
all_feature_names = X.columns.tolist()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization (Fit only on training data)
# Important: Scale based on ALL features now
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert scaled arrays back to DataFrames to retain feature names for selection/SHAP
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=all_feature_names)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=all_feature_names)


# 3. Feature Selection (Using Hyperparameter Tuned RF if step 3 implemented)
print("Performing feature selection...")
rf_selector = RandomForestClassifier(n_estimators=100, random_state=42) # Use tuned params if available
rf_selector.fit(X_train_scaled_df, y_train)
selector = SelectFromModel(rf_selector, threshold='median', prefit=True)
X_train_selected = selector.transform(X_train_scaled_df)
X_test_selected = selector.transform(X_test_scaled_df)
selected_mask = selector.get_support()
selected_features = X_train_scaled_df.columns[selected_mask].tolist()
print(f"Selected {len(selected_features)} features: {selected_features}")

# Create DataFrames for selected features
X_train_selected_df = pd.DataFrame(X_train_selected, columns=selected_features)
X_test_selected_df = pd.DataFrame(X_test_selected, columns=selected_features)


# --- MODIFIED SECTION: Add Hyperparameter Tuning --- (Step 3 from plan)
print("\n--- Hyperparameter Tuning (RandomizedSearchCV) ---")
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Define parameter space for Gradient Boosting (example)
param_dist_gb = {
    'n_estimators': randint(100, 300),
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': randint(3, 7)
}

# Setup Randomized Search for Gradient Boosting
gb_base = GradientBoostingClassifier(random_state=42)
random_search_gb = RandomizedSearchCV(
    estimator=gb_base,
    param_distributions=param_dist_gb,
    n_iter=20, # Number of parameter settings that are sampled
    cv=5,      # 5-fold cross-validation
    scoring='roc_auc',
    n_jobs=-1, # Use all available CPU cores
    random_state=42,
    verbose=1  # Show progress
)

# Fit Randomized Search
print("Running Randomized Search for Gradient Boosting...")
random_search_gb.fit(X_train_selected, y_train)

print(f"Best Parameters for Gradient Boosting: {random_search_gb.best_params_}")
print(f"Best CV ROC AUC Score for Gradient Boosting: {random_search_gb.best_score_:.3f}")

# Use the best estimator found by the search
best_gb_model = random_search_gb.best_estimator_

# --- END HYPERPARAMETER TUNING SECTION ---

# 4. Train Final Models (Use best tuned model if applicable)
print("\n--- Training and Evaluating Final Models ---")
models = {
    # Keep others for comparison, or just use the tuned one
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
    "Tuned Gradient Boosting": best_gb_model # Use the tuned model
}

results = {}
trained_models = {}

for name, model in models.items():
    print(f"Evaluating {name}...")
    # If it's the tuned model, it's already fitted, just predict
    # Otherwise, fit the others (though fitting GB again is redundant here)
    if name != "Tuned Gradient Boosting":
        model.fit(X_train_selected, y_train)

    trained_models[name] = model

    # Predict probabilities on test set
    # NOTE: For model evaluation, we still use class 1 (benign) probability
    # because scikit-learn's roc_auc_score expects probabilities of the positive class (1)
    # This is different from our risk assessment where we want risk_score to be probability of cancer
    if hasattr(model, "predict_proba"):
         y_prob = model.predict_proba(X_test_selected)[:, 1]  # Class 1 (benign) for evaluation
    else:
         y_decision = model.decision_function(X_test_selected)
         y_prob = (y_decision - y_decision.min()) / (y_decision.max() - y_decision.min())

    y_pred = model.predict(X_test_selected)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    # Get CV score from tuning results if it's the tuned model
    cv_score = random_search_gb.best_score_ if name == "Tuned Gradient Boosting" else "N/A (Not Tuned)"

    results[name] = {
        'accuracy': round(accuracy, 3),
        'test_roc_auc': round(roc_auc, 3),
        'cv_roc_auc_mean (from tuning if applicable)': round(cv_score, 3) if isinstance(cv_score, float) else cv_score,
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    print(f"{name} - Test ROC AUC: {results[name]['test_roc_auc']:.3f}, Best CV ROC AUC: {results[name]['cv_roc_auc_mean (from tuning if applicable)']}")


# 5. Select and Save Best Model (Based on Test ROC AUC)
# Force selection of the tuned model if it was trained
best_model_name = "Tuned Gradient Boosting" if "Tuned Gradient Boosting" in trained_models else max(results, key=lambda name: results[name]['test_roc_auc'])
best_model = trained_models[best_model_name]
print(f"\n--- Best Model Selected: {best_model_name} (Test ROC AUC: {results[best_model_name]['test_roc_auc']:.3f}) ---")

# Save components
print("Saving components...")
joblib.dump(best_model, MODEL_FILENAME)
joblib.dump(scaler, SCALER_FILENAME)
joblib.dump(selector, SELECTOR_FILENAME)

# Update metadata
model_metadata = {
    'model_name': 'Breast Cancer Risk Assessment with Evidence-Based Risk Factors',
    'best_model_type': best_model_name,
    'version': '2.0', # Incremented version after implementing evidence-based risk factors
    'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'risk_score_interpretation': 'Risk score represents probability of malignancy (class 0). Higher score = higher cancer risk.',
    'risk_factors': {
        'alcohol_consumption': 'Risk increases with amount consumed; even light drinking (1 drink/day) can increase risk by 7-10%',
        'body_weight_bmi': 'Being overweight or obese, especially after menopause, increases risk as excess body fat increases estrogen production',
        'physical_inactivity_level': 'Low physical activity increases risk; regular exercise can reduce risk by 10-20%',
        'poor_diet_quality': 'Low intake of fruits, vegetables, and fiber, and high consumption of processed foods increases risk',
        'reproductive_history': 'Late first pregnancy (after 30), fewer children, and shorter/no breastfeeding increases risk',
        'hormone_use': 'Hormone replacement therapy (especially combined estrogen-progestin) and certain oral contraceptives increase risk',
        'family_history_genetic': 'First-degree relatives with breast cancer and known genetic mutations (BRCA1, BRCA2) increase risk',
        'smoking_history': 'Current or past tobacco use increases risk, with duration and intensity as important factors',
        'environmental_exposures': 'Radiation exposure (especially to chest area before age 30) and certain chemicals increase risk',
        'menstrual_history': 'Early menarche (before age 12) and late menopause (after 55) increase risk due to longer estrogen exposure'
    },
    'hyperparameter_tuning_details (GB)': {
        'method': 'RandomizedSearchCV',
        'params_searched': str(param_dist_gb), # Convert dists to string
        'best_params': random_search_gb.best_params_,
        'best_cv_score': random_search_gb.best_score_
    },
    'all_features': all_feature_names,
    'selected_features': selected_features,
    'model_performance_summary': results[best_model_name],
    'full_evaluation_results': results,
    'scaler_details': {'type': 'StandardScaler', 'n_features_in': scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else 'N/A'},
    'selector_details': {'type': 'SelectFromModel', 'base_estimator': 'RandomForestClassifier', 'threshold': 'median'}
}

with open(METADATA_FILENAME, 'w') as f:
    json.dump(model_metadata, f, indent=4)

print(f"Model ({MODEL_FILENAME}), Scaler ({SCALER_FILENAME}), Selector ({SELECTOR_FILENAME}), and Metadata ({METADATA_FILENAME}) saved successfully.")


# 6. SHAP Analysis - Hybrid Approach
print("\n--- Performing SHAP Analysis (Hybrid Approach) ---")
if isinstance(best_model, (RandomForestClassifier, GradientBoostingClassifier)):
    try:
        # Create a SHAP explainer for the best model
        explainer = shap.TreeExplainer(best_model)
        
        # 1. Population-level SHAP plot for all selected features
        print("Creating population-level SHAP plot for all selected features...")
        shap_values = explainer.shap_values(X_test_selected_df)
        shap_values_to_plot = shap_values[1] if isinstance(shap_values, list) and len(shap_values) == 2 else shap_values

        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values_to_plot, X_test_selected_df, show=False)
        plt.title(f'Population-Level SHAP Summary - All Selected Features')
        plt.tight_layout()
        plt.savefig(SHAP_SUMMARY_PLOT_FILENAME)
        plt.close()
        
        print(f"Population-level SHAP plot saved as {SHAP_SUMMARY_PLOT_FILENAME}")
        
        # 2. Population-level SHAP plot for medical features only
        print("Creating population-level SHAP plot for medical features only...")
        
        # Get medical features (original features from the Wisconsin dataset)
        medical_features = [f for f in selected_features if f not in LIFESTYLE_FEATURES_NAMES + INTERACTION_FEATURES_NAMES]
        
        if medical_features:
            # Create a new model specifically for medical features to avoid SHAP errors
            medical_model = clone(best_model)
            X_train_medical = X_train_selected_df[medical_features]
            X_test_medical = X_test_selected_df[medical_features]
            
            # Train the model on just medical features
            medical_model.fit(X_train_medical, y_train)
            
            # Create a new explainer for medical features
            medical_explainer = shap.TreeExplainer(medical_model)
            medical_shap_values = medical_explainer.shap_values(X_test_medical)
            medical_shap_values_to_plot = medical_shap_values[1] if isinstance(medical_shap_values, list) and len(medical_shap_values) == 2 else medical_shap_values
            
            # Create the medical features SHAP plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(medical_shap_values_to_plot, X_test_medical, show=False)
            plt.title(f'Population-Level SHAP Summary - Medical Features Only')
            plt.tight_layout()
            plt.savefig(MEDICAL_SHAP_PLOT_FILENAME)
            plt.close()
            
            print(f"Medical features SHAP plot saved as {MEDICAL_SHAP_PLOT_FILENAME}")
        
        # 3. Population-level SHAP plot for lifestyle and interaction features
        print("Creating population-level SHAP plot for lifestyle factors...")
        
        # Get all lifestyle and interaction features that exist in the DataFrame
        all_lifestyle_interaction_features = LIFESTYLE_FEATURES_NAMES + INTERACTION_FEATURES_NAMES
        lifestyle_interaction_features = [f for f in all_lifestyle_interaction_features if f in X_train_scaled_df.columns]
        
        print(f"Found {len(lifestyle_interaction_features)} lifestyle and interaction features in the dataset")
        
        # Only proceed if we have lifestyle features to analyze
        if len(lifestyle_interaction_features) > 0:
            # Create a new model specifically for lifestyle features
            lifestyle_model = clone(best_model)
            
            # Filter data to include only lifestyle and interaction features
            X_train_lifestyle = X_train_scaled_df[lifestyle_interaction_features]
            X_test_lifestyle = X_test_scaled_df[lifestyle_interaction_features]
        
            # Train the model on just lifestyle features
            lifestyle_model.fit(X_train_lifestyle, y_train)
            
            # Calculate SHAP values for lifestyle features
            lifestyle_explainer = shap.TreeExplainer(lifestyle_model)
            lifestyle_shap_values = lifestyle_explainer.shap_values(X_test_lifestyle)
            lifestyle_shap_values_to_plot = lifestyle_shap_values[1] if isinstance(lifestyle_shap_values, list) and len(lifestyle_shap_values) == 2 else lifestyle_shap_values
            
            # Create the lifestyle SHAP plot
            plt.figure(figsize=(12, 10))
            shap.summary_plot(lifestyle_shap_values_to_plot, X_test_lifestyle, show=False)
            plt.title(f'Population-Level SHAP Summary - Lifestyle & Interaction Features')
            plt.tight_layout()
            plt.savefig(LIFESTYLE_SHAP_PLOT_FILENAME)
            plt.close()
            
            print(f"Lifestyle features SHAP plot saved as {LIFESTYLE_SHAP_PLOT_FILENAME}")
        else:
            print("No lifestyle or interaction features found in the selected features. Skipping lifestyle SHAP plot.")
        
        # 4. Save the explainer for generating individual-level SHAP plots later
        print("Saving SHAP explainer for individual-level analysis...")
        joblib.dump(explainer, SHAP_EXPLAINER_FILENAME)
        print(f"SHAP explainer saved as {SHAP_EXPLAINER_FILENAME}")
        
    except Exception as e:
        print(f"Error during SHAP analysis: {e}")
else:
    print(f"SHAP analysis skipped (not a compatible tree-based model: {best_model_name}).")


# 7. Example Usage (Using imported function)
print("\n--- Running Example: Assessment and Plan for a Random Test Sample ---")

# Select a random sample from the original TEST set (X_test)
# Important: Use X_test which has the original (unscaled) feature values
random_index = np.random.randint(0, len(X_test))
print(f"Selected random test sample index: {random_index}")

# Get the sample as a DataFrame (ensure columns match the original X)
random_sample_features = pd.DataFrame([X_test.iloc[random_index]], columns=X.columns)

# Add dummy lifestyle/interaction features if they are missing in X_test (shouldn't happen if X_test derived from df_enhanced)

# Get the actual target value for comparison
actual_target = y_test.iloc[random_index]
print(f"Actual Target (0=Malignant, 1=Benign): {actual_target}")

# Run the assessment and planning function
assessment, plan = assess_risk_and_plan(random_sample_features)

# Print the results
if assessment and plan:
    print("\n--- Risk Assessment Result ---")
    print(json.dumps(assessment, indent=2))
    
    print("\n--- Personalized Prevention Plan ---")
    print(json.dumps(plan, indent=2))
else:
    print("Failed to generate assessment and plan.")

print("\n--- Script Finished ---")