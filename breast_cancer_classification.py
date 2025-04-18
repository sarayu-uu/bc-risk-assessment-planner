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
import joblib
import shap
import json
from datetime import datetime
import random
import warnings

# Import necessary functions/classes from the utility file
from risk_assessment_utils import assess_risk_and_plan, MODEL_FILENAME, SCALER_FILENAME, SELECTOR_FILENAME, METADATA_FILENAME

# Ignore convergence warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)


# --- Constants --- (Now mostly defined in utils, maybe keep SHAP plot names here)
SHAP_SUMMARY_PLOT_FILENAME = 'shap_summary_plot.png'


# Cell 3: Enhanced Dataset Creation (Function is now in utils)
def create_enhanced_dataset():
    # Define constants needed within this function
    LIFESTYLE_FEATURES_NAMES = [
        'physical_activity', 'diet_quality', 'stress_level',
        'sleep_quality', 'alcohol_consumption', 'smoking_history'
    ]
    INTERACTION_FEATURES_NAMES = [
        'activity_immune_interaction', 'diet_cell_interaction', 'stress_immune_interaction'
    ]

    # Load original dataset
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    # Add synthetic lifestyle features (0-1 scale, higher is better/more)
    np.random.seed(42) # for reproducibility
    for feature in LIFESTYLE_FEATURES_NAMES:
        df[feature] = np.random.uniform(0.1, 0.9, len(df))

    # Create interaction features
    df['activity_immune_interaction'] = df['physical_activity'] * df['worst concave points']
    df['diet_cell_interaction'] = df['diet_quality'] * df['mean texture']
    df['stress_immune_interaction'] = (1 - df['stress_level']) * df['mean smoothness']

    print("--- Enhanced Dataset Created ---")
    print(f"Added features: {LIFESTYLE_FEATURES_NAMES + INTERACTION_FEATURES_NAMES}")
    print(f"New shape: {df.shape}")
    return df, list(data.feature_names)


# Cell 4: Personalized Prevention Plan Generator (Class is now in utils)
# --- REMOVED CLASS DEFINITION ---
# (The lines below defining the class and its docstring were leftovers and are now removed)


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
    'model_name': 'Breast Cancer Risk Assessment with Lifestyle Factors',
    'best_model_type': best_model_name,
    'version': '1.3', # Incremented version after fixing risk score interpretation
    'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'risk_score_interpretation': 'Risk score represents probability of malignancy (class 0). Higher score = higher cancer risk.',
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

# Clean results for JSON serialization
for model_res in model_metadata['full_evaluation_results'].values():
    if 'classification_report' in model_res:
        for report_key, report_val in model_res['classification_report'].items():
            if isinstance(report_val, dict):
                for metric, value in report_val.items():
                    if isinstance(value, (np.int32, np.int64)):
                        report_val[metric] = int(value)
                    elif isinstance(value, (np.float32, np.float64)):
                        report_val[metric] = float(value)
            elif isinstance(report_val, (np.int32, np.int64)):
                model_res['classification_report'][report_key] = int(report_val)
            elif isinstance(report_val, (np.float32, np.float64)):
                model_res['classification_report'][report_key] = float(report_val)

with open(METADATA_FILENAME, 'w') as f:
    json.dump(model_metadata, f, indent=4)

print(f"✅ Model ({MODEL_FILENAME}), Scaler ({SCALER_FILENAME}), Selector ({SELECTOR_FILENAME}), and Metadata ({METADATA_FILENAME}) saved successfully.")


# 6. SHAP Analysis
print("\n--- Performing SHAP Analysis ---")
if isinstance(best_model, (RandomForestClassifier, GradientBoostingClassifier)):
    try:
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_test_selected_df)
        shap_values_to_plot = shap_values[1] if isinstance(shap_values, list) and len(shap_values) == 2 else shap_values

        # Create two SHAP summary plots - one for all features and one for lifestyle features
        # 1. Plot for all features
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values_to_plot, X_test_selected_df, show=False)
        plt.title(f'SHAP Summary Plot for {best_model_name} - All Features')
        plt.tight_layout()
        plt.savefig(SHAP_SUMMARY_PLOT_FILENAME)
        plt.close()
        
        # We're not creating a separate SHAP plot for lifestyle parameters as requested
        
        print(f"SHAP summary plot saved as {SHAP_SUMMARY_PLOT_FILENAME}")
    except Exception as e:
        print(f"Error during SHAP analysis: {e}")
else:
    print(f"SHAP analysis skipped (not a compatible tree-based model: {best_model_name}).")


# 7. Define Risk Assessment and Prevention Plan Functions (Now imported)
# --- REMOVED FUNCTION DEFINITION ---


# 8. Example Usage (Using imported function)
print("\n--- Running Example: Assessment and Plan for a Random Test Sample ---")

# Select a random sample from the original TEST set (X_test)
# Important: Use X_test which has the original (unscaled) feature values
random_index = np.random.randint(0, len(X_test))
print(f"Selected random test sample index: {random_index}")

# Get the sample as a DataFrame (ensure columns match the original X)
random_sample_features = pd.DataFrame([X_test.iloc[random_index]], columns=X.columns)

# Add dummy lifestyle/interaction features if they are missing in X_test (shouldn't happen if X_test derived from df_enhanced)
# This is a safeguard - ideally X_test already has all columns.
missing_cols = set(all_feature_names) - set(random_sample_features.columns)
if missing_cols:
    print(f"Warning: Random sample missing columns for assessment: {missing_cols}. Adding dummies (0). This might indicate an issue.")
    for col in missing_cols:
        random_sample_features[col] = 0


# Get the actual target for comparison
actual_target = y_test.iloc[random_index]
print(f"Actual Target (0=Malignant, 1=Benign): {actual_target}")

# Ensure the imported function is used
assessment, plan = assess_risk_and_plan(random_sample_features)

if assessment and plan:
    print("\n--- Risk Assessment Result ---")
    print(json.dumps(assessment, indent=2))

    print("\n--- Personalized Prevention Plan ---")
    print(json.dumps(plan, indent=2))
else:
    print("\nFailed to generate assessment and plan for the example sample.")

print("\n--- Script Finished ---")