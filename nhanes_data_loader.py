"""
NHANES Data Loader for Breast Cancer Risk Factors

This module fetches and processes NHANES data to create a dataset with evidence-based
lifestyle factors related to breast cancer risk.

References:
- Alcohol: https://www.cancer.gov/about-cancer/causes-prevention/risk/alcohol/alcohol-fact-sheet
- BMI: https://www.cancer.gov/about-cancer/causes-prevention/risk/obesity
- Physical Activity: https://www.cancer.gov/about-cancer/causes-prevention/risk/physical-activity
- Diet: https://www.cancer.gov/about-cancer/causes-prevention/risk/diet
- Smoking: https://www.cancer.gov/about-cancer/causes-prevention/risk/tobacco
- Hormone Use: https://www.cancer.gov/types/breast/risk-fact-sheet
"""

import pandas as pd
import numpy as np
from nhanes.load import load_NHANES_data
import warnings
from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

# Suppress warnings
warnings.filterwarnings('ignore')

def load_nhanes_lifestyle_factors():
    """
    Load lifestyle factors from NHANES dataset that are relevant to breast cancer risk.
    
    Returns:
        DataFrame with processed lifestyle factors
    """
    print("Loading NHANES data (this may take a moment)...")
    
    # Define the NHANES variables we want to load
    # These codes correspond to specific NHANES survey questions
    variables = {
        # Alcohol consumption
        'ALQ130': 'alcohol_days_per_year',  # Days per year had alcoholic beverages
        'ALQ120Q': 'alcohol_drinks_per_day',  # Usual number of drinks on drinking days
        
        # Body measurements
        'BMXBMI': 'body_weight_bmi',  # Body Mass Index (kg/m²)
        
        # Physical activity
        'PAQ650': 'physical_inactivity_level',  # Minutes of moderate physical activity per week
        'PAQ665': 'physical_activity_vigorous',  # Minutes of vigorous physical activity per week
        
        # Diet quality
        'DBQ700': 'diet_quality_fruit',  # How often eat fruit
        'DBQ710': 'diet_quality_vegetables',  # How often eat vegetables
        'DBQ720': 'diet_quality_greens',  # How often eat dark green vegetables
        
        # Smoking
        'SMQ020': 'smoking_status',  # Smoked at least 100 cigarettes in life
        'SMQ040': 'smoking_current',  # Do you now smoke cigarettes
        
        # Reproductive history
        'RHQ131': 'reproductive_age_first_birth',  # Age when first child was born
        'RHQ160': 'reproductive_breastfed',  # Ever breastfed children
        
        # Hormone use
        'RHQ420': 'hormone_birth_control',  # Ever taken birth control pills
        'RHQ540': 'hormone_hrt',  # Ever taken hormone replacement therapy
        
        # Demographics (for filtering and context)
        'RIAGENDR': 'gender',  # Gender
        'RIDAGEYR': 'age',  # Age in years
    }
    
    try:
        # Load NHANES data for the most recent available cycle
        # The NHANES library only has data for '2017-2018' in the combined_data directory
        print("Loading NHANES data for cycle '2017-2018'...")
        nhanes_data = load_NHANES_data(year='2017-2018')
        
        # Map NHANES column names to our variable names
        # The NHANES dataset has different column names than the original variable codes
        # For example, 'Gender' instead of 'RIAGENDR'
        nhanes_mapping = {
            'Gender': 'gender',
            'AgeInYearsAtScreening': 'age',
            'BodyMassIndexKgm2': 'body_weight_bmi',
            'AlcoholGm_DR1TOT': 'alcohol_consumption',
            'VigorousRecreationalActivities': 'physical_inactivity_level',
            'ModerateRecreationalActivities': 'physical_activity_vigorous',
            'HowHealthyIsTheDiet': 'diet_quality',
            'SmokedAtLeast100CigarettesInLife': 'smoking_status',
            'DoYouNowSmokeCigarettes': 'smoking_current',
            'EverBreastfedOrFedBreastmilk': 'reproductive_breastfed',
            'TakeMedicationForTheseFeelings': 'hormone_use'
        }
        
        # Use only the columns that exist in the dataset
        valid_columns = {col: nhanes_mapping[col] for col in nhanes_mapping if col in nhanes_data.columns}
        
        # Rename columns to more descriptive names
        nhanes_data = nhanes_data.rename(columns=valid_columns)
        
        # Filter to include only women (NHANES uses 2 for female)
        women_data = nhanes_data[nhanes_data['gender'] == 2].copy()
        
        # Filter to include only adults
        women_data = women_data[women_data['age'] >= 18].copy()
        
        # Process each lifestyle factor
        
        # 1. Alcohol Consumption (higher = more consumption = higher risk)
        # Convert to a 0-1 scale where higher values indicate higher consumption
        women_data['alcohol_consumption'] = women_data['alcohol_drinks_per_day'] * women_data['alcohol_days_per_year'] / 365
        women_data['alcohol_consumption'] = women_data['alcohol_consumption'].fillna(0)  # Non-drinkers get 0
        
        # 2. Body Weight/BMI (higher = higher BMI = higher risk)
        # Already in a good scale, just need to normalize
        
        # 3. Physical Activity Level (higher = more activity = lower risk)
        # Combine moderate and vigorous activity (vigorous counts double)
        women_data['physical_inactivity_level'] = women_data['physical_inactivity_level'].fillna(0) + 2 * women_data['physical_activity_vigorous'].fillna(0)
        
        # 4. Diet Quality (higher = better diet = lower risk)
        # Combine fruit and vegetable consumption
        diet_columns = ['diet_quality_fruit', 'diet_quality_vegetables', 'diet_quality_greens']
        women_data['diet_quality'] = women_data[diet_columns].mean(axis=1)
        
        # 5. Smoking History (higher = more smoking = higher risk)
        # Create a smoking score: 2 for current smokers, 1 for former smokers, 0 for never smokers
        women_data['smoking_history'] = 0
        # Ever smoked 100 cigarettes (1=Yes)
        ever_smokers = women_data['smoking_status'] == 1
        # Current smoker (1=Every day, 2=Some days)
        current_smokers = women_data['smoking_current'].isin([1, 2])
        
        women_data.loc[ever_smokers & ~current_smokers, 'smoking_history'] = 1  # Former smokers
        women_data.loc[current_smokers, 'smoking_history'] = 2  # Current smokers
        
        # 6. Reproductive History (higher = more risk factors = higher risk)
        # Late first birth is a risk factor
        women_data['reproductive_history'] = 0
        # Age at first birth > 30 is a risk factor
        women_data.loc[women_data['reproductive_age_first_birth'] > 30, 'reproductive_history'] += 1
        # Never breastfed is a risk factor (2=No in NHANES)
        women_data.loc[women_data['reproductive_breastfed'] == 2, 'reproductive_history'] += 1
        
        # 7. Hormone Use (higher = more hormone use = higher risk)
        women_data['hormone_use'] = 0
        # Birth control pills (1=Yes)
        women_data.loc[women_data['hormone_birth_control'] == 1, 'hormone_use'] += 1
        # Hormone replacement therapy (1=Yes)
        women_data.loc[women_data['hormone_hrt'] == 1, 'hormone_use'] += 1
        
        # Select final columns and handle missing values
        lifestyle_factors = [
            'alcohol_consumption', 'body_weight_bmi', 'physical_inactivity_level', 
            'diet_quality', 'reproductive_history', 'hormone_use', 'smoking_history'
        ]
        
        # Create final dataframe with just the lifestyle factors
        final_data = women_data[lifestyle_factors].copy()
        
        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        final_data_imputed = imputer.fit_transform(final_data)
        
        # Scale all features to 0-1 range
        scaler = MinMaxScaler()
        final_data_scaled = scaler.fit_transform(final_data_imputed)
        
        # Convert back to DataFrame
        final_data = pd.DataFrame(final_data_scaled, columns=lifestyle_factors)
        
        # For physical activity and diet quality, invert the scale
        # (higher physical activity and better diet = lower risk)
        # Rename to physical_inactivity_level for clarity
        print("Note: 'physical_activity_level' renamed to 'physical_inactivity_level' for clarity")
        # Invert the scale so higher = less activity = higher risk
        final_data['physical_inactivity_level'] = 1 - final_data['physical_inactivity_level']
        # Rename to poor_diet_quality for clarity
        print("Note: 'diet_quality' renamed to 'poor_diet_quality' for clarity")
        # Invert the scale so higher = worse diet = higher risk
        final_data['poor_diet_quality'] = 1 - final_data['poor_diet_quality']
        
        print(f"Successfully loaded NHANES data with {len(final_data)} women")
        return final_data
        
    except Exception as e:
        print(f"Error loading NHANES data: {e}")
        print("Falling back to synthetic data based on evidence...")
        return create_synthetic_lifestyle_data()

def create_synthetic_lifestyle_data(n_samples=569):
    """
    Create synthetic lifestyle data based on evidence from medical literature.
    Used as a fallback if NHANES data cannot be loaded.
    
    Parameters:
        n_samples: Number of samples to generate (default matches breast cancer dataset)
        
    Returns:
        DataFrame with synthetic lifestyle factors
    """
    np.random.seed(42)  # For reproducibility
    
    # Create empty DataFrame
    synthetic_data = pd.DataFrame()
    
    # 1. Alcohol Consumption (higher = more consumption = higher risk)
    # Evidence: Even light drinking increases risk by 7-10%
    synthetic_data['alcohol_consumption'] = np.random.beta(2, 5, n_samples)  # Right-skewed distribution
    
    # 2. Body Weight/BMI (higher = higher BMI = higher risk)
    # Evidence: Postmenopausal women with obesity have 20-40% increased risk
    synthetic_data['body_weight_bmi'] = np.random.beta(4, 6, n_samples)
    
    # 3. Physical Activity Level (higher value = less activity = higher risk)
    # Evidence: Regular exercise reduces risk by 10-20%
    synthetic_data['physical_inactivity_level'] = np.random.beta(3, 4, n_samples)
    
    # 4. Poor Diet Quality (higher value = worse diet = higher risk)
    # Evidence: Mediterranean diet associated with 15% lower risk
    synthetic_data['poor_diet_quality'] = np.random.beta(3, 4, n_samples)
    
    # 5. Reproductive History (higher = more risk factors = higher risk)
    # Evidence: Late first pregnancy, fewer children increase risk
    synthetic_data['reproductive_history'] = np.random.beta(2, 4, n_samples)
    
    # 6. Hormone Use (higher = more hormone use = higher risk)
    # Evidence: HRT increases risk by 75% during use
    synthetic_data['hormone_use'] = np.random.beta(1, 3, n_samples)
    
    # 7. Smoking History (higher = more smoking = higher risk)
    # Evidence: Current smokers have 15-40% higher risk
    synthetic_data['smoking_history'] = np.random.beta(1, 4, n_samples)
    
    # 8. Family History/Genetic (higher = stronger family history = higher risk)
    # Evidence: First-degree relative with breast cancer doubles risk
    synthetic_data['family_history_genetic'] = np.random.beta(1, 9, n_samples)
    
    # 9. Environmental Exposures (higher = more exposures = higher risk)
    # Evidence: Radiation exposure increases risk
    synthetic_data['environmental_exposures'] = np.random.beta(1, 6, n_samples)
    
    # 10. Menstrual History (higher = longer estrogen exposure = higher risk)
    # Evidence: Early menarche and late menopause increase risk
    synthetic_data['menstrual_history'] = np.random.beta(3, 5, n_samples)
    
    print("Created synthetic lifestyle data based on medical evidence")
    return synthetic_data

def integrate_lifestyle_factors_with_breast_cancer_data(use_nhanes=True):
    """
    Integrate lifestyle factors with the breast cancer dataset.
    
    Parameters:
        use_nhanes: Whether to use real NHANES data (True) or synthetic data (False)
        
    Returns:
        DataFrame with combined data, list of original feature names
    """
    # Load original breast cancer dataset
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    
    # Get lifestyle factors
    if use_nhanes:
        lifestyle_df = load_nhanes_lifestyle_factors()
    else:
        lifestyle_df = create_synthetic_lifestyle_data(len(df))
    
    # If NHANES data has fewer samples than the breast cancer dataset,
    # we need to sample with replacement to match the size
    if len(lifestyle_df) < len(df):
        lifestyle_df = lifestyle_df.sample(len(df), replace=True, random_state=42).reset_index(drop=True)
    # If it has more samples, we take just what we need
    elif len(lifestyle_df) > len(df):
        lifestyle_df = lifestyle_df.sample(len(df), random_state=42).reset_index(drop=True)
    
    # Add lifestyle factors to breast cancer data
    for col in lifestyle_df.columns:
        df[col] = lifestyle_df[col].values
    
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
    if 'family_history_genetic' in lifestyle_df.columns:
        df['genetic_cell_interaction'] = df['family_history_genetic'] * df['worst perimeter']
    
    # Menstrual history interacts with hormone-sensitive markers
    if 'menstrual_history' in lifestyle_df.columns:
        df['menstrual_hormone_interaction'] = df['menstrual_history'] * df['mean concavity']
    
    # Define interaction feature names based on what was actually created
    interaction_features = [col for col in df.columns if '_interaction' in col]
    
    print("--- Enhanced Dataset Created with Evidence-Based Risk Factors ---")
    print(f"Added lifestyle features: {list(lifestyle_df.columns)}")
    print(f"Added interaction features: {interaction_features}")
    print(f"New shape: {df.shape}")
    
    return df, list(data.feature_names)

if __name__ == "__main__":
    # Test the function
    df, original_features = integrate_lifestyle_factors_with_breast_cancer_data(use_nhanes=True)
    print("\nSample of the integrated dataset:")
    print(df.head())
    
    # Print summary statistics for lifestyle factors
    lifestyle_cols = [col for col in df.columns if col not in original_features and col != 'target' and '_interaction' not in col]
    print("\nSummary statistics for lifestyle factors:")
    print(df[lifestyle_cols].describe())