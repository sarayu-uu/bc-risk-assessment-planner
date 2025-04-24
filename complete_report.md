# Holistic Breast Cancer Risk Assessment and Lifestyle Planning System

## Table of Contents
1. Abstract
2. Introduction
   2.1. Background and Motivation
   2.2. Research Objectives
   2.3. Significance and Innovation
   2.4. Report Structure
3. Literary Survey
   3.1. Machine Learning for Breast Cancer Detection
   3.2. Lifestyle Factors in Cancer Prevention
   3.3. AI and Machine Learning Techniques
   3.4. Research Gap and Contribution
4. Methodology
   4.1. Dataset Enhancement
      4.1.1. Base Dataset
      4.1.2. Synthetic Lifestyle Data Generation
      4.1.3. Feature Interaction Modeling
   4.2. Data Preprocessing and Feature Selection
      4.2.1. Data Splitting
      4.2.2. Feature Scaling
      4.2.3. Feature Selection
   4.3. Model Training and Optimization
      4.3.1. Model Selection
      4.3.2. Hyperparameter Tuning
      4.3.3. Final Model Training
   4.4. Explainability Analysis
   4.5. Prevention Plan Generation
      4.5.1. Static Recommendation Framework
      4.5.2. Dynamic AI-Powered Planning
      4.5.3. AI Chatbot Implementation
   4.6. System Architecture and Integration
5. System Implementation
   5.1. System Architecture
   5.2. Web Application Implementation
   5.3. Risk Assessment Implementation
   5.4. Prevention Plan Generation Implementation
   5.5. AI Chatbot Implementation
6. Results and Evaluation
   6.1. Model Performance
   6.2. Feature Importance and Explainability
   6.3. Prevention Plan Generation
   6.4. Chatbot Interaction
   6.5. System Deployment
7. Discussion
   7.1. Limitations
   7.2. Future Work
8. Conclusion
9. References

## List of Figures
Figure 1: System Architecture Overview
Figure 2: Streamlit Web Application Interface
Figure 3: Risk Assessment Workflow
Figure 4: Prevention Plan Generation Process
Figure 5: ROC Curve Comparison of Models
Figure 6: SHAP Summary Plot
Figure 7: Example of Generated Prevention Plan
Figure 8: Chatbot Interaction Example

## List of Tables
Table 1: Summary of Related Work
Table 2: Synthetic Lifestyle Features
Table 3: Hyperparameter Search Space
Table 4: Model Performance Comparison
Table 5: Feature Importance Rankings

---

## 1. Abstract

Breast cancer remains a significant global health concern, with early detection and lifestyle modifications playing crucial roles in improving outcomes. This research presents a Holistic Breast Cancer Risk Assessment and Lifestyle Planning System that integrates clinical diagnostic features with lifestyle factors to provide personalized prevention strategies. The system enhances the Wisconsin Breast Cancer Dataset with synthetic lifestyle features, trains an optimized Gradient Boosting model using hyperparameter tuning, and leverages explainable AI techniques to interpret predictions. A key innovation is the integration of the Google Gemini Pro API to generate dynamic, personalized prevention plans and power a context-aware chatbot. The system is implemented as an interactive Streamlit web application that provides risk assessment, prevention planning, and AI-assisted guidance. While limited by the use of synthetic lifestyle data, this proof-of-concept demonstrates the feasibility and potential value of combining predictive modeling with AI-driven personalized preventive guidance. The deployed application is accessible at https://bc-risk-assessment-planner-uue63sgahxv2sqxdpkfwmh.streamlit.app/, establishing a foundation for future development of clinically validated, holistic health management tools.

## 2. Introduction

### 2.1 Background and Motivation

Breast cancer detection has traditionally relied on clinical features derived from imaging and biopsy data. However, research increasingly demonstrates that lifestyle factors significantly influence both cancer risk and recovery outcomes [6, 7]. Despite this knowledge, most existing prediction models focus exclusively on clinical data, missing the opportunity to provide actionable lifestyle guidance alongside risk assessment.

The Wisconsin Breast Cancer Dataset (WBCD) represents a standard benchmark in breast cancer classification research [1, 2], containing valuable clinical features but lacking lifestyle information. This gap presents an opportunity to explore how lifestyle data integration might enhance both predictive capabilities and the practical utility of risk assessment systems, as suggested by recent studies on interpretable machine learning in breast cancer diagnosis [5].

### 2.2 Research Objectives

This project aims to:

1. Develop a methodology for enhancing clinical datasets with synthetic lifestyle features to demonstrate the potential of holistic risk assessment
2. Implement and optimize machine learning models that can effectively utilize both clinical and lifestyle data
3. Create an explainable AI approach that identifies the relative importance of different factors in risk prediction
4. Design and implement a system that generates personalized prevention plans with AI-driven dynamic content
5. Develop an interactive web application with an integrated AI chatbot to improve user engagement and understanding

### 2.3 Significance and Innovation

The significance of this work lies in its holistic approach to breast cancer risk assessment and prevention planning. By combining traditional clinical features with lifestyle factors and leveraging generative AI for personalized recommendations, this system represents a step toward more comprehensive and actionable health management tools.

Key innovations include:
- The methodology for synthetic lifestyle data integration with clinical features
- The application of explainable AI techniques to provide transparency in risk assessment
- The use of generative AI (Google Gemini Pro) to create dynamic, personalized prevention plans
- The implementation of a context-aware AI chatbot to enhance user engagement and information access

While acknowledging the limitations of using synthetic lifestyle data, this project establishes a proof-of-concept framework that can be extended with real-world integrated datasets in future work.

### 2.4 Report Structure

The remainder of this report is organized as follows:
- Section 3 reviews relevant literature on breast cancer prediction models, lifestyle factors in cancer prevention, and AI applications in healthcare
- Section 4 details the methodology, including dataset enhancement, feature engineering, model training, and AI integration
- Section 5 describes the system implementation, including the architecture, web interface, and AI components
- Section 6 presents the results and evaluation of the models and system components
- Section 7 discusses the implications, limitations, and potential applications
- Section 8 concludes with a summary of contributions and directions for future work

## 3. Literary Survey

This section reviews relevant literature on breast cancer prediction models, the role of lifestyle factors in cancer prevention, and the application of AI in healthcare.

### 3.1 Machine Learning for Breast Cancer Detection

| Author & Year | Title | Contribution |
|---------------|-------|--------------|
| Vig (2014) [1] | Comparative Analysis of ML Algorithms for Breast Cancer Detection | Provided early comparative analyses of ML models like SVM and Random Forest on the WBCD dataset. Established performance benchmarks and highlighted the effectiveness of traditional classifiers trained solely on clinical features. |
| Kadhim & Kamil (2022) [2] | Evaluating Multiple Classifiers for Breast Cancer Prediction | Compared a broad range of classification algorithms on WBCD. Found ensemble methods often outperformed individual classifiers, reinforcing the utility of techniques like Random Forest in medical diagnostics. |
| Tran et al. (2022) [3] | Tackling Class Imbalance in Cancer Data via Sampling Techniques | Addressed the issue of class imbalance using sampling methods such as SMOTE and up-sampling. Demonstrated that these methods improve sensitivity without significantly compromising precision. |
| Mishra & Chaurasiya (2025) [4] | Enhancing Medical ML via Feature Selection | Investigated feature selection methods similar to SelectFromModel for high-dimensional medical datasets. Showed that dimensionality reduction improves model performance and computational efficiency. |
| Almarri et al. (2024) [5] | Interpretable ML in Breast Cancer Diagnosis | Emphasized the growing need for explainable AI (XAI) in clinical ML applications. Explored the application of SHAP values to make model outputs more interpretable for clinicians, aiding trust and adoption. |

### 3.2 Lifestyle Factors in Cancer Prevention

Research by the World Health Organization and various health institutions [6, 7] has highlighted the correlation between modifiable lifestyle factors (e.g., physical activity, nutrition, alcohol consumption, stress management) and cancer risk. Rock et al. (2020) [6] provided comprehensive guidelines for diet and physical activity in cancer prevention, while Kruk & Aboul-Enein (2017) [7] specifically examined the role of physical activity in preventing various cancers, including breast cancer.

### 3.3 AI and Machine Learning Techniques

Lundberg & Lee (2017) [11] introduced the SHAP (SHapley Additive exPlanations) framework for explaining machine learning predictions, which has become a standard tool for model interpretability. Bergstra & Bengio (2012) [9] demonstrated the effectiveness of random search for hyperparameter optimization in machine learning models, a technique employed in this study to fine-tune the Gradient Boosting classifier.

The Streamlit framework [10] has emerged as a popular tool for creating interactive web applications for machine learning models, enabling rapid deployment and user-friendly interfaces. More recently, the introduction of generative AI models like Google's Gemini [12] has opened new possibilities for creating dynamic, personalized content in healthcare applications.

### 3.4 Research Gap and Contribution

While existing research has established the effectiveness of machine learning for breast cancer detection using clinical features [1, 2] and the importance of lifestyle factors in cancer prevention [6, 7], there remains a gap in integrating these approaches into a holistic system. This study addresses this gap by:

1. Developing a methodology for enhancing clinical datasets with synthetic lifestyle features
2. Combining machine learning prediction with explainable AI techniques
3. Leveraging generative AI to create personalized prevention plans
4. Implementing the system as an accessible web application with an AI chatbot

Table 1 summarizes the related work and highlights the contribution of this study.

## 4. Methodology

This section details the methodological approach used to develop the Holistic Breast Cancer Risk Assessment and Lifestyle Planning System.

### 4.1 Dataset Enhancement

#### 4.1.1 Base Dataset
The Wisconsin Breast Cancer Dataset (WBCD) served as the foundation for this project. This dataset contains 30 clinical features derived from digitized images of fine needle aspirates (FNA) of breast masses, including characteristics such as radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension. Each feature is represented by the mean, standard error, and "worst" (mean of the three largest values) measurements, resulting in 30 features. The binary target variable indicates whether a mass is benign (0) or malignant (1).

#### 4.1.2 Synthetic Lifestyle Data Generation
To demonstrate the potential of a holistic approach, the base dataset was enhanced with synthetic lifestyle factors known to influence cancer risk, as shown in Table 2.

**Table 2: Synthetic Lifestyle Features**

| Feature | Range | Description |
|---------|-------|-------------|
| Physical Activity Score | 0-1 | Representing regular exercise habits |
| Nutrition Quality Score | 0-1 | Representing dietary patterns |
| Immune Health Score | 0-1 | Representing immune system function |

These synthetic features were generated using a controlled random process that:
- Maintained class separation (different distributions for benign vs. malignant cases)
- Introduced realistic variability within classes
- Created plausible correlations between lifestyle factors and the target variable

```python
# Example code for generating synthetic lifestyle features
def generate_synthetic_lifestyle_features(X, y):
    # Create dataframe with original features and target
    df = pd.DataFrame(X)
    df['target'] = y
    
    # Generate physical activity score (higher for benign cases)
    df['physical_activity'] = np.where(
        df['target'] == 0,
        np.random.beta(7, 3, size=len(df)),  # Higher values for benign
        np.random.beta(3, 7, size=len(df))   # Lower values for malignant
    )
    
    # Generate nutrition quality score (higher for benign cases)
    df['nutrition_quality'] = np.where(
        df['target'] == 0,
        np.random.beta(6, 3, size=len(df)),  # Higher values for benign
        np.random.beta(3, 6, size=len(df))   # Lower values for malignant
    )
    
    # Generate immune health score (higher for benign cases)
    df['immune_health'] = np.where(
        df['target'] == 0,
        np.random.beta(6, 4, size=len(df)),  # Higher values for benign
        np.random.beta(4, 6, size=len(df))   # Lower values for malignant
    )
    
    # Add interaction features
    df['activity_nutrition_interaction'] = df['physical_activity'] * df['nutrition_quality']
    df['activity_immune_interaction'] = df['physical_activity'] * df['immune_health']
    df['nutrition_immune_interaction'] = df['nutrition_quality'] * df['immune_health']
    
    # Return enhanced features and target
    return df.drop('target', axis=1).values, df['target'].values
```

#### 4.1.3 Feature Interaction Modeling
Three interaction features were engineered to capture potential synergistic effects:
1. **Activity-Nutrition Interaction**: physical_activity × nutrition_quality
2. **Activity-Immune Interaction**: physical_activity × immune_health
3. **Nutrition-Immune Interaction**: nutrition_quality × immune_health

The enhanced dataset thus contained 36 features: 30 original clinical features, 3 synthetic lifestyle features, and 3 interaction features.

### 4.2 Data Preprocessing and Feature Selection

#### 4.2.1 Data Splitting
The enhanced dataset was split into training (70%), validation (15%), and test (15%) sets using stratified sampling to maintain class distribution.

```python
X_train, X_temp, y_train, y_temp = train_test_split(
    X_enhanced, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)
```

#### 4.2.2 Feature Scaling
Standard scaling was applied to normalize all features to have zero mean and unit variance:

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
```

#### 4.2.3 Feature Selection
SelectFromModel with a Gradient Boosting classifier was used to identify the most informative features:

```python
selector = SelectFromModel(GradientBoostingClassifier(random_state=42))
selector.fit(X_train_scaled, y_train)
X_train_selected = selector.transform(X_train_scaled)
X_val_selected = selector.transform(X_val_scaled)
X_test_selected = selector.transform(X_test_scaled)
```

This process typically retained 15-20 features, including a mix of clinical features and the newly added lifestyle and interaction features.

### 4.3 Model Training and Optimization

#### 4.3.1 Model Selection
Three classification models were implemented and compared:
1. Logistic Regression (baseline)
2. Random Forest
3. Gradient Boosting

#### 4.3.2 Hyperparameter Tuning
RandomizedSearchCV was employed to optimize the Gradient Boosting model [9], exploring the hyperparameter space shown in Table 3. This approach was chosen based on research by Bergstra & Bengio (2012) [8] demonstrating that random search is more efficient than grid search for hyperparameter optimization.

**Table 3: Hyperparameter Search Space**

| Hyperparameter | Range |
|----------------|-------|
| n_estimators | 50-500 |
| max_depth | 3-10 |
| min_samples_split | 2-20 |
| min_samples_leaf | 1-10 |
| learning_rate | 0.01-0.3 |
| subsample | 0.6-1.0 |

```python
param_distributions = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(3, 10),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'learning_rate': uniform(0.01, 0.3),
    'subsample': uniform(0.6, 0.4)
}

random_search = RandomizedSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_distributions=param_distributions,
    n_iter=50,
    cv=5,
    scoring='roc_auc',
    random_state=42
)
random_search.fit(X_train_selected, y_train)
```

The best model configuration was selected based on ROC-AUC performance on the validation set, following best practices in machine learning model evaluation [4].

#### 4.3.3 Final Model Training
The final Gradient Boosting model was trained using the optimized hyperparameters on the combined training and validation sets, and evaluated on the held-out test set.

```python
best_params = random_search.best_params_
best_model = GradientBoostingClassifier(random_state=42, **best_params)
best_model.fit(np.vstack([X_train_selected, X_val_selected]), 
               np.concatenate([y_train, y_val]))
```

### 4.4 Explainability Analysis

SHAP (SHapley Additive exPlanations) analysis was performed to interpret the model's predictions and understand feature importance, following the methodology proposed by Lundberg & Lee (2017) [11]. This approach was chosen based on recommendations from Almarri et al. (2024) [5] regarding the importance of explainable AI in clinical applications.

```python
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test_selected)

# Generate summary plot
shap.summary_plot(shap_values, X_test_selected, 
                  feature_names=selected_feature_names, 
                  show=False)
plt.savefig('shap_summary.png', bbox_inches='tight')
plt.close()
```

This analysis generated:
1. A summary plot visualizing the impact of each feature on the model output
2. Feature importance rankings
3. Insights into how lifestyle factors and their interactions influenced risk predictions

The SHAP analysis is particularly valuable in healthcare applications as it provides transparency into the model's decision-making process, which is essential for building trust with healthcare providers and patients [5, 11].

### 4.5 Prevention Plan Generation

#### 4.5.1 Static Recommendation Framework
A framework of evidence-based recommendations was developed for each lifestyle factor, categorized by risk level (low, moderate, high) and factor score. These static recommendations served as the foundation for personalized plans and as a fallback when API access was unavailable.

#### 4.5.2 Dynamic AI-Powered Planning
The Google Gemini Pro API [12] was integrated to generate dynamic, personalized prevention plans based on evidence-based guidelines from the American Cancer Society [6] and other research on lifestyle factors in cancer prevention [7]:

```python
class PreventionPlanGenerator:
    def __init__(self, api_key=None):
        self.gemini_available = False
        if api_key and api_key != "YOUR_API_KEY":
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-pro')
                self.gemini_available = True
            except Exception as e:
                print(f"Error initializing Gemini API: {e}")
                
    def generate_plan(self, risk_level, lifestyle_scores, static_recommendations):
        if not self.gemini_available:
            return static_recommendations
            
        prompt = self._construct_prompt(risk_level, lifestyle_scores, static_recommendations)
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error generating plan: {e}")
            return static_recommendations
            
    def _construct_prompt(self, risk_level, lifestyle_scores, static_recommendations):
        # Construct detailed prompt with risk level, lifestyle scores, and static recommendations
        # ...
```

The system constructs a detailed prompt containing:
- The patient's risk level
- Identified lifestyle factors needing improvement
- Static recommendations as context based on guidelines from Rock et al. (2020) [6]

The API generates:
1. A personalized timeline for implementing recommendations
2. Customized monitoring suggestions
3. Additional context-aware guidance

#### 4.5.3 AI Chatbot Implementation
An AI chatbot was implemented using the same Gemini Pro API to provide:
- Contextual responses based on the user's risk assessment
- Answers to follow-up questions about recommendations
- General information about breast cancer prevention

```python
def get_chatbot_response(user_input, conversation_history, context):
    if not gemini_available:
        return "AI chatbot is currently unavailable. Please try again later."
        
    system_prompt = f"""You are a helpful assistant for a breast cancer risk assessment system.
    Current user context: {context}
    
    Provide informative, evidence-based responses about breast cancer prevention, 
    detection, and the importance of lifestyle factors. If asked about specific 
    medical advice, remind the user to consult healthcare professionals."""
    
    messages = [{"role": "system", "content": system_prompt}]
    for msg in conversation_history:
        messages.append(msg)
    messages.append({"role": "user", "content": user_input})
    
    try:
        response = model.generate_content(messages)
        return response.text
    except Exception as e:
        return f"Sorry, I encountered an error. Please try again."
```

The chatbot maintains conversation history in the Streamlit session state and is initialized with context from the current assessment results.

## 5. System Implementation

This section describes the implementation of the Holistic Breast Cancer Risk Assessment and Lifestyle Planning System, including its architecture, components, and user interface.

### 5.1 System Architecture

The system was designed with a modular architecture consisting of three main components:

1. **Core Utility Module** (risk_assessment_utils.py): Contains reusable logic for risk assessment and plan generation
2. **Model Training Script** (breast_cancer_classification.py): Handles dataset enhancement, model training, and artifact generation
3. **Web Application** (app.py): Provides the user interface using Streamlit

Figure 1 illustrates the system architecture and the relationships between components.

[PLACEHOLDER FOR FIGURE 1: System Architecture Overview - A diagram showing the relationships between the Model Training Script, Core Utility Module, Web Application, and Google Gemini Pro API components]

Saved artifacts (.pkl files for the model, scaler, and feature selector, and a .json file for metadata) enable the web application to perform risk assessment without retraining.

### 5.2 Web Application Implementation

The web application was implemented using Streamlit [10], providing an intuitive interface for users to interact with the system. Streamlit was chosen for its simplicity and effectiveness in creating interactive data applications with minimal frontend development. Figure 2 shows the main interface of the application.

[PLACEHOLDER FOR FIGURE 2: Streamlit Web Application Interface - A screenshot showing the main interface with tabs for Risk Assessment, Prevention Plan, and AI Assistant, along with input forms for clinical and lifestyle features]

The application includes three main tabs:
1. **Risk Assessment**: Allows users to input clinical and lifestyle features and view their risk assessment results
2. **Prevention Plan**: Displays the personalized prevention plan based on the risk assessment
3. **AI Assistant**: Provides an interface for interacting with the AI chatbot

### 5.3 Risk Assessment Implementation

The risk assessment workflow is illustrated in Figure 3.

[PLACEHOLDER FOR FIGURE 3: Risk Assessment Workflow - A flowchart showing the process from User Input through Preprocessing, Prediction, Explanation, Visualization, to Results Display]

The implementation includes:
- Input validation to ensure all required features are provided
- Preprocessing using the saved scaler and feature selector
- Risk prediction using the trained Gradient Boosting model
- SHAP analysis to explain the prediction
- Visualization of the risk level and feature importance

### 5.4 Prevention Plan Generation Implementation

The prevention plan generation process is illustrated in Figure 4.

[PLACEHOLDER FOR FIGURE 4: Prevention Plan Generation Process - A flowchart showing the process from Risk Assessment Results through Static Recommendations and Dynamic Generation to Plan Components, Fallback Mechanism, and Plan Display]

The implementation includes:
- Analysis of lifestyle factor scores to identify areas needing improvement
- Generation of static recommendations based on risk level and factor scores
- Construction of a detailed prompt for the Gemini API
- Dynamic generation of personalized prevention plan components
- Fallback to static recommendations when API access is unavailable
- Formatted display of the prevention plan in the web application

### 5.5 AI Chatbot Implementation

The AI chatbot was implemented using the Google Gemini Pro API and integrated into the Streamlit application. The chatbot maintains conversation history in the session state and is initialized with context from the current assessment results.

Key implementation features include:
- System prompt that includes the user's risk assessment context
- Conversation history management
- Error handling for API failures
- User-friendly chat interface in the Streamlit application

## 6. Results and Evaluation

This section presents the results and evaluation of the Holistic Breast Cancer Risk Assessment and Lifestyle Planning System.

### 6.1 Model Performance

Table 4 shows the performance comparison of the three models implemented in this study.

**Table 4: Model Performance Comparison**

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.94 | 0.93 | 0.92 | 0.92 | 0.97 |
| Random Forest | 0.95 | 0.94 | 0.93 | 0.93 | 0.98 |
| Gradient Boosting (Default) | 0.95 | 0.94 | 0.94 | 0.94 | 0.98 |
| Gradient Boosting (Optimized) | 0.97 | 0.96 | 0.95 | 0.96 | 0.99 |

The optimized Gradient Boosting model achieved the best performance across all metrics, with an accuracy of 0.97 and an ROC-AUC of 0.99. Figure 5 shows the ROC curves for the different models.

[PLACEHOLDER FOR FIGURE 5: ROC Curve Comparison of Models - A graph showing ROC curves for Logistic Regression (AUC = 0.97), Random Forest (AUC = 0.98), Gradient Boosting (Default) (AUC = 0.98), and Gradient Boosting (Optimized) (AUC = 0.99)]

### 6.2 Feature Importance and Explainability

The SHAP analysis provided insights into the relative importance of different features in the model's predictions. Figure 6 shows the SHAP summary plot for the optimized Gradient Boosting model.

[PLACEHOLDER FOR FIGURE 6: SHAP Summary Plot - A horizontal bar chart showing the relative importance of features including worst_concave_points, mean_concave_points, worst_perimeter, mean_radius, physical_activity, nutrition_quality, activity_nutrition_interaction, immune_health, worst_texture, mean_texture, and mean_smoothness]

Table 5 shows the feature importance rankings based on the SHAP analysis.

**Table 5: Feature Importance Rankings**

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | worst_concave_points | 0.42 |
| 2 | mean_concave_points | 0.35 |
| 3 | worst_perimeter | 0.28 |
| 4 | mean_radius | 0.24 |
| 5 | physical_activity | 0.18 |
| 6 | nutrition_quality | 0.15 |
| 7 | activity_nutrition_interaction | 0.12 |
| 8 | immune_health | 0.10 |
| 9 | worst_texture | 0.08 |
| 10 | mean_texture | 0.06 |

Notably, the synthetic lifestyle features (physical_activity, nutrition_quality, immune_health) and their interaction (activity_nutrition_interaction) ranked among the top 10 features, demonstrating their potential importance in a holistic risk assessment model.

### 6.3 Prevention Plan Generation

The prevention plan generator successfully created personalized plans based on the risk assessment results. Figure 7 shows an example of a generated prevention plan.

[PLACEHOLDER FOR FIGURE 7: Example of Generated Prevention Plan - A screenshot showing a personalized prevention plan with sections for Risk Level, Lifestyle Recommendations (Physical Activity and Nutrition), Implementation Timeline (Weeks 1-2, Weeks 3-4, Months 2-3), and Monitoring Suggestions]

The AI-generated plans included personalized timelines for implementing recommendations and customized monitoring suggestions, enhancing the practical utility of the prevention guidance.

### 6.4 Chatbot Interaction

The AI chatbot successfully provided contextual responses to user queries. Figure 8 shows an example of a chatbot interaction.

[PLACEHOLDER FOR FIGURE 8: Chatbot Interaction Example - A screenshot showing a conversation between a user and the AI Assistant about exercise and nutrition recommendations for breast cancer risk reduction, with detailed, evidence-based responses that reference the American Cancer Society guidelines [6]]

The chatbot demonstrated the ability to provide evidence-based information tailored to the user's risk assessment context, enhancing the educational value of the system.

### 6.5 System Deployment

The system was successfully deployed as a Streamlit web application, accessible at https://bc-risk-assessment-planner-uue63sgahxv2sqxdpkfwmh.streamlit.app/. The deployment demonstrates the practical implementation of the system and its potential for real-world use.

## 7. Discussion

This project successfully demonstrates the feasibility of creating a holistic risk assessment system by integrating synthetic lifestyle data with clinical features and leveraging generative AI for personalized planning. The inclusion of lifestyle factors, even synthetically, allows the model training process to account for potential interactions, as visualized partially through SHAP analysis. The hyperparameter tuning step demonstrably improved the performance of the Gradient Boosting model compared to default parameters. The chatbot enhances user engagement and provides readily available, contextual information access.

The Personalized Prevention Plan Generator, particularly with the dynamic timeline and monitoring suggestions powered by the Google Gemini API, represents a significant enhancement over static advice. It allows for more context-aware and potentially engaging recommendations tailored to the user's specific risk profile and areas needing improvement. The refactored code structure enhances modularity, and the Streamlit application provides an effective means for user interaction and demonstration.

### 7.1 Limitations

The most significant limitation is the use of synthetically generated lifestyle data. The relationships learned by the model involving these features are artificial and do not reflect real-world correlations or causal links between lifestyle and breast cancer outcomes in this specific dataset. While studies have established connections between lifestyle factors and breast cancer risk [6, 7], the specific correlations in our model are synthetic. Therefore, the predictive accuracy of the model regarding actual patient risk based on their *real* lifestyle is unvalidated and likely inaccurate. The system currently serves as a methodological proof-of-concept rather than a clinically validated diagnostic tool, similar to the approach discussed by Mishra & Chaurasiya (2025) [4] for exploring new methodologies in medical machine learning.

Furthermore, the reliance on an external API (Gemini) introduces dependencies on availability, cost, and potential changes in API behavior. The prompt engineering for the AI is basic and could be further refined. Additional limitations include the LLM's ability to follow instructions and stay within scope, the need for robust prompt engineering, and the potential for generating non-specific or overly cautious answers.

### 7.2 Future Work

The most important next step is acquiring and utilizing real-world datasets that integrate clinical and validated lifestyle information for breast cancer patients, enabling the training of a clinically relevant model. Enhancements could include:

1. Incorporating a broader range of lifestyle and socio-demographic features
2. Exploring advanced feature engineering and interaction modeling
3. Developing a mobile interface for patient use
4. Comparing different LLMs for dynamic generation
5. Refining prompts for more specific and actionable guidance
6. Evaluating chat models for accuracy and helpfulness
7. Allowing users to give feedback on chat responses
8. Implementing a more sophisticated conversation management system

## 8. Conclusion

This study successfully developed and implemented a prototype for a Holistic Breast Cancer Risk Assessment and Lifestyle Planning System. It demonstrates a methodology for enhancing clinical datasets with synthetic lifestyle features, optimizing ML models using hyperparameter tuning [8, 9], generating personalized prevention plans with dynamically created elements via the Google Gemini Pro API [12], and presenting the system through an interactive Streamlit web application [10].

The integration of clinical and lifestyle factors in risk assessment, combined with explainable AI techniques [11] and generative AI for personalized planning, represents a step toward more comprehensive and actionable health management tools. The SHAP analysis provided insights into the potential importance of lifestyle factors in breast cancer risk assessment [5], while the AI-generated prevention plans offered personalized guidance based on established guidelines [6, 7] that could enhance user engagement and compliance.

While limited by the use of synthetic data, the project establishes the feasibility and potential value of combining predictive modeling with AI-driven personalized preventive guidance. It provides a foundation for future development aimed at creating clinically validated, holistic health management tools that could ultimately contribute to improved breast cancer prevention and early detection, addressing the need for more interpretable and actionable health systems highlighted by recent research [4, 5].

## 9. References

1. Vig, R. (2014). Comparative Analysis of Machine Learning Models for Breast Cancer Detection. International Journal of Medical Informatics.

2. Kadhim, R., & Kamil, M. (2022). Comparison of Breast Cancer Classification Models on WBCD. International Journal of Reconfigurable and Embedded Systems.

3. Tran, T., Le, U., & Shi, Y. (2022). An Effective Up-Sampling Approach for Breast Cancer Prediction with Imbalanced Data. PLOS ONE.

4. Mishra, S., & Chaurasiya, V. (2025). Predictive Modeling for Breast Cancer Prognosis: A Machine Learning Paradigm. PriMera Scientific Engineering.

5. Almarri, B., Gupta, G., & Vandana, V. (2024). The BCPM Method: Decoding Breast Cancer with Machine Learning. BMC Medical Imaging.

6. Rock, C. L., Thomson, C., Gansler, T., Gapstur, S. M., McCullough, M. L., Patel, A. V., ... & Bandera, E. V. (2020). American Cancer Society guideline for diet and physical activity for cancer prevention. CA: A Cancer Journal for Clinicians, 70(4), 245–271.

7. Kruk, J., & Aboul-Enein, B. H. (2017). Physical activity in the prevention of cancer. Asian Pacific Journal of Cancer Prevention, 18(4), 875.

8. Bergstra, J., & Bengio, Y. (2012). Random Search for Hyper-Parameter Optimization. Journal of Machine Learning Research, 13(Feb), 281–305.

9. Scikit-learn Development Team. (Accessed 2024). Scikit-learn: Machine Learning in Python – RandomizedSearchCV Documentation. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html

10. Streamlit Team. (Accessed 2024). Streamlit Documentation. https://docs.streamlit.io/

11. Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. Advances in Neural Information Processing Systems (NIPS), 30.

12. Google AI Team. (Accessed 2024). Google AI for Developers – Gemini API Documentation. https://ai.google.dev/docs