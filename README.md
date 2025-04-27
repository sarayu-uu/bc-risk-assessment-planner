# Breast Cancer Risk Assessment & AI Planner

This application provides a holistic breast cancer risk assessment and generates personalized prevention plans.

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Required packages (install using `pip install -r requirements.txt`)

### Environment Variables
This application uses environment variables for API keys. To set them up:

1. Copy the `.env.example` file to a new file named `.env`:
   ```
   cp .env.example .env
   ```

2. Get a Google API key for Gemini AI:
   - Visit https://makersuite.google.com/app/apikey
   - Create a new API key
   - Copy the API key

3. Edit the `.env` file and replace the placeholder with your actual API key:
   ```
   GOOGLE_API_KEY=your_actual_api_key_here
   ```

### Running the Application
To run the Streamlit app:
```
streamlit run app.py
```

## Features
- Breast cancer risk assessment based on medical and lifestyle factors
- Personalized prevention plans
- SHAP explanations for model predictions
- AI-powered chatbot for answering questions (requires valid Google API key)

## Note on API Keys
- Never commit your actual API keys to GitHub
- The `.env` file is included in `.gitignore` to prevent accidental commits
- Always use the `.env` file or environment variables to store sensitive information