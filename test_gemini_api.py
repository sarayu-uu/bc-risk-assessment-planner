"""
Test script to verify if the Gemini API key is working correctly.
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key
api_key = os.getenv("GOOGLE_API_KEY")
print(f"API Key found: {api_key[:5]}...{api_key[-5:] if api_key else 'None'}")

try:
    # Configure the Gemini API
    genai.configure(api_key=api_key)
    
    # Try to create a model
    model = genai.GenerativeModel('gemini-1.5-flash')
    print("Model created successfully")
    
    # Try a simple generation
    response = model.generate_content("Hello, how are you?")
    print("\nAPI Response:")
    print(response.text)
    
    print("\nAPI KEY IS WORKING CORRECTLY!")
except Exception as e:
    print(f"\nERROR: {e}")
    print("\nThis usually means:")
    print("1. The API key is invalid")
    print("2. The model name is incorrect")
    print("3. There's a network issue")
    print("\nPlease get a valid API key from https://makersuite.google.com/app/apikey")