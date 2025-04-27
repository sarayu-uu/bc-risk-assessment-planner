"""
Test script to verify if the Gemini API works in Streamlit.
"""

import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Page config
st.set_page_config(page_title="Gemini API Test", page_icon="🧪")
st.title("Gemini API Test")

# Get API key
api_key = os.getenv("GOOGLE_API_KEY")
st.write(f"API Key found: {api_key[:5]}...{api_key[-5:] if api_key else 'None'}")

try:
    # Configure the Gemini API
    genai.configure(api_key=api_key)
    
    # Try to create a model
    model = genai.GenerativeModel('gemini-1.5-flash')
    st.success("✅ Model created successfully")
    
    # Try a simple generation
    if st.button("Test API"):
        with st.spinner("Generating response..."):
            response = model.generate_content("Hello, how are you?")
            st.write("API Response:")
            st.write(response.text)
        
        st.success("✅ API KEY IS WORKING CORRECTLY!")
except Exception as e:
    st.error(f"ERROR: {e}")
    st.write("This usually means:")
    st.write("1. The API key is invalid")
    st.write("2. The model name is incorrect")
    st.write("3. There's a network issue")
    st.write("Please get a valid API key from https://makersuite.google.com/app/apikey")