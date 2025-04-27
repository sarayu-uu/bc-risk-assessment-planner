"""
Setup API Key for Streamlit

This script helps you securely set up your Google API key for use with Streamlit.
It will update your .streamlit/secrets.toml file with your API key.
"""

import os
import toml
from pathlib import Path

def setup_api_key():
    print("\n=== Streamlit API Key Setup ===\n")
    
    # Get the API key from the user
    api_key = input("Enter your Google API key: ").strip()
    
    if not api_key:
        print("No API key provided. Exiting.")
        return
    
    # Create .streamlit directory if it doesn't exist
    streamlit_dir = Path(".streamlit")
    streamlit_dir.mkdir(exist_ok=True)
    
    # Path to secrets.toml
    secrets_path = streamlit_dir / "secrets.toml"
    
    # Create or update secrets.toml
    secrets = {}
    if secrets_path.exists():
        try:
            secrets = toml.load(secrets_path)
            print("Loaded existing secrets.toml file.")
        except Exception as e:
            print(f"Error loading existing secrets.toml: {e}")
    
    # Update the API key
    secrets["GOOGLE_API_KEY"] = api_key
    
    # Write back to secrets.toml
    try:
        with open(secrets_path, "w") as f:
            toml.dump(secrets, f)
        print(f"\nAPI key successfully saved to {secrets_path.absolute()}")
        print("\nYou can now run your Streamlit app with:")
        print("streamlit run app.py")
    except Exception as e:
        print(f"Error saving secrets.toml: {e}")

if __name__ == "__main__":
    setup_api_key()