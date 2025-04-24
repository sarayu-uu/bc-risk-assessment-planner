"""
Test script for NHANES library
"""

from nhanes.load import load_NHANES_data
import sys

def test_nhanes():
    try:
        # Try with string format
        print("Trying with string format '2017-2018'...")
        data = load_NHANES_data(['BMXBMI'], '2017-2018')
        print(f"Success! Data shape: {data.shape}")
        return
    except Exception as e:
        print(f"Error with string format: {e}")
    
    try:
        # Try with integer format
        print("Trying with integer format 2017...")
        data = load_NHANES_data(['BMXBMI'], 2017)
        print(f"Success! Data shape: {data.shape}")
        return
    except Exception as e:
        print(f"Error with integer format: {e}")
    
    try:
        # Try with no year parameter
        print("Trying with default year parameter...")
        data = load_NHANES_data(['BMXBMI'])
        print(f"Success! Data shape: {data.shape}")
        return
    except Exception as e:
        print(f"Error with default year: {e}")
    
    print("All attempts failed.")

if __name__ == "__main__":
    print(f"Python version: {sys.version}")
    print(f"NHANES library path: {load_NHANES_data.__module__}")
    test_nhanes()