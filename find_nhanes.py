"""
Find NHANES library location
"""

import nhanes
import os

print(f"NHANES library location: {nhanes.__file__}")
print(f"NHANES library directory: {os.path.dirname(nhanes.__file__)}")
print("\nListing contents of NHANES directory:")
nhanes_dir = os.path.dirname(nhanes.__file__)
for item in os.listdir(nhanes_dir):
    print(f"  - {item}")