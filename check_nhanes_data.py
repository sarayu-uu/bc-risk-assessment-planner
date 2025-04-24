"""
Check NHANES combined_data directory
"""

import nhanes
import os

nhanes_dir = os.path.dirname(nhanes.__file__)
combined_data_dir = os.path.join(nhanes_dir, 'combined_data')

print(f"NHANES combined_data directory: {combined_data_dir}")
print(f"Directory exists: {os.path.exists(combined_data_dir)}")

if os.path.exists(combined_data_dir):
    print("\nListing contents of combined_data directory:")
    for item in os.listdir(combined_data_dir):
        print(f"  - {item}")
        item_path = os.path.join(combined_data_dir, item)
        if os.path.isdir(item_path):
            print(f"    Contents of {item}:")
            for subitem in os.listdir(item_path):
                print(f"      - {subitem}")
else:
    print("combined_data directory does not exist.")