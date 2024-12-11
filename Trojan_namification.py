import pandas as pd
import numpy as np

# Load your dataset without column names
data = pd.read_csv("./CCCS-CIC-Malicious-CSVs/Trojan.csv", header=None)

# Generate random column names
num_columns = len(data.columns)
random_names = [f"feature_{i}" for i in range(num_columns)]

# Assign random column names to the dataset
data.columns = random_names

# Add a new column named "Trojan" filled with One
data['Trojan'] = 1

# Save the dataset with random column names and the "Trojan" column to another file
data.to_csv(".Preprocessing_dataset/namified_Torjan_with_Trojan.csv", index=False)
