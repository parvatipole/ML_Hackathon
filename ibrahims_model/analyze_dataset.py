import pandas as pd
import numpy as np

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('UNSW_NB15_training-set.csv')

# Basic information
print(f"Dataset shape: {df.shape}")
print("\nData types:")
print(df.dtypes)

# Label distribution
print("\nLabel distribution:")
print(df['label'].value_counts())
print(f"Percentage of anomalies: {df['label'].mean() * 100:.2f}%")

# Attack category distribution
print("\nAttack category distribution:")
print(df['attack_cat'].value_counts())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum().sum())

# Save basic statistics to a file
print("\nSaving basic statistics to stats.txt...")
with open('stats.txt', 'w') as f:
    f.write(f"Dataset shape: {df.shape}\n\n")
    f.write("Label distribution:\n")
    f.write(f"{df['label'].value_counts().to_string()}\n\n")
    f.write("Attack category distribution:\n")
    f.write(f"{df['attack_cat'].value_counts().to_string()}\n\n")
    f.write("Data types:\n")
    f.write(f"{df.dtypes.to_string()}\n\n")
    f.write("Basic statistics for numeric columns:\n")
    f.write(f"{df.describe().to_string()}\n\n")

print("Analysis complete!") 