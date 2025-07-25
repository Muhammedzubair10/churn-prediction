import pandas as pd
df = pd.read_csv(r"D:\churn-prediciton-project-1\dataset\churn-bigml-20.csv")
print("--- First 5 Rows ---")
print(df.head())

print("\n--- Dataset Shape ---")
print(df.shape)

print("\n--- Dataset Info ---")
df.info()

# Check for null values
print("\n--- Null Value Count ---")
print(df.isnull().sum())

# Get basic statistical details
print("\n--- Statistical Summary ---")
print(df.describe())

# print all rows and columns
print(df.to_string())
