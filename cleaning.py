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

import pandas as pd

# Load the dataset
df = pd.read_csv(r"D:\churn-prediciton-project-1\dataset\churn-bigml-20.csv")

# --- Data Cleaning and Preprocessing ---

# 1. Standardize column names to snake_case
df.columns = df.columns.str.lower().str.replace(' ', '_')

# 2. Encode binary categorical features
df['international_plan'] = df['international_plan'].apply(lambda x: 1 if x == 'Yes' else 0)
df['voice_mail_plan'] = df['voice_mail_plan'].apply(lambda x: 1 if x == 'Yes' else 0)
df['churn'] = df['churn'].astype(int)

# 3. One-Hot Encode the 'state' column
df = pd.get_dummies(df, columns=['state'], drop_first=True)

# 4. Drop redundant 'charge' columns and area_code (often has low predictive power)
# The charge is directly proportional to minutes, so it's redundant.
charge_cols = ['total_day_charge', 'total_eve_charge', 'total_night_charge', 'total_intl_charge']
df = df.drop(charge_cols, axis=1)
df = df.drop('area_code', axis=1) # Also dropping area code

# --- Save the Cleaned Dataset ---
df.to_csv('dataset/churn_cleaned.csv', index=False)

print("--- Cleaned Data Head ---")
print(df.head())
print("\n--- Cleaned Data Info ---")
df.info()
