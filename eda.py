import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
df_cleaned = pd.read_csv('dataset/churn_cleaned.csv')

# Set plot style
sns.set(style="whitegrid")

# --- 1. Churn Distribution ---
plt.figure(figsize=(6, 4))
sns.countplot(x='churn', data=df_cleaned)
plt.title('Churn Distribution (0 = No, 1 = Yes)')
plt.show()

# --- 2. Churn by International Plan ---
plt.figure(figsize=(7, 5))
sns.countplot(x='international_plan', hue='churn', data=df_cleaned)
plt.title('Churn by International Plan')
plt.xlabel('International Plan (0 = No, 1 = Yes)')
plt.legend(['No Churn', 'Churn'])
plt.show()

# --- 3. Churn by Customer Service Calls ---
plt.figure(figsize=(10, 6))
sns.countplot(x='customer_service_calls', hue='churn', data=df_cleaned)
plt.title('Churn by Number of Customer Service Calls')
plt.xlabel('Customer Service Calls')
plt.legend(['No Churn', 'Churn'])
plt.show()

# --- 4. Churn by Total Day Minutes ---
plt.figure(figsize=(10, 6))
sns.boxplot(x='churn', y='total_day_minutes', data=df_cleaned)
plt.title('Churn vs. Total Day Minutes')
plt.xlabel('Churn (0 = No, 1 = Yes)')
plt.ylabel('Total Day Minutes')
plt.show()

# --- 5. Correlation Heatmap for Numerical Features ---
# We select a subset of main numerical features for readability
numerical_cols = ['account_length', 'number_vmail_messages', 'total_day_minutes',
                  'total_day_calls', 'total_eve_minutes', 'total_eve_calls',
                  'total_night_minutes', 'total_night_calls', 'total_intl_minutes',
                  'total_intl_calls', 'customer_service_calls', 'churn']

plt.figure(figsize=(12, 10))
corr_matrix = df_cleaned[numerical_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Key Features')
plt.show()
