# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the cleaned dataset
df = pd.read_csv('dataset/churn_cleaned.csv')

# --- 1. Define Features (X) and Target (y) ---
X = df.drop('churn', axis=1)
y = df['churn']

# --- 2. Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 3. Feature Scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 4. Train Logistic Regression Model ---
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

# --- 5. Train Decision Tree Model ---
dec_tree = DecisionTreeClassifier(random_state=42)
dec_tree.fit(X_train_scaled, y_train) # Tree models don't strictly require scaling, but it's fine to use

# --- 6. Evaluate Models ---
# Logistic Regression Evaluation
y_pred_log = log_reg.predict(X_test_scaled)
print("--- Logistic Regression Results ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_log):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log))
print("Classification Report:\n", classification_report(y_test, y_pred_log))


# Decision Tree Evaluation
y_pred_tree = dec_tree.predict(X_test_scaled)
print("\n--- Decision Tree Results ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_tree):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_tree))
print("Classification Report:\n", classification_report(y_test, y_pred_tree))



# (Code from Day 5 is above)
# --- 7. Cross-Validation and Final Model Selection ---
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import joblib

# Cross-validate the Decision Tree
cv_scores = cross_val_score(DecisionTreeClassifier(random_state=42), X, y, cv=5)
print(f"\n--- Decision Tree Cross-Validation Scores ---")
print(f"Scores: {cv_scores}")
print(f"Average CV Score: {cv_scores.mean():.4f}")

# Plot ROC Curve for the Decision Tree
y_pred_proba = dec_tree.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - Decision Tree')
plt.legend(loc="lower right")
plt.show()

# --- 8. Save the Final Model ---
# The Decision Tree is the better model. We'll save the scaler and the model.
model_columns = X.columns
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(dec_tree, 'models/churn_model.pkl')
joblib.dump(model_columns, 'models/model_columns.pkl')

print("\nFinal Decision Tree model and scaler have been saved to the 'models/' directory.")