# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

# ----------------------------
# Step 1: Load and Inspect Data
# ----------------------------
# Read CSV file 
df = pd.read_csv('loan_prediction.csv')

# Display basic info and first few rows
print("Initial Data Info:")
print(df.info())
print("\nFirst 5 Rows:")
print(df.head())

# -----------------------------
# Step 2: Visualize Missing Values
# -----------------------------
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()

# -----------------------------
# Step 3: Handle Missing Values
# -----------------------------
# Fill categorical missing values with mode
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])

# Fill numerical missing values with median
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median())

# -----------------------------------
# Step 4: Clean and Convert Features
# -----------------------------------
# Convert '3+' in Dependents to 3 (numeric)
df['Dependents'] = df['Dependents'].replace('3+', '3').astype(int)

# Standardize Loan_Status to uppercase (Y/N)
df['Loan_Status'] = df['Loan_Status'].str.upper()

# -----------------------------------
# Step 5: Class Balance Check
# -----------------------------------
plt.figure(figsize=(6, 4))
sns.countplot(x='Loan_Status', data=df)
plt.title('Class Distribution (Before Balancing)')
plt.show()

print("\nClass Distribution (Before Balancing):")
print(df['Loan_Status'].value_counts(normalize=True))

# -----------------------------------
# Step 6: Encode Categorical Variables
# -----------------------------------
# Identify categorical columns
categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']

# Apply One-Hot Encoding (drop first to avoid dummy trap)
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Encode target variable (Loan_Status) to 0/1
df_encoded['Loan_Status'] = df_encoded['Loan_Status'].map({'Y': 1, 'N': 0})

# -----------------------------------
# Step 7: Separate Features and Target
# -----------------------------------
X = df_encoded.drop(['Loan_ID', 'Loan_Status'], axis=1)
y = df_encoded['Loan_Status']

# -----------------------------------
# Step 8: Balance Classes with SMOTE
# -----------------------------------
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Recombine balanced data
balanced_df = pd.concat([X_balanced, y_balanced], axis=1)


# -----------------------------------
# Step 9: Final Data Visualization
# -----------------------------------
plt.figure(figsize=(10, 6))
sns.pairplot(balanced_df[['ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Loan_Status']])
plt.suptitle('Final Data Distribution', y=1.02)
plt.show()

# -----------------------------------
# Step 10: Save Processed Data
# -----------------------------------
balanced_df.to_csv('processed_loan_data.csv', index=False)
print("\nPreprocessing complete! Processed data saved to 'processed_loan_data.csv'")
