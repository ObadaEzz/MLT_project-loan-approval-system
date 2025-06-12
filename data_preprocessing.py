# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
# Step 2: Feature Engineering
# -----------------------------
# Create new features
df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['EMI'] = df['LoanAmount'] / df['Loan_Amount_Term']
df['Balance_Income'] = df['Total_Income'] - (df['EMI'] * 1000)  # Converting EMI to monthly
df['Income_per_dependent'] = df['Total_Income'] / (df['Dependents'].replace('3+', '3').astype(float) + 1)

# Log transform skewed features
df['ApplicantIncome'] = np.log1p(df['ApplicantIncome'])
df['CoapplicantIncome'] = np.log1p(df['CoapplicantIncome'])
df['LoanAmount'] = np.log1p(df['LoanAmount'])
df['Total_Income'] = np.log1p(df['Total_Income'])

# -----------------------------
# Step 3: Visualize Missing Values
# -----------------------------
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()

print("\nMissing Values Summary:")
print(df.isnull().sum())

# -----------------------------
# Step 4: Handle Missing Values
# -----------------------------
# Fill categorical missing values with mode
categorical_columns = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History']
for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Fill numerical missing values with median
numerical_columns = ['LoanAmount', 'Loan_Amount_Term']
for col in numerical_columns:
    df[col] = df[col].fillna(df[col].median())

# -----------------------------------
# Step 5: Feature Analysis and Visualization
# -----------------------------------
# Correlation Analysis
plt.figure(figsize=(12, 8))
correlation = df.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Distribution of numerical features
numerical_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(2, 2, i)
    sns.histplot(data=df, x=feature, hue='Loan_Status', bins=30)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()

# -----------------------------------
# Step 6: Encode Categorical Variables
# -----------------------------------
# Identify categorical columns
categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Dependents']

# Apply One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Encode target variable (Loan_Status) to 0/1
df_encoded['Loan_Status'] = df_encoded['Loan_Status'].map({'Y': 1, 'N': 0})

# -----------------------------------
# Step 7: Feature Scaling
# -----------------------------------
# Scale numerical features
scaler = StandardScaler()
numerical_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 
                     'Total_Income', 'EMI', 'Balance_Income', 'Income_per_dependent']
df_encoded[numerical_features] = scaler.fit_transform(df_encoded[numerical_features])

# -----------------------------------
# Step 8: Separate Features and Target
# -----------------------------------
X = df_encoded.drop(['Loan_ID', 'Loan_Status'], axis=1)
y = df_encoded['Loan_Status']

# -----------------------------------
# Step 9: Balance Classes with SMOTE
# -----------------------------------
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Show class distribution after balancing
plt.figure(figsize=(8, 5))
sns.countplot(x=y_balanced)
plt.title('Class Distribution After SMOTE')
plt.show()

# Recombine balanced data
balanced_df = pd.concat([pd.DataFrame(X_balanced, columns=X.columns), 
                        pd.Series(y_balanced, name='Loan_Status')], axis=1)

# -----------------------------------
# Step 10: Save Processed Data
# -----------------------------------
balanced_df.to_csv('processed_loan_data.csv', index=False)
print("\nPreprocessing complete! Processed data saved to 'processed_loan_data.csv'")
print("\nShape of processed data:", balanced_df.shape)
