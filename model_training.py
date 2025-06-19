# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Load preprocessed data
print("Loading preprocessed data...")
df = pd.read_csv("processed_loan_data.csv")
df = df.astype(float)

# Split features and target
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print('Training set shape:', X_train.shape)
print('Testing set shape:', X_test.shape)

# Define models with their hyperparameters for grid search
models = {
    'Logistic Regression': {
        'model': LogisticRegression(random_state=42, max_iter=10000),
        'params': {
            'C': [0.001, 0.01, 0.1, 1, 10],
            'solver': ['liblinear', 'saga']
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        }
    },
    'XGBoost': {
        'model': XGBClassifier(random_state=42, eval_metric='logloss'),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1]
        }
    }
}

# Function to evaluate models
def evaluate_model(y_true, y_pred, y_pred_proba=None):
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # ROC Curve
    if y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()

# Train and evaluate models
best_models = {}
best_scores = {}
all_metrics = {}

print("\nTraining and evaluating models...")
for name, model_info in models.items():
    print(f"\nTraining {name}...")
    
    # Perform GridSearchCV
    grid_search = GridSearchCV(
        model_info['model'],
        model_info['params'],
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Store best model and score
    best_models[name] = grid_search.best_estimator_
    best_scores[name] = grid_search.best_score_
    
    print(f"Best parameters for {name}:", grid_search.best_params_)
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Make predictions
    y_pred = grid_search.predict(X_test)
    y_pred_proba = grid_search.predict_proba(X_test)[:, 1]
    
    # Calculate all metrics
    all_metrics[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'AUC-ROC': roc_curve(y_test, y_pred_proba)[2],
        'Cross-Val Score': best_scores[name]
    }
    
    print(f"\nEvaluation metrics for {name}:")
    evaluate_model(y_test, y_pred, y_pred_proba)

# Create comprehensive model comparison table
print("\n" + "="*100)
print("Model Comparison Table")
print("="*100)

# Convert metrics to DataFrame
comparison_df = pd.DataFrame(all_metrics).round(4).T

# Sort by accuracy
comparison_df = comparison_df.sort_values('Accuracy', ascending=False)

# Add ranking column
comparison_df['Rank'] = range(1, len(comparison_df) + 1)

# Reorder columns to show rank first
cols = ['Rank'] + [col for col in comparison_df.columns if col != 'Rank']
comparison_df = comparison_df[cols]

# Print comparison table with formatting
print("\nModel Performance Comparison:")
print("-" * 120)
print(comparison_df.to_string())
print("-" * 120)

# Print best model summary
best_model_name = comparison_df.index[0]
print(f"\nBest Model: {best_model_name}")
print(f"Best Model Parameters: {best_models[best_model_name].get_params()}")

# Save the best model and scaler
import joblib
joblib.dump(best_models[best_model_name], 'best_loan_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
print("\nBest model saved as 'best_loan_model.joblib'")
print("Scaler saved as 'scaler.joblib'")

# Plot feature importance for the best model if it's a tree-based model
if hasattr(best_models[best_model_name], 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_models[best_model_name].feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
    plt.title(f'Top 10 Most Important Features ({best_model_name})')
    plt.tight_layout()
    plt.show()