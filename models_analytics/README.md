# Loan Approval System - Models & Analytics

This part contains the machine learning models and analytics files for the Loan Approval System.

## Contents

- `best_loan_model.joblib` - Trained Random Forest model
- `scaler.joblib` - StandardScaler for feature normalization
- `feature_importance.png` - Feature importance visualization
- `correlation_matrix.png` - Correlation analysis visualization
- `README.md` - This file

## Model Details

### Random Forest Model
- Accuracy: 87%
- Precision: 85%
- Recall: 86%
- F1 Score: 86%

### Model Features
1. Applicant Information
   - Gender
   - Marital Status
   - Dependents
   - Education
   - Employment Status

2. Financial Information
   - Applicant Income
   - Co-applicant Income
   - Loan Amount
   - Loan Term
   - Credit History

3. Property Information
   - Property Area

## Analytics Files

### feature_importance.png
- Shows the relative importance of each feature
- Helps in understanding key decision factors
- Used for model interpretation

### correlation_matrix.png
- Displays relationships between features
- Helps identify multicollinearity
- Used for feature selection

## Integration

These files are used by:
1. Core backend for predictions
2. EDA module for analysis
3. Report generation for insights
4. Frontend for visualization

## Usage

1. Load the model:
   ```python
   import joblib
   model = joblib.load('best_loan_model.joblib')
   ```

2. Load the scaler:
   ```python
   scaler = joblib.load('scaler.joblib')
   ```

3. Make predictions:
   ```python
   # Scale features
   scaled_features = scaler.transform(features)
   # Make prediction
   prediction = model.predict(scaled_features)
   ```

## Contributing

1. Create a new branch for your changes
2. Make your modifications
3. Test thoroughly
4. Submit a pull request

## Notes

- The model is trained on processed data
- Features are standardized before prediction
- Model performance is regularly monitored
- Analytics are updated with new data 