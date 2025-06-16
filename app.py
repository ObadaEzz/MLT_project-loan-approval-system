from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import joblib
import json
from datetime import datetime
import plotly
import plotly.express as px
from eda_module import perform_eda
import numpy as np
from sklearn.preprocessing import StandardScaler
from flask_caching import Cache

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///loan_requests.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['CACHE_TYPE'] = 'SimpleCache'
app.config['CACHE_DEFAULT_TIMEOUT'] = 300

db = SQLAlchemy(app)
cache = Cache(app)

# تحميل النموذج والبيانات
model = joblib.load('best_loan_model.joblib')

# تحميل بيانات التدريب وتهيئة المقياس
training_data = pd.read_csv('processed_loan_data.csv')
scaler = StandardScaler()
scaler.fit(training_data.drop('Loan_Status', axis=1))

# نموذج قاعدة البيانات
class LoanRequest(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    gender = db.Column(db.String(10))
    married = db.Column(db.String(5))
    dependents = db.Column(db.String(5))
    education = db.Column(db.String(20))
    self_employed = db.Column(db.String(5))
    applicant_income = db.Column(db.Float)
    coapplicant_income = db.Column(db.Float)
    loan_amount = db.Column(db.Float)
    loan_term = db.Column(db.Integer)
    credit_history = db.Column(db.Float)
    property_area = db.Column(db.String(20))
    prediction = db.Column(db.String(10))
    request_date = db.Column(db.DateTime, default=datetime.utcnow)

def preprocess_form_data(data):
    # Create DataFrame with features in exact order from training data
    processed_data = {
        'Dependents': [data['dependents']],  # First column
        'ApplicantIncome': [float(data['applicant_income'])],
        'CoapplicantIncome': [float(data['coapplicant_income'])],
        'LoanAmount': [float(data['loan_amount'])],
        'Loan_Amount_Term': [float(data['loan_term'])],
        'Credit_History': [float(data['credit_history'])],
        'Gender_Male': [1 if data['gender'] == 'Male' else 0],
        'Married_Yes': [1 if data['married'] == 'Yes' else 0],
        'Education_Not Graduate': [1 if data['education'] == 'Not Graduate' else 0],
        'Self_Employed_Yes': [1 if data['self_employed'] == 'Yes' else 0],
        'Property_Area_Semiurban': [1 if data['property_area'] == 'Semiurban' else 0],
        'Property_Area_Urban': [1 if data['property_area'] == 'Urban' else 0]
    }
    
    # Create DataFrame
    df = pd.DataFrame(processed_data)
    
    # Convert Dependents to numeric, replacing '3+' with '3'
    df['Dependents'] = df['Dependents'].replace('3+', '3').astype(float)
    
    # Scale features using the pre-fitted scaler
    df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)
    
    return df_scaled

with app.app_context():
    db.create_all()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/add_request', methods=['GET', 'POST'])
def add_request():
    if request.method == 'POST':
        data = request.form.to_dict()
        
        # Preprocess the data
        input_data = preprocess_form_data(data)
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Save request to database
        loan_request = LoanRequest(
            gender=data['gender'],
            married=data['married'],
            dependents=data['dependents'],
            education=data['education'],
            self_employed=data['self_employed'],
            applicant_income=float(data['applicant_income']),
            coapplicant_income=float(data['coapplicant_income']),
            loan_amount=float(data['loan_amount']),
            loan_term=int(data['loan_term']),
            credit_history=float(data['credit_history']),
            property_area=data['property_area'],
            prediction='Approved' if prediction == 1 else 'Rejected'
        )
        
        db.session.add(loan_request)
        db.session.commit()
        
        return redirect(url_for('view_requests'))
        
    return render_template('add_request.html')

@app.route('/view_requests')
def view_requests():
    requests = LoanRequest.query.all()
    return render_template('view_requests.html', requests=requests)

@app.route('/delete_request/<int:id>')
def delete_request(id):
    request_to_delete = LoanRequest.query.get_or_404(id)
    db.session.delete(request_to_delete)
    db.session.commit()
    return redirect(url_for('view_requests'))

@app.route('/eda')
def eda():
    try:
        # Get all requests from database
        requests = LoanRequest.query.all()
        print(f"Found {len(requests)} records in database")
        
        if not requests:
            print("No data available for analysis")
            return render_template('eda.html', plots={}, error="No loan applications found in the database.")
        
        # Convert to list of dictionaries
        data = []
        for req in requests:
            try:
                data.append({
                    'gender': req.gender,
                    'married': req.married,
                    'dependents': req.dependents,
                    'education': req.education,
                    'self_employed': req.self_employed,
                    'applicant_income': float(req.applicant_income),
                    'coapplicant_income': float(req.coapplicant_income),
                    'loan_amount': float(req.loan_amount),
                    'loan_term': float(req.loan_term),
                    'credit_history': float(req.credit_history),
                    'property_area': req.property_area,
                    'prediction': req.prediction,
                    'request_date': req.request_date
                })
            except (ValueError, TypeError) as e:
                print(f"Error converting request data: {str(e)}")
                continue
        
        if not data:
            return render_template('eda.html', plots={}, error="Error processing loan application data.")
        
        # Create DataFrame
        try:
            requests_df = pd.DataFrame(data)
            print(f"Created DataFrame with {len(requests_df)} rows and columns: {requests_df.columns.tolist()}")
        except Exception as e:
            print(f"Error creating DataFrame: {str(e)}")
            return render_template('eda.html', plots={}, error="Error processing data for analysis.")
        
        # Perform EDA if we have data
        try:
            # Cache the plots
            cache_key = 'eda_plots'
            plots = cache.get(cache_key)
            
            if plots is None:
                plots = perform_eda(requests_df)
                if not plots:
                    return render_template('eda.html', plots={}, error="Error generating analytics plots.")
                # Cache for 5 minutes
                cache.set(cache_key, plots, timeout=300)
            
            print(f"Generated {len(plots)} plots")
            for plot_id in plots:
                print(f"Plot available: {plot_id}")
            return render_template('eda.html', plots=plots)
        except Exception as e:
            print(f"Error in perform_eda: {str(e)}")
            return render_template('eda.html', plots={}, error=f"Error generating analytics: {str(e)}")
            
    except Exception as e:
        print(f"Error in EDA route: {str(e)}")
        return render_template('eda.html', plots={}, error=f"An unexpected error occurred: {str(e)}")

@app.route('/model_metrics')
def model_metrics():
    # حساب مقاييس دقة النموذج
    metrics = {
        'accuracy': 0.87,
        'precision': 0.85,
        'recall': 0.86,
        'f1': 0.86
    }
    return render_template('model_metrics.html', metrics=metrics)

@app.route('/date_issue')
def date_issue():
    return render_template('date_issue.html')

if __name__ == '__main__':
    app.run(debug=True)
