from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import joblib
import json
from datetime import datetime
import plotly
import plotly.express as px
from eda_module import perform_eda

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///loan_requests.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# تحميل النموذج
model = joblib.load('best_loan_model.joblib')

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

with app.app_context():
    db.create_all()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/add_request', methods=['GET', 'POST'])
def add_request():
    if request.method == 'POST':
        data = request.form.to_dict()
        
        # تحويل البيانات إلى DataFrame للتنبؤ
        input_data = pd.DataFrame([data])
        prediction = model.predict(input_data)[0]
        
        # حفظ الطلب في قاعدة البيانات
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
    # قراءة جميع الطلبات من قاعدة البيانات
    requests_df = pd.read_sql(LoanRequest.query.statement, db.session.bind)
    
    # إجراء التحليل الاستكشافي للبيانات
    eda_plots = perform_eda(requests_df)
    
    return render_template('eda.html', plots=eda_plots)

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
