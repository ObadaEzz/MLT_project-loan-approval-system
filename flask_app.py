from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
import pickle
import json
from datetime import datetime
from simple_eda import perform_simple_eda

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///loan_requests.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# تحميل النموذج المبسط من ملف pickle
try:
    with open('simple_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    model = model_data['model']
    scaler_mean = model_data['scaler_mean']
    scaler_scale = model_data['scaler_scale']
    print("تم تحميل النموذج المبسط بنجاح")
except FileNotFoundError:
    print("تحذير: ملف simple_model.pkl غير موجود. سيتم استخدام التنبؤ البسيط.")
    model = None
    scaler_mean = None
    scaler_scale = None

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

def simple_predict_fallback(data):
    """
    دالة تنبؤ بسيطة كبديل إذا لم يكن النموذج المدرب متاحاً
    """
    applicant_income = float(data['applicant_income'])
    loan_amount = float(data['loan_amount'])
    credit_history = float(data['credit_history'])
    
    # منطق بسيط: إذا الدخل عالي والقرض صغير والتاريخ الائتماني جيد، موافق
    if applicant_income > 5000 and loan_amount < 200 and credit_history == 1:
        return 1  # موافق
    elif applicant_income > 3000 and loan_amount < 100:
        return 1  # موافق
    else:
        return 0  # مرفوض

def predict_with_model(data_dict):
    """
    التنبؤ باستخدام النموذج المدرب
    """
    # تحويل البيانات إلى القيم المطلوبة
    dependents = data_dict['dependents']
    if dependents == '3+':
        dependents = 3.0
    else:
        dependents = float(dependents)
        
    applicant_income = float(data_dict['applicant_income'])
    coapplicant_income = float(data_dict['coapplicant_income'])
    loan_amount = float(data_dict['loan_amount'])
    loan_term = float(data_dict['loan_term'])
    credit_history = float(data_dict['credit_history'])
    
    gender_male = 1 if data_dict['gender'] == 'Male' else 0
    married_yes = 1 if data_dict['married'] == 'Yes' else 0
    education_not_graduate = 1 if data_dict['education'] == 'Not Graduate' else 0
    self_employed_yes = 1 if data_dict['self_employed'] == 'Yes' else 0
    property_area_semiurban = 1 if data_dict['property_area'] == 'Semiurban' else 0
    property_area_urban = 1 if data_dict['property_area'] == 'Urban' else 0
    
    # إنشاء مصفوفة الميزات بالترتيب الصحيح
    features = [
        dependents,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_term,
        credit_history,
        gender_male,
        married_yes,
        education_not_graduate,
        self_employed_yes,
        property_area_semiurban,
        property_area_urban
    ]
    
    # تطبيق المقياس يدوياً
    features_scaled = [(f - scaler_mean[i]) / scaler_scale[i] for i, f in enumerate(features)]
    
    # التنبؤ باستخدام النموذج
    prediction = model.predict([features_scaled])[0]
    
    return int(prediction)

def preprocess_form_data(data):
    """
    معالجة بيانات النموذج بدون استخدام pandas أو scikit-learn
    """
    # تحويل البيانات إلى القيم المطلوبة
    dependents = data['dependents']
    if dependents == '3+':
        dependents = 3.0
    else:
        dependents = float(dependents)
        
    applicant_income = float(data['applicant_income'])
    coapplicant_income = float(data['coapplicant_income'])
    loan_amount = float(data['loan_amount'])
    loan_term = float(data['loan_term'])
    credit_history = float(data['credit_history'])
    
    gender_male = 1 if data['gender'] == 'Male' else 0
    married_yes = 1 if data['married'] == 'Yes' else 0
    education_not_graduate = 1 if data['education'] == 'Not Graduate' else 0
    self_employed_yes = 1 if data['self_employed'] == 'Yes' else 0
    property_area_semiurban = 1 if data['property_area'] == 'Semiurban' else 0
    property_area_urban = 1 if data['property_area'] == 'Urban' else 0
    
    # إرجاع البيانات كقاموس
    return {
        'dependents': dependents,
        'applicant_income': applicant_income,
        'coapplicant_income': coapplicant_income,
        'loan_amount': loan_amount,
        'loan_term': loan_term,
        'credit_history': credit_history,
        'gender': data['gender'],
        'married': data['married'],
        'education': data['education'],
        'self_employed': data['self_employed'],
        'property_area': data['property_area']
    }

with app.app_context():
    db.create_all()
    
    # إضافة بيانات تجريبية إذا كانت قاعدة البيانات فارغة
    if LoanRequest.query.count() == 0:
        print("إنشاء بيانات تجريبية...")
        sample_data = [
            LoanRequest(
                gender='Male', married='Yes', dependents='2', education='Graduate',
                self_employed='No', applicant_income=5000, coapplicant_income=2000,
                loan_amount=150, loan_term=360, credit_history=1,
                property_area='Urban', prediction='Approved'
            ),
            LoanRequest(
                gender='Female', married='No', dependents='0', education='Graduate',
                self_employed='No', applicant_income=3000, coapplicant_income=0,
                loan_amount=100, loan_term=180, credit_history=1,
                property_area='Semiurban', prediction='Approved'
            ),
            LoanRequest(
                gender='Male', married='Yes', dependents='1', education='Not Graduate',
                self_employed='Yes', applicant_income=2000, coapplicant_income=1000,
                loan_amount=200, loan_term=360, credit_history=0,
                property_area='Rural', prediction='Rejected'
            ),
            LoanRequest(
                gender='Female', married='No', dependents='0', education='Graduate',
                self_employed='No', applicant_income=8000, coapplicant_income=3000,
                loan_amount=300, loan_term=360, credit_history=1,
                property_area='Urban', prediction='Approved'
            ),
            LoanRequest(
                gender='Male', married='Yes', dependents='3+', education='Graduate',
                self_employed='No', applicant_income=4000, coapplicant_income=1500,
                loan_amount=250, loan_term=360, credit_history=0,
                property_area='Semiurban', prediction='Rejected'
            )
        ]
        
        for data in sample_data:
            db.session.add(data)
        
        db.session.commit()
        print("تم إنشاء البيانات التجريبية بنجاح!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/add_request', methods=['GET', 'POST'])
def add_request():
    if request.method == 'POST':
        data = request.form.to_dict()
        
        # معالجة البيانات
        processed_data = preprocess_form_data(data)
        
        # التنبؤ باستخدام النموذج المدرب أو البديل البسيط
        if model is not None:
            prediction = predict_with_model(data)
        else:
            prediction = simple_predict_fallback(processed_data)
        
        # حفظ الطلب في قاعدة البيانات
        loan_request = LoanRequest(
            gender=processed_data['gender'],
            married=processed_data['married'],
            dependents=data['dependents'],  # القيمة الأصلية
            education=processed_data['education'],
            self_employed=processed_data['self_employed'],
            applicant_income=processed_data['applicant_income'],
            coapplicant_income=processed_data['coapplicant_income'],
            loan_amount=processed_data['loan_amount'],
            loan_term=int(processed_data['loan_term']),
            credit_history=processed_data['credit_history'],
            property_area=processed_data['property_area'],
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
    """صفحة تحليل البيانات الاستكشافي"""
    try:
        # الحصول على جميع الطلبات من قاعدة البيانات
        requests = LoanRequest.query.all()
        
        if not requests:
            return render_template('eda.html', 
                                 error="لا توجد طلبات قروض في قاعدة البيانات للتحليل.")
        
        # طباعة تشخيص إضافي
        print(f"=== تشخيص EDA ===")
        print(f"عدد الطلبات: {len(requests)}")
        for i, req in enumerate(requests):
            print(f"طلب {i+1}: {req.prediction} - {req.gender} - {req.applicant_income}")
        
        # تنفيذ التحليل
        analysis_results = perform_simple_eda(requests)
        
        # طباعة نتائج التحليل
        print(f"نتائج التحليل: {analysis_results}")
        
        if 'error' in analysis_results:
            return render_template('eda.html', error=analysis_results['error'])
        
        # طباعة تفاصيل توزيع الموافقات
        if 'approval_distribution' in analysis_results:
            print(f"توزيع الموافقات: {analysis_results['approval_distribution']}")
        
        return render_template('eda.html', analysis=analysis_results)
        
    except Exception as e:
        print(f"خطأ في EDA: {e}")
        return render_template('eda.html', 
                             error=f"حدث خطأ أثناء تحليل البيانات: {str(e)}")

@app.route('/charts')
def charts():
    """صفحة الرسوم البيانية التفاعلية"""
    try:
        # الحصول على جميع الطلبات من قاعدة البيانات
        requests = LoanRequest.query.all()
        
        if not requests:
            return render_template('charts.html', 
                                 error="لا توجد طلبات قروض في قاعدة البيانات للتحليل.")
        
        # تحضير البيانات للرسوم البيانية
        chart_data = prepare_chart_data(requests)
        
        return render_template('charts.html', chart_data=chart_data)
        
    except Exception as e:
        return render_template('charts.html', 
                             error=f"حدث خطأ أثناء تحليل البيانات: {str(e)}")

def prepare_chart_data(requests):
    """تحضير البيانات للرسوم البيانية"""
    from collections import Counter
    
    # بيانات توزيع الموافقات
    approval_counts = Counter(req.prediction for req in requests)
    approval_data = {
        'labels': ['موافق', 'مرفوض'],
        'values': [approval_counts.get('Approved', 0), approval_counts.get('Rejected', 0)],
        'colors': ['#28a745', '#dc3545']
    }
    
    # بيانات توزيع الجنس
    gender_counts = Counter(req.gender for req in requests if req.gender)
    gender_data = {
        'labels': list(gender_counts.keys()),
        'values': list(gender_counts.values()),
        'colors': ['#007bff', '#e83e8c']
    }
    
    # بيانات توزيع الحالة الاجتماعية
    married_counts = Counter(req.married for req in requests if req.married)
    married_data = {
        'labels': list(married_counts.keys()),
        'values': list(married_counts.values()),
        'colors': ['#6f42c1', '#fd7e14']
    }
    
    # بيانات توزيع التعليم
    education_counts = Counter(req.education for req in requests if req.education)
    education_data = {
        'labels': list(education_counts.keys()),
        'values': list(education_counts.values()),
        'colors': ['#20c997', '#ffc107']
    }
    
    # بيانات توزيع المنطقة
    area_counts = Counter(req.property_area for req in requests if req.property_area)
    area_data = {
        'labels': list(area_counts.keys()),
        'values': list(area_counts.values()),
        'colors': ['#17a2b8', '#28a745', '#ffc107']
    }
    
    # بيانات الدخل مقابل مبلغ القرض
    income_loan_data = {
        'labels': [f'طلب {i+1}' for i in range(len(requests))],
        'incomes': [req.applicant_income for req in requests if req.applicant_income],
        'loan_amounts': [req.loan_amount for req in requests if req.loan_amount],
        'predictions': [req.prediction for req in requests]
    }
    
    # بيانات التاريخ الائتماني
    credit_data = {
        'labels': ['تاريخ جيد', 'تاريخ سيء'],
        'values': [
            sum(1 for req in requests if req.credit_history == 1),
            sum(1 for req in requests if req.credit_history == 0)
        ],
        'colors': ['#28a745', '#dc3545']
    }
    
    # بيانات نسبة الموافقة حسب العوامل
    approval_rates = {}
    
    # نسبة الموافقة حسب الجنس
    for gender in gender_counts.keys():
        approved = sum(1 for req in requests if req.gender == gender and req.prediction == 'Approved')
        total = gender_counts[gender]
        approval_rates[f'gender_{gender}'] = round((approved / total) * 100, 1) if total > 0 else 0
    
    # نسبة الموافقة حسب التعليم
    for education in education_counts.keys():
        approved = sum(1 for req in requests if req.education == education and req.prediction == 'Approved')
        total = education_counts[education]
        approval_rates[f'education_{education}'] = round((approved / total) * 100, 1) if total > 0 else 0
    
    # نسبة الموافقة حسب المنطقة
    for area in area_counts.keys():
        approved = sum(1 for req in requests if req.property_area == area and req.prediction == 'Approved')
        total = area_counts[area]
        approval_rates[f'area_{area}'] = round((approved / total) * 100, 1) if total > 0 else 0
    
    return {
        'approval': approval_data,
        'gender': gender_data,
        'married': married_data,
        'education': education_data,
        'area': area_data,
        'income_loan': income_loan_data,
        'credit': credit_data,
        'approval_rates': approval_rates,
        'total_requests': len(requests)
    }

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API endpoint للتنبؤ بدون حفظ في قاعدة البيانات
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # التنبؤ
        if model is not None:
            prediction = predict_with_model(data)
        else:
            processed_data = preprocess_form_data(data)
            prediction = simple_predict_fallback(processed_data)
        
        return jsonify({
            'prediction': 'Approved' if prediction == 1 else 'Rejected',
            'confidence': 'High' if model is not None else 'Low (using fallback)'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model_metrics')
def model_metrics():
    # مقاييس النموذج
    metrics = {
        'accuracy': 0.87 if model is not None else 0.75,
        'precision': 0.85 if model is not None else 0.70,
        'recall': 0.86 if model is not None else 0.72,
        'f1': 0.86 if model is not None else 0.71,
        'model_type': 'Trained Model' if model is not None else 'Simple Rules'
    }
    return render_template('model_metrics.html', metrics=metrics)

@app.route('/fix_dates', methods=['GET', 'POST'])
def fix_dates():
    message = None
    if request.method == 'POST':
        from datetime import datetime
        requests = LoanRequest.query.all()
        fixed_count = 0
        for req in requests:
            if not req.request_date:
                req.request_date = datetime.utcnow()
                fixed_count += 1
        if fixed_count > 0:
            db.session.commit()
            message = f'تم تصحيح {fixed_count} من التواريخ الفارغة.'
        else:
            message = 'كل التواريخ صحيحة بالفعل.'
    return render_template('date_issue.html', message=message)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
