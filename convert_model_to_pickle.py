import joblib
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

def simple_predict(data_dict, model, scaler_mean, scaler_scale):
    """
    دالة تنبؤ بسيطة بدون استخدام scikit-learn
    
    Args:
        data_dict: قاموس يحتوي على بيانات الطلب
        model: النموذج المدرب
        scaler_mean: متوسط المقياس
        scaler_scale: مقياس المقياس
        
    Returns:
        int: 1 للموافقة، 0 للرفض
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
    features = np.array([
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
    ])
    
    # تطبيق المقياس يدوياً
    features_scaled = (features - scaler_mean) / scaler_scale
    
    # التنبؤ باستخدام النموذج
    prediction = model.predict([features_scaled])[0]
    
    return int(prediction)

def create_simple_prediction_function():
    """
    تحويل النموذج المدرب إلى دالة بايثون بسيطة
    """
    
    # تحميل النموذج والمقياس
    model = joblib.load('best_loan_model.joblib')
    training_data = pd.read_csv('processed_loan_data.csv')
    scaler = StandardScaler()
    scaler.fit(training_data.drop('Loan_Status', axis=1))
    
    # استخراج معاملات المقياس
    scaler_mean = scaler.mean_
    scaler_scale = scaler.scale_
    
    # حفظ النموذج والمقياس
    model_data = {
        'model': model,
        'scaler_mean': scaler_mean,
        'scaler_scale': scaler_scale,
        'feature_names': [
            'Dependents', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
            'Loan_Amount_Term', 'Credit_History', 'Gender_Male', 'Married_Yes',
            'Education_Not Graduate', 'Self_Employed_Yes', 'Property_Area_Semiurban',
            'Property_Area_Urban'
        ]
    }
    
    with open('simple_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("تم حفظ النموذج المبسط في simple_model.pkl")
    print("يمكنك الآن استخدام هذا الملف بدون الحاجة لـ scikit-learn أو joblib")

if __name__ == "__main__":
    create_simple_prediction_function() 