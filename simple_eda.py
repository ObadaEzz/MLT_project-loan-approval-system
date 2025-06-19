import json
from collections import Counter
import math

def calculate_basic_stats(data_list):
    """حساب الإحصائيات الأساسية"""
    if not data_list:
        return {}
    
    n = len(data_list)
    total = sum(data_list)
    mean = total / n
    
    # حساب الانحراف المعياري
    variance = sum((x - mean) ** 2 for x in data_list) / n
    std_dev = math.sqrt(variance)
    
    return {
        'count': n,
        'mean': round(mean, 2),
        'std': round(std_dev, 2),
        'min': min(data_list),
        'max': max(data_list)
    }

def create_approval_distribution(requests):
    """إنشاء توزيع الموافقات"""
    if not requests:
        return None
    
    approval_counts = Counter(req.prediction for req in requests)
    total = len(requests)
    
    # التأكد من وجود قيم
    approved_count = approval_counts.get('Approved', 0)
    rejected_count = approval_counts.get('Rejected', 0)
    
    data = {
        'labels': ['Approved', 'Rejected'],
        'values': [approved_count, rejected_count],
        'percentages': [round((approved_count / total) * 100, 1), round((rejected_count / total) * 100, 1)]
    }
    
    return data

def create_income_analysis(requests):
    """تحليل الدخل"""
    if not requests:
        return None
    
    try:
        incomes = [req.applicant_income for req in requests if req.applicant_income is not None]
        approved_incomes = [req.applicant_income for req in requests if req.prediction == 'Approved' and req.applicant_income is not None]
        rejected_incomes = [req.applicant_income for req in requests if req.prediction == 'Rejected' and req.applicant_income is not None]
        
        return {
            'all_incomes': calculate_basic_stats(incomes) if incomes else {},
            'approved_incomes': calculate_basic_stats(approved_incomes) if approved_incomes else {},
            'rejected_incomes': calculate_basic_stats(rejected_incomes) if rejected_incomes else {}
        }
    except Exception as e:
        print(f"Error in income analysis: {e}")
        return None

def create_loan_amount_analysis(requests):
    """تحليل مبالغ القروض"""
    if not requests:
        return None
    
    try:
        amounts = [req.loan_amount for req in requests if req.loan_amount is not None]
        approved_amounts = [req.loan_amount for req in requests if req.prediction == 'Approved' and req.loan_amount is not None]
        rejected_amounts = [req.loan_amount for req in requests if req.prediction == 'Rejected' and req.loan_amount is not None]
        
        return {
            'all_amounts': calculate_basic_stats(amounts) if amounts else {},
            'approved_amounts': calculate_basic_stats(approved_amounts) if approved_amounts else {},
            'rejected_amounts': calculate_basic_stats(rejected_amounts) if rejected_amounts else {}
        }
    except Exception as e:
        print(f"Error in loan amount analysis: {e}")
        return None

def create_categorical_analysis(requests, field_name):
    """تحليل المتغيرات الفئوية"""
    if not requests:
        return None
    
    try:
        # الحصول على قيم الحقل
        field_values = []
        for req in requests:
            if hasattr(req, field_name):
                value = getattr(req, field_name)
                if value is not None:
                    field_values.append(value)
        
        if not field_values:
            return None
        
        # حساب التوزيع
        value_counts = Counter(field_values)
        
        # حساب نسب الموافقة لكل قيمة
        approval_rates = {}
        for value in value_counts.keys():
            approved_count = sum(1 for req in requests 
                               if getattr(req, field_name) == value and req.prediction == 'Approved')
            total_count = value_counts[value]
            approval_rate = (approved_count / total_count) * 100 if total_count > 0 else 0
            approval_rates[value] = round(approval_rate, 1)
        
        return {
            'value_counts': dict(value_counts),
            'approval_rates': approval_rates
        }
    except Exception as e:
        print(f"Error in categorical analysis for {field_name}: {e}")
        return None

def create_credit_history_analysis(requests):
    """تحليل التاريخ الائتماني"""
    if not requests:
        return None
    
    try:
        credit_histories = [req.credit_history for req in requests if req.credit_history is not None]
        approved_credit = [req.credit_history for req in requests if req.prediction == 'Approved' and req.credit_history is not None]
        rejected_credit = [req.credit_history for req in requests if req.prediction == 'Rejected' and req.credit_history is not None]
        
        return {
            'all_credit': calculate_basic_stats(credit_histories) if credit_histories else {},
            'approved_credit': calculate_basic_stats(approved_credit) if approved_credit else {},
            'rejected_credit': calculate_basic_stats(rejected_credit) if rejected_credit else {}
        }
    except Exception as e:
        print(f"Error in credit history analysis: {e}")
        return None

def perform_simple_eda(requests):
    """تنفيذ تحليل البيانات الاستكشافي البسيط"""
    if not requests:
        return {
            'error': 'لا توجد بيانات متاحة للتحليل'
        }
    
    try:
        results = {
            'total_requests': len(requests),
            'approval_distribution': create_approval_distribution(requests),
            'income_analysis': create_income_analysis(requests),
            'loan_amount_analysis': create_loan_amount_analysis(requests),
            'credit_history_analysis': create_credit_history_analysis(requests),
            'gender_analysis': create_categorical_analysis(requests, 'gender'),
            'married_analysis': create_categorical_analysis(requests, 'married'),
            'education_analysis': create_categorical_analysis(requests, 'education'),
            'property_area_analysis': create_categorical_analysis(requests, 'property_area'),
            'self_employed_analysis': create_categorical_analysis(requests, 'self_employed')
        }
        
        return results
        
    except Exception as e:
        print(f"Error in perform_simple_eda: {e}")
        return {
            'error': f'خطأ في تحليل البيانات: {str(e)}'
        } 