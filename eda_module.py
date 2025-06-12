import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json

def perform_eda(df):
    plots = {}
    
    # 1. توزيع الموافقات/الرفض
    approval_dist = px.pie(df, names='prediction', title='توزيع الموافقات والرفض')
    plots['approval_distribution'] = json.loads(approval_dist.to_json())
    
    # 2. متوسط الدخل حسب حالة القرض
    income_by_status = px.box(df, x='prediction', y='applicant_income', 
                            title='توزيع دخل المتقدمين حسب حالة القرض')
    plots['income_distribution'] = json.loads(income_by_status.to_json())
    
    # 3. توزيع مبالغ القروض
    loan_dist = px.histogram(df, x='loan_amount', 
                           title='توزيع مبالغ القروض',
                           nbins=30)
    plots['loan_amount_distribution'] = json.loads(loan_dist.to_json())
    
    # 4. نسبة الموافقات حسب مستوى التعليم
    education_approval = df.groupby(['education', 'prediction']).size().unstack()
    education_approval_pct = education_approval.div(education_approval.sum(axis=1), axis=0)
    
    education_fig = px.bar(education_approval_pct.reset_index(), 
                          x='education', y=['Approved', 'Rejected'],
                          title='نسبة الموافقات حسب مستوى التعليم')
    plots['education_approval'] = json.loads(education_fig.to_json())
    
    # 5. تحليل الدخل المشترك
    coapplicant_analysis = px.scatter(df, x='applicant_income', y='coapplicant_income',
                                    color='prediction', 
                                    title='العلاقة بين دخل المتقدم ودخل المشارك')
    plots['coapplicant_analysis'] = json.loads(coapplicant_analysis.to_json())
    
    return plots
