#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Load the model
#model = joblib.load('XGBoost.pkl')


# Age（年龄）：指受检者的年龄
# Sex（性别）：指受检者的性别，通常为“男”或“女”。
# GLU（血糖）：血液中的葡萄糖水平，正常范围通常为：空腹血糖：70-99 mg/dL 餐后2小时：<140 mg/dL
# CREA（肌酐）：反映肾功能的指标，正常范围一般为：男性：0.74-1.35 mg/dL 女性：0.59-1.04 mg/dL
# DBIL（直接胆红素）：反映肝脏的功能和胆汁的排泄，正常范围一般为：0.1-0.3 mg/dL
# 
# IBIL（间接胆红素）：与直接胆红素相对，正常范围一般为：0.2-0.8 mg/dL
# 
# UA（尿酸）：反映身体内尿酸的水平，正常范围通常为：男性：3.4-7.0 mg/dL 女性：2.4-6.0 mg/dL
# 
# AST（天冬氨酸氨基转移酶）：肝功能的指标之一，正常范围通常为：10-40 U/L
# 
# ALT（丙氨酸氨基转移酶）：另一个反映肝功能的指标，正常范围一般为：7-56 U/L
# UREA（尿素）：反映肾功能及蛋白质代谢的指标，正常范围一般为：7-20 mg/dL
# TBIL（总胆红素）：血液中的胆红素总量，正常范围通常为：0.1-1.2 mg/dL
# BUN/Scr（血尿素氮与肌酐比率）：评估肾功能的一种指标，比率通常在10:1到20:1之间。
# 

# In[ ]:





# In[3]:


import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib


# In[4]:


# 加载模型
model = joblib.load("C:\\Users\\lenovo\\毕设-20212133050\\XGBoost.pkl")  # 替换为你的模型路径


# In[5]:


# 页面标题
st.title('Alzheimer\'s Disease Prediction')

feature_names = ['Age', 'Sex', 'GLU', 'CREA', 'DBIL', 'IBIL', 'UA', 'AST', 'ALT', 'UREA', 'TBIL', 'BUN/Scr']

# 创建表单
with st.form("data_form"):
    age = st.number_input('年龄', min_value=0, max_value=120, value=50, step=1)
    sex = st.selectbox("Sex (0=Female, 1=Male):", 
                       options=[0, 1], format_func=lambda x: 'Female (0)' if x == 0 else 'Male (1)')
    st.write("You selected:", "Female" if sex == 0 else "Male")
    glu = st.number_input('GLU', min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    crea = st.number_input('CREA', min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    dbil = st.number_input('DBIL', min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    ibil = st.number_input('IBIL', min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    ua = st.number_input('UA', min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    ast = st.number_input('AST', min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    alt = st.number_input('ALT', min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    urea = st.number_input('UREA', min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    tbil = st.number_input('TBIL', min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    bun_scr = st.number_input('BUN/Scr', min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    
    submit_button = st.form_submit_button(label='提交')

if submit_button:
    feature_values = [age, sex, glu, crea, dbil, ibil, ua, ast, alt, urea, tbil, bun_scr]
    features = np.array([feature_values], dtype=np.float32)
    
    st.write('您输入的参数如下:')
    st.write(f'年龄: {age}')
    st.write(f'性别: {"Female" if sex == 0 else "Male"}')
    st.write(f'GLU: {glu}')
    # 其他参数显示...
    
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    
    st.write(f"**预测类别:** {predicted_class}")
    st.write(f"**预测概率:** {predicted_proba[predicted_class]:.2f}%")
    
    # 生成建议
    probability = predicted_proba[predicted_class] * 100    
if predicted_class == 1
          advice = f"根据模型预测，您有较高的阿尔茨海默症风险（预测概率为{probability:.2f}%）。建议您尽快前往医院进行进一步检查，以便早期发现和干预。    "
el
            advice = f"根据模型预测，您患阿尔茨海默症的风险较低（预测概率为{probability:.2f}%）。但请注意，这不能完全排除患病的可能性。建议您保持健康的生活方式，定期进行体    检。"
st.write(advice)
    
    # 生成SHAP图
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))
    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=120)
    plt.close()  # 关闭图像
    st.image("shap_force_plot.png")


# In[ ]:




