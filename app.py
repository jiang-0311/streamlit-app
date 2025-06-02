#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
import os


# In[3]:
# 自定义 CSS - 统一字体但标题加粗
st.markdown("""
<style>
    /* 统一标题和表单的字体 */
    h1, .stNumberInput, .stSelectbox, .stButton, .stText {
        font-family: 'SimHei', sans-serif;
    }
    
    /* 标题保持加粗但调整大小 */
    h1 {
        font-size: 29px !important;
        font-weight: 700 !important;
        color: #333 !important;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    /* 表单元素字体大小和样式 */
    .stNumberInput input, .stSelectbox select, .stButton button, .stText {
        font-size: 16px !important;
        font-weight: 400 !important;  /* 普通字体粗细 */
    }
    
    /* 美化表单元素 */
    .stNumberInput input, .stSelectbox select {
        border-radius: 5px;
        padding: 0.5rem;
        border: 1px solid #ccc;
    }
    
    .stButton button {
        background-color: #1E88E5;
        color: white;
        border-radius: 5px;
        padding: 0.75rem 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# 加载模型

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'models', 'XGBoost2.pkl')
scaler_path = os.path.join(script_dir, 'models', 'scaler_model.pkl')
                                                             

model =joblib.load(model_path)  
scaler = joblib.load(scaler_path)                                                                   


# In[ ]:


# 页面标题
st.title('Alzheimer\'s Disease Prediction')
feature_names = ['Age','Gender','GLU', 'CREA', 'DBIL', 'IBIL', 'UA', 'AST', 'ALT', 'UREA', 'TBIL','BUN/Scr']

# 创建表单
with st.form("data_form"):

    age = st.number_input('Age', min_value=0, max_value=100, value=50, step=1)
    gender = st.selectbox("Gender (0=Female, 1=Male):", 
                          options=[0, 1], format_func=lambda x: 'Female (0)' if x == 0 else 'Male (1)')
    st.write("You selected:", "Female" if gender == 0 else "Male")
    glu = st.number_input('GLU（mmol/L）', min_value=0.0, max_value=30.0, value=10.0, step=0.1)
    crea = st.number_input('CREA（μmol/L）', min_value=0.0, max_value=200.0, value=50.0, step=0.1)
    dbil = st.number_input('DBIL（μmol/L）', min_value=0.0, max_value=35.0, value=10.0, step=0.1)
    ibil = st.number_input('IBIL（μmol/L）', min_value=0.0, max_value=60.0, value=10.0, step=0.1)
    ua = st.number_input('UA（μmol/L)', min_value=0.0, max_value=500.0, value=50.0, step=0.1)
    ast = st.number_input('AST（U/L）', min_value=0.0, max_value=300.0, value=50.0, step=0.1)
    alt = st.number_input('ALT（U/L）', min_value=0.0, max_value=300.0, value=50.0, step=0.1)
    urea = st.number_input('UREA（mmol/L）', min_value=0.0, max_value=40.0, value=10.0, step=0.1)
    tbil = st.number_input('TBIL（μmol/L）', min_value=0.0, max_value=90.0, value=50.0, step=0.1)
    bun_scr = st.number_input('BUN/Scr', min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    
    submit_button = st.form_submit_button(label='提交')

if submit_button:
    # 获取原始特征值
    feature_values = [age,gender,glu, crea, dbil, ibil, ua, ast, alt, urea, tbil,bun_scr]
    features_raw = np.array([feature_values], dtype=np.float32)
    
    # 打印调试信息
    #st.write("原始数据形状:", features_raw.shape)
    #st.write("标准化器均值:", scaler.mean_)
    #st.write("标准化器标准差:", scaler.scale_)
    
    
    features_scaled = scaler.transform(features_raw)
    

    st.subheader('输入参数与标准化结果')
    col1, col2 = st.columns(2)
    
    with col1:
        st.write('**原始输入值:**')
        st.write(pd.DataFrame([feature_values], columns=feature_names))
        
    with col2:
        st.write('**标准化后值:**')
        st.write(pd.DataFrame(features_scaled, columns=feature_names))
    
    
    predicted_class = model.predict(features_scaled)[0]
    predicted_proba = model.predict_proba(features_scaled)[0]
    
    # 显示
    st.subheader('预测结果')
    if predicted_class == 1:
        st.write("**预测类别:** 阿尔茨海默症")
    else:
        st.write("**预测类别:** 正常")
    st.write(f"**预测概率:** {predicted_proba[predicted_class]:.2f}%")
    
    
     # 生成建议
    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1:
        advice = f"根据模型预测，您有较高的阿尔茨海默症风险（预测概率为{probability:.2f}%）。建议您尽快前往医院进行进一步检查，以便早期发现和干预。"
    else:
        advice = f"根据模型预测，您患阿尔茨海默症的风险较低（预测概率为{probability:.2f}%）。但请注意，这不能完全排除患病的可能性。建议您保持健康的生活方式，定期进行体检。"
    st.write(advice)
    
    # 生成SHAP解释
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame(features_scaled, columns=feature_names))
    shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        pd.DataFrame(features_scaled, columns=feature_names),
        matplotlib=True
    )
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=120)
    plt.close()
    st.image("shap_force_plot.png")


# In[ ]:




