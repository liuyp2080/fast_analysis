import streamlit as st
from streamlit_shap import st_shap
import shap
from sklearn.model_selection import train_test_split
import xgboost
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer  
from plotly import graph_objects as go
from plotly import express as px
import statsmodels.api as sm

def load_model(X, y, model_type):
    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
    d_train = xgboost.DMatrix(X_train, label=y_train)
    d_test = xgboost.DMatrix(X_test, label=y_test)
    if model_type=='回归':
        params = {
            "eta": 0.01,
            "objective": "reg:linear",
            "subsample": 0.5,
            "base_score": np.mean(y_train),
            "eval_metric": "rmse",
            "n_jobs": -1,
        }
    else:
        params = {
            "eta": 0.01,
            "objective": "binary:logistic",
            "subsample": 0.5,
            "base_score": np.mean(y_train),
            "eval_metric": "auc",
            "n_jobs": -1,
        }
    model = xgboost.train(params, d_train, 10, evals = [(d_test, "test")], verbose_eval=100, early_stopping_rounds=20)
    return model

def load_model_liner(X, y):
    model = sm.OLS(y, X).fit()
    
    return model
if __name__ == '__main__':
    st.title("🧮SHAP 分析")
    X,y = st.session_state['X'], st.session_state['y']
    selected_features=st.session_state['selected_features']
    X=X[selected_features]
    X_display,y_display = X,y

    model_type=st.session_state['model_type']
    model = load_model(X,y,model_type=model_type)#load_model(X, y)

    # compute SHAP values
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    with st.container():
        st_shap(shap.plots.violin(shap_values),height=400)
        '说明：SHAP分析的汇总图，展示每个特征的SHAP值。X轴上的0点代表该特征预测值的平均值，正负代表了预测值的变化的方向（以平均值为参照）。'
    '***'
    with st.container():
        select_var=st.selectbox('选择变量',X.columns)
        if 'select_var' not in st.session_state:
            st.session_state['select_var'] = select_var

        st_shap(shap.plots.scatter(shap_values[:,select_var],))
        '说明：SHAP分析的单个变量的散点图，展示某个特征的SHAP值。'
    if 'shap_values' not in st.session_state:
        st.session_state['shap_values'] = shap_values
    else:
        st.session_state['shap_values'] = shap_values
    '***'
    #-----------------立方样条曲线拟合-----------------
    shap_values=st.session_state['shap_values']
    x_spline=shap_values[:,select_var].data# 特征值
    y_spline=shap_values[:,select_var].values# SHAP值
    
    with st.container():
        st.header('限制性立方样条曲线拟合')
        col1,col2=st.columns(2)
        with col1:
            n_knots=st.slider("n_knots", 2, 10, 3, 1)
        with col2:
            degree=st.slider("degree", 1, 10, 2, 1)
        model_spline = make_pipeline(SplineTransformer(n_knots=n_knots, degree=degree), Ridge(alpha=1e-3))
        model_spline.fit(x_spline.reshape(-1, 1), y_spline)
        y_plot = model_spline.predict(x_spline.reshape(-1, 1))
      
        #仅仅是单个变量的plot数据
        single_plot_data=pd.DataFrame({select_var:shap_values[:,select_var].data,'shap_'+select_var:shap_values[:,select_var].values,'spline':y_plot})
        single_plot_data=single_plot_data.sort_values(by=select_var)

        data_scatter = go.Scatter(x=single_plot_data[select_var],y=single_plot_data['shap_'+select_var],mode='markers',name='shap_'+select_var)
        fig2 = go.Figure(data_scatter,layout=go.Layout(title=go.layout.Title(text='shap_'+select_var+"的限制性立方样条拟合"))) 
        fig2.add_trace(go.Scatter(x=single_plot_data[select_var],y=single_plot_data['spline'],mode='lines',name='spline',line=dict(color='red')))
        
        annotation_number=st.number_input("竖线位置", min_value=0.0, max_value=float(max(single_plot_data[select_var])), value=float(np.median(single_plot_data[select_var])), step=1.0)
        annotation_number2=st.number_input("竖线位置", min_value=0.0, max_value=float(max(single_plot_data[select_var])), value=5+float(np.median(single_plot_data[select_var])), step=1.0)

        fig2.add_vline(x=annotation_number, line_dash='dash', annotation_text='{}'.format(annotation_number))
        fig2.add_vline(x=annotation_number2, line_dash='dash', annotation_text='{}'.format(annotation_number2))
 
        st.plotly_chart(fig2)
        '说明：限制性立方样条曲线拟合以上的散点图。可以用于确定关键的点对应的特征值，比如SHAP值为0的点对应的特征值，或者曲线的拐点。\n'
        '技巧：曲线寻找拐点的时候，可以调高参数n_knots，使其更加精细地拟合数据，借此找到拐点。'
    '***'   
        #SHAP interaction values
    explainer_tree = shap.TreeExplainer(model)
    interaction_values = explainer_tree.shap_interaction_values(X)
    if 'interaction_vlaues' not in st.session_state:
        st.session_state.interaction_values = interaction_values
    else:
        st.session_state.interaction_values = interaction_values
    
    #-----------------------------多因素回归----------------------------------------------------------------------------
    with st.container():
        st.header('单因素回归分析')
        '在通过RCS拟合确定了数据的关键点之后，如果发现关键点前后的数据趋势是不同的，可以通过单因素线性回归来进一步描述关键点前后的趋势，包括给出OR（RR）值，以及相应的统计P值。'
        key_num=st.number_input('关键转折点', min_value=min(single_plot_data[select_var]), max_value=max(single_plot_data[select_var]),  value=np.median(single_plot_data[select_var]), step=1.0)
        direction=st.radio('拟合方向',('关键点之前','关键点之后'),horizontal=True)
        
        #筛选数据，使用原始数据，而不是SHAP数据
        if direction=='关键点之前':
            X_liner=single_plot_data[single_plot_data[select_var]<=key_num][select_var]
            y_liner=single_plot_data[single_plot_data[select_var]<=key_num]['shap_'+select_var]
        else:
            X_liner=single_plot_data[single_plot_data[select_var]>key_num][select_var]
            y_liner=single_plot_data[single_plot_data[select_var]>key_num]['shap_'+select_var]

        model_liner_fit=load_model_liner(X_liner,np.abs(y_liner))
        st.write(model_liner_fit.summary())
        '说明：因为SHAP值的正负代表了预测值的变化的方向，并不是数据的大小，所以在进行单因素分析的时候对SHAP值做了取绝对值的处理。'
        
    