import streamlit as st
from streamlit_shap import st_shap
import shap
from sklearn.model_selection import train_test_split
import xgboost
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer  
from plotly import graph_objects as go



def load_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
    d_train = xgboost.DMatrix(X_train, label=y_train)
    d_test = xgboost.DMatrix(X_test, label=y_test)
    params = {
        "eta": 0.01,
        "objective": "reg:linear",
        "subsample": 0.5,
        "base_score": np.mean(y_train),
        "eval_metric": "rmse",
        "n_jobs": -1,
    }
    model = xgboost.train(params, d_train, 10, evals = [(d_test, "test")], verbose_eval=100, early_stopping_rounds=20)
    return model


if __name__ == '__main__':
    st.title("SHAP in Streamlit")
    X,y = st.session_state['X'], st.session_state['y']
    selected_features=st.session_state['selected_features']
    X=X[selected_features]
    X_display,y_display = X,y

    # st.write(X_display)
    model = load_model(X,y)#load_model(X, y)

    # compute SHAP values
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    with st.container():
        st_shap(shap.plots.beeswarm(shap_values),height=400)
    '***'
    with st.container():
        select_var=st.selectbox('选择变量',X.columns)

        st_shap(shap.plots.scatter(shap_values[:,select_var]))
    if 'shap_values' not in st.session_state:
        st.session_state['shap_values'] = shap_values
    else:
        st.session_state['shap_values'] = shap_values
    '***'
    #立方样条曲线拟合
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
      
        
        plot_data=pd.DataFrame({select_var:shap_values[:,select_var].data,'shap_'+select_var:shap_values[:,select_var].values,'spline':y_plot})
        # plot_data_melt=pd.melt(plot_data,id_vars=[select_var])
        plot_data=plot_data.sort_values(by=select_var)
        #95%置信区间

        data = go.Scatter(x=plot_data[select_var],y=plot_data['shap_'+select_var],mode='markers',name='shap_'+select_var)
        fig2 = go.Figure(data) 
        fig2.add_trace(go.Scatter(x=plot_data[select_var],y=plot_data['spline'],mode='lines',name='spline',line=dict(color='red')))
        
        annotation_number=st.number_input("竖线位置", min_value=0.0, max_value=float(max(plot_data[select_var])), value=float(np.median(plot_data[select_var])), step=1.0)

        fig2.add_vline(x=annotation_number, line_dash='dash', annotation_text='{}'.format(annotation_number))
        # Add the 95% CI as a shaded region
        # fig2.add_trace(go.Scatter(x=plot_data[select_var], y=ci_lower, mode='lines', name='CI lower', line=dict(color='rgba(0, 0, 255, 0.2)')))
        # fig2.add_trace(go.Scatter(x=plot_data[select_var], y=ci_upper, mode='lines', name='CI upper', line=dict(color='rgba(0, 0, 255, 0.2)')))

        st.plotly_chart(fig2)

    # explainer_tree = shap.TreeExplainer(model)
    # shap_values_tree = explainer_tree.shap_values(X)
    # with st.container():
    #     num=st.number_input('选择前几个样本', min_value=0, max_value=X.shape[0]-1, value=1, step=1, key='sample_num')
    # # st_shap(shap.plots.waterfall(shap_values[0]), height=400)
    #     st_shap(shap.force_plot(explainer_tree.expected_value, shap_values_tree[0:num,:], X_display.iloc[0:num,:]), height=400, width=800)
    #     # st_shap(shap.force_plot(explainer_tree.expected_value, shap_values_tree[:num,:], X_display.iloc[:num,:]), height=400)#
    