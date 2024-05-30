import streamlit as st
import shap
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import statsmodels as sm 
import pandas as pd 
import statsmodels.api as sm
from streamlit_shap import st_shap
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from sklearn.linear_model import Ridge, LogisticRegression
def load_model_liner(X, y):
    model = sm.OLS(y, X).fit()
    return model

   
if __name__ == '__main__':
    st.title('🤝SHAP交互作用')
    
    '***'
    st.header('SHAP交互作用热力图')
    selcected_features=st.session_state.selected_features
    interaction_values=st.session_state.interaction_values
    interaction_matrix = np.abs(interaction_values).sum(0)
    shap_values=st.session_state.shap_values
    select_var=st.session_state['select_var']
    X=st.session_state.X
    y=st.session_state.y
    outcome=st.session_state.outcome
    with st.container():      
        #SHAP interaction values        
        heat= go.Figure(data=go.Heatmap(z=interaction_matrix,
                                        x=selcected_features,
                                        y=selcected_features,
                                        ))
        st.plotly_chart(heat)
        '说明：SHAP交互作用值热力图，可视化特征之间的交互，可以展示特征之间的关联性。'
        
    # #-------------------shap 依赖性图-----------------------------------------------
    # '***'
    # with st.container():
    #     st.header('SHAP 依赖性图')
    #     # st.write(shap_values)
    #     st_shap(shap.dependence_plot(select_var,shap_values, X),height=400)
    #     '说明：SHAP 依赖性图，可视化特征之间的交互，可以展示特征之间的关联性。'
        
    #-------------------限制性立方样条曲线拟合---------------------------------------
    '***'
    with st.container():
        st.header('限制性立方样条曲线拟合')
        col1,col2=st.columns(2)


        selected_x=shap_values[:,select_var].data
        selected_y=shap_values[:,select_var].values
        
        selected_df=pd.DataFrame({select_var:selected_x,'shap_'+select_var:selected_y})
        
        selcected_features2=np.delete(selcected_features, np.where(selcected_features==select_var))
        
        subgroup_feature=st.selectbox('选择分层变量(要求是分组变量)',selcected_features2)
        
        with col1:
            n_knots=st.slider("n_knots", 2, 10, 3, 1)
        with col2:
            degree=st.slider("degree", 1, 10, 2, 1) 
        
        filter_x=shap_values[:,subgroup_feature].data
        filter_y=shap_values[:,subgroup_feature].values
        
        filter_df=pd.DataFrame({subgroup_feature:filter_x,'shap_'+subgroup_feature:filter_y})
        
        combined_df=pd.concat([selected_df,filter_df],axis=1)
        combined_df['shap_sum']=combined_df['shap_'+select_var]+combined_df['shap_'+subgroup_feature]
        combined_df=combined_df.sort_values(by=select_var)

        x_spline=combined_df[select_var].values
        y_spline=combined_df['shap_sum']
        

        model_spline = make_pipeline(SplineTransformer(n_knots=n_knots, degree=degree), Ridge(alpha=1e-3))
        model_spline.fit(x_spline.reshape(-1, 1), y_spline)
        y_plot = model_spline.predict(x_spline.reshape(-1, 1))
        
        combined_df['y_plot']=y_plot
        
            #分层数据            
        fig_subgroup=px.scatter(combined_df,x=select_var,y='shap_'+select_var,color=subgroup_feature,title='分层数据的散点图')
        fig_subgroup.add_trace(go.Scatter(x=x_spline,y=y_plot,mode='lines',name='spline',line=dict(color='red')))

        st.plotly_chart(fig_subgroup)

        fig_sub_line=px.line(combined_df,x=select_var,y='y_plot',color=subgroup_feature,title='分层数据的线性拟合')
        st.plotly_chart(fig_sub_line)
        
    #-------------------亚组单因素回归----------------------------------
    '***'
    with st.container():
        st.header('分层单因素回归分析')
        '根据交互作用变量进行分组，如果数据的趋势发生变化，说明变量间具有交互作用。'
        label=st.selectbox('选择亚组变量的标签',combined_df[subgroup_feature].unique())
        key_number=st.number_input("设置"+select_var+"切点", min_value=0.0, max_value=float(max(combined_df[select_var])), value=float(np.median(combined_df[select_var])), step=1.0)
        direction=st.radio('拟合方向',('关键点之前','关键点之后'),horizontal=True)
        if direction=='关键点之前':
            X_subgroup=combined_df[(combined_df[subgroup_feature]==label)&(combined_df[select_var]<=key_number)][select_var].values
            y_subgroup=combined_df[(combined_df[subgroup_feature]==label)&(combined_df[select_var]<=key_number)]['shap_sum'].values
        else:
            X_subgroup=combined_df[(combined_df[subgroup_feature]==label)&(combined_df[select_var]>key_number)][select_var].values
            y_subgroup=combined_df[(combined_df[subgroup_feature]==label)&(combined_df[select_var]>key_number)]['shap_sum'].values
            
        model_liner = load_model_liner(X=X_subgroup, y=np.abs(y_subgroup))
        st.write(model_liner.summary())
            
        
        
        
    