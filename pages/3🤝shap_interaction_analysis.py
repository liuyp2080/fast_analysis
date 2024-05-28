import streamlit as st
import shap
import numpy as np
import plotly.graph_objects as go
import statsmodels as sm 
import pandas as pd 
import statsmodels.api as sm


def load_model_liner(X, y,model_type):
    if model_type=='回归': 
        model = sm.OLS(y, X).fit()
        
    else:
        model=sm.GLM(y,X,family=sm.families.Binomial()).fit()
    return model

   
if __name__ == '__main__':
    st.title('🤝SHAP交互作用')
    
    '***'
    st.header('SHAP交互作用热力图')
    selcected_features=st.session_state.selected_features
    interaction_values=st.session_state.interaction_values
    interaction_matrix = np.abs(interaction_values).sum(0)
    with st.container():      
        #SHAP interaction values        
        heat= go.Figure(data=go.Heatmap(z=interaction_matrix,
                                        x=selcected_features,
                                        y=selcected_features,
                                        ))
        st.plotly_chart(heat)
        '说明：SHAP交互作用值热力图，可视化特征之间的交互，可以展示特征之间的关联性。'
        
    #-------------------多因素回归-----------
    '***'
    with st.container():
        st.header('多因素线性回归分析')
        '为了说明变量间的交互作用，采用分层分析+多因素回归方法。'
        df_liner=pd.concat([st.session_state.X[selcected_features],st.session_state.y],axis=1)
        model_type=st.session_state.model_type
        outcome=st.session_state.outcome
        subgroup_feature=st.selectbox('选择分层变量(要求是分组变量)',selcected_features)
        label=st.selectbox('选择分层变量的标签',df_liner[subgroup_feature].unique())
        # st.write(st.session_state)
        submit=st.button('确定')
        if submit:
            X_liner=df_liner[df_liner[subgroup_feature]==label].drop([outcome,subgroup_feature],axis=1)
            y=df_liner[df_liner[subgroup_feature]==label][outcome]
            model_liner = load_model_liner(X=X_liner, y=y, model_type=model_type)
            st.write(model_liner.summary())
            
        
        
        
    