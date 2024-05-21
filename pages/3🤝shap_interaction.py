import streamlit as st
import shap
import numpy as np
import plotly.graph_objects as go  
   
if __name__ == '__main__':
    st.title('🤝SHAP交互作用')

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
        
    