import streamlit as st
import numpy as np
import pandas as pd
import arfs.feature_selection as arfsfs
import arfs.feature_selection.allrelevant as arfsgroot
from arfs.feature_selection import (
    MinRedundancyMaxRelevance,
    GrootCV,
    MissingValueThreshold,
    UniqueValuesThreshold,
    CollinearityThreshold,
    make_fs_summary,
)
from arfs.utils import LightForestClassifier, LightForestRegressor
from arfs.benchmark import highlight_tick, compare_varimp, sklearn_pimp_bench
from lightgbm import LGBMRegressor, LGBMClassifier
import matplotlib.pyplot as plt
from arfs.preprocessing import OrdinalEncoderPandas



if __name__ == '__main__':
    st.title("变量筛选")
    model_type=st.radio('模型类别选择', ('回归', '分类'),horizontal=True)
    upload_file=st.file_uploader('上传数据' , type=['csv'], key='upload_file')
    if upload_file is not None:
        data = pd.read_csv(upload_file)
    else:
        if model_type=='回归':
            data = pd.read_csv('Housing_Price_Data.csv')
        else:
            data = pd.read_csv('SAHeart.csv')
        
    X_drop=st.multiselect('排除的指标（比如编号等）', data.columns)
    outcome=st.selectbox('选择结局指标（其它指标作为预测指标）', data.columns)
    
    data=data.drop(X_drop, axis=1)
    
    X=data.drop([outcome],axis=1)
    X=OrdinalEncoderPandas().fit_transform(X)
    y=data[outcome]
    
  
    if 'X' not in st.session_state:
        st.session_state.X = X
        st.session_state.y = y
    else:
        st.session_state.X = X
        st.session_state.y = y
    # 数据筛选
    
    '***'
    with st.container():
        if model_type=='回归':
            model_lg = LGBMRegressor(random_state=42, verbose=-1)
        else:
            model_lg = LGBMClassifier(random_state=42, verbose=-1)
            
        st.header(' Leshy 算法（来自python-arfs包）')
        importance=st.radio('算法中重要性参数', ('shap', 'naive'),horizontal=True)
        feat_selector = arfsgroot.Leshy(
            model_lg, n_estimators=20, verbose=1, max_iter=10, random_state=42, importance=importance
        )
        feat_selector.fit(X, y, sample_weight=None)
        
        selected_features = feat_selector.get_feature_names_out()
        if 'selected_features' not in st.session_state:
            st.session_state.selected_features = selected_features
        else:
            st.session_state.selected_features = selected_features
        
        
        fig = feat_selector.plot_importance(n_feat_per_inch=5)
        fig.set_size_inches(10, 5)
        # highlight synthetic random variable
        fig = highlight_tick(figure=fig, str_match="random")
        fig = highlight_tick(figure=fig, str_match="genuine", color="green")

        dic_ft_select = pd.DataFrame({'name':X.columns, 'rank':feat_selector.ranking_,'selected':feat_selector.support_})
        dic_ft_select=dic_ft_select.sort_values('rank',ascending=True)
    

        st.write(fig)
        st.write(dic_ft_select.T)
    '***'
    with st.container():
        st.header('数据展示：')
        st.write(st.session_state.X)