import streamlit as st
import numpy as np
import pandas as pd
import arfs.feature_selection as arfsfs
from sklearn.pipeline import Pipeline
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

#--------------------------------------
import hmac
import streamlit as st


def check_password():
    """Returns `True` if the user had a correct password."""

    def login_form():
        """Form with widgets to collect user information"""
        with st.form("Credentials"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.form_submit_button("Log in", on_click=password_entered)

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["username"] in st.secrets[
            "passwords"
        ] and hmac.compare_digest(
            st.session_state["password"],
            st.secrets.passwords[st.session_state["username"]],
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the username or password.
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    # Return True if the username + password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show inputs for username + password.
    login_form()
    if "password_correct" in st.session_state:
        st.error("😕 User not known or password incorrect")
    return False



if __name__ == '__main__':
    
    if not check_password():
        st.stop()

    # Main Streamlit app starts here
    st.write("Here goes your normal Streamlit app...")
    st.button("Click me")

    st.title("🔎变量筛选")
    model_type=st.radio('模型类别选择', ('回归', '二分类'),horizontal=True)
    if model_type not in st.session_state:
        st.session_state.model_type = model_type

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
    if 'outcome' not in st.session_state:
        st.session_state.outcome = outcome
    else:
        st.session_state.outcome = outcome
    
    data=data.drop(X_drop, axis=1)
    
    X=data.drop([outcome],axis=1)
    #basic selected
    basic_fs_pipeline = Pipeline(
    [
        ("missing", arfsfs.MissingValueThreshold(threshold=0.05)),
        ("unique", arfsfs.UniqueValuesThreshold(threshold=1)),
        ("cardinality", arfsfs.CardinalityThreshold(threshold=10)),
        ("collinearity", arfsfs.CollinearityThreshold(threshold=0.75)),
        ("encoder", OrdinalEncoderPandas())
    ]
)

    X=basic_fs_pipeline.fit_transform(X, y=None)
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
        feat_selector = arfsgroot.Leshy(
            model_lg, n_estimators=20, verbose=1, max_iter=10, random_state=42, importance='naive'
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
        st.write('说明：图中蓝色框表示被选中的特征,对应表格中rank为1的变量，而红色框表示被排除的特征。后续的SHAP分析聚焦在被选中的特征上进行，因而增加了分析的速度和效率。')
    '***'
    with st.container():
        st.header('数据展示：')
        st.write(st.session_state.X)