import streamlit as st 

st.header('📑使用说明')
st.write('首先进行变量筛选，通过变量筛选，将分析的重点集中在结局变量相关的特征上，有助于增加分析的速度和效率，避免在不必要的特征上花费时间和算力的浪费；\
    然后进行SHAP分析，SHAP值的计算需要大量的运算，过多的变量会导致会导致计算时间较长，也体现了变量筛选的必要性；\
    最后进行曲线拟合，通过将散点图拟合到曲线上，可以确定关键点的坐标，进而说明变量之间的关系。')
'***'
st.header('🎯用途说明')
st.write('1. 进行预分析，快速定位变量的重要性，从而确定分析的重点。')
st.write('2. 监测变量的收集，及时发现所收集变量是否与结局变量相关，如不相关则要及时调整，避免时间和精力的浪费。')

'***'
st.header('🔧制作说明')
st.write('1. 变量筛选采用的模型是lightgbm，包括回归和分类，采用arfs包中的Leshy算法进行变量筛选，参数采用默认值。')
st.write('2. SHAP分析的模型xgboost，采用的是固定的训练参数，为的是加快运算的速度，精度上可能有所欠缺。')
st.write('3. 曲线拟合采用的方法scikit-learn包中的SplineTransformer函数。')
'***'
with st.container():
    st.sidebar.image('logo.jpg', width=300,)
    st.sidebar.write('欢迎扫码关注微信公众号，获取更多数据分析资讯！')