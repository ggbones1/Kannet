# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 15:29:35 2023

@author: 24397
"""

import streamlit as st
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'SimHei' # 设置字体为SimHei
# 读取Excel文件函数
def read_excel(file):
    df = pd.read_excel(file)
    return df

# 数据预处理函数，将输入/输出数据分离并转换为numpy数组
def preprocess_data(df, inputs, outputs):
    X = df[inputs].values
    y = df[outputs].values

    return X, y

def main():
    # 设置网页标题和头部信息
    st.set_page_config(page_title='BP Neural Network Prediction', page_icon=':chart_with_upwards_trend:')
    st.header('BP神经网络预测应用')

    # 创建侧边栏控件以选择Excel文件和设置模型参数
    file = st.sidebar.file_uploader('导入数据集(Excel格式)', type=['xlsx'])
    
    # 如果已经上传了文件，则显示一个数据预览表格，并允许用户选择要使用的输入/输出列。
    if file is not None:
        data = read_excel(file)
        st.write("## 数据预览")
        st.write(data.head())

        inputs = st.sidebar.multiselect('选择输入列', list(data.columns))
        outputs = st.sidebar.multiselect('选择输出列', list(data.columns))

        X, y = preprocess_data(data, inputs, outputs)

        # 设置BP神经网络的默认参数值
        hidden_layer_sizes = st.sidebar.slider('隐藏层规模', min_value=1, max_value=100, value=(10,))
        activation = st.sidebar.selectbox('激励函数', ['identity', 'logistic', 'tanh', 'relu'], index=3)
        solver = st.sidebar.selectbox('优化器', ['lbfgs', 'sgd', 'adam'], index=2)
        alpha = st.sidebar.slider('正则化系数(alpha)', min_value=0.0, max_value=1.0, value=0.0001, step=0.0001)
        learning_rate_init = st.sidebar.slider('学习率(learning_rate_init)', min_value=0.0001, max_value=1.0, value=0.001, step=0.0001)
        max_iter = st.sidebar.slider('最大迭代次数(max_iter)', min_value=10, max_value=10000, value=200)

        # 创建并拟合BP神经网络模型
        model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                             activation=activation,
                             solver=solver,
                             alpha=alpha,
                             learning_rate_init=learning_rate_init,
                             max_iter=max_iter)

        model.fit(X,y)

        # 进行预测，并计算R2和均方误差指标
        y_pred = model.predict(X)
        r2score = r2_score(y,y_pred)
        mse = mean_squared_error(y,y_pred)

        # 显示模型评估指标的结果
        st.write("## 模型评估")
        st.write(f"R2 score: {r2score:.4f}")
        st.write(f"均方误差(MSE): {mse:.4f}")

        # 绘制每个输出的预测和实际值的折线图
        for i, output in enumerate(outputs):
            fig, ax = plt.subplots()
            ax.plot(y[:, i], '-o', label='True')
            ax.plot(y_pred[:, i], '-o', label='Predicted')
            ax.set_title(f'{output} 预测结果')
            ax.set_xlabel('样本序号')
            ax.set_ylabel(output)
            ax.legend()
            st.pyplot(fig)

if __name__ == '__main__':
    main()