# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 13:01:42 2023

@author: jczeng
"""
from scipy.stats import norm
import scipy.stats as stats
import pandas as pd
import numpy as np
'''
状态方程 X(t+1) = 0.81385936 * X(t) + Vw(0.047418267)**2+0.00062985
观测方程 Y(t+1) = X(t+1) * dT + Ve(0.007422390190403894)
初始化状态变量 X(0) =0
方差 P(0)=0.01
以及过程噪声 Vw(0.047418267**2)
观测噪声 Ve(0.007422390190403894)

预测：
X-(t+1) = 0.81385936 * X(t) + Vw(0.007422390190403894)**2+0.00062985
P-(t+1) = F * P(t) * F^T + Vw


修正：
K = P-(t+1) / (P-(t+1) + Ve)
X+(t+1)=X-(t+1)+K(real-X-(t+1))
P+(t+1) = (1 - K) * P-(t+1)
n = data.shape[0]

# 初始化状态变量和误差协方差
X0 = [0] * n
P0 = [1] * n

# 设置过程噪声和观测噪声的方差
Vw = 0.047418267 ** 2
Ve = 0.007422390190403894
'''

def kalman_filter(X, P, A, C, Q, R, measurement):
    # Prediction Step
    X_pred = np.dot(A, X)
    P_pred = np.dot(A, np.dot(P, A.T)) + Q
    # Update Step
    K = np.dot(P_pred, np.dot(C.T, np.linalg.inv(np.dot(C, np.dot(P_pred, C.T)) + R)))
    X = X_pred + np.dot(K, (measurement.reshape(-1, 1) - np.dot(C, X_pred)))
    P = P_pred - np.dot(K, np.dot(C, P_pred))
    return X, P

def Buymark(data):
    data['1d']=data['close_qfq'].pct_change(1).shift(-1)
    data['Bstr']=np.sqrt(((data['1d']**2).cumsum())/(data.index+1))
    res=[]
    X = np.array([[0.0],
              [0.0]])  # 初始化状态估计值 X，假设为二维列向量
    P = np.eye(2)  # 初始化协方差矩阵 P，假设为2x2的单位矩阵
# Define state transition matrix A (replace this with your specific value)
    A = np.array([[1, 1],
              [0.0, 1.0]])
# Define measurement matrix C (replace this with your specific value)
    C = np.array([[1.0, 0.0],
              [0.0, 1.0]])
# Define process noise covariance Q and measurement noise covariance R (replace these with your specific values)
    Q = np.array([[0.01, 0.0],
              [0.0, 0.01]])
    R = np.array([[0.047418267 ** 2, 0.0],
              [0.0,  0.00022975044431238655]])
# Simulate measurements (replace this with your actual measurements)
    measurements = np.array(data[['1d','Bstr']])
    

    for measurement in measurements:
        E, P = kalman_filter(X, P, A, C, Q, R, measurement)
        res.append(E.reshape(1,-1))

    flattened_list = [item[0] for item in res]
    res=pd.DataFrame(flattened_list)
    res['cdf']=norm.cdf(0,res[0],res[1])
    res['cdf']=res['cdf']*100
    data['fE']=res[0]
    data['fstr']=res[1]
    data['Bpro']=res['cdf']
    data['Bpro']=data['Bpro'].shift(1)
    return data



