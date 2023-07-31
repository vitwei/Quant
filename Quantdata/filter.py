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
'''

data=pd.read_feather('C:/Users/jczeng/Desktop/David Hwang/st/ktest.feather')
Xadd=data['pre'].tolist()
n=data.shape[0]
X0=list(range(n))
X1=list(range(n))
P0=list(range(n))
P1=list(range(n))
#设置初值为0
for i in range(n):
    X0[i]=0
    X1[i]=0
    P0[i]=0
    P1[i]=0
    
'''
n = data.shape[0]

# 初始化状态变量和误差协方差
X0 = [0] * n
P0 = [1] * n

# 设置过程噪声和观测噪声的方差
Vw = 0.047418267 ** 2
Ve = 0.007422390190403894

Y = data['real'].tolist()

for i in range(1, n):
    # 预测步骤
    X_pred = 0.81385936 * X0[i - 1] + Vw + 0.00062985+Xadd[i]
    P_pred = P0[i - 1] + Vw
    # 更新步骤
    K = P_pred / (P_pred + Ve)
    X1 = X_pred + K * (Y[i] - X_pred)
    P1 = (1 - K) * P_pred
    # 保存修正后的状态估计值和误差协方差
    X0[i] = X1
    P0[i] = P1
    
res=pd.DataFrame()
res['pre']=X0
res['real']=Y
res.plot()
'''
'''
U记做神秘力量
U（t+1)=A*U(t)+Vw（噪声）Vw(个股收益率方差噪声)
Z记做收益率
Z（t+1）=U（t+1）*dt(采样间隔）+X（深度学习拟合出来的信息）+Ve(全局收益率方差噪声)

import numpy as np
from scipy.optimize import minimize
import pandas as pd
data=pd.read_feather('C:/Users/jczeng/Desktop/David Hwang/st/ktest.feather')
Z_data=data['real']
X_data=data['pre']
size=data.shape[0]
Z_data = np.array(Z_data)
X_data = np.array(X_data)
Vw = np.random.normal(0, 0.047418267,size)
Ve = np.random.normal(0, 0.0861532947158952,size)
def state_equation(U, A, Vw):
    return A*U + Vw
# 定义观测方程的函数
def observation_equation(U, X, Ve):
    return U + X + Ve
def loss_function(params, *args):
    U_t, A = params[:2]
    Z_data, X_data, Vw, Ve = args
    U_t_next = state_equation(U_t, A, Vw)
    Z_predicted = observation_equation(U_t_next, X_data, Ve)
    return np.sum((Z_data - Z_predicted) ** 2)
U_t_initial_guess = -0.00185044
A_initial_guess = 1
params = [U_t_initial_guess, A_initial_guess]
args = (Z_data, X_data, Vw, Ve)
result = minimize(loss_function, params, args=args, method='L-BFGS-B')
result.x
'''


'''
import numpy as np

res=[]
def kalman_filter(X, P, A, C, Q, R, measurement):
    # Prediction Step
    X_pred = np.dot(A, X)
    P_pred = np.dot(A, np.dot(P, A.T)) + Q
    # Update Step
    K = np.dot(P_pred, np.dot(C.T, np.linalg.inv(np.dot(C, np.dot(P_pred, C.T)) + R)))
    X = X_pred + np.dot(K, (measurement.reshape(-1, 1) - np.dot(C, X_pred)))
    P = P_pred - np.dot(K, np.dot(C, P_pred))
    return X, P
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
measurements = np.array(data[['real','var']])


for measurement in measurements:
    E, P = kalman_filter(X, P, A, C, Q, R, measurement)
    res.append(E.reshape(1,-1))




flattened_list = [item[0] for item in res]
res=pd.DataFrame(flattened_list)
res['cdf']=norm.cdf(0,res[0],res[1])
res['cdf']=res['cdf']*100




'''




