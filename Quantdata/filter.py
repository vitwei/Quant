# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 13:01:42 2023

@author: jczeng
"""
import numpy as np
from pykalman import KalmanFilter
import matplotlib.pyplot as plt
import pandas as pd


data=pd.read_feather('database')

kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)

# 进行滤波
(fmeans, fcovs)  = kf.filter(data)

# 绘制结果
plt.figure(figsize=(10, 6))
plt.plot(data, color='blue', label='True Values')
plt.plot(fmeans, color='red', linestyle='dashed', label='Filtered States')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('One-Dimensional Kalman Filtering')
plt.legend()
plt.show()

'''
kf = KalmanFilter(em_vars=['transition_covariance', 'observation_covariance','transition_matrices','observation_matrices'],observation_matrices=np.array(1))
kf = kf.em(data, n_iter=10)
'''


    
    
    

    
    
    
    
    
    
    
    
    
    