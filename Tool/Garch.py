# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 09:49:55 2023

@author: jczeng
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf
import arch
from statsmodels.tsa.stattools import adfuller


lag_acf = acf(returns, nlags=250)
lag_pacf = pacf(returns, nlags=250)
plt.figure(figsize=(12, 6))
plt.stem(lag_pacf)
plt.axhline(y=0, linestyle='--', color='gray')  # 添加水平虚线，表示0
plt.axhline(y=-1.96/np.sqrt(len(returns)), linestyle='--', color='red')  # 添加置信区间上界
plt.axhline(y=1.96/np.sqrt(len(returns)), linestyle='--', color='red')   # 添加置信区间下界
plt.xlabel('Lag')
plt.ylabel('PACF')
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()

garch_model = arch.arch_model(returns, vol='Garch', p=1, q=1)
result = garch_model.fit()
print(result.summary())

# Plot the conditional volatility
cond_volatility = result.conditional_volatility
plt.figure(figsize=(12, 6))
plt.plot(cond_volatility)
plt.xlabel('Time')
plt.ylabel('Conditional Volatility')
plt.title('GARCH Model - Conditional Volatility')
plt.show()


for i in range(1,30):
    for j in range(1,30):
        try:
            model = arch.arch_model(returns, vol='Garch', p=i, q=j)
            result = model.fit(disp='off')
            aic = result.aic
            if aic < best_aic:
                best_aic = aic
                best_pq = (i, j)
        except:
            continue
            
