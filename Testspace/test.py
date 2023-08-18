# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 09:47:08 2023

@author: jczeng
"""
from scipy.fft import fft, fftfreq,ifft
import scipy.stats
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
data=pd.read_feather('Database/autodata.feather')
ts_code='600552.SH'
temp=data[data['ts_code']==ts_code[:-3]]

def SMA(data,col,window):
    middle_band = data[col].rolling(window=window).mean()
    return middle_band.tolist()

def Buline(data,col,window=20,num_std=2):
    '''
    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    col : TYPE
        DESCRIPTION.
    window : TYPE, optional
        DESCRIPTION. The default is 20.
    num_std : TYPE, optional
        DESCRIPTION. The default is 2.
    注意日期排序
    Returns
    -------
    data : TYPE
        DESCRIPTION.

    '''
    middle_band=SMA(data,col,window)
    # 计算标准差
    moving_std = data[col].rolling(window=window).std()
    # 计算上轨和下轨
    upper_band = middle_band + num_std * moving_std
    lower_band = middle_band - num_std * moving_std
    data[str(window)+'SMA']=middle_band
    data['upper_band']=upper_band
    data['lower_band']=lower_band
    data['moving_std']=moving_std
    data['cdf']=scipy.stats.norm.cdf(data[col],data[str(window)+'SMA'],data['moving_std'])
    data['cdf']=data['cdf']*100
    return data
    
def calculate_kdj(data,col,n=9,high=None,low=None):
    """
    K线（%K）：K线是随机指标中的快速线，
    表示当前价格与一段时间内最低价之间的相对位置。
    K值的计算公式为：K = (C - L) / (H - L) * 100，
    其中C是当前收盘价，
    L是过去一段时间内的最低价，
    H是过去一段时间内的最高价。
    K值的取值范围在0到100之间。
    D线（%D）：D线是K线的平滑线，用于平滑K值，
    一般采用3天的简单移动平均线。
    D值的计算方法是对K值进行3天的简单移动平均，即D = MA(K, 3)。
    J线（%J）：J线是K线与D线的差值，
    用于提供更多的信息。
    J值的计算公式为：J = 3 * K - 2 * D。
    n=windows大小
    M1表示K线平滑的周期
    """
    data = data.reset_index(drop=True)
    # 计算最近n天的最高价和最低价
    if high==None and low==None:
        data['lowest_low'] = data[col].rolling(window=n).min()
        data['highest_high'] = data[col].rolling(window=n).max()
    else:
        data['lowest_low'] = data[low].rolling(window=n).min()
        data['highest_high'] = data[high].rolling(window=n).max()

    # 计算RSV值
    data['rsv'] = (data[col] - data['lowest_low']) / (data['highest_high'] - data['lowest_low']) * 100
    # 初始化K和D值
    data['k'] = 50
    data['d'] = 50
    # 计算K值和D值
    for i in range(n, len(data)):
        data.loc[i, 'k'] = 2/3 * data.loc[i-1, 'k'] + 1/3 * data.loc[i, 'rsv']
        data.loc[i, 'd'] = 2/3 * data.loc[i-1, 'd'] + 1/3 * data.loc[i, 'k']

    # Calculate J value
    data['j'] = 3 * data['k'] - 2 * data['d']

    return data



def ff(res,weights):
    columns_to_transform = ['vol_ratio', 'turn_over', 'net_mf_vol']
    fvalue=pd.DataFrame(fft(res[columns_to_transform]))
    weighted_fft = weights[0] * fvalue[0] + weights[1] * fvalue[1] + weights[2] * fvalue[2]
    weighted_fft=np.array(weighted_fft)
    reconstructed_signal = ifft(weighted_fft)
    synthesized_signal = np.real(reconstructed_signal)
    sns.lineplot(res['logreturn'])
    sns.lineplot(np.real(reconstructed_signal))
    


    