# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 09:47:08 2023

@author: jczeng
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
data=pd.read_feather('Database/autodata.feather')
ts_code='600552.SH'
temp=data[data['ts_code']==ts_code[:-3]]

def SMA(data,window):
    middle_band = data['close_qfq'].rolling(window=window).mean()
    return middle_band.tolist()

def Buline(data,window=20,num_std=2):
    middle_band=SMA(data,window)
    # 计算标准差
    moving_std = data['close_qfq'].rolling(window=window).std()
    # 计算上轨和下轨
    upper_band = middle_band + num_std * moving_std
    lower_band = middle_band - num_std * moving_std
    data[str(window)+'SMA']=middle_band
    data['upper_band']=upper_band
    data['lower_band']=lower_band
    return data
    
def calculate_kdj(data, n=9):
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
    data['lowest_low'] = data['low_qfq'].rolling(window=n).min()
    data['highest_high'] = data['high_qfq'].rolling(window=n).max()
    # 计算RSV值
    data['rsv'] = (data['close_qfq'] - data['lowest_low']) / (data['highest_high'] - data['lowest_low']) * 100
    # 初始化K和D值
    data['k'] = 50
    data['d'] = 50
    # 计算K值和D值
    for i in range(n, len(data)):
        data.loc[i, 'k'] = 2/3 * data.loc[i-1, 'k'] + 1/3 * data.loc[i, 'rsv']
        data.loc[i, 'd'] = 2/3 * data.loc[i-1, 'd'] + 1/3 * data.loc[i, 'k']
    # 计算J值
    data['j'] = 3 * data['k'] - 2 * data['d']
    return data

