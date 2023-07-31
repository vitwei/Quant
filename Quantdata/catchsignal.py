# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 15:18:14 2023

@author: vitmcs
"""


import pandas as pd
import numpy as np
import torch 
import math
import time
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import scipy
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error
import tushare as ts
from sklearn.preprocessing import StandardScaler

def catch(ts_code,sc,rg,stddata):
    tscode=ts_code
    pro = ts.pro_api("92bfb7a5df70a386927cf4cb1c2b5809df1f45ba402f32dc605111f6")
    df = pro.stk_factor(**{
    "ts_code": tscode ,
    "start_date": "20220801"
}, fields=[
    "ts_code",
    "trade_date",
    "close_qfq",
    'open_qfq',
    'high_qfq',
    'low_qfq',
    'vol'
])
    df['trade_date']=pd.to_datetime(df['trade_date'])
    df.sort_values(by='trade_date',ascending=True,inplace=True)
    for i in range(1,101):
        df[str(i)+'d']=df['close_qfq'].pct_change(i)
    df.dropna(inplace=True)
    tlist=df.columns.tolist()[7:]
    df[tlist]=sc.transform(df[tlist])
    x=df[tlist]
    xt=torch.from_numpy(x.values.astype(np.float32)).cpu()
    rg.eval()
    y=rg(xt)
    y=y.cpu()
    df=df[["ts_code",
        "trade_date",
        "close_qfq",
        'open_qfq',
        'high_qfq',
        'low_qfq','vol']]
    df['pre']=y.detach().numpy()
    df['ts_code']=df['ts_code'].str[:-3]
    df=pd.merge(df,stddata)
    cdf_threshold = scipy.stats.norm.cdf(0, loc=df['pre'], scale=df['std'])
    cdf_threshold *= 100
    df['pro']=cdf_threshold
    return df

'''
def catch_offline(ts_code,sc,rg,stddata,qfqdata):
    tscode=ts_code
    pro = ts.pro_api("92bfb7a5df70a386927cf4cb1c2b5809df1f45ba402f32dc605111f6")

    df['trade_date']=pd.to_datetime(df['trade_date'])
    df.sort_values(by='trade_date',ascending=True,inplace=True)
    for i in range(1,101):
        df[str(i)+'d']=df['close_qfq'].pct_change(i)
    df.dropna(inplace=True)
    tlist=df.columns.tolist()[3:103]
    df[tlist]=sc.transform(df[tlist])
    x=df[tlist]
    xt=torch.from_numpy(x.values.astype(np.float32)).cpu()
    rg.eval()
    y=rg(xt)
    y=y.cpu()
    df=df[['ts_code' ,'trade_date',  'close_qfq']]
    df['pre']=y.detach().numpy()
    df['ts_code']=df['ts_code'].str[:-3]
    df=pd.merge(df,stddata)
    cdf_threshold = scipy.stats.norm.cdf(0, loc=df['pre'], scale=df['std'])
    cdf_threshold *= 100
    df['pro']=cdf_threshold

    daily = pro.daily(**{
    "ts_code": tscode,
    "trade_date": "",
    "start_date": "20220801"
})
    daily['trade_date']=pd.to_datetime(daily['trade_date'])
    daily.sort_values(by='trade_date',ascending=True,inplace=True)
    daily['ts_code']=daily['ts_code'].str[:-3]
    df=pd.merge(df,daily)
    return df'''







def catch_norm0(ts_code,sc,rg,stddata):
    tscode=ts_code
    pro = ts.pro_api("92bfb7a5df70a386927cf4cb1c2b5809df1f45ba402f32dc605111f6")
    df = pro.stk_factor(**{
    "ts_code": tscode ,
    "start_date": "20000101"
}, fields=[
    "ts_code",
    "trade_date",
    "close_qfq",
    'open_qfq',
    'high_qfq',
    'low_qfq',
    'vol'
])
    df['trade_date']=pd.to_datetime(df['trade_date'])
    df.sort_values(by='trade_date',ascending=True,inplace=True)
    for i in range(1,101):
        df[str(i)+'d']=df['close_qfq'].pct_change(i)
    df.dropna(inplace=True)
    tlist=df.columns.tolist()[7:]
    df[tlist]=sc.transform(df[tlist])
    x=df[tlist]
    xt=torch.from_numpy(x.values.astype(np.float32)).cpu()
    rg.eval()
    y=rg(xt)
    y=y.cpu()
    df=df[["ts_code",
        "trade_date",
        "close_qfq",
        'open_qfq',
        'high_qfq',
        'low_qfq','vol']]
    df['pre']=y.detach().numpy()
    df['ts_code']=df['ts_code'].str[:-3]
    df=pd.merge(df,stddata)
    threshold=scipy.stats.norm.cdf(df['pre'],0,df['std'])
    threshold=threshold*100
    #buy_threshold = scipy.stats.norm.ppf(0.7,0,df['std'])
    #sell_threshold=scipy.stats.norm.ppf(0.3,0,df['std'])
    df['pro']=threshold
    return df




