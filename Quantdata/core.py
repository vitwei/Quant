# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 23:50:37 2023

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
from sklearn.metrics import mean_squared_error,mean_absolute_error
import tushare as ts
from sklearn.preprocessing import StandardScaler
from Quantdata.catchsignal import catch,catch_norm0
import multiprocessing
from multiprocessing import Pool, cpu_count
from functools import partial


pro = ts.pro_api("92bfb7a5df70a386927cf4cb1c2b5809df1f45ba402f32dc605111f6")
<<<<<<< HEAD
data=pd.read_feather('Database/std.feather')
=======
data=pd.read_feather('Database/20212022std2.feather')
>>>>>>> 70cb2f8b0aef91778921e394c6f41fac65579033
stock = pro.stock_basic()
stock=stock.loc[stock['symbol'].isin(data['ts_code'])]
tslist=stock[['ts_code']]
df300 = pro.index_weight(**{
    "index_code": "000300.SH"
}, fields=[
    "index_code",
    "con_code",
    "trade_date",
    "weight"
])
ts300=df300.loc[df300['con_code'].isin(tslist['ts_code'])]
ts300=ts300[['con_code']]
ts300.drop_duplicates(inplace=True)
ts300=ts300['con_code'].tolist()
tslist['ts_code'].tolist()
res=pd.DataFrame()
std=pd.read_feather('Database/标准化信息.feather')
sc=StandardScaler(with_mean=False, with_std=False)
sc.mean_=std['mean']
sc.scale_=std['std']

class MLPrg(nn.Module):
    def __init__(self):
        super(MLPrg,self).__init__()
        self.hidden1=nn.Linear(in_features=100,out_features=512,bias=True)
        self.drop=nn.Dropout(0.2)
        self.hidden2=nn.Linear(in_features=512,out_features=256)
        self.predict=nn.Linear(256,1)
        self.bn1=nn.BatchNorm1d(512)
        self.bn2=nn.BatchNorm1d(256)
    def forward(self,x):
        x=F.tanh(self.hidden1(x))
        x=self.drop(x)
        x=F.tanh(self.hidden2(x))
        x=self.drop(x)
        output=self.predict(x)
        return output[:,0]
if __name__ == '__main__':
    rg = torch.load('modelbase/bestks0.025.pt').cpu()
    with Pool(processes=cpu_count()-1) as pool:
        catchsignal1 = partial(catch_norm0, stddata=data)
        catchsignal2 = partial(catchsignal1, rg=rg)
        catchsignal3 = partial(catchsignal2, sc=sc)
        results=pool.map(catchsignal3, tslist.sample(500)['ts_code'].tolist())
'''
class MLPrg(nn.Module):
    def __init__(self):
        super(MLPrg,self).__init__()
        self.hidden1=nn.Linear(in_features=100,out_features=512,bias=True)
        self.hidden2=nn.Linear(in_features=512,out_features=256)
        self.hidden3=nn.Linear(in_features=256,out_features=64)
        self.predict=nn.Linear(64,1)
        self.drop=nn.Dropout(0.2)
    def forward(self,x):
        x=F.tanh(self.hidden1(x))
        x=self.drop(x)
        x=F.tanh(self.hidden2(x))
        x=self.drop(x)
        x=F.tanh(self.hidden3(x))
        x=self.drop(x)
        output=self.predict(x)
        return output[:,0]
if __name__ == '__main__':
    rg = torch.load('J:/quant_trade/modelbase/99.pt').cpu()
    with Pool(processes=cpu_count()-1) as pool:
        catchsignal1 = partial(catch, stddata=data)
        catchsignal2 = partial(catchsignal1, rg=rg)
        catchsignal3 = partial(catchsignal2, sc=sc)
        results=pool.map(catchsignal3, ts300)
'''

'''
res=pd.concat(results)
res.index=range(res.shape[0])
res.to_feather('Database/autodata.feather')
'''

