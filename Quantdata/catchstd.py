# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 11:16:14 2023

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
from sklearn.metrics import r2_score
import gc

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

data=pd.read_feather('J:/quant_trade/database/trains.feather')
tlist=data.columns.tolist()
tlist=tlist[3:103]
x=data[tlist]
xt=torch.from_numpy(x.values.astype(np.float32)).cpu()
y=data[['1W']]
y['ts_code']=data['ts_code']
y['trade_date']=data['trade_date']
del data
del x
gc.collect()
rg.eval()
yt=rg(xt)
yt=yt.detach().numpy()
y['pre']=yt
y['bias']=y['1W']-y['pre']
r2=r2_score(y['pre'], y['1W'])
def getstd(x):
    a=x.shape[0]
    b=x['bias']**2
    x['std']=np.sqrt(b.sum()/a)
    return x

res=y.groupby(by='ts_code').apply(lambda x:getstd(x))
res=res[['ts_code','std']]
res.drop_duplicates(inplace=True)
res.index=range(res.shape[0])
res.to_feather('std.feather')
'''
plt.figure(figsize=(16,9))
plt.scatter(range(res.shape[0]),res['1W'],c='blue')
plt.plot(range(res.shape[0]),res['pre']+res['std'],c='r')
plt.plot(range(res.shape[0]),res['pre']-res['std'],c='r')
plt.plot(range(res.shape[0]),res['pre']+2*res['std'],c='r')
plt.plot(range(res.shape[0]),res['pre']-2*res['std'],c='r')
plt.plot(range(res.shape[0]),res['pre']+3*res['std'],c='r')
plt.plot(range(res.shape[0]),res['pre']-3*res['std'],c='r')
'''