# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 15:56:43 2023

@author: jczeng
"""

import pandas as pd
import numpy as np
import scipy
import random
import math
import seaborn as sns
from numpy.stats import norm 
'''
粒子生成
1.设置初值 假设X0满足正态分布
2.根据X0的样本的分布生成X0的N个粒子
3.生成权重Wi=1/N
预测
1.根据X0的N个粒子生成N个X1即(n）X1=F（(n）x0）+Q Q服从（0，Q）正态分布 
2.由n个X1粒子生成N个(n）F（X1）即观测方程Y1-
修正
1.记观测为Y1
2.根据观测修正下N个权重       (n）Wi+1=（Y1-(n）Y1-）*(n）Wi
3.将粒子权重归一化
4.继续预测
'''
QE=0.047418267
Qstd=0.084838168
data=pd.read_feather('C:/Users/jczeng/Desktop/David Hwang/st/ktest.feather')
shape=data.shape[0]
#生成X+结果数组
Xres=list(range(shape))
Sres=list(range(shape))
#定位Y
realx=data['real']
realx=realx.tolist()
#粒子数
n=10**2
#粒子生成和权重生成
XE=list(range(n))
Xstd=list(range(n))
WE=list(range(n))
Wstd=list(range(n))
var=list(range(n))
#设置初值为0
for i in range(n):
    XE[i]=0
    Xstd[i]=0
    var[i]=0
for i in range(shape):
    Xres[i]=0
    Sres[i]=0
#生成Xtemp数组
Stemp=Xstd.copy()
Xtemp=XE.copy()
#生成权重Wi=1/N
for i in range(n):
    WE[i]=1/n
    Wstd[i]=1/n
#预测

for i in range(len(realx)): 
    #时序处理
    for j in range(n): 
        XE[j]=0.81385936*XE[j]+random.normalvariate(0,QE)+0.00062985
        Xstd[j]=random.normalvariate(0,Qstd)+Xstd[j]
        #Xant[j]=Xant[j]+random.normalvariate(0,Q)
        #这是100个粒子的Y1-
        var[j]=(realx[i]-XE[j])**2
    std=np.sqrt(sum(var)/n)
    for j in range(n):
        #这是100个粒子的权重
        WE[j]=((2*math.pi*QE)**(-0.5))*math.exp(-(((realx[i]-XE[j])**2)/(2*QE)))*WE[j]
        Wstd[j]=((2*math.pi*Qstd)**(-0.5))*math.exp(-(((std-Xstd[j])**2)/(2*Qstd)))*Wstd[j]
    #权重归一化         
    WE = [w/sum(WE) for w in WE]
    Wstd=[w/sum(Wstd) for w in Wstd]
    #权重复制
    CE=list(range(n))
    Cstd=list(range(n))
    for k in range(n):
        CE[k]=0
        Cstd[k]=0
    CE[0]=WE[0]
    Cstd[0]=Wstd[0]
    for j in range(1,n):
        CE[j]=CE[j-1]+WE[j]
        Cstd[j]=Cstd[j-1]+Wstd[j]
#重采样
#转盘子，生成随机数，看落在哪个区间
#首先我们要重采样n个粒子，粒子数要与之前相同
    for j in range(n):
        a=random.uniform(0,1)
        for k in range(n):
            if a<CE[k] and a<Cstd[k]:
                Xtemp[j]=XE[k]
                Stemp[j]=Xstd[k]
                break
            if a<CE[k]:
                Xtemp[j]=XE[k]
                break
            if a<Cstd[k]:
                Stemp[j]=Xstd[k]
                break
    #更新
    XE=Xtemp.copy()
    Xstd=Stemp.copy()
    #恢复权重
    for j in range(n):
        WE[j]=1/n
        Wstd[j]=1/n
    #粒子集合归一
    Xres[i]=sum(Xtemp)/n
    Sres[i]=sum(Xstd)/n


res=pd.DataFrame()
res['real']=realx
res['pre']=Xres
res['std']=np.abs(Sres)
res['pre+std']=res['pre']+res['std']
res['pre-std']=res['pre']-res['std']
res['pre+2std']=res['pre']+2*res['std']
res['pre-2std']=res['pre']-2*res['std']
res['pre+3std']=res['pre']+3*res['std']
res['pre-3std']=res['pre']-3*res['std']
res['stand+']=QE
res['stand-']=-QE
res['cdf']=norm.cdf(0,res['pre'],res['std'])



res.index=data['trade_date']
res['preal1']=res['real'].shift(-1)
res['preal2']=res['real'].shift(-2)
res['preal3']=res['real'].shift(-3)
res['preal4']=res['real'].shift(-4)
res['preal5']=res['real'].shift(-5)

sns.scatterplot(x=res.index,y=res['preal'],c='black')
#sns.lineplot(x=res.index,y=res['pre+std'],c='r')
#sns.lineplot(x=res.index,y=res['pre-std'],c='b')
#sns.lineplot(x=res.index,y=res['pre+2std'],c='orange')
#sns.lineplot(x=res.index,y=res['pre-2std'],c='green')
sns.lineplot(x=res.index,y=res['pre+3std'],c='yellow')
sns.lineplot(x=res.index,y=res['pre-3std'],c='purple')
sns.lineplot(x=res.index,y=res['stand-'],c='red')
sns.lineplot(x=res.index,y=0,c='red')
sns.lineplot(x=res.index,y=res['stand+'],c='red')


