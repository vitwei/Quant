# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 14:50:07 2023

@author: jczeng
"""
import backtrader as bt
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt 
import tushare as ts


class CustomPandasData(bt.feeds.PandasData):
    lines = ('open', 'high', 'low', 'close', 'volume', 'openinterest', 'pre','pro')
    params = (('open', 0), ('high', 1), ('low', 2), ('close', 3), ('volume', 4), ('openinterest', 5), ('pre', 9),('pro', 13))


#这是策略
class SMA(bt.Strategy):
    def __init__(self):
        self.dataclose = self.data0.close
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.distribution = self.data0.pro
        self.profit_target = 0.05  # 5%的收益目标
        self.stop_target =-0.2  #止损目标
    def next(self): 
        if not self.position:
            if self.distribution[0] < 40:
                self.buy()
        else:
            current_profit = (self.data.close[0] - self.position.price) / self.position.price
            if current_profit >= self.profit_target:
                self.close()
            if current_profit <=self.stop_target:
                self.close()

def autotest(ts_code,data):
    tscode=ts_code
    pro = ts.pro_api("92bfb7a5df70a386927cf4cb1c2b5809df1f45ba402f32dc605111f6")
    dl=data.loc[data['ts_code']==tscode[:-3]]
    dl['trade_date']=dl['trade_date'].astype(str)
    dl['Datetime']=pd.to_datetime(dl['trade_date'])
    dl.set_index('Datetime',inplace=True)
    dl.sort_index(inplace=True,ascending=True)
    df = pro.daily(**{"ts_code": tscode})
    df['trade_date']=df['trade_date'].astype(str)
    df['Datetime']=pd.to_datetime(df['trade_date'])
    df.set_index('Datetime',inplace=True)
    df['openinterest']=0
    df['volume']=df['vol']
    df=df[['open','high','low','close','volume','openinterest']]
    df.sort_index(inplace=True,ascending=True)
    res=pd.merge(df,dl,left_index=True,right_index=True)
    dp = CustomPandasData(dataname=res, timeframe=bt.TimeFrame.Days)
    cerebro=bt.Cerebro()
    data=bt.feeds.PandasData(dataname=df,fromdate=dt.datetime(2021,1,1),todate=dt.datetime(2022,1,1),timeframe=bt.TimeFrame.Days)
    cerebro.adddata(dp)
    cerebro.addstrategy(SMA)
    cerebro.addanalyzer(bt.analyzers.DrawDown,_name='DrawDown')
    cerebro.addanalyzer(bt.analyzers.AnnualReturn,_name='AnnualReturn')
    cerebro.broker.setcash(1000.0)#钱钱
    cerebro.broker.setcommission(commission=0.0006)#手续费
    cerebro.addsizer(bt.sizers.PercentSizer,percents=90)#购买仓位
    result=cerebro.run()
    res=result[0].analyzers.AnnualReturn.get_analysis()
    returns_list = list(res.values())
    returns_list.append(result[0].analyzers.DrawDown.get_analysis()['max']['drawdown'])
    returns_list.append(ts_code)
    return returns_list
    