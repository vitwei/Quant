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
    params = (('open',0), ('high', 1), ('low', 2), ('close', 3), ('volume', 4), ('openinterest',5), ('pre',6),('pro',7))


class st(bt.Strategy):#只能卖后买
    def __init__(self):
        self.dataclose = self.data0.close
        self.distribution = self.data0.pro
        self.profit_target = 0.04  # 5%的收益目标
        self.stop_target = -0.05
        self.daily_return_threshold = 0.000386
        self.holding_days = 0  # 持仓天数
        self.total_return = 0  # 总收益率
        self.buy_transaction =pd.DataFrame(columns=['成本','day']) # 买入交易记录
        self.buy_times=-1
        self.stop_mark=False
        self.stop_loss=0.01
        self.log=pd.DataFrame(columns=['日期','行为','价格','信号','价值','操作利润','平均收益','平均日收益'])
        self.sma5=bt.ind.SimpleMovingAverage(self.dataclose,period=5)
        self.order = None
        self.pvalue=self.broker.get_value()
        
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return
        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                buy_price = self.data0.close[0]
                buy_transaction = pd.DataFrame({'日期':[self.data0.datetime.date(0)],'成本': [order.executed.price], 'day': [1]})
                self.buy_transaction = pd.concat([self.buy_transaction, buy_transaction])
                self.buy_times+=1
                #log = pd.DataFrame({'日期':[self.data0.datetime.date(0)],'行为': ['买入'],'价格': [order.executed.price],
                 #                  '信号':['买入成功'],'价值':[order.executed.value]})
                #self.log = pd.concat([self.log, log])
            elif order.issell():
                self.buy_transaction =pd.DataFrame(columns=['成本','day'])
                #log = pd.DataFrame({'日期':[self.data0.datetime.date(0)],'行为': ['卖出'],'价格': [order.executed.price],
                 #                 '信号':['卖出成功'],'平均收益':[self.total_return],'平均日收益':[self.average_return],
                  #                '操作利润':[order.executed.pnl]})
                #self.log = pd.concat([self.log, log])
                self.buy_times=-1
      #  elif order.status in [order.Canceled, order.Margin, order.Rejected]:
                #log = pd.DataFrame({'日期':[self.data0.datetime.date(0)],'行为': ['取消'],'价格': [order.executed.price],
                 #                  '信号':['操作取消'],'平均收益':[self.total_return],'平均日收益':[self.average_return]})
                #self.log = pd.concat([self.log, log])
        # Write down: no pending order
        self.order = None
    def next(self):
        buy_price = self.data0.close[0]  # 获取当前的买入价格
        if self.position:
            current_value=self.broker.get_value()
            pnl=(current_value-self.pvalue)/self.pvalue
            self.total_return=(self.data.close[0]-self.buy_transaction['成本'])/self.buy_transaction['成本']
            self.total_return=self.total_return.mean()
            if not self.buy_transaction.empty and not self.buy_transaction['day'].isnull().all():
                self.average_return = self.total_return / int(self.buy_transaction['day'].mean())
            else:
                self.average_return = np.nan
            if self.total_return>self.profit_target:
            #    log = pd.DataFrame({'日期':[self.data0.datetime.date(0)],'行为': ['卖出'],'价格': [self.data.close[0]],'信号':['4%到手'],'平均收益':[self.total_return],'平均日收益':[self.average_return]})
           #     self.log = pd.concat([self.log, log])
                self.order =self.sell(exectype=bt.Order.Market)
            elif  self.average_return < self.daily_return_threshold and self.buy_transaction['day'].max()>=60:
           #     log = pd.DataFrame({'日期':[self.data0.datetime.date(0)],'行为': ['卖出'], '价格': [self.data.close[0]],'信号':['日均较低'],'平均收益':[self.total_return],'平均日收益':[self.average_return]})
           #     self.log = pd.concat([self.log, log])
                self.order =self.sell(exectype=bt.Order.Market)
            elif self.total_return <=self.stop_target:
                self.stop_mark=True
            #    log = pd.DataFrame({'日期':[self.data0.datetime.date(0)],'行为': ['卖出'], '价格': [self.data.close[0]],'信号':['止损'],'平均收益':[self.total_return],'平均日收益':[self.average_return]})
             #   self.log = pd.concat([self.log, log])
                self.order =self.sell(exectype=bt.Order.Market)
            elif self.distribution[0]<30:
            #    log = pd.DataFrame({'日期':[self.data0.datetime.date(0)],'行为': ['卖出'], '价格': [self.data.close[0]],'信号':['卖出分布'],'平均收益':[self.total_return],'平均日收益':[self.average_return]})
            #    self.log = pd.concat([self.log, log])
                self.order =self.sell(exectype=bt.Order.Market)
            elif pnl<=-self.stop_loss:
                self.stop_mark=True
           #     log = pd.DataFrame({'日期':[self.data0.datetime.date(0)],'行为': ['卖出'], '价格': [self.data.close[0]],'信号':['本金止损'],'平均收益':[self.total_return],'平均日收益':[self.average_return]})
           #     self.log = pd.concat([self.log, log])
                self.order =self.sell(exectype=bt.Order.Market)
            if not self.buy_transaction.empty:
                self.buy_transaction['day']+=1
        elif self.distribution[0] >70 and self.stop_mark==False:   
            if self.buy_times==-1:
          #      log = pd.DataFrame({'日期':[self.data0.datetime.date(0)],'行为': ['买入'],'价格': [buy_price],'信号':[self.distribution[0]]})
          #      self.log = pd.concat([self.log, log])
                self.order = self.buy()
            elif self.buy_times!=-1 and self.total_return>0 and self.buy_times<3:
          #      log = pd.DataFrame({'日期':[self.data0.datetime.date(0)],'行为': ['买入'],'价格': [buy_price],'信号':[self.distribution[0]]})
          #      self.log = pd.concat([self.log, log])
                self.order = self.buy()
   # def stop(self):
    #    self.log.to_excel('log4.xlsx',index=None)








def autotest(ts_code,data):#need add daily(qfq)
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
    cerebro.addstrategy(st)
    cerebro.addanalyzer(bt.analyzers.DrawDown,_name='DrawDown')
    cerebro.addanalyzer(bt.analyzers.AnnualReturn,_name='AnnualReturn')
    cerebro.broker.setcash(1000.0)
    cerebro.broker.setcommission(commission=0.0006)
    cerebro.addsizer(bt.sizers.PercentSizer,percents=90)
    result=cerebro.run()
    res=result[0].analyzers.AnnualReturn.get_analysis()
    returns_list = list(res.values())
    returns_list.append(result[0].analyzers.DrawDown.get_analysis()['max']['drawdown'])
    returns_list.append(ts_code)
    return returns_list

def autotest_offline(ts_code,data):
    tscode=ts_code
    df=data.loc[data['ts_code']==tscode[:-3]]
    df['trade_date']=df['trade_date'].astype(str)
    df['Datetime']=pd.to_datetime(df['trade_date'])
    df['volume']=df['vol']
    df['openinterest']=0
    df.set_index('Datetime',inplace=True)
    df.sort_index(inplace=True,ascending=True)
    df=df[['open_qfq', 'high_qfq', 'low_qfq','close_qfq', 'volume', 'openinterest', 'pre','pro']]
    cerebro=bt.Cerebro()
    data=CustomPandasData(dataname=df,fromdate=dt.datetime(2023,1,1),todate=dt.datetime(2024,1,1),timeframe=bt.TimeFrame.Days)
    cerebro.adddata(data)
    cerebro.addstrategy(st)
    cerebro.addanalyzer(bt.analyzers.DrawDown,_name='DrawDown')
    cerebro.addanalyzer(bt.analyzers.AnnualReturn,_name='AnnualReturn')
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.0006)
    cerebro.addsizer(bt.sizers.PercentSizer,percents=50)
    result=cerebro.run()
    res=result[0].analyzers.AnnualReturn.get_analysis()
    returns_list = list(res.values())
    returns_list.append(result[0].analyzers.DrawDown.get_analysis()['max']['drawdown'])
    returns_list.append(tscode)
    return returns_list
    