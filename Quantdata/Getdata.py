# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 22:59:05 2023

@author: vitmcs
"""


# 导入tushare
import tushare as ts
import pandas as pd
# 初始化pro接口
pro = ts.pro_api('2f8168a08ce71c9655c5b633b3dc17088bfc6f5a07e629f38f1c3d1f')

# 拉取数据
def Getdata(key):
  df = pro.daily(**{
    "ts_code": key,
    "trade_date": "",
    "start_date": "",
    "end_date": "",
    "offset": "",
    "limit": ""
}, fields=[
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "change",
    "pct_chg",
    "vol",
    "amount"])
  df['Datetime']=pd.to_datetime(df['trade_date'])
  df.set_index('Datetime',inplace=True)
  df.sort_index(ascending=True,inplace=True)
  df.rename(columns={'vol':'volume'},inplace=True)
  df['openinterest']=-1
  df=df[['open', 'high', 'low', 'close','volume','openinterest']]
  return df


