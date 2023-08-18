# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 11:16:41 2023

@author: jczeng
"""


import pandas as pd


def ratew(res,key,w):
    basetime=5*w
    df=res.copy()
    df['M(t+'+str(w)+'w)']=df[key].shift(-basetime)
    df[str(w)+'W']=(df['M(t+'+str(w)+'w)']-df[key])/df[key]
    df.drop(columns='M(t+'+str(w)+'w)',inplace=True)
    return df

def ratem(res,key,month):
    basetime=22*month
    df=res.copy()
    df['M(t+'+str(month)+'M)']=df[key].shift(-basetime)
    df[str(month)+'M']=(df['M(t+'+str(month)+'M)']-df[key])/df[key]
    df.drop(columns='M(t+'+str(month)+'M)',inplace=True)
    return df


def ratey(res,key,year):
    basetime=250*year
    df=res.copy()
    df['M(t+'+str(year)+'Y)']=df[key].shift(-basetime)
    df[str(year)+'Y']=(df['M(t+'+str(year)+'Y)']-df[key])/df[key]
    df.drop(columns='M(t+'+str(year)+'Y)',inplace=True)
    return df


def winrate1(ts_code,data):
    data=data.loc[data['ts_code']==ts_code]
    data=Buline(data,'close_qfq',14)
    data['mark']=np.nan
    data.loc[data['cdf']<20,['mark']]=True
    a=data.loc[(data['mark']==True)&(data['1W']>0)].shape[0]
    b=data.loc[(data['mark']==True)&(data['1W']<0)].shape[0]
    return [a/(a+b),data['ts_code'].iat[0]]
    

def winrate2(ts_code,data):
    data=data.loc[data['ts_code']==ts_code]
    data=Buline(data,'close_qfq',14)
    data=calculate_kdj(data,'close_qfq',high='high_qfq',low='low_qfq')
    data['mark']=np.nan
    data.loc[(data['cdf']<20)&(data['k']<20),['mark']]=True
    a=data.loc[(data['mark']==True)&(data['1W']>0)].shape[0]
    b=data.loc[(data['mark']==True)&(data['1W']<0)].shape[0]
    return [a/(a+b),data['ts_code'].iat[0]]

def winrate3(ts_code,data):
    data=data.loc[data['ts_code']==ts_code]
    data=Buline(data,'close_qfq',14)
    data=calculate_kdj(data,'close_qfq',high='high_qfq',low='low_qfq')
    data['mark']=np.nan
    data.loc[(data['cdf']<20)&(data['k']<20),['mark']]=True
    col=['5D',
    '6D', '7D', '8D', '9D', '10D', '11D', '12D', '13D', '14D', '15D', '16D',
    '17D', '18D', '19D', '20D', '21D', '22D']
    data[col]=scipy.stats.norm.cdf(data[col],0,0.075385072)*100
    data['win']=(data.loc[:, col] > 50).sum(axis=1) >= 11
    a=data.loc[(data['mark']==True)&(data['win']==True)].shape[0]
    b=data.loc[(data['mark']==True)&(data['win']!=True)].shape[0]
    return [a/(a+b),data['ts_code'].iat[0]]

def describe_group(group):
    columns=['1W', '2W', '3W', '4W',
           '2M', '3M', '4M', '5M', '6M', '7M', '8M', '9M', '10M', '11M', '12M',
           '2Y', '3Y', '4Y', '5Y']
    return group[columns].describe(percentiles=[.05, .1, .2, .25, .3, .4, .5, .6, .7, .75, .8, .9, .95])


def save_to_excel(industry, dataframes):
    combined_df = pd.concat(dataframes)
    file_name = f"{industry}.xlsx"
    combined_df.to_excel(file_name)  # 设置index=False以避免保存索引列
    
    
def save_to_excel1(args):
    industry, dataframes = args
    with pd.ExcelWriter(f"{industry}.xlsx") as writer:
        base_df = pd.read_excel(f"{industry}.xlsx", sheet_name='Sheet1')
        base_df.to_excel(writer, sheet_name='Sheet1', index=False)
        for i, df in enumerate(dataframes):
            sheet_name = f'Sheet{i+2}'
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    


