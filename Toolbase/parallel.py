# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 11:16:41 2023

@author: jczeng
"""


from Testspace.test import winrate1,winrate2,winrate3
from multiprocessing import Pool,cpu_count
from functools import partial
import pandas as pd



df=pd.read_feather('Database/autodata.feather')
def W(df):
    for i in range(5,23):
        df[str(i)+'D']=(df['close_qfq'].shift(-1*i)-df['close_qfq'])/df['close_qfq']
    return df
df=df.groupby(by='ts_code').apply(lambda x:W(x))
tslist=pd.unique(df['ts_code']).tolist()

if __name__ == '__main__':
    with Pool(processes=cpu_count()-1) as pool:
        winrate_partial = partial(winrate3, data=df)
        results=pool.map(winrate_partial, tslist)
    results=pd.DataFrame(results)