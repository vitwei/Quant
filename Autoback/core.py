# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 16:25:16 2023

@author: vitmcs
"""

import pandas as pd
import numpy as np
import multiprocessing
from multiprocessing import Pool, cpu_count
from functools import partial
import tushare as ts
from base.autoback import autotest_offline
#from base.autoback import autotest


pro = ts.pro_api("92bfb7a5df70a386927cf4cb1c2b5809df1f45ba402f32dc605111f6")
df = pro.stock_basic()
df300 = pro.index_weight(**{
    "index_code": "000300.SH"
}, fields=[
    "index_code",
    "con_code",
    "trade_date",
    "weight"
])
data=pd.read_feather('J:/quant_trade/autodata.feather')
tl=pd.unique(data['ts_code'])
tl=tl.tolist()
df=df.loc[df['symbol'].isin(tl)]
tl=df['ts_code']
tl=pd.DataFrame(tl)
tl300=tl.loc[tl['ts_code'].isin(df300['con_code'])]


if __name__ == '__main__':
    with Pool(processes=cpu_count()-1) as pool:
        autotest_partial = partial(autotest_offline, data=data)
        results=pool.map(autotest_partial, tl300['ts_code'].tolist())
    results=pd.DataFrame(results,columns=['annualreturn','maxback','ts_code'])

'''
if __name__ == '__main__':
    with Pool(processes=cpu_count()-1) as pool:
        autotest_partial = partial(autotest, data=data)
        results=pool.map(autotest_partial, tl['ts_code'].tolist())
    results=pd.DataFrame(results,columns=['annualreturn','maxback','ts_code'])
'''

    
    

