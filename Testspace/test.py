# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 09:47:08 2023

@author: jczeng
"""
from scipy.fft import fft, fftfreq,ifft
import scipy.stats
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
data=pd.read_feather('Database/autodata.feather')
ts_code='600552.SH'
temp=data[data['ts_code']==ts_code[:-3]]


    


    